"""Record real scheduler decisions on explicitly synthetic resource snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .upstream_adapter import (
    ExperimentError,
    FIFOSequencer,
    FixtureState,
    PassRecording,
    ResourcePolicyValidator,
    SessionProvisioner,
    create_agent_selector,
    schedule_pass,
)

BACKEND_AI_VERSION = "26.9.7"
SOURCE_COMMIT = "332a308e34c5c7bcd57939850fe032cfd8d8ca00"


def _provenance() -> dict[str, object]:
    try:
        installed_version = version("backend.ai-manager")
    except PackageNotFoundError as error:
        raise ExperimentError("Install the pinned backend.ai-manager distribution first") from error
    if installed_version != BACKEND_AI_VERSION:
        raise ExperimentError(
            f"Expected backend.ai-manager {BACKEND_AI_VERSION}, found {installed_version}"
        )
    component_hashes = {}
    for component in (
        SessionProvisioner,
        ResourcePolicyValidator,
        FIFOSequencer,
        create_agent_selector,
    ):
        component_hashes[f"{component.__module__}.{component.__name__}"] = hashlib.sha256(
            Path(inspect.getfile(component)).read_bytes()
        ).hexdigest()
    return {
        "distribution": "backend.ai-manager",
        "version": installed_version,
        "source_commit": SOURCE_COMMIT,
        "source_url": f"https://github.com/lablup/backend.ai/tree/{SOURCE_COMMIT}",
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "component_sha256": component_hashes,
    }


def _expect(actual: object, expected: object, message: str, checks: list[str]) -> None:
    if actual != expected:
        raise ExperimentError(f"{message}: expected {expected!r}, observed {actual!r}")
    checks.append(message)


def _outcomes(result: PassRecording) -> dict[str, str]:
    return {item["session_id"]: item["outcome"] for item in result["decisions"]}


async def _controls() -> list[dict[str, object]]:
    controls = []

    bob_first = FixtureState(pending=["bob", "alice"])
    result = await schedule_pass(bob_first)
    checks: list[str] = []
    _expect(
        _outcomes(result),
        {"bob": "scheduled", "alice": "failed"},
        "Putting Bob first lets his one-CPU request fit",
        checks,
    )
    _expect(
        bob_first.allocations,
        {"current": 2, "bob": 1},
        "Bob-first order leaves one free CPU slot",
        checks,
    )
    controls.append(
        {
            "name": "bob_first",
            "passed": True,
            "checks": checks,
            "observation": result,
        }
    )

    quota = FixtureState()
    result = await schedule_pass(quota, user_cpu_limits={"alice": 3})
    checks = []
    _expect(
        _outcomes(result),
        {"alice": "failed", "bob": "scheduled"},
        "Alice's quota rejection allows the scheduler to attempt Bob",
        checks,
    )
    alice_phases = result["upstream_records"]["alice"]["phases"]
    _expect(
        any(
            phase["name"] == "validation" and phase["status"] == "failed" for phase in alice_phases
        ),
        True,
        "Alice fails during upstream validation in the quota control",
        checks,
    )
    _expect(
        any(phase["name"] == "agent_selection" for phase in alice_phases),
        False,
        "The quota control rejects Alice before placement",
        checks,
    )
    controls.append(
        {
            "name": "quota_does_not_block_bob",
            "passed": True,
            "inputs": {"alice_cpu_quota": 3},
            "checks": checks,
            "observation": result,
        }
    )

    unchanged = FixtureState()
    first = await schedule_pass(unchanged)
    second = await schedule_pass(unchanged)
    checks = []
    _expect(
        _outcomes(first),
        _outcomes(second),
        "Retrying unchanged capacity preserves the failure and skip",
        checks,
    )
    _expect(
        first["snapshot"],
        second["snapshot"],
        "Retrying the blocked queue does not leak allocations",
        checks,
    )
    controls.append(
        {
            "name": "unchanged_retry",
            "passed": True,
            "checks": checks,
            "observations": [first, second],
        }
    )
    return controls


async def _run() -> dict[str, object]:
    provenance = _provenance()
    state = FixtureState()
    events: list[dict[str, object]] = []
    checks: list[str] = []

    def append_event(kind: str, actor: str, description: str, **result: object) -> None:
        events.append(
            {
                "sequence": len(events),
                "kind": kind,
                "recorded_at": datetime.now(UTC).isoformat(),
                "action": {"actor": actor, "description": description},
                "decisions": [],
                "snapshot": state.snapshot(),
                **result,
            }
        )

    append_event(
        "initial_state",
        "fixture",
        "Supply four CPU slots, a two-slot allocation, then Alice and Bob in FIFO order.",
    )

    first = await schedule_pass(state)
    _expect(
        _outcomes(first),
        {"alice": "failed", "bob": "skipped"},
        "Alice fails capacity and Bob is left unattempted",
        checks,
    )
    _expect(
        state.allocations,
        {"current": 2},
        "The first pass leaves two free slots unallocated",
        checks,
    )
    append_event("scheduling_pass", "scheduler", "Run the real Backend.AI provisioner.", **first)

    state.release("current")
    append_event(
        "fixture_release", "fixture", "Supply a new snapshot with the current allocation removed."
    )
    second = await schedule_pass(state)
    _expect(
        _outcomes(second),
        {"alice": "scheduled", "bob": "failed"},
        "Alice receives four slots and Bob then fails capacity",
        checks,
    )
    _expect(state.allocations, {"alice": 4}, "Alice's allocation fills the worker", checks)
    append_event(
        "scheduling_pass", "scheduler", "Run the provisioner after the supplied release.", **second
    )

    state.release("alice")
    append_event(
        "fixture_release", "fixture", "Supply a new snapshot with Alice's allocation removed."
    )
    third = await schedule_pass(state)
    _expect(
        _outcomes(third),
        {"bob": "scheduled"},
        "Bob receives one slot after the supplied Alice release",
        checks,
    )
    _expect(state.allocations, {"bob": 1}, "The last pass leaves three free slots", checks)
    _expect(state.pending, [], "The pending queue is empty after Bob is scheduled", checks)
    append_event(
        "scheduling_pass", "scheduler", "Run the provisioner for Bob's remaining request.", **third
    )

    return {
        "schema_version": 1,
        "experiment": {
            "name": "queue-and-release",
            "title": "Queue order and resource release",
            "mode": "scheduler_component",
            "resource": "cpu",
            "capacity": 4,
            "scope": (
                "Real Backend.AI scheduling code; synthetic inventory; in-memory persistence. "
                "Release snapshots are supplied inputs. No manager, agent, database, container, "
                "or workload lifecycle runs. Allocated does not mean running."
            ),
            "assumptions": [
                "One compatible synthetic worker with four CPU slots and 16 GiB memory",
                "Equal priorities; FIFO preserves the supplied Alice-before-Bob list",
                "Each session requests 1 GiB; memory is not the limiting resource",
                "Main-scenario user quotas allow four CPUs; project and domain quotas are absent",
                "Preemption, container limits, dependencies, and session groups are inactive",
            ],
        },
        "provenance": provenance,
        "events": events,
        "controls": await _controls(),
        "verification": {"passed": True, "checks": checks},
    }


def run_experiment() -> dict[str, object]:
    """Return a JSON-serializable recording or raise if an expectation fails."""
    return asyncio.run(_run())
