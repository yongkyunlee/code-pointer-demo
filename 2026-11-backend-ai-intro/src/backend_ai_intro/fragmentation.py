"""Exercise Backend.AI's installed placement code with synthetic GPU inventories.

The harness supplies resource observations and requests. Backend.AI's selector
factory, filters, strategies and state trackers make every placement decision.
No queue, process, GPU execution or throughput is simulated.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

VERSION = "26.9.7"
RELEASE_COMMIT = "332a308e34c5c7bcd57939850fe032cfd8d8ca00"
CASE_INPUTS = (
    ("small_first", (1, 3, 4), ("node-a", "node-b")),
    ("reversed_node_order", (1, 3, 4), ("node-b", "node-a")),
    ("largest_first", (4, 1, 3), ("node-a", "node-b")),
)
STRATEGIES = ("concentrated", "dispersed")


class FragmentationError(RuntimeError):
    """The installed implementation or recorded evidence failed a required check."""


def _require(condition: bool, message: str) -> None:
    # Ordinary checks remain active under python -O.
    if not condition:
        raise FragmentationError(message)


def _source_evidence() -> list[dict[str, Any]]:
    """Hash loaded placement modules and verify their installed wheel RECORD hashes."""
    distributions = {
        name: importlib.metadata.distribution(name)
        for name in ("backend.ai-manager", "backend.ai-common")
    }
    records = {
        name: {str(path): path for path in distribution.files or []}
        for name, distribution in distributions.items()
    }
    evidence = []
    for name, module in sorted(sys.modules.items()):
        if not (
            name.startswith("ai.backend.manager.sokovan.scheduler.provisioner.selectors.")
            or name.startswith("ai.backend.manager.views.sokovan.")
            or name
            in {
                "ai.backend.common.types",
                "ai.backend.common.data.entity.resource_slot",
                "ai.backend.manager.data.session.options",
            }
        ):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        owner = (
            "backend.ai-common" if name.startswith("ai.backend.common.") else "backend.ai-manager"
        )
        root = Path(distributions[owner].locate_file("")).resolve()
        _require(
            path.is_relative_to(root), f"{name} was imported outside its installed distribution"
        )
        relative = str(path.relative_to(root))
        entry = records[owner].get(relative)
        _require(entry is not None and entry.hash is not None, f"Missing wheel hash for {name}")
        _require(entry.hash.mode == "sha256", f"Unsupported wheel hash for {name}")
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        _require(encoded == entry.hash.value, f"Installed source differs from wheel RECORD: {name}")
        evidence.append(
            {
                "module": name,
                "distribution": owner,
                "path": relative,
                "sha256": digest.hex(),
                "matches_wheel_record": True,
            }
        )
    _require(len(evidence) > 5, "Placement source evidence is incomplete")
    return evidence


async def _cases() -> list[dict[str, Any]]:
    from ai.backend.common.data.entity.resource_group import ResourceGroupID
    from ai.backend.common.data.entity.resource_slot import ResourceSlotName
    from ai.backend.common.types import (
        AgentId,
        AgentSelectionStrategy,
        ArchName,
        PreemptionOrder,
        SessionId,
    )
    from ai.backend.manager.data.session.options import AgentSelectionPolicy
    from ai.backend.manager.sokovan.scheduler.provisioner.selectors.pool import (
        create_agent_selector,
    )
    from ai.backend.manager.sokovan.scheduler.provisioner.selectors.selector import (
        AgentSelectionCriteria,
    )
    from ai.backend.manager.sokovan.scheduler.provisioner.selectors.tracker import AgentStateTracker
    from ai.backend.manager.sokovan.scheduler.provisioner.selectors.types import (
        ResourceRequirements,
    )
    from ai.backend.manager.views.sokovan.agent import (
        AgentInfo,
        AgentLimit,
        AgentResource,
        SlotResource,
    )
    from ai.backend.manager.views.sokovan.workload import ResourceRequest

    gpu = ResourceSlotName("cuda.device")

    def remaining(trackers: list[AgentStateTracker]) -> dict[str, int]:
        values = {}
        for tracker in trackers:
            value = tracker.remaining_slots()[gpu]
            _require(value == value.to_integral_value(), "Unexpected fractional whole-GPU slots")
            values[str(tracker.original_agent.agent_id)] = int(value)
        return values

    cases = []
    for case_id, requests, names in CASE_INPUTS:
        runs = []
        for strategy_name in STRATEGIES:
            strategy = AgentSelectionStrategy(strategy_name)
            selector = create_agent_selector(["cuda.device", "cpu", "mem"])
            trackers = [
                AgentStateTracker(
                    original_agent=AgentInfo(
                        agent_id=AgentId(name),
                        agent_addr=f"tcp://{name}:6001",
                        architecture=ArchName("x86_64"),
                        container_count=0,
                        resources=AgentResource(
                            slots={
                                gpu: SlotResource(
                                    capacity=Decimal(4), reserved=Decimal(0), used=Decimal(0)
                                )
                            }
                        ),
                    )
                )
                for name in names
            ]
            steps = []
            for index, amount in enumerate(requests):
                before = remaining(trackers)
                criteria = AgentSelectionCriteria(
                    session_id=SessionId(UUID(int=index + 1)),
                    resource_group_id=ResourceGroupID(UUID(int=100)),
                    requirements=[
                        ResourceRequirements(
                            requested_slots=ResourceRequest(slots={gpu: Decimal(amount)}),
                            required_architecture=ArchName("x86_64"),
                            container_count=1,
                        )
                    ],
                    agent_selection_policy=AgentSelectionPolicy.PREFERRED,
                    designated_agent_ids=None,
                    job_priority=0,
                    victim_candidates=None,
                    session_group=None,
                )
                result = await selector.compute_placements(
                    strategy,
                    trackers,
                    criteria,
                    AgentLimit(max_container_count=None),
                    PreemptionOrder.OLDEST,
                )
                steps.append(
                    {
                        "sequence": index + 1,
                        "request_id": f"request-{index + 1}",
                        "requested_gpu_slots": amount,
                        "status": "cannot_place" if result.failures else "placed",
                        "selected_agents": [
                            str(selection.selected_agent.agent_id)
                            for selection in result.selections
                        ],
                        "preempting_session_ids": [
                            str(victim)
                            for selection in result.selections
                            for victim in selection.preempting_session_ids
                        ],
                        "free_before": before,
                        "free_after": remaining(trackers),
                        "failures": [
                            {
                                "requirement_index": failure.requirement_index,
                                "filter": failure.filter_name,
                                "missing_slots_on_best_candidate": {
                                    str(slot): str(value)
                                    for slot, value in failure.missing_slots.items()
                                },
                            }
                            for failure in result.failures
                        ],
                    }
                )
            runs.append({"strategy": strategy_name, "steps": steps})
        cases.append(
            {
                "id": case_id,
                "node_input_order": list(names),
                "request_order": list(requests),
                "strategies": runs,
            }
        )
    return cases


def validate_recording(recording: dict[str, Any]) -> None:
    """Check recorded outcomes and conservation before reporting success or drawing."""
    _require(recording.get("schema_version") == 1, "Expected schema_version 1")
    experiment = recording.get("experiment", {})
    _require(experiment.get("mode") == "component_synthetic_gpu", "Wrong experiment mode")
    cases = recording.get("cases", [])
    _require(len(cases) == len(CASE_INPUTS), "Missing experiment or control case")
    for case, (case_id, requests, names) in zip(cases, CASE_INPUTS, strict=True):
        _require(case["id"] == case_id, "Unexpected case order")
        _require(case["node_input_order"] == list(names), "Unexpected node order")
        _require(case["request_order"] == list(requests), "Unexpected request order")
        runs = case["strategies"]
        _require([run["strategy"] for run in runs] == list(STRATEGIES), "Missing strategy")
        for run in runs:
            _require(len(run["steps"]) == len(requests), "Incomplete request history")
            previous = dict.fromkeys(names, 4)
            for index, (step, amount) in enumerate(zip(run["steps"], requests, strict=True)):
                _require(step["sequence"] == index + 1, "Unexpected step sequence")
                _require(step["request_id"] == f"request-{index + 1}", "Unexpected request id")
                _require(step["requested_gpu_slots"] == amount, "Unexpected GPU request")
                _require(step["free_before"] == previous, "Discontinuous resource accounting")
                after = step["free_after"]
                _require(set(after) == set(names), "Unexpected node inventory")
                _require(
                    all(type(value) is int and 0 <= value <= 4 for value in after.values()),
                    "Invalid free slot count",
                )
                _require(not step["preempting_session_ids"], "Unexpected preemption")
                should_fail = (
                    run["strategy"] == "dispersed" and requests == (1, 3, 4) and index == 2
                )
                if should_fail:
                    _require(step["status"] == "cannot_place", "Expected fragmentation rejection")
                    _require(not step["selected_agents"], "Rejected request selected an agent")
                    _require(after == previous, "Rejected request leaked an allocation")
                    _require(sorted(after.values()) == [1, 3], "Expected fragmented free capacity")
                    _require(len(step["failures"]) == 1, "Missing placement failure evidence")
                    failure = step["failures"][0]
                    _require(failure["requirement_index"] == 0, "Unexpected failed requirement")
                    _require(failure["filter"] == "resource", "Expected resource-fit failure")
                    _require(
                        failure["missing_slots_on_best_candidate"] == {"cuda.device": "1"},
                        "Expected one-GPU shortfall on the best candidate",
                    )
                else:
                    _require(step["status"] == "placed", "Expected successful placement")
                    _require(not step["failures"], "Placed request contains failures")
                    selected = step["selected_agents"]
                    _require(len(selected) == 1 and selected[0] in names, "Invalid selected node")
                    expected = previous.copy()
                    expected[selected[0]] -= amount
                    _require(after == expected, "Placement accounting does not match the request")
                previous = after
            expected_end = (
                [1, 3] if run["strategy"] == "dispersed" and requests == (1, 3, 4) else [0, 0]
            )
            _require(sorted(previous.values()) == expected_end, "Unexpected final inventory")


def run_experiment() -> dict[str, Any]:
    """Run all cases against the pinned installed package, or raise on any failure."""
    _require(
        sys.version_info[:2] == (3, 13) and sys.version_info.micro >= 18,
        "This experiment requires Python >=3.13.18,<3.14",
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("backend.ai-manager", "backend.ai-common")
    }
    _require(all(value == VERSION for value in versions.values()), f"Expected Backend.AI {VERSION}")
    cases = asyncio.run(_cases())
    project = Path(__file__).resolve().parents[2]
    lock = project / "uv.lock"
    recording = {
        "schema_version": 1,
        "experiment": {
            "id": "placement-fragmentation",
            "mode": "component_synthetic_gpu",
            "scope": (
                "Real factory-created AgentSelector.compute_placements, filters, strategies and "
                "state trackers; synthetic whole-GPU inventories. Placement acceptance is not "
                "container execution. No services, database, queue, CUDA, GPU workload or timing."
            ),
            "request_semantics": "Each request must fit on one node; requests remain allocated.",
        },
        "fixture": {
            "resource_slot": "cuda.device",
            "gpu_slots_per_node": 4,
            "architecture": "x86_64",
            "initial_reserved_slots": 0,
            "initial_used_slots": 0,
            "initial_container_count": 0,
            "container_limit": None,
            "resource_priority": ["cuda.device", "cpu", "mem"],
            "omitted_resource_dimensions": ["cpu", "mem"],
            "designated_agents": None,
            "session_group": None,
            "preemption_candidates": None,
        },
        "provenance": {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "installed_distributions": versions,
            "source": "Installed PyPI distributions; loaded source matches wheel RECORD hashes.",
            "upstream_release_tag": VERSION,
            "upstream_release_commit_reference": RELEASE_COMMIT,
            "release_reference_note": (
                "The commit identifies the release source reference, not an embedded wheel commit."
            ),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
            if lock.exists()
            else None,
            "modules": _source_evidence(),
        },
        "cases": cases,
    }
    validate_recording(recording)
    recording["checks"] = {
        "all_passed": True,
        "concentrated_places_all_three": True,
        "dispersed_rejects_last_request_with_four_free_slots": True,
        "reversed_node_order_preserves_fit_result": True,
        "largest_first_allows_both_strategies_to_place_all_requests": True,
        "every_step_conserves_capacity": True,
        "rejected_request_leaves_inventory_unchanged": True,
        "loaded_source_matches_wheel_record": True,
    }
    return recording


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fragmentation/recording.json")
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--recording", type=Path, help="Validate and render an existing recording")
    args = parser.parse_args(argv)
    try:
        if args.render_dir:
            from backend_ai_intro.fragmentation_render import invalidate_outputs

            invalidate_outputs(args.render_dir, preserve=args.recording or args.output)
        if not args.recording:
            # A failed rerun must not leave a previous successful recording at this target.
            args.output.unlink(missing_ok=True)
        if args.recording:
            recording = json.loads(args.recording.read_text())
            validate_recording(recording)
        else:
            recording = run_experiment()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(recording, indent=2, sort_keys=True) + "\n")
            print(f"Recorded placement evidence: {args.output}")
        for case in recording["cases"]:
            for run in case["strategies"]:
                statuses = ", ".join(step["status"] for step in run["steps"])
                print(f"{case['id']} / {run['strategy']}: {statuses}")
        if args.render_dir:
            from backend_ai_intro.fragmentation_render import render_recording

            # Reload newly written JSON so figures derive from the saved evidence.
            saved = json.loads((args.recording or args.output).read_text())
            for kind, path in render_recording(
                saved, args.render_dir, source_recording=args.recording or args.output
            ).items():
                print(f"{kind}: {path}")
    except (FragmentationError, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Fragmentation experiment failed: {exc}", file=sys.stderr)
        return 1
    print("All placement checks passed. GPU inventory is synthetic; no GPU work was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
