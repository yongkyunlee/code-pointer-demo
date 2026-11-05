"""Record an actual CPU-only Backend.AI session lifecycle through its public SDK.

This module never substitutes a scheduler or changes database state. It creates
three ordinary interactive sessions in an otherwise empty, four-CPU resource
group, observes scheduling history, and terminates only its own sessions.
Credentials are read from the environment and are never included in recordings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

BACKEND_AI_VERSION = "26.9.7"
TERMINAL_STATES = frozenset({"TERMINATED", "CANCELLED", "ERROR"})
REQUESTS = {"current": 2, "alice": 4, "bob": 1}
LABELS = {"current": "Current", "alice": "Alice", "bob": "Bob"}


class LiveExperimentError(RuntimeError):
    """The observed system did not satisfy the experiment's prerequisites or checks."""


@dataclass(frozen=True)
class LiveConfig:
    endpoint: str
    image: str
    resource_group: str = "default"
    domain: str = "default"
    project: str = "default"
    architecture: str = "aarch64"
    poll_interval: float = 1.0
    stage_timeout: float = 300.0
    request_timeout: float = 30.0
    memory: str = "1g"

    def validate(self) -> None:
        address = urlsplit(self.endpoint)
        if address.scheme not in {"http", "https"} or address.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Use an isolated local manager endpoint on localhost")
        if address.username or address.password or address.query or address.fragment:
            raise ValueError("The endpoint must not contain credentials, queries, or fragments")
        if not self.image.strip():
            raise ValueError("A registered Backend.AI image is required")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.resource_group):
            raise ValueError("Resource-group name contains unsupported characters")
        if not 0.1 <= self.poll_interval <= 30:
            raise ValueError("poll_interval must be between 0.1 and 30 seconds")
        if not 5 <= self.stage_timeout <= 3600:
            raise ValueError("stage_timeout must be between 5 and 3600 seconds")
        if not 1 <= self.request_timeout <= 120:
            raise ValueError("request_timeout must be between 1 and 120 seconds")


def _slots(value: Any, label: str) -> dict[str, str]:
    """Decode GraphQL JSONString fields without executing resource representations."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise LiveExperimentError(f"{label} is not a resource-slot object")
    result = {}
    for key, raw_amount in value.items():
        try:
            amount = Decimal(str(raw_amount))
        except InvalidOperation as error:
            raise LiveExperimentError(f"{label}.{key} is not numeric") from error
        if not amount.is_finite() or amount < 0:
            raise LiveExperimentError(f"{label}.{key} must be finite and nonnegative")
        result[str(key)] = str(amount)
    return result


def _cpu(slots: Mapping[str, str]) -> Decimal:
    return Decimal(slots.get("cpu", "0"))


def _json_object(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Remove credentials even if a server diagnostic echoes them in free text."""
    if isinstance(value, str):
        for secret in sorted((item for item in secrets if item), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if re.search(r"secret|password|authorization|access[_-]?key", key, re.I)
                else _redact(item, secrets)
            )
            for key, item in value.items()
        }
    return value


class _Recorder:
    def __init__(
        self,
        config: LiveConfig,
        output: Path,
        *,
        owned_session_ids: set[str] | None = None,
        owned_session_names: set[str] | None = None,
    ) -> None:
        self.config = config
        self.output = output
        self.start = time.monotonic()
        self.started_at = datetime.now(UTC)
        self.run_id = uuid4().hex
        self.sessions: dict[str, Any] = {}
        self.creation_attempts: dict[str, str] = {}
        self.owned_session_ids = owned_session_ids if owned_session_ids is not None else set()
        self.owned_session_names = owned_session_names if owned_session_names is not None else set()
        self.secrets = (
            os.environ.get("BACKEND_ACCESS_KEY", ""),
            os.environ.get("BACKEND_SECRET_KEY", ""),
        )
        self.agent_id: str | None = None
        self.history: dict[str, dict[str, dict[str, Any]]] = {}
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "experiment": {
                "mode": "live_cpu",
                "name": "CPU queue order and session release",
                "scope": "Actual manager, agent, containers, and public API observations",
                "sampling_limit": (
                    "Session and agent queries are sequential, not an atomic snapshot"
                ),
            },
            "configuration": {
                "capacity": 4,
                "resource_group": config.resource_group,
                "image": config.image,
                "architecture": config.architecture,
                "poll_interval_seconds": config.poll_interval,
                "stage_timeout_seconds": config.stage_timeout,
                "session_requests": REQUESTS,
            },
            "provenance": {
                "client_version": version("backend.ai-client"),
                "python": platform.python_version(),
                "platform": platform.system(),
                "architecture": platform.machine(),
            },
            "samples": [],
            "actions": [],
            "checkpoints": [],
            "scheduling_history": {},
            "status_history": {},
            "verification": {"passed": False, "checks": []},
            "cleanup": {"complete": False, "errors": []},
        }

    def clock(self) -> dict[str, Any]:
        return {
            "observed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.start, 6),
        }

    def save(self) -> None:
        self.data["scheduling_history"] = {
            label: sorted(rows.values(), key=lambda row: row["created_at"])
            for label, rows in self.history.items()
        }
        _write_json(self.output, _redact(self.data, self.secrets))

    async def bounded[T](self, operation: Awaitable[T]) -> T:
        async with asyncio.timeout(self.config.request_timeout):
            return await operation

    def action(self, stage: str, operation: str, label: str, **details: Any) -> None:
        self.data["actions"].append(
            {
                "sequence": len(self.data["actions"]),
                **self.clock(),
                "stage": stage,
                "actor": "experiment operator",
                "operation": operation,
                "session": label,
                **details,
            }
        )
        self.save()

    async def preflight(self, api: Any) -> None:
        from ai.backend.client.output.fields import agent_fields, scaling_group_fields

        manager_status = await self.bounded(api.Manager.status())
        manager_versions = {node["version"] for node in manager_status.get("nodes", [])}
        if manager_versions != {BACKEND_AI_VERSION}:
            raise LiveExperimentError("The live manager must run Backend.AI 26.9.7")
        self.data["provenance"]["manager_versions"] = sorted(manager_versions)

        group = await self.bounded(
            api.ScalingGroup.detail(
                self.config.resource_group,
                fields=tuple(
                    scaling_group_fields[key] for key in ("name", "is_active", "scheduler")
                ),
            )
        )
        if not group or not group["is_active"] or group["scheduler"].lower() != "fifo":
            raise LiveExperimentError("The experiment requires an active FIFO resource group")
        deadline = time.monotonic() + self.config.stage_timeout
        while True:
            agents = await self.bounded(
                api.Agent.paginated_list(
                    status="ALIVE",
                    scaling_group=self.config.resource_group,
                    page_size=2,
                    fields=tuple(
                        agent_fields[key]
                        for key in (
                            "id",
                            "status",
                            "scaling_group",
                            "available_slots",
                            "occupied_slots",
                            "version",
                        )
                    ),
                )
            )
            if agents.total_count:
                break
            if time.monotonic() >= deadline:
                raise LiveExperimentError("Timed out waiting for the CPU agent to become ALIVE")
            await asyncio.sleep(self.config.poll_interval)
        if agents.total_count != 1 or len(agents.items) != 1:
            raise LiveExperimentError("The experiment requires exactly one ALIVE agent")
        agent = agents.items[0]
        if agent["version"] != BACKEND_AI_VERSION:
            raise LiveExperimentError("The live agent must run Backend.AI 26.9.7")
        self.data["provenance"]["agent_version"] = agent["version"]
        if _cpu(_slots(agent["available_slots"], "agent.available_slots")) != 4:
            raise LiveExperimentError("Configure the agent to announce exactly four CPU slots")
        if _cpu(_slots(agent["occupied_slots"], "agent.occupied_slots")) != 0:
            raise LiveExperimentError("The resource group already has CPU allocations")
        pending = await self.bounded(
            api.ComputeSession.paginated_list(
                status="PENDING",
                filter=f'scaling_group == "{self.config.resource_group}"',
                page_size=1,
            )
        )
        if pending.total_count:
            raise LiveExperimentError("The resource group already has pending sessions")
        self.agent_id = str(agent["id"])
        self.data["verification"]["checks"].append(
            "Preflight found one empty four-CPU agent and an active FIFO resource group"
        )
        self.save()

    async def create(self, api: Any, label: str) -> None:
        from ai.backend.client.exceptions import BackendAPIError

        name = f"cp-demo-{self.run_id}-{label}"
        self.action("submit", "create_requested", label, name=name, requested_cpu=REQUESTS[label])
        self.creation_attempts[label] = name
        self.owned_session_names.add(name)
        try:
            session = await self.bounded(
                api.ComputeSession.get_or_create(
                    self.config.image,
                    name=name,
                    type_="interactive",
                    priority=10,
                    enqueue_only=True,
                    no_reuse=True,
                    resources={"cpu": REQUESTS[label], "mem": self.config.memory},
                    architecture=self.config.architecture,
                    domain_name=self.config.domain,
                    group_name=self.config.project,
                    scaling_group=self.config.resource_group,
                    mounts=[],
                )
            )
        except BackendAPIError as error:
            if 400 <= error.status < 500:
                # A definitive rejection must not cause cleanup of a preexisting name.
                self.creation_attempts.pop(label)
                self.owned_session_names.discard(name)
            raise
        if not session.created or session.id is None:
            self.creation_attempts.pop(label)
            self.owned_session_names.discard(name)
            raise LiveExperimentError("Session creation did not return a new session identifier")
        self.owned_session_ids.add(str(session.id))
        self.sessions[label] = session
        self.creation_attempts.pop(label)
        self.action("submit", "create_accepted", label, session_id=str(session.id))

    async def recover_creation(self, api: Any, label: str, name: str) -> None:
        """Find a server-accepted session whose create response was interrupted."""
        from ai.backend.client.output.fields import session_fields

        response = await self.bounded(
            api.ComputeSession.paginated_list(
                filter=f'name == "{name}"',
                page_size=2,
                fields=tuple(session_fields[key] for key in ("id", "name", "created_at")),
            )
        )
        if not response.total_count:
            return
        if response.total_count != 1 or len(response.items) != 1:
            raise LiveExperimentError("Cannot safely identify the interrupted creation")
        item = response.items[0]
        created_at = datetime.fromisoformat(item["created_at"])
        if item["name"] != name or created_at < self.started_at:
            raise LiveExperimentError("Interrupted creation lookup did not match this run")
        self.owned_session_ids.add(str(UUID(item["id"])))
        self.owned_session_names.add(name)
        self.sessions[label] = api.ComputeSession.from_session_id(UUID(item["id"]))
        self.action("cleanup", "interrupted_creation_found", label, session_id=item["id"])

    async def terminate(self, label: str, stage: str) -> None:
        session = self.sessions[label]
        self.action(stage, "termination_requested", label, session_id=str(session.id))
        await self.bounded(session.destroy())
        self.action(stage, "termination_accepted", label, session_id=str(session.id))

    async def get_history(self, api: Any, label: str) -> list[dict[str, Any]]:
        from ai.backend.common.dto.manager.query import UUIDFilter
        from ai.backend.common.dto.manager.scheduling_history import (
            OrderDirection,
            SearchSessionHistoryRequest,
            SessionHistoryFilter,
            SessionHistoryOrder,
            SessionHistoryOrderField,
        )

        response = await self.bounded(
            api.SchedulingHistory.list_session_history(
                SearchSessionHistoryRequest(
                    filter=SessionHistoryFilter(
                        session_id=UUIDFilter(equals=self.sessions[label].id)
                    ),
                    order=[
                        SessionHistoryOrder(
                            field=SessionHistoryOrderField.CREATED_AT,
                            direction=OrderDirection.DESC,
                        )
                    ],
                    limit=100,
                ),
            )
        )
        rows = [row.model_dump(mode="json") for row in response.items]
        saved = self.history.setdefault(label, {})
        for row in rows:
            saved[row["id"]] = row
        return rows[:10]

    async def sample(self, api: Any, stage: str) -> dict[str, Any]:
        from ai.backend.client.output.fields import agent_fields, session_node_fields

        session_fields = tuple(
            session_node_fields[key]
            for key in (
                "id",
                "status",
                "status_info",
                "status_data",
                "requested_slots",
                "occupied_slots",
                "created_at",
                "terminated_at",
            )
        )
        observed = []
        history = {}
        for label, session in self.sessions.items():
            item = await self.bounded(session.detail(fields=session_fields))
            if not item:
                raise LiveExperimentError(f"Session {label} is missing from the manager response")
            observed.append(
                {
                    "id": label,
                    "label": LABELS[label],
                    "session_id": str(session.id),
                    "requested_cpu": REQUESTS[label],
                    "status": item["status"],
                    "status_info": item.get("status_info"),
                    "status_data": _json_object(item.get("status_data")),
                    "requested_slots": _slots(item["requested_slots"], "session.requested_slots"),
                    "occupied_slots": _slots(item["occupied_slots"], "session.occupied_slots"),
                    "created_at": item.get("created_at"),
                    "terminated_at": item.get("terminated_at"),
                }
            )
            history[label] = await self.get_history(api, label)
        agent = await self.bounded(
            api.Agent.detail(
                self.agent_id,
                fields=tuple(
                    agent_fields[key]
                    for key in (
                        "id",
                        "status",
                        "available_slots",
                        "occupied_slots",
                    )
                ),
            )
        )
        if not isinstance(agent, Mapping):
            raise LiveExperimentError("Agent detail is missing from the manager response")
        sample = {
            "sequence": len(self.data["samples"]),
            **self.clock(),
            "stage": stage,
            "sessions": observed,
            "agent": {
                "id": str(agent["id"]),
                "status": agent["status"],
                "available_slots": _slots(agent["available_slots"], "agent.available_slots"),
                "occupied_slots": _slots(agent["occupied_slots"], "agent.occupied_slots"),
            },
            "latest_scheduling_history": history,
        }
        self.data["samples"].append(sample)
        self.save()
        return sample

    async def wait(
        self,
        api: Any,
        stage: str,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        allow_error_states: bool = False,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self.config.stage_timeout):
                while True:
                    sample = await self.sample(api, stage)
                    for session in sample["sessions"]:
                        if session["status"] == "ERROR" and not allow_error_states:
                            raise LiveExperimentError(
                                f"Session {session['id']} entered ERROR during {stage}"
                            )
                    if predicate(sample):
                        return sample
                    await asyncio.sleep(self.config.poll_interval)
        except TimeoutError as error:
            raise LiveExperimentError(
                f"Timed out waiting for {stage}; partial observations saved"
            ) from error

    def checkpoint(self, label: str, sample: dict[str, Any], summary: str) -> None:
        self.data["checkpoints"].append(
            {
                "label": label,
                "sample_sequence": sample["sequence"],
                "decision_summary": summary,
            }
        )
        self.data["verification"]["checks"].append(summary)
        self.save()


def _session(sample: dict[str, Any], label: str) -> dict[str, Any]:
    return next(item for item in sample["sessions"] if item["id"] == label)


def _running_with_cpu(sample: dict[str, Any], label: str, cpu: int) -> bool:
    item = _session(sample, label)
    return item["status"] == "RUNNING" and _cpu(item["occupied_slots"]) == cpu


def _pending_schedule_result(row: dict[str, Any], result: str) -> bool:
    return (
        row.get("result") == result
        and row.get("phase") == "schedule-sessions"
        and row.get("from_status") == "PENDING"
        and row.get("to_status") == "PENDING"
    )


def _fifo_pass(row: dict[str, Any]) -> tuple[str, str] | None:
    """The pinned scheduler shares this sequencing step across one scheduling pass."""
    for step in row.get("sub_steps", []):
        if (
            step.get("step") == "FIFOSequencer"
            and step.get("result") == "SUCCESS"
            and step.get("started_at")
            and step.get("ended_at")
        ):
            return step["started_at"], step["ended_at"]
    return None


def _alice_cpu_shortfall(row: dict[str, Any]) -> bool:
    if not _pending_schedule_result(row, "NEED_RETRY"):
        return False
    steps = row.get("sub_steps", [])
    if not any(
        step.get("step") == "ResourcePolicyValidator" and step.get("result") == "SUCCESS"
        for step in steps
    ) or not any(
        step.get("step") == "resource" and step.get("result") == "FAILURE" for step in steps
    ):
        return False
    # Backend.AI 26.9.7 reports deficits in the selector's message, not a typed field.
    # Reject additional deficits so memory or another resource cannot explain the wait.
    deficits = re.findall(r"^\s*- ([^:\n]+): missing=(.+)$", row.get("message") or "", re.M)
    if len(deficits) != 1 or deficits[0][0] != "cpu":
        return False
    try:
        return Decimal(deficits[0][1].rstrip(")")) == 2
    except InvalidOperation:
        return False


def _skipped_while_pending(sample: dict[str, Any]) -> bool:
    alice = _session(sample, "alice")
    if not (
        _running_with_cpu(sample, "current", 2)
        and alice["status"] == "PENDING"
        and _session(sample, "bob")["status"] == "PENDING"
        and _cpu(sample["agent"]["available_slots"]) == 4
        and _cpu(sample["agent"]["occupied_slots"]) == 2
    ):
        return False
    history = sample["latest_scheduling_history"]
    alice_attempts = [row for row in history.get("alice", []) if _alice_cpu_shortfall(row)]
    for row in history.get("bob", []):
        if not _pending_schedule_result(row, "SKIPPED"):
            continue
        if row.get("message") != (
            "Not attempted: the resource group ran out of resources at the earlier session "
            f"{alice['session_id']}"
        ):
            continue
        fifo_pass = _fifo_pass(row)
        if fifo_pass and any(_fifo_pass(attempt) == fifo_pass for attempt in alice_attempts):
            return True
    return False


@asynccontextmanager
async def _connected_session(config: LiveConfig, api_config: Any, http: Any) -> AsyncIterator[Any]:
    """Allow a freshly launched manager to become reachable before submitting work."""
    import aiohttp
    from ai.backend.client.exceptions import BackendAPIError, BackendClientError
    from ai.backend.client.session import AsyncSession

    deadline = time.monotonic() + config.stage_timeout
    while True:
        api = AsyncSession(config=api_config, aiohttp_session=http)
        try:
            async with asyncio.timeout(config.request_timeout):
                await api.open()
                await api.Manager.status()
            break
        except (
            BackendAPIError,
            BackendClientError,
            aiohttp.ClientError,
            OSError,
            TimeoutError,
        ) as error:
            await api.close()
            if isinstance(error, BackendAPIError) and error.status in {401, 403}:
                raise LiveExperimentError(
                    "The manager rejected the demo API credentials"
                ) from error
            if time.monotonic() >= deadline:
                raise LiveExperimentError("Timed out waiting for the manager API") from error
            await asyncio.sleep(config.poll_interval)
    try:
        yield api
    finally:
        await api.close()


async def _run(
    config: LiveConfig,
    output: Path,
    *,
    owned_session_ids: set[str] | None = None,
    owned_session_names: set[str] | None = None,
) -> dict[str, Any]:
    import aiohttp
    from ai.backend.client.config import APIConfig
    from ai.backend.client.output.fields import session_node_fields

    access_key = os.environ.get("BACKEND_ACCESS_KEY", "")
    secret_key = os.environ.get("BACKEND_SECRET_KEY", "")
    if not access_key or not secret_key:
        raise LiveExperimentError(
            "Set BACKEND_ACCESS_KEY and BACKEND_SECRET_KEY for the demo account"
        )
    if os.environ.get("BACKEND_VFOLDER_MOUNTS") or os.environ.get("SORNA_VFOLDER_MOUNTS"):
        raise LiveExperimentError("Unset automatic vfolder mounts for this isolated experiment")
    recorder = _Recorder(
        config,
        output,
        owned_session_ids=owned_session_ids,
        owned_session_names=owned_session_names,
    )
    api_config = APIConfig(
        endpoint=config.endpoint,
        endpoint_type="api",
        domain=config.domain,
        group=config.project,
        access_key=access_key,
        secret_key=secret_key,
        vfolder_mounts=[],
        connection_timeout=10,
        read_timeout=config.request_timeout,
    )
    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        async with _connected_session(config, api_config, http) as api:
            try:
                await recorder.preflight(api)
                await recorder.create(api, "current")
                sample = await recorder.wait(
                    api,
                    "current_running",
                    lambda row: (
                        _running_with_cpu(row, "current", 2)
                        and _cpu(row["agent"]["occupied_slots"]) == 2
                    ),
                )
                recorder.checkpoint(
                    "Current running", sample, "Current is RUNNING with two CPU slots"
                )
                await recorder.create(api, "alice")
                await recorder.create(api, "bob")
                sample = await recorder.wait(api, "queue_blocked", _skipped_while_pending)
                recorder.checkpoint(
                    "Alice and Bob pending",
                    sample,
                    "Alice passed policy validation but lacked two CPU slots; "
                    "the same FIFO pass skipped Bob behind Alice while Current remained RUNNING",
                )
                await recorder.terminate("current", "end_current")
                sample = await recorder.wait(
                    api,
                    "alice_running",
                    lambda row: (
                        _running_with_cpu(row, "alice", 4)
                        and _session(row, "current")["status"] == "TERMINATED"
                        and _cpu(_session(row, "current")["occupied_slots"]) == 0
                        and _session(row, "bob")["status"] == "PENDING"
                        and _cpu(row["agent"]["occupied_slots"]) == 4
                    ),
                )
                recorder.checkpoint(
                    "Alice running",
                    sample,
                    "Current has released its allocation; Alice is RUNNING with four CPU slots "
                    "and Bob is PENDING",
                )
                await recorder.terminate("alice", "end_alice")
                sample = await recorder.wait(
                    api,
                    "bob_running",
                    lambda row: (
                        _running_with_cpu(row, "bob", 1)
                        and _session(row, "alice")["status"] == "TERMINATED"
                        and _cpu(_session(row, "alice")["occupied_slots"]) == 0
                        and _cpu(row["agent"]["occupied_slots"]) == 1
                    ),
                )
                recorder.checkpoint(
                    "Bob running",
                    sample,
                    "Alice has released her allocation; Bob is RUNNING with one CPU slot",
                )
                recorder.data["verification"]["passed"] = True
            except BaseException as error:
                recorder.data["failure"] = {"type": type(error).__name__}
                recorder.save()
                raise
            finally:
                cleanup_errors = []
                for label, name in recorder.creation_attempts.items():
                    try:
                        await recorder.recover_creation(api, label, name)
                    except Exception as error:
                        cleanup_errors.append(
                            {
                                "session": label,
                                "stage": "recover_creation",
                                "type": type(error).__name__,
                            }
                        )
                for label in reversed(recorder.sessions):
                    try:
                        # The legacy REST get_info endpoint rejects ended sessions.
                        # A captured terminal state with zero allocation is already
                        # sufficient evidence that this session needs no termination.
                        latest = recorder.data["samples"][-1] if recorder.data["samples"] else None
                        previous = (
                            next((item for item in latest["sessions"] if item["id"] == label), None)
                            if latest
                            else None
                        )
                        if (
                            previous
                            and previous["status"] in TERMINAL_STATES
                            and _cpu(previous["occupied_slots"]) == 0
                        ):
                            continue
                        info = await recorder.bounded(
                            recorder.sessions[label].detail(
                                fields=(session_node_fields["status"],),
                            )
                        )
                        if not info:
                            raise LiveExperimentError(f"Cannot observe cleanup state for {label}")
                        if info["status"] not in TERMINAL_STATES:
                            await recorder.terminate(label, "cleanup")
                    except Exception as error:
                        cleanup_errors.append({"session": label, "type": type(error).__name__})
                if recorder.sessions and not cleanup_errors:
                    try:
                        sample = await recorder.wait(
                            api,
                            "cleanup_released",
                            lambda row: (
                                all(item["status"] in TERMINAL_STATES for item in row["sessions"])
                                and all(
                                    _cpu(item["occupied_slots"]) == 0 for item in row["sessions"]
                                )
                                and _cpu(row["agent"]["occupied_slots"]) == 0
                            ),
                            allow_error_states=not recorder.data["verification"]["passed"],
                        )
                        recorder.checkpoint(
                            "All allocations released",
                            sample,
                            "All experiment sessions ended and the agent reports "
                            "zero allocated CPU slots",
                        )
                        for label, session in recorder.sessions.items():
                            detail = await recorder.bounded(
                                session.detail(
                                    fields=(session_node_fields["status_history"],),
                                )
                            )
                            recorder.data["status_history"][label] = _json_object(
                                detail["status_history"]
                            )
                    except Exception as error:
                        cleanup_errors.append(
                            {"stage": "observe_release", "type": type(error).__name__}
                        )
                recorder.data["cleanup"] = {
                    "complete": not cleanup_errors,
                    "errors": cleanup_errors,
                }
                if cleanup_errors:
                    recorder.data["verification"]["passed"] = False
                recorder.save()
    if not recorder.data["verification"]["passed"]:
        raise LiveExperimentError(
            "The run or cleanup did not complete; inspect the saved recording"
        )
    return _redact(recorder.data, recorder.secrets)


def run_live_experiment(
    config: LiveConfig,
    output: Path,
    *,
    owned_session_ids: set[str] | None = None,
    owned_session_names: set[str] | None = None,
) -> dict[str, Any]:
    """Record the experiment and expose owned identifiers for outer cleanup.

    Caller-provided sets are updated in place before any further asynchronous
    operation. Names cover an interrupted creation whose server-generated ID is
    still unknown; they must be resolved through the API or database, since
    Backend.AI's Docker labels contain session IDs rather than session names.
    """
    config.validate()
    if version("backend.ai-client") != BACKEND_AI_VERSION:
        raise LiveExperimentError(
            f"This experiment requires backend.ai-client {BACKEND_AI_VERSION}"
        )
    return asyncio.run(
        _run(
            config,
            output,
            owned_session_ids=owned_session_ids,
            owned_session_names=owned_session_names,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091")
    parser.add_argument("--image", required=True)
    parser.add_argument("--resource-group", default="default")
    parser.add_argument("--domain", default="default")
    parser.add_argument("--project", default="default")
    parser.add_argument("--architecture", default="aarch64")
    parser.add_argument("--stage-timeout", type=float, default=300)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/live-recording.json"))
    args = parser.parse_args()
    output = args.output
    config = LiveConfig(**{key: value for key, value in vars(args).items() if key != "output"})
    result = run_live_experiment(config, output)
    print(f"Saved {len(result['samples'])} observed samples to {output}")


if __name__ == "__main__":
    main()
