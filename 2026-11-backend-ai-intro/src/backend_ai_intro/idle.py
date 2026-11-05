"""Observe utilization-based reclamation and saved-file recovery in a live install.

All workload and policy mutations use Backend.AI's public APIs. CPU activity comes
from the real agent's Prometheus endpoint. Missing metrics are never treated as
zero usage. The operator does not end the idle session during the measured stage.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .idle_setup import (
    BASELINE_SECONDS,
    CPU_THRESHOLD_MILLICORES,
    IDLE_SECONDS,
    METRIC_PRESET_ID,
    PROMETHEUS_IMAGE,
    PROMETHEUS_PORT,
)
from .live import (
    BACKEND_AI_VERSION,
    TERMINAL_STATES,
    LiveConfig,
    LiveExperimentError,
    _connected_session,
    _cpu,
    _json_object,
    _Recorder,
    _redact,
    _running_with_cpu,
    _session,
    _slots,
)

REQUESTS = {"idle": 3, "active": 1, "waiting": 1, "reopened": 1}
LABELS = {
    "idle": "Idle notebook",
    "active": "Active CPU control",
    "waiting": "Waiting useful work",
    "reopened": "Reopened notebook",
}
METRIC_QUERY = (
    "sum by (session_id) (rate(backendai_container_utilization"
    '{container_metric_name="cpu_used",value_type="current"}[5m]))'
)


def _metric_values(payload: dict[str, Any]) -> dict[str, str]:
    if payload.get("status") != "success" or payload.get("data", {}).get("resultType") != "vector":
        raise LiveExperimentError("Prometheus did not return an instant vector")
    values: dict[str, str] = {}
    for row in payload["data"]["result"]:
        session_id = row.get("metric", {}).get("session_id")
        if not session_id:
            continue
        try:
            value = Decimal(row["value"][1])
        except (InvalidOperation, KeyError, IndexError, TypeError) as error:
            raise LiveExperimentError("Invalid CPU utilization sample") from error
        if not value.is_finite() or value < 0:
            raise LiveExperimentError("CPU utilization must be finite and nonnegative")
        values[session_id] = str(value)
    return values


def _activity_contrast(sample: dict[str, Any]) -> bool:
    values = sample.get("cpu_utilization_millicores", {})
    if values.get("idle") is None or values.get("active") is None:
        return False
    return (
        Decimal(values["idle"]) < CPU_THRESHOLD_MILLICORES
        and Decimal(values["active"]) >= CPU_THRESHOLD_MILLICORES
        and _running_with_cpu(sample, "idle", 3)
        and _running_with_cpu(sample, "active", 1)
    )


def _reclaimed(sample: dict[str, Any]) -> bool:
    idle = _session(sample, "idle")
    active_value = sample.get("cpu_utilization_millicores", {}).get("active")
    return (
        idle["status"] == "TERMINATED"
        and idle.get("status_info") == "idle-timeout"
        and _cpu(idle["occupied_slots"]) == 0
        and _running_with_cpu(sample, "active", 1)
        and active_value is not None
        and Decimal(active_value) >= CPU_THRESHOLD_MILLICORES
        and _running_with_cpu(sample, "waiting", 1)
    )


def checker_payload(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "checker_type": "utilization",
        "target_session_types": ["interactive"],
        "initial_grace_period_seconds": 0,
        "checker_spec": {
            "utilization": {
                "max_underutilized_duration_seconds": IDLE_SECONDS,
                "threshold": {
                    "preset_id": METRIC_PRESET_ID,
                    "threshold": str(CPU_THRESHOLD_MILLICORES),
                    "filter_labels": [
                        {"key": "container_metric_name", "value": "cpu_used"},
                        {"key": "value_type", "value": "current"},
                    ],
                    "group_labels": ["session_id"],
                },
            },
        },
    }


def _included_all(payload: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
    """A successful active control must have the same policy as the idle session."""
    expected = {(str(row["checker_id"]), str(row["session_id"])) for row in targets}
    items = payload.get("items", [])
    actual = {(str(row["checker_id"]), str(row["session_id"])) for row in items}
    return not payload.get("failed") and actual == expected and len(items) == len(expected)


class _IdleRecorder(_Recorder):
    def __init__(self, *args: Any, http: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.http = http
        self.data["experiment"].update(
            mode="live_idle_cpu",
            name="Idle allocation reclaimed while active work survives",
            scope="Actual CPU sessions, measured utilization, idle policy and persistent vfolder",
            limits="CPU demonstration; does not measure GPU performance or preserve process memory",
        )
        self.data["configuration"].update(
            session_requests=REQUESTS,
            idle_seconds=IDLE_SECONDS,
            threshold_millicores=CPU_THRESHOLD_MILLICORES,
            baseline_seconds=BASELINE_SECONDS,
            metric_query=METRIC_QUERY,
            metric_preset_id=METRIC_PRESET_ID,
            metric_interpretation=(
                "Backend.AI 26.9.7 exports cumulative CPU milliseconds for cpu_used/current. "
                "The seeded five-minute rate preset converts milliseconds per second to "
                "millicores. cpu_util/current is not used because this release exports "
                "the raw cumulative measurement there before applying its internal rate hook."
            ),
            prometheus_image=PROMETHEUS_IMAGE,
            identities="One generated administrator account; names describe workload roles",
        )

    async def create(self, api: Any, label: str, mounts: list[str] | None = None) -> None:
        from ai.backend.client.exceptions import BackendAPIError

        name = f"cp-idle-{self.run_id}-{label}"
        self.creation_attempts[label] = name
        self.owned_session_names.add(name)
        self.action("submit", "create_requested", label, name=name, requested_cpu=REQUESTS[label])
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
                    mounts=mounts or [],
                )
            )
        except BackendAPIError as error:
            if 400 <= error.status < 500:
                self.creation_attempts.pop(label)
                self.owned_session_names.discard(name)
            raise
        if not session.created or session.id is None:
            self.creation_attempts.pop(label)
            self.owned_session_names.discard(name)
            raise LiveExperimentError("Creation did not return a new session")
        self.owned_session_ids.add(str(session.id))
        self.sessions[label] = session
        self.creation_attempts.pop(label)
        self.action("submit", "create_accepted", label, session_id=str(session.id))

    async def sample(self, api: Any, stage: str) -> dict[str, Any]:
        from ai.backend.client.output.fields import agent_fields, session_node_fields

        observed = []
        history = {}
        for label, session in self.sessions.items():
            item = await self.bounded(
                session.detail(
                    fields=tuple(
                        session_node_fields[key]
                        for key in (
                            "status",
                            "status_info",
                            "status_data",
                            "requested_slots",
                            "occupied_slots",
                            "created_at",
                            "terminated_at",
                        )
                    )
                )
            )
            if not item:
                raise LiveExperimentError(f"Session {label} disappeared")
            observed.append(
                {
                    "id": label,
                    "label": LABELS[label],
                    "session_id": str(session.id),
                    "requested_cpu": REQUESTS[label],
                    **{
                        key: item.get(key)
                        for key in (
                            "status",
                            "status_info",
                            "created_at",
                            "terminated_at",
                        )
                    },
                    "status_data": _json_object(item.get("status_data")),
                    "requested_slots": _slots(item["requested_slots"], "requested_slots"),
                    "occupied_slots": _slots(item["occupied_slots"], "occupied_slots"),
                }
            )
            history[label] = await self.get_history(api, label)
        agent = await self.bounded(
            api.Agent.detail(
                self.agent_id,
                fields=tuple(
                    agent_fields[key]
                    for key in ("id", "status", "available_slots", "occupied_slots")
                ),
            )
        )
        async with self.http.get(
            f"http://127.0.0.1:{PROMETHEUS_PORT}/api/v1/query",
            params={"query": METRIC_QUERY},
        ) as response:
            response.raise_for_status()
            metrics = _metric_values(await response.json())
        sample = {
            "sequence": len(self.data["samples"]),
            **self.clock(),
            "stage": stage,
            "sessions": observed,
            "agent": {
                "id": agent["id"],
                "status": agent["status"],
                "available_slots": _slots(agent["available_slots"], "available_slots"),
                "occupied_slots": _slots(agent["occupied_slots"], "occupied_slots"),
            },
            "cpu_utilization_millicores": {
                label: metrics.get(str(session.id)) for label, session in self.sessions.items()
            },
            "latest_scheduling_history": history,
        }
        self.data["samples"].append(sample)
        self.save()
        return sample

    async def execute(self, label: str, code: str, expected: str) -> dict[str, Any]:
        self.action("workload", "execute_code", label, expected_marker=expected)
        responses = []
        async with asyncio.timeout(self.config.stage_timeout):
            response = await self.bounded(self.sessions[label].execute(code=code, mode="query"))
            while True:
                responses.append(response)
                if response.get("status") == "finished":
                    break
                if response.get("status") not in {"continued", "waiting-input"}:
                    # The API can return partial execution while code is still running.
                    if not response.get("runId"):
                        raise LiveExperimentError("Code execution returned no continuation ID")
                if response.get("status") == "waiting-input":
                    raise LiveExperimentError("The workload unexpectedly requested input")
                response = await self.bounded(
                    self.sessions[label].execute(
                        run_id=response["runId"],
                        code="",
                        mode="continue",
                    )
                )
        # Check actual console text, not an echoed request or JSON metadata.
        console = "".join(
            str(entry[1])
            for part in responses
            for entry in part.get("console", [])
            if isinstance(entry, (list, tuple)) and len(entry) >= 2
        )
        if expected not in console or response.get("exitCode", 0) not in {0, None}:
            self.data.setdefault("execution_failures", []).append(responses)
            self.save()
            raise LiveExperimentError(f"Workload {label} did not produce its expected result")
        result = {"session": label, **self.clock(), "console": console, "responses": responses}
        self.data.setdefault("workload_results", []).append(result)
        self.save()
        return result


async def _cleanup(
    recorder: _IdleRecorder, api: Any, v2: Any, checker_id: Any, folder_name: str | None
) -> None:
    from ai.backend.client.output.fields import session_node_fields

    errors = []
    if checker_id is not None:
        try:
            await recorder.bounded(v2.idle_checker.admin_purge(checker_id))
            recorder.action("cleanup", "purge_idle_checker", "policy")
        except Exception as error:
            errors.append({"stage": "purge_policy", "type": type(error).__name__})
    for label, name in list(recorder.creation_attempts.items()):
        try:
            await recorder.recover_creation(api, label, name)
        except Exception as error:
            errors.append(
                {"stage": "recover_creation", "session": label, "type": type(error).__name__}
            )
    for label in reversed(recorder.sessions):
        try:
            info = await recorder.bounded(
                recorder.sessions[label].detail(
                    fields=(session_node_fields["status"],),
                )
            )
            if not info:
                raise LiveExperimentError("Missing cleanup status")
            if info["status"] not in TERMINAL_STATES:
                await recorder.terminate(label, "cleanup")
        except Exception as error:
            errors.append({"stage": "terminate", "session": label, "type": type(error).__name__})
    released = False
    if recorder.sessions:
        try:
            sample = await recorder.wait(
                api,
                "cleanup_released",
                lambda row: (
                    all(
                        item["status"] in TERMINAL_STATES and _cpu(item["occupied_slots"]) == 0
                        for item in row["sessions"]
                    )
                    and _cpu(row["agent"]["occupied_slots"]) == 0
                ),
                allow_error_states=True,
            )
            released = True
            recorder.checkpoint(
                "All allocations released",
                sample,
                "All owned sessions ended and allocated CPU returned to zero",
            )
        except Exception as error:
            errors.append({"stage": "observe_release", "type": type(error).__name__})
    if folder_name and (released or not recorder.sessions):
        try:
            await recorder.bounded(api.VFolder(folder_name).delete())
            recorder.action("cleanup", "delete_vfolder", "storage", name=folder_name)
        except Exception as error:
            errors.append({"stage": "delete_vfolder", "type": type(error).__name__})
    recorder.data["cleanup"] = {"complete": not errors, "errors": errors}
    if errors:
        recorder.data["verification"]["passed"] = False
    recorder.save()


async def _run(
    config: LiveConfig,
    output: Path,
    *,
    owned_session_ids: set[str] | None,
    owned_session_names: set[str] | None,
) -> dict[str, Any]:
    import aiohttp
    from ai.backend.client.config import APIConfig
    from ai.backend.client.v2 import ClientConfig, HMACAuth
    from ai.backend.client.v2.v2_registry import V2ClientRegistry
    from ai.backend.common.dto.manager.v2.idle_checker.request import CreateIdleCheckerInput
    from ai.backend.common.dto.manager.v2.session.request import IncludeSessionIdleChecksInput
    from yarl import URL

    access_key, secret_key = (
        os.environ.get(key, "")
        for key in (
            "BACKEND_ACCESS_KEY",
            "BACKEND_SECRET_KEY",
        )
    )
    if not access_key or not secret_key:
        raise LiveExperimentError("Demo API credentials are required")
    if os.environ.get("BACKEND_VFOLDER_MOUNTS") or os.environ.get("SORNA_VFOLDER_MOUNTS"):
        raise LiveExperimentError("Unset automatic vfolder mounts for this isolated experiment")
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
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.request_timeout),
    ) as http:
        recorder = _IdleRecorder(
            config,
            output,
            http=http,
            owned_session_ids=owned_session_ids,
            owned_session_names=owned_session_names,
        )
        async with _connected_session(config, api_config, http) as api:
            v2 = await V2ClientRegistry.create(
                ClientConfig(endpoint=URL(config.endpoint), read_timeout=config.request_timeout),
                HMACAuth(access_key, secret_key),
            )
            checker_id = None
            folder_name = None
            try:
                await recorder.preflight(api)
                name = f"cp-idle-{recorder.run_id}"
                folder = await recorder.bounded(
                    api.VFolder.create(
                        name,
                        host="local:volume1",
                    )
                )
                folder_name = name
                recorder.data["storage"] = {"name": name, "creation_response": folder}
                recorder.action("setup", "create_vfolder", "storage", name=name)
                created = await recorder.bounded(
                    v2.idle_checker.admin_create(
                        CreateIdleCheckerInput.model_validate(checker_payload(name)),
                    )
                )
                checker_id = created.idle_checker.id
                recorder.data["idle_policy"] = created.model_dump(mode="json")
                recorder.action("setup", "create_unassigned_idle_checker", "policy")
                await recorder.create(api, "idle", [name])
                await recorder.create(api, "active")
                await recorder.wait(
                    api,
                    "sessions_running",
                    lambda row: (
                        _running_with_cpu(row, "idle", 3) and _running_with_cpu(row, "active", 1)
                    ),
                )
                content = f"Saved work from idle experiment {recorder.run_id}\n"
                digest = hashlib.sha256(content.encode()).hexdigest()
                saved_path = f"/home/work/{name}/saved.txt"
                await recorder.execute(
                    "idle",
                    (
                        "from pathlib import Path\nimport hashlib\n"
                        f"p=Path({saved_path!r})\np.write_text({content!r})\n"
                        "print('SAVED_SHA256='+hashlib.sha256(p.read_bytes()).hexdigest())\n"
                    ),
                    f"SAVED_SHA256={digest}",
                )
                recorder.data["storage"].update(saved_file="saved.txt", sha256=digest)
                busy_code = (
                    "import time, hashlib\n"
                    "deadline=time.monotonic()+1200\nvalue=b'backend-ai-active-control'\n"
                    "while time.monotonic()<deadline:\n"
                    "    value=hashlib.sha256(value).digest()\n"
                )
                await recorder.execute(
                    "active",
                    (
                        "import subprocess, sys\n"
                        f"p=subprocess.Popen([sys.executable, '-c', {busy_code!r}], "
                        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                        "stderr=subprocess.DEVNULL, start_new_session=True)\n"
                        "print('ACTIVE_PROCESS_STARTED='+str(p.pid))\n"
                    ),
                    "ACTIVE_PROCESS_STARTED=",
                )
                await recorder.create(api, "waiting")
                await recorder.wait(
                    api,
                    "contrast_visible",
                    lambda row: (
                        _activity_contrast(row) and _session(row, "waiting")["status"] == "PENDING"
                    ),
                )
                baseline_start = time.monotonic()
                while time.monotonic() - baseline_start < BASELINE_SECONDS:
                    sample = await recorder.sample(api, "policy_off")
                    if not (
                        _running_with_cpu(sample, "idle", 3)
                        and _running_with_cpu(sample, "active", 1)
                        and _session(sample, "waiting")["status"] == "PENDING"
                    ):
                        raise LiveExperimentError("Policy-off control changed unexpectedly")
                    await asyncio.sleep(max(config.poll_interval, 2))
                if not _activity_contrast(sample):
                    raise LiveExperimentError("Baseline lacks measured active/idle contrast")
                recorder.checkpoint(
                    "Policy off: work remains blocked",
                    sample,
                    "The idle notebook held three CPUs beyond the planned idle duration; "
                    "active work consumed CPU and the one-CPU request remained pending",
                )
                targets = [
                    {"checker_id": checker_id, "session_id": recorder.sessions[label].id}
                    for label in ("idle", "active")
                ]
                included = await recorder.bounded(
                    v2.session.include_idle_checks(
                        IncludeSessionIdleChecksInput.model_validate({"targets": targets}),
                    )
                )
                inclusion = included.model_dump(mode="json")
                if not _included_all(inclusion, targets):
                    raise LiveExperimentError("The idle policy was not attached to every control")
                recorder.data["idle_policy_inclusion"] = inclusion
                recorder.action(
                    "policy_on", "include_idle_checks", "policy", sessions=["idle", "active"]
                )
                enabled_at = time.monotonic()
                sample = await recorder.wait(api, "automatic_reclamation", _reclaimed)
                recorder.data["measurements"] = {
                    "policy_on_to_waiting_running_seconds": round(time.monotonic() - enabled_at, 3),
                    "policy_off_observation_seconds": BASELINE_SECONDS,
                    "interpretation": "Observed timings include periodic checks and polling",
                }
                recorder.checkpoint(
                    "Idle capacity reclaimed automatically",
                    sample,
                    "The notebook ended with idle-timeout; waiting work started while "
                    "the active CPU control remained running above the same threshold",
                )
                await recorder.execute(
                    "waiting",
                    (
                        "import hashlib\n"
                        "print('USEFUL_RESULT='+hashlib.sha256(b'queued-work-completed').hexdigest())\n"
                    ),
                    "USEFUL_RESULT=" + hashlib.sha256(b"queued-work-completed").hexdigest(),
                )
                await recorder.create(api, "reopened", [name])
                await recorder.wait(
                    api, "reopened_running", lambda row: _running_with_cpu(row, "reopened", 1)
                )
                await recorder.execute(
                    "reopened",
                    (
                        "from pathlib import Path\nimport hashlib\n"
                        f"print('RESTORED_SHA256='+hashlib.sha256(Path({saved_path!r}).read_bytes())"
                        ".hexdigest())\n"
                    ),
                    f"RESTORED_SHA256={digest}",
                )
                sample = await recorder.sample(api, "saved_work_restored")
                recorder.checkpoint(
                    "Saved file restored in a fresh session",
                    sample,
                    "A newly created session read the same saved-file SHA-256 from the vfolder",
                )
                recorder.data["verification"]["passed"] = True
            except BaseException as error:
                recorder.data["failure"] = {"type": type(error).__name__}
                recorder.save()
                raise
            finally:
                await _cleanup(recorder, api, v2, checker_id, folder_name)
                await v2.close()
    if not recorder.data["verification"]["passed"]:
        raise LiveExperimentError("Idle run or cleanup failed; inspect the partial recording")
    return _redact(recorder.data, recorder.secrets)


def run_idle_experiment(
    config: LiveConfig,
    output: Path,
    *,
    owned_session_ids: set[str] | None = None,
    owned_session_names: set[str] | None = None,
) -> dict[str, Any]:
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
