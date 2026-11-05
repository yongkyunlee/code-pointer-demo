"""Check that missing metrics and ordinary termination cannot prove idle recovery."""

from copy import deepcopy

import pytest
from ai.backend.common.dto.manager.v2.idle_checker.request import CreateIdleCheckerInput

from backend_ai_intro.idle import (
    _activity_contrast,
    _included_all,
    _metric_values,
    _reclaimed,
    checker_payload,
)
from backend_ai_intro.live import LiveExperimentError


def sample():
    return {
        "sessions": [
            {
                "id": "idle",
                "status": "TERMINATED",
                "status_info": "idle-timeout",
                "occupied_slots": {"cpu": "0"},
            },
            {"id": "active", "status": "RUNNING", "occupied_slots": {"cpu": "1"}},
            {"id": "waiting", "status": "RUNNING", "occupied_slots": {"cpu": "1"}},
        ],
        "cpu_utilization_millicores": {"idle": "2.5", "active": "970"},
    }


def test_recovery_requires_idle_reason_release_and_active_measured_work():
    observed = sample()
    assert _reclaimed(observed)
    for field, value in (
        ("status_info", "user-requested"),
        ("occupied_slots", {"cpu": "3"}),
        ("status", "TERMINATING"),
    ):
        changed = deepcopy(observed)
        changed["sessions"][0][field] = value
        assert not _reclaimed(changed)
    for metric in (None, "0", "99"):
        changed = deepcopy(observed)
        changed["cpu_utilization_millicores"]["active"] = metric
        assert not _reclaimed(changed)


def test_missing_idle_metrics_are_not_zero_usage():
    observed = sample()
    observed["sessions"][0].update(status="RUNNING", occupied_slots={"cpu": "3"})
    assert _activity_contrast(observed)
    observed["cpu_utilization_millicores"].pop("idle")
    assert not _activity_contrast(observed)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "broken"])
def test_invalid_metric_values_fail_closed(value):
    payload = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"session_id": "owned-session"}, "value": [1, value]},
            ],
        },
    }
    with pytest.raises(LiveExperimentError):
        _metric_values(payload)


def test_empty_prometheus_result_stays_missing():
    assert (
        _metric_values(
            {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [],
                },
            }
        )
        == {}
    )


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"checker_id": "policy", "session_id": "idle"}],
        [
            {"checker_id": "policy", "session_id": "idle"},
            {"checker_id": "policy", "session_id": "idle"},
        ],
    ],
)
def test_partial_or_duplicate_inclusion_cannot_establish_the_active_control(items):
    targets = [{"checker_id": "policy", "session_id": session} for session in ("idle", "active")]
    assert not _included_all({"items": items, "failed": []}, targets)
    assert _included_all({"items": list(reversed(targets)), "failed": []}, targets)


def test_policy_is_utilization_only_and_sdk_accepts_payload():
    policy = CreateIdleCheckerInput.model_validate(checker_payload("test-only"))
    assert policy.checker_spec.network is None
    assert policy.checker_spec.session_lifetime is None
    filters = {
        entry.key: entry.value for entry in policy.checker_spec.utilization.threshold.filter_labels
    }
    assert filters == {"container_metric_name": "cpu_used", "value_type": "current"}
    assert str(policy.checker_spec.utilization.threshold.preset_id) == (
        "d5d83b5e-3463-4b5f-beda-1afbe4761e3c"
    )


def test_setup_changes_only_generated_config_and_pins_prometheus(tmp_path):
    import json

    import tomlkit

    from backend_ai_intro.idle_setup import configure_runtime

    manager = tmp_path / "manager.toml"
    manager.write_text("[manager]\nnum-proc = 1\n")
    compose_path = configure_runtime(tmp_path)
    config = tomlkit.parse(manager.read_text())
    assert config["manager"]["num-proc"] == 1
    assert config["metric"]["addr"]["port"] == 29090
    service = json.loads(compose_path.read_text())["services"]["prometheus"]
    assert "@sha256:" in service["image"]
    assert "2s" in (tmp_path / "idle-prometheus.yml").read_text()


def test_current_active_worker_must_survive_even_when_cpu_rate_is_high(tmp_path):
    import asyncio
    import contextlib
    import io
    import subprocess
    import sys

    from backend_ai_intro.idle import _IdleRecorder
    from backend_ai_intro.live import LiveConfig

    async def check(process):
        class ExecutionSession:
            async def execute(self, *, code, mode):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(code, {"p": process})
                return {
                    "status": "finished",
                    "exitCode": 0,
                    "console": [["stdout", output.getvalue()]],
                }

        recorder = _IdleRecorder(
            LiveConfig(endpoint="http://127.0.0.1:8080", image="test"),
            tmp_path / "recording.json",
            http=None,
        )
        recorder.sessions["active"] = ExecutionSession()
        await recorder.verify_active_control(process.pid)

    # A container can remain RUNNING and its five-minute rate stay high after
    # this worker has exited. The fresh in-container check must still fail.
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait(timeout=5)
    assert _reclaimed(sample())
    with pytest.raises(LiveExperimentError, match="expected result"):
        asyncio.run(check(exited))
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        asyncio.run(check(alive))
    finally:
        alive.terminate()
        alive.wait(timeout=5)


def test_folder_cleanup_waits_for_physical_deletion_before_purging():
    import asyncio
    from types import SimpleNamespace
    from uuid import UUID

    from backend_ai_intro.idle import _delete_saved_folder

    folder_id = UUID(int=123)
    events = []
    statuses = iter(["delete-ongoing", "delete-complete"])

    class Folder:
        def __init__(self, name, *, id):
            assert name == "owned" and id == folder_id

        async def delete(self):
            events.append("trash")

        async def delete_trash(self):
            events.append("request_storage_delete")

        @classmethod
        async def paginated_list(cls, *, fields, filter):
            assert filter == f'id == "{folder_id}"'
            status = next(statuses)
            events.append(status)
            return SimpleNamespace(items=[{"id": str(folder_id), "status": status}])

        async def purge(self):
            assert events[-1] == "delete-complete"
            events.append("purge")

    async def bounded(operation):
        return await operation

    recorder = SimpleNamespace(
        data={"storage": {"creation_response": {"id": folder_id.hex}}},
        config=SimpleNamespace(stage_timeout=5, poll_interval=0.001),
        bounded=bounded,
        action=lambda *args, **kwargs: None,
    )
    asyncio.run(_delete_saved_folder(recorder, SimpleNamespace(VFolder=Folder), "owned"))
    assert events == [
        "trash",
        "request_storage_delete",
        "delete-ongoing",
        "delete-complete",
        "purge",
    ]
    assert recorder.data["storage"]["deletion_status"] == "delete-complete"
