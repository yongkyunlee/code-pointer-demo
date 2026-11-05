"""Test live-runner boundaries without making API calls or starting containers."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from ai.backend.client.exceptions import BackendAPIError

from backend_ai_intro import live


def config(**overrides):
    return live.LiveConfig(
        **{"endpoint": "http://127.0.0.1:8091", "image": "test-image", **overrides}
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com",
        "http://localhost.example.com",
        "http://test:credential@localhost",
        "http://localhost?key=value",
        "http://localhost#fragment",
        "ftp://localhost",
    ],
)
def test_live_endpoint_rejects_remote_hosts_and_embedded_credentials(endpoint):
    with pytest.raises(ValueError):
        config(endpoint=endpoint).validate()


@pytest.mark.parametrize("endpoint", ["http://localhost:8091", "http://127.0.0.1", "http://[::1]"])
def test_local_endpoints_are_accepted(endpoint):
    config(endpoint=endpoint).validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"stage_timeout": 0},
        {"request_timeout": 0},
        {"poll_interval": float("nan")},
        {"resource_group": 'default" or name != "'},
        {"image": " "},
    ],
)
def test_invalid_run_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides).validate()


def test_resource_slot_strings_preserve_decimal_values():
    assert live._slots('{"cpu": "0.5", "mem": "1073741824"}', "slots") == {
        "cpu": "0.5",
        "mem": "1073741824",
    }


@pytest.mark.parametrize("amount", ["-1", "NaN", "Infinity", "not a number", True])
def test_invalid_resource_amount_is_rejected(amount):
    with pytest.raises(live.LiveExperimentError):
        live._slots({"cpu": amount}, "slots")


def test_invalid_resource_json_is_rejected():
    with pytest.raises(json.JSONDecodeError):
        live._slots("{not JSON}", "slots")


def test_capacity_check_requires_an_actual_skipped_history_record():
    sample = {
        "sessions": [{"id": name, "status": "PENDING"} for name in ("alice", "bob")],
        "agent": {"occupied_slots": {"cpu": "2"}},
        "latest_scheduling_history": {"bob": [{"result": "FAILURE"}]},
    }
    assert not live._skipped_while_pending(sample)
    sample["latest_scheduling_history"]["bob"] = [
        {
            "result": "SKIPPED",
            "phase": "another-phase",
            "from_status": "PENDING",
            "to_status": "PENDING",
        }
    ]
    assert not live._skipped_while_pending(sample)
    sample["latest_scheduling_history"]["bob"][0]["phase"] = "schedule-sessions"
    assert live._skipped_while_pending(sample)
    sample["sessions"][1]["status"] = "SCHEDULED"
    assert not live._skipped_while_pending(sample)


def test_allocated_session_is_not_treated_as_running():
    sample = {"sessions": [{"id": "bob", "status": "SCHEDULED", "occupied_slots": {"cpu": "1"}}]}
    assert not live._running_with_cpu(sample, "bob", 1)
    sample["sessions"][0]["status"] = "RUNNING"
    assert live._running_with_cpu(sample, "bob", 1)
    sample["sessions"][0]["occupied_slots"]["cpu"] = "0"
    assert not live._running_with_cpu(sample, "bob", 1)


def test_stage_timeout_cancels_a_stalled_sample(tmp_path, monkeypatch):
    recorder = live._Recorder(config(stage_timeout=0.02), tmp_path / "recording.json")

    async def stalled_sample(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(recorder, "sample", stalled_sample)

    async def run():
        with pytest.raises(live.LiveExperimentError, match="Timed out waiting for test-stage"):
            await asyncio.wait_for(recorder.wait(None, "test-stage", lambda _: False), timeout=1)

    asyncio.run(run())


def test_saved_diagnostics_redact_credentials(tmp_path, monkeypatch):
    # Deliberately fake credentials exercise the recording boundary.
    access, secret = "test-access-not-a-real-key", "test-secret-not-a-real-key"
    monkeypatch.setenv("BACKEND_ACCESS_KEY", access)
    monkeypatch.setenv("BACKEND_SECRET_KEY", secret)
    recorder = live._Recorder(config(), tmp_path / "recording.json")
    recorder.data["diagnostics"] = {
        "message": f"A server echoed {access} and {secret}",
        "nested": [{"authorization": "another-sensitive-value"}],
    }
    recorder.save()
    content = recorder.output.read_text()
    assert access not in content
    assert secret not in content
    assert "another-sensitive-value" not in content
    assert "[REDACTED]" in content


class FakeSession:
    def __init__(self):
        self.id = uuid4()
        self.destroy_calls = 0
        self.status = "RUNNING"
        self.detail_calls = []

    async def get_info(self):
        raise AssertionError("The REST endpoint rejects terminated sessions")

    async def detail(self, *, fields):
        self.detail_calls.append(tuple(field.field_ref for field in fields))
        return {"status": self.status, "status_history": {}}

    async def destroy(self):
        self.destroy_calls += 1
        self.status = "TERMINATED"

    async def get_status_history(self):
        raise AssertionError("Use GraphQL for terminated session history")


def test_interrupted_create_is_recovered_by_exact_run_name_and_id(tmp_path):
    owned_ids, owned_names = set(), set()
    recorder = live._Recorder(
        config(),
        tmp_path / "recording.json",
        owned_session_ids=owned_ids,
        owned_session_names=owned_names,
    )
    owned = FakeSession()
    looked_up = []

    class FakeComputeSession:
        async def get_or_create(self, *_args, **kwargs):
            self.name = kwargs["name"]
            raise TimeoutError("Simulate a lost creation response")

        async def paginated_list(self, **kwargs):
            looked_up.append(kwargs["filter"])
            return SimpleNamespace(
                total_count=1,
                items=[
                    {
                        "id": str(owned.id),
                        "name": self.name,
                        "created_at": (recorder.started_at + timedelta(seconds=1)).isoformat(),
                    }
                ],
            )

        def from_session_id(self, session_id):
            assert session_id == owned.id
            return owned

    api = SimpleNamespace(ComputeSession=FakeComputeSession())

    async def run():
        with pytest.raises(TimeoutError):
            await recorder.create(api, "current")
        name = recorder.creation_attempts["current"]
        assert name == f"cp-demo-{recorder.run_id}-current"
        assert recorder.sessions == {}
        assert owned_names == {name}
        assert owned_ids == set()
        await recorder.recover_creation(api, "current", name)
        assert owned_ids == {str(owned.id)}
        await recorder.terminate("current", "cleanup")
        assert looked_up == [f'name == "{name}"']
        assert owned.destroy_calls == 1

    asyncio.run(run())


def test_accepted_creation_preserves_owned_id_if_recording_the_response_fails(
    tmp_path, monkeypatch
):
    owned_ids, owned_names = set(), set()
    recorder = live._Recorder(
        config(),
        tmp_path / "recording.json",
        owned_session_ids=owned_ids,
        owned_session_names=owned_names,
    )
    session_id = uuid4()

    async def accepted(*_args, **_kwargs):
        return SimpleNamespace(created=True, id=session_id)

    def action(_stage, operation, _label, **_details):
        if operation == "create_accepted":
            raise OSError("Injected recording failure after the accepted response")

    monkeypatch.setattr(recorder, "action", action)
    api = SimpleNamespace(ComputeSession=SimpleNamespace(get_or_create=accepted))
    with pytest.raises(OSError, match="recording failure"):
        asyncio.run(recorder.create(api, "current"))
    assert owned_ids == {str(session_id)}
    assert owned_names == {f"cp-demo-{recorder.run_id}-current"}


@pytest.mark.parametrize("mismatch", ["name", "old_timestamp", "ambiguous"])
def test_recovery_refuses_unrelated_or_ambiguous_sessions(tmp_path, mismatch):
    recorder = live._Recorder(config(), tmp_path / "recording.json")
    name = f"cp-demo-{recorder.run_id}-current"
    reconstructed = []

    async def listing(**_kwargs):
        return SimpleNamespace(
            total_count=2 if mismatch == "ambiguous" else 1,
            items=[
                {
                    "id": str(uuid4()),
                    "name": "unrelated" if mismatch == "name" else name,
                    "created_at": (
                        recorder.started_at
                        + timedelta(seconds=-1 if mismatch == "old_timestamp" else 1)
                    ).isoformat(),
                }
            ],
        )

    api = SimpleNamespace(
        ComputeSession=SimpleNamespace(
            paginated_list=listing,
            from_session_id=lambda session_id: reconstructed.append(session_id),
        )
    )
    with pytest.raises(live.LiveExperimentError):
        asyncio.run(recorder.recover_creation(api, "current", name))
    assert recorder.sessions == {}
    assert reconstructed == []


def test_definitive_creation_rejection_does_not_enable_name_based_cleanup(tmp_path):
    recorder = live._Recorder(config(), tmp_path / "recording.json")

    async def rejected(*_args, **_kwargs):
        raise BackendAPIError(409, "Conflict", {"title": "A session with that name exists"})

    api = SimpleNamespace(ComputeSession=SimpleNamespace(get_or_create=rejected))
    with pytest.raises(BackendAPIError):
        asyncio.run(recorder.create(api, "current"))
    assert recorder.creation_attempts == {}
    assert recorder.sessions == {}


def test_failure_cleanup_only_terminates_sessions_owned_by_this_run(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_ACCESS_KEY", "fake-access")
    monkeypatch.setenv("BACKEND_SECRET_KEY", "fake-secret")
    monkeypatch.delenv("BACKEND_VFOLDER_MOUNTS", raising=False)
    monkeypatch.delenv("SORNA_VFOLDER_MOUNTS", raising=False)
    owned, unrelated = FakeSession(), FakeSession()

    @asynccontextmanager
    async def connected(*_args):
        yield SimpleNamespace(unrelated_session=unrelated)

    async def preflight(_recorder, _api):
        return None

    async def create_then_fail(recorder, _api, label):
        recorder.sessions[label] = owned
        raise live.LiveExperimentError("Injected failure after creation")

    async def cleanup_sample(recorder, _api, stage, predicate):
        assert stage == "cleanup_released"
        sample = {
            "sequence": 0,
            "sessions": [{"id": "current", "status": owned.status, "occupied_slots": {"cpu": "0"}}],
            "agent": {"occupied_slots": {"cpu": "0"}},
        }
        assert predicate(sample)
        recorder.data["samples"].append(sample)
        return sample

    monkeypatch.setattr(live, "_connected_session", connected)
    monkeypatch.setattr(live._Recorder, "preflight", preflight)
    monkeypatch.setattr(live._Recorder, "create", create_then_fail)
    monkeypatch.setattr(live._Recorder, "wait", cleanup_sample)
    output = tmp_path / "recording.json"
    with pytest.raises(live.LiveExperimentError, match="Injected failure"):
        asyncio.run(live._run(config(), output))
    assert owned.destroy_calls == 1
    assert unrelated.destroy_calls == 0
    saved = json.loads(output.read_text())
    assert saved["verification"]["passed"] is False
    assert saved["failure"]["type"] == "LiveExperimentError"
    assert saved["cleanup"] == {"complete": True, "errors": []}


def test_cleanup_uses_known_terminal_observations_without_rest_lookups(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_ACCESS_KEY", "fake-access")
    monkeypatch.setenv("BACKEND_SECRET_KEY", "fake-secret")
    monkeypatch.delenv("BACKEND_VFOLDER_MOUNTS", raising=False)
    monkeypatch.delenv("SORNA_VFOLDER_MOUNTS", raising=False)
    sessions = {label: FakeSession() for label in ("current", "alice", "bob")}
    sessions["current"].status = sessions["alice"].status = "TERMINATED"

    def sample():
        return {
            "sequence": 0,
            "sessions": [
                {
                    "id": label,
                    "status": session.status,
                    "occupied_slots": {"cpu": "0" if session.status == "TERMINATED" else "1"},
                }
                for label, session in sessions.items()
            ],
            "agent": {"occupied_slots": {"cpu": "0"}},
        }

    @asynccontextmanager
    async def connected(*_args):
        yield None

    async def preflight(_recorder, _api):
        return None

    async def populate_then_fail(recorder, _api, _label):
        recorder.sessions.update(sessions)
        recorder.data["samples"].append(sample())
        raise live.LiveExperimentError("Injected failure after terminal observations")

    async def cleanup_sample(recorder, _api, stage, predicate):
        assert stage == "cleanup_released"
        observation = sample()
        assert predicate(observation)
        recorder.data["samples"].append(observation)
        return observation

    monkeypatch.setattr(live, "_connected_session", connected)
    monkeypatch.setattr(live._Recorder, "preflight", preflight)
    monkeypatch.setattr(live._Recorder, "create", populate_then_fail)
    monkeypatch.setattr(live._Recorder, "wait", cleanup_sample)
    output = tmp_path / "recording.json"
    with pytest.raises(live.LiveExperimentError, match="Injected failure"):
        asyncio.run(live._run(config(), output))

    for label in ("current", "alice"):
        assert sessions[label].destroy_calls == 0
        assert sessions[label].detail_calls == [("status_history",)]
    assert sessions["bob"].destroy_calls == 1
    assert sessions["bob"].detail_calls == [("status",), ("status_history",)]
    assert json.loads(output.read_text())["cleanup"] == {"complete": True, "errors": []}
