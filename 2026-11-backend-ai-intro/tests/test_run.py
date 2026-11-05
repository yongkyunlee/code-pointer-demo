"""Check orchestration cleanup using fake child processes, never Docker."""

import asyncio
import importlib
import json
import os
import signal
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("run")


@pytest.mark.parametrize("image_id", ["sha256:" + "a" * 64, "sha256:" + "b" * 64])
def test_image_identity_accepts_config_or_manifest_id_when_repository_digest_matches(
    runner, image_id
):
    # Classic stores identify the image by its config; containerd may use its manifest.
    pin = "registry.example.test/cpu-image@sha256:" + "a" * 64
    runner.verify_image_identity({"Id": image_id, "RepoDigests": [pin]}, pin)


def test_image_identity_rejects_a_wrong_digest_even_if_image_id_matches(runner):
    digest = "sha256:" + "a" * 64
    pin = "registry.example.test/cpu-image@" + digest
    inspection = {
        "Id": digest,
        "RepoDigests": ["registry.example.test/cpu-image@sha256:" + "c" * 64],
    }
    with pytest.raises(RuntimeError, match="pinned manifest digest"):
        runner.verify_image_identity(inspection, pin)


@pytest.mark.parametrize("inspection", [{"Id": "sha256:" + "a" * 64}, {"RepoDigests": []}])
def test_image_identity_requires_a_repository_digest(runner, inspection):
    pin = "registry.example.test/cpu-image@sha256:" + "a" * 64
    with pytest.raises(RuntimeError, match="pinned manifest digest"):
        runner.verify_image_identity(inspection, pin)


def test_environment_restores_existing_values_and_removes_new_values(runner, monkeypatch):
    monkeypatch.setenv("BACKEND_DEMO_TEST_EXISTING", "before")
    monkeypatch.delenv("BACKEND_DEMO_TEST_NEW", raising=False)
    with pytest.raises(RuntimeError, match="injected"):
        with runner.environment(
            {"BACKEND_DEMO_TEST_EXISTING": "during", "BACKEND_DEMO_TEST_NEW": "temporary"}
        ):
            assert os.environ["BACKEND_DEMO_TEST_EXISTING"] == "during"
            assert os.environ["BACKEND_DEMO_TEST_NEW"] == "temporary"
            raise RuntimeError("injected")
    assert os.environ["BACKEND_DEMO_TEST_EXISTING"] == "before"
    assert "BACKEND_DEMO_TEST_NEW" not in os.environ


def test_runtime_lock_refuses_a_second_owner_and_keeps_the_inode(runner, tmp_path):
    runtime = tmp_path / "runtime"
    with runner.runtime_lock(runtime):
        inode = (runtime / "run.lock").stat().st_ino
        with pytest.raises(RuntimeError, match="already using"):
            with runner.runtime_lock(runtime):
                pytest.fail("A second caller acquired the same runtime")
    assert (runtime / "run.lock").stat().st_ino == inode
    with runner.runtime_lock(runtime):
        assert (runtime / "run.lock").stat().st_ino == inode


def test_runtime_lock_is_released_when_the_run_raises(runner, tmp_path):
    with pytest.raises(RuntimeError, match="injected"):
        with runner.runtime_lock(tmp_path):
            raise RuntimeError("injected")
    with runner.runtime_lock(tmp_path):
        pass


class FakeChild:
    def __init__(self, pid, *, exited=False, needs_kill=False):
        self.pid = pid
        self.exited = exited
        self.needs_kill = needs_kill
        self.wait_timeouts = []

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.needs_kill and len(self.wait_timeouts) == 1:
            raise subprocess.TimeoutExpired("fake-child", timeout)
        return 0


def test_service_failure_stops_only_owned_process_groups(runner, tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    children = [FakeChild(101), FakeChild(102, exited=True), FakeChild(103, needs_kill=True)]
    starts, signals = [], []

    def start(args, **kwargs):
        starts.append((args, kwargs))
        return children[len(starts) - 1]

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="injected"):
        with runner.services(tmp_path):
            raise RuntimeError("injected")

    assert len(starts) == 3
    assert all(kwargs["start_new_session"] for _, kwargs in starts)
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in starts)
    assert set(signals) == {(101, signal.SIGTERM), (103, signal.SIGTERM), (103, signal.SIGKILL)}
    assert children[2].wait_timeouts == [20, 5]
    assert children[0].wait_timeouts == children[1].wait_timeouts == [20]


def test_partial_service_startup_cleans_up_the_child_that_did_start(runner, tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    child = FakeChild(201)
    signals = []
    calls = 0

    def start(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected spawn failure")
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(OSError, match="spawn failure"):
        with runner.services(tmp_path):
            pytest.fail("The setup should fail before yielding")
    assert signals == [(201, signal.SIGTERM)]
    assert child.wait_timeouts == [20]


def test_reproduce_holds_runtime_lock_for_prepare_through_cleanup(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def work(_output, _source, _timeout):
        with pytest.raises(RuntimeError, match="already using"):
            with runner.runtime_lock(tmp_path / ".runtime"):
                pytest.fail("The runtime lock was not held")

    monkeypatch.setattr(runner, "_reproduce", work)
    runner.reproduce(tmp_path / "output", None, 30)
    with runner.runtime_lock(tmp_path / ".runtime"):
        pass


def test_lost_creation_lookup_uses_exact_parameterized_names(runner, tmp_path, monkeypatch):
    import asyncpg

    (tmp_path / "credentials.json").write_text(json.dumps({"database_password": "fake-password"}))
    recovered = uuid4()
    names = {"cp-demo-bob", "cp-demo-'quoted'-alice"}
    calls = []

    class Connection:
        async def fetch(self, query, parameters):
            calls.append((query, parameters))
            return [{"id": recovered}]

        async def close(self):
            calls.append("closed")

    async def connect(**kwargs):
        assert kwargs["host"] == "127.0.0.1"
        return Connection()

    monkeypatch.setattr(asyncpg, "connect", connect)
    assert asyncio.run(runner.resolve_owned_session_ids(tmp_path, names)) == {str(recovered)}
    query, parameters = calls[0]
    assert "name = ANY($1::text[])" in query
    assert parameters == sorted(names)
    assert not any(name in query for name in names)
    assert calls[-1] == "closed"


def test_workload_cleanup_removes_only_exact_accepted_and_recovered_session_ids(
    runner, tmp_path, monkeypatch
):
    accepted_id, recovered_id, unrelated_id = (str(uuid4()) for _ in range(3))
    containers = {accepted_id: "owned-accepted", recovered_id: "owned-recovered"}
    queried, removed = [], []

    async def resolve(runtime, names):
        assert runtime == tmp_path
        assert names == {"exact-run-owned-name"}
        return {recovered_id}

    def output(*args):
        if args[:3] == ("docker", "ps", "-aq"):
            session_id = args[-1].removeprefix("label=ai.backend.session-id=")
            queried.append(session_id)
            container = containers[session_id]
            return "" if container in removed else container
        assert args[:2] == ("docker", "inspect")
        owner = next(key for key, value in containers.items() if value == args[2])
        return json.dumps([{"Config": {"Labels": {"ai.backend.session-id": owner}}}])

    def command(args, **kwargs):
        assert args[:3] == ["docker", "rm", "-f"]
        assert kwargs["check"] is True
        removed.append(args[3])

    monkeypatch.setattr(runner, "resolve_owned_session_ids", resolve)
    monkeypatch.setattr(runner, "command_output", output)
    monkeypatch.setattr(runner.subprocess, "run", command)
    result = runner.cleanup_workload_containers(tmp_path, {accepted_id}, {"exact-run-owned-name"})
    assert set(queried) == {accepted_id, recovered_id}
    assert unrelated_id not in queried
    assert set(removed) == {"owned-accepted", "owned-recovered"}
    assert result == {"complete": True, "checked_session_count": 2, "removed_container_count": 2}


def test_workload_cleanup_refuses_an_inspection_with_a_different_owner(
    runner, tmp_path, monkeypatch
):
    owned_id, unrelated_id = str(uuid4()), str(uuid4())
    removed = []

    async def resolve(*_args):
        return set()

    def output(*args):
        if args[1] == "ps":
            return "container-with-unexpected-owner"
        return json.dumps([{"Config": {"Labels": {"ai.backend.session-id": unrelated_id}}}])

    monkeypatch.setattr(runner, "resolve_owned_session_ids", resolve)
    monkeypatch.setattr(runner, "command_output", output)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: removed.append(args))
    with pytest.raises(RuntimeError, match="ownership changed"):
        runner.cleanup_workload_containers(tmp_path, {owned_id}, set())
    assert removed == []


def test_workload_cleanup_propagates_unresolved_ownership_before_touching_docker(
    runner, tmp_path, monkeypatch
):
    async def unavailable(*_args):
        raise OSError("Database unavailable")

    monkeypatch.setattr(runner, "resolve_owned_session_ids", unavailable)
    monkeypatch.setattr(
        runner, "command_output", lambda *_args: pytest.fail("Docker must not run before ownership")
    )
    with pytest.raises(OSError, match="Database unavailable"):
        runner.cleanup_workload_containers(tmp_path, {str(uuid4())}, {"owned-name"})


def test_cleanup_failure_preserves_compose_database_for_recovery(runner, tmp_path, monkeypatch):
    commands = []

    def prepare(_source, runtime, _cpu_count):
        (runtime / "logs").mkdir(parents=True)

    def fail_bootstrap(*_args, **_kwargs):
        raise RuntimeError("Injected experiment failure")

    def fail_cleanup(*_args):
        raise OSError("Injected Docker cleanup failure")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "command_output", lambda *_args: "4")
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "bootstrap", fail_bootstrap)
    monkeypatch.setattr(runner, "cleanup_workload_containers", fail_cleanup)
    monkeypatch.setattr(runner.subprocess, "run", lambda args, **_kwargs: commands.append(args))
    with pytest.raises(OSError, match="Docker cleanup failure"):
        runner._reproduce(tmp_path / "output", None, 30)
    assert len(commands) == 1
    assert "up" in commands[0]
    assert not any("down" in command for command in commands)
