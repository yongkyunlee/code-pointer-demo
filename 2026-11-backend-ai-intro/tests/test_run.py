"""Check orchestration cleanup using fake child processes, never Docker."""

import asyncio
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("run")
    monkeypatch.setattr(module, "require_supported_live_host", lambda: None)
    return module


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
    assert set(signals) == {
        (pid, number) for pid in (101, 102, 103) for number in (signal.SIGTERM, signal.SIGKILL)
    }
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
    assert signals == [(201, signal.SIGTERM), (201, signal.SIGKILL)]
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
    monkeypatch.setattr(
        runner,
        "command_output",
        lambda *_args: "unix:///tmp/test-docker.sock" if "context" in _args else "4",
    )
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "require_clean_project", lambda *_args: None)
    monkeypatch.setattr(runner, "bootstrap", fail_bootstrap)
    monkeypatch.setattr(runner, "cleanup_workload_containers", fail_cleanup)
    monkeypatch.setattr(runner.subprocess, "run", lambda args, **_kwargs: commands.append(args))
    with pytest.raises(OSError, match="Docker cleanup failure"):
        runner._reproduce(tmp_path / "output", None, 30)
    assert len(commands) == 1
    assert "up" in commands[0]
    assert not any("down" in command for command in commands)


@pytest.mark.parametrize("timeout", [-1, 0, 4, 3601, float("nan"), float("inf")])
def test_invalid_timeout_fails_before_lock_or_docker(runner, tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_reproduce", lambda *_args: pytest.fail("Run must not start"))
    with pytest.raises(ValueError, match="stage_timeout"):
        runner.reproduce(tmp_path / "output", None, timeout)
    assert not (tmp_path / ".runtime").exists()
    assert not (tmp_path / "output").exists()


def test_failed_rerun_discards_old_success_artifacts_and_preserves_unrelated_files(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    for name in (
        "recording.json",
        "scheduler.gif",
        "storyboard.png",
        "keyframe.png",
        "manifest.json",
        "environment-cleanup.json",
    ):
        (output / name).write_text("old successful run")
    (output / "my-notes.txt").write_text("keep")

    def fail(*_args):
        raise RuntimeError("a diagnostic containing a secret")

    monkeypatch.setattr(runner, "_reproduce", fail)
    with pytest.raises(RuntimeError):
        runner.reproduce(output, None, 30)
    assert sorted(path.name for path in output.iterdir()) == ["my-notes.txt", "run-status.json"]
    assert (output / "my-notes.txt").read_text() == "keep"
    assert json.loads((output / "run-status.json").read_text()) == {
        "passed": False,
        "failure": {"type": "RuntimeError"},
    }


def test_outer_failure_preserves_experiment_checks_without_copying_diagnostics(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "output"

    def fail(output, *_args):
        (output / "recording.json").write_text(json.dumps({"verification": {"passed": True}}))
        raise OSError("private diagnostic")

    monkeypatch.setattr(runner, "_reproduce", fail)
    with pytest.raises(OSError):
        runner.reproduce(output, None, 30)
    recording = json.loads((output / "recording.json").read_text())
    assert recording == {"verification": {"passed": False}, "run_failure": {"type": "OSError"}}
    assert "private diagnostic" not in (output / "run-status.json").read_text()


@pytest.mark.parametrize(
    "endpoint", ["ssh://remote", "tcp://127.0.0.1:2375", "npipe:////./pipe/docker_engine"]
)
def test_remote_or_unsupported_docker_endpoint_rejected_before_setup(
    runner, tmp_path, monkeypatch, endpoint
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "command_output", lambda *_args: endpoint)
    monkeypatch.setattr(
        runner, "prepare", lambda *_args: pytest.fail("Must reject before preparing")
    )
    with pytest.raises(RuntimeError, match="local Docker engine"):
        runner._reproduce(tmp_path / "output", None, 30)


@pytest.mark.parametrize("failure_stage", ["workload_containers", "infrastructure"])
def test_cleanup_failure_persists_result_and_does_not_render(
    runner, tmp_path, monkeypatch, failure_stage
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "recording.json").write_text(json.dumps({"verification": {"passed": True}}))
    commands = []

    def prepare(_source, runtime, _cpu_count):
        (runtime / "logs").mkdir(parents=True)

    def bootstrap(*_args, **_kwargs):
        raise RuntimeError("bootstrap failure")

    def cleanup(*_args):
        if failure_stage == "workload_containers":
            raise OSError("private cleanup diagnostic")
        return {"complete": True, "checked_session_count": 0, "removed_container_count": 0}

    def command(args, **_kwargs):
        commands.append(args)
        if "down" in args:
            raise OSError("private teardown diagnostic")

    monkeypatch.setattr(
        runner,
        "command_output",
        lambda *_args: "unix:///tmp/docker.sock" if "context" in _args else "4",
    )
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "require_clean_project", lambda *_args: None)
    monkeypatch.setattr(runner, "bootstrap", bootstrap)
    monkeypatch.setattr(runner, "cleanup_workload_containers", cleanup)
    monkeypatch.setattr(runner.subprocess, "run", command)
    monkeypatch.setattr(
        runner, "save_artifacts", lambda *_args: pytest.fail("Do not render failure")
    )
    with pytest.raises(OSError):
        runner._reproduce(output, None, 30)
    recording = json.loads((output / "recording.json").read_text())
    outcome = recording["environment_cleanup"]
    assert outcome["complete"] is False
    assert outcome["stage"] == failure_stage
    assert outcome["error"] == {"type": "OSError"}
    assert recording["verification"]["passed"] is False
    assert json.loads((output / "environment-cleanup.json").read_text()) == outcome
    assert any("down" in args for args in commands) == (failure_stage == "infrastructure")


@pytest.mark.parametrize("retained", ["container", "volume"])
def test_retained_project_resources_are_preserved(runner, tmp_path, monkeypatch, retained):
    (tmp_path / "credentials.json").write_text(json.dumps({"project_name": "owned-project"}))
    commands = []

    def output(*args):
        commands.append(args)
        return retained if ("ps" in args) == (retained == "container") else ""

    monkeypatch.setattr(runner, "command_output", output)
    with pytest.raises(RuntimeError, match="database have been preserved"):
        runner.require_clean_project(["docker", "compose"], tmp_path)
    assert not any("up" in command or "down" in command or "rm" in command for command in commands)
    assert "label=com.docker.compose.project=owned-project" in commands[-1]


def test_termination_handlers_are_restored(runner):
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    with pytest.raises(RuntimeError):
        with runner.termination_signals():
            assert all(signal.getsignal(number) != before[number] for number in before)
            raise RuntimeError("injected")
    assert all(signal.getsignal(number) == previous for number, previous in before.items())


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGHUP])
def test_termination_unwinds_real_child_processes_and_records_failure(runner, tmp_path, number):
    # The real OS signal must unwind services(), not only a mocked context manager.
    script = tmp_path / "supervisor.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        f"sys.path.insert(0, {str(Path(runner.__file__).parent)!r})\n"
        "import run\n"
        "run.require_supported_live_host = lambda: None\n"
        f"run.ROOT = pathlib.Path({str(tmp_path)!r})\n"
        "original_popen = subprocess.Popen\n"
        "def launch(*args, **kwargs):\n"
        "    child = original_popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True)\n"
        "    with (run.ROOT / 'children').open('a') as stream: stream.write(str(child.pid)+'\\n')\n"
        "    return child\n"
        "def work(output, source, timeout):\n"
        "    (run.ROOT / '.runtime' / 'logs').mkdir(parents=True, exist_ok=True)\n"
        "    run.subprocess.Popen = launch\n"
        "    with run.services(run.ROOT / '.runtime'):\n"
        "        try:\n"
        "            (run.ROOT / 'ready').touch()\n"
        "            time.sleep(60)\n"
        "        finally:\n"
        "            (run.ROOT / 'cleaning').touch()\n"
        "            time.sleep(0.3)\n"
        "run._reproduce = work\n"
        "run.reproduce(run.ROOT / 'output', None, 30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def wait_for(path):
        deadline = time.monotonic() + 10
        while not path.exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                pytest.fail(f"Supervisor did not reach {path.name}")
            time.sleep(0.02)

    try:
        wait_for(tmp_path / "ready")
        child.send_signal(number)
        wait_for(tmp_path / "cleaning")
        child.send_signal(number)  # A second termination must not interrupt cleanup.
        child.communicate(timeout=10)
        assert child.returncode != 0
        status = json.loads((tmp_path / "output/run-status.json").read_text())
        assert status == {"passed": False, "failure": {"type": "ExperimentInterrupted"}}
        for pid in (tmp_path / "children").read_text().splitlines():
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid), 0)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        if (tmp_path / "children").exists():
            for pid in (tmp_path / "children").read_text().splitlines():
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("argument", ["--output=custom", "--output"])
def test_fifo_wrapper_preserves_explicit_output(runner, tmp_path, monkeypatch, argument):
    import runpy

    wrapper = Path(runner.__file__).parent / "experiments/01-fifo-queue/run.py"
    calls = []
    original = runpy.run_path
    monkeypatch.setattr(runpy, "run_path", lambda *args, **kwargs: calls.append(list(sys.argv)))
    arguments = [str(wrapper), argument] + (["custom"] if argument == "--output" else [])
    monkeypatch.setattr(sys, "argv", arguments)
    original(str(wrapper), run_name="__main__")
    assert calls[0][1:] == arguments[1:]
    # Importing the wrapper should not start an experiment.
    calls.clear()
    original(str(wrapper), run_name="wrapper_import")
    assert calls == []


def test_services_kill_descendants_after_leaders_already_exited(runner, tmp_path, monkeypatch):
    import psutil

    (tmp_path / "logs").mkdir()
    original_popen = subprocess.Popen
    leaders = []
    descendants = []

    def start(_args, **_kwargs):
        index = len(leaders)
        pid_path = tmp_path / f"descendant-{index}"
        code = (
            "import os, pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        leader = original_popen(
            [
                sys.executable,
                "-c",
                f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{code!r}])",
            ],
            start_new_session=True,
        )
        leaders.append(leader)
        deadline = time.monotonic() + 5
        while not pid_path.exists():
            if time.monotonic() >= deadline:
                pytest.fail("Descendant did not start")
            time.sleep(0.02)
        descendants.append(psutil.Process(int(pid_path.read_text())))
        leader.wait(timeout=5)
        return leader

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    try:
        with runner.services(tmp_path):
            assert all(leader.poll() is not None for leader in leaders)
            assert all(process.is_running() for process in descendants)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all(
                not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
                for process in descendants
            ):
                break
            time.sleep(0.02)
        else:
            pytest.fail("A descendant survived process-group cleanup")
    finally:
        for leader in leaders:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "system,machine", [("Linux", "aarch64"), ("Windows", "AMD64"), ("Darwin", "x86_64")]
)
def test_unsupported_live_host_fails_before_any_setup(
    runner, tmp_path, monkeypatch, system, machine
):
    import runtime_setup

    monkeypatch.setattr(runtime_setup.platform, "system", lambda: system)
    monkeypatch.setattr(runtime_setup.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        runner, "require_supported_live_host", runtime_setup.require_supported_live_host
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "command_output", lambda *_args: pytest.fail("No Docker command"))
    monkeypatch.setattr(runner, "runtime_lock", lambda *_args: pytest.fail("No runtime lock"))
    monkeypatch.setattr(
        runtime_setup, "ensure_source", lambda *_args: pytest.fail("No Git/source work")
    )
    with pytest.raises(RuntimeError, match="macOS ARM64 with Docker Desktop"):
        runner.reproduce(tmp_path / "output", None, 30)
    with pytest.raises(RuntimeError, match="macOS ARM64 with Docker Desktop"):
        runtime_setup.prepare(None, tmp_path / ".runtime", 4)
    assert list(tmp_path.iterdir()) == []
