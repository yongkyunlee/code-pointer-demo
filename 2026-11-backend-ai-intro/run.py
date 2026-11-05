"""Reproduce the live CPU-only experiment and clean up its isolated services.

Run from a fresh checkout with `uv sync --locked`, then `uv run python run.py`.
Docker must be running with at least four CPUs and sufficient memory. Runtime
credentials and service logs remain private in .runtime; public artifacts contain
only the allowlisted observations saved by the experiment recorder.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from backend_ai_intro.cli import clear_generated_artifacts, save_artifacts
from backend_ai_intro.live import LiveConfig, run_live_experiment
from runtime_setup import PORTS, bootstrap, prepare

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_failure(output: Path, error: BaseException) -> None:
    """Record failure types without copying potentially private diagnostics."""
    write_json(
        output / "run-status.json", {"passed": False, "failure": {"type": type(error).__name__}}
    )
    path = output / "recording.json"
    if path.exists():
        recording = json.loads(path.read_text())
        recording["run_failure"] = {"type": type(error).__name__}
        write_json(path, recording)


def command_output(*args: str, timeout: float = 30) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def verify_image_identity(inspection: dict, pinned_reference: str) -> None:
    """Use the repository digest across Docker's classic and containerd stores."""
    if pinned_reference not in inspection.get("RepoDigests", []):
        raise RuntimeError("The local workload image does not match its pinned manifest digest")


async def resolve_owned_session_ids(runtime: Path, names: set[str]) -> set[str]:
    """Recover accepted creates whose API responses were lost, by exact run names."""
    import asyncpg

    if not names:
        return set()
    credentials = json.loads((runtime / "credentials.json").read_text())
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=PORTS["postgres"],
        user="postgres",
        database="backend",
        password=credentials["database_password"],
        timeout=10,
        command_timeout=10,
    )
    try:
        rows = await connection.fetch(
            "SELECT id FROM sessions WHERE name = ANY($1::text[])", sorted(names)
        )
        return {str(row["id"]) for row in rows}
    finally:
        await connection.close()


def cleanup_workload_containers(runtime: Path, ids: set[str], names: set[str]) -> dict:
    """Remove only exact run-owned sessions, including after API cleanup fails.

    Kernel containers are outside Compose. If ownership cannot be resolved or
    Docker cleanup fails, raise before removing the database needed for recovery.
    """
    owned_ids = ids | asyncio.run(resolve_owned_session_ids(runtime, names))
    removed: list[str] = []
    for session_id in sorted(owned_ids):
        session_id = str(UUID(session_id))
        query = ("docker", "ps", "-aq", "--filter", f"label=ai.backend.session-id={session_id}")
        containers = command_output(*query).split()
        for container_id in containers:
            inspection = json.loads(command_output("docker", "inspect", container_id))[0]
            if inspection["Config"]["Labels"].get("ai.backend.session-id") != session_id:
                raise RuntimeError("Container ownership changed during cleanup")
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=True,
                timeout=30,
                stdout=subprocess.DEVNULL,
            )
            removed.append(container_id)
        if command_output(*query):
            raise RuntimeError("A run-owned workload container remains after cleanup")
    return {
        "complete": True,
        "checked_session_count": len(owned_ids),
        "removed_container_count": len(removed),
    }


@contextlib.contextmanager
def environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def runtime_lock(runtime: Path) -> Iterator[None]:
    """Prevent two invocations from sharing or tearing down the same runtime."""
    import fcntl

    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Keep this inode in place after unlocking so concurrent callers share the lock.
    with (runtime / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another experiment is already using this checkout") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextlib.contextmanager
def services(runtime: Path) -> Iterator[None]:
    """Start and stop only children created by this invocation."""
    children: list[subprocess.Popen] = []
    try:
        for name, module, config in (
            ("storage", "ai.backend.storage.server", "storage-proxy.toml"),
            ("manager", "ai.backend.manager.server", "manager.toml"),
            ("agent", "ai.backend.agent.server", "agent.toml"),
        ):
            with (runtime / "logs" / f"{name}.log").open("wb") as log:
                children.append(
                    subprocess.Popen(
                        [sys.executable, "-u", "-m", module, "-f", str(runtime / config)],
                        cwd=runtime,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        yield
    finally:
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in reversed(children):
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)


def reproduce(output: Path, source: Path | None, timeout: float, *, scenario: str = "fifo") -> None:
    if scenario not in {"fifo", "idle"}:
        raise ValueError("Unknown live experiment")
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError(
            "The live setup supports macOS and Linux; use Linux through WSL on Windows"
        )
    if not 5 <= timeout <= 3600:
        raise ValueError("stage_timeout must be between 5 and 3600 seconds")
    with runtime_lock(ROOT / ".runtime"):
        output.mkdir(parents=True, exist_ok=True)
        clear_generated_artifacts(output, include_recording=True)
        write_json(output / "run-status.json", {"passed": False, "state": "running"})
        try:
            if scenario == "fifo":
                _reproduce(output, source, timeout)
            else:
                _reproduce(output, source, timeout, scenario=scenario)
        except BaseException as error:
            save_failure(output, error)
            raise
        else:
            write_json(output / "run-status.json", {"passed": True, "state": "complete"})


def _reproduce(
    output: Path, source: Path | None, timeout: float, *, scenario: str = "fifo"
) -> None:
    runtime = ROOT / ".runtime"
    docker_endpoint = command_output(
        "docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"
    )
    if not docker_endpoint.startswith("unix:///"):
        raise RuntimeError("Use a local Docker engine through a Unix socket for this experiment")
    cpu_count = int(command_output("docker", "info", "--format", "{{.NCPU}}"))
    prepare(source, runtime, cpu_count)
    compose = [
        "docker",
        "compose",
        "--env-file",
        str(runtime / ".env"),
        "-f",
        str(ROOT / "compose.yaml"),
    ]
    if scenario == "idle":
        from backend_ai_intro.idle_setup import configure_runtime

        compose.extend(["-f", str(configure_runtime(runtime))])
    output.mkdir(parents=True, exist_ok=True)
    compose_started = False
    owned_ids: set[str] = set()
    owned_names: set[str] = set()
    recording = None
    try:
        print("Starting isolated infrastructure...", flush=True)
        compose_started = True
        subprocess.run(
            [*compose, "up", "-d", "--wait", "--wait-timeout", "120"], check=True, timeout=900
        )
        print("Initializing Backend.AI from the pinned release...", flush=True)
        # Bootstrap CLI output may contain generated account details. Keep it private.
        with (runtime / "logs/bootstrap.log").open("w") as log:
            metadata = bootstrap(runtime, log=log)
        credentials = json.loads((runtime / "credentials.json").read_text())["users"]["admin"]
        architecture = "aarch64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"
        image = str(metadata["image"]["name"])
        pinned_image = str(metadata["image"]["pinned_reference"])
        print("Preparing the CPU workload image...", flush=True)
        subprocess.run(["docker", "pull", pinned_image], check=True, timeout=900)
        subprocess.run(["docker", "tag", pinned_image, image], check=True, timeout=30)
        inspection = json.loads(command_output("docker", "image", "inspect", image))[0]
        verify_image_identity(inspection, pinned_image)
        values = {
            "BACKEND_ACCESS_KEY": credentials["access_key"],
            "BACKEND_SECRET_KEY": credentials["secret_key"],
            "DOCKER_HOST": docker_endpoint,
        }
        config = LiveConfig(
            endpoint=f"http://127.0.0.1:{PORTS['manager']}",
            image=image,
            architecture=architecture,
            stage_timeout=timeout,
        )
        with environment(values), services(runtime):
            print("Recording sessions, scheduler decisions, and resource release...", flush=True)
            experiment = run_live_experiment
            if scenario == "idle":
                from backend_ai_intro.idle import run_idle_experiment

                experiment = run_idle_experiment
            recording = experiment(
                config,
                output / "recording.json",
                owned_session_ids=owned_ids,
                owned_session_names=owned_names,
            )
        if not recording["verification"]["passed"]:
            raise RuntimeError("The experiment did not pass its verification checks")
        recording["provenance"]["installation"] = {
            "source_commit": "332a308e34c5c7bcd57939850fe032cfd8d8ca00",
            "workload_manifest_digest": metadata["image"]["manifest_digest"],
            "workload_config_digest": metadata["image"]["config_digest"],
            "docker_image_id": inspection["Id"],
            "dependency_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        }
    finally:
        if compose_started:
            cleanup = {"complete": False, "stage": "workload_containers"}
            try:
                cleanup = cleanup_workload_containers(runtime, owned_ids, owned_names)
                # Workload removal alone does not establish infrastructure cleanup.
                cleanup["complete"] = False
                cleanup["stage"] = "infrastructure"
                print("Removing this experiment's infrastructure and data volumes...", flush=True)
                subprocess.run([*compose, "down", "--volumes"], check=True, timeout=120)
                cleanup["complete"] = True
                cleanup["stage"] = "complete"
            except BaseException as error:
                cleanup["error"] = {"type": type(error).__name__}
                raise
            finally:
                path = output / "recording.json"
                if recording is None and path.exists():
                    recording = json.loads(path.read_text())
                if recording is not None:
                    recording["environment_cleanup"] = cleanup
                    if not cleanup["complete"]:
                        recording.setdefault("verification", {})["passed"] = False
                    write_json(path, recording)
                # Setup can fail before the recorder creates its first sample.
                write_json(output / "environment-cleanup.json", cleanup)
    if recording is not None:
        save_artifacts(recording, output)
        print(f"Recorded and rendered the experiment in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/live")
    parser.add_argument("--source", type=Path, help="Optional existing Backend.AI clone at the pin")
    parser.add_argument("--stage-timeout", type=float, default=300)
    parser.add_argument("--scenario", choices=("fifo", "idle"), default="fifo")
    args = parser.parse_args()
    reproduce(args.output.resolve(), args.source, args.stage_timeout, scenario=args.scenario)


if __name__ == "__main__":
    main()
