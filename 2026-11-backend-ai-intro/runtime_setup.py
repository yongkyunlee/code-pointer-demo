"""Prepare an isolated CPU-only Backend.AI 26.9.7 development installation.

The upstream checkout is read through ``git show`` at an exact commit. Existing
files in that checkout are never changed. All generated secrets and runtime
configuration are written under --runtime, which must remain gitignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO

import tomlkit
from zmq.auth import create_certificates

SOURCE_COMMIT = "332a308e34c5c7bcd57939850fe032cfd8d8ca00"
SOURCE_URL = "https://github.com/lablup/backend.ai.git"
NAMESPACE = "code-pointer-backend-ai-intro"
PORTS = {
    "postgres": 28100,
    "valkey": 28110,
    "etcd": 28120,
    "manager": 28091,
    "manager_internal": 28092,
    "agent": 26011,
    "agent_metrics": 26013,
    "agent_sock": 26017,
    "storage_client": 26021,
    "storage_manager": 26022,
    "storage_internal": 26023,
}
FIXTURES = (
    "example-container-registries-harbor.json",
    "example-users.json",
    "example-keypairs.json",
    "example-resource-slot-types.json",
    "example-resource-presets.json",
    "example-runtime-variants.json",
    "example-runtime-variant-presets.json",
    "example-login-client-types.json",
    "example-role-presets.json",
    "example-prometheus-query-preset-categories.json",
    "example-prometheus-query-presets.json",
    "example-retention-policies.json",
)


def require_supported_live_host() -> None:
    """Limit live service setup to the host configuration verified end to end."""
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "Live experiments require macOS ARM64 with Docker Desktop. "
            "Use the component experiments or replay saved recordings on other hosts."
        )


def workload_image() -> tuple[str, dict[str, str]]:
    """Return the image pin used by the verified macOS ARM64 live setup."""
    require_supported_live_host()
    pins = json.loads(Path(__file__).with_name("images.json").read_text())
    return "aarch64", pins["aarch64"]


def upstream(source: Path, relative: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "show", f"{SOURCE_COMMIT}:{relative}"], text=True
    )


def ensure_source(source: Path | None, runtime: Path) -> Path:
    """Fetch only the pinned commit when a developer checkout is not supplied."""
    checkout = source.resolve() if source is not None else runtime / "upstream.git"
    if source is None and not checkout.exists():
        subprocess.run(
            ["git", "init", "--bare", str(checkout)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    result = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        if source is not None:
            raise ValueError(f"The supplied source checkout lacks commit {SOURCE_COMMIT}")
        subprocess.run(
            ["git", "-C", str(checkout), "fetch", "--depth=1", SOURCE_URL, SOURCE_COMMIT],
            check=True,
            timeout=300,
        )
    return checkout


def private_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
    path.chmod(0o600)


def address(port: int) -> dict[str, str | int]:
    return {"host": "127.0.0.1", "port": port}


def prepare(source: Path | None, runtime: Path, cpu_count: int) -> None:
    require_supported_live_host()
    if cpu_count < 4:
        raise ValueError("Docker must expose at least four CPUs")
    runtime.mkdir(parents=True, exist_ok=True)
    runtime.chmod(0o700)
    source = ensure_source(source, runtime)
    for subdir in (
        "var",
        "scratches",
        "vfolders/volume1",
        "logs",
        "fixtures/manager",
        "keys",
    ):
        (runtime / subdir).mkdir(parents=True, exist_ok=True)
    secret_file = runtime / "credentials.json"
    if secret_file.exists():
        credentials = json.loads(secret_file.read_text())
        # Preserve the project used by the first local development run.
        credentials.setdefault("project_name", "code-pointer-backend-ai-intro")
    else:
        credentials = {
            "project_name": "code-pointer-backend-ai-" + secrets.token_hex(5),
            "database_password": secrets.token_hex(24),
            "storage_secret": secrets.token_hex(32),
            "storage_manager_secret": secrets.token_hex(32),
            "wsproxy_api_token": secrets.token_urlsafe(24),
            "users": {},
        }
    # macOS Unix-domain sockets have a short pathname limit. The repository
    # location can be arbitrarily long, so socket files need a short private root.
    if "ipc_path" not in credentials:
        credentials["ipc_path"] = tempfile.mkdtemp(prefix="cp-bai-", dir="/tmp")
    ipc_path = Path(credentials["ipc_path"])
    if ipc_path.is_symlink():
        raise ValueError("Refusing a symlink as the IPC directory")
    ipc_path.mkdir(mode=0o700, exist_ok=True)

    for name in ("manager", "agent"):
        if not (runtime / "keys" / f"{name}.key_secret").exists():
            create_certificates(str(runtime / "keys"), name)
        (runtime / "keys" / f"{name}.key_secret").chmod(0o600)

    def template(name: str):
        data = tomlkit.parse(upstream(source, name))
        data["etcd"].update(namespace=NAMESPACE, addr=address(PORTS["etcd"]))
        data["logging"]["console"]["colored"] = False
        return data

    manager = template("configs/manager/halfstack.toml")
    manager["db"].update(addr=address(PORTS["postgres"]), password=credentials["database_password"])
    manager["manager"].update(
        {
            "num-proc": 1,
            "service-addr": address(PORTS["manager"]),
            "internal-addr": address(PORTS["manager_internal"]),
            "announce-addr": address(PORTS["manager"]),
            "announce-internal-addr": address(PORTS["manager_internal"]),
            "rpc-auth-manager-keypair": str(runtime / "keys/manager.key_secret"),
            "ipc-base-path": str(ipc_path),
            "pid-file": str(runtime / "manager.pid"),
            "hide-agents": False,
            "agent-selection-resource-priority": ["cpu", "mem"],
        }
    )
    private_write(runtime / "manager.toml", tomlkit.dumps(manager))

    agent = template("configs/agent/halfstack.toml")
    agent["agent"].update(
        {
            "id": "code-pointer-cpu-worker",
            "rpc-listen-addr": address(PORTS["agent"]),
            "service-addr": address(PORTS["agent_metrics"]),
            "metadata-server-bind-host": "127.0.0.1",
            "agent-sock-port": PORTS["agent_sock"],
            "announce-addr": address(PORTS["agent_metrics"]),
            "rpc-auth-manager-public-key": str(runtime / "keys/manager.key"),
            "rpc-auth-agent-keypair": str(runtime / "keys/agent.key_secret"),
            "ipc-base-path": str(ipc_path),
            "var-base-path": str(runtime / "var"),
            "pid-file": str(runtime / "agent.pid"),
            "allow-compute-plugins": [],
            "mount-path": str(runtime / "vfolders"),
            "image-commit-path": str(runtime / "commits"),
        }
    )
    agent["container"].update(
        {
            "scratch-root": str(runtime / "scratches"),
            "port-range": [31000, 31100],
            "kernel-uid": -1,
            "kernel-gid": -1,
        }
    )
    agent["resource"].update(
        {"reserved-cpu": cpu_count - 4, "reserved-mem": "1G", "reserved-disk": "1G"}
    )
    agent.pop("watcher", None)
    private_write(runtime / "agent.toml", tomlkit.dumps(agent))

    storage = template("configs/storage-proxy/halfstack.toml")
    storage["storage-proxy"].update(
        {
            "node-id": "code-pointer-storage",
            "ipc-base-path": str(ipc_path),
            "secret": credentials["storage_secret"],
            "pid-file": str(runtime / "storage.pid"),
        }
    )
    storage["api"]["client"].update(
        {"service-addr": address(PORTS["storage_client"]), "ssl-enabled": False}
    )
    storage["api"]["manager"].update(
        {
            "service-addr": address(PORTS["storage_manager"]),
            "internal-addr": address(PORTS["storage_internal"]),
            "announce-addr": address(PORTS["storage_manager"]),
            "announce-internal-addr": address(PORTS["storage_internal"]),
            "ssl-enabled": False,
            "secret": credentials["storage_manager_secret"],
        }
    )
    for key in ("ssl-cert", "ssl-privkey"):
        storage["api"]["manager"].pop(key, None)
    storage["volume"] = {"volume1": {"backend": "vfs", "path": str(runtime / "vfolders/volume1")}}
    storage.pop("artifact-storages", None)
    private_write(runtime / "storage-proxy.toml", tomlkit.dumps(storage))
    (runtime / "vfolders/volume1/version.txt").write_text("3\n")

    alembic = upstream(source, "configs/manager/halfstack.alembic.ini")
    alembic = alembic.replace(
        "postgres:develove@localhost:8100",
        f"postgres:{credentials['database_password']}@127.0.0.1:{PORTS['postgres']}",
    )
    private_write(runtime / "alembic.ini", alembic)

    fixture_data = {
        name: json.loads(upstream(source, f"fixtures/manager/{name}")) for name in FIXTURES
    }
    # The pinned example uses strings, but the database BIGINT binding requires
    # integers for shared_memory. Normalize only that fixture field.
    for preset in fixture_data["example-resource-presets.json"]["resource_presets"]:
        if preset.get("shared_memory") is not None:
            preset["shared_memory"] = int(preset["shared_memory"])
    user_data = fixture_data["example-users.json"]
    aliases = {"admin": "admin", "user": "alice", "user2": "bob"}
    by_uuid = {}
    for user in user_data["users"]:
        alias = aliases.get(user["username"], user["username"])
        record = credentials["users"].setdefault(
            alias,
            {
                "password": secrets.token_urlsafe(24),
                "access_key": secrets.token_hex(10).upper(),
                "secret_key": secrets.token_urlsafe(32),
                "uuid": user["uuid"],
            },
        )
        user["email"] = f"{alias}@demo.invalid"
        user["password"] = record["password"]
        record["email"] = user["email"]
        by_uuid[user["uuid"]] = record
    for group in user_data["scaling_groups"]:
        group["wsproxy_api_token"] = credentials.setdefault(
            "wsproxy_api_token", secrets.token_urlsafe(24)
        )
    for pair in fixture_data["example-keypairs.json"]["keypairs"]:
        record = by_uuid[pair["user"]]
        pair["access_key"], pair["secret_key"] = record["access_key"], record["secret_key"]
    for name, data in fixture_data.items():
        private_write(runtime / "fixtures/manager" / name, json.dumps(data, indent=2) + "\n")

    private_write(secret_file, json.dumps(credentials, indent=2) + "\n")
    private_write(
        runtime / ".env",
        f"BACKEND_DEMO_DB_PASSWORD={credentials['database_password']}\n"
        f"BACKEND_DEMO_PROJECT={credentials['project_name']}\n",
    )
    config = {
        "redis": {"addr": f"127.0.0.1:{PORTS['valkey']}"},
        "docker": {"image": {"auto_pull": "tag"}},
    }
    private_write(runtime / "etcd.config.json", json.dumps(config, indent=2))
    volumes = {
        "_types": {"group": "", "user": ""},
        "default_host": "local:volume1",
        "proxies": {
            "local": {
                "client_api": f"http://127.0.0.1:{PORTS['storage_client']}",
                "manager_api": f"http://127.0.0.1:{PORTS['storage_manager']}",
                "secret": credentials["storage_manager_secret"],
                "ssl_verify": "false",
            }
        },
        "exposed_volume_info": "percentage",
    }
    private_write(runtime / "etcd.volumes.json", json.dumps(volumes, indent=2))
    private_write(
        runtime / "runtime.json",
        json.dumps(
            {
                "ports": PORTS,
                "source_commit": SOURCE_COMMIT,
                "cpu_capacity": 4,
                "ipc_path": str(ipc_path),
            },
            indent=2,
        ),
    )


async def verify_bootstrap(runtime: Path) -> dict[str, object]:
    """Verify database state because upstream image CLI can log errors and exit zero."""
    import asyncpg

    credentials = json.loads((runtime / "credentials.json").read_text())
    connection = await asyncpg.connect(
        host="127.0.0.1",
        port=PORTS["postgres"],
        database="backend",
        user="postgres",
        password=credentials["database_password"],
        timeout=10,
    )
    try:
        group = await connection.fetchrow(
            "SELECT scheduler, is_active FROM scaling_groups WHERE name = 'default'"
        )
        if group is None or group["scheduler"] != "fifo" or not group["is_active"]:
            raise RuntimeError("The demo resource group is not active with FIFO scheduling")
        for alias in ("admin", "alice", "bob"):
            user = credentials["users"][alias]
            found = await connection.fetchval(
                "SELECT count(*) FROM keypairs WHERE access_key = $1 AND is_active",
                user["access_key"],
            )
            if found != 1:
                raise RuntimeError(f"The {alias} API key is not present and active")
        image = await connection.fetchrow(
            "SELECT i.name, i.tag, i.registry, i.project, i.architecture, i.config_digest "
            "FROM image_aliases AS a JOIN images AS i ON i.id = a.image "
            "WHERE a.alias = 'python'"
        )
        if image is None:
            raise RuntimeError("Image scan or alias creation failed; python alias is missing")
        architecture, pin = workload_image()
        metadata = dict(image)
        metadata["config_digest"] = metadata["config_digest"].strip()
        if metadata["architecture"] != architecture or metadata["name"] != pin["name"]:
            raise RuntimeError("The python alias does not match the pinned native workload image")
        if metadata["config_digest"] != pin["config_digest"]:
            raise RuntimeError(
                "The workload image tag no longer matches the checked-in config digest. "
                "Refusing to run with changed image metadata."
            )
        metadata["manifest_digest"] = pin["manifest_digest"]
        metadata["pinned_reference"] = pin["pinned_reference"]
        result = {"scheduler": group["scheduler"], "image": metadata}
        private_write(runtime / "bootstrap-result.json", json.dumps(result, indent=2))
        return result
    finally:
        await connection.close()


def bootstrap(runtime: Path, log: TextIO | None = None) -> dict[str, object]:
    """Initialize only this installation; infrastructure must already be healthy."""
    command = [sys.executable, "-m", "ai.backend.manager.cli", "-f", str(runtime / "manager.toml")]

    def manager(*args: str) -> None:
        subprocess.run(
            [*command, *args],
            cwd=runtime,
            check=True,
            timeout=300,
            stdout=log,
            stderr=subprocess.STDOUT if log is not None else None,
        )

    manager("etcd", "put-json", "config", str(runtime / "etcd.config.json"))
    manager("etcd", "put-json", "volumes", str(runtime / "etcd.volumes.json"))
    manager("schema", "oneshot", "-f", str(runtime / "alembic.ini"))
    for name in FIXTURES:
        manager("fixture", "populate", str(runtime / "fixtures/manager" / name))
    manager("permissions", "provision")
    architecture, pin = workload_image()
    image = pin["name"]
    manager("image", "rescan", image)
    manager("image", "alias", "python", image, architecture)
    return asyncio.run(verify_bootstrap(runtime))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Existing Backend.AI checkout; otherwise fetch the pinned commit into the runtime",
    )
    parser.add_argument(
        "--runtime", type=Path, default=Path(__file__).resolve().parent / ".runtime"
    )
    parser.add_argument(
        "--docker-cpus",
        type=int,
        default=8,
        help="Actual Docker CPU capacity, verified with docker info",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Initialize the database and registry after compose services are healthy",
    )
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    prepare(args.source, runtime, args.docker_cpus)
    print(f"Prepared configuration in {runtime}; secrets remain in that directory.")
    if args.bootstrap:
        bootstrap(runtime)


if __name__ == "__main__":
    main()
