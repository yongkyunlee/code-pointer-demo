"""Configure real Prometheus observations for the isolated idle experiment."""

from __future__ import annotations

import json
from pathlib import Path

import tomlkit

PROMETHEUS_PORT = 29090
PROMETHEUS_IMAGE = (
    "prom/prometheus:v3.1.0@sha256:6559acbd5d770b15bb3c954629ce190ac3cbbdb2b7f1c30f0385c4e05104e218"
)
METRIC_PRESET_ID = "d5d83b5e-3463-4b5f-beda-1afbe4761e3c"
CPU_THRESHOLD_MILLICORES = 100
IDLE_SECONDS = 60
BASELINE_SECONDS = 75


def configure_runtime(runtime: Path) -> Path:
    """Add an isolated metrics service; called after common runtime preparation.

    The returned compose override belongs to the same isolated compose project.
    No checkers are enabled here. The live experiment creates one through the API
    and includes only its own sessions after recording the policy-off control.
    """
    runtime = runtime.resolve()
    manager_path = runtime / "manager.toml"
    manager = tomlkit.parse(manager_path.read_text())
    manager["metric"] = {"addr": {"host": "127.0.0.1", "port": PROMETHEUS_PORT}}
    manager_path.write_text(tomlkit.dumps(manager))
    prometheus_path = runtime / "idle-prometheus.yml"
    prometheus_path.write_text(
        "global:\n  scrape_interval: 2s\n  evaluation_interval: 2s\n"
        "scrape_configs:\n  - job_name: backend-ai-demo-agent\n"
        "    metrics_path: /metrics\n    static_configs:\n"
        "      - targets: ['host.docker.internal:26013']\n"
    )
    prometheus_path.chmod(0o644)
    compose_path = runtime / "idle-compose.yaml"
    # JSON is valid YAML and safely quotes arbitrary absolute paths.
    compose = {
        "services": {
            "prometheus": {
                "image": PROMETHEUS_IMAGE,
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.retention.time=1h",
                ],
                "ports": [f"127.0.0.1:{PROMETHEUS_PORT}:9090"],
                "volumes": [f"{prometheus_path}:/etc/prometheus/prometheus.yml:ro"],
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "healthcheck": {
                    "test": [
                        "CMD",
                        "wget",
                        "-q",
                        "-O",
                        "/dev/null",
                        "http://127.0.0.1:9090/-/ready",
                    ],
                    "interval": "2s",
                    "timeout": "2s",
                    "retries": 30,
                },
            },
        },
    }
    compose_path.write_text(json.dumps(compose, indent=2) + "\n")
    provenance = {
        "compose_file": str(compose_path),
        "compose_override": str(compose_path),
        "prometheus_image": PROMETHEUS_IMAGE,
        "prometheus_endpoint": f"http://127.0.0.1:{PROMETHEUS_PORT}",
        "metric_preset_id": METRIC_PRESET_ID,
        "metric": "rate(cpu_used/current[5m])",
        "metric_unit": "millicores",
        "threshold": CPU_THRESHOLD_MILLICORES,
        "max_underutilized_duration_seconds": IDLE_SECONDS,
        "policy_off_observation_seconds": BASELINE_SECONDS,
        "scope": "Only the experiment's idle and active sessions are manually included",
    }
    (runtime / "idle-setup.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return compose_path
