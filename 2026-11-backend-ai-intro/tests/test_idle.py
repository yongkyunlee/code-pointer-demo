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
