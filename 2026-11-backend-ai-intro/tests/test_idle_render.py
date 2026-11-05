"""Do not turn a failed supervised run into successful-looking idle evidence."""

import pytest

from backend_ai_intro.idle_render import render_idle_recording


@pytest.mark.parametrize("field", ["failure", "run_failure"])
def test_idle_replay_rejects_failure_even_after_scenario_succeeded(field, tmp_path):
    recording = {
        "verification": {"passed": True},
        "cleanup": {"complete": True},
        "environment_cleanup": {"complete": True},
        field: {"type": "RuntimeError"},
    }
    with pytest.raises(ValueError, match="failed idle run"):
        render_idle_recording(recording, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cleanup", [None, {"complete": False}])
def test_idle_replay_requires_finished_infrastructure_cleanup(cleanup, tmp_path):
    recording = {"verification": {"passed": True}, "cleanup": {"complete": True}}
    if cleanup is not None:
        recording["environment_cleanup"] = cleanup
    with pytest.raises(ValueError, match="infrastructure cleanup"):
        render_idle_recording(recording, tmp_path)
    assert not list(tmp_path.iterdir())
