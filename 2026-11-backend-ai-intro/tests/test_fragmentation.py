"""Verify the real placement experiment and reject corrupted evidence."""

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from PIL import Image

from backend_ai_intro.fragmentation import (
    FragmentationError,
    main,
    run_experiment,
    validate_recording,
)
from backend_ai_intro.fragmentation_render import render_recording


@pytest.fixture(scope="module")
def placement_recording():
    return run_experiment()


def test_installed_selector_reproduces_fragmentation_and_controls(placement_recording):
    recording = placement_recording
    assert recording["checks"]["all_passed"]
    assert recording["provenance"]["installed_distributions"]["backend.ai-manager"] == "26.9.7"
    assert all(module["matches_wheel_record"] for module in recording["provenance"]["modules"])
    assert len(recording["cases"]) == 3
    assert recording["cases"][0]["strategies"][1]["steps"][-1]["free_after"] == {
        "node-a": 3,
        "node-b": 1,
    }


def test_rejection_must_preserve_free_capacity(placement_recording):
    invalid = deepcopy(placement_recording)
    invalid["cases"][0]["strategies"][1]["steps"][-1]["free_after"]["node-a"] = 2
    with pytest.raises(FragmentationError, match="leaked an allocation"):
        validate_recording(invalid)


def test_report_requires_actual_resource_failure(placement_recording):
    invalid = deepcopy(placement_recording)
    invalid["cases"][0]["strategies"][1]["steps"][-1]["failures"] = []
    with pytest.raises(FragmentationError, match="failure evidence"):
        validate_recording(invalid)


def test_wrong_node_or_slot_delta_cannot_be_presented_as_a_placement(placement_recording):
    invalid = deepcopy(placement_recording)
    invalid["cases"][0]["strategies"][0]["steps"][0]["selected_agents"] = ["node-b"]
    with pytest.raises(FragmentationError, match="does not match"):
        validate_recording(invalid)


def test_largest_first_control_is_required(placement_recording):
    invalid = deepcopy(placement_recording)
    invalid["cases"].pop()
    with pytest.raises(FragmentationError, match="control case"):
        validate_recording(invalid)


def test_cli_exits_nonzero_on_invalid_recording(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version": 999}')
    assert main(["--recording", str(path)]) == 1


def test_graphics_replay_four_recorded_inventory_states(placement_recording, tmp_path):
    outputs = render_recording(placement_recording, tmp_path)
    with Image.open(outputs["gif"]) as image:
        assert image.n_frames == 4
        assert image.size == (1280, 800)
    with Image.open(outputs["comparison"]) as image:
        assert image.size == (1280, 800)
    with Image.open(outputs["storyboard"]) as image:
        assert image.size == (1280, 3200)


def test_invalid_evidence_does_not_produce_graphics(placement_recording, tmp_path):
    invalid = deepcopy(placement_recording)
    invalid["cases"][0]["strategies"][1]["steps"][-1]["status"] = "placed"
    with pytest.raises(FragmentationError, match="fragmentation rejection"):
        render_recording(invalid, tmp_path)
    assert not list(tmp_path.iterdir())


def test_failed_replay_removes_old_figures_but_preserves_input_and_unrelated_files(
    placement_recording, tmp_path
):
    render_recording(placement_recording, tmp_path)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("unrelated")
    invalid = deepcopy(placement_recording)
    invalid["cases"].pop()
    source = tmp_path / "recording.json"
    source.write_text(json.dumps(invalid))
    assert main(["--recording", str(source), "--render-dir", str(tmp_path)]) == 1
    assert {path.name for path in tmp_path.iterdir()} == {"recording.json", "keep.txt"}
    assert unrelated.read_text() == "unrelated"
    assert json.loads(source.read_text()) == invalid


def test_manifest_binds_figures_to_saved_recording(placement_recording, tmp_path):
    source = tmp_path / "recording.json"
    source.write_text(json.dumps(placement_recording))
    outputs = render_recording(placement_recording, tmp_path, source_recording=source)
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["recording_file_sha256"] == sha256(source.read_bytes()).hexdigest()
    for key, artifact in manifest["artifacts"].items():
        assert artifact["sha256"] == sha256(outputs[key].read_bytes()).hexdigest()


def test_failed_component_rerun_invalidates_previous_recording_and_figures(
    placement_recording, tmp_path, monkeypatch
):
    from backend_ai_intro import fragmentation

    source = tmp_path / "recording.json"
    source.write_text(json.dumps(placement_recording))
    render_recording(placement_recording, tmp_path, source_recording=source)

    def fail():
        raise FragmentationError("required check failed")

    monkeypatch.setattr(fragmentation, "run_experiment", fail)
    assert main(["--output", str(source), "--render-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.iterdir())
