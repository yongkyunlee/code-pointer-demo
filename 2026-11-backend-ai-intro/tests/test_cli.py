"""Failed reruns must not retain a previous run's successful visual evidence."""

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from backend_ai_intro.cli import clear_generated_artifacts, replay, save_artifacts

DERIVED_NAMES = ("scheduler.gif", "storyboard.png", "keyframe.png", "manifest.json")


def stale_artifacts(output):
    for name in DERIVED_NAMES:
        (output / name).write_text("previous successful run")


def test_failed_save_keeps_new_evidence_without_old_success_artifacts(recording, tmp_path):
    stale_artifacts(tmp_path)
    invalid = deepcopy(recording)
    invalid["events"] = []
    invalid["verification"]["passed"] = False

    with pytest.raises(ValueError, match="events"):
        save_artifacts(invalid, tmp_path)

    assert json.loads((tmp_path / "recording.json").read_text()) == invalid
    assert not any((tmp_path / name).exists() for name in DERIVED_NAMES)


@pytest.mark.parametrize("contents", ['{"schema_version": 999}', "not JSON", None])
def test_failed_replay_clears_old_artifacts_and_preserves_input(tmp_path, contents):
    stale_artifacts(tmp_path)
    path = tmp_path / "recording.json"
    if contents is not None:
        path.write_text(contents)

    with pytest.raises((ValueError, FileNotFoundError)):
        replay(path, tmp_path)

    assert not any((tmp_path / name).exists() for name in DERIVED_NAMES)
    if contents is not None:
        assert path.read_text() == contents


def test_cleanup_removes_only_known_outputs_and_can_preserve_recording(tmp_path):
    stale_artifacts(tmp_path)
    for name in ("recording.json", ".recording.json.tmp", "recording.json.tmp"):
        (tmp_path / name).write_text("old recording")
    (tmp_path / ".scheduler.gif.tmp").write_text("interrupted render")
    (tmp_path / "notes.txt").write_text("keep this")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "recording.json").write_text("unrelated")

    clear_generated_artifacts(tmp_path)
    assert (tmp_path / "recording.json").read_text() == "old recording"
    assert not (tmp_path / ".scheduler.gif.tmp").exists()
    clear_generated_artifacts(tmp_path, include_recording=True)

    assert not any((tmp_path / name).exists() for name in DERIVED_NAMES)
    assert not (tmp_path / "recording.json").exists()
    assert not (tmp_path / ".recording.json.tmp").exists()
    assert not (tmp_path / "recording.json.tmp").exists()
    assert (tmp_path / "notes.txt").read_text() == "keep this"
    assert (tmp_path / "other" / "recording.json").read_text() == "unrelated"


def test_successful_save_replaces_manifest_with_matching_hashes(recording, tmp_path):
    stale_artifacts(tmp_path)
    outputs = save_artifacts(recording, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["recording_sha256"]
        == hashlib.sha256((tmp_path / "recording.json").read_bytes()).hexdigest()
    )
    assert manifest["artifacts"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs.values()
    }


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_manifest_failure_removes_new_unbound_visuals(recording, tmp_path, monkeypatch, error_type):
    path = tmp_path / "recording.json"
    original = json.dumps(recording)
    path.write_text(original)
    write_text = Path.write_text

    def fail_manifest(path, content, *args, **kwargs):
        if path.name == "manifest.json":
            assert all((tmp_path / name).is_file() for name in DERIVED_NAMES[:3])
            write_text(path, "partial manifest")
            raise error_type("injected manifest failure")
        return write_text(path, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    with pytest.raises(error_type, match="manifest failure"):
        replay(path, tmp_path)

    assert path.read_text() == original
    assert not any((tmp_path / name).exists() for name in DERIVED_NAMES)


def test_failed_component_attempt_does_not_keep_previous_recording(tmp_path, monkeypatch):
    from backend_ai_intro import cli, experiment

    stale_artifacts(tmp_path)
    (tmp_path / "recording.json").write_text('{"verification": {"passed": true}}')
    sidecars = {"run-status.json", "environment-cleanup.json"}
    sidecars |= {f".{name}.tmp" for name in sidecars} | {f"{name}.tmp" for name in sidecars}
    for name in sidecars:
        (tmp_path / name).write_text('{"status": "passed"}')

    def fail():
        raise RuntimeError("injected scheduling failure")

    monkeypatch.setattr(experiment, "run_experiment", fail)
    monkeypatch.setattr(sys, "argv", ["backend-ai-intro", "component", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="scheduling failure"):
        cli.main()

    assert not (tmp_path / "recording.json").exists()
    assert not any((tmp_path / name).exists() for name in DERIVED_NAMES)
    assert not any((tmp_path / name).exists() for name in sidecars)
