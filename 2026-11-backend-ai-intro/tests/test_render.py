"""The visual must be a replay of a valid recording, not a second simulation."""

from copy import deepcopy
from hashlib import sha256

import pytest
from PIL import Image, ImageChops

from backend_ai_intro.render import render_recording, validate_recording


def test_saved_events_become_readable_image_files(recording, tmp_path):
    outputs = render_recording(recording, tmp_path)

    assert set(outputs) == {"gif", "storyboard", "keyframe"}
    assert all(path.parent == tmp_path and path.is_file() for path in outputs.values())
    with Image.open(outputs["gif"]) as image:
        assert image.n_frames == len(recording["events"])
        assert image.width >= 1000
        for frame in range(image.n_frames):
            image.seek(frame)
            assert image.info["duration"] >= 100

    for key in ("storyboard", "keyframe"):
        with Image.open(outputs[key]) as image:
            assert image.format == "PNG"
            assert image.width >= 1000
            rgb = image.convert("RGB")
            background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
            assert ImageChops.difference(rgb, background).getbbox() is not None


def test_rendering_the_same_recording_is_deterministic(recording, tmp_path):
    first = render_recording(recording, tmp_path / "first")
    second = render_recording(recording, tmp_path / "second")

    assert {key: sha256(path.read_bytes()).hexdigest() for key, path in first.items()} == {
        key: sha256(path.read_bytes()).hexdigest() for key, path in second.items()
    }


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_unknown_recording_schema_is_rejected(recording, version):
    invalid = deepcopy(recording)
    invalid["schema_version"] = version
    with pytest.raises(ValueError):
        validate_recording(invalid)


def test_empty_recording_is_rejected(recording):
    invalid = deepcopy(recording)
    invalid["events"] = []
    with pytest.raises(ValueError):
        validate_recording(invalid)


def test_inconsistent_available_capacity_is_rejected(recording):
    invalid = deepcopy(recording)
    node = invalid["events"][0]["snapshot"]["node"]
    node["free"] = node["capacity"] + 1
    with pytest.raises(ValueError):
        validate_recording(invalid)


def test_negative_allocation_is_rejected(recording):
    invalid = deepcopy(recording)
    invalid["events"][0]["snapshot"]["node"]["allocations"][0]["slots"] = -1
    with pytest.raises(ValueError):
        validate_recording(invalid)


def test_unknown_session_reference_is_rejected(recording):
    invalid = deepcopy(recording)
    invalid["events"][0]["snapshot"]["node"]["allocations"][0]["session_id"] = "missing"
    with pytest.raises(ValueError):
        validate_recording(invalid)


def test_container_running_is_not_a_component_recording_status(recording):
    invalid = deepcopy(recording)
    invalid["events"][0]["snapshot"]["sessions"][0]["status"] = "running"
    with pytest.raises(ValueError):
        validate_recording(invalid)
