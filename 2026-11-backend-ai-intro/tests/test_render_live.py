"""Small synthetic API fixtures test the renderer; these are not recorded runs."""

from copy import deepcopy

import pytest
from PIL import Image

from backend_ai_intro import render


@pytest.fixture
def live_recording():
    samples = []
    statuses = [
        ("RUNNING", "PENDING", "PENDING"),
        ("RUNNING", "PENDING", "PENDING"),
        ("TERMINATED", "RUNNING", "PENDING"),
        ("TERMINATED", "TERMINATED", "RUNNING"),
        ("TERMINATED", "TERMINATED", "TERMINATED"),
    ]
    for sequence, (state, occupied) in enumerate(zip(statuses, [2, 2, 4, 1, 0], strict=True)):
        samples.append(
            {
                "sequence": sequence,
                "elapsed_seconds": sequence * 10.0,
                "sessions": [
                    {
                        "id": name,
                        "label": name.title(),
                        "requested_cpu": request,
                        "status": status,
                        "occupied_slots": {"cpu": str(request if status == "RUNNING" else 0)},
                    }
                    for name, request, status in zip(
                        ("current", "alice", "bob"), (2, 4, 1), state, strict=True
                    )
                ],
                "agent": {
                    "available_slots": {"cpu": "4"},
                    "occupied_slots": {"cpu": str(occupied)},
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment": {"mode": "live_cpu"},
        "samples": samples,
        "checkpoints": [
            {
                "label": f"Synthetic test checkpoint {sequence}",
                "sample_sequence": sequence,
                "decision_summary": "Synthetic test data, not an observed experiment.",
            }
            for sequence in range(5)
        ],
    }


def test_live_renderer_and_dispatch_keep_one_frame_per_checkpoint(live_recording, tmp_path):
    direct = render.render_live_recording(live_recording, tmp_path / "direct")
    dispatched = render.render_recording(live_recording, tmp_path / "dispatch")
    assert set(direct) == set(dispatched) == {"gif", "storyboard", "keyframe"}
    for key in direct:
        assert direct[key].read_bytes() == dispatched[key].read_bytes()
    with Image.open(dispatched["gif"]) as image:
        assert image.n_frames == len(live_recording["checkpoints"]) == 5


def test_actual_api_statuses_are_drawn_without_conversion(live_recording, tmp_path, monkeypatch):
    live_recording["samples"][1]["sessions"][1]["status"] = "PREPARING"
    before = deepcopy(live_recording)
    labels = []
    original = render._Canvas.wrapped

    def capture(canvas, x, y, value, width, **kwargs):
        labels.append(value)
        return original(canvas, x, y, value, width, **kwargs)

    monkeypatch.setattr(render._Canvas, "wrapped", capture)
    render.render_live_recording(live_recording, tmp_path)
    assert "PREPARING" in labels
    assert "RUNNING" in labels
    assert "TERMINATED" in labels
    assert live_recording == before


def test_checkpoint_must_reference_an_existing_observation(live_recording):
    live_recording["checkpoints"][-1]["sample_sequence"] = 999
    with pytest.raises(ValueError, match="reference"):
        render.validate_recording(live_recording)


def test_repeated_checkpoint_reference_is_rejected(live_recording):
    live_recording["checkpoints"][1]["sample_sequence"] = 0
    with pytest.raises(ValueError, match="reference"):
        render.validate_recording(live_recording)


def test_separate_api_reads_do_not_need_identical_occupancy(live_recording):
    # Session occupancy was read before the manager released current's allocation.
    # Agent occupancy was read after that release. Neither observation is rewritten.
    sample = live_recording["samples"][0]
    sample["agent"]["occupied_slots"]["cpu"] = "0"
    assert sample["sessions"][0]["occupied_slots"]["cpu"] == "2"
    render.validate_recording(live_recording)


def test_available_slots_is_total_capacity_not_remaining_capacity(live_recording):
    capacity, allocated = render._live_capacity(live_recording["samples"][0])
    assert capacity == 4
    assert allocated == 2
    assert capacity - allocated == 2


def test_agent_used_counters_are_not_labeled_total_allocations(
    live_recording, tmp_path, monkeypatch
):
    labels = []
    original = render._Canvas.text

    def capture(canvas, x, y, value, *args, **kwargs):
        labels.append(value)
        return original(canvas, x, y, value, *args, **kwargs)

    monkeypatch.setattr(render._Canvas, "text", capture)
    render.render_live_recording(live_recording, tmp_path)
    assert "AGENT-REPORTED USED CPU" in labels
    assert "Capacity minus used = 2 CPU slots" in labels
    assert not any(" free" in label.lower() or " allocated" in label.lower() for label in labels)


@pytest.mark.parametrize("elapsed", [float("inf"), float("nan"), -1, True])
def test_invalid_elapsed_observation_is_rejected(live_recording, elapsed):
    live_recording["samples"][0]["elapsed_seconds"] = elapsed
    with pytest.raises(ValueError, match="elapsed_seconds"):
        render.validate_recording(live_recording)
