"""Draw idle-policy checkpoints from recorded API and Prometheus observations."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .render import INK, MUTED, PALETTE, _Canvas, _live_checkpoints


def render_idle_recording(recording: Mapping[str, Any], output: Path) -> dict[str, Path]:
    if not recording.get("verification", {}).get("passed"):
        raise ValueError("Only a verified idle recording can be rendered")
    if not recording.get("cleanup", {}).get("complete"):
        raise ValueError("Idle experiment cleanup must be complete")
    selected = _live_checkpoints(recording)
    frames = []
    for index, (checkpoint, sample) in enumerate(selected):
        canvas = _Canvas(1200, 720)
        canvas.text(40, 30, "Automatic idle-session reclamation", 30)
        canvas.text(40, 78, "Backend.AI 26.9.7 | Recorded CPU sessions and activity", 19, MUTED)
        canvas.line(40, 116, 1160)
        canvas.text(40, 138, f"{index + 1}. {checkpoint['label']}", 25)
        canvas.text(1160, 148, f"t = {sample['elapsed_seconds']:.1f}s", 19, MUTED, "ra")
        canvas.text(40, 197, "SESSION", 16, MUTED)
        canvas.text(420, 197, "STATE", 16, MUTED)
        canvas.text(650, 197, "RUNNING CPU SLOTS", 16, MUTED)
        canvas.text(1160, 197, "CPU RATE", 16, MUTED, "ra")
        for row, session in enumerate(sample["sessions"]):
            y = 237 + row * 62
            color, border = PALETTE[row % len(PALETTE)]
            canvas.text(40, y + 7, session["label"], 21)
            canvas.text(420, y + 9, session["status"], 18, MUTED)
            # Session occupancy can fall back to requested resources before
            # startup. Filled blocks require an observed RUNNING session.
            allocated = (
                int(float(session["occupied_slots"].get("cpu", "0")))
                if session["status"] == "RUNNING"
                else 0
            )
            for slot in range(4):
                canvas.rect(
                    (650 + slot * 49, y, 692 + slot * 49, y + 40),
                    color if slot < allocated else "#f2f4f6",
                    border if slot < allocated else "#d9dfe6",
                    radius=6,
                )
            metric = sample.get("cpu_utilization_millicores", {}).get(session["id"])
            if session["status"] == "PENDING":
                canvas.text(857, y + 10, f"asks {session['requested_cpu']}", 16, MUTED)
            activity = "unavailable" if metric is None else f"{float(metric):.1f} mCPU"
            if session["status"] != "RUNNING":
                activity = "not running"
            canvas.text(1160, y + 9, activity, 19, MUTED, "ra")
        canvas.rect((40, 505, 1160, 614), "#f1f6fa", radius=10)
        canvas.wrapped(
            60,
            526,
            checkpoint["decision_summary"],
            1080,
            size=21,
            lines=3,
            leading=28,
        )
        used = float(sample["agent"]["occupied_slots"].get("cpu", "0"))
        canvas.text(
            40,
            638,
            f"Agent reports {used:g} / 4 CPU slots used. Pending requests are unfilled.",
            18,
            INK,
        )
        canvas.text(
            40,
            670,
            "CPU rate uses a five-minute window (available history during startup). "
            "1000 mCPU = one core.",
            17,
            MUTED,
        )
        frames.append(canvas.finish())
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "gif": output / "scheduler.gif",
        "storyboard": output / "storyboard.png",
        "keyframe": output / "keyframe.png",
    }
    temporary = {name: path.with_name(f".{path.name}.tmp") for name, path in paths.items()}
    try:
        frames[0].save(
            temporary["gif"],
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=4500,
            loop=0,
        )
        frames[min(1, len(frames) - 1)].save(temporary["keyframe"], format="PNG")
        rows = (len(frames) + 1) // 2
        board = Image.new("RGB", (1200, rows * 360), "white")
        for index, frame in enumerate(frames):
            board.paste(
                frame.resize((600, 360), Image.Resampling.LANCZOS),
                ((index % 2) * 600, (index // 2) * 360),
            )
        board.save(temporary["storyboard"], format="PNG")
        for name, path in paths.items():
            temporary[name].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths
