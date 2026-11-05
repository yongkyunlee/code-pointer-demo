"""Draw placement observations from saved JSON without running a scheduler."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from backend_ai_intro.fragmentation import validate_recording

WIDTH, HEIGHT, SCALE = 1280, 800, 2
INK = "#192735"
MUTED = "#536373"
LINE = "#d9e2e8"
FREE = ("#edf5f0", "#679276")
REQUEST_COLORS = ("#547cb9", "#9466bd", "#d08a32")
FILENAMES = {
    "gif": "placement.gif",
    "comparison": "comparison.png",
    "storyboard": "storyboard.png",
    "manifest": "manifest.json",
}


def invalidate_outputs(output_dir: Path, *, preserve: Path | None = None) -> None:
    """Remove only this renderer's artifacts so failed attempts cannot reuse old figures."""
    targets = [output_dir / filename for filename in FILENAMES.values()]
    if preserve is not None and preserve.resolve() in {path.resolve() for path in targets}:
        raise ValueError("The input recording must not use a reserved graphics filename")
    for path in targets:
        path.unlink(missing_ok=True)


@lru_cache(maxsize=12)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size * SCALE)


class _Canvas:
    def __init__(self) -> None:
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "#ffffff")
        self.draw = ImageDraw.Draw(self.image)

    def text(self, x: int, y: int, value: str, size: int = 24, color: str = INK) -> None:
        self.draw.text((x * SCALE, y * SCALE), value, fill=color, font=_font(size), anchor="lt")

    def box(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        fill: str,
        border: str | None = None,
    ) -> None:
        self.draw.rounded_rectangle(
            (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE),
            radius=12 * SCALE,
            fill=fill,
            outline=border,
            width=2 * SCALE,
        )

    def finish(self) -> Image.Image:
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def _frame(recording: dict[str, Any], step_count: int) -> Image.Image:
    canvas = _Canvas()
    case = recording["cases"][0]
    canvas.text(42, 28, "Placement fragmentation", size=38)
    canvas.text(
        42, 79, "Backend.AI 26.9.7 | real placement code | two synthetic four-GPU nodes", 22, MUTED
    )
    title = (
        "Initial inventory: 8 free GPU slots"
        if step_count == 0
        else f"Request {step_count}: {case['request_order'][step_count - 1]} GPU slots on one node"
    )
    canvas.box(42, 125, WIDTH - 84, 65, "#f3f6f9")
    canvas.text(64, 144, title, 28)

    for column, run in enumerate(case["strategies"]):
        x = 42 + column * 620
        canvas.text(x, 216, run["strategy"].upper(), 27)
        assigned: dict[str, list[int]] = {name: [] for name in case["node_input_order"]}
        steps = run["steps"][:step_count]
        # These colors label accepted requests from JSON. They are not device IDs.
        for index, step in enumerate(steps):
            if step["status"] == "placed":
                assigned[step["selected_agents"][0]].extend([index] * step["requested_gpu_slots"])
        free = steps[-1]["free_after"] if steps else dict.fromkeys(assigned, 4)
        for row, (node, colors) in enumerate(assigned.items()):
            y = 267 + row * 149
            canvas.box(x, y, 574, 130, "#fafbfd", LINE)
            canvas.text(x + 18, y + 15, node, 23)
            canvas.text(x + 385, y + 16, f"{free[node]} / 4 free", 22, MUTED)
            for slot in range(4):
                left = x + 19 + slot * 136
                if slot < len(colors):
                    request = colors[slot]
                    canvas.box(left, y + 54, 126, 56, REQUEST_COLORS[request])
                    canvas.text(left + 37, y + 72, f"R{request + 1}", 23, "#ffffff")
                else:
                    canvas.box(left, y + 54, 126, 56, FREE[0], FREE[1])
                    canvas.text(left + 32, y + 72, "Free", 22, "#42654c")

        accepted = sum(step["status"] == "placed" for step in steps)
        canvas.text(x, 578, f"Accepted requests: {accepted} / {step_count}", 23, MUTED)
        if not steps:
            result, detail, background, foreground = (
                "Ready for the same requests",
                "Order: 1, then 3, then 4 GPU slots",
                "#f3f6f9",
                INK,
            )
        elif steps[-1]["status"] == "cannot_place":
            result, detail, background, foreground = (
                "4 free slots; request cannot fit",
                "Free capacity is split 3 + 1 across nodes.",
                "#fff0ec",
                "#a1482b",
            )
        else:
            result, detail, background, foreground = (
                f"Request {step_count} accepted",
                (
                    "All three requests fit in the same inventory."
                    if step_count == 3
                    else (
                        f"Selected {steps[-1]['selected_agents'][0]}; "
                        f"{sum(free.values())} slots remain free."
                    )
                ),
                "#edf6f0",
                "#326044",
            )
        canvas.box(x, 617, 574, 88, background)
        canvas.text(x + 16, 631, result, 25, foreground)
        canvas.text(x + 16, 669, detail, 20, foreground)

    canvas.text(
        42,
        737,
        "Slots show placement accounting, not GPU activity. Blocks are not physical device IDs.",
        20,
        MUTED,
    )
    canvas.text(
        42, 768, "No containers or GPU jobs ran. GIF playback timing is for readability.", 18, MUTED
    )
    return canvas.finish()


def render_recording(
    recording: dict[str, Any], output_dir: Path, *, source_recording: Path | None = None
) -> dict[str, Path]:
    """Write a four-frame replay and PNGs after validating the saved evidence."""
    invalidate_outputs(output_dir, preserve=source_recording)
    validate_recording(recording)
    if source_recording is not None and json.loads(source_recording.read_text()) != recording:
        raise ValueError("The source recording does not match the observations being rendered")
    frames = [_frame(recording, count) for count in range(4)]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in FILENAMES.items()}
    with TemporaryDirectory(prefix=".render-", dir=output_dir) as temporary:
        staged = {key: Path(temporary) / name for key, name in FILENAMES.items()}
        frames[-1].save(staged["comparison"])
        frames[0].save(
            staged["gif"],
            save_all=True,
            append_images=frames[1:],
            duration=[2200, 2200, 2800, 5000],
            loop=0,
            disposal=2,
        )
        storyboard = Image.new("RGB", (WIDTH, HEIGHT * len(frames)), "white")
        for index, frame in enumerate(frames):
            storyboard.paste(frame, (0, index * HEIGHT))
        storyboard.save(staged["storyboard"])
        canonical = json.dumps(recording, sort_keys=True, separators=(",", ":")).encode()
        manifest = {
            "schema_version": 1,
            "recording_canonical_json_sha256": hashlib.sha256(canonical).hexdigest(),
            "canonical_json_format": "UTF-8 json.dumps(sort_keys=True, separators=(',', ':'))",
            "recording_file_sha256": (
                hashlib.sha256(source_recording.read_bytes()).hexdigest()
                if source_recording is not None
                else None
            ),
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifacts": {
                key: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in staged.items()
                if key != "manifest"
            },
        }
        staged["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            for key, path in staged.items():
                path.replace(paths[key])
        except OSError:
            invalidate_outputs(output_dir, preserve=source_recording)
            raise
    return paths
