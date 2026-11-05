"""Run the component experiment or replay a saved recording."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def save_artifacts(recording: dict, output: Path) -> dict[str, Path]:
    """Save the evidence first, then draw only from that saved JSON."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / "recording.json"
    temporary = output / ".recording.json.tmp"
    temporary.write_text(json.dumps(recording, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return replay(path, output)


def replay(recording_path: Path, output: Path) -> dict[str, Path]:
    """Render a recording and bind the generated files to its content hash."""
    from .render import render_recording

    raw = recording_path.read_bytes()
    recording = json.loads(raw)
    paths = render_recording(recording, output)
    manifest = {
        "schema_version": 1,
        "recording_sha256": hashlib.sha256(raw).hexdigest(),
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    component = commands.add_parser("component", help="Run the real scheduler on synthetic inputs")
    component.add_argument("--output", type=Path, default=Path("artifacts/component"))
    render = commands.add_parser("render", help="Replay saved evidence without starting services")
    render.add_argument("recording", type=Path)
    render.add_argument("--output", type=Path, default=Path("artifacts/replay"))
    args = parser.parse_args()
    if args.command == "component":
        from .experiment import run_experiment

        paths = save_artifacts(run_experiment(), args.output)
    else:
        paths = replay(args.recording, args.output)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
