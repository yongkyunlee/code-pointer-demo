#!/usr/bin/env python3
"""Run or replay the independent placement-fragmentation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from backend_ai_intro.fragmentation import main  # noqa: E402

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent / "artifacts" / "latest"
    defaults = ["--output", str(directory / "recording.json"), "--render-dir", str(directory)]
    raise SystemExit(main(defaults + sys.argv[1:]))
