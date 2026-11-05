"""Run the original live FIFO experiment using the shared pinned installation."""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
if "--output" not in sys.argv:
    sys.argv.extend(["--output", str(HERE / "artifacts/live")])
runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
