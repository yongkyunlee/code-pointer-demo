"""Run the original live FIFO experiment using the shared pinned installation."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    sys.path[:0] = [str(root), str(root / "src")]
    arguments = sys.argv[1:]
    if not any(arg == "--output" or arg.startswith("--output=") for arg in arguments):
        arguments.extend(["--output", str(here / "artifacts/live")])
    sys.argv = [str(root / "run.py"), *arguments]
    runpy.run_path(str(root / "run.py"), run_name="__main__")
