"""One isolated GPU worker per checkpoint. Launch via the generated notebook."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch


def save(path, obj):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices=["ouro", "qwen", "paired"])
    parser.add_argument("--phases", nargs="+", default=["pilot", "heldout"],
                        choices=["pilot", "heldout"])
    parser.add_argument("--tasks", nargs="+", default=["cue", "memory", "carry"],
                        choices=["cue", "memory", "carry"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    base = Path(__file__).resolve().parent
    if not torch.cuda.is_available() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this worker in a Kaggle GPU notebook")
    metadata = {
        "platform": "kaggle", "engine": args.worker, "gpu": torch.cuda.get_device_name(),
        "gpu_capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda": torch.version.cuda, "torch": str(torch.__version__),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(base.glob("*.py"))},
    }
    failed = []

    def measure(label, experiment, phase, operation):
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        print(f"BEGIN {label}", flush=True)
        try:
            operation()
            torch.cuda.synchronize()
        except Exception:
            error = traceback.format_exc()
            save(out / f"{label}.failure.json", {**metadata, "status": "failed", "error": error})
            print(error, flush=True)
            failed.append(label)
            return False
        save(out / f"{label}.receipt.json", {
            **metadata, "status": "complete", "experiment": experiment, "phase": phase,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        print(f"COMPLETE {label} in {time.perf_counter() - start:.1f}s", flush=True)
        return True

    if args.worker == "paired":
        import paired
        measure("paired-training-control", "training", "control", lambda: paired.run(out))
    else:
        from model_engine import Engine
        engine = Engine.load(args.worker)
        diag = engine.validate()
        save(out / f"{args.worker}-native-validation.json", {**metadata, **diag})
        print(f"Native parity {args.worker}: {diag}", flush=True)
        if args.validate_only:
            return
        passed = {}
        for phase in args.phases:
            for name in args.tasks:
                if phase == "heldout" and passed.get(name) is False:
                    continue
                def run_task():
                    importlib.import_module(name).run(engine, phase, out)
                passed[name] = measure(f"{args.worker}-{name}-{phase}", name, phase, run_task)
    if failed:
        sys.exit(f"Failed workloads: {failed}")


if __name__ == "__main__":
    main()
