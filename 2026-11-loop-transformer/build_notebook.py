"""Build a self-contained Kaggle notebook from the adjacent experiment sources."""
import argparse
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent


def cell(kind, source):
    return {"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {})}


SETUP = """
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

work = Path("/kaggle/working")
temp = Path("/kaggle/temp")
src = temp / "loop-source"
lib = temp / "loop-py"
out = work / "results"
for directory in (src, lib, out):
    directory.mkdir(parents=True, exist_ok=True)
assert torch.cuda.device_count(), "Select a GPU accelerator in Kaggle Session options"
free = {str(p): round(shutil.disk_usage(p).free / 2**30, 2) for p in (temp, work)}
print("Free GiB", free, flush=True)
assert free[str(temp)] > 8 and free[str(work)] > 1, "Insufficient free storage"
env = dict(os.environ, HF_HOME=str(temp / "loop-hf"),
           HF_HUB_CACHE=str(temp / "loop-hf" / "hub"), HF_HUB_DISABLE_XET="1",
           PYTHONPATH=str(lib) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
           MPLBACKEND="Agg")
deps = ["transformers==4.55.0", "tokenizers==0.21.4", "huggingface-hub==0.36.0",
        "accelerate==1.10.1"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                "--no-deps", "--target", str(lib), *deps], check=True)
print("GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
"""


def build(path, validation):
    modules = (["model_engine", "runner"] if validation else
               ["model_engine", "runner", "cue", "memory", "carry", "paired", "verify_results"])
    cells = [cell("markdown", "# Loop Transformer experiments\n\nFour-pass Ouro-1.4B, a frozen Qwen2.5-1.5B comparison, and a controlled small-model study. Checkpoint weights and dependencies live in Kaggle temporary storage. The results directory contains the measurements and validation records."),
             cell("code", dedent(SETUP).strip())]
    for module in modules:
        name = module + ".py"
        contents = (ROOT / name).read_text()
        cells.append(cell("code", f'%%writefile /kaggle/temp/loop-source/{name}\n' + contents))
    pilots = [[model, "--phases", "pilot"] for model in ("ouro", "qwen")]
    remaining = [["ouro", "--phases", "heldout", "--tasks", task] for task in ("carry", "memory", "cue")]
    remaining += [["qwen", "--phases", "heldout", "--tasks", task] for task in ("carry", "memory", "cue")]
    waves = ([[["ouro", "--validate-only"], ["qwen", "--validate-only"]]] if validation
             else [[*pilots, ["paired"]], remaining])
    launch = f"waves = {waves!r}\n" + dedent("""
        import concurrent.futures
        import hashlib
        import json
        import zipfile
        import time
        import queue

        def archive_results(filename):
            path = work / filename
            tmp = path.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
                for p in sorted(out.rglob("*")):
                    if p.is_file() and p.suffix != ".tmp":
                        z.write(p, p.relative_to(work))
                for p in sorted(src.glob("*.py")):
                    z.write(p, "source/" + p.name)
            tmp.replace(path)
            return path

        # A shorter workload may use the next available card while the other continues.
        slots = queue.Queue()
        for card in range(torch.cuda.device_count()):
            slots.put(card)
        def worker(args):
            card = slots.get()
            name = "-".join(arg for arg in args if not arg.startswith("--"))
            try:
                if "heldout" in args and not (out / f"{args[0]}-{args[-1]}-pilot.receipt.json").is_file():
                    print(f"Skipped {name}: pilot failed", flush=True)
                    return name, 1
                filename = out / f"{name}.log"
                with filename.open("w", buffering=1) as log:
                    p = subprocess.Popen([sys.executable, "-u", str(src / "runner.py"),
                                          *args, "--out", str(out)], env=dict(env, CUDA_VISIBLE_DEVICES=str(card)),
                                         cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    reported = 0.
                    for line in p.stdout:
                        log.write(line)
                        event = any(k in line for k in ["BEGIN ", "COMPLETE ", "Native parity", "Error", "Exception", "Traceback"])
                        if event or time.monotonic() - reported > 90:
                            print(f"[{name}] {line.rstrip()[:650]}", flush=True)
                            reported = time.monotonic()
                    code = p.wait()
                print(f"{name} exit {code}", flush=True)
                return name, code
            finally:
                slots.put(card)
        with concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
            outcomes = {}
            for jobs in waves:
                pending = {pool.submit(worker, job) for job in jobs}
                while pending:
                    done, pending = concurrent.futures.wait(pending, timeout=300,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    outcomes.update(f.result() for f in done)
                    if pending or len(waves) > 1:
                        archive_results("loop-progress.zip")
        print(outcomes)
        archive = archive_results("loop-results.zip")
        print("Result archive", archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest())
        assert not any(outcomes.values()), outcomes
    """).strip()
    cells.append(cell("code", launch))
    if not validation:
        cells.append(cell("code", dedent("""
            import json
            verification = subprocess.run([sys.executable, str(src / "verify_results.py"), str(out),
                                           "--source", str(src), "--json", str(out / "audit.json")],
                                          cwd=src, capture_output=True, text=True, env=env)
            print(verification.stdout[-12000:], verification.stderr[-4000:], flush=True)
            final_archive = archive_results("loop-results.zip")
            print("Verified output bundle", final_archive.stat().st_size,
                  hashlib.sha256(final_archive.read_bytes()).hexdigest(), flush=True)
            verification.check_returncode()
        """).strip()))
    for i, content in enumerate(cells):
        content["id"] = f"loop-study-{i:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}}}
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    print(build(args.path, args.validation))
