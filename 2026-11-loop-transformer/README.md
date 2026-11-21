# Loop Transformer experiments

This reader report follows an arithmetic carry through a looped Transformer and compares its behavior on memory and repeated-clue tasks. It includes charts and an explorer for the words produced in 400 recorded checkpoint, language and sum combinations. The saved Kaggle run passed all 13 workload checks. [Download the standalone HTML](https://github.com/yongkyunlee/code-pointer-demo/raw/refs/heads/main/2026-11-loop-transformer/index.html)

## Reproduce the Kaggle run

1. Create a fresh Kaggle notebook. Use **File → Import Notebook** to import [loop-transformer-experiments.ipynb](loop-transformer-experiments.ipynb).
2. Enable **Internet** and select **GPU T4 ×2** under **Session options**. Choose **Run All**. The notebook downloads both checkpoints to Kaggle temporary storage and checks for more than 8 GiB free there and more than 1 GiB in the output filesystem.
3. Wait for the last cell to print `{"verified": 13}` and `Verified output bundle`. Download `/kaggle/working/loop-results.zip`. It contains the raw measurements, figures, GPU timings, source hashes and executed Python files. The published copy is [evidence/kaggle-results.zip](evidence/kaggle-results.zip).

The recorded run used two Tesla T4 GPUs, PyTorch 2.11.0+cu130 and Transformers 4.55.0. The checkpoint weights total 5.55 GiB. The longest workload, Ouro’s held-out carry experiment, took 59 minutes. The final run reused the two previously measured memory pilots. A fresh notebook runs those pilots as well.

## Verify the saved measurements locally

Run these commands from this demo directory with Python 3.11 or newer. The audit uses the standard library. It checks the source files against the GPU receipts, the held-out families against the pilots, and the recorded measurements against the raw files. The archive expands to about 31 MiB.

```sh
run_dir=$(mktemp -d)
python3 -m zipfile -e evidence/kaggle-results.zip "$run_dir"
python3 verify_results.py "$run_dir/results" --source . --json "$run_dir/audit.json"
cmp "$run_dir/audit.json" evidence/audit.json
```

The expected archive SHA-256 is `9a26ba409eab9ccd02fedef298cbdb478875c7ca05e92beae3e57555c52c33ad`. The audit should report 13 verified workloads. [Saved audit](evidence/audit.json)

To recompute the paired checkpoint differences and bootstrap intervals, use a Python environment with NumPy and PyTorch. The local replay used Python 3.12, NumPy 2.5.5 and CPU PyTorch 2.8.0. In the same shell, run the following commands with that environment active.

```sh
python3 compare_results.py "$run_dir/results" --source . \
  --json "$run_dir/checkpoint-comparisons.json"
cmp "$run_dir/checkpoint-comparisons.json" evidence/checkpoint-comparisons.json
```

It reports six comparisons, one for each task and phase. The held-out sets contain 100 arithmetic families, 120 memory families and 100 repeated-clue families per checkpoint. All families contribute to the main estimates. None met the strict shared clean-success rule on any held-out task. [Saved checkpoint comparisons](evidence/checkpoint-comparisons.json)

## Models and controls

- **Ouro-1.4B** executes 24 shared blocks four times with frozen FP16 weights and eager attention. [Checkpoint `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)
- **Qwen2.5-1.5B** executes 28 distinct blocks in four groups of seven. It uses frozen FP16 weights and math SDPA. For both models, the adapter matched the native logits and all four stage boundaries exactly in the Kaggle validation. [Checkpoint `8faed761d45a263340a0528343f099c05c9a4323`](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
- **Small-model training control** compares a tied stack, an untied stack of the same width and a narrower untied stack with a similar parameter count. It uses three seeds and matches the token order within each seed. Its results apply to these synthetic tasks and three trained seeds. The frozen checkpoints also differ in tokenizer, training and executed depth. [Training and scoring code](paired.py)

If you edit the experiment modules, rebuild the self-contained notebook before importing it into Kaggle. To generate a short notebook that checks only native-forward parity for the two checkpoints, add `--validation` and choose a separate output name.

```sh
python3 build_notebook.py loop-transformer-experiments.ipynb
python3 build_notebook.py loop-transformer-validation.ipynb --validation
```

## Rebuild the reader report

Open `index.html` directly to use the figures and swap explorer offline. The explorer shows the stage-three answer-position swaps for all 100 held-out arithmetic families, both checkpoints, and both recipient languages. Every displayed response comes from the saved Kaggle archive.

The HTML builder uses Matplotlib (3.11.2 in the published build) and the Python standard library. It embeds the figures and the selected measurement fields in a single file.

```sh
python3 report/build.py
```

The original [memory qualification archive](evidence/memory-qualification.zip) preserves the pilot measurements that the final run reused.
