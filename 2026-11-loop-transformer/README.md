# Loop Transformer experiments

Follow a single question through a looped Transformer and a regular Transformer. The interactive report maps each visit to the block that ran. Select a square to see which token the output head favored there, inspect the effect of the intervention, or compare all four visits to the same Ouro weights. The report includes three tasks, the frozen Ouro-1.4B and Qwen2.5-1.5B checkpoints, and a separate comparison of small models trained at matched width and depth. [Open the interactive report](https://htmlpreview.github.io/?https://github.com/yongkyunlee/code-pointer-demo/blob/main/2026-11-loop-transformer/index.html) · [Download the standalone HTML](https://github.com/yongkyunlee/code-pointer-demo/raw/refs/heads/main/2026-11-loop-transformer/index.html)

The [experiment lab](https://htmlpreview.github.io/?https://github.com/yongkyunlee/code-pointer-demo/blob/main/2026-11-loop-transformer/experiment-lab.html) is a shorter companion for first-time readers. Each experiment asks for a prediction, shows the intervention on both architectures block by block, and then reveals the recorded answers beside the matched small-model result.

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

## Reproduce the per-block trace

Import [the depth-trace notebook](loop-transformer-depth-trace.ipynb) into a new Kaggle notebook with internet access and select GPU T4 ×2. Run all cells. The last cell combines the two checkpoint runs into `/kaggle/working/depth-traces.json`. [Published measurements](evidence/depth-traces.json)

The notebook replays one held-out example per task under two conditions. After each decoder block it projects the hidden state at the final question token through the checkpoint’s own final normalization and language-model head. It records the unrestricted top token and the difference between candidate logits. The carry trace compares “eight” with “seven”, the coin trace compares the actual box B with the stronger of A and L, and the locker trace compares “apple” with “orange”. Each curve follows a single prompt. The early readouts describe what the final head can read from that state. The separate swap and attention-cut experiments test how changing a state affects the eventual output.

The runner checks native forward parity, the final projected logits, and 16 previously archived margins across both checkpoints. Parameter sharing determines the physical block column, while execution depth counts each visit. To rebuild the trace notebook after editing its source, use the following command.

```sh
python3 build_notebook.py loop-transformer-depth-trace.ipynb --depth-trace
```

## Rebuild the reader report

Open `index.html` directly to use the figures and swap explorer offline. The explorer shows the stage-three answer-position swaps for all 100 held-out arithmetic families, both checkpoints, and both recipient languages. The coin walkthrough uses held-out scene 024, and the locker walkthrough uses scene 035. Each shows the recorded model answers for that example beside the aggregate results.

The HTML builder uses Matplotlib (3.11.2 in the published build) and the Python standard library. It embeds the figures and the selected measurement fields in a single file.

```sh
python3 report/build.py
```

The experiment lab has its own builder, which uses the Python standard library and reuses the report's scene extractors. It checks the embedded values against the archive and writes only `experiment-lab.html`.

```sh
python3 report/build_experiment_lab.py
```

The original [memory qualification archive](evidence/memory-qualification.zip) preserves the pilot measurements that the final run reused.
