# Trace a looped and a regular Transformer

Follow one arithmetic carry, one moving coin and one locker clue through two pretrained models. Ouro applies the same 24 blocks four times. Qwen applies 28 different blocks once. The notebook reads the answer after every block, then compares how the readout changes when you swap a state, restrict access to the story, or cut an attention connection.

1. Download the [depth-trace notebook](loop-transformer-depth-trace.ipynb) and import it with **File → Import Notebook** in Kaggle.
2. Enable **Internet**, select **GPU T4 ×2**, and choose **Run All**. The notebook checks that Kaggle has enough free space and loads the pinned model weights on the worker.
3. Look for `Verified per-block traces` in the final cell. Download `/kaggle/working/depth-traces.json`. It contains 744 readouts across both models and both conditions, including the block visited, top token and answer margin.

On the full-story coin question, the top token at Ouro’s block 24 changes from L to B to B to L across the four visits. Explore the [interactive report](https://htmlpreview.github.io/?https://github.com/yongkyunlee/code-pointer-demo/blob/2e192997a45a779f2d2c3c2d3717a3948dec56bd/2026-11-loop-transformer/index.html) or download the [original 13-workload study notebook](https://github.com/yongkyunlee/code-pointer-demo/raw/f4a4a3d876c90a20c9d6c6b3a2466f34a7b108a3/2026-11-loop-transformer/loop-transformer-experiments.ipynb) to reproduce the full study.
