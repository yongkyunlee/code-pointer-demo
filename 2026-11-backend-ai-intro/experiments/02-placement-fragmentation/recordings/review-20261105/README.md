# Placement review rerun, November 5, 2026

This component rerun executes the installed Backend.AI 26.9.7 placement code against synthetic GPU inventories. All strategy/order controls passed. It does not start containers or execute GPU workloads.

The recording, images, and manifest are copied unchanged from the run. The original files in the parent `recordings/` directory remain unchanged.

From `2026-11-backend-ai-intro/`, replay the evidence:

```sh
uv run --locked python experiments/02-placement-fragmentation/run.py --recording experiments/02-placement-fragmentation/recordings/review-20261105/recording.json --render-dir /tmp/backend-ai-placement-review-replay
```

`manifest.json` binds the recording and generated images with SHA-256 hashes. The recording also identifies its harness, dependency lock, and loaded placement modules.
