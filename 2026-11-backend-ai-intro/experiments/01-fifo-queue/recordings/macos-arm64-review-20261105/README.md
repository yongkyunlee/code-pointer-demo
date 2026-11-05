# FIFO review rerun, November 5, 2026

This macOS ARM64 rerun records 45 API samples from the real Backend.AI 26.9.7 manager, agent, storage proxy, and CPU containers. It passed the same-pass FIFO checks, session cleanup, infrastructure cleanup, and complete-run checks. This was the final FIFO run after the shared supervisor cleanup changes.

The recording, images, and manifests are copied unchanged from the run. The original `../macos-arm64/` reference remains unchanged. Elapsed times describe this run and are not a performance benchmark. These observations do not establish Linux-host behavior or physical GPU execution.

From `2026-11-backend-ai-intro/`, replay without Docker:

```sh
uv run --locked backend-ai-intro render experiments/01-fifo-queue/recordings/macos-arm64-review-20261105/recording.json --output /tmp/backend-ai-fifo-review-replay
```

`manifest.json` binds the recording and generated images with SHA-256 hashes. `run-status.json` records complete-run success; `environment-cleanup.json` records infrastructure cleanup.
