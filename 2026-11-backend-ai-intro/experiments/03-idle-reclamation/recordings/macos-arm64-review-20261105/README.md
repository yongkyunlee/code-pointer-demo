# Idle reclamation review rerun, November 5, 2026

This macOS ARM64 rerun records 248 API samples. After the 75-second policy-off control, waiting work was observed running 110.389 seconds after policy attachment. The idle session ended with `idle-timeout`, the active worker passed fresh liveness checks, and a new session read the matching saved-file SHA-256. Vfolder cleanup reached `delete-complete` before its database record was purged. Session and infrastructure cleanup passed.

The recording, images, and manifests are copied unchanged from the run. The original `../macos-arm64/` reference remains unchanged. This run includes the new liveness, physical-deletion, and failed-replay checks. It preceded the final shared supervisor changes for process-group cleanup, which were checked separately by the final FIFO rerun and regression tests. These observations do not establish Linux-host behavior.

Elapsed times describe this run and include periodic checks and polling. They are not a performance benchmark. This is CPU execution, not physical GPU execution.

From `2026-11-backend-ai-intro/`, replay without Docker:

```sh
uv run --locked backend-ai-intro render experiments/03-idle-reclamation/recordings/macos-arm64-review-20261105/recording.json --output /tmp/backend-ai-idle-review-replay
```

`manifest.json` binds the recording and generated images with SHA-256 hashes. `run-status.json` records complete-run success; `environment-cleanup.json` records infrastructure cleanup.
