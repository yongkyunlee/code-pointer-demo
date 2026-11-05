# Recorded local run

Recorded on November 5, 2026, with macOS ARM64, Python 3.13.18, and Backend.AI 26.9.7.

The runner saved 45 API samples. All operational checks, API cleanup, and final Docker ownership checks passed.

| Observed checkpoint | Seconds since recording began |
|---|---:|
| Current running | 20.22 |
| Alice and Bob pending | 22.05 |
| Alice running | 37.83 |
| Bob running | 54.18 |
| All allocations released | 58.18 |

Elapsed times describe this run, not a performance benchmark. Infrastructure setup happened before recording began. The GIF uses fixed frame durations for readability.

Reproduce the experiment with the commands in [the experiment README](../../README.md). Replay this exact recording with `uv run --locked backend-ai-intro render experiments/01-fifo-queue/recordings/macos-arm64/recording.json --output artifacts/replay`.
