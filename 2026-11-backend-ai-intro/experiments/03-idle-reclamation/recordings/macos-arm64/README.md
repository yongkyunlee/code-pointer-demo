# Recorded idle-reclamation run

Recorded November 5, 2026, with macOS ARM64, Python 3.13.18, Backend.AI 26.9.7,
and the pinned Prometheus 3.1.0 image. All verification, API cleanup, and Docker
ownership checks passed. The recording has 273 sequential API samples.

| Observed checkpoint | Seconds since recording began |
|---|---:|
| Policy-off observation completed | 211.15 |
| Idle session ended automatically and waiting work started | 353.71 |
| Fresh session read the saved file with the same SHA-256 | 367.36 |
| All allocations released | 373.43 |

The policy was attached at 213.36 seconds. The waiting session was observed
running about 140 seconds later. The 60-second underutilization setting is not a
60-second startup guarantee. Initial-grace, judgement, sweep, scheduling, and
startup processing add delay. CPU rates use a five-minute lookback with the
available history during startup. The agent's Docker Desktop CPU sampling was
intermittent; the runner waited for actual metric contrast before its 75-second
policy-off observation. Missing metrics were not accepted as zero activity.

The idle and active controls were attached to the same utilization checker.
The successful inclusion pairs in the response match the exact requested pairs.
No operator termination was requested before the automatic-reclamation
checkpoint. The queued job produced a checked result, and the saved and restored
file hashes match. All sessions used one generated administrator account.

`manifest.json` binds the visualization to `recording.json`. The run-status and
environment-cleanup files record successful supervision and teardown. GIF frame
durations are presentation choices; timestamps come from the recording.

See [the experiment instructions](../../README.md) for reproduction and scope.
