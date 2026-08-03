# Cold mixed MPS v2 preflight

Date: 2026-08-02  
Source commit: `8305711cc1e9e3963fc62a5d9e92bd92fd4d8133`  
Config: `configs/neural/cold-mixed-mps-15m-preflight-v2.json`

## Purpose

Validate the exact first-stage profile before a ten-hour cold training run:
large v2 identity, 75% `strong-field-pool-v1` opponents, eight CPU actors, MPS
learning, 8,192-transition PPO minibatches, bounded one-update policy lag, and
rotating recovery checkpoints. Raw rollout transitions are intentionally
ephemeral.

## Training result

The 900-second budget completed 31 updates and 59,520 games with no training
failure:

| Measurement | Result |
| --- | ---: |
| Total decisions | 4,365,317 |
| Trainable transitions | 1,097,082 |
| Mean collection decisions/s | 8,542.5 |
| Mean update duration | 27.02 s |
| Mean MPS learner time/update | 11.49 s |
| Mean collection overlap/update | 7.76 s |
| Accepted one-update-stale batches | 17 |
| Rejected stale batches | 13 |
| Maximum applied approximate KL | 0.00963 |
| Maximum applied clip fraction | 0.11369 |
| Final value explained variance | 0.5302 |

At the measured end-to-end rate, ten hours projects to roughly 1,240 updates,
2.38 million games, and 43.9 million trainable transitions. Rejected stale
batches are the main throughput tax, but every applied update remained inside
both configured safety limits.

## Storage and recovery

The 31 compact metric records occupy 1.06 MB. The final checkpoint and each
periodic recovery checkpoint occupy about 19 MB; the run retained two periodic
snapshots and occupied 59 MB total. At the projected ten-hour update count,
metrics plus the final and two retained recovery checkpoints should remain near
100 MB, rather than several gigabytes of per-transition diagnostics.

Both periodic checkpoints and the final `vector_ppo_large_v2_g59520` checkpoint
passed checksum/load validation. Resuming the final checkpoint completed update
32 successfully. Exporting it as a model-only inference bundle also succeeded.

## Diagnostic tournament

The exported update-31 checkpoint played a 300-game A-E, 3-5-player diagnostic
tournament against the six unique bots in `strong-field-pool-v1`. There were no
bot faults.

| Rank | Bot | PL rating | Win rate | Mean money |
| ---: | --- | ---: | ---: | ---: |
| 1 | surplus-v10 | 1724.85 | 0.622 | 59.42 |
| 2 | fixed-objective-overlay-v3 | 1603.74 | 0.456 | 55.42 |
| 3 | aggressive-v2 | 1522.99 | 0.222 | 48.04 |
| 4 | balanced-v3 | 1451.69 | 0.105 | 46.59 |
| 5 | fixed-bid-tuned-v1 | 1437.98 | 0.134 | 44.84 |
| 6 | vector_ppo_large_v2_g59520 | 1433.25 | 0.135 | 44.89 |
| 7 | passive-v3 | 1325.49 | 0.035 | 40.74 |

This is a pipeline diagnostic, not promotion evidence: 300 games have wide
sampling uncertainty and bootstrap intervals were intentionally disabled. The
cold model is not yet competitive with the field leaders, which is expected
after only 15 minutes.
