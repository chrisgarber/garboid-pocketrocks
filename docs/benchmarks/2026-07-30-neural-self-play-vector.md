# Neural self-play vector benchmark

This note records the first measured SDK-vector self-play path for recurrent
PPO. It is a throughput and correctness baseline, not a strength result.

## Workload

Self-play runs directly on the PocketRocks SDK `BatchSimEngine`. A balanced
update covers every live value chart (`live-A` through `live-E`) at three,
four, and five players. The smoke uses 100 games in each of these 15 cells:
1,500 complete games total. All seats act from one frozen collection snapshot,
and all trainable seat trajectories enter PPO after collection finishes.

## Smoke throughput

The measurements below use the current small recurrent model. Collection rates
include neural observation encoding and batched policy/value inference.
`Total update` also includes PPO and update-boundary work.

| CPU collection path | Actors | Collection (s) | Games/s | Decisions/s | Total update (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single vector | 1 | 14.402 | 104.15 | 7,676.84 | 28.359 |
| Parallel vector | 8 | 11.522 | 130.19 | 9,596.00 | 24.902 |
| Final integrated persistent-pool path | 8 | 9.448 | 158.76 | 11,702.35 | 22.989 |

Eight CPU vector actors improve collection by about 25% in games and decisions
per second, and reduce total update time by 3.457 seconds. The gain is useful
but sublinear, so actor count should remain a measured setting. The final
integrated smoke was another independent run rather than a controlled paired
benchmark, so its additional gain should not be attributed solely to the pool.
The persistent pool's structural benefit appears on multi-update runs: actors
remain alive while each update receives an exact fresh policy snapshot.

## Raw engine ceiling

The SDK engine alone sustains approximately 2,214 games/s at batch 64 and
10,984 games/s at batch 1,024. These engine-only rates exclude neural encoding,
history replay, inference, rollout retention, and PPO. Their large gap from the
end-to-end smoke shows that neural data preparation, inference, and training
dominate; the game engine is not the present bottleneck.

## Accelerator calibration

A separate 960-game calibration compared complete collection and a one-epoch
PPO update:

| Device/path | Collection (s) | Complete decisions/s | PPO (s) |
| --- | ---: | ---: | ---: |
| CPU, one vector actor | 9.009 | 4,070 | 8.335 |
| MPS, one vector actor | 52.991 | 1,053 | 14.032 |

For the current small model and workload, CPU wins both collection and PPO, so
the small profile should use CPU. This is machine- and model-dependent:
`device: auto` and `workers: auto` calibrate representative
collection-plus-PPO candidates before a run.

The checkpoint-stable capacity profiles now contain 125,620 (`small`),
443,132 (`medium`), and 1,654,156 (`large`) parameters. A 300-game,
one-PPO-epoch probe measured:

| Profile/device | Collection decisions/s | Collection (s) | PPO (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Medium CPU | 5,635 | 3.931 | 4.804 | 8.735 |
| Medium MPS | 1,155 | 19.177 | 4.405 | 23.581 |
| Large CPU | 3,358 | 6.600 | 10.423 | 17.023 |
| Large MPS | 1,112 | 19.922 | 5.201 | 25.123 |

MPS cuts large-profile PPO time roughly in half, but synchronous inference and
collection still make CPU faster end to end at this scale. Hardware
calibration now times the configured PPO epoch count so longer large-profile
runs can select MPS if learner savings outweigh collection cost.

## Checkpoints and value calibration

Durable checkpoints save model and optimizer state, RNG state, run
configuration, completed updates/games/decisions, per-cell coverage, lineage,
file checksums, and a canonical parameter digest. They load fail-closed and can
resume at an update boundary.

Each update records value-head accuracy against realized training returns:

- MAE and RMSE for absolute error;
- signed bias for systematic over- or under-prediction;
- explained variance and prediction/target correlation when defined;
- size-balanced buckets comparing mean predicted and mean realized return;
- the same summaries globally and by live chart, player count, and decision
  phase.

Calibration here means whether predicted returns agree with realized returns;
it is separate from hardware calibration, which selects a device and actor
configuration by measured throughput.

## Run commands and plan

Install the locked optional dependencies once:

```bash
uv sync --locked --extra neural
```

Run the 1,500-game acceptance smoke:

```bash
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke
```

Run the committed approximately ten-minute profile in a new directory:

```bash
uv run --extra neural garboid-train train \
  --config configs/neural/initial-10m.json \
  --output-dir artifacts/neural-10m
```

If its checkpoints and rollout/PPO metrics are healthy, start the separate
large-profile lineage bounded to at most eight hours:

```bash
uv run --extra neural garboid-train train \
  --config configs/neural/long-8h.json \
  --output-dir artifacts/neural-8h
```

Every output directory must be new or empty. The eight-hour command is the
planned next stage; this report does not claim that a long run was started.
Medium and large checkpoints are intentionally not cross-resumable.
