# Neural self-play bot

This package trains a recurrent actor-critic entirely through PocketRocks
self-play. The implemented training envelope covers every live value chart
(`live-A` through `live-E`) with three, four, and five players. Every seat uses
the same frozen policy snapshot during collection, and every seat trajectory is
then used by PPO.

This is a training system, not yet a registered live remote `NeuralBot`. Two
frozen checkpoints are registered for local simulation and the standard
tournament:

- `vector_ppo_small_v1_g1500`, the 1,500-game smoke policy;
- `vector_ppo_large_v1_g350k`, the rounded release name for the large policy
  trained for exactly 349,860 games and 196 updates.

Their inference manifests retain exact training ages and repository provenance.
Together they exercise legal SDK-compatible inference, balanced self-play, PPO
updates, value diagnostics, throughput calibration, and update-boundary
checkpoint/resume.

## Install

From the repository root:

```bash
uv sync --locked --extra neural
```

All commands below also run from the repository root. Output directories must
be new or empty.

Run the standard tournament, including both frozen neural policies, with:

```bash
uv run --extra neural garboid-tournament
```

## Model and information boundary

`NeuralPolicy` is stateless across calls. It encodes the complete padded public
history from a new zero GRU state on every inference, so interleaved SDK games
cannot leak recurrent state into one another.

```text
current public/private snapshot ──> snapshot MLP ───────┐
five learner-relative seats ──────> shared seat MLP ────┼─> two-layer trunk
complete public history ──────────> event MLP + GRU ────┘        ├─> bid logits
                                                               ├─> reveal logits
                                                               └─> scalar value
```

The snapshot includes the decision phase, player count, current action and
resources, learner-relative priority seat, public ruleset/chart data, active
and possible objectives, and the acting seat's private hand. Each of five seat
slots contains 41 public features for cash, won resources, revealed
information, and objective ownership; the acting seat is rotated to relative
seat zero. History events contain setup, turn-opened, resolved-bid, and public
reveal data, with actors and resolved bids rotated into the same relative-seat
frame.

Both policy and value heads see exactly the information available to a
deployable SDK bot:

- the current SDK `DecisionContext`;
- public `RulesetKnowledge` and cumulative public history;
- the acting seat's current private hand;
- the legal-action mask.

They do not receive opponent hands, deck order, unresolved sealed bids,
simulator RNG state, or an omniscient critic state.

The current checkpoint-stable capacity profiles are:

| Profile | Snapshot / seat / event / GRU / trunk widths | Parameters |
| --- | --- | ---: |
| `small` | 128 / 32 / 64 / 64 / 128 | 125,620 |
| `medium` | 256 / 64 / 128 / 128 / 256 | 443,132 |
| `large` | 512 / 128 / 256 / 256 / 512 | 1,654,156 |

Completed checkpoints receive lossless local IDs in the form
`vector_ppo_<profile>_v1_g<completed-games>`, for example
`vector_ppo_large_v1_g1000000`. Network size and training age are therefore
visible in tournament/reporting identities. The tournament's
`vector_ppo_large_v1_g350k` release name is an explicit rounded alias requested
for the 349,860-game checkpoint; its manifest remains the source of truth for
the exact age.

The trunk feeds three heads:

- 101 bid logits: universal pass plus bids 1 through 100;
- 6 reveal logits: universal pass plus five hand positions;
- one scalar expected return.

The active phase head is projected into the 106-action universal action space.
Context-illegal actions are masked to negative infinity and therefore have
exactly zero probability. Collection samples each decision from a stable,
row-specific seed; greedy evaluation chooses the lowest-index maximum through
`argmax`.

The training encoder supports a maximum five-card hand and 77 public events,
which covers the finite A-E, three-to-five-player live envelope. The older
Stage 1 helpers remain available for the legacy live-A/three-player smoke.

## What one training update does

`plan_mirror_episodes` builds a balanced matrix of 15 cells: five charts times
three player counts. `games_per_cell=N` therefore collects `15 * N` complete
games per update. In every currently orchestrated game, all seats use a frozen
copy of the current policy and all seats are trainable.

Collection uses the SDK's NumPy `BatchSimEngine`. Plans are grouped into
homogeneous player-count batches while each engine row retains its own chart
and deterministic engine seed. Bid and reveal requests are sorted, encoded in
batches, and split only at `max_inference_batch`. Per-decision sampling seeds
make chosen actions independent of worker completion order and batch packing.

The collector records each trainable transition's observation, mask, sampled
action, old log probability, old value, decomposed reward, terminal flags, and
chart/player/phase metadata. The default reward is:

- normalized change in public financial potential;
- terminal conversion from public potential to final money;
- a `1.0` first-place bonus shared among tied winners.

Optional placement and event shaping bonuses default to zero.

PPO then:

1. packs all complete seat trajectories into immutable NumPy arrays;
2. computes GAE independently per seat trajectory with `gamma=1.0`,
   `lambda=0.95`, and zero terminal bootstrap;
3. forms value targets as `advantage + old value`;
4. normalizes advantages over the complete rollout;
5. shuffles deterministically once per epoch and trains in minibatches;
6. applies clipped policy loss, half squared-error value loss, entropy bonus,
   and gradient clipping with a persistent Adam optimizer.

`gamma=1.0` is intentional: decisions are not uniformly spaced because an
auction winner can receive an additional reveal decision. The value head is a
deployable-information critic trained to estimate the resulting future return,
not hidden game state.

## Batching, actors, and device calibration

The resolved collection path depends on device and worker count:

- one worker: one process runs SDK vector batches and batched inference on the
  selected CPU, CUDA, or MPS device;
- CPU with multiple workers: spawned vector actors each receive a frozen model
  copy, run `BatchSimEngine` batches, and perform local CPU inference;
- accelerator with multiple workers: spawned game workers send inference
  requests to one centrally batched accelerator policy.

The committed automatic calibration currently benchmarks bounded candidates:
one or eight CPU vector actors, plus one-vector-actor CUDA/MPS candidates when
available. It measures a complete balanced collection and the configured PPO
epoch count, then selects the highest decisions/second result.
`benchmark.json` records every successful result and the selected candidate.
Set both `device` and
`parallel.workers` explicitly to skip calibration.

`learner_threads` controls Torch learner threads. CPU vector actors force one
Torch intra-op and inter-op thread each to avoid oversubscription.

## Determinism and committed run profiles

The self-play smoke keeps `deterministic_algorithms=true` (the configuration
default). Engine seeds, per-seat decision seeds, model initialization, and PPO
epoch shuffles are derived from the root seed. Pin CPU and one worker when an
exact reproducibility check matters:

```bash
uv run --extra neural garboid-train smoke \
  --device cpu \
  --workers 1 \
  --output-dir artifacts/neural-smoke
```

The default smoke pins the measured eight-actor CPU path and runs one PPO
update with 100 games in each cell: 1,500 games total. It verifies full cell
coverage, legal/fault-free collection, finite value metrics, checkpoint digest
replay, and reloadable optimizer/progress state. It writes
`self-play-smoke-result.json` in addition to the normal run artifacts. Use
`--games-per-cell 1 --workers 1` for a 15-game developer probe.

The approximately ten-minute and up-to-eight-hour profiles deliberately set
`deterministic_algorithms=false`, allowing accelerator-compatible kernels.
Both use hardware calibration when `device` or `workers` is `auto`:

```bash
# Current medium-profile, approximately ten-minute plan
uv run --extra neural garboid-train train \
  --config configs/neural/initial-10m.json \
  --output-dir artifacts/neural-10m

# Current large-profile, up-to-eight-hour plan; starts a separate lineage
uv run --extra neural garboid-train train \
  --config configs/neural/long-8h.json \
  --output-dir artifacts/neural-8h
```

Wall-clock limits are checked only between complete PPO updates. A final update
is never interrupted and can overshoot the nominal limit. The two profiles use
different model shapes (`medium` versus `large`), so the large run cannot
resume the medium checkpoint.

To compare model capacity and device behavior without starting a durable run,
use the real-rollout profile benchmark:

```bash
uv run --extra neural python \
  scripts/training/benchmark_neural_profiles.py \
  --profiles small medium large \
  --devices cpu mps \
  --games-per-cell 20
```

It prints one JSON record per profile/device pair with parameter count,
collection games/decisions per second, and one-epoch PPO time.

### Run status

The current code path is implemented and a 1,500-game small-profile smoke
artifact exists in `artifacts/neural-self-play-smoke`: one update, 1,500 games,
113,948 decisions, zero illegal actions/faults, and successful checkpoint and
resume probes. That artifact was produced by an earlier repository commit, so
it is evidence for the implemented path rather than a fresh verification of
the current working tree.

An earlier approximately ten-minute artifact also exists in
`artifacts/neural-initial-10m`: 99 updates and 11,880 games. It used the former
small model and former 8,192-decision target. It is not a run of the current
`medium` `initial-10m.json`.

The current medium ten-minute plan and current large eight-hour plan have not
been started in the checked-in artifacts. No long-run strength claim is made.

## Checkpoint, resume, inspect, and evaluate commands

Every completed update atomically replaces:

```text
RUN/checkpoints/latest/
  manifest.json
  model.pt
  optimizer.pt
  rng.pt
  metrics.json
```

The bundle includes the encoder/model/run configurations, repository commit,
progress and per-cell counts, lineage, model and optimizer state, saved Torch
RNG state, parameter digest, and checksums for every payload. Loading is
fail-closed on extra/missing files, schema or tensor incompatibility,
non-finite state, or checksum/digest mismatch.

Inspect progress and the support contract:

```bash
uv run --extra neural garboid-train inspect \
  --checkpoint artifacts/neural-10m/checkpoints/latest \
  --format json
```

Resume at the next update boundary into a new output directory:

```bash
uv run --extra neural garboid-train resume \
  --checkpoint artifacts/neural-10m/checkpoints/latest \
  --output-dir artifacts/neural-10m-resumed \
  --max-additional-updates 10
```

Omit `--max-additional-updates` to use the checkpointed run budget. A
`--config` override may change compatible run controls, but it may not change
the root seed or model profile.

The CLI exposes an `evaluate` command:

```bash
uv run --extra neural garboid-train evaluate \
  --checkpoint artifacts/neural-10m/checkpoints/latest \
  --config configs/neural/initial-10m.json \
  --output artifacts/neural-10m/evaluation.json
```

At present this command only validates/inspects the checkpoint and writes that
metadata plus the evaluation-config path. It does **not** play evaluation
games or produce a strength report.

## Metrics and run files

A normal run directory contains:

```text
RUN/
  resolved-config.json
  benchmark.json
  metrics.jsonl
  checkpoints/latest/...
```

- `resolved-config.json` is the complete post-calibration configuration.
- `benchmark.json` contains calibration candidates and the selected device /
  actor settings (or an empty result list when settings were explicit).
- `metrics.jsonl` contains one JSON object per completed update.
- `checkpoints/latest/metrics.json` duplicates the most recent update metrics
  inside the checksummed checkpoint.

Each update record includes duration and games-per-cell; collection totals,
cell coverage, elapsed/inference/queue/IPC/worker time, games/second,
decisions/second, and inference batch sizes; and PPO losses, entropy,
approximate KL, clip fraction, gradient norms, optimizer-step counts, and
transition counts.

Value diagnostics compare rollout-time predictions with realized GAE return
targets. They include mean prediction/target, MAE, RMSE, bias, explained
variance, correlation, and ten size-balanced calibration buckets. The same
metrics are emitted globally and by chart, player count, and early/middle/late
game phase. `None` is used where variance or correlation is undefined; JSON
never receives NaN sentinels.

## Implemented versus scaffolded

Implemented in the trainer:

- balanced A-E / 3-5-player all-seat mirror self-play;
- SDK `BatchSimEngine` vector collection;
- frozen rollout snapshots and row-seeded legal action sampling;
- single-process and multi-actor CPU collection;
- CPU/CUDA/MPS discovery and throughput calibration;
- packed PPO, value diagnostics, atomic latest checkpoints, resume, and
  inspection.

Present as configuration fields or helper modules, but not yet connected to
the training loop:

- paired strength evaluation, confidence intervals, and promotion;
- heuristic/champion evaluation at start, interval, or end;
- checkpoint-league mixing (`league_fraction` is currently ignored);
- interval checkpoints and retention (`checkpoint_interval_seconds` and
  `keep_periodic_checkpoints` are currently ignored);
- periodic evaluation (`evaluation_interval_seconds`, `evaluate_at_start`, and
  `evaluate_at_end` are currently ignored);
- a CLI strength evaluator (the current `evaluate` command is metadata-only);
- a registered deployable SDK `NeuralBot`.

The orchestration currently writes `checkpoints/latest` after every update and
always calls `plan_mirror_episodes`; checkpoint league fields remain empty.
