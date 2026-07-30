# Parallel Neural Self-Play Training Design

**Date:** 2026-07-29

**Status:** Approved for implementation

## Purpose

Extend the existing Stage 1 recurrent PPO proof into a durable, measurable
self-play trainer. One shared policy/value network will train against frozen
copies of itself across every live value chart and supported player count:

- charts A, B, C, D, and E;
- three, four, and five players;
- all seats represented in training;
- optional historical checkpoint opponents after bootstrap.

The implementation must preserve the live information boundary, use resumable
training checkpoints, report the quality of its estimated value function, and
use measured throughput to configure smoke, ten-minute, and eight-hour runs.

## Existing foundation

The repository already has:

- the deterministic `GameEngine` and public-history adapters;
- fixed universal action encoding and legal-action masks;
- learner-relative observations with five seat slots;
- a roughly 125,000-parameter GRU actor-critic;
- gamma-one GAE and clipped PPO;
- deterministic Stage 1 rollout collection;
- inference-only model checkpoints;
- a Stage 1 CPU smoke command.

The measured warm Stage 1 baseline on the development machine is 16
three-player games, 331 learner decisions, one PPO update, and checkpoint
replay in 1.03 seconds. The raw simulator sustains approximately 431, 343, and
255 random-policy games per second for three, four, and five players
respectively. Neural history encoding and batch-size-one inference, rather than
the game engine, are therefore the first optimization target.

## Chosen approach

Use batched all-seat self-play with one central optimizer.

At the beginning of a PPO update, copy the trainable model into an immutable
collection snapshot. Every seat in a mirror game uses that snapshot. All
actions produced by the snapshot are on-policy, so every seat trajectory may
enter the PPO batch. The trainable model changes only after the complete
rollout has been collected.

After bootstrap, a configured minority of games may use older immutable
checkpoints. In those league games, only trajectories from seats controlled by
the current collection snapshot enter PPO; historical-policy trajectories are
opponents, not learner samples. Heuristic bots remain evaluation baselines and
are not training opponents.

This design avoids multiple independent PPO learners. A single optimizer
preserves a clear on-policy boundary, one checkpoint lineage, and reproducible
resume behavior.

## Support envelope

The neural encoder configuration expands from live-A/three-player support to:

- rulesets `live-A` through `live-E`;
- player counts `(3, 4, 5)`;
- five relative seat slots;
- maximum bid 100;
- maximum private hand size five;
- a history bound derived from the maximum public history over every supported
  chart and player count.

All live charts share deck/setup limits today, but support is declared and
checkpointed as a finite set rather than inferred at load time. A ruleset
outside the stored envelope is rejected.

The policy and value heads continue to consume only:

- the acting seat's `DecisionContext`;
- public `RulesetKnowledge`;
- public cumulative history;
- that seat's private hand;
- its legal-action mask.

They never consume another seat's hand, shuffled deck order, unresolved sealed
bids, simulator RNG state, or terminal scores before termination.

## Deterministic episode planning

Each episode plan records:

- global episode index and PPO update index;
- chart name and player count;
- engine seed;
- one sampling seed per seat;
- the policy identity assigned to each seat;
- whether each seat's trajectory is trainable.

Seeds derive from the root seed, stable namespaces, and explicit indices.
Episode plans do not consume scheduling-dependent RNG state. Workers may finish
out of order, but trajectories are restored to plan/seat order before metric
aggregation and PPO minibatch planning.

Training uses balanced round-robin coverage across the 15 chart/player-count
cells. A complete coverage cycle contains at least one game from every cell.
Run metrics report planned and completed counts for every cell.

## Batched game collection

A `SelfPlayGame` owns one engine transition, public-history adapter, per-seat
reward trackers, and per-seat trajectory builders. It exposes all currently
pending seat decisions without resolving any of them.

A `BatchedSelfPlayCollector` advances many active games:

1. gather pending contexts from every active game;
2. encode each context from its own seat-relative view;
3. group requests by assigned immutable policy;
4. batch each group through the appropriate policy;
5. sample with the request's dedicated generator;
6. validate every selected universal action;
7. submit the entire pending decision batch to its game;
8. append public events and per-seat rewards;
9. refill completed active-game slots from the deterministic plan.

All simultaneous bids are selected before any bid is submitted, so batching
cannot leak one seat's sealed action to another.

Production rollouts retain packed encoded observations, masks, actions, old log
probabilities, old values, rewards, termination flags, and provenance. Full
policy logits are retained only by diagnostic tests. Packed array/tensor
storage avoids one large Python object graph for the 1,500-game smoke.

## Parallel execution

Parallel training has three layers:

1. multiple active games in a collector;
2. spawned CPU workers for game execution and observation encoding;
3. batched inference and PPO minibatches on the selected Torch device.

The main process owns policy modules, sampling generators, the inference batch
queue, PPO, metrics, and checkpoints. Spawned workers own engine state and
public-history/reward bookkeeping. Workers exchange bounded, schema-checked
messages with the main process. They do not load or mutate the trainable model.

The central process drains pending inference requests until reaching the
configured maximum batch size or a short maximum queue delay. It returns
actions and old policy quantities to the originating workers. This allows CPU
simulation to progress in parallel while keeping accelerator inference
batched and model ownership unambiguous.

The supported worker setting is `1`, a positive explicit count, or `auto`.
`auto` benchmarks candidate worker/active-game configurations and chooses the
highest stable decisions-per-second result. Worker count must not change
planned games, sampled actions, ordered deterministic metrics, or the final
parameter digest on the same device.

## Device support

Training accepts `auto`, `cpu`, `cuda`, and `mps`.

- Game simulation and observation construction remain on CPU.
- Batched collection inference and PPO run on the selected Torch device.
- `auto` benchmarks CPU and each available accelerator with representative
  batches, then selects the fastest complete collection-plus-update path.
- CPU remains the portable CI and correctness device.
- Same-device resume is exact at an update boundary.
- Cross-device training is supported but is not required to be bit-identical.

Checkpoints store device-independent state dictionaries and load through an
explicit `map_location`. Live batch-size-one inference chooses its device
independently from training; CPU is the default unless a live inference
benchmark demonstrates lower latency on an accelerator.

## PPO and estimated value function

The existing actor-critic architecture remains the base model. The scalar value
head predicts the acting seat's expected undiscounted training return from the
same deployable information as the policy.

Each seat trajectory has its own gamma-one GAE sequence. Reveal decisions do
not introduce artificial temporal discounting. The value target is the
trajectory return produced by the configured accounting and terminal reward
contract.

Every update reports:

- value loss;
- mean absolute error and root mean squared error;
- signed prediction bias;
- explained variance;
- predicted/realized return correlation when defined;
- calibration buckets containing prediction count, mean prediction, and mean
  realized return;
- the same value summaries by chart, player count, and early/middle/late game
  phase when each slice has enough samples.

Undefined correlations or explained variance from constant targets are
reported as null with a reason, not NaN. Non-finite values, targets, losses, or
gradients are hard failures.

PPO metrics additionally include policy loss, entropy, approximate KL, clip
fraction, probability ratio summaries, optimizer steps, and pre/post clipping
gradient norms.

## Evaluation and checkpoint league

Training return never promotes a checkpoint.

Held-out evaluation uses fixed disjoint seeds. For every chart and player
count, candidate-versus-incumbent games rotate the candidate through every
seat. The primary paired utility is:

```text
normalized final-money delta + first-place-share delta
```

Reports also include outright wins, tied firsts, mean rank, final money, pass
rate, bid distribution, action wins, illegal actions, and faults. Paired
bootstrap confidence intervals quantify uncertainty.

The initial run may finish without a promotion if its interval is
inconclusive. The long run adds a candidate to the immutable league only when
its configured paired gate passes. The league retains the champion and older
promoted snapshots so training does not depend only on the latest policy.

## Durable checkpoints

Training writes versioned directories containing:

```text
checkpoint/
  manifest.json
  model.pt
  optimizer.pt
  rng.pt
  metrics.json
```

The manifest records:

- checkpoint/encoder/action schema versions and hashes;
- repository and dependency versions;
- encoder/model/reward/PPO/curriculum/opponent configurations;
- root seed and next update/episode/decision counters;
- chart/player-count coverage;
- lineage, champion, and league identities;
- checksums and the canonical parameter digest.

`rng.pt` contains only tensor generator states needed for exact continuation.
Episode/environment seeds remain derivable from counters and the root seed.
Model and optimizer files contain state dictionaries, not application objects.

Saving writes a temporary sibling directory, flushes files, validates the
complete bundle, and atomically renames it. Loading rejects extra or missing
files, incompatible schemas/configurations, insufficient bounds, corrupt
checksums, non-finite tensors, or an invalid optimizer structure.

An inference export contains only the manifest and model state needed for
deployment. It cannot be used as a resumable training bundle.

At an update boundary, resuming must reproduce the uninterrupted next episode
plans, rollout metrics, PPO metrics, model parameters, optimizer state, and
next checkpoint digest on the same machine/device configuration.

## CLI and artifacts

The end-state commands are:

```text
garboid-train smoke
garboid-train train --config CONFIG.json --output-dir RUN
garboid-train resume --checkpoint CHECKPOINT --output-dir RUN
garboid-train evaluate --checkpoint CHECKPOINT --config CONFIG.json
garboid-train inspect --checkpoint CHECKPOINT
```

Run directories are new unless `resume` names an exact compatible checkpoint.
They contain the resolved configuration, append-only update metrics, benchmark
results, checkpoints, evaluations, league metadata, and selected replays.

SIGINT/SIGTERM stop scheduling new games, finish or cancel the current
uncommitted update according to configuration, and leave the last atomic
checkpoint valid. Wall-time limits are checked before beginning an update
using the rolling p95 update duration so a run stops near its requested
envelope.

## Performance metrics and calibration

Every update records:

- games and decisions per second overall and by chart/player count;
- collection, encoding, inference, queue wait, IPC, PPO, evaluation, and
  checkpoint wall time;
- mean, p50, and p95 inference batch size;
- active-game and worker utilization;
- transitions per optimizer step;
- accelerator/CPU utilization metadata available from Torch and the standard
  library;
- peak process memory where available.

Cold dependency/model startup is separate from warm training time.

Calibration uses a small balanced subset of the 15 cells to compare candidate
device, worker, active-game, and inference-batch settings. It chooses the
fastest configuration that completes without faults and writes the resolved
choice before the measured run. It does not adapt learning hyperparameters.

## Run profiles

### Smoke

The default smoke runs 100 games in every chart/player-count cell:

```text
5 charts * 3 player counts * 100 games = 1,500 games
```

It:

- performs the bounded parallel/device calibration;
- freezes one mirror policy for collection;
- completes all 1,500 games with all seats represented;
- performs PPO training over the collected trajectories;
- reports cell-specific batching and throughput metrics;
- reports policy and value-function diagnostics;
- verifies that at least one parameter changes;
- atomically saves and reloads a training checkpoint;
- replays a canonical inference fixture;
- verifies one small update-boundary resume continuation.

The smoke measures mechanics and scale. It does not claim strategy strength.
Tests may override games per cell to one; the user-facing default remains 100.

### Initial run

The initial profile has a 600-second wall-time envelope. It:

- uses the smoke-selected device/parallel configuration;
- samples the 15 cells evenly by completed decisions;
- targets enough decisions per PPO update for a measured 10–30 second update;
- saves recovery checkpoints approximately every two minutes;
- runs held-out baseline and final evaluations;
- writes a recommendation for long-run decisions/update, checkpoint interval,
  and evaluation interval based on observed overhead.

### Long run

The long profile resumes an initial-run checkpoint and has a maximum
28,800-second wall-time envelope. It:

- uses the resolved throughput-based PPO batch size;
- saves approximately every 15 minutes and at clean shutdown;
- runs paired candidate evaluation approximately every 30 minutes;
- permits historical league games after the first qualifying promotion;
- keeps the latest, best, and periodic recovery checkpoints;
- keeps evaluation plus checkpoint overhead near or below 15 percent.

Intervals are converted to update boundaries in the resolved configuration.
No checkpoint or evaluation interrupts a PPO update.

## Failure handling

Training stops with a specific error on:

- any illegal sampled action or nonzero illegal probability;
- engine or bot faults;
- incomplete or truncated games;
- non-finite observations, logits, probabilities, values, returns, losses, or
  gradients;
- missing required cell coverage;
- worker death, malformed IPC, or an inference response mismatch;
- corrupt/incompatible checkpoints;
- same-device update-boundary resume divergence.

No failure silently substitutes a heuristic action, pass, CPU device, smaller
curriculum, or unverified checkpoint.

## Test strategy

Implementation follows red-green-refactor TDD.

Unit and integration tests cover:

- support bounds for charts A-E and player counts three through five;
- balanced deterministic 15-cell episode planning;
- per-seat canonical observations and rewards;
- sealed bids remaining private until simultaneous resolution;
- every mirror seat producing an on-policy trajectory;
- collection weights remaining frozen for a rollout;
- serial and batched seeded collection equivalence;
- worker-count-independent plans, actions, ordered metrics, and digest;
- hand-calculated GAE, PPO, and value-quality metrics;
- paired evaluation seeds and full seat rotations;
- CPU/CUDA/MPS device validation and explicit unavailable-device errors;
- atomic checkpoint checksums, schema rejection, and optimizer/RNG restore;
- exact same-device update-boundary resume;
- CLI configuration, artifact, inspect, and overwrite protections.

The full smoke acceptance requires:

- exactly 100 completed games in each of the 15 cells;
- zero illegal actions, illegal probability, bot faults, and engine faults;
- finite policy, value, return, loss, and gradient metrics;
- changed trainable parameters;
- nonempty throughput and batching metrics;
- a valid training checkpoint reload;
- a verified canonical inference replay;
- verified update-boundary continuation.

Playing strength is evaluated separately on held-out paired games. A smoke run
passing does not imply that the policy improved.

## Implementation sequence

1. Expand the encoder/support envelope and deterministic 15-cell plans.
2. Implement all-seat game state and serial batched collection.
3. Add packed rollout storage, complete PPO metrics, and value diagnostics.
4. Add atomic resumable checkpoints and exact resume.
5. Add the parallel worker/central inference protocol.
6. Add device calibration and throughput reporting.
7. Add held-out evaluation and the checkpoint league.
8. Add CLI commands and the smoke/initial/long configurations.
9. Run the 1,500-game smoke, then use its measurements to finalize the
   ten-minute and eight-hour resolved plans.
