# Neural self-play

This package trains a recurrent actor-critic entirely through PocketRocks
self-play. The supported envelope covers value charts A through E and three,
four, and five players. Every seat uses the same frozen policy snapshot during
collection, and every seat trajectory is available to PPO.

Two immutable checkpoint-backed policies are registered for local simulation
and tournaments:

- `vector_ppo_small_v1_g1500`;
- `vector_ppo_large_v1_g350k`, a rounded release alias whose manifest records
  the exact 349,860-game age.

The checkpoints themselves are immutable local simulation identities, not
remote live bots. The separate public live wrapper `ppo-large-teen` currently
delegates to `vector_ppo_large_v1_g350k` and connects through
`bot_dd7807c1-93bc-4f70-80c8-a2f2d7d26429`. Run it alongside the other public
bots with:

```bash
uv run --extra neural garboid-bots
```

The frozen checkpoint name remains immutable for reproducibility. The public
alias is the moving latest pointer for compatible large teen releases.

## Install and smoke

```bash
uv sync --locked --extra neural
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke
```

The one supported smoke path uses the production trainer over all 15
chart/player-count cells. It trains, validates the checksummed checkpoint, and
resumes for one additional update. Use a small deterministic developer probe
with:

```bash
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke-small \
  --games-per-cell 1 \
  --device cpu \
  --workers 1
```

A smoke proves collection, legal masking, PPO, checkpoint loading, and resume.
It is not evidence of playing strength.

## Information and model boundary

The policy follows the repository's
[public-information decision](../../../docs/architecture/public-information-boundary.md).
It receives the current SDK context, public rules knowledge and history, the
acting seat's hand, and the legal-action mask. It does not receive opponent
hands, deck order, unresolved bids, engine RNG, or omniscient critic state.

Inputs are learner-relative: the acting seat rotates to seat zero, public
actors and bids rotate consistently, and unused seat/hand/history positions
are masked. Snapshot, seat, and event encoders feed a GRU over the complete
public history, then shared policy and value layers. Inference starts from a
zero recurrent state on every call, preventing state leakage between
interleaved games.

The active bid or reveal head is projected into one universal action space.
Illegal actions receive zero probability. Collection samples from
row-specific seeds; greedy tournament inference uses lowest-index `argmax`.

## Collection and PPO

Each update plans a balanced matrix of five charts by three player counts.
Games are grouped into homogeneous-player-count SDK batch engines while each
row retains its own chart and deterministic engine and decision seeds.

The collector records immutable observations, masks, sampled actions, old log
probabilities and values, decomposed rewards, terminal flags, and
chart/player/phase metadata. The default reward combines:

- normalized change in public financial potential;
- terminal conversion from that potential to final money;
- a first-place bonus shared among tied winners.

PPO computes GAE independently per seat trajectory with `gamma=1.0`,
`lambda=0.95`, and zero terminal bootstrap. It normalizes advantages across
the rollout, uses one locally seeded CPU permutation per epoch, applies clipped
policy and value losses plus entropy regularization, clips gradients, and
keeps a persistent Adam optimizer. The update does not consume global Torch
RNG state.

## Workers and devices

- One worker runs vector games and batched inference in one process.
- Multiple CPU actors receive frozen model copies and perform local inference.
- Multiple accelerator workers send requests to one centrally batched CUDA or
  MPS policy.

Automatic calibration measures bounded device/worker candidates and writes
`benchmark.json`. Set both device and worker count explicitly to skip
calibration. CPU actors use one Torch thread each to avoid oversubscription.

Engine, decision, model-initialization, and PPO shuffle seeds derive from the
root seed. Pin CPU and one worker when exact repeatability matters. See the
[deterministic-evaluation decision](../../../docs/architecture/deterministic-evaluation.md).

## Durable training

Start one of the committed profiles:

```bash
uv run --extra neural garboid-train train \
  --config configs/neural/initial-10m.json \
  --output-dir artifacts/neural-10m

uv run --extra neural garboid-train train \
  --config configs/neural/long-8h.json \
  --output-dir artifacts/neural-8h
```

The profiles use different model shapes and start separate lineages. Wall-time
budgets are checked between complete PPO updates; an update is never
interrupted.

Training writes:

```text
RUN/
  resolved-config.json
  benchmark.json
  metrics.jsonl
  checkpoints/latest/
    manifest.json
    metrics.json
    model.pt
    optimizer.pt
    rng.pt
```

Each completed update atomically replaces `checkpoints/latest`. The manifest
contains configuration, progress, lineage, repository provenance, tensor
metadata, parameter digest, and payload checksums. Loading fails closed on
missing or extra files, checksum mismatch, incompatible schemas or tensors,
and non-finite state.

Inspect or resume:

```bash
uv run --extra neural garboid-train inspect \
  --checkpoint artifacts/neural-10m/checkpoints/latest \
  --format json

uv run --extra neural garboid-train resume \
  --checkpoint artifacts/neural-10m/checkpoints/latest \
  --output-dir artifacts/neural-10m-resumed \
  --max-additional-updates 10
```

Resume starts at the next update boundary in a new output directory. Compatible
run controls may be overridden, but the root seed and model profile may not.

Historical checkpoint configurations remain readable. Training rejects
non-default interval checkpoints, checkpoint retention, start/periodic/final
evaluation, and checkpoint-league mixing before creating an output directory
because those controls are not implemented by the current runtime.

## Completed evaluation and promotion

The frozen large v1 checkpoint passed the held-out promotion gate against the
small v1 checkpoint across 480 matched pairs and 960 games, with no bot faults
or failed games. The
[dated benchmark note](../../../docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-promotion.md)
explains the result, its limits, and checkpoint provenance. The immutable
[promotion artifacts](../../../docs/benchmarks/promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500/)
contain the authoritative report, all paired game summaries, and the expanded
corpora.

The evidence supports the large v1 checkpoint over the small v1 checkpoint on
that held-out corpus. The promotion gate does not move aliases, and this result
does not claim that the large policy is the best bot overall.

## Metrics and extension points

Per-update JSON records collection throughput, inference batching and queue
timing, worker timing, reward totals, PPO losses, entropy, approximate KL,
clip fraction, gradient norms, and optimizer steps. Value diagnostics include
error, bias, explained variance, correlation, and calibration globally and by
chart, player count, and game phase. Undefined statistics serialize as `null`,
never NaN.

Current extension seams include leagues, new model profiles, and additional
remote neural wrappers. Activating one requires real runtime behavior, tests,
documentation, and the
[identity policy](../../../docs/architecture/immutable-bot-identities.md);
configuration fields alone do not make a feature supported.
