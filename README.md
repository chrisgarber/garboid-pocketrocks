# Garboid PocketRocks

Garboid PocketRocks builds live bots, deterministic simulation and evaluation
tools, and local self-play training for
[PocketRocks](https://pocketrocks.xyz/). The current live rules and the SDK
version pinned in [`pyproject.toml`](pyproject.toml) are authoritative. This
repository does not maintain a second copy of the game rules.

## Architecture

Every strategy implements the same synchronous brain contract. Live wrappers
attach a server identity; local simulation and training call the brain
directly. The SDK owns game state and transitions, while Garboid owns
orchestration, replay, evaluation, and learning interfaces.

```text
Live server -> PocketRocksFastBot --+
                                    +-> BotBrain -> SDK BotDecision
SDK simulator -> DecisionContext ---+
```

Only SDK-visible public information and the acting bot's private hand may reach
a strategy. See the [architecture decisions](docs/README.md#architecture-decisions)
for the complete authority, information, identity, and determinism contracts.

## Setup

Install [mise](https://mise.jdx.dev/) and Git, then run:

```bash
mise install
uv sync --locked
cp .env.example .env
```

Set the secret `POCKETROCKS_API_KEY` only in `.env`; never commit that file.
Bot IDs are public code constants.

## Commands

Run one seeded live random bot:

```bash
uv run garboid-random-bot --seed 42
```

Run all live bot wrappers together:

```bash
uv run --extra neural garboid-bots
```

The command starts random, aggressive, balanced, passive, and
`ppo-large-teen` in separate threads. Each bot uses its committed public ID and
the shared SDK settings from `.env`. Press Ctrl+C to stop the local process.

`ppo-large-teen` is the public latest alias for the large teen policy family.
It currently runs the immutable `vector_ppo_large_v1_g350k` checkpoint (exactly
349,860 self-play games) under public ID
`bot_dd7807c1-93bc-4f70-80c8-a2f2d7d26429`. Historical checkpoints keep their
versioned local identities; the public alias may advance to a later compatible
large teen checkpoint.

The SDK reads its server, capacity, logging, and API-key settings from the
environment. The tests use its in-memory `FakeTransport` and never connect to
the live service.

Run deterministic local matches:

```bash
uv run garboid-simulate \
  --bots random,aggressive,balanced \
  --games 1000 \
  --players 3 \
  --ruleset live-A \
  --seed 42 \
  --workers 4
```

Add `--format json` for structured results or `--replay-dir PATH` for
per-game replay JSON. See the [simulator runbook](src/garboid_pocketrocks/simulator/README.md).

Rank the curated field with the deterministic tournament:

```bash
uv run --extra neural garboid-tournament \
  --output-dir artifacts/tournaments/default
```

See the [tournament runbook](src/garboid_pocketrocks/tournament/README.md) for
scheduling, rating semantics, artifacts, and reproduction.

Turn any tournament artifact directory into an interactive field and bot
behavior report:

```bash
uv run garboid-visualize tournament-results
```

Use `--decision-reports` on the tournament command for objective, auction,
loan, opponent, and cash-pressure insights. See the
[visualizer runbook](src/garboid_pocketrocks/visualizer/README.md) for metric
definitions and the visualization roadmap.

Run the held-out final exam before promoting a candidate:

```bash
uv run --extra neural garboid-promote \
  --candidate vector_ppo_large_v1_g350k \
  --incumbent vector_ppo_small_v1_g1500 \
  --output-dir artifacts/promotions/neural-comparison
```

The committed opponent pool contains all v1/v2 heuristics and the frozen
350k PPO policy. Compared identities are removed automatically from ordinary
opponent seats before matched games are planned.

See the [promotion runbook](src/garboid_pocketrocks/promotion/README.md) for
the matched-game contract, evidence, and failure reasons.

Search fixed heuristic coefficient grids on development games:

```bash
uv run garboid-evolve-heuristic \
  --manifest configs/evolution/balanced-v3-search-v1.json \
  --development-corpus configs/promotion/development-balanced-v3-broad-v1.json \
  --output-dir artifacts/evolution/balanced-v3-search-v1
```

See the [heuristic evolution runbook](src/garboid_pocketrocks/evolution/README.md)
for the development-only boundary, deterministic evidence, and frozen-candidate
handoff to held-out promotion.

Full simulation, training, evolution, and promotion receipts belong under the
gitignored `artifacts/` tree. Commit versioned inputs, released checkpoints or
frozen candidates, and concise benchmark conclusions instead of raw per-game
logs and repeated corpus snapshots.

Install the optional neural dependencies and run the production-backed smoke:

```bash
uv sync --locked --extra neural
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke
```

The supported neural commands are `smoke`, `train`, `resume`, and `inspect`.
See the [neural runbook](src/garboid_pocketrocks/neural/README.md).

## Released identities

Released bot names, coefficients, checkpoint-backed policies, and real remote
bot IDs are immutable. A behavior-changing strategy update requires either an
explicit new version or an explicit decision to update the current version in
place. Read the [identity decision](docs/architecture/immutable-bot-identities.md)
and the [bot-versioning workflow](.agents/skills/versioning-bots/SKILL.md)
before changing strategy behavior.

The local-only `surplus` heuristic alias currently selects immutable
`surplus-v10`. Generations v1-v10 remain explicitly selectable for deterministic
simulation. The [initial development ladder](docs/benchmarks/2026-08-02-surplus-heuristic-ladder.md),
the [objective-tuning report](docs/benchmarks/2026-08-02-surplus-v7-objective-tuning.md),
the [opponent-threat report](docs/benchmarks/2026-08-02-surplus-v8-opponent-objective-threat.md),
the [liquidity and objective-progress report](docs/benchmarks/2026-08-02-surplus-v9-liquidity-objective-progress.md),
and the [action-aware liquidity report](docs/benchmarks/2026-08-02-surplus-v10-action-aware-liquidity.md)
record the hypotheses, fixed-seed comparisons, ablations, and negative experiments.

## Documentation

Start at the [documentation index](docs/README.md). Current operational
runbooks live beside the
[simulator](src/garboid_pocketrocks/simulator/README.md),
[tournament](src/garboid_pocketrocks/tournament/README.md),
[promotion](src/garboid_pocketrocks/promotion/README.md),
[heuristic evolution](src/garboid_pocketrocks/evolution/README.md),
[heuristics](src/garboid_pocketrocks/heuristics/README.md), and
[neural](src/garboid_pocketrocks/neural/README.md) packages.
