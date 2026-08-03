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

The curated field includes `fixed-objective-overlay-v3`, a tuned fixed-target
bot that caps resource-auction bids at the exact cash-and-tiebreak amount needed
to guarantee a win. Its development evidence and full-field result are recorded
in the [dated benchmark report](docs/benchmarks/2026-08-02-fixed-objective-overlay-v3.md).

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
`surplus-v11`. Generations v1-v11 remain explicitly selectable for deterministic
simulation. The [initial development ladder](docs/benchmarks/2026-08-02-surplus-heuristic-ladder.md),
the [objective-tuning report](docs/benchmarks/2026-08-02-surplus-v7-objective-tuning.md),
the [opponent-threat report](docs/benchmarks/2026-08-02-surplus-v8-opponent-objective-threat.md),
the [liquidity and objective-progress report](docs/benchmarks/2026-08-02-surplus-v9-liquidity-objective-progress.md),
the [action-aware liquidity report](docs/benchmarks/2026-08-02-surplus-v10-action-aware-liquidity.md),
and the [net-loan valuation report](docs/benchmarks/2026-08-03-surplus-v11-net-loan-valuation.md)
record the hypotheses, fixed-seed comparisons, ablations, and negative experiments.

### Surplus research backlog

These are hypotheses, not measured results. The effectiveness score is a
subjective estimate of likely tournament value from 1 (small) to 5 (large),
including both expected upside and how often the feature should affect play.

| Candidate feature | Estimated effectiveness | Why it may help |
|---|---:|---|
| Net loan proceeds and marginal liquidity value | 5/5 | Score each legal loan bid using post-win cash (`cash + principal - bid`) and the projected shortfall it removes. V10 can legally bid from the proceeds, but its fixed principal fraction pays the same fee for a small or large shortage. Selected for v11. |
| Opponent-specific clearing-price and win-probability model | 4.5/5 | Replace the pooled upper-quartile price with seat-aware bid distributions conditioned on action, cash, game phase, and observed behavior; bid only when marginal win probability justifies the next dollar. |
| Objective reachability and race probability | 4.5/5 | Discount objective progress when too few matching resources remain and increase completion or denial value when a rival is likely to claim first. |
| Short-horizon bundle lookahead | 4/5 | Compare buying now with the expected marginal value and cash cost of plausible remaining Auction1/Auction2 bundles instead of valuing each offer mostly in isolation. |
| Strategic information reveal selection | 4/5 | Stop always revealing the first private card; reveal information that leaks the least about valuable suits and objective plans while accounting for information already public. |
| Endgame cash and action-deck conversion | 3.5/5 | Reduce reserves when too few useful auctions remain, price investments by remaining horizon, and stop borrowing when the remaining deck cannot productively use the proceeds. |
| Opponent liquidity and denial pressure | 3.5/5 | Use rival cash, loans, holdings, and objective proximity to identify who can contest an auction and when one extra bid has meaningful denial value. |
| Portfolio concentration and diversification | 3/5 | Penalize cards with weak future marginal chart value and reward bundles that preserve multiple feasible objective routes. |
| Tie-break-aware bid increments | 2.5/5 | Use the current tie-break seat to decide when matching an expected clearing price is enough and when the bot must bid one more. |

## Documentation

Start at the [documentation index](docs/README.md). Current operational
runbooks live beside the
[simulator](src/garboid_pocketrocks/simulator/README.md),
[tournament](src/garboid_pocketrocks/tournament/README.md),
[promotion](src/garboid_pocketrocks/promotion/README.md),
[heuristic evolution](src/garboid_pocketrocks/evolution/README.md),
[heuristics](src/garboid_pocketrocks/heuristics/README.md), and
[neural](src/garboid_pocketrocks/neural/README.md) packages.
