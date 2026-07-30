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
uv run garboid-bots
```

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
  --output-dir tournament-results
```

See the [tournament runbook](src/garboid_pocketrocks/tournament/README.md) for
scheduling, rating semantics, artifacts, and reproduction.

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

## Documentation

Start at the [documentation index](docs/README.md). Current operational
runbooks live beside the
[simulator](src/garboid_pocketrocks/simulator/README.md),
[tournament](src/garboid_pocketrocks/tournament/README.md),
[heuristics](src/garboid_pocketrocks/heuristics/README.md), and
[neural](src/garboid_pocketrocks/neural/README.md) packages.
