# Garboid PocketRocks

Bots, deterministic simulation, evaluation, and local training environments for
[PocketRocks](https://pocketrocks.xyz/).

The project builds bots for
[jaiparera/pocketrockscompetition](https://github.com/jaiparera/pocketrockscompetition)
and connects them to the live service through the
[PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk).
The current live rules and pinned SDK are authoritative; the competition
repository is a secondary reference, not a runtime dependency.

## Architecture

Each strategy is a synchronous `BotBrain`. A live bot combines that brain with
a public bot identity through `PocketRocksFastBot`, while local runners call the
same brain directly:

```text
Live server -> PocketRocksFastBot --+
                                    +-> BotBrain -> SDK BotDecision
Simulator -> DecisionContext -------+
```

The immutable game engine produces SDK `DecisionContext` batches and consumes
SDK `BotDecision` values. Match runners, replay, Monte Carlo evaluation,
PettingZoo, and Gymnasium all drive that one engine, so training logic does not
reimplement game rules.

## Requirements and setup

- [mise](https://mise.jdx.dev/)
- Git

Mise installs the Python 3.14 release line and uv version declared by this
repository. uv manages the virtual environment and locked dependencies.

```bash
mise install
uv sync --locked
cp .env.example .env
```

Do not commit `.env`. `POCKETROCKS_API_KEY` is secret; bot IDs are public class
constants.

## Live random bot

`RandomBot` extends `PocketRocksFastBot`, uses the committed public identity
`RandomBot.BOT_ID`, and delegates decisions to `RandomBotBrain`. The brain:

- samples uniformly from every legal integer bid, treating zero as pass;
- samples uniformly from every revealable card index;
- passes when no bid or reveal is available.

After putting the API key in `.env`, run:

```bash
uv run garboid-random-bot
```

For a reproducible decision sequence:

```bash
uv run garboid-random-bot --seed 42
```

The SDK reads its server, capacity, logging, and API-key settings from the
environment. The tests use its in-memory `FakeTransport` and never connect to
the live service.

## Local simulation and Monte Carlo

Run 1,000 deterministic random-bot matches:

```bash
uv run garboid-simulate \
  --bots random,random,random \
  --games 1000 \
  --players 3 \
  --ruleset live-A \
  --seed 42
```

Use `--format json` for structured output, `--workers N` for local process
workers, and `--replay-dir PATH` to save one versioned JSON replay per game.
Seeds are stable across runs and worker counts.

Evaluation reports games, outright wins, first-place ties, rank counts, final
money samples, first-place margins, seat buckets, ruleset buckets, decision
counts, and faults. Distribution helpers provide mean, median, population
spread, and quantiles.

## Configurable rules

`LIVE_RULESET` models the current public 30-resource and 30-action decks,
3–5-player setup values, four active objectives, and value chart A.
`live_ruleset("A")` through `live_ruleset("E")` select any live chart.

Training can use:

- `FixedRulesetSampler` for one ruleset;
- `WeightedRulesetSampler` for a finite weighted pool;
- `RulesetVariationSampler` for validated combinations of resource counts,
  action counts, player setup, charts, and objective configuration.

Brains receive public `RulesetKnowledge`; shuffled deck order and hidden cards
remain private.

## RL environments

The fixed action encoding is:

- `0`: pass, including SDK `submitBid(0)`;
- `1..max_bid`: positive bids;
- remaining actions: reveal indices.

Every observation includes an action mask, the SDK-visible state, and optional
public ruleset features. It never accepts engine state or exposes hidden deck
order and opponent hands.

PettingZoo exposes the underlying multi-agent game:

```python
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator import FixedRulesetSampler
from garboid_pocketrocks.training import EnvironmentBounds, PocketRocksAECEnv

env = PocketRocksAECEnv(
    ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
    player_count=3,
    bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
)
env.reset(seed=42)
```

Gymnasium trains one learner against synchronous opponent brains:

```python
from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.training import PocketRocksEnv

env = PocketRocksEnv(
    opponent_specs=(
        BotSpec.from_bot_class(RandomBot),
        BotSpec.from_bot_class(RandomBot),
    ),
    ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
    player_count=3,
    bounds=EnvironmentBounds(max_bid=100, max_hand_size=5),
)
observation, info = env.reset(seed=42)
```

The default intermediate reward is the normalized change in public accounting
potential:

```text
cash + investment locks + investment payouts
     - loan principals + claimed objective payouts
```

Terminal resource value supplies the remaining normalized final-money change.
A configurable win bonus is divided among tied winners. Reward components are
returned separately for auditing. No neural policy or training algorithm is
included yet.

## Quality checks

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Roadmap

1. ✅ Establish the repository scaffold and quality gates.
2. ✅ Build and live-test a random SDK bot.
3. ✅ Implement the deterministic engine, replay, Monte Carlo, and RL
   environments.
4. Design and implement value-heuristic bot strategies.
5. Run seeded round-robin evaluations and compare strategies.
6. Build and locally train a neural policy.

## License

Garboid PocketRocks is available under the [MIT License](LICENSE).
