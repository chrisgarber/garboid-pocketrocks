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

Run the three heuristic profiles against one another on another live chart:

```bash
uv run garboid-simulate \
  --bots aggressive,balanced,passive \
  --games 1000 \
  --players 3 \
  --ruleset live-E \
  --seed 42 \
  --workers 4 \
  --format json
```

Unversioned heuristic names track the latest generation. Use explicit names
to reproduce or compare historical generations:

```bash
uv run garboid-simulate \
  --bots balanced-v1,balanced-v2,passive-v2 \
  --games 1000 \
  --players 3 \
  --seed 42 \
  --workers 4
```

Evaluation reports games, outright wins, first-place ties, rank counts, final
money samples, first-place margins, seat buckets, ruleset buckets, decision
counts, and faults. Distribution helpers provide mean, median, population
spread, and quantiles.

## Multiplayer bot tournaments

Rank every registered simulator bot with 10,000 deterministic games:

```bash
uv run garboid-tournament \
  --games 10000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 42 \
  --workers 4 \
  --output-dir tournament-results
```

Those game, player, chart, seed, and bootstrap settings are the defaults.
Lineups contain distinct bot identities, condition exposure is balanced across
the 15 chart/player-count cells, and seats are rotated fairly. A fixed seed
produces identical game summaries, rankings, and bootstrap intervals regardless
of worker count. Workers accelerate both match simulation and interval fitting.

The shared registry includes the random bot, the latest unversioned heuristic
profiles, and the explicit v1 and v2 heuristic generations, so the full default
has enough distinct identities for five-player games. Adding future `BotSpec`
entries to that registry automatically includes them in tournaments. Use
`--bots` or `--exclude-bots` for a reproducible subset and
`--bootstrap-samples 0` for quick experiments.

The estimator fits complete multiplayer finishes with a tie-aware
Plackett–Luce model. `worth` is the fitted positive strength normalized across
the field. `PL rating` is the same result on a familiar display scale:

```text
1500 + 400 * log10(worth / geometric mean worth)
```

It is not a sequence of pairwise Elo updates. A 400-point difference represents
10:1 worth odds. Weak ghost comparisons keep undefeated, winless, and newly
added bots finite; confidence intervals bootstrap complete games rather than
pairwise fragments.

Each run writes:

- `ratings.csv` for spreadsheets and diffs;
- `summary.json` with exact configuration, model diagnostics, condition
  statistics, and calibration bins;
- `report.html` with the leaderboard, bootstrap intervals, PL rating versus
  mean winning money, and pairwise calibration implied by the listwise fit.

The global worth averages across value charts, player counts, opponent
mixtures, and seats. Use the condition statistics and calibration chart to
spot interactions that a single global number cannot represent.

To reproduce the 100,000-game heuristic analysis and its tournament, auction,
information-asymmetry, value-chart, endgame, and cash-constraint
visualizations, follow the
[heuristic bot visualization runbook](docs/analysis/heuristic-bot-visualizations.md).

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

## Bayesian heuristic bots

The heuristic evaluator uses only information available through the live SDK.
For each resource suit, it removes publicly won cards, public reveals, the
bot's own hand, and the current auction offer from the configured finite deck.
It then treats the remaining opponent hand slots as an exact hypergeometric
sample from the remaining cards. This exchangeability assumption is a
maximum-entropy baseline: it is deterministic and does not leak simulator
state, but it does not yet model opponents choosing reveals strategically.

The resulting posterior gives an expected terminal price for each suit.
Conditioning on one suit also changes the other suits because every card comes
from the same finite population. The evaluator values:

```text
resource lot = sum(offered card count * expected terminal price)

auction win = resource lot + objective value - bid
              + option(cash - bid) - option(cash)
              + future(cash - bid) - future(cash)

loan win = -bid
           + option(cash + principal - bid) - option(cash)
           + future(cash + principal - bid) - future(cash)

investment win = fixed payout
                 + option(cash - bid) - option(cash)
                 + future(cash - bid) - future(cash)
```

The `option(...)` term is an increasing, concave value for cash that remains
available for later auctions. It shrinks with the fraction of biddable
resources remaining and reaches zero at the end of the game. Terminal-dollar
accounting stays exact. The v2 `future(...)` term separately protects cash up
to the public remaining-resource horizon; v1 sets its weight to zero.

An objective completed by the offered lot receives its full payout. Incomplete
progress receives a smaller shaped value based on the positive change in
squared progress, multiplied by the remaining-resource horizon and reduced
when opponents are at least as close to the same objective. This progress term
is heuristic option value, not predicted cash.

Every legal integer bid is evaluated. The reservation bid is the largest bid
with nonnegative win value, and each profile shades that reservation bid.
Released generations are immutable:

| Generation | Profile | Liquidity | Future cash | Objective progress | Bid shading |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | aggressive | 0.75 | 0.00 | 0.25 | 0.05 |
| v1 | balanced | 0.40 | 0.00 | 0.20 | 0.25 |
| v1 | passive | 0.15 | 0.00 | 0.15 | 0.50 |
| v2 | aggressive | 0.75 | 1.50 | 0.25 | 0.05 |
| v2 | balanced | 0.40 | 0.75 | 0.20 | 0.25 |
| v2 | passive | 0.15 | 0.60 | 0.15 | 0.30 |

The remote-capable heuristic identities are:

| CLI name | Brain | Bot wrapper | `BotSpec` | Bot ID |
| --- | --- | --- | --- | --- |
| `aggressive` | `AggressiveHeuristicBrain` | `AggressiveHeuristicBot` | `AGGRESSIVE_HEURISTIC_BOT_SPEC` | `bot_00000000-0000-4000-8000-00000000000a` |
| `balanced` | `BalancedHeuristicBrain` | `BalancedHeuristicBot` | `BALANCED_HEURISTIC_BOT_SPEC` | `bot_00000000-0000-4000-8000-00000000000b` |
| `passive` | `PassiveHeuristicBrain` | `PassiveHeuristicBot` | `PASSIVE_HEURISTIC_BOT_SPEC` | `bot_00000000-0000-4000-8000-00000000000c` |

These three IDs are development-only placeholders, not registered live bot
IDs. Replace the relevant class constant before connecting one of these
wrappers to the live service.

Historical `*-v1` and `*-v2` generations are local brain/spec pairs, not
remote bot wrappers. Their versioned name is also their internal simulation
identity (`BotSpec.bot_id`), so they do not pretend to have a server-issued
bot ID. The CLI constructs those local specs from the versioned brain
classes. Python callers that need the prebuilt specs import them from
`garboid_pocketrocks.bots.heuristic`; configured spec instances are not
re-exported from the `bots` package. `aggressive`, `balanced`, and `passive`
remain aliases to v2.

Fast bot wrappers reconcile the chart, starting cash, private-card count, and
objective state exposed by each SDK context with their configured ruleset
knowledge. Pass an explicit `Ruleset` when a game uses different resource or
action deck counts, because the SDK context does not expose those initial
counts.

Reveal decisions use the same finite-population model from an observer's
perspective, without access to the bot's hand. For each candidate suit, the
policy conditions on revealing that card and measures the resulting price
change across every suit, weighted by opponents' public holdings. It exposes
the card with the smallest total opponent benefit; ties use the lowest hand
index.

`BidEvaluation` exposes the full bid curve, posterior, reservation bid, chosen
bid, and additive value breakdown:

```python
from garboid_pocketrocks.heuristics import HeuristicValuator
from garboid_pocketrocks.heuristics.profiles import BALANCED_PROFILE
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator import GameEngine

ruleset = live_ruleset("A")
transition = GameEngine.start(ruleset, player_count=3, seed=42)
assert transition.pending is not None
_, context = transition.pending.contexts[0]

evaluation = HeuristicValuator(BALANCED_PROFILE).evaluate_bid(
    context,
    ruleset.knowledge(3),
)
chosen = evaluation.points[evaluation.chosen_bid]
print(evaluation.reservation_bid, evaluation.chosen_bid)
print(chosen.breakdown)
```

See the [heuristic design](docs/superpowers/specs/2026-07-28-heuristic-bots-design.md)
for the complete model and the
[v1 tournament benchmark](docs/benchmarks/2026-07-28-heuristic-v1.md) for
seeded results and behavioral metrics.

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
returned separately for auditing.

## Stage 1 neural PPO smoke

Torch remains an optional dependency. Install the locked neural environment
and run the deterministic mechanics smoke with:

```bash
uv sync --locked --extra neural
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke
```

The default runs exactly two PPO updates with 16 complete live-A, three-player
games per update on CPU. The learner seat rotates while the other seats use
the frozen balanced and passive heuristic bots. It verifies deterministic
planning, public-history encoding, legal-action masking, gamma-one GAE, one
epoch of clipped PPO, gradient clipping, and checkpoint replay. It is a
mechanics test, not evidence of playing strength.

The output contains `smoke-result.json` and an inference-only checkpoint:

```text
checkpoint/
  manifest.json
  model.pt
```

Stage 1 checkpoints intentionally omit optimizer and random-number-generator
state, so runs cannot be resumed. Stage 1 also has no opponent league, varied
ruleset curriculum, or registered live neural bot wrapper.

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
4. ✅ Implement and validate Bayesian value-heuristic bot strategies.
5. ✅ Run seeded round-robin evaluations and compare strategy behavior.
6. Build and locally train an imperfect-information neural policy through
   self-play.

## License

Garboid PocketRocks is available under the [MIT License](LICENSE).
