# Deterministic Simulator, Monte Carlo Runner, and RL Environments Design

**Date:** 2026-07-28
**Status:** Approved for written review

## Purpose

Build a deterministic PocketRocks game engine that supports the current live
rules and configurable training rulesets. Use that engine for synchronous bot
matches, reproducible Monte Carlo evaluation, replay, a standard multi-agent RL
environment, and a convenient single-agent training environment.

This milestone also separates each bot's synchronous decision logic from the
official SDK's asynchronous runtime. A bot becomes a public identity plus a
brain class. The live bot delegates SDK requests to the same brain that local
simulation uses.

## Goals

- Model the current live PocketRocks rules exactly as the default preset.
- Make deck composition, action counts, value chart, setup distribution, and
  objective selection configurable through validated rulesets.
- Produce the official SDK's `DecisionContext` and consume its `BotDecision`
  type at every policy boundary.
- Keep the game engine deterministic, independent of bots, and stepwise.
- Preserve a complete event log and support exact replay.
- Run many seeded games across fixed or sampled rulesets and fair seat
  assignments.
- Expose multi-agent and single-agent RL interfaces over the same transitions.
- Support intermediate financial rewards without leaking hidden information.
- Keep observations and actions suitable for later neural-network training.

## Non-Goals

This milestone does not:

- implement a value-heuristic strategy;
- choose or train a neural-network architecture;
- add a reinforcement-learning algorithm;
- emulate the SDK transport or wire protocol during simulation;
- reproduce the competition repository's stale rule differences;
- support arbitrary alternative meanings for actions, tie-breaking, or scoring;
- add a graphical simulator or interactive renderer;
- require the competition repository at runtime.

Alternative numeric and composition settings are supported. Alternative core
game semantics require a future design.

## Sources of Truth

The source priority is:

1. the current [live PocketRocks rules](https://pocketrocks.xyz/rules);
2. the pinned [PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk/tree/597857446d47ac0890609a4767cad561578a2519);
3. observed live behavior and committed conformance fixtures;
4. the [competition repository](https://github.com/jaiparera/pocketrockscompetition)
   only where it agrees with the first three sources.

This ordering is necessary because the competition engine currently differs
from the live rules. In particular, it uses a 25-card action deck instead of
the live 30-card deck and caps loan bids at current cash instead of current
cash plus principal.

## Architecture

```text
                        public identity + brain factory
                                      |
Live server -> PocketRocksFastBot -> BotBrain -> SDK BotDecision
                                      ^
                                      |
Ruleset -> GameEngine -> DecisionContext batch
                 |                |
                 |                +-> MatchRunner -> BotBrain factories
                 |                +-> Multi-agent RL adapter
                 |                +-> Single-agent RL adapter
                 |
                 +-> immutable state + domain events + replay
```

The engine is the only implementation of game rules. Runners and environments
coordinate decisions but never duplicate transitions or scoring.

## Bot Identity and Brain Boundary

### Synchronous brain

The policy contract is synchronous and returns official SDK decisions:

```python
class BotBrain(Protocol):
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision: ...
```

A brain may own immutable model parameters or per-instance state such as an
RNG. Simulations create a fresh brain for every seat in every game so state
cannot leak between matches.

The live SDK can serve multiple games from one bot process. Brains intended for
live use therefore cannot assume that mutable state belongs to one game unless
a future SDK supplies a stable game identifier. Feed-forward decisions and
shared, read-only model inference are safe. The random brain's RNG is allowed
to span live requests; deterministic per-game random behavior is guaranteed
only by simulation's fresh brain instances.

### SDK bridge

`PocketRocksFastBot` subclasses the official `PocketRocksBot` and provides:

- public class constants `BOT_ID` and `BOT_NAME`;
- a brain type or overridable brain factory;
- `choose_decision_sync(context) -> BotDecision`;
- the required async `choose_decision(context)`, which immediately delegates
  to the synchronous method;
- constructor overrides for tests.

Each concrete bot binds identity and policy construction:

```python
class RandomBot(PocketRocksFastBot):
    BOT_ID = "bot_e0e2c541-1615-4f47-983c-224e7d888d89"
    BOT_NAME = "random"
    BRAIN_CLASS = RandomBotBrain
```

Bot IDs are public source metadata. API keys remain secret environment
configuration. `RANDOM_BOT_ID` is removed from `.env.example` and the local
`.env`; the live wrapper passes `BOT_ID` to the SDK. Tests may override it.

### Bot specifications

`BotSpec` is the runner-facing immutable description of a bot:

- stable name;
- public bot ID;
- brain factory.

`BotSpec.from_bot_class(RandomBot)` derives this metadata without constructing
an SDK runtime. The simulator instantiates brains, never live SDK bot objects.

## Rulesets

### Ruleset data

`Ruleset` is immutable, serializable, and validated. It contains:

- a stable name;
- five resource-card counts, one per SDK suit;
- six action-card counts, one per SDK action;
- starting cash for each supported player count;
- private-card count for each supported player count;
- one six-bucket value chart, with the final bucket representing five or more;
- whether objectives are enabled;
- a unique objective-ID pool;
- the number of active objectives selected from that pool.

Player count and seed are match parameters because the same ruleset may support
3-, 4-, and 5-player games.

Validation requires:

- exactly five nonnegative resource counts;
- exactly six nonnegative action counts;
- supported player counts between 3 and 5;
- positive starting cash and nonnegative private-card counts;
- enough resource cards for setup plus at least one biddable card;
- enough configured auction capacity to exhaust all biddable resources;
- a six-value chart;
- known, unique objective IDs;
- an active-objective count within the configured pool.

An invalid ruleset fails before setup. The engine never reshuffles an
insufficient action deck to conceal a configuration error.

### Live preset

`LIVE_RULESET` contains:

- six resource cards in each of five suits, 30 total;
- action counts:
  - Auction 1: 12;
  - Auction 2: 8;
  - Loan 10: 3;
  - Loan 20: 2;
  - Invest 5: 3;
  - Invest 10: 2;
- starting cash:
  - 3 players: $30;
  - 4 players: $25;
  - 5 players: $20;
- private cards per player:
  - 3 players: 5;
  - 4 players: 4;
  - 5 players: 3;
- value chart A by default: `(0, 4, 8, 12, 16, 20)`;
- objectives enabled, with four unique active objectives sampled without
  replacement from the SDK's 30 unique objective definitions.

The other current live charts are named presets:

- B: `(20, 16, 12, 8, 4, 0)`;
- C: `(0, 2, 5, 9, 14, 20)`;
- D: `(20, 18, 15, 11, 6, 0)`;
- E: `(0, 4, 10, 18, 6, 0)`.

### Ruleset sampling and public knowledge

`RulesetSampler` supports:

- one exact ruleset;
- a weighted set of named rulesets;
- a seeded distribution over bounded numeric and composition fields.

Every match records the fully resolved ruleset. Sampling depends only on the
root seed and game index, not execution order or worker count.
Bounded distributions generate valid rulesets by construction or fail during
sampler validation rather than retrying unpredictably during a run.

`RulesetKnowledge` contains pregame-public settings but no shuffled deck order,
dealt hidden cards, or future randomness. It is supplied separately from
`DecisionContext` because the SDK does not carry fixed resource or action deck
composition. The live wrapper uses `LIVE_RULESET` knowledge. Training may
optionally mask selected knowledge fields for domain-randomization experiments,
but live-fidelity mode exposes all public rules.

## Engine Model

### Immutable state

Frozen data types represent:

- cards and actions with deterministic IDs;
- per-seat cash, won resources, private hand, revealed information, loans,
  investments, and objectives;
- shuffled resource and action decks;
- currently visible resources and action;
- active objectives;
- priority-marker seat;
- turn and phase;
- accumulated public history.

The complete state contains hidden information and is never passed directly to
a policy.

### Pure transition API

The core API is functionally stepwise:

```python
GameEngine.start(ruleset, player_count, seed) -> EngineTransition
GameEngine.step(state, decisions_by_seat) -> EngineTransition
```

`EngineTransition` contains:

- the new immutable state;
- emitted domain events;
- the next pending decision batch;
- termination status;
- the final result when terminated.

All setup randomness is resolved into shuffled immutable sequences. Applying a
decision transition itself requires no global RNG.

### Decision phases

The engine has three phases:

1. **Bidding:** every seat receives an SDK-compatible `DecisionContext` for the
   same pre-bid state and submits one `BotDecision`.
2. **Reveal:** when the winner still has a private card, only that seat receives
   a reveal context.
3. **Terminal:** remaining private cards are revealed and final scores are
   calculated.

The engine requires exactly the seats active in the current decision batch.
Missing, duplicate, or unexpected seats are errors.

### SDK-compatible contexts

Contexts populate every public SDK field from engine state:

- player count and starting cash;
- selected value chart and active objective IDs;
- current action and offered resource IDs for auctions, zero-padded to two
  entries; financial actions expose `(0, 0)`, matching the SDK;
- cash, won-resource counts, revealed-information counts, and owned objectives
  by seat;
- bot seat and ordered private hand;
- legal maximum bid and revealable count;
- deterministic request ID.

Synthetic timing fields use a non-expiring sentinel. Context conformance ignores
only request ID and timing fields. `metadata` remains empty so bots cannot
depend on simulator-only information; public ruleset knowledge is the separate
brain/environment input.

## Live Rule Transitions

### Setup

- Build and shuffle resource and action decks from the ruleset.
- Deal the configured private cards to each seat.
- Reveal up to two biddable resources.
- Select active objectives without replacement.
- Select the initial priority marker uniformly from occupied seats.
- Open the first valid action.

### Bidding and priority

- `pass` and `submitBid(0)` both have auction value zero.
- Auction and investment bids range from zero through current cash.
- Loan bids range through current cash plus that loan's principal.
- The highest bid wins.
- For a tie, scan seat order beginning immediately after the current priority
  marker and wrap; the first tied seat wins.
- An all-zero round uses the same tie rule and awards the action for zero.
- The winner becomes the next priority marker.

Wrong decision kinds, negative bids, and bids above the context maximum are
illegal.

### Actions

- **Auction 1:** deduct the bid and award the first visible resource.
- **Auction 2:** deduct the bid and award both visible resources, or the single
  remaining resource.
- **Loan 10/20:** deduct the bid, add principal to cash, and record principal
  for final repayment.
- **Invest 5/10:** deduct and lock the bid, then record the locked amount and
  fixed final payout.

After a resource award, claim every still-unowned active objective newly
satisfied by the winner. Claims do not consume resources and multiple
objectives may be claimed on one transition.

### Reveals

After every win, the winner reveals one private card when any remain. A valid
selection identifies an index in the ordered hand. A reveal-phase `pass` is
legal under the SDK and triggers deterministic auto-reveal of the first card,
matching the server's documented timeout fallback without introducing new
randomness.

### End and scoring

After the biddable resource deck is exhausted, finish the current round,
perform its winner reveal, reveal all remaining private cards, and score:

```text
final money =
    cash
  + sum(won resource count × final per-suit chart value)
  + objective payouts
  + returned investment locks and investment payouts
  - loan principals
```

The final value-display count chooses the chart bucket, capped at the final
five-or-more bucket.

Equal final money is a genuine tie. Results use shared competition ranks;
presentation may use seat order for stable sorting but does not convert a tie
into a win.

## Domain Events and Replay

Events describe setup and every externally meaningful transition, including:

- game setup;
- turn opened;
- decisions submitted;
- auction resolved;
- resources awarded;
- loan or investment acquired;
- objective claimed;
- information revealed or auto-revealed;
- game ended and scores calculated;
- bot fault and fallback, when a forgiving runner is explicitly enabled.

A replay contains schema version, resolved ruleset, player count, root/game
seed, bot names, submitted decisions, and events. Reapplying recorded decisions
must reproduce the event stream and final state. Replay files contain no API
keys or live credentials.

## Match and Monte Carlo Runners

### Match runner

`MatchRunner`:

- accepts a ruleset, player count, seed, and ordered `BotSpec` lineup;
- creates a fresh brain for every seat;
- synchronously resolves every decision batch;
- returns a structured match result and replay.

The core engine is strict. Runner fault modes are:

- `raise` (default for development and tests);
- `record_and_pass` for long evaluations, which records the exception or
  illegal decision and substitutes pass.

### Monte Carlo runner

`MonteCarloRunner`:

- accepts bot specs, game count, player-count distribution, ruleset sampler,
  root seed, and worker count;
- samples lineups when more bots exist than seats;
- balances or randomizes seat assignments reproducibly;
- derives game and brain seeds by game index;
- optionally uses local process workers;
- combines results in game-index order.

Worker scheduling cannot change sampled games or results.
Process workers require importable brain classes and serializable bot specs;
worker count one remains available for local closures and experimental brains.

Metrics include:

- games played;
- outright wins and first-place ties;
- placement distribution and mean rank;
- final-money mean, median, spread, and quantiles;
- score margin from first place;
- per-seat and per-ruleset breakdowns;
- decision and bot-fault counts.

A console command runs named lineups and emits a human-readable table or
structured JSON. Large replays and reports default to ignored artifact paths.

## RL Environments

### Shared observation encoder

One encoder converts `DecisionContext` plus `RulesetKnowledge` to bounded numeric
features. It includes:

- active/inactive phase and decision kind;
- player count, bot seat, starting cash, and current cash;
- current action and visible resources;
- priority seat;
- padded per-seat cash, won-resource, revealed-information, and objective
  ownership arrays;
- ordered private hand and length;
- value chart and active-objective mask;
- public ruleset features such as initial resource and action counts and setup
  distribution;
- action mask.

Arrays pad to configured maxima for five seats, five suits, 30 known objectives,
the ruleset sampler's maximum hand size, and its maximum legal bid.
Environment construction validates that these bounds cover the sampler's
entire declared support before the first reset. Hidden opponent cards, deck
order, and unobservable random state are excluded.

### Action encoding

For adapter bounds `max_bid` and `max_hand_size`, the fixed discrete space is:

- `0`: pass;
- `1..max_bid`: submit that positive bid;
- `max_bid + 1 .. max_bid + max_hand_size`: reveal indices
  `0..max_hand_size - 1`.

The mask enables pass plus legal bids during bidding, and pass plus existing
hand indices during reveal. Encoding and decoding round-trip through
`BotDecision`.

### Multi-agent environment

The standard multi-agent adapter follows PettingZoo's AEC contract because
PocketRocks mixes simultaneous sealed bidding with a winner-only reveal:

- during bidding it cycles through seats and stores actions without changing
  observable game state;
- after the final seat acts, it submits the joint batch to the engine;
- during reveal, only the winner acts;
- no policy observes another seat's sealed bid before resolution.

The adapter exposes standard spaces, masks, rewards, terminations, truncations,
and per-agent `info`, and passes PettingZoo's API test.

### Single-agent environment

The Gymnasium wrapper controls one learner seat and accepts opponent brain
factories. It:

- randomizes or fixes the learner seat;
- returns only the learner's encoded observation;
- asks opponent brains for their sealed bids;
- automatically resolves opponent-only reveal phases;
- accumulates rewards across internal opponent transitions;
- returns at the learner's next decision or game end;
- passes Gymnasium's environment checker.

Training one policy against frozen opponents is the initial workflow. Opponent
pools may later include random brains, heuristic brains, and frozen snapshots
of earlier learned policies. The environment remains multi-agent underneath, so
self-play does not require redesign.

## Rewards

### Default accounting potential

For each seat:

```text
public potential =
    cash
  + locked investment amounts
  + investment payouts
  - loan principals
  + claimed objective payouts
```

After each engine transition, the default intermediate reward is the change in
this potential divided by starting cash.

Won-resource value is excluded until terminal scoring because its true value
depends on hidden information. Revealing it through immediate reward would give
the learner information unavailable to a live bot.

At terminal, final resource value enters the potential and supplies the
remaining final-money delta. The sum of default accounting rewards equals:

```text
(final money - starting cash) / starting cash
```

The terminal transition then adds a configurable win bonus. Seats tied for
first divide that bonus equally. Because final money already orders players
within a game, the bonus emphasizes winning without discarding useful signal
from strong losses.

### Configuration and diagnostics

`RewardConfig` controls:

- accounting-delta weight;
- win bonus;
- optional placement bonuses;
- optional public event bonuses;
- invalid-action penalty and fallback behavior.

Optional event shaping is disabled by default. Every returned `info` contains
the separate accounting, terminal-resource, placement, shaping, and penalty
components so reward behavior is auditable.

## Error Handling

Specific domain exceptions cover:

- invalid rulesets or incompatible RL bounds;
- invalid setup;
- wrong phase;
- missing or unexpected acting seats;
- illegal decision kind or value;
- replay divergence;
- brain construction or decision failure.

The engine never catches these errors or changes decisions. Forgiving behavior
belongs only to an explicitly configured match runner or RL adapter, which
records the original fault before substituting pass.

## Testing Strategy

Implementation follows test-driven development.

### Rule and transition tests

Focused tests cover:

- live-preset constants and ruleset validation;
- seeded deck construction, dealing, objectives, and initial priority;
- bid limits, pass, zero bids, all tie positions, and all-zero rounds;
- every action type and partial final Auction 2;
- immediate and multiple objective claims;
- selected, passed, and automatic reveals;
- termination and every scoring component;
- final-score ties.

### Property and invariant tests

Generated valid rulesets and decision sequences verify:

- every card remains in exactly one valid location;
- cash and paper movements reconcile with events;
- one objective has at most one owner;
- no awarded resource exceeds configured supply;
- legal contexts and masks agree;
- final score recomputation matches the engine;
- terminal games have no biddable resources or pending decisions.

### SDK conformance

Curated live-preset histories are built through both our engine and the SDK's
public `scenario(...)` helper. All meaningful `DecisionContext` fields must
match. Timing and generated request IDs are intentionally excluded.

Reference constants use the SDK's public `Suit`, `ActionId`, `OBJECTIVES`, and
objective-payout helpers rather than internal protocol modules.

### Brain and runner tests

Tests prove:

- `RandomBotBrain` retains seeded random behavior;
- the live async `RandomBot` wrapper and synchronous brain return equal
  decisions for equal inputs;
- brains are fresh per seat and match;
- replays reproduce events and scores exactly;
- Monte Carlo output is identical across repeated runs and worker counts;
- seat and ruleset breakdowns reconcile with aggregate totals;
- strict and forgiving fault modes behave as documented.

### RL contract tests

Tests cover:

- observation shapes, padding, dtypes, and bounds;
- action encode/decode and masks for every phase;
- no hidden-state fields in observations;
- public ruleset conditioning and optional masks;
- intermediate and terminal reward decomposition;
- randomized learner seating;
- automatic opponent advancement;
- Gymnasium's environment checker;
- PettingZoo's AEC API test.

### Quality and performance

The complete existing quality gate remains:

```text
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

A local benchmark reports games per second for 3-, 4-, and 5-player random
matches across multiple rulesets. CI does not assert a machine-dependent timing
threshold.

## Implementation Boundaries and Parallel Work

The engine contracts and SDK-context adapter are foundational and must land
first. After those interfaces are verified, implementation can safely split:

- one subagent refactors the live random bot into identity plus brain;
- one subagent builds replay and Monte Carlo runners;
- one subagent builds observation, reward, and RL adapters;
- the primary agent integrates, reviews, and runs cross-boundary verification.

Subagents work in separate files after the shared interfaces are fixed. The
implementation plan will define exact file ownership and merge checkpoints.

## Documentation

The README will document:

- live bot identity and API-key configuration;
- synchronous brains and SDK wrappers;
- running one deterministic match;
- running seeded Monte Carlo evaluations;
- fixed and sampled rulesets;
- replaying a match;
- using the multi-agent and single-agent environments;
- reward semantics and action masks;
- the distinction between simulator fidelity and competition-reference code.

## Acceptance Criteria

The milestone is complete when:

- `LIVE_RULESET` matches current live rules and all five value charts;
- configurable valid rulesets run for 3–5 players;
- `RandomBot` uses a static public ID and delegates to `RandomBotBrain`;
- one engine drives synchronous matches and both RL environments;
- engine contexts conform to SDK scenarios for curated live histories;
- replay, seed, and worker-count determinism tests pass;
- Monte Carlo reports fair seat and ruleset breakdowns;
- Gymnasium and PettingZoo contract checks pass;
- rewards reconcile to normalized final money plus configured bonuses without
  hidden-information leakage;
- all local quality checks and GitHub Actions pass;
- usage and architectural boundaries are documented.

## References

- [PocketRocks live rules](https://pocketrocks.xyz/rules)
- [PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk/tree/597857446d47ac0890609a4767cad561578a2519)
- [SDK `DecisionContext`](https://github.com/jaiparera/pocketrocks-python-sdk/blob/597857446d47ac0890609a4767cad561578a2519/src/pocketrocks/types.py)
- [SDK public reference helpers](https://github.com/jaiparera/pocketrocks-python-sdk/blob/597857446d47ac0890609a4767cad561578a2519/src/pocketrocks/reference.py)
- [PocketRocks competition reference](https://github.com/jaiparera/pocketrockscompetition)
- [Gymnasium custom environment guidance](https://gymnasium.farama.org/introduction/create_custom_env/)
- [PettingZoo AEC API](https://pettingzoo.farama.org/api/aec/)
- [RLlib multi-agent environments](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [Repository scaffold design](2026-07-28-repository-scaffold-design.md)
- [Random bot SDK design](2026-07-28-random-bot-sdk-design.md)
