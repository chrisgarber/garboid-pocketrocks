# Bayesian Heuristic Valuation and Opponent Bots Design

**Date:** 2026-07-28
**Status:** Approved by delegated design review

## Purpose

Build an auditable, live-compatible heuristic engine that estimates the value
of the current PocketRocks action from the official SDK context. Validate the
engine independently, then use it to create aggressive, balanced, and passive
opponents and compare them in the deterministic Monte Carlo simulator.

This is the next milestone after the simulator and RL environment foundation.
Neural training remains a separate design and implementation cycle.

## Sources of Truth

The source priority remains:

1. the current live PocketRocks rules;
2. the pinned PocketRocks Python SDK and its public `DecisionContext`;
3. the deterministic simulator's SDK-conformance tests;
4. the competition repository only where it agrees with the first three.

The heuristic may consume only `DecisionContext`, `RulesetKnowledge`, and
public SDK objective definitions. It must not consume `GameState`, replay
seeds, hidden deck order, opponent hands, simulator events, or other
simulator-only data.

## Goals

- Estimate terminal per-card resource value from all information available to
  a live bot:
  - configured resource counts;
  - all publicly won resources;
  - all publicly revealed private cards;
  - the bot's own private hand;
  - the resource cards offered in the current auction.
- Treat the remaining unknown cards as a finite population and compute the
  exact hypergeometric expectation for opponents' unrevealed private cards.
- Add full value for objectives completed by the current bundle and a smaller,
  nonlinear value for useful partial progress.
- Preserve the exact terminal-dollar accounting of cash, auction bids, loans,
  and investments while adding a time-sensitive option value for liquid cash.
- Produce an auditable valuation breakdown and an integer legal bid.
- Use one valuation implementation for aggressive, balanced, and passive
  profiles.
- Sanity-check the evaluator before constructing bot wrappers.
- Add behavioral Monte Carlo metrics that prove the profiles are distinct.
- Run seeded tournaments across charts A-E and 3-5 players without faults.

## Non-Goals

This milestone does not:

- infer a strategically selected-reveal model from data;
- maintain mutable per-game history in a live brain;
- use simulator-only hidden information;
- implement full opponent bid distributions or equilibrium bidding;
- run determinized rollouts in every live decision;
- guarantee that one hand-tuned profile is strongest;
- train a neural policy.

## Considered Approaches

### Flat weighted score

Assign a fixed value to each chart bucket, objective step, loan, and
investment. This is easy to implement but fails to use the known resource-deck
composition and handles decreasing or non-monotone charts poorly.

### Bayesian marginal utility

Use an exact finite-population posterior for final reveal counts, nonlinear
objective progress, and dollar-equivalent action utility at each legal bid.
This is deterministic, explainable, fast at PocketRocks scale, and testable
with strong invariants.

This is the selected approach.

### Determinized rollouts

Sample hidden hands and future decks, then simulate candidate bids against
assumed opponent policies. This may later calibrate or audit the heuristic, but
it is slower, noisy, and risks strategy fusion if treated as perfect
information. It is not part of this milestone.

## Live Information Boundary

The SDK provides:

- the current action and resource cards on offer;
- every seat's cash, won-resource counts, revealed-card counts, and claimed
  objectives;
- the bot's ordered private hand;
- the tiebreak marker and legal maximum bid;
- the active objectives and value chart;
- ruleset resource/action counts and setup values through
  `RulesetKnowledge`.

The SDK does not provide:

- opponent hands or either deck order;
- turn index or consumed action-card history;
- historical bids;
- outstanding loan and investment positions;
- unoffered buffered resource cards;
- a stable live game identifier.

The evaluator must therefore be stateless. Game progress uses the remaining
biddable-resource fraction as a live-compatible horizon proxy.

## Package Architecture

```text
src/garboid_pocketrocks/heuristics/
  belief.py       finite-population resource belief
  objectives.py   requirement vectors and progress
  cash.py         horizon and cash option value
  valuation.py    action utility and bid evaluation

src/garboid_pocketrocks/bots/
  heuristic.py    thin BotBrain and live wrappers

DecisionContext + RulesetKnowledge
              |
              v
       HeuristicValuator
         /     |      \
    belief  objective  cash
         \     |      /
          BidEvaluation
                |
                v
       HeuristicBotBrain
```

Pure domain components are kept independent of bot IDs, the SDK runtime, and
the simulator. `HeuristicBotBrain` only selects a bid or reveal from their
results.

## Public Types

```python
@dataclass(frozen=True, slots=True)
class HeuristicProfile:
    name: str
    liquidity_strength: float
    objective_progress_weight: float
    bid_shading: float


@dataclass(frozen=True, slots=True)
class SuitBelief:
    suit: Suit
    known_terminal_reveals: int
    unseen_suit_count: int
    unseen_population: int
    opponent_hidden_slots: int
    terminal_price_pmf: tuple[float, ...]
    expected_terminal_price: float


@dataclass(frozen=True, slots=True)
class BeliefState:
    suits: tuple[SuitBelief, ...]
    expected_future_biddable_counts: tuple[float, ...]
    normalized_horizon: float


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    objective_id: int
    completion_value: float
    progress_value: float


@dataclass(frozen=True, slots=True)
class ValueBreakdown:
    resource: float
    objective_completion: float
    objective_progress: float
    terminal_cash: float
    liquidity: float
    total: float


@dataclass(frozen=True, slots=True)
class BidPoint:
    bid: int
    win_delta: float
    breakdown: ValueBreakdown


@dataclass(frozen=True, slots=True)
class BidEvaluation:
    belief: BeliefState
    points: tuple[BidPoint, ...]
    reservation_bid: int
    chosen_bid: int
```

The concrete API is:

```python
class HeuristicValuator:
    def __init__(self, profile: HeuristicProfile) -> None: ...

    def evaluate_bid(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BidEvaluation: ...

    def choose_reveal(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> int: ...
```

All numeric outputs must be finite. Invalid or internally inconsistent
contexts raise `HeuristicInputError` in the evaluator. The brain catches only
that error and safely passes; unexpected programming errors remain visible.

`HeuristicProfile.__post_init__` requires every coefficient to be finite,
`liquidity_strength >= 0`, `0 <= objective_progress_weight <= 1`, and
`0 <= bid_shading <= 1`.

`terminal_price_pmf` always contains exactly six probabilities indexed by the
capped final-reveal bucket `0..5`. It is not indexed by distinct price because
one chart may repeat a price in multiple buckets.

## Bayesian Resource Belief

Let:

- `D_s` be the configured initial count of suit `s`;
- `W_s` be the total publicly won count of suit `s`;
- `R_s` be the total publicly revealed private count of suit `s`;
- `H_s` be the count of suit `s` in this bot's current private hand;
- `Y_s` be the count of suit `s` in the current offered bundle;
- `h0` be initial private cards per player;
- `P` be player count.

Define `Y_s` as nonzero offered resources only when `decision_kind` is
`submitBid` and the action is Auction 1 or Auction 2. It is zero for financial
bids and every reveal request. The unseen suit population is:

```text
U_s = D_s - W_s - R_s - H_s - Y_s
```

The number of still-hidden opponent private cards is known exactly:

```text
M = sum(
    h0 - sum(revealed_info_counts_by_seat[seat])
    for seat != bot_seat
)
```

The unknown opponents' aggregate suit counts are modeled as a multivariate
hypergeometric sample of size `M` from population `U`. Each suit's expected
price needs only its univariate hypergeometric marginal:

```text
X_s ~ Hypergeometric(sum(U), U_s, M)
T_s = R_s + H_s + X_s
price_s = E[value_chart[min(T_s, 5)]]
```

`price_s` is the expected terminal value of one newly won resource card of
suit `s`. A two-card lot's base resource value is the sum of its card values;
two cards of the same suit contribute twice the same unit price.

The offered bundle is subtracted only for a bidding context. During the reveal
phase it has already entered `won_resource_counts_by_seat`, so subtracting it
again would double-count it.

The central conservation identity is:

```text
sum(U) - M = B - W - q
```

where `q = sum(Y)`. The left side is the unknown pool not occupying opponent
private hands; the right side is the remaining biddable pool after removing
the current offered lot. Construction fails if this identity does not hold.

The expected future biddable supply is:

```text
E[future_biddable_s] = U_s * (1 - M / sum(U))
```

Zero-population cases resolve deterministically rather than dividing by zero.
Every PMF is computed with exact integer combinations and normalized once at
the boundary.

### Belief invariants

- no unseen count is negative;
- `0 <= M <= sum(U)`;
- PMF entries are finite and nonnegative and sum to one;
- each expected terminal price is within the chart's minimum and maximum;
- moving one known card from this bot's hand to its public reveals leaves its
  own terminal price belief unchanged;
- hidden engine states with the same SDK context produce identical beliefs.
- conservation holds separately for auction bidding, financial bidding, and
  the post-auction reveal context that preserves `current_resource_ids`.

## Game-Progress Horizon

Initial biddable resources are:

```text
B = sum(resource_counts) - P * h0
```

Let `W` be the total number of resources already won. Let `q` be the current
lot size for Auction 1 or Auction 2, otherwise zero. The post-action normalized
horizon is:

```text
tau = clamp((B - W - q) / B, 0, 1)
```

This is not an exact remaining-turn count because the SDK omits consumed
action history. It is a monotone, stateless proxy: financial actions early in
the game have a larger `tau`, and every auction moves it toward zero.

## Objective Valuation

Every objective is converted to minimal requirement vectors:

- `same2` and `same3`: one vector per suit;
- `different3` and `different4`: one vector per combination of suits;
- `twoPairs4`: one vector per pair of suits with two of each;
- suit-specific objectives: their SDK requirement vector.

For holdings `c` and a requirement vector `r`:

```text
distance(c, r) = sum(max(required - owned, 0))
progress(c, r) = 1 - distance(c, r) / sum(r)
```

An objective's progress is the maximum across its requirement vectors.
Completion predicates must be exhaustively conformance-tested against the
engine for all 30 SDK objectives.

Objectives already owned by any seat are worth zero. For every still-unowned
active objective:

- if the offered bundle newly completes it, add its full SDK payout;
- otherwise add only the positive change in shaped progress:

```text
progress_value =
    payout
    * objective_progress_weight
    * max(after_progress**2 - before_progress**2, 0)
    * tau
    * contest_factor
```

`contest_factor` is:

```text
1 / (1 + number of opponents whose public distance <= our post-win distance)
```

The square makes one-card-away progress more valuable than an equal amount of
initial progress. Multiplication by `tau` makes incomplete progress decay as
opportunities disappear. Immediate completion always receives the full payout
and is not discounted by time or competition. Multiple objectives completed
by one bundle are each counted once, matching the engine.

For a completed objective, `completion_value == payout` and
`progress_value == 0`; the `otherwise` branch prevents progress shaping from
being added to the same objective.

The shaped progress term is deliberately labeled heuristic option value, not
literal expected money.

## Cash and Time Value

One dollar of cash always retains exactly one terminal dollar. Liquid cash also
has temporary option value because it can win later auctions:

```text
cash_utility(c, tau) = c + option(c, tau)
option(c, tau) =
    liquidity_strength
    * tau
    * kappa
    * log1p(c / kappa)
kappa = max(starting_cash / 2, 1)
```

This option function is increasing and concave. Cash scarcity therefore
matters more than the same absolute change at high cash. At `tau == 0`, option
value is zero and only terminal dollars remain.

For current cash `c`, bid `b`, resource/objective gross value `G`, loan
principal `P`, and investment payout `K`:

```text
auction_delta(b) =
    G - b + option(c - b, tau) - option(c, tau)

loan_delta(b) =
    -b + option(c + P - b, tau) - option(c, tau)

investment_delta(b) =
    K + option(c - b, tau) - option(c, tau)
```

These formulas preserve scoring:

- an auction bid is permanently spent and also reduces temporary liquidity;
- a loan's principal is repaid, its bid is permanently spent, and its value is
  only the temporary liquidity it creates;
- an investment bid returns at scoring, so it costs only temporary liquidity
  while the fixed payout is genuine terminal profit.

At zero horizon, a loan is worth `-bid` and an investment is worth its payout
regardless of lock size. Loans therefore naturally become worthless late;
investment lock costs naturally disappear late.

Every legal integer bid from zero through `legal_max_amount` is evaluated.
There is no continuous approximation or rounding ambiguity.

## Reservation and Bid Selection

The reservation bid is the largest legal bid whose `win_delta` is
nonnegative. It is an economic upper bound, not the bid submitted to a
first-price auction.

The initial opponent bots use deterministic bid shading:

```text
chosen_bid = floor(reservation_bid * (1 - bid_shading))
```

The chosen bid is clamped to the legal range and never exceeds the reservation
bid. A chosen bid of zero becomes `BotDecision.pass_turn()`.

The engine's exact clockwise tiebreak remains authoritative. The initial
profiles do not add randomized bid jitter or pretend to know opponent bid
distributions. A later calibration may enumerate tie-aware win probabilities
from observed self-play data.

## Reveal Selection

Reveal order does not change final resource prices because every private card
is revealed by scoring. Its only strategic effect is the information exposed
to opponents.

For each distinct candidate suit, the evaluator compares the public
terminal-price expectation before and after revealing that suit, from an
observer perspective that does not know this bot's hand. In the observer's
pre-reveal posterior, this bot's remaining hand is part of the total hidden
private slots. In the post-reveal posterior, one card of the candidate suit is
removed from the unseen population, the hidden-slot count decreases by one,
and the known reveal count for that suit increases by one.

Because conditioning one suit changes the finite population for every suit,
the information influence proxy sums all cross-suit price changes:

```text
sum(
    opponent_won_count[seat, suit]
    * (
        post_reveal_public_expected_price[suit]
        - pre_reveal_public_expected_price[suit]
    )
    for seat != bot_seat
    for suit in all_suits
)
```

Choose the card with the smallest opponent benefit. Equal suit scores choose
the lowest hand index, making the policy deterministic. The initial bid
valuation assigns this forced reveal a monetary cost of zero; converting the
proxy to dollars before calibration would be false precision.

## Bot Profiles and Identities

All profiles use the same evaluator and differ only through immutable
configuration:

| Profile | `liquidity_strength` | `objective_progress_weight` | `bid_shading` |
| --- | ---: | ---: | ---: |
| aggressive | 0.75 | 0.25 | 0.05 |
| balanced | 0.40 | 0.20 | 0.25 |
| passive | 0.15 | 0.15 | 0.50 |

The table is materialized as the module-level constants
`AGGRESSIVE_PROFILE`, `BALANCED_PROFILE`, and `PASSIVE_PROFILE`.

Interpretation:

- Aggressive values early liquidity highly, takes small bid margins, pursues
  objective options more strongly, and should favor loans while treating
  early investment lockup as expensive.
- Balanced uses middle settings.
- Passive values temporary liquidity less, demands a large bid margin, and
  waits for inexpensive resource lots and investments.

Concrete classes are:

```text
AggressiveHeuristicBrain / AggressiveHeuristicBot
BalancedHeuristicBrain   / BalancedHeuristicBot
PassiveHeuristicBrain    / PassiveHeuristicBot
```

Bot IDs are committed public class constants. Until live bot registrations
exist, the syntactically valid development-only IDs are:

```text
aggressive: bot_00000000-0000-4000-8000-00000000000a
balanced:   bot_00000000-0000-4000-8000-00000000000b
passive:    bot_00000000-0000-4000-8000-00000000000c
```

These IDs are not registered and the wrappers must not be started against the
live service until they are replaced. The wrappers do not add a separate
runtime rejection—the server remains authoritative—but no live console script
is exposed for them and the README labels them simulator-only. Replacing an ID
does not change its brain or simulator specification.

Each concrete bot has a top-level class and importable `build_brain` classmethod
that constructs its corresponding top-level brain class. Profiles are passed
as module constants; no closure, lambda, or `functools.partial` is stored in a
`BotSpec`. Spawn-worker picklability is a required test.

The CLI registry exposes `aggressive`, `balanced`, and `passive` alongside
`random`.

## Behavioral Monte Carlo Metrics

Existing score/rank/seat/ruleset statistics remain. Add per-bot behavior
statistics derived from replay decisions and domain events:

- bidding decisions and passes;
- mean nonzero bid;
- wins by all six action IDs;
- resource cards won;
- objectives claimed;
- reveal choices.

Metrics must aggregate by bot identity without changing deterministic game
planning or replay.

The exact definitions are:

- `bidding_requests`: replay decisions whose action kind is `pass` or
  `submitBid`;
- `passes`: bidding `pass` decisions plus `submitBid(0)`;
- `pass_rate`: `passes / bidding_requests`, or `0.0` when there are no bidding
  requests;
- `nonzero_bids`: positive `submitBid` values;
- `mean_nonzero_bid`: the arithmetic mean of `nonzero_bids`, or `0.0` for an
  empty sample;
- `reveal_choices`: `selectInfoToReveal` decisions, counted by selected hand
  index;
- `wins_by_action`: `AUCTION_RESOLVED` events counted by `event.seat` and
  `event.action_id`;
- `resource_cards_won`: the length of `resource_ids` in
  `RESOURCES_AWARDED` events;
- `objectives_claimed`: the length of `objective_ids` in
  `OBJECTIVE_CLAIMED` events.

Decision metrics are read from replay decisions by seat. Win and acquisition
metrics are read from events by seat. The two streams do not need positional
joining for these aggregate definitions.

## Test Strategy

Implementation follows red-green-refactor TDD.

### Belief tests

- exact deterministic value with no opponent hidden cards;
- PMF normalization and support;
- constant-chart invariance;
- own hand-to-reveal metamorphic invariance;
- charts A-E, including decreasing and non-monotone chart E;
- Hypothesis card-conservation properties across legal 3-5-player contexts;
- rejection of inconsistent known counts.

### Objective tests

- requirement vectors match engine completion for all 30 objectives and
  generated holdings;
- already-owned objectives contribute zero;
- immediate single and multiple completions use exact payouts once;
- completion explicitly reports full payout and zero progress value;
- incomplete progress is bounded below payout;
- near-complete progress exceeds initial progress;
- incomplete progress decreases with horizon and stronger competition.

### Cash/action tests

- cash option value is finite, increasing, and concave;
- option value is zero at zero horizon;
- each action curve matches the exact accounting equations;
- loan benefit decreases as `tau` decreases, while investment value increases
  because its lock cost decreases;
- resource, loan, and investment curves are nonincreasing in bid;
- every reservation and selected bid is legal.

### Brain tests

- valuation is identical for identical contexts and profiles;
- reveal ties select the lowest index;
- all decisions pass `DecisionContext.validate`;
- aggressive, balanced, and passive wrappers expose distinct names and IDs;
- every `BotSpec` is picklable and usable by spawned workers;
- expected profile ordering is tested on curated resource, loan, and investment
  contexts without asserting universal win-rate ordering.

### Integration and benchmark tests

- full games complete without faults for all profiles, charts A-E, and 3-5
  players;
- serial and parallel Monte Carlo results are identical;
- a fixed 1,000-game three-player tournament rotates seats fairly;
- at least two profiles differ by ten percentage points in pass rate and by
  one dollar in mean nonzero bid;
- aggressive wins more loans than passive;
- no hard assertion is made that one profile must win most often.

The benchmark report records the fixed seed, ruleset, games per second, score
metrics, and behavioral metrics. If the separation criteria fail, profile
parameters are tuned from the report and the fixed benchmark becomes a
regression fixture.

## Documentation and Commands

README examples will include:

```bash
uv run garboid-simulate \
  --bots aggressive,balanced,passive \
  --games 1000 \
  --players 3 \
  --ruleset live-A \
  --seed 42 \
  --workers 4
```

The evaluator's breakdown is also exposed through a small Python API example
so a future bot or training tool can inspect why a bid was selected.

## Risks and Follow-Ups

- Opponents choose which private cards to reveal strategically, so the
  exchangeability posterior is an approximation. It is the maximum-entropy
  baseline and can later be calibrated from self-play.
- The resource horizon is not an exact remaining-turn count because the live
  context omits action history.
- Outstanding loan and investment positions are not visible, limiting
  opponent final-score estimates.
- Objective progress is utility shaping rather than literal expected money.
- Deterministic bid shading can be exploitable. Recorded bid metrics can later
  fit tie-aware opponent bid distributions.
- Determinized rollouts may later serve as an offline teacher, but they must
  sample information sets and simultaneous opponent bids without leaking true
  hidden state.

## Delegated Design Review

The user requested that design questions be resolved through subagents instead
of interrupting implementation.

- **Critical design:** Sol, extra-high reasoning, recommended exact Bayesian
  marginal utility and identified the live-information and accounting
  constraints.
- **Validation:** Terra, medium reasoning, validated dimensional action
  accounting, profile separation criteria, and missing behavioral metrics.
- **Decision:** adopt Bayesian marginal utility, keep nonlinear objective
  progress because it is an explicit product requirement, and defer learned
  opponent models and rollout search.
