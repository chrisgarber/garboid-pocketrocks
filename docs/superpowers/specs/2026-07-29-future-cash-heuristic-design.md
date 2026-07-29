# Future-Auction Cash Heuristic Design

**Date:** 2026-07-29
**Status:** Approved

## Purpose

Make the heuristic bots account explicitly for the opportunity cost of spending
cash before future resource auctions. Calibrate the three profiles so they
remain recognizable but competitively balanced: balanced should be the
strongest general strategy, while aggressive and passive should trail only
slightly.

## Evidence

The established 100,000-game tournament produced outright win rates of:

- aggressive: 16.856%;
- balanced: 15.497%;
- passive: 65.725%.

A separate 20,000-game replay diagnostic showed that the late collapse in
two-resource auction prices is primarily a cash constraint:

- the market winning bid falls from $14.97 in turns 1-5 to $5.17 after turn 12;
- estimated two-resource gross value falls only about 5-11%;
- mean late two-resource-auction cash is $3.97 aggressive, $3.92 balanced, and
  $7.64 passive;
- the legal cash cap binds 94.0% of aggressive, 95.2% of balanced, and 87.3% of
  passive late two-resource valuations.

The current logarithmic liquidity term recognizes that early cash is useful,
but does not model a reserve for future auction opportunities. Fixed bid
shading then causes aggressive and balanced bots to commit too much cash
early. Passive wins disproportionately because its 50% shading preserves cash
incidentally.

## Goals

- Give retained cash explicit marginal value while future resource auctions
  remain.
- Let the reserve target shrink monotonically to zero as biddable resources
  are consumed.
- Keep the reserve soft: a sufficiently valuable resource bundle or objective
  completion may justify crossing it.
- Apply the same opportunity cost to auctions, loans, and investments.
- Preserve the live-information boundary: use only `DecisionContext` and
  `RulesetKnowledge`.
- Preserve personality on equivalent contexts:
  - aggressive submits the highest bid;
  - balanced submits an intermediate bid;
  - passive submits the lowest bid.
- Preserve aggregate personality:
  - aggressive spends the most on early resource auctions;
  - passive retains the most terminal liquid cash;
  - balanced remains between the two.
- Tune overall outright win rates so balanced performs best and all profiles
  remain approximately even.

## Non-Goals

- Predict opponent bid distributions.
- Read simulator-only turn indices, deck order, replay history, or outstanding
  financial positions.
- Build a learned policy or determinized rollout search.
- Guarantee that balanced wins every individual value chart.
- Hard-code bids or outcomes to meet benchmark targets.

## Considered Approaches

### Soft future-cash reserve

Add a piecewise-linear utility for cash retained below a horizon-dependent
reserve target. The reserve changes the marginal economics of a bid without
making any bid illegal.

This is the selected approach. It is live-compatible, auditable, and directly
addresses the measured cash constraint.

### Hard spending cap

Subtract a reserve from the legal maximum. This guarantees retained cash but
prevents an aggressive bot from crossing the reserve for a valuable objective
completion. The discontinuity also makes profile calibration brittle.

### Opponent-aware auction model

Estimate future opposing bids from cash, profile assumptions, and public
history. This could improve first-price shading later, but historical bids and
stable opponent identity are not available in the current stateless live
boundary. It is outside this change.

## Cash Model

Keep the existing concave cash option value:

```text
log_value(cash) =
    liquidity_strength
    * horizon
    * kappa
    * log(1 + cash / kappa)

kappa = max(starting_cash / 2, 1)
```

Add a future-auction reserve target:

```text
reserve_target = starting_cash * horizon
```

The existing `BeliefState.normalized_horizon` is the post-action fraction of
biddable resources remaining. It is the best stateless, live-compatible proxy
for future resource-auction opportunity.

For a profile weight `future_cash_weight`, define:

```text
future_cash_value(cash) =
    future_cash_weight * min(cash, reserve_target)
```

The value is flat above the target. Spending cash above the target has no new
penalty; spending below it carries a constant marginal opportunity cost.
Because the target approaches zero with the resource horizon, this protection
disappears naturally at the end of the game.

For each legal bid, cash economics become:

```text
liquidity_delta =
    log_value(post_action_cash) - log_value(current_cash)

future_cash_delta =
    future_cash_value(post_action_cash) - future_cash_value(current_cash)

win_delta =
    gross_action_value
    + terminal_cash_delta
    + liquidity_delta
    + future_cash_delta
```

This applies uniformly:

- a resource-auction bid spends cash and may cross the reserve;
- an investment locks cash and carries the same temporary opportunity cost;
- a loan adds spendable cash and receives positive future-cash value early;
- all reserve effects are zero at a zero resource horizon.

## Auditable Types

Extend `HeuristicProfile`:

```python
@dataclass(frozen=True, slots=True)
class HeuristicProfile:
    name: str
    liquidity_strength: float
    future_cash_weight: float
    objective_progress_weight: float
    bid_shading: float
```

`future_cash_weight` must be finite and nonnegative.

Keep the old logarithmic delta and the new reserve delta separate:

```python
@dataclass(frozen=True, slots=True)
class ActionEconomics:
    bid: int
    terminal_cash: float
    liquidity: float
    future_cash: float
    win_delta: float


@dataclass(frozen=True, slots=True)
class ValueBreakdown:
    resource: float
    objective_completion: float
    objective_progress: float
    terminal_cash: float
    liquidity: float
    future_cash: float
    total: float
```

This separation makes the new mechanism visible in tests and later
diagnostics instead of hiding it inside the existing liquidity term.

## Profile Calibration

Objective-progress weights remain fixed at their current values so calibration
does not simultaneously change resource, objective, and cash semantics.

Start calibration with:

| Profile | Liquidity strength | Future cash weight | Objective progress | Bid shading |
|---|---:|---:|---:|---:|
| Aggressive | 0.75 | 1.00 | 0.25 | 0.05 |
| Balanced | 0.40 | 0.75 | 0.20 | 0.25 |
| Passive | 0.15 | 0.25 | 0.15 | 0.50 |

The stronger compensating reserve weight for aggressive is intentional: it
currently exhausts cash fastest. Personality is defined by observable
behavior, not by requiring every internal conservatism coefficient to have the
same ordering.

Calibration may adjust only `future_cash_weight` and `bid_shading`.
`liquidity_strength` and `objective_progress_weight` remain fixed. Every
candidate must satisfy:

```text
0 <= aggressive bid shading
   < balanced bid shading
   < passive bid shading
   <= 0.65

future_cash_weight >= 0 for every profile
```

Use a deterministic 20,000-game calibration tournament with root seed
`20260730`. Change one coefficient family at a time:

1. adjust future-cash weights to correct cash exhaustion;
2. adjust bid shading only if the win-rate target remains out of range;
3. select the lowest-complexity candidate satisfying the targets.

The established 100,000-game tournament with root seed `20260729` is reserved
for final validation and must not drive further tuning.

## Acceptance Criteria

The focused unit and property suites must pass with:

- finite action economics for every legal bid;
- nonincreasing `win_delta` as bid increases;
- zero future-cash effect at zero horizon;
- no future-cash penalty while both current and post-action cash remain above
  the reserve target;
- a negative future-cash delta when a bid spends below the target;
- a positive future-cash delta for an early loan;
- identical evaluations for identical public contexts;
- legal deterministic decisions for charts A-E and 3-5 players.

Canonical equal-information contexts must preserve:

```text
aggressive chosen bid > balanced chosen bid > passive chosen bid
```

The 20,000-game calibration tournament and untouched 100,000-game validation
tournament target:

- balanced has the highest overall outright win rate;
- balanced wins 34-38% outright;
- aggressive and passive each win 29-34% outright;
- the largest overall outright-win-rate gap is at most 7 percentage points;
- aggressive early resource-auction spending is greater than balanced, which
  is greater than passive;
- passive mean terminal liquid cash is greater than balanced, which is greater
  than aggressive;
- the late two-resource market winning bid improves from the measured $5.17
  baseline;
- no bot faults occur.

Win-rate ranges are calibration targets, not runtime invariants. If no
candidate meets every range without violating personality ordering, select
the candidate with the smallest win-rate spread that leaves balanced first,
then report the unmet target explicitly.

## Testing Strategy

Use test-driven development.

1. Add direct tests for the new future-cash value and action-curve deltas in
   `tests/heuristics/test_cash.py`.
2. Update profile validation and ordering tests in
   `tests/heuristics/test_profiles.py`.
3. Add valuation tests showing that the same resource bundle is bid more
   conservatively with a longer horizon and that the three profiles preserve
   bid ordering in `tests/heuristics/test_valuation.py`.
4. Extend finite-breakdown and generated-context assertions in
   `tests/heuristics/test_sanity.py`.
5. Run the focused heuristic suite before calibration.
6. Run the deterministic calibration and final validation tournaments.
7. Run the complete project test suite and static checks after selecting the
   final constants.

## Files

- `src/garboid_pocketrocks/heuristics/profiles.py`
- `src/garboid_pocketrocks/heuristics/cash.py`
- `src/garboid_pocketrocks/heuristics/valuation.py`
- `tests/heuristics/test_profiles.py`
- `tests/heuristics/test_cash.py`
- `tests/heuristics/test_valuation.py`
- `tests/heuristics/test_sanity.py`
- `docs/benchmarks/2026-07-29-future-cash-heuristic.md`

The simulator and SDK adapters remain unchanged.
