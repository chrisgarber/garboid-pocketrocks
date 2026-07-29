# Future-Auction Cash Heuristic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a soft future-auction cash reserve and calibrate the aggressive, balanced, and passive profiles so balanced performs best while all three remain behaviorally distinct and competitively close.

**Architecture:** Extend the pure cash-economics layer with a separately auditable piecewise-linear future-cash utility driven by the existing normalized resource horizon. Thread that delta through valuation breakdowns, add one profile coefficient, then tune only future-cash weights and bid shading through deterministic calibration before an untouched 100,000-game validation.

**Tech Stack:** Python 3.14, frozen dataclasses, pytest, Hypothesis, deterministic PocketRocks simulator, uv

## Global Constraints

- The evaluator may consume only `DecisionContext`, `RulesetKnowledge`, and public SDK definitions.
- The simulator and SDK adapters must remain unchanged.
- `future_cash_weight` must be finite and nonnegative.
- Objective-progress weights and existing logarithmic liquidity strengths remain fixed during calibration.
- Bid shading must remain ordered: aggressive < balanced < passive.
- Every action-economics component and valuation output must remain finite.
- The same public context must continue to produce a deterministic legal decision.
- Calibration uses 20,000 games with root seed `20260730`.
- Final validation uses 100,000 games with root seed `20260729` and must not drive further tuning.

---

### Task 1: Extend the profile contract

**Files:**
- Modify: `tests/heuristics/test_profiles.py`
- Modify: `tests/heuristics/test_valuation.py`
- Modify: `src/garboid_pocketrocks/heuristics/profiles.py`

**Interfaces:**
- Produces: `HeuristicProfile(name, liquidity_strength, future_cash_weight, objective_progress_weight, bid_shading)`
- Produces: validated nonnegative finite `future_cash_weight`

- [ ] **Step 1: Update test profile construction and add failing validation coverage**

Update every direct `HeuristicProfile` construction to pass a
`future_cash_weight`. Add these cases to the coefficient validation table in
`tests/heuristics/test_profiles.py`:

```python
("future_cash_weight", -0.1),
("future_cash_weight", float("nan")),
```

Construct the profile under test with:

```python
HeuristicProfile(
    name="test",
    liquidity_strength=(
        value if field == "liquidity_strength" else 0.4
    ),
    future_cash_weight=(
        value if field == "future_cash_weight" else 0.5
    ),
    objective_progress_weight=(
        value if field == "objective_progress_weight" else 0.2
    ),
    bid_shading=value if field == "bid_shading" else 0.25,
)
```

- [ ] **Step 2: Run the focused profile and valuation tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_profiles.py tests/heuristics/test_valuation.py -q
```

Expected: collection or construction failures because
`HeuristicProfile` does not yet accept `future_cash_weight`.

- [ ] **Step 3: Add the profile field and initial calibration values**

Change the dataclass coefficient order to:

```python
name: str
liquidity_strength: float
future_cash_weight: float
objective_progress_weight: float
bid_shading: float
```

Include `future_cash_weight` in finite validation and reject values below zero.
Set the initial constants to:

```python
AGGRESSIVE_PROFILE = HeuristicProfile("aggressive", 0.75, 1.00, 0.25, 0.05)
BALANCED_PROFILE = HeuristicProfile("balanced", 0.40, 0.75, 0.20, 0.25)
PASSIVE_PROFILE = HeuristicProfile("passive", 0.15, 0.25, 0.15, 0.50)
```

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit the profile contract**

```bash
git add src/garboid_pocketrocks/heuristics/profiles.py \
  tests/heuristics/test_profiles.py tests/heuristics/test_valuation.py
git commit -m "feat: add future cash profile weight"
```

### Task 2: Add pure future-cash economics

**Files:**
- Modify: `tests/heuristics/test_cash.py`
- Modify: `src/garboid_pocketrocks/heuristics/cash.py`

**Interfaces:**
- Produces: `future_cash_value(cash: int, *, horizon: float, starting_cash: int, weight: float) -> float`
- Extends: `evaluate_action_curve(..., future_cash_weight: float, ...)`
- Extends: `ActionEconomics.future_cash: float`

- [ ] **Step 1: Write failing unit tests for the reserve utility**

Import `future_cash_value` and add:

```python
def test_future_cash_value_is_zero_at_zero_horizon() -> None:
    assert future_cash_value(
        30,
        horizon=0.0,
        starting_cash=30,
        weight=1.0,
    ) == 0.0


def test_future_cash_value_is_flat_above_reserve_target() -> None:
    assert future_cash_value(
        30,
        horizon=0.5,
        starting_cash=30,
        weight=0.75,
    ) == future_cash_value(
        15,
        horizon=0.5,
        starting_cash=30,
        weight=0.75,
    )


def test_future_cash_value_has_linear_marginal_value_below_target() -> None:
    assert future_cash_value(
        14,
        horizon=0.5,
        starting_cash=30,
        weight=0.75,
    ) - future_cash_value(
        13,
        horizon=0.5,
        starting_cash=30,
        weight=0.75,
    ) == pytest.approx(0.75)
```

Extend invalid-input parametrization with negative and nonfinite `weight`.

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_cash.py -q
```

Expected: import failure because `future_cash_value` does not exist.

- [ ] **Step 3: Implement the minimal pure utility**

Add:

```python
def future_cash_value(
    cash: int,
    *,
    horizon: float,
    starting_cash: int,
    weight: float,
) -> float:
    cash = _require_nonnegative_integer(cash, "cash")
    horizon = _require_finite_number(horizon, "horizon")
    starting_cash = _require_positive_integer(starting_cash, "starting_cash")
    weight = _require_finite_number(weight, "weight")
    if not 0.0 <= horizon <= 1.0:
        raise ValueError("horizon must be between zero and one")
    if weight < 0.0:
        raise ValueError("weight must be nonnegative")
    value = weight * min(float(cash), starting_cash * horizon)
    if not math.isfinite(value):
        raise ValueError("future cash value must be finite")
    return value
```

- [ ] **Step 4: Run the utility tests to verify GREEN**

Run the command from Step 2.

Expected: the new direct tests pass; existing action-curve tests still pass.

- [ ] **Step 5: Write failing action-curve delta tests**

Add `future_cash_weight` to every `evaluate_action_curve` call. Use `0.0` in
existing tests that assert legacy exact values. Add:

```python
def test_bid_crossing_reserve_has_separate_future_cash_cost() -> None:
    curve = evaluate_action_curve(
        action_id=ActionId.AUCTION2,
        cash=20,
        legal_max=10,
        horizon=0.5,
        starting_cash=30,
        liquidity_strength=0.0,
        future_cash_weight=0.75,
        gross_value=20.0,
    )

    assert curve[5].future_cash == 0.0
    assert curve[6].future_cash == pytest.approx(-0.75)
    assert curve[10].future_cash == pytest.approx(-3.75)


def test_early_loan_receives_positive_future_cash_value() -> None:
    point = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=5,
        legal_max=0,
        horizon=0.8,
        starting_cash=30,
        liquidity_strength=0.0,
        future_cash_weight=0.75,
        gross_value=0.0,
    )[0]

    assert point.future_cash == pytest.approx(7.5)
```

- [ ] **Step 6: Run the action-curve tests to verify RED**

Run the command from Step 2.

Expected: failures because `evaluate_action_curve` and `ActionEconomics` do
not expose future-cash economics.

- [ ] **Step 7: Thread the reserve delta through the action curve**

Add `future_cash: float` to `ActionEconomics`. Add the
`future_cash_weight` parameter to `evaluate_action_curve`, validate it, compute
current and post-action reserve values, and include their difference in
`win_delta`.

- [ ] **Step 8: Run cash tests and verify GREEN**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_cash.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit the cash model**

```bash
git add src/garboid_pocketrocks/heuristics/cash.py tests/heuristics/test_cash.py
git commit -m "feat: value cash reserved for future auctions"
```

### Task 3: Expose future cash in bid valuation

**Files:**
- Modify: `tests/heuristics/test_valuation.py`
- Modify: `tests/heuristics/test_sanity.py`
- Modify: `src/garboid_pocketrocks/heuristics/valuation.py`

**Interfaces:**
- Extends: `ValueBreakdown.future_cash: float`
- Preserves: `BidPoint.win_delta == ValueBreakdown.total`

- [ ] **Step 1: Write failing breakdown tests**

Extend every breakdown-component sum and finiteness assertion with
`breakdown.future_cash`. Add:

```python
def test_zero_horizon_has_no_future_cash_component() -> None:
    won = ((5, 5, 5, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    result = HeuristicValuator(BALANCED_PROFILE).evaluate_bid(
        make_context(
            action_id=ActionId.AUCTION1,
            current_resources=(1, 0),
            won=won,
            cash=(20, 20, 20),
            legal_max=20,
        ),
        make_knowledge(),
    )

    assert all(point.breakdown.future_cash == 0.0 for point in result.points)
```

- [ ] **Step 2: Run valuation and sanity tests to verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_valuation.py tests/heuristics/test_sanity.py -q
```

Expected: failures because `ValueBreakdown.future_cash` is absent and the
evaluator does not pass `future_cash_weight`.

- [ ] **Step 3: Implement the auditable breakdown**

Add `future_cash` to `ValueBreakdown`, pass
`self.profile.future_cash_weight` to `evaluate_action_curve`, pass each
`ActionEconomics.future_cash` into `_bid_point`, and include it in the total.

- [ ] **Step 4: Run focused heuristic tests and verify GREEN**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics -q
```

Expected: PASS.

- [ ] **Step 5: Commit valuation integration**

```bash
git add src/garboid_pocketrocks/heuristics/valuation.py \
  tests/heuristics/test_valuation.py tests/heuristics/test_sanity.py
git commit -m "feat: expose future cash in bid valuations"
```

### Task 4: Lock in personality and horizon behavior

**Files:**
- Modify: `tests/heuristics/test_valuation.py`
- Modify: `tests/heuristics/test_profiles.py`

**Interfaces:**
- Consumes: calibrated profile constants and `HeuristicValuator`
- Produces: regression coverage for observable personality ordering

- [ ] **Step 1: Add failing behavioral tests**

Create equal-information early and late contexts with a constant value chart.
Assert:

```python
early_results = tuple(
    HeuristicValuator(profile).evaluate_bid(early_context, knowledge)
    for profile in (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE)
)
assert (
    early_results[0].chosen_bid
    > early_results[1].chosen_bid
    > early_results[2].chosen_bid
)

for profile in (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE):
    early = HeuristicValuator(profile).evaluate_bid(early_context, knowledge)
    late = HeuristicValuator(profile).evaluate_bid(late_context, knowledge)
    assert early.points[early.chosen_bid].breakdown.future_cash < 0.0
    assert late.points[late.chosen_bid].breakdown.future_cash == 0.0
```

The late context must use won-resource counts that make the normalized horizon
zero without changing the constant per-card value.

- [ ] **Step 2: Run the tests to verify behavior**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_profiles.py tests/heuristics/test_valuation.py -q
```

Expected before calibration: either PASS with the initial coefficients or a
personality-ordering failure that must be corrected only by the allowed
profile constants.

- [ ] **Step 3: Make the minimum allowed constant adjustment**

If Step 2 fails, adjust only `future_cash_weight` or `bid_shading` while
maintaining aggressive < balanced < passive bid shading. Do not change
valuation formulas or objective weights.

- [ ] **Step 4: Run the full heuristic suite**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics tests/benchmarks/test_heuristic_tournament.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit behavioral coverage**

```bash
git add src/garboid_pocketrocks/heuristics/profiles.py \
  tests/heuristics/test_profiles.py tests/heuristics/test_valuation.py
git commit -m "test: preserve heuristic personalities"
```

### Task 5: Calibrate and independently validate the profiles

**Files:**
- Modify: `src/garboid_pocketrocks/heuristics/profiles.py`
- Create: `docs/benchmarks/2026-07-29-future-cash-heuristic.md`
- Use without committing: `/private/tmp/calibrate_future_cash_heuristics.py`
- Use without committing: `/private/tmp/run_pocketrocks_100k.py`
- Use without committing: `/private/tmp/diagnose_late_resource_auctions.py`

**Interfaces:**
- Consumes: deterministic simulator, profile constants
- Produces: final profile constants and benchmark record

- [ ] **Step 1: Create the disposable calibration runner**

Build `/private/tmp/calibrate_future_cash_heuristics.py` from the existing
Monte Carlo runner. It must run 20,000 games over equally weighted charts A-E
with root seed `20260730` and report:

- outright win rates;
- mean score and rank;
- pass rate and mean nonzero bid;
- early resource-auction payments per game;
- mean terminal liquid cash;
- late two-resource market winning bid;
- faults.

- [ ] **Step 2: Run the initial calibration candidate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run python /private/tmp/calibrate_future_cash_heuristics.py
```

Record the exact constants and output.

- [ ] **Step 3: Tune future-cash weights**

Change only `future_cash_weight`, one profile at a time. Prefer round
increments of `0.10` or `0.25`. Keep any candidate only if it reduces the
win-rate spread or moves balanced toward first place without violating:

```text
aggressive early resource spend
    > balanced early resource spend
    > passive early resource spend

passive terminal cash
    > balanced terminal cash
    > aggressive terminal cash
```

- [ ] **Step 4: Tune bid shading only if needed**

If future-cash weights alone do not place balanced first within a 7-point
spread, adjust bid shading in increments of `0.05`, preserving:

```text
0 <= aggressive < balanced < passive <= 0.65
```

Do not revisit any coefficient after starting the final validation run.

- [ ] **Step 5: Run focused tests with selected constants**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics tests/benchmarks/test_heuristic_tournament.py -q
```

Expected: PASS.

- [ ] **Step 6: Run untouched 100,000-game validation**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run python /private/tmp/run_pocketrocks_100k.py
```

Then run the late-auction diagnostic at 100,000 games without changing the
selected constants.

Expected:

- balanced ranks first overall;
- balanced 34-38% outright wins;
- aggressive and passive each 29-34%;
- maximum spread at most 7 percentage points;
- late two-resource price exceeds $5.17;
- personality aggregate ordering holds;
- zero faults.

If the validation misses a range, do not retune against seed `20260729`.
Document the selected calibration result, validation result, and exact miss.

- [ ] **Step 7: Write the benchmark record**

Document:

- baseline and final constants;
- calibration and validation seeds;
- win rates, ties, mean score, rank, and behavior metrics;
- early/middle/late two-resource cash, gross value, cap-binding rate, and
  winning price;
- per-chart win rates and scores;
- which acceptance criteria passed or missed;
- why no validation-driven retuning was performed.

- [ ] **Step 8: Commit final constants and benchmark**

```bash
git add src/garboid_pocketrocks/heuristics/profiles.py \
  docs/benchmarks/2026-07-29-future-cash-heuristic.md
git commit -m "feat: calibrate future cash heuristics"
```

### Task 6: Complete project verification

**Files:**
- Verify only

**Interfaces:**
- Consumes: completed heuristic implementation
- Produces: repository-wide verification evidence

- [ ] **Step 1: Run formatting and static checks**

Run the repository-configured formatter, linter, and type checker through the
commands declared in `pyproject.toml`.

Expected: all checks pass without modifying unrelated neural files.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check HEAD^
git status --short
git log --oneline -8
```

Confirm only heuristic, focused test, specification, plan, and benchmark files
belong to this change. Preserve all unrelated pre-existing neural and script
changes.
