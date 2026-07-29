# Bayesian Heuristic Bots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a live-compatible Bayesian action valuator, then use it for aggressive, balanced, and passive PocketRocks opponents with behavioral Monte Carlo evaluation.

**Architecture:** Independent pure modules compute finite-population resource beliefs, objective progress, and cash option value. `HeuristicValuator` composes them into an auditable integer bid curve; only after its standalone sanity gate passes does a thin `BotBrain` layer expose the three profiles. Existing replay decisions and game events feed behavioral metrics without changing engine semantics.

**Tech Stack:** Python 3.14, pinned PocketRocks SDK, standard-library `math`/`dataclasses`, pytest, Hypothesis, uv, Ruff, mypy.

## Global Constraints

- Consume only `DecisionContext`, `RulesetKnowledge`, and public SDK `OBJECTIVES`; never pass `GameState`, events, replay seeds, or hidden simulator state into heuristic code.
- Use exact integer-combination hypergeometric probabilities; add no SciPy dependency.
- Keep every evaluator function deterministic and stateless.
- Use the exact profile constants and development-only bot IDs in the approved design.
- Follow red-green-refactor: every production behavior starts with a focused failing test that is observed failing for the intended reason.
- Keep `RandomBot` behavior and the deterministic engine unchanged.
- Run focused pytest, Ruff, and mypy gates after each task and commit each independently.

## Execution Topology

Tasks 1, 2, and 3 have disjoint production/test files and may run in parallel.
Task 2 accepts a primitive `progress_weight` so it does not import Task 1.
Task 4 begins only after all three are integrated. Tasks 5-8 are sequential
because each is a reviewable gate over the preceding behavior.

When parallel subagents share this worktree, they stop before each task's
commit step. The root reviewer stages only that task's exact file list and
creates its commit after reviewing and running the integrated gate; agents
must not commit or stage one another's visible changes.

---

### Task 1: Validated Profiles and Bayesian Resource Belief

**Files:**
- Create: `src/garboid_pocketrocks/heuristics/__init__.py`
- Create: `src/garboid_pocketrocks/heuristics/errors.py`
- Create: `src/garboid_pocketrocks/heuristics/profiles.py`
- Create: `src/garboid_pocketrocks/heuristics/belief.py`
- Create: `tests/heuristics/__init__.py`
- Create: `tests/heuristics/helpers.py`
- Create: `tests/heuristics/test_profiles.py`
- Create: `tests/heuristics/test_belief.py`

**Interfaces:**
- Consumes: SDK `DecisionContext`, `Suit`; `RulesetKnowledge`.
- Produces: `HeuristicInputError`, `HeuristicProfile`, three module-level profile constants, `SuitBelief`, `BeliefState`, and `build_belief`.

- [ ] **Step 1: Add context builders and failing profile-validation tests**

Create `tests/heuristics/helpers.py` with a complete keyword-driven SDK context builder:

```python
from __future__ import annotations

from pocketrocks import OBJECTIVES, ActionId, DecisionContext

from garboid_pocketrocks.rules import RulesetKnowledge


def make_knowledge(
    *,
    private_cards: int = 0,
    resource_counts: tuple[int, ...] = (2, 2, 2, 2, 2),
    value_chart: tuple[int, ...] = (0, 4, 8, 12, 16, 20),
) -> RulesetKnowledge:
    return RulesetKnowledge(
        name="heuristic-test",
        player_count=3,
        starting_cash=30,
        private_cards_per_player=private_cards,
        resource_counts=resource_counts,
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=value_chart,
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def make_context(
    *,
    decision_kind: str = "submitBid",
    action_id: ActionId = ActionId.AUCTION1,
    current_resources: tuple[int, int] = (1, 0),
    cash: tuple[int, ...] = (30, 30, 30),
    won: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    revealed: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    owned_objectives: tuple[tuple[int, ...], ...] = ((), (), ()),
    objectives: tuple[int, ...] = (),
    hand: tuple[int, ...] = (),
    legal_max: int | None = 30,
    bot_seat: int = 0,
    value_chart: tuple[int, ...] = (0, 4, 8, 12, 16, 20),
) -> DecisionContext:
    return DecisionContext(
        request_id="heuristic-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=value_chart,
        objective_ids=objectives,
        current_action_id=int(action_id),
        current_resource_ids=current_resources,
        cash_by_seat=cash,
        tiebreak_seat=2,
        won_resource_counts_by_seat=won,
        revealed_info_counts_by_seat=revealed,
        owned_objective_ids_by_seat=owned_objectives,
        bot_seat=bot_seat,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )
```

In `tests/heuristics/test_profiles.py`, require finite/ranged coefficients and
the exact constants:

```python
import pytest

from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("liquidity_strength", -0.1),
        ("liquidity_strength", float("nan")),
        ("objective_progress_weight", -0.1),
        ("objective_progress_weight", 1.1),
        ("bid_shading", -0.1),
        ("bid_shading", 1.1),
    ),
)
def test_profile_rejects_invalid_coefficient(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        HeuristicProfile(
            name="invalid",
            liquidity_strength=value if field == "liquidity_strength" else 0.4,
            objective_progress_weight=(value if field == "objective_progress_weight" else 0.2),
            bid_shading=value if field == "bid_shading" else 0.25,
        )


def test_named_profiles_have_expected_ordering() -> None:
    assert (
        AGGRESSIVE_PROFILE.liquidity_strength
        > BALANCED_PROFILE.liquidity_strength
        > PASSIVE_PROFILE.liquidity_strength
    )
    assert (
        AGGRESSIVE_PROFILE.bid_shading < BALANCED_PROFILE.bid_shading < PASSIVE_PROFILE.bid_shading
    )
```

- [ ] **Step 2: Run profile tests and observe the missing-module failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_profiles.py -q
```

Expected: collection fails because `garboid_pocketrocks.heuristics.profiles`
does not exist.

- [ ] **Step 3: Implement the profile and error types**

Create `profiles.py` with the frozen dataclass, complete validation, and exact
constants:

```python
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeuristicProfile:
    name: str
    liquidity_strength: float
    objective_progress_weight: float
    bid_shading: float

    def __post_init__(self) -> None:
        coefficients = (
            self.liquidity_strength,
            self.objective_progress_weight,
            self.bid_shading,
        )
        if not self.name:
            raise ValueError("profile name must be nonempty")
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("profile coefficients must be finite")
        if self.liquidity_strength < 0:
            raise ValueError("liquidity strength must be nonnegative")
        if not 0 <= self.objective_progress_weight <= 1:
            raise ValueError("objective progress weight must be between zero and one")
        if not 0 <= self.bid_shading <= 1:
            raise ValueError("bid shading must be between zero and one")


AGGRESSIVE_PROFILE = HeuristicProfile("aggressive", 0.75, 0.25, 0.05)
BALANCED_PROFILE = HeuristicProfile("balanced", 0.40, 0.20, 0.25)
PASSIVE_PROFILE = HeuristicProfile("passive", 0.15, 0.15, 0.50)
```

Create `errors.py`:

```python
class HeuristicInputError(ValueError):
    """Raised when an SDK context contradicts public ruleset knowledge."""
```

- [ ] **Step 4: Add failing belief examples and conservation properties**

In `test_belief.py`, cover:

```python
from dataclasses import replace

import pytest
from pocketrocks import ActionId, Suit

from garboid_pocketrocks.heuristics.belief import build_belief
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from tests.heuristics.helpers import make_context, make_knowledge


def test_no_hidden_private_cards_uses_deterministic_chart_bucket() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        value_chart=(7, 7, 7, 7, 7, 7),
    )
    belief = build_belief(
        context,
        make_knowledge(value_chart=context.value_chart),
    )
    brick = belief.suits[int(Suit.BRICK) - 1]
    assert brick.terminal_price_pmf == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert brick.expected_terminal_price == 7.0


def test_reveal_context_does_not_subtract_preserved_offer_twice() -> None:
    won = ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    context = make_context(
        decision_kind="selectInfoToReveal",
        current_resources=(1, 0),
        won=won,
        hand=(2,),
        legal_max=None,
    )
    belief = build_belief(context, make_knowledge(private_cards=1))
    assert (
        sum(suit.unseen_suit_count for suit in belief.suits) - belief.suits[0].opponent_hidden_slots
        == 6
    )


def test_financial_bid_has_no_offered_resources() -> None:
    context = make_context(
        action_id=ActionId.LOAN10,
        current_resources=(0, 0),
    )
    belief = build_belief(context, make_knowledge())
    assert belief.normalized_horizon == 1.0


def test_inconsistent_known_cards_are_rejected() -> None:
    context = make_context(
        won=((3, 0, 0, 0, 0),) + ((0, 0, 0, 0, 0),) * 2,
    )
    with pytest.raises(HeuristicInputError, match="known card"):
        build_belief(context, make_knowledge(resource_counts=(2, 2, 2, 2, 2)))


def test_constant_chart_stays_constant() -> None:
    context = make_context(value_chart=(9, 9, 9, 9, 9, 9))
    belief = build_belief(
        context,
        make_knowledge(private_cards=0, value_chart=context.value_chart),
    )
    assert all(suit.expected_terminal_price == 9.0 for suit in belief.suits)
```

- [ ] **Step 5: Run belief tests and observe the missing API failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_belief.py -q
```

Expected: collection fails because `build_belief` is missing.

- [ ] **Step 6: Implement exact finite-population belief**

Implement:

```python
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


def _hypergeometric_probability(
    population: int,
    successes: int,
    draws: int,
    selected: int,
) -> float:
    if selected < 0 or selected > successes or draws - selected > population - successes:
        return 0.0
    return (
        math.comb(successes, selected)
        * math.comb(population - successes, draws - selected)
        / math.comb(population, draws)
    )
```

`build_belief` must:

1. validate all matrix widths/counts and actor/setup agreement;
2. set the offered count only for an auction bidding request;
3. calculate `U`, `M`, `B`, `W`, `q`, and assert
   `sum(U) - M == B - W - q`;
4. aggregate the marginal PMF into six capped reveal buckets;
5. calculate expected price from `context.value_chart`;
6. calculate expected future biddable counts;
7. calculate the post-action horizon;
8. return immutable tuples.

- [ ] **Step 7: Expand generated belief properties and run the full focused gate**

Add a Hypothesis strategy that starts real games through `GameEngine`, advances
them with legal random decisions, and feeds only produced SDK contexts to
`build_belief`. Assert conservation, PMF normalization, bounds, and own
hand-to-reveal invariance.

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_profiles.py tests/heuristics/test_belief.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check \
    src/garboid_pocketrocks/heuristics/__init__.py \
    src/garboid_pocketrocks/heuristics/errors.py \
    src/garboid_pocketrocks/heuristics/profiles.py \
    src/garboid_pocketrocks/heuristics/belief.py \
    tests/heuristics/__init__.py tests/heuristics/helpers.py \
    tests/heuristics/test_profiles.py tests/heuristics/test_belief.py
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy \
    src/garboid_pocketrocks/heuristics/__init__.py \
    src/garboid_pocketrocks/heuristics/errors.py \
    src/garboid_pocketrocks/heuristics/profiles.py \
    src/garboid_pocketrocks/heuristics/belief.py \
    tests/heuristics/__init__.py tests/heuristics/helpers.py \
    tests/heuristics/test_profiles.py tests/heuristics/test_belief.py
```

Expected: all focused tests and static checks pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  src/garboid_pocketrocks/heuristics/__init__.py \
  src/garboid_pocketrocks/heuristics/errors.py \
  src/garboid_pocketrocks/heuristics/profiles.py \
  src/garboid_pocketrocks/heuristics/belief.py \
  tests/heuristics/__init__.py tests/heuristics/helpers.py \
  tests/heuristics/test_profiles.py tests/heuristics/test_belief.py
git commit -m "feat: add Bayesian resource beliefs"
```

---

### Task 2: Objective Requirements and Nonlinear Progress

**Files:**
- Create: `src/garboid_pocketrocks/heuristics/objectives.py`
- Create: `tests/heuristics/test_objectives.py`

**Interfaces:**
- Consumes: SDK `OBJECTIVES`; public holdings/objective ownership, normalized horizon, and a validated progress-weight float supplied by the caller.
- Produces: `ObjectiveValue`, `requirement_vectors`, `objective_distance`, `objective_is_met`, and `evaluate_objectives`.

- [ ] **Step 1: Write failing requirement and valuation tests**

Cover all patterns and the exact completion/progress split:

```python
from pocketrocks import OBJECTIVES

from garboid_pocketrocks.heuristics.objectives import (
    evaluate_objectives,
    objective_is_met,
    requirement_vectors,
)


def test_every_sdk_objective_has_requirement_vectors() -> None:
    for objective_id in OBJECTIVES:
        vectors = requirement_vectors(objective_id)
        assert vectors
        assert all(len(vector) == 5 and sum(vector) > 0 for vector in vectors)


def test_immediate_completion_has_full_payout_and_no_progress() -> None:
    values = evaluate_objectives(
        active_objective_ids=(6,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=((1, 0, 0, 0, 0),) * 3,
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=0.5,
        progress_weight=0.2,
    )
    assert values[0].completion_value == OBJECTIVES[6].payout
    assert values[0].progress_value == 0.0


def test_near_completion_progress_exceeds_initial_progress() -> None:
    initial = evaluate_objectives(
        active_objective_ids=(2,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=1.0,
        progress_weight=0.2,
    )[0]
    near = evaluate_objectives(
        active_objective_ids=(2,),
        owned_objective_ids_by_seat=((), (), ()),
        won_resource_counts_by_seat=(
            (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        bot_seat=0,
        offered_counts=(1, 0, 0, 0, 0),
        horizon=1.0,
        progress_weight=0.2,
    )[0]
    assert near.progress_value > initial.progress_value
```

Add an exhaustive generated conformance test comparing `objective_is_met` to
the engine's behavior by placing generated holdings in a state and observing
objective claims for every objective ID.

- [ ] **Step 2: Run tests and observe the missing-module failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_objectives.py -q
```

Expected: collection fails because `heuristics.objectives` is missing.

- [ ] **Step 3: Implement requirement vectors and shaped marginal progress**

Create immutable `ObjectiveValue` and implement:

```python
@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    objective_id: int
    completion_value: float
    progress_value: float

    @property
    def total(self) -> float:
        return self.completion_value + self.progress_value


def _progress(counts: tuple[int, ...], vectors: tuple[tuple[int, ...], ...]) -> float:
    return max(
        1.0
        - (
            sum(max(required - owned, 0) for owned, required in zip(counts, vector, strict=True))
            / sum(vector)
        )
        for vector in vectors
    )
```

Generate vectors with `itertools.combinations` for flexible patterns. In
`evaluate_objectives`, skip any objective owned by any seat; use full payout
and zero progress on immediate completion; otherwise calculate the positive
squared-progress change times `progress_weight`, horizon, and the exact public
contest factor from the design.

- [ ] **Step 4: Run focused and static gates**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_objectives.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check src/garboid_pocketrocks/heuristics/objectives.py \
    tests/heuristics/test_objectives.py
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy src/garboid_pocketrocks/heuristics/objectives.py \
    tests/heuristics/test_objectives.py
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/garboid_pocketrocks/heuristics/objectives.py \
  tests/heuristics/test_objectives.py
git commit -m "feat: value objective completion and progress"
```

---

### Task 3: Cash Option Value and Action Curves

**Files:**
- Create: `src/garboid_pocketrocks/heuristics/cash.py`
- Create: `tests/heuristics/test_cash.py`

**Interfaces:**
- Consumes: cash, horizon, starting cash, liquidity strength, action kind, gross action value, and legal maximum.
- Produces: `cash_option_value`, `ActionEconomics`, and `evaluate_action_curve`.

- [ ] **Step 1: Write failing accounting and monotonicity tests**

```python
import math

from pocketrocks import ActionId

from garboid_pocketrocks.heuristics.cash import (
    cash_option_value,
    evaluate_action_curve,
)


def test_option_value_is_zero_at_zero_horizon() -> None:
    assert cash_option_value(30, horizon=0.0, starting_cash=30, strength=0.75) == 0.0


def test_option_value_is_increasing_and_concave() -> None:
    values = [
        cash_option_value(cash, horizon=1.0, starting_cash=30, strength=0.75)
        for cash in (0, 10, 20, 30)
    ]
    assert values == sorted(values)
    assert values[1] - values[0] > values[2] - values[1] > values[3] - values[2]


def test_zero_horizon_action_accounting_is_exact() -> None:
    loan = evaluate_action_curve(
        action_id=ActionId.LOAN10,
        cash=20,
        legal_max=30,
        horizon=0.0,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )
    investment = evaluate_action_curve(
        action_id=ActionId.INVEST5,
        cash=20,
        legal_max=20,
        horizon=0.0,
        starting_cash=30,
        liquidity_strength=0.75,
        gross_value=0.0,
    )
    assert [point.win_delta for point in loan[:3]] == [0.0, -1.0, -2.0]
    assert all(math.isclose(point.win_delta, 5.0) for point in investment)
```

Add parameterized tests proving all three action curves are nonincreasing in
bid and loan benefit/investment lock cost move in the required horizon
direction.

- [ ] **Step 2: Run and observe the missing-module failure**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_cash.py -q
```

Expected: collection fails because `heuristics.cash` is missing.

- [ ] **Step 3: Implement the exact action equations**

```python
@dataclass(frozen=True, slots=True)
class ActionEconomics:
    bid: int
    terminal_cash: float
    liquidity: float
    win_delta: float


def cash_option_value(
    cash: int,
    *,
    horizon: float,
    starting_cash: int,
    strength: float,
) -> float:
    if cash < 0 or not 0 <= horizon <= 1:
        raise ValueError("cash and horizon are outside economic bounds")
    kappa = max(starting_cash / 2.0, 1.0)
    return strength * horizon * kappa * math.log1p(cash / kappa)
```

`evaluate_action_curve` must enumerate `range(legal_max + 1)` and calculate:

```python
if action_id in (ActionId.AUCTION1, ActionId.AUCTION2):
    terminal_cash = -float(bid)
    post_cash = cash - bid
elif action_id in (ActionId.LOAN10, ActionId.LOAN20):
    principal = 10 if action_id is ActionId.LOAN10 else 20
    terminal_cash = -float(bid)
    post_cash = cash + principal - bid
else:
    payout = 5 if action_id is ActionId.INVEST5 else 10
    terminal_cash = float(payout)
    post_cash = cash - bid
liquidity = option(post_cash) - option(cash)
win_delta = gross_value + terminal_cash + liquidity
```

- [ ] **Step 4: Run focused and static gates**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_cash.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check src/garboid_pocketrocks/heuristics/cash.py \
    tests/heuristics/test_cash.py
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy src/garboid_pocketrocks/heuristics/cash.py \
    tests/heuristics/test_cash.py
```

Expected: pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/garboid_pocketrocks/heuristics/cash.py \
  tests/heuristics/test_cash.py
git commit -m "feat: model time-sensitive cash utility"
```

---

### Task 4: Compose the Auditable Heuristic Valuator

**Files:**
- Create: `src/garboid_pocketrocks/heuristics/valuation.py`
- Modify: `src/garboid_pocketrocks/heuristics/__init__.py`
- Create: `tests/heuristics/test_valuation.py`

**Interfaces:**
- Consumes: Task 1 belief/profile, Task 2 objectives, Task 3 cash curves.
- Produces: `ValueBreakdown`, `BidPoint`, `BidEvaluation`, and `HeuristicValuator.evaluate_bid`.

- [ ] **Step 1: Write failing end-to-end valuation tests**

```python
from pocketrocks import ActionId

import pytest

from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from tests.heuristics.helpers import make_context, make_knowledge

NO_LIQUIDITY = HeuristicProfile("test", 0.0, 0.0, 0.0)


def test_constant_ten_dollar_resource_has_ten_dollar_reservation() -> None:
    chart = (10, 10, 10, 10, 10, 10)
    context = make_context(value_chart=chart, legal_max=30)
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        context,
        make_knowledge(value_chart=chart),
    )
    assert result.reservation_bid == 10
    assert result.chosen_bid == 10
    assert result.points[10].breakdown.resource == 10.0
    assert result.points[10].win_delta == 0.0


def test_zero_horizon_loan_passes_and_investment_can_lock_all_cash() -> None:
    won = ((2, 2, 2, 2, 2), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    loan_context = make_context(
        action_id=ActionId.LOAN10,
        current_resources=(0, 0),
        won=won,
        cash=(20, 20, 20),
        legal_max=30,
    )
    investment_context = make_context(
        action_id=ActionId.INVEST5,
        current_resources=(0, 0),
        won=won,
        cash=(20, 20, 20),
        legal_max=20,
    )
    evaluator = HeuristicValuator(NO_LIQUIDITY)
    assert evaluator.evaluate_bid(loan_context, make_knowledge()).chosen_bid == 0
    assert evaluator.evaluate_bid(investment_context, make_knowledge()).chosen_bid == 20


@pytest.mark.parametrize(
    "profile",
    (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE),
)
@pytest.mark.parametrize(
    ("action_id", "resources", "legal_max"),
    (
        (ActionId.AUCTION1, (1, 0), 30),
        (ActionId.LOAN10, (0, 0), 40),
        (ActionId.INVEST5, (0, 0), 30),
    ),
)
def test_profile_shading_never_exceeds_reservation_or_legal_maximum(
    profile: HeuristicProfile,
    action_id: ActionId,
    resources: tuple[int, int],
    legal_max: int,
) -> None:
    result = HeuristicValuator(profile).evaluate_bid(
        make_context(
            action_id=action_id,
            current_resources=resources,
            legal_max=legal_max,
        ),
        make_knowledge(),
    )
    assert 0 <= result.chosen_bid <= result.reservation_bid <= legal_max


def test_breakdown_components_sum_to_each_point_delta() -> None:
    result = HeuristicValuator(NO_LIQUIDITY).evaluate_bid(
        make_context(),
        make_knowledge(),
    )
    for point in result.points:
        breakdown = point.breakdown
        expected = (
            breakdown.resource
            + breakdown.objective_completion
            + breakdown.objective_progress
            + breakdown.terminal_cash
            + breakdown.liquidity
        )
        assert breakdown.total == expected == point.win_delta
```

- [ ] **Step 2: Run and observe the missing-module failure**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_valuation.py -q
```

Expected: collection fails because `heuristics.valuation` is missing.

- [ ] **Step 3: Implement valuation composition**

Create the frozen breakdown/result types from the design. In
`evaluate_bid`:

1. reject non-bid contexts, missing actions, or missing legal maxima;
2. build the belief;
3. count the offered bundle;
4. calculate resource value only for Auction 1/2;
5. evaluate objective completion/progress only for Auction 1/2;
6. set `gross_value = resource + objective_completion + objective_progress`;
7. evaluate every legal bid with `evaluate_action_curve`;
8. preserve cash/liquidity components in each `ValueBreakdown`;
9. choose the largest bid with `win_delta >= 0` as reservation;
10. choose `floor(reservation * (1 - profile.bid_shading))`;
11. assert the chosen bid is legal and finite.

- [ ] **Step 4: Run focused and static gates**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check src/garboid_pocketrocks/heuristics tests/heuristics
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy src/garboid_pocketrocks/heuristics tests/heuristics
```

Expected: pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/garboid_pocketrocks/heuristics tests/heuristics
git commit -m "feat: compose heuristic action valuations"
```

---

### Task 5: Standalone Heuristic Sanity Gate

**Files:**
- Create: `tests/heuristics/test_sanity.py`
- Modify only if a failing sanity property exposes a defect:
  `src/garboid_pocketrocks/heuristics/*.py`

**Interfaces:**
- Consumes: complete `HeuristicValuator`.
- Produces: evidence that the evaluator works across engine-generated public contexts before any bot class exists.

- [ ] **Step 1: Write generated context and metamorphic sanity tests**

The test must:

- run real engine states for charts A-E and 3-5 players;
- collect every bidding context before terminal;
- evaluate all three profiles;
- assert every number is finite and every bid is legal;
- assert identical SDK contexts produce identical valuations despite different
  hidden engine state;
- move an own hand card into revealed counts and assert resource expectations
  are unchanged;
- compare early and late financial contexts to verify loan and investment time
  direction;
- verify charts B/D/E are not treated as monotone-increasing.

Use at least 25 Hypothesis examples per player count.

- [ ] **Step 2: Run the sanity gate before bot implementation**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_sanity.py -q
```

Expected on the first run: at least one assertion fails if the composed
evaluator has an integration error. If it passes immediately, record that the
focused implementation already satisfies the broader properties. If it fails,
add a focused regression that reproduces the defect, observe that regression
fail, fix only the proven defect, and rerun both tests.

- [ ] **Step 3: Run all heuristic and simulator regression tests**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics tests/simulator -q
```

Expected: pass. Do not start Task 6 until this gate is green.

- [ ] **Step 4: Commit the sanity gate and any proven fixes**

```bash
git add tests/heuristics/test_sanity.py src/garboid_pocketrocks/heuristics
git commit -m "test: sanity-check heuristic valuations"
```

---

### Task 6: Reveal Policy and Three Heuristic Bot Profiles

**Files:**
- Create: `src/garboid_pocketrocks/heuristics/reveals.py`
- Create: `src/garboid_pocketrocks/bots/heuristic.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/heuristics/valuation.py`
- Create: `tests/heuristics/test_reveals.py`
- Create: `tests/bots/test_heuristic_bots.py`

**Interfaces:**
- Consumes: Task 1 finite-population formulas, `HeuristicValuator.evaluate_bid`, existing `PocketRocksFastBot`.
- Produces: `choose_reveal`, `HeuristicBotBrain`, three brain classes, three bot classes, and three picklable `BotSpec` values.

- [ ] **Step 1: Write failing reveal-policy tests**

Require:

- no-hand contexts raise `HeuristicInputError`;
- every returned index is legal;
- repeated suits choose their lowest index;
- the influence calculation includes price changes in all five suits;
- a deterministic tie chooses index zero.

- [ ] **Step 2: Run and observe the missing reveal API**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics/test_reveals.py -q
```

Expected: collection fails because `heuristics.reveals` is missing.

- [ ] **Step 3: Implement cross-suit observer influence**

Implement in `reveals.py`:

```python
def build_observer_price_vector(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    *,
    revealed_suit: Suit | None = None,
) -> tuple[float, ...]:
    """Expected prices for an observer who does not know the actor's hand."""
```

Before a candidate reveal, all remaining private cards—including the actor's
hand—are hidden slots. After a candidate reveal, decrement both the unseen
suit population and hidden slots and increment that suit's known reveal count.

For every distinct suit in `current_hand_suit_ids`:

1. calculate the observer price vector with no candidate reveal;
2. calculate it after revealing the candidate suit;
3. sum every opponent's held count times every suit-price delta;
4. select the minimum `(influence, first_hand_index)` tuple.

Expose it through `HeuristicValuator.choose_reveal`.

- [ ] **Step 4: Write failing brain and identity tests**

Tests must assert:

```python
assert AggressiveHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000a"
assert BalancedHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000b"
assert PassiveHeuristicBot.BOT_ID == "bot_00000000-0000-4000-8000-00000000000c"
```

Also require distinct names, legal bid/reveal decisions, safe pass only for
`HeuristicInputError`, deterministic construction, `pickle.dumps(BotSpec)`,
and a two-worker Monte Carlo smoke using all three specs.

- [ ] **Step 5: Implement top-level brains and bot classes**

`HeuristicBotBrain.choose_decision` must:

```python
try:
    if context.decision_kind == "selectInfoToReveal":
        return BotDecision.select_info_to_reveal(self.valuator.choose_reveal(context, ruleset))
    bid = self.valuator.evaluate_bid(context, ruleset).chosen_bid
    return BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid)
except HeuristicInputError:
    return BotDecision.pass_turn()
```

Define top-level `AggressiveHeuristicBrain`, `BalancedHeuristicBrain`, and
`PassiveHeuristicBrain` constructors with module profile constants. Define
top-level bot wrappers with importable classmethod factories. Do not expose
live console scripts for development IDs.

- [ ] **Step 6: Run focused and static gates**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/heuristics tests/bots/test_heuristic_bots.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check src/garboid_pocketrocks/heuristics \
    src/garboid_pocketrocks/bots tests/heuristics tests/bots
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy src/garboid_pocketrocks/heuristics \
    src/garboid_pocketrocks/bots tests/heuristics tests/bots
```

Expected: pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add src/garboid_pocketrocks/heuristics \
  src/garboid_pocketrocks/bots tests/heuristics tests/bots
git commit -m "feat: add three heuristic bot profiles"
```

---

### Task 7: Behavioral Monte Carlo Statistics

**Files:**
- Modify: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Modify: `tests/simulator/test_monte_carlo.py`
- Modify: `tests/simulator/test_cli.py`

**Interfaces:**
- Consumes: replay `BotDecision` values and domain events from completed matches.
- Produces: `BehaviorStatistics` attached to each `BotStatistics`, plus table/JSON CLI output.

- [ ] **Step 1: Write failing metric-definition tests**

Construct a completed match with scripted brains and assert:

- `pass` and `submitBid(0)` both count as bidding passes;
- reveal decisions do not enter the pass-rate denominator;
- empty `nonzero_bids` yields `mean_nonzero_bid() == 0.0`;
- action wins, resource cards, and objectives count from their exact events;
- duplicate copies of one bot identity aggregate together;
- serial and two-worker results are identical including behavior.

- [ ] **Step 2: Run and observe missing behavior statistics**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/simulator/test_monte_carlo.py -q
```

Expected: failure because `BotStatistics.behavior` is missing.

- [ ] **Step 3: Implement behavior accumulation**

Create:

```python
@dataclass(frozen=True, slots=True)
class BehaviorStatistics:
    bidding_requests: int
    passes: int
    nonzero_bids: tuple[int, ...]
    reveal_choices: tuple[int, ...]
    wins_by_action: tuple[int, ...]
    resource_cards_won: int
    objectives_claimed: int

    def pass_rate(self) -> float:
        return self.passes / self.bidding_requests if self.bidding_requests else 0.0

    def mean_nonzero_bid(self) -> float:
        return float(statistics.mean(self.nonzero_bids)) if self.nonzero_bids else 0.0
```

`wins_by_action` contains exactly six entries indexed by `ActionId - 1`.

Update `_aggregate` to:

- inspect replay decisions by seat for bidding/pass/bid/reveal metrics;
- inspect events by seat for wins/resources/objectives;
- merge accumulators by bot ID;
- freeze sorted tuples in the result.

No positional join between replay steps and events is required.

- [ ] **Step 4: Add behavior columns to table output and structured JSON**

Add `pass_rate`, `mean_bid`, `resource_wins`, and `objectives` columns to the
table. `asdict` already carries the raw behavior object into JSON; add focused
JSON assertions for its exact keys and integer/float representations.

- [ ] **Step 5: Run focused and static gates**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/simulator/test_monte_carlo.py \
    tests/simulator/test_cli.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run ruff check src/garboid_pocketrocks/simulator tests/simulator
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "feat: report Monte Carlo bot behavior"
```

---

### Task 8: CLI Integration, Tournaments, Tuning, and Documentation

**Files:**
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Modify: `README.md`
- Create: `tests/benchmarks/test_heuristic_tournament.py`
- Modify: `tests/test_integration.py`
- Create: `docs/benchmarks/2026-07-28-heuristic-v1.md`
- Modify if benchmark evidence requires calibration:
  `src/garboid_pocketrocks/heuristics/profiles.py`

**Interfaces:**
- Consumes: three bot classes, behavior metrics, Monte Carlo runner.
- Produces: registered CLI bot names, fixed seeded benchmark, calibrated v1 profiles, and user documentation.

- [ ] **Step 1: Write failing CLI and full-game integration tests**

Require:

- `--bots aggressive,balanced,passive` runs;
- `payload["result"]` is identical at workers 1 and 2 while
  `payload["configuration"]["workers"]` records the requested worker count;
- each profile completes games without faults across charts A-E and player
  counts 3-5;
- each returned decision remains legal through full matches.

The all-chart/player-count smoke lineups are exactly:

- 3 players: aggressive, balanced, passive;
- 4 players: aggressive, balanced, passive, random;
- 5 players: aggressive, balanced, passive, random, random.

For chart letter `chart` and player count `players`, use
`root_seed = 42 + 100 * (ord(chart) - ord("A")) + players`. Run 15 games for
each of the 15 smoke combinations.

- [ ] **Step 2: Register the three bot specifications**

Add `BotSpec.from_bot_class(AggressiveHeuristicBot)`,
`BotSpec.from_bot_class(BalancedHeuristicBot)`, and
`BotSpec.from_bot_class(PassiveHeuristicBot)` to `_BOT_REGISTRY`. Keep
`BotSpec.from_bot_class(RandomBot)`.
Document that development-only heuristic IDs are for simulation and must be
replaced before live connection.

- [ ] **Step 3: Write and run the fixed tournament test**

The benchmark test uses:

```python
MonteCarloConfig(
    bot_specs=(
        BotSpec.from_bot_class(AggressiveHeuristicBot),
        BotSpec.from_bot_class(BalancedHeuristicBot),
        BotSpec.from_bot_class(PassiveHeuristicBot),
    ),
    games=1000,
    player_counts=(3,),
    ruleset_sampler=FixedRulesetSampler(live_ruleset("A")),
    root_seed=42,
)
```

Assert zero faults, fair seat counts, serial/parallel equality, at least ten
percentage points of pass-rate separation between two profiles, at least one
dollar of mean-nonzero-bid separation, and more aggressive than passive loan
wins. Do not assert win-rate ordering.

This deterministic 1,000-game test remains in the default pytest suite. Its
serial-plus-two-worker wall time must stay below 30 seconds on the development
machine; if it exceeds that threshold, retain the 1,000-game benchmark as an
explicit command and add a 100-game default regression using the same seed and
only invariants that hold at that smaller sample size.

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run pytest tests/benchmarks/test_heuristic_tournament.py -q
```

If a separation assertion fails, inspect the recorded statistics, adjust only
the three named profile constants, rerun the focused heuristic tests, and
repeat the exact seed-42 benchmark. Record every accepted final coefficient in
the benchmark report.

- [ ] **Step 4: Benchmark all live charts and player counts**

Run at least 1,000 games for each chart/player-count combination with available
CPU workers. Record:

- elapsed time and games/second;
- outright/tied-first rates, mean rank, and mean final money;
- pass rate and mean nonzero bid;
- action wins and objective claims;
- any profile separation or chart-specific failure mode.

Write exact commands, commit SHA, Python version, worker count, seeds, and
tables to `docs/benchmarks/2026-07-28-heuristic-v1.md`.

- [ ] **Step 5: Update README**

Document:

- Bayesian resource belief and its exchangeability assumption;
- objective completion versus shaped progress;
- loan/investment/resource cash equations in plain language;
- profile meanings;
- CLI tournament examples;
- how to inspect `BidEvaluation`;
- development-only IDs;
- benchmark link;
- roadmap status.

- [ ] **Step 6: Run the complete quality gate**

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv lock --check
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run garboid-simulate --bots aggressive,balanced,passive \
    --games 10 --players 3 --ruleset live-A --seed 42 --workers 2
```

Expected: all gates pass and CLI output includes distinct behavior metrics.

- [ ] **Step 7: Request independent milestone review**

Ask a Sol extra-high reviewer to inspect:

- public-information leakage;
- belief conservation and hypergeometric correctness;
- objective completion/progress accounting;
- financial equations;
- reveal cross-suit conditioning;
- spawn picklability;
- metric definitions and benchmark claims.

Fix every Critical/Important issue with a failing regression test first.

- [ ] **Step 8: Commit and push the completed heuristic milestone**

```bash
git add \
  README.md \
  docs/benchmarks/2026-07-28-heuristic-v1.md \
  src/garboid_pocketrocks/simulator/cli.py \
  tests/benchmarks/test_heuristic_tournament.py \
  tests/test_integration.py
# If benchmark evidence changed the profile constants, also stage:
# git add src/garboid_pocketrocks/heuristics/profiles.py
git commit -m "feat: complete Bayesian heuristic bot milestone"
git push origin main
```

Watch the GitHub Actions run for the exact pushed SHA and do not begin neural
training until it is green.
