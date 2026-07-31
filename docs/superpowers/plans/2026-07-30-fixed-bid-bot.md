# Fixed-Bid Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic local simulation bot that bids 6 for one resource, 12 for two resources, 1 for either loan, and 7 for either investment.

**Architecture:** Put the stateless policy in a focused `bots/fixed_bid.py` module and expose it through a local-only `BotSpec`. Register that spec with simulation and tournament discovery while leaving the remote live launcher untouched.

**Tech Stack:** Python 3.14, PocketRocks SDK decision types, pytest, Ruff, mypy.

## Global Constraints

- Existing bot names, identities, policies, and latest aliases remain unchanged.
- The local simulation name and identity are both exactly `fixed-bid`.
- Bid targets are exactly 6 for `AUCTION1`, 12 for `AUCTION2`, 1 for `LOAN10` and `LOAN20`, and 7 for `INVEST5` and `INVEST10`.
- A positive legal maximum caps the target; a missing or nonpositive maximum produces a pass.
- Reveal decisions select index `0` when possible and pass otherwise.
- Unknown action identifiers produce a pass.
- Do not add a remote wrapper, live `BOT_ID`, live launcher entry, user configuration, or comparative-strength claim.
- Follow red-green-refactor: no production behavior is added before its focused test has failed for the expected reason.

---

### Task 1: Fixed-Bid Policy and Local Specification

**Files:**
- Create: `tests/bots/test_fixed_bid_bot.py`
- Create: `src/garboid_pocketrocks/bots/fixed_bid.py`

**Interfaces:**
- Consumes: `BotBrain.choose_decision(context: DecisionContext, ruleset: RulesetKnowledge) -> BotDecision` and `BotSpec.for_simulation(name: str, brain_factory: BrainFactory) -> BotSpec`.
- Produces: `FixedBidBotBrain`, `_target_bid(action_id: int) -> int | None`, and `FIXED_BID_BOT_SPEC`.

- [ ] **Step 1: Write failing decision and specification tests**

Create `tests/bots/test_fixed_bid_bot.py`:

```python
from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_BOT_SPEC, FixedBidBotBrain
from garboid_pocketrocks.knowledge import RulesetKnowledge


def _knowledge() -> RulesetKnowledge:
    return RulesetKnowledge(
        name="fixed-bid-test",
        player_count=3,
        starting_cash=30,
        private_cards_per_player=0,
        resource_counts=(2, 2, 2, 2, 2),
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def _context(
    *,
    action_id: ActionId | int = ActionId.AUCTION1,
    decision_kind: str = "submitBid",
    legal_max: int | None = 30,
    revealable_count: int = 0,
) -> DecisionContext:
    return DecisionContext(
        request_id="fixed-bid-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(action_id),
        current_resource_ids=(0, 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(),
        legal_max_amount=legal_max,
        revealable_count=revealable_count,
    )


@pytest.mark.parametrize(
    ("action_id", "expected"),
    (
        (ActionId.AUCTION1, 6),
        (ActionId.AUCTION2, 12),
        (ActionId.LOAN10, 1),
        (ActionId.LOAN20, 1),
        (ActionId.INVEST5, 7),
        (ActionId.INVEST10, 7),
    ),
)
def test_fixed_bid_brain_uses_action_target(action_id: ActionId, expected: int) -> None:
    context = _context(action_id=action_id)

    decision = FixedBidBotBrain().choose_decision(context, _knowledge())

    assert decision == BotDecision.submit_bid(expected)
    assert context.is_legal(decision)


@pytest.mark.parametrize(
    ("action_id", "legal_max"),
    (
        (ActionId.AUCTION1, 4),
        (ActionId.AUCTION2, 9),
        (ActionId.LOAN10, 0),
        (ActionId.INVEST5, 3),
    ),
)
def test_fixed_bid_brain_caps_target_at_legal_maximum(
    action_id: ActionId,
    legal_max: int,
) -> None:
    context = _context(action_id=action_id, legal_max=legal_max)

    decision = FixedBidBotBrain().choose_decision(context, _knowledge())

    expected = BotDecision.pass_turn() if legal_max == 0 else BotDecision.submit_bid(legal_max)
    assert decision == expected
    assert context.is_legal(decision)


@pytest.mark.parametrize("legal_max", (None, 0, -1))
def test_fixed_bid_brain_passes_without_positive_bid_limit(legal_max: int | None) -> None:
    context = _context(legal_max=legal_max)

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == BotDecision.pass_turn()


def test_fixed_bid_brain_passes_for_unknown_action() -> None:
    context = replace(_context(), current_action_id=999)

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == BotDecision.pass_turn()


@pytest.mark.parametrize(
    ("revealable_count", "expected"),
    (
        (2, BotDecision.select_info_to_reveal(0)),
        (0, BotDecision.pass_turn()),
    ),
)
def test_fixed_bid_brain_reveals_first_option_when_available(
    revealable_count: int,
    expected: BotDecision,
) -> None:
    context = _context(
        decision_kind="selectInfoToReveal",
        legal_max=None,
        revealable_count=revealable_count,
    )

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == expected


def test_fixed_bid_spec_is_local_deterministic_and_builds_fresh_brains() -> None:
    left = FIXED_BID_BOT_SPEC.make_brain(seed=11)
    right = FIXED_BID_BOT_SPEC.make_brain(seed=999)
    context = _context(action_id=ActionId.AUCTION2)

    assert FIXED_BID_BOT_SPEC.name == "fixed-bid"
    assert FIXED_BID_BOT_SPEC.bot_id == "fixed-bid"
    assert isinstance(left, FixedBidBotBrain)
    assert isinstance(right, FixedBidBotBrain)
    assert left is not right
    assert left.choose_decision(context, _knowledge()) == right.choose_decision(
        context,
        _knowledge(),
    )
```

- [ ] **Step 2: Run the focused test and verify the red state**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run pytest -n=0 tests/bots/test_fixed_bid_bot.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'garboid_pocketrocks.bots.fixed_bid'`.

- [ ] **Step 3: Implement the minimal fixed-bid policy**

Create `src/garboid_pocketrocks/bots/fixed_bid.py`:

```python
from __future__ import annotations

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge

_TARGET_BIDS = {
    ActionId.AUCTION1: 6,
    ActionId.AUCTION2: 12,
    ActionId.LOAN10: 1,
    ActionId.LOAN20: 1,
    ActionId.INVEST5: 7,
    ActionId.INVEST10: 7,
}


def _target_bid(action_id: int) -> int | None:
    try:
        action = ActionId(action_id)
    except ValueError:
        return None
    return _TARGET_BIDS.get(action)


class FixedBidBotBrain:
    """Deterministic baseline that bids a fixed amount for each action."""

    def __init__(self, seed: int | None = None) -> None:
        del seed

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            if context.revealable_count <= 0:
                return BotDecision.pass_turn()
            return BotDecision.select_info_to_reveal(0)

        legal_max = context.legal_max_amount
        target = _target_bid(context.current_action_id)
        if legal_max is None or legal_max <= 0 or target is None:
            return BotDecision.pass_turn()
        return BotDecision.submit_bid(min(target, legal_max))


FIXED_BID_BOT_SPEC = BotSpec.for_simulation("fixed-bid", FixedBidBotBrain)
```

- [ ] **Step 4: Run the focused test and verify the green state**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run pytest -n=0 tests/bots/test_fixed_bid_bot.py -q
```

Expected: all tests in `test_fixed_bid_bot.py` pass with no warnings.

- [ ] **Step 5: Commit the policy slice**

```bash
git add tests/bots/test_fixed_bid_bot.py src/garboid_pocketrocks/bots/fixed_bid.py
git commit -m "feat: add fixed-bid bot policy"
```

### Task 2: Package and Registry Integration

**Files:**
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `tests/bots/test_registry.py`
- Modify: `tests/bots/test_launcher.py`

**Interfaces:**
- Consumes: `FIXED_BID_BOT_SPEC` and `FixedBidBotBrain` from Task 1.
- Produces: public `garboid_pocketrocks.bots.FixedBidBotBrain`, registry lookup `BOT_SPECS_BY_NAME["fixed-bid"]`, and default tournament inclusion.

- [ ] **Step 1: Write failing package and registry tests**

Update `tests/bots/test_registry.py` so `test_registered_bot_specs_have_unique_names_and_ids` expects `fixed-bid` immediately after `random`, and so the default-field test expects it immediately after `random`:

```python
assert tuple(spec.name for spec in specs) == (
    "random",
    "fixed-bid",
    "aggressive",
    "balanced",
    "passive",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "vector_ppo_small_v1_g1500",
    "vector_ppo_large_v1_g350k",
)
```

```python
assert tuple(spec.name for spec in specs) == (
    "random",
    "fixed-bid",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "vector_ppo_small_v1_g1500",
    "vector_ppo_large_v1_g350k",
)
```

Add this assertion to `tests/bots/test_registry.py`:

```python
def test_fixed_bid_brain_is_exported_from_bots_package() -> None:
    from garboid_pocketrocks.bots import FixedBidBotBrain
    from garboid_pocketrocks.bots.fixed_bid import FixedBidBotBrain as DefinedBrain

    assert FixedBidBotBrain is DefinedBrain
```

Leave `tests/bots/test_launcher.py` unchanged; its existing
`test_registry_contains_every_live_wrapper_in_stable_order` is the regression
test proving the local-only bot does not enter the live launcher.

- [ ] **Step 2: Run registry and launcher tests and verify the red state**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run pytest -n=0 tests/bots/test_registry.py tests/bots/test_launcher.py -q
```

Expected: registry name assertions fail because `fixed-bid` is absent, and the
package export test fails with `ImportError`; the existing launcher test
continues to pass.

- [ ] **Step 3: Register and export the local bot**

In `src/garboid_pocketrocks/bots/__init__.py`, import the brain:

```python
from garboid_pocketrocks.bots.fixed_bid import FixedBidBotBrain
```

Add `"FixedBidBotBrain"` to `__all__`.

In `src/garboid_pocketrocks/bots/registry.py`, import the spec:

```python
from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_BOT_SPEC
```

Insert `FIXED_BID_BOT_SPEC` immediately after the random spec in `BOT_SPECS`.
Replace the index-dependent start of `DEFAULT_TOURNAMENT_BOT_SPECS` with:

```python
DEFAULT_TOURNAMENT_BOT_SPECS = (
    BOT_SPECS[0],
    FIXED_BID_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
```

Do not modify `src/garboid_pocketrocks/bots/launcher.py`.

- [ ] **Step 4: Run registry and launcher tests and verify the green state**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run pytest -n=0 tests/bots/test_registry.py tests/bots/test_launcher.py -q
```

Expected: all registry and launcher tests pass with no warnings.

- [ ] **Step 5: Commit the integration slice**

```bash
git add src/garboid_pocketrocks/bots/__init__.py src/garboid_pocketrocks/bots/registry.py tests/bots/test_registry.py
git commit -m "feat: register fixed-bid simulation bot"
```

### Task 3: User Documentation and Complete Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the registered `fixed-bid` simulation name from Task 2.
- Produces: user-facing simulation and tournament documentation.

- [ ] **Step 1: Document the fixed-bid baseline**

After the historical heuristic simulation example in `README.md`, add:

```markdown
Use `fixed-bid` as a deterministic baseline. It bids 6 for one resource, 12
for two resources, 1 for either loan, and 7 for either investment. If it
cannot afford the target, it bids the maximum legal amount.
```

Update the tournament registry description to say:

```markdown
The shared registry includes the random and fixed-bid baselines, the latest
unversioned heuristic profiles, the explicit v1 and v2 heuristic generations,
the frozen `vector_ppo_small_v1_g1500` smoke policy, and the large
`vector_ppo_large_v1_g350k` policy trained for exactly 349,860 games. The
curated default field uses random, fixed-bid, the six distinct versioned
heuristic policies, and both neural policies; it omits the unversioned aliases
because they duplicate v2 behavior.
```

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run pytest
```

Expected: the complete test suite passes with zero failures.

- [ ] **Step 3: Run lint and formatting verification**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run ruff check .
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run ruff format --check .
```

Expected: both commands exit zero with no lint or formatting violations.

- [ ] **Step 4: Run static type checking**

Run:

```bash
UV_CACHE_DIR=/tmp/garboid-pocketrocks-uv uv run mypy
```

Expected: mypy exits zero with no errors.

- [ ] **Step 5: Review the final diff and commit documentation**

Run:

```bash
git diff --check
git status --short
```

Confirm the diff contains only the approved fixed-bid bot, its tests,
registry/package integration, and README documentation. Then commit:

```bash
git add README.md
git commit -m "docs: describe fixed-bid baseline"
```
