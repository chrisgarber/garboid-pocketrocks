# SDK Greedy Value Tournament Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the PocketRocks SDK's published `GreedyValueBot` in Garboid's curated default tournament under the immutable local identity `sdk-greedy-value-v1`.

**Architecture:** A focused synchronous `BotBrain` adapter imports and invokes the pinned SDK sample policy directly. The adapter only drives SDK decision coroutines that complete without suspension, preserving tournament throughput and failing explicitly if the upstream contract changes. The existing registry and tournament scheduler then treat the versioned local spec like every other competitor.

**Tech Stack:** Python 3.14, PocketRocks Python SDK commit `51cad378ee1e70a78e39ebbb25957ea003444873`, pytest, Ruff, strict mypy, multiprocessing tournament runner.

## Global Constraints

- Keep the local simulation identity exactly `sdk-greedy-value-v1`.
- Keep `pocketrocks-python-sdk` pinned to commit `51cad378ee1e70a78e39ebbb25957ea003444873`.
- Import `GreedyValueBot` from `pocketrocks.sim.sample_bots`; do not copy or modify its policy.
- Use `BotSpec.for_simulation`; do not define a remote `BOT_ID` or add the bot to the live launcher.
- Preserve all existing bot identities, checkpoints, decisions, schedule logic, rating logic, and reporting logic.
- Do not add general asynchronous-bot support to the simulator.
- Use TDD: observe every new behavior test fail for the intended reason before implementing it.
- Do not run or commit a new full 15,000-game benchmark in this change.

---

### Task 1: Add the frozen synchronous SDK policy adapter

**Files:**
- Create: `src/garboid_pocketrocks/bots/sdk_samples.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Create: `tests/bots/test_sdk_samples.py`

**Interfaces:**
- Consumes: `GreedyValueBot.choose_decision(context) -> Coroutine[Any, Any, BotDecision]`, `BotBrain.choose_decision(context, ruleset) -> BotDecision`, and `BotSpec.for_simulation(name, brain_factory)`.
- Produces: `SdkGreedyValueV1Brain`, `SDK_GREEDY_VALUE_V1_BOT_SPEC`, and private `_run_immediate_decision(coroutine) -> BotDecision`.

- [ ] **Step 1: Write a contract test that fails inside the test because the adapter module is absent**

Create `tests/bots/test_sdk_samples.py` with the initial import contract:

```python
from __future__ import annotations

from importlib import import_module


def test_sdk_greedy_value_adapter_has_frozen_local_identity() -> None:
    module = import_module("garboid_pocketrocks.bots.sdk_samples")

    spec = module.SDK_GREEDY_VALUE_V1_BOT_SPEC
    assert spec.name == "sdk-greedy-value-v1"
    assert spec.bot_id == "sdk-greedy-value-v1"
    assert not hasattr(module.SdkGreedyValueV1Brain, "BOT_ID")
```

- [ ] **Step 2: Run the contract test and verify the intended red state**

Run:

```bash
uv run pytest -n=0 tests/bots/test_sdk_samples.py -q
```

Expected: `FAILED` in
`test_sdk_greedy_value_adapter_has_frozen_local_identity` with
`ModuleNotFoundError: No module named 'garboid_pocketrocks.bots.sdk_samples'`.
The failure occurs inside the test, not during test collection.

- [ ] **Step 3: Add only the public adapter shape needed by the contract test**

Create `src/garboid_pocketrocks/bots/sdk_samples.py`:

```python
from __future__ import annotations

from collections.abc import Coroutine
from typing import Any

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge


def _run_immediate_decision(
    coroutine: Coroutine[Any, Any, BotDecision],
) -> BotDecision:
    del coroutine
    raise NotImplementedError


class SdkGreedyValueV1Brain:
    """Frozen synchronous adapter for the SDK's first greedy-value sample policy."""

    def __init__(self, seed: int | None = None) -> None:
        del seed

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise NotImplementedError


SDK_GREEDY_VALUE_V1_BOT_SPEC = BotSpec.for_simulation(
    "sdk-greedy-value-v1",
    SdkGreedyValueV1Brain,
)
```

Do not export the configured spec from `garboid_pocketrocks.bots`. Add only the
brain import before the registry import in `src/garboid_pocketrocks/bots/__init__.py`:

```python
from garboid_pocketrocks.bots.sdk_samples import SdkGreedyValueV1Brain
```

Add `"SdkGreedyValueV1Brain"` to `__all__`.

- [ ] **Step 4: Run the contract test and verify the public shape is green**

Run:

```bash
uv run pytest -n=0 tests/bots/test_sdk_samples.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Expand the tests to pin behavior before implementing the bridge**

Replace `tests/bots/test_sdk_samples.py` with:

```python
from __future__ import annotations

import asyncio
import inspect
import pickle
from importlib import import_module
from pathlib import Path

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.sim.sample_bots import GreedyValueBot

from garboid_pocketrocks.bots.sdk_samples import (
    SDK_GREEDY_VALUE_V1_BOT_SPEC,
    SdkGreedyValueV1Brain,
    _run_immediate_decision,
)
from garboid_pocketrocks.knowledge import canonical_knowledge

SDK_REVISION = "51cad378ee1e70a78e39ebbb25957ea003444873"


def _bid_context(
    *,
    resources: tuple[int, int],
    hand: tuple[int, ...],
    legal_max: int,
    revealed_brick: int = 0,
) -> DecisionContext:
    return DecisionContext(
        request_id="sdk-greedy-value-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=resources,
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=(
            (revealed_brick, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def _reveal_context() -> DecisionContext:
    return DecisionContext(
        request_id="sdk-greedy-value-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="selectInfoToReveal",
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=(int(Suit.BRICK), 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(int(Suit.BRICK), int(Suit.WOOD)),
        legal_max_amount=None,
        revealable_count=2,
    )


def _sdk_decision(context: DecisionContext) -> BotDecision:
    bot = GreedyValueBot(api_key="test-key", bot_id="test-bot", reconnect=False)
    return asyncio.run(bot.choose_decision(context))


def test_sdk_greedy_value_adapter_has_frozen_local_identity() -> None:
    module = import_module("garboid_pocketrocks.bots.sdk_samples")

    spec = module.SDK_GREEDY_VALUE_V1_BOT_SPEC
    assert spec.name == "sdk-greedy-value-v1"
    assert spec.bot_id == "sdk-greedy-value-v1"
    assert not hasattr(module.SdkGreedyValueV1Brain, "BOT_ID")


def test_sdk_dependency_revision_is_pinned_for_v1_reproducibility() -> None:
    dependency = (
        "pocketrocks-python-sdk @ "
        f"git+https://github.com/chrisgarber/pocketrocks-python-sdk.git@{SDK_REVISION}"
    )

    assert f'"{dependency}",' in Path("pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("context", "expected"),
    (
        (
            _bid_context(
                resources=(int(Suit.BRICK), 0),
                hand=(int(Suit.BRICK), int(Suit.BRICK)),
                revealed_brick=1,
                legal_max=30,
            ),
            BotDecision.submit_bid(12),
        ),
        (
            _bid_context(
                resources=(int(Suit.WOOD), 0),
                hand=(int(Suit.BRICK),),
                legal_max=30,
            ),
            BotDecision.submit_bid(0),
        ),
        (
            _bid_context(
                resources=(int(Suit.BRICK), 0),
                hand=(int(Suit.BRICK), int(Suit.BRICK)),
                revealed_brick=1,
                legal_max=7,
            ),
            BotDecision.submit_bid(7),
        ),
        (_reveal_context(), BotDecision.select_info_to_reveal(0)),
    ),
)
def test_sdk_greedy_value_v1_pins_and_matches_sdk_decisions(
    context: DecisionContext,
    expected: BotDecision,
) -> None:
    actual = SdkGreedyValueV1Brain().choose_decision(context, canonical_knowledge(3))

    assert actual == expected
    assert actual == _sdk_decision(context)
    assert context.is_legal(actual)


def test_sdk_greedy_value_v1_spec_is_pickle_safe_and_deterministic() -> None:
    restored = pickle.loads(pickle.dumps(SDK_GREEDY_VALUE_V1_BOT_SPEC))
    context = _bid_context(
        resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.BRICK),),
        legal_max=30,
    )
    knowledge = canonical_knowledge(3)

    assert restored == SDK_GREEDY_VALUE_V1_BOT_SPEC
    assert restored.make_brain(seed=1).choose_decision(
        context, knowledge
    ) == restored.make_brain(seed=999).choose_decision(context, knowledge)


async def _suspending_decision() -> BotDecision:
    await asyncio.sleep(0)
    return BotDecision.pass_turn()


def test_immediate_bridge_rejects_and_closes_a_suspending_coroutine() -> None:
    coroutine = _suspending_decision()

    with pytest.raises(RuntimeError, match="must complete synchronously"):
        _run_immediate_decision(coroutine)

    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
```

- [ ] **Step 6: Run the expanded tests and verify the intended red state**

Run:

```bash
uv run pytest -n=0 tests/bots/test_sdk_samples.py -q
```

Expected: the identity and dependency-pin tests pass. Decision tests fail with
`NotImplementedError`, and the suspending-coroutine test fails because the
placeholder bridge does not raise the required explicit error.

- [ ] **Step 7: Implement the minimal no-suspension SDK bridge**

Replace `src/garboid_pocketrocks/bots/sdk_samples.py` with:

```python
from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, cast

from pocketrocks import BotDecision, DecisionContext
from pocketrocks.sim.sample_bots import GreedyValueBot

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge


def _run_immediate_decision(
    coroutine: Coroutine[Any, Any, BotDecision],
) -> BotDecision:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return cast(BotDecision, completed.value)
    coroutine.close()
    raise RuntimeError("SDK sample bot decisions must complete synchronously")


class SdkGreedyValueV1Brain:
    """Frozen synchronous adapter for the SDK's first greedy-value sample policy."""

    def __init__(self, seed: int | None = None) -> None:
        del seed
        self._bot = GreedyValueBot()

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        return _run_immediate_decision(self._bot.choose_decision(context))


SDK_GREEDY_VALUE_V1_BOT_SPEC = BotSpec.for_simulation(
    "sdk-greedy-value-v1",
    SdkGreedyValueV1Brain,
)
```

- [ ] **Step 8: Run the adapter tests and focused static checks**

Run:

```bash
uv run pytest -n=0 tests/bots/test_sdk_samples.py -q
uv run ruff format src/garboid_pocketrocks/bots/sdk_samples.py \
  src/garboid_pocketrocks/bots/__init__.py tests/bots/test_sdk_samples.py
uv run ruff check src/garboid_pocketrocks/bots/sdk_samples.py \
  src/garboid_pocketrocks/bots/__init__.py tests/bots/test_sdk_samples.py
uv run mypy src/garboid_pocketrocks/bots/sdk_samples.py tests/bots/test_sdk_samples.py
```

Expected: all adapter tests pass; Ruff and mypy exit zero.

- [ ] **Step 9: Commit the adapter**

```bash
git add src/garboid_pocketrocks/bots/sdk_samples.py \
  src/garboid_pocketrocks/bots/__init__.py tests/bots/test_sdk_samples.py
git commit -m "feat: adapt SDK greedy value bot"
```

### Task 2: Register the bot in the curated tournament and document it

**Files:**
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `tests/bots/test_registry.py`
- Modify: `tests/tournament/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `SDK_GREEDY_VALUE_V1_BOT_SPEC` from Task 1.
- Produces: `BOT_SPECS_BY_NAME["sdk-greedy-value-v1"]`, inclusion in `DEFAULT_TOURNAMENT_BOT_SPECS`, CLI-selectable simulation/tournament identity, and documented default-roster behavior.

- [ ] **Step 1: Update registry expectations before changing production registration**

In `tests/bots/test_registry.py`, add `"sdk-greedy-value-v1"` immediately after
`"passive-v2"` in both expected name tuples:

```python
assert tuple(spec.name for spec in specs) == (
    "random",
    "aggressive",
    "balanced",
    "passive",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "sdk-greedy-value-v1",
    "vector_ppo_small_v1_g1500",
    "vector_ppo_large_v1_g350k",
)
```

```python
assert tuple(spec.name for spec in specs) == (
    "random",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "sdk-greedy-value-v1",
    "vector_ppo_small_v1_g1500",
    "vector_ppo_large_v1_g350k",
)
```

In `tests/tournament/test_cli.py`, add `import json`, add
`"sdk-greedy-value-v1"` after `"passive-v2"` in
`test_bot_filters_use_curated_defaults_when_include_is_omitted`, and extend
`test_cli_runs_all_conditions_with_current_registry` after its artifact
assertions:

```python
summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
configured_names = tuple(item["name"] for item in summary["configuration"]["bots"])
sdk_row = next(
    row for row in summary["leaderboard"] if row["bot_name"] == "sdk-greedy-value-v1"
)

assert "sdk-greedy-value-v1" in configured_names
assert sdk_row["faults"] == 0
```

- [ ] **Step 2: Run the focused tests and verify registry assertions fail**

Run:

```bash
uv run pytest -n=0 tests/bots/test_registry.py \
  tests/tournament/test_cli.py::test_bot_filters_use_curated_defaults_when_include_is_omitted \
  tests/tournament/test_cli.py::test_cli_runs_all_conditions_with_current_registry -q
```

Expected: registry/default-name assertions fail because
`sdk-greedy-value-v1` is absent. The subprocess smoke test either fails to find
the SDK row or remains blocked behind the same missing registration.

- [ ] **Step 3: Register the versioned local spec**

In `src/garboid_pocketrocks/bots/registry.py`, import:

```python
from garboid_pocketrocks.bots.sdk_samples import SDK_GREEDY_VALUE_V1_BOT_SPEC
```

Add it after the six explicit heuristic generations and before the neural
policies in `BOT_SPECS`:

```python
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    SDK_GREEDY_VALUE_V1_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
```

Add it in the same position in `DEFAULT_TOURNAMENT_BOT_SPECS`:

```python
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    SDK_GREEDY_VALUE_V1_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
```

Do not modify the live launcher registry.

- [ ] **Step 4: Document the SDK policy and updated default field**

In the local simulation section of `README.md`, after the versioned heuristic
example, add:

````markdown
The pinned SDK also supplies `GreedyValueBot`, which bids according to the
value its private hand implies for the offered suits. Garboid imports that
policy directly under the frozen local identity `sdk-greedy-value-v1`:

```bash
uv run garboid-simulate \
  --bots sdk-greedy-value-v1,balanced-v2,passive-v2 \
  --games 1000 \
  --players 3 \
  --seed 42
```

This is a local SDK sample opponent, not a Garboid live bot, so it has no
remote bot ID.
````

In the multiplayer tournament registry paragraph, replace the roster
description with:

```markdown
The shared registry includes the random bot, the latest unversioned heuristic
profiles, the explicit v1 and v2 heuristic generations, the frozen SDK
`sdk-greedy-value-v1` policy, the frozen `vector_ppo_small_v1_g1500` smoke
policy, and the large `vector_ppo_large_v1_g350k` policy trained for exactly
349,860 games. The curated default field uses random, the six distinct
versioned heuristic policies, the SDK greedy-value policy, and both neural
policies; it omits the unversioned aliases because they duplicate v2 behavior.
Use `--bots` or `--exclude-bots` for a reproducible subset and
`--bootstrap-samples 0` for quick experiments.
```

- [ ] **Step 5: Run registry, CLI, package-boundary, and simulator selection tests**

Run:

```bash
uv run pytest -n=0 tests/bots/test_registry.py tests/tournament/test_cli.py \
  tests/simulator/test_cli.py tests/test_package.py -q
```

Expected: all tests pass. The tournament subprocess writes CSV, JSON, and HTML;
the summary includes `sdk-greedy-value-v1` with zero faults.

- [ ] **Step 6: Run repository-wide formatting and static validation**

Run:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
```

Expected: every command exits zero with no warnings or errors.

- [ ] **Step 7: Run the full test suite with optional neural policies available**

Run:

```bash
uv run --extra neural pytest -q
```

Expected: the full suite passes with no failures, errors, or warnings caused by
the new adapter.

- [ ] **Step 8: Verify the final diff and commit the tournament integration**

Run:

```bash
git diff --check
git status --short
git diff -- src/garboid_pocketrocks/bots/registry.py \
  tests/bots/test_registry.py tests/tournament/test_cli.py README.md
```

Confirm that no SDK dependency, live bot ID, launcher, neural checkpoint,
schedule, rating, or reporting file changed.

Then commit:

```bash
git add src/garboid_pocketrocks/bots/registry.py tests/bots/test_registry.py \
  tests/tournament/test_cli.py README.md
git commit -m "feat: add SDK greedy value bot to tournament"
```
