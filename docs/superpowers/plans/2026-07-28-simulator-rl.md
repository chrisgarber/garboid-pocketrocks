# Deterministic Simulator and RL Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live-rules-compatible deterministic PocketRocks engine, synchronous bot brains, replayable Monte Carlo evaluation, and standard multi-agent and single-agent RL environments with configurable rulesets.

**Architecture:** A pure stepwise engine owns all rules and produces SDK `DecisionContext` batches while consuming SDK `BotDecision` values. Live SDK bots delegate to synchronous brains; match runners and RL adapters drive the same engine without transport or credentials. Immutable rules, state, events, and explicit seed derivation make games replayable and independent of worker scheduling.

**Tech Stack:** Python 3.14, uv, official pinned PocketRocks SDK, frozen dataclasses, pytest, Hypothesis, NumPy, Gymnasium, PettingZoo, Ruff, and strict mypy.

## Global Constraints

- Python remains `>=3.14,<3.15`, selected by mise and locked by uv.
- Keep the SDK pinned to commit `597857446d47ac0890609a4767cad561578a2519`.
- The live rules and SDK are authoritative; do not copy or depend on the competition engine.
- Use SDK `Suit`, `ActionId`, `OBJECTIVES`, `DecisionContext`, and `BotDecision` at public boundaries.
- The engine is synchronous, deterministic, strict, and independent of bot implementations.
- Every simulated brain is fresh per seat and game.
- `RandomBot.BOT_ID` is the public constant `bot_e0e2c541-1615-4f47-983c-224e7d888d89`.
- API keys remain only in ignored `.env`; tests never inspect or connect with live credentials.
- Policy observations may include public `RulesetKnowledge` but never hidden cards, deck order, or engine RNG state.
- Rulesets may vary numeric and composition fields, but action meanings, tie-breaking, reveals, and scoring retain live semantics.
- Follow TDD: demonstrate each new behavior failing before implementing it.
- Preserve strict mypy and the existing full quality gate.

## Execution Order and Parallel Boundaries

```text
Task 1: dependencies + rulesets
        |
        +---- Task 2: bot identity/brain refactor
        |
        +---- Task 3: immutable state/events/setup
                       |
                       Task 4: engine/context/transitions
                       |
                       Task 5: conformance/invariants
                              /                    \
              Task 6 -> Task 7              Task 8 -> Task 9
             replay    Monte Carlo          RL codec   environments
                              \                    /
                               Task 10: CLI/docs/integration
```

After Task 1, Tasks 2 and 3 may use separate subagents because their file
ownership does not overlap. After Task 5, the Task 6–7 runner branch and Task
8–9 RL branch may use separate subagents. Task 10 stays with the primary agent
for integration and review.

---

### Task 1: Dependencies, Ruleset Model, and Live Presets

**Files:**
- Create: `src/garboid_pocketrocks/rules.py`
- Create: `tests/test_rules.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: SDK `Suit`, `ActionId`, and `OBJECTIVES`
- Produces: `PlayerSetup`, `Ruleset`, `RulesetKnowledge`, `LIVE_RULESET`,
  `VALUE_CHARTS`, and `RulesetValidationError`
- Produces dependencies used by later tasks: `hypothesis`, `numpy`,
  `gymnasium`, and `pettingzoo`

- [ ] **Step 1: Add failing live-preset and validation tests**

Create `tests/test_rules.py` with these concrete cases:

```python
from dataclasses import replace

import pytest
from pocketrocks import ActionId

from garboid_pocketrocks.rules import (
    LIVE_RULESET,
    VALUE_CHARTS,
    PlayerSetup,
    Ruleset,
    RulesetValidationError,
)


def test_live_ruleset_matches_current_server_rules() -> None:
    assert LIVE_RULESET.resource_counts == (6, 6, 6, 6, 6)
    assert LIVE_RULESET.action_count(ActionId.AUCTION1) == 12
    assert LIVE_RULESET.action_count(ActionId.AUCTION2) == 8
    assert LIVE_RULESET.action_count(ActionId.LOAN10) == 3
    assert LIVE_RULESET.action_count(ActionId.LOAN20) == 2
    assert LIVE_RULESET.action_count(ActionId.INVEST5) == 3
    assert LIVE_RULESET.action_count(ActionId.INVEST10) == 2
    assert LIVE_RULESET.setup_for(3) == PlayerSetup(3, 30, 5)
    assert LIVE_RULESET.setup_for(4) == PlayerSetup(4, 25, 4)
    assert LIVE_RULESET.setup_for(5) == PlayerSetup(5, 20, 3)
    assert LIVE_RULESET.value_chart == VALUE_CHARTS["A"]
    assert len(LIVE_RULESET.objective_pool) == 30
    assert LIVE_RULESET.active_objective_count == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resource_counts", (6, 6), "five resource counts"),
        ("action_counts", (12, 8), "six action counts"),
        ("value_chart", (0, 4), "six value-chart buckets"),
        ("active_objective_count", 31, "active objective count"),
    ],
)
def test_ruleset_rejects_invalid_shapes(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(RulesetValidationError, match=message):
        replace(LIVE_RULESET, **{field: value})


def test_ruleset_rejects_insufficient_auction_capacity() -> None:
    with pytest.raises(RulesetValidationError, match="auction capacity"):
        replace(
            LIVE_RULESET,
            action_counts=(1, 0, 0, 0, 0, 0),
        )


def test_ruleset_knowledge_resolves_player_setup() -> None:
    knowledge = LIVE_RULESET.knowledge(4)
    assert knowledge.player_count == 4
    assert knowledge.starting_cash == 25
    assert knowledge.private_cards_per_player == 4
    assert knowledge.resource_counts == LIVE_RULESET.resource_counts
    assert knowledge.action_counts == LIVE_RULESET.action_counts
```

- [ ] **Step 2: Run the rules tests and verify the red state**

Run:

```bash
uv run pytest tests/test_rules.py -q
```

Expected: collection fails because `garboid_pocketrocks.rules` does not exist.

- [ ] **Step 3: Add dependencies**

Run:

```bash
uv add numpy gymnasium pettingzoo
uv add --dev hypothesis
```

Expected: `pyproject.toml` declares `numpy`, `gymnasium`, and `pettingzoo` as
runtime dependencies and `hypothesis` as a development dependency; `uv.lock`
resolves successfully under Python 3.14.

- [ ] **Step 4: Implement immutable rules and validation**

Create `src/garboid_pocketrocks/rules.py` with these public types and constants:

```python
from __future__ import annotations

from dataclasses import dataclass

from pocketrocks import ActionId, OBJECTIVES


class RulesetValidationError(ValueError):
    """Raised before setup when a ruleset cannot produce a valid game."""


@dataclass(frozen=True, slots=True)
class PlayerSetup:
    player_count: int
    starting_cash: int
    private_cards_per_player: int


@dataclass(frozen=True, slots=True)
class RulesetKnowledge:
    name: str
    player_count: int
    starting_cash: int
    private_cards_per_player: int
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int
    objectives_enabled: bool


@dataclass(frozen=True, slots=True)
class Ruleset:
    name: str
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    player_setups: tuple[PlayerSetup, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int = 4
    objectives_enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.resource_counts) != 5:
            raise RulesetValidationError("ruleset requires five resource counts")
        if any(count < 0 for count in self.resource_counts):
            raise RulesetValidationError("resource counts must be nonnegative")
        if len(self.action_counts) != 6:
            raise RulesetValidationError("ruleset requires six action counts")
        if any(count < 0 for count in self.action_counts):
            raise RulesetValidationError("action counts must be nonnegative")
        if len(self.value_chart) != 6:
            raise RulesetValidationError("ruleset requires six value-chart buckets")
        if len(set(self.objective_pool)) != len(self.objective_pool):
            raise RulesetValidationError("objective IDs must be unique")
        if any(objective_id not in OBJECTIVES for objective_id in self.objective_pool):
            raise RulesetValidationError("objective pool contains an unknown ID")
        if not 0 <= self.active_objective_count <= len(self.objective_pool):
            raise RulesetValidationError("active objective count exceeds objective pool")
        if not self.objectives_enabled and self.active_objective_count != 0:
            raise RulesetValidationError(
                "disabled objectives require active objective count zero"
            )
        for setup in self.player_setups:
            self._validate_setup(setup)

    def _validate_setup(self, setup: PlayerSetup) -> None:
        if not 3 <= setup.player_count <= 5:
            raise RulesetValidationError("player count must be between 3 and 5")
        if setup.starting_cash <= 0:
            raise RulesetValidationError("starting cash must be positive")
        if setup.private_cards_per_player < 0:
            raise RulesetValidationError("private-card count must be nonnegative")
        biddable = sum(self.resource_counts) - (
            setup.player_count * setup.private_cards_per_player
        )
        if biddable <= 0:
            raise RulesetValidationError("setup must leave a biddable resource")
        auction_capacity = self.action_counts[ActionId.AUCTION1 - 1] + (
            2 * self.action_counts[ActionId.AUCTION2 - 1]
        )
        if auction_capacity < biddable:
            raise RulesetValidationError("action deck has insufficient auction capacity")

    def setup_for(self, player_count: int) -> PlayerSetup:
        for setup in self.player_setups:
            if setup.player_count == player_count:
                return setup
        raise RulesetValidationError(
            f"ruleset {self.name!r} does not support {player_count} players"
        )

    def action_count(self, action_id: ActionId) -> int:
        return self.action_counts[int(action_id) - 1]

    def knowledge(self, player_count: int) -> RulesetKnowledge:
        setup = self.setup_for(player_count)
        return RulesetKnowledge(
            name=self.name,
            player_count=player_count,
            starting_cash=setup.starting_cash,
            private_cards_per_player=setup.private_cards_per_player,
            resource_counts=self.resource_counts,
            action_counts=self.action_counts,
            value_chart=self.value_chart,
            objective_pool=self.objective_pool,
            active_objective_count=self.active_objective_count,
            objectives_enabled=self.objectives_enabled,
        )


VALUE_CHARTS: dict[str, tuple[int, ...]] = {
    "A": (0, 4, 8, 12, 16, 20),
    "B": (20, 16, 12, 8, 4, 0),
    "C": (0, 2, 5, 9, 14, 20),
    "D": (20, 18, 15, 11, 6, 0),
    "E": (0, 4, 10, 18, 6, 0),
}

LIVE_RULESET = Ruleset(
    name="live-A",
    resource_counts=(6, 6, 6, 6, 6),
    action_counts=(12, 8, 3, 2, 3, 2),
    player_setups=(
        PlayerSetup(3, 30, 5),
        PlayerSetup(4, 25, 4),
        PlayerSetup(5, 20, 3),
    ),
    value_chart=VALUE_CHARTS["A"],
    objective_pool=tuple(sorted(OBJECTIVES)),
    active_objective_count=4,
)
```

Also export `live_ruleset(chart: str = "A", objectives_enabled: bool = True) ->
Ruleset`, implemented with `dataclasses.replace`, so callers can select charts
B–E and disable objectives without mutating the preset. When objectives are
disabled, it must also set `active_objective_count=0`.

- [ ] **Step 5: Run focused and static checks**

Run:

```bash
uv run pytest tests/test_rules.py -q
uv run ruff check src/garboid_pocketrocks/rules.py tests/test_rules.py
uv run mypy src/garboid_pocketrocks/rules.py tests/test_rules.py
```

Expected: all rules tests pass and both static checks report no issues.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/garboid_pocketrocks/rules.py tests/test_rules.py
git commit -m "feat: add configurable PocketRocks rulesets"
```

---

### Task 2: Synchronous Brain Contract and Static Bot Identity

**Files:**
- Create: `src/garboid_pocketrocks/bots/base.py`
- Modify: `src/garboid_pocketrocks/bots/random_bot.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `tests/bots/test_random_bot.py`
- Modify: `.env.example`
- Modify: local ignored `.env` without reading or printing its API-key value
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `Ruleset`, `RulesetKnowledge`, and `LIVE_RULESET` from Task 1
- Produces: `BotBrain`, `BrainFactory`, `BotSpec`, `PocketRocksFastBot`,
  `RandomBotBrain`, and refactored `RandomBot`

- [ ] **Step 1: Replace async-logic tests with brain and bridge tests**

Extend `tests/bots/test_random_bot.py` with:

```python
from garboid_pocketrocks.bots.base import BotSpec, PocketRocksFastBot
from garboid_pocketrocks.bots.random_bot import RandomBot, RandomBotBrain
from garboid_pocketrocks.rules import LIVE_RULESET

RANDOM_BOT_ID = "bot_e0e2c541-1615-4f47-983c-224e7d888d89"


def test_random_bot_has_static_public_identity() -> None:
    assert issubclass(RandomBot, PocketRocksFastBot)
    assert RandomBot.BOT_ID == RANDOM_BOT_ID
    assert RandomBot.BOT_NAME == "random"


def test_random_brain_and_async_bridge_return_same_decision() -> None:
    context = _bid_context(7)
    brain = RandomBotBrain(seed=42)
    expected = brain.choose_decision(context, LIVE_RULESET.knowledge(3))
    bot = _bot(seed=42)

    assert _choose(bot, context) == expected
    assert bot.choose_decision_sync(context) == RandomBotBrain(
        seed=42
    ).choose_decision(context, LIVE_RULESET.knowledge(3))


def test_bot_spec_builds_fresh_brains() -> None:
    spec = BotSpec.from_bot_class(RandomBot)
    left = spec.make_brain(seed=11)
    right = spec.make_brain(seed=11)
    assert left is not right
    assert spec.bot_id == RANDOM_BOT_ID
```

Delete the prior dotenv-specific test
`test_main_loads_random_bot_id_from_dotenv`; preserve all legality,
reproducibility, and fake-transport tests.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
uv run pytest tests/bots/test_random_bot.py -q
```

Expected: collection fails because `bots.base` and `RandomBotBrain` do not
exist.

- [ ] **Step 3: Implement the shared brain and SDK bridge**

Create `src/garboid_pocketrocks/bots/base.py` with:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from pocketrocks import BotDecision, DecisionContext, PocketRocksBot

from garboid_pocketrocks.rules import LIVE_RULESET, Ruleset, RulesetKnowledge


class BotBrain(Protocol):
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        """Return one synchronous SDK decision."""


BrainFactory = Callable[[int | None], BotBrain]


class PocketRocksFastBot(PocketRocksBot):
    BOT_ID: ClassVar[str]
    BOT_NAME: ClassVar[str]

    def __init__(
        self,
        *,
        seed: int | None = None,
        brain: BotBrain | None = None,
        ruleset: Ruleset = LIVE_RULESET,
        **sdk_options: Any,
    ) -> None:
        sdk_options.setdefault("bot_id", self.BOT_ID)
        super().__init__(**sdk_options)
        self._brain = brain if brain is not None else self.build_brain(seed)
        self._ruleset = ruleset

    @classmethod
    def build_brain(cls, seed: int | None) -> BotBrain:
        raise NotImplementedError

    def choose_decision_sync(self, context: DecisionContext) -> BotDecision:
        knowledge = self._ruleset.knowledge(context.player_count)
        return self._brain.choose_decision(context, knowledge)

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
        return self.choose_decision_sync(context)


@dataclass(frozen=True, slots=True)
class BotSpec:
    name: str
    bot_id: str
    brain_factory: BrainFactory

    @classmethod
    def from_bot_class(
        cls,
        bot_class: type[PocketRocksFastBot],
    ) -> BotSpec:
        return cls(
            name=bot_class.BOT_NAME,
            bot_id=bot_class.BOT_ID,
            brain_factory=bot_class.build_brain,
        )

    def make_brain(self, *, seed: int | None = None) -> BotBrain:
        return self.brain_factory(seed)
```

Refactor `random_bot.py` so `RandomBotBrain` owns the existing RNG and exact
decision behavior, and `RandomBot` only declares identity and builds the brain:

```python
class RandomBotBrain:
    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "submitBid":
            max_amount = context.legal_max_amount
            if max_amount is None or max_amount <= 0:
                return BotDecision.pass_turn()
            amount = self._random.randint(0, max_amount)
            return (
                BotDecision.pass_turn()
                if amount == 0
                else BotDecision.submit_bid(amount)
            )
        if context.revealable_count <= 0:
            return BotDecision.pass_turn()
        return BotDecision.select_info_to_reveal(
            self._random.randrange(context.revealable_count)
        )


class RandomBot(PocketRocksFastBot):
    BOT_ID = "bot_e0e2c541-1615-4f47-983c-224e7d888d89"
    BOT_NAME = "random"

    @classmethod
    def build_brain(cls, seed: int | None) -> RandomBotBrain:
        return RandomBotBrain(seed=seed)
```

Remove `_random_bot_id`, `os`, and `dotenv` imports. `main()` returns to
`RandomBot(seed=args.seed).run()`.

- [ ] **Step 4: Move bot identity out of environment files**

In `.env.example`, remove `RANDOM_BOT_ID` and explain that concrete bot IDs are
public class constants. Apply the same one-line removal to ignored `.env`
without reading or printing `POCKETROCKS_API_KEY`. Remove direct
`python-dotenv` from `pyproject.toml`, then run:

```bash
uv lock
```

Expected: the SDK retains its transitive dotenv dependency while this project
no longer imports it directly.

- [ ] **Step 5: Run bot tests and the installed command check**

Run:

```bash
uv run pytest tests/bots/test_random_bot.py -q
uv run garboid-random-bot --help
uv run ruff check src/garboid_pocketrocks/bots tests/bots
uv run mypy src/garboid_pocketrocks/bots tests/bots
```

Expected: all existing and new bot tests pass; help exits zero; static checks
pass. Do not make a live network connection.

- [ ] **Step 6: Commit**

```bash
git add .env.example pyproject.toml uv.lock src/garboid_pocketrocks/bots tests/bots
git commit -m "refactor: separate bot identity and synchronous brain"
```

---

### Task 3: Immutable Game State, Domain Events, and Seeded Setup

**Files:**
- Create: `src/garboid_pocketrocks/simulator/model.py`
- Create: `src/garboid_pocketrocks/simulator/events.py`
- Create: `src/garboid_pocketrocks/simulator/setup.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Create: `tests/simulator/test_setup.py`
- Create: `tests/simulator/test_invariants.py`

**Interfaces:**
- Consumes: `Ruleset` and SDK `Suit`/`ActionId`
- Produces: `Phase`, `ResourceCard`, `ActionCard`, `LoanPosition`,
  `InvestmentPosition`, `PlayerState`, `GameState`, `GameResult`,
  `EventKind`, `GameEvent`, `SetupResult`, and `build_setup`

- [ ] **Step 1: Write failing deterministic setup tests**

Create `tests/simulator/test_setup.py`:

```python
from collections import Counter

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.model import Phase
from garboid_pocketrocks.simulator.setup import build_setup


def test_setup_is_reproducible_and_conserves_cards() -> None:
    left = build_setup(LIVE_RULESET, player_count=3, seed=123)
    right = build_setup(LIVE_RULESET, player_count=3, seed=123)

    assert left == right
    assert left.state.phase is Phase.BIDDING
    all_cards = [
        *left.state.resource_deck,
        *left.state.visible_resources,
        *(card for player in left.state.players for card in player.private_hand),
    ]
    assert len(all_cards) == 30
    assert len({card.card_id for card in all_cards}) == 30
    assert set(Counter(card.suit for card in all_cards).values()) == {6}


def test_live_setup_counts_by_player_count() -> None:
    for players, cash, hand_size in ((3, 30, 5), (4, 25, 4), (5, 20, 3)):
        setup = build_setup(LIVE_RULESET, player_count=players, seed=7)
        assert len(setup.state.players) == players
        assert {player.cash for player in setup.state.players} == {cash}
        assert {len(player.private_hand) for player in setup.state.players} == {
            hand_size
        }
        assert len(setup.state.visible_resources) == 2
        assert len(setup.state.active_objective_ids) == 4
        assert len(setup.state.action_deck) == 29
```

- [ ] **Step 2: Run setup tests and verify they fail**

Run:

```bash
uv run pytest tests/simulator/test_setup.py -q
```

Expected: collection fails because the simulator model and setup modules do not
exist.

- [ ] **Step 3: Implement immutable model types**

Create `model.py` with frozen, slotted dataclasses:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import ActionId, Suit

from garboid_pocketrocks.rules import Ruleset


Seat = int


class Phase(StrEnum):
    BIDDING = "bidding"
    REVEAL = "reveal"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ResourceCard:
    card_id: int
    suit: Suit


@dataclass(frozen=True, slots=True)
class ActionCard:
    card_id: int
    action_id: ActionId


@dataclass(frozen=True, slots=True)
class LoanPosition:
    principal: int
    winning_bid: int


@dataclass(frozen=True, slots=True)
class InvestmentPosition:
    locked: int
    payout: int


@dataclass(frozen=True, slots=True)
class PlayerState:
    seat: Seat
    cash: int
    private_hand: tuple[ResourceCard, ...] = ()
    revealed_info: tuple[ResourceCard, ...] = ()
    won_resources: tuple[ResourceCard, ...] = ()
    loans: tuple[LoanPosition, ...] = ()
    investments: tuple[InvestmentPosition, ...] = ()
    owned_objective_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    ruleset: Ruleset
    player_count: int
    seed: int
    turn_index: int
    phase: Phase
    players: tuple[PlayerState, ...]
    resource_deck: tuple[ResourceCard, ...]
    action_deck: tuple[ActionCard, ...]
    visible_resources: tuple[ResourceCard, ...]
    current_action: ActionCard | None
    active_objective_ids: tuple[int, ...]
    priority_seat: Seat
    reveal_seat: Seat | None = None


@dataclass(frozen=True, slots=True)
class Score:
    seat: Seat
    final_money: int
    rank: int


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[Score, ...]
```

Create `events.py` with a frozen `EventKind(StrEnum)` covering every event named
in the design and a frozen `GameEvent` whose optional typed fields are:
`turn_index`, `seat`, `action_id`, `amount`, `resource_ids`,
`objective_ids`, `scores`, and `automatic`.

- [ ] **Step 4: Implement deterministic setup**

Create `setup.py`:

```python
@dataclass(frozen=True, slots=True)
class SetupResult:
    state: GameState
    events: tuple[GameEvent, ...]


def build_setup(
    ruleset: Ruleset,
    *,
    player_count: int,
    seed: int,
) -> SetupResult:
```

Implement the body in this exact order:

1. Resolve `ruleset.setup_for(player_count)`.
2. Build resource cards in ascending suit then copy order and action cards in
   ascending `ActionId` then copy order.
3. Shuffle both lists with one local `random.Random(seed)`.
4. Deal private cards one at a time in seat order until every hand reaches the
   configured size.
5. Move up to two cards to `visible_resources`.
6. Shuffle a copy of the objective pool with the same RNG and take the first
   active count, or none when disabled.
7. Choose the initial priority seat with `rng.randrange(player_count)`.
8. Move the first action to `current_action`; retain the rest in `action_deck`.
9. Return a bidding `GameState` plus setup and turn-opened events.

Never use module-global randomness.

- [ ] **Step 5: Add invariant helpers and tests**

In `tests/simulator/test_invariants.py`, define reusable assertions
`assert_resource_conservation(state)` and `assert_objective_ownership(state)`.
Test the initial state for every player count and ten seeds. The card assertion
must compare exact per-suit totals to `ruleset.resource_counts`, not only the
overall count.

- [ ] **Step 6: Run focused tests and static checks**

Run:

```bash
uv run pytest tests/simulator/test_setup.py tests/simulator/test_invariants.py -q
uv run ruff check src/garboid_pocketrocks/simulator tests/simulator
uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: setup and invariant tests pass; static checks pass.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "feat: add deterministic simulator state and setup"
```

---

### Task 4: SDK Context Adapter and Complete Engine Transitions

**Files:**
- Create: `src/garboid_pocketrocks/simulator/errors.py`
- Create: `src/garboid_pocketrocks/simulator/context.py`
- Create: `src/garboid_pocketrocks/simulator/engine.py`
- Modify: `src/garboid_pocketrocks/simulator/model.py`
- Modify: `src/garboid_pocketrocks/simulator/events.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Create: `tests/simulator/test_context.py`
- Create: `tests/simulator/test_engine.py`
- Create: `tests/simulator/test_scoring.py`

**Interfaces:**
- Consumes: Task 3 state/setup/events and SDK decision/reference types
- Produces: `SimulationError`, `WrongPhase`, `ActingSeatMismatch`,
  `IllegalDecision`, `DecisionBatch`, `EngineTransition`, and `GameEngine`

- [ ] **Step 1: Add failing context and bid-resolution tests**

Create tests that assert:

```python
def test_bidding_context_matches_live_loan_limit() -> None:
    transition = GameEngine.resume(
        state_with_action(ActionId.LOAN20, cash=(7, 12, 30))
    )
    contexts = transition.pending.contexts_by_seat
    assert contexts[0].legal_max_amount == 27
    assert contexts[1].legal_max_amount == 32
    assert contexts[2].legal_max_amount == 50
    assert all(context.current_resource_ids == (0, 0) for context in contexts.values())


def test_tie_scans_after_priority_and_updates_marker() -> None:
    transition = bidding_transition(priority_seat=1, action=ActionId.INVEST5)
    resolved = GameEngine.step(
        transition.state,
        {0: bid(4), 1: bid(1), 2: bid(4)},
    )
    assert resolved.state.priority_seat == 2
    assert resolved.state.players[2].cash == 26


def test_all_passes_award_action_for_zero() -> None:
    transition = bidding_transition(priority_seat=2, action=ActionId.LOAN10)
    resolved = GameEngine.step(
        transition.state,
        {seat: BotDecision.pass_turn() for seat in range(3)},
    )
    assert resolved.state.priority_seat == 0
    assert resolved.state.players[0].cash == 40
```

Provide fixture builders in `tests/simulator/helpers.py` rather than mutating
private engine attributes.

- [ ] **Step 2: Run the targeted tests and verify the red state**

Run:

```bash
uv run pytest \
  tests/simulator/test_context.py \
  tests/simulator/test_engine.py \
  tests/simulator/test_scoring.py -q
```

Expected: collection fails because `GameEngine` and context types do not exist.

- [ ] **Step 3: Implement errors and context construction**

`context.py` must expose:

```python
@dataclass(frozen=True, slots=True)
class DecisionBatch:
    phase: Phase
    contexts: tuple[tuple[Seat, DecisionContext], ...]

    @property
    def contexts_by_seat(self) -> dict[Seat, DecisionContext]:
        return dict(self.contexts)

    @property
    def acting_seats(self) -> tuple[Seat, ...]:
        return tuple(seat for seat, _ in self.contexts)


def build_decision_batch(state: GameState) -> DecisionBatch:
```

For bidding, build one context per seat. For reveal, build only the winner's
context. Use deterministic request IDs
`sim-{seed}-{turn_index}-{phase}-{seat}`, `received_at=0`, and
`deadline_at=2**63 - 1`. Derive all matrices from state in seat order.
Financial actions expose `(0, 0)` resources. Auction 1 pads its second resource
with zero. Loan maximums add principal; other maxima equal cash.

In `errors.py`, make all specific errors subclass `SimulationError` and include
phase/seat/value in messages.

- [ ] **Step 4: Implement engine start, validation, and bidding transitions**

`engine.py` must expose:

```python
@dataclass(frozen=True, slots=True)
class EngineTransition:
    state: GameState
    events: tuple[GameEvent, ...]
    pending: DecisionBatch | None
    result: GameResult | None

    @property
    def terminated(self) -> bool:
        return self.result is not None


```

`GameEngine` provides three static entry points:

- `start(ruleset: Ruleset, *, player_count: int, seed: int) -> EngineTransition`
  validates the ruleset, performs setup, and returns the first pending batch;
- `resume(state: GameState) -> EngineTransition` builds a pending batch from an
  existing immutable state without changing it, for focused tests and replay
  diagnostics;
- `step(state: GameState, decisions_by_seat: Mapping[Seat, BotDecision]) ->
  EngineTransition` validates and applies one decision step.

Implement bidding behavior in this order:

1. Require bidding phase and exactly every occupied seat.
2. Validate each decision with that seat's context.
3. Convert pass to zero.
4. Select the maximum bid and scan from `priority_seat + 1`, wrapping.
5. Deduct the winner's bid.
6. Apply Auction 1, Auction 2, Loan 10/20, or Invest 5/10 exactly as specified.
7. Claim all newly satisfied active objectives in active-objective order.
8. Set the winner as priority.
9. Enter reveal phase if the winner has a private card; otherwise advance.

Use `dataclasses.replace` and tuple reconstruction; never mutate a `GameState`
or `PlayerState`.

- [ ] **Step 5: Implement reveal, turn advancement, termination, and scoring**

Reveal behavior:

- require only `reveal_seat`;
- accept a valid SDK reveal index;
- map pass to index zero;
- move exactly that card from private hand to revealed information;
- increment turn and open the next action.

Turn advancement:

- preserve visible resources across financial actions;
- after auctions, refill visible resources to two while cards remain;
- terminate only when both the face-down resource deck and visible resources
  are empty;
- otherwise consume the next action card;
- raise `SimulationError` if a validated ruleset nevertheless exhausts actions.

Terminal scoring must use:

```python
resource_value = sum(
    count_owned(player, suit)
    * state.ruleset.value_chart[min(total_revealed(state, suit), 5)]
    for suit in Suit
)
final_money = (
    player.cash
    + resource_value
    + sum(objective_payout(oid) or 0 for oid in player.owned_objective_ids)
    + sum(position.locked + position.payout for position in player.investments)
    - sum(position.principal for position in player.loans)
)
```

Reveal every remaining private card before calculating totals. Assign shared
competition ranks with `rank = 1 + count(players with strictly greater money)`;
equal totals receive the same rank, so totals `$10, $10, $5` produce ranks
`1, 1, 3`.

- [ ] **Step 6: Expand tests across every live action and terminal edge**

Add parameterized assertions for:

- Auction 1 and Auction 2 resource movement/refill;
- final Auction 2 with one remaining resource;
- each loan's legal maximum and score repayment;
- each investment's lock and final return;
- immediate, exclusive, and multiple objective claims;
- selected reveal and pass auto-reveal;
- wrong seat sets and illegal decisions raising their exact domain errors;
- terminal resource valuation for charts A–E;
- tied final scores receiving equal rank.

- [ ] **Step 7: Run engine tests and static checks**

Run:

```bash
uv run pytest tests/simulator/test_context.py tests/simulator/test_engine.py tests/simulator/test_scoring.py -q
uv run ruff check src/garboid_pocketrocks/simulator tests/simulator
uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: all engine cases pass and static checks report no issues.

- [ ] **Step 8: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "feat: implement deterministic PocketRocks engine"
```

---

### Task 5: SDK Conformance and Generated Invariant Coverage

**Files:**
- Create: `tests/simulator/test_sdk_conformance.py`
- Modify: `tests/simulator/test_invariants.py`
- Modify: `tests/simulator/helpers.py`
- Modify if a demonstrated mismatch requires it:
  `src/garboid_pocketrocks/simulator/context.py`
- Modify if a demonstrated mismatch requires it:
  `src/garboid_pocketrocks/simulator/engine.py`

**Interfaces:**
- Consumes: complete engine from Task 4 and SDK `scenario(...)`
- Produces: verified compatibility boundary required by runner and RL branches

- [ ] **Step 1: Write a failing narrated-history conformance test**

Drive one seeded three-player engine through an Auction 2, Loan 10, and Auction
1. In parallel, append equivalent `.turn(...)`, `.auction(...)`, and
`.reveal(...)` calls to an SDK scenario using emitted engine events.

At every pending decision batch compare:

```python
CONFORMANT_FIELDS = (
    "player_count",
    "starting_cash",
    "value_chart",
    "objective_ids",
    "current_action_id",
    "current_resource_ids",
    "cash_by_seat",
    "tiebreak_seat",
    "won_resource_counts_by_seat",
    "revealed_info_counts_by_seat",
    "owned_objective_ids_by_seat",
    "bot_seat",
    "current_hand_suit_ids",
    "legal_max_amount",
    "revealable_count",
)

for field in CONFORMANT_FIELDS:
    assert getattr(actual, field) == getattr(expected, field), field
```

Do not compare request IDs or timing.

- [ ] **Step 2: Run conformance and capture the first real mismatch**

Run:

```bash
uv run pytest tests/simulator/test_sdk_conformance.py -x -vv
```

Expected before adjustment: at least one narrated field mismatch or missing
fixture capability demonstrates the test is exercising independent paths.
If the first implementation already matches, temporarily invert one asserted
field to prove the test fails, restore it, and continue.

- [ ] **Step 3: Correct only demonstrated fidelity mismatches**

For each mismatch:

1. Confirm the live rule or SDK reconstruction behavior.
2. Add a focused regression assertion naming the field.
3. Change only the engine/context derivation responsible.
4. Re-run with `-x` before proceeding to the next mismatch.

Do not alter tests to accept competition-engine behavior.

- [ ] **Step 4: Add Hypothesis invariants**

Use `@given(st.integers(min_value=0, max_value=2**32 - 1))` across 3-, 4-, and
5-player random-brain games. For every transition assert:

- exact per-suit card conservation;
- unique card IDs;
- at most one owner per objective;
- all cash values are nonnegative after action resolution;
- pending contexts accept every action enabled by their calculated legal range;
- terminal states have no pending batch or remaining biddable resources;
- independent score recomputation equals `GameResult`.

Set `@settings(max_examples=40, deadline=None)` to keep CI deterministic.

- [ ] **Step 5: Run the simulator verification slice**

Run:

```bash
uv run pytest tests/simulator -q
uv run ruff check src/garboid_pocketrocks/simulator tests/simulator
uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: all example and generated tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "test: verify simulator invariants and SDK conformance"
```

---

### Task 6: Replay and Synchronous Match Runner

**Files:**
- Create: `src/garboid_pocketrocks/simulator/replay.py`
- Create: `src/garboid_pocketrocks/simulator/runner.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Create: `tests/simulator/test_replay.py`
- Create: `tests/simulator/test_runner.py`

**Interfaces:**
- Consumes: `BotSpec`, `BotBrain`, `Ruleset`, `GameEngine`
- Produces: `FaultMode`, `BotFault`, `MatchReplay`, `MatchResult`,
  `MatchRunner.run(...)`, and `replay_match(...)`

- [ ] **Step 1: Write failing match and replay tests**

Tests must prove:

```python
def test_match_runner_is_reproducible_and_uses_fresh_brains() -> None:
    lineup = tuple(BotSpec.from_bot_class(RandomBot) for _ in range(3))
    left = MatchRunner.run(
        lineup,
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=91,
    )
    right = MatchRunner.run(
        lineup,
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=91,
    )
    assert left.result == right.result
    assert left.replay == right.replay


def test_replay_reproduces_events_and_result() -> None:
    match = run_random_match(seed=17)
    replayed = replay_match(match.replay)
    assert replayed.events == match.events
    assert replayed.result == match.result


def test_replay_json_round_trip_is_lossless(tmp_path: Path) -> None:
    original = run_random_match(seed=18).replay
    path = tmp_path / "match.json"
    save_replay(original, path)
    assert load_replay(path) == original


def test_record_and_pass_records_brain_failure() -> None:
    match = MatchRunner.run(
        lineup_with_raising_brain(),
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=5,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )
    assert len(match.faults) == 1
    assert match.faults[0].seat == 0
```

Also assert `FaultMode.RAISE` propagates the original exception.

- [ ] **Step 2: Run the runner tests and verify they fail**

Run:

```bash
uv run pytest tests/simulator/test_runner.py tests/simulator/test_replay.py -q
```

Expected: collection fails because runner/replay modules do not exist.

- [ ] **Step 3: Implement replay schema**

`MatchReplay` is a frozen dataclass containing:

```python
schema_version: int
ruleset: Ruleset
player_count: int
seed: int
root_seed: int | None
game_index: int | None
bot_names: tuple[str, ...]
decisions: tuple[tuple[int, tuple[tuple[int, BotDecision], ...]], ...]
events: tuple[GameEvent, ...]
```

The outer decision tuple stores a monotonic decision-step index followed by
sorted seat/decision pairs. `MatchReplay.to_dict()` and
`MatchReplay.from_dict()` define a versioned, stable JSON representation;
`save_replay(replay, path)` and `load_replay(path)` write and read it.
`replay_match` starts the engine from recorded configuration, verifies each
pending seat set, reapplies decisions, and raises `ReplayDivergence` if the
regenerated events differ at any step.

- [ ] **Step 4: Implement `MatchRunner`**

Define:

```python
class FaultMode(StrEnum):
    RAISE = "raise"
    RECORD_AND_PASS = "record_and_pass"


@dataclass(frozen=True, slots=True)
class BotFault:
    turn_index: int
    seat: int
    bot_name: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class MatchResult:
    result: GameResult
    events: tuple[GameEvent, ...]
    faults: tuple[BotFault, ...]
    replay: MatchReplay


```

`MatchRunner.run(lineup: Sequence[BotSpec], *, ruleset: Ruleset,
player_count: int, seed: int, fault_mode: FaultMode = FaultMode.RAISE) ->
MatchResult` is the synchronous match entry point.

Derive each brain seed with
`random.Random(seed).randrange(2**63)` in seat order. On every batch, call each
brain synchronously with its context and
`ruleset.knowledge(player_count)`. In forgiving mode, record construction,
decision, and engine legality failures, append a `BOT_FAULT` domain event, and
substitute `BotDecision.pass_turn()`. Direct matches leave replay `root_seed`
and `game_index` as `None`; Monte Carlo fills both provenance fields. Never
catch `KeyboardInterrupt`, `SystemExit`, or `BaseException`.

- [ ] **Step 5: Run focused and full simulator tests**

Run:

```bash
uv run pytest tests/simulator/test_runner.py tests/simulator/test_replay.py -q
uv run pytest tests/simulator -q
uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: replay and runner tests pass without regressing engine tests.

- [ ] **Step 6: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "feat: add replayable synchronous match runner"
```

---

### Task 7: Ruleset Sampling and Parallel Monte Carlo Evaluation

**Files:**
- Create: `src/garboid_pocketrocks/simulator/sampling.py`
- Create: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Modify: `src/garboid_pocketrocks/simulator/__init__.py`
- Create: `tests/simulator/test_sampling.py`
- Create: `tests/simulator/test_monte_carlo.py`

**Interfaces:**
- Consumes: `Ruleset`, `BotSpec`, and `MatchRunner`
- Produces: `RulesetSampler`, `FixedRulesetSampler`,
  `WeightedRulesetSampler`, `RulesetVariationSampler`,
  `MonteCarloConfig`, `BotStatistics`, `MonteCarloResult`, and
  `MonteCarloRunner.run(...)`

- [ ] **Step 1: Write failing sampler and determinism tests**

Cover:

```python
def test_weighted_sampler_is_seeded_by_game_index() -> None:
    sampler = WeightedRulesetSampler(
        ((live_ruleset("A"), 1), (live_ruleset("E"), 2))
    )
    assert [sampler.sample(root_seed=7, game_index=i) for i in range(20)] == [
        sampler.sample(root_seed=7, game_index=i) for i in range(20)
    ]


def test_variation_sampler_can_change_each_public_axis() -> None:
    sampler = RulesetVariationSampler(
        base=LIVE_RULESET,
        resource_count_options=((6, 6, 6, 6, 6), (5, 6, 7, 6, 6)),
        action_count_options=(
            LIVE_RULESET.action_counts,
            (14, 7, 3, 2, 2, 2),
        ),
        setup_options=(
            LIVE_RULESET.player_setups,
            (
                PlayerSetup(3, 35, 4),
                PlayerSetup(4, 28, 3),
                PlayerSetup(5, 22, 2),
            ),
        ),
        value_chart_options=(VALUE_CHARTS["A"], VALUE_CHARTS["E"]),
        objective_options=(
            (LIVE_RULESET.objective_pool, 4, True),
            (LIVE_RULESET.objective_pool[:12], 3, True),
        ),
    )
    samples = {sampler.sample(root_seed=13, game_index=i) for i in range(100)}
    assert len({sample.resource_counts for sample in samples}) == 2
    assert len({sample.action_counts for sample in samples}) == 2
    assert len({sample.player_setups for sample in samples}) == 2
    assert len({sample.value_chart for sample in samples}) == 2
    assert len(
        {
            (
                sample.objective_pool,
                sample.active_objective_count,
                sample.objectives_enabled,
            )
            for sample in samples
        }
    ) == 2


def test_worker_count_does_not_change_monte_carlo_result() -> None:
    config = small_random_config(games=30, seed=101)
    assert MonteCarloRunner.run(config, workers=1) == MonteCarloRunner.run(
        config,
        workers=2,
    )
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```bash
uv run pytest tests/simulator/test_sampling.py tests/simulator/test_monte_carlo.py -q
```

Expected: collection fails because sampling and Monte Carlo modules do not
exist.

- [ ] **Step 3: Implement fixed, weighted, and variation samplers**

Define a runtime-checkable protocol:

```python
class RulesetSampler(Protocol):
    def support(self) -> tuple[Ruleset, ...]:
        """Return every ruleset this sampler can produce."""

    def sample(self, *, root_seed: int, game_index: int) -> Ruleset:
        """Resolve one valid ruleset without global random state."""
```

`FixedRulesetSampler` always returns its ruleset and exposes a one-item support.
`WeightedRulesetSampler` validates positive integer weights and uses
`random.Random(derive_seed(root_seed, "ruleset", game_index)).choices(...)`;
its support contains each distinct configured ruleset once.
`RulesetVariationSampler` accepts finite nonempty options for each configurable
field plus correlated `(objective_pool, active_objective_count,
objectives_enabled)` options, validates every Cartesian product during
construction, and samples one option per field with a derived local RNG. Its
support is that validated Cartesian product. Give every generated ruleset a
stable name formed from the base name plus the first 12 hexadecimal characters
of a SHA-256 digest over its canonically serialized public fields.

Use a stable hash-independent seed derivation:

```python
def derive_seed(root_seed: int, namespace: str, index: int) -> int:
    payload = f"{root_seed}:{namespace}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

- [ ] **Step 4: Implement deterministic Monte Carlo plans and aggregation**

`MonteCarloConfig` contains:

```python
bot_specs: tuple[BotSpec, ...]
games: int
player_counts: tuple[int, ...]
ruleset_sampler: RulesetSampler
root_seed: int
fault_mode: FaultMode = FaultMode.RAISE
capture_replays: bool = False
```

Before workers start, construct immutable game jobs in index order. Choose
player count from its configured tuple with a derived RNG. If bots outnumber
seats, sample without replacement. Shuffle the selected lineup once, then
rotate by `game_index % player_count` to balance seats. Use
`derive_seed(root_seed, "game", game_index)` as the match seed.

Aggregate in game-index order into frozen per-bot statistics containing games,
outright wins, first-place ties, rank counts, final-money samples, score
margins, per-seat buckets, per-ruleset buckets, decision counts, and faults.
Expose calculated mean, median, population spread, and quantiles as methods,
not stored duplicate state. `MonteCarloResult` also contains ordered per-game
summaries and a replay tuple. The replay tuple is populated in game-index order
when `capture_replays` is true and is empty otherwise.

Use `ProcessPoolExecutor` only when `workers > 1`. Catch pickling failures and
raise a clear `SimulationError` naming the unpicklable bot spec. Do not silently
fall back to serial execution.

- [ ] **Step 5: Test fairness, reconciliation, and workers**

Add assertions that:

- every requested game is represented once;
- aggregate games equal per-seat and per-ruleset totals;
- three identical bots occupy each seat equally over a game count divisible by
  three;
- same seed produces byte-for-byte equal dataclasses;
- different root seeds produce different game jobs;
- an unpicklable closure works with one worker and fails clearly with two.

- [ ] **Step 6: Run Monte Carlo and static checks**

Run:

```bash
uv run pytest tests/simulator/test_sampling.py tests/simulator/test_monte_carlo.py -q
uv run ruff check src/garboid_pocketrocks/simulator tests/simulator
uv run mypy src/garboid_pocketrocks/simulator tests/simulator
```

Expected: sampling, fairness, aggregation, and worker determinism pass.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "feat: add deterministic Monte Carlo evaluation"
```

---

### Task 8: RL Bounds, Action Codec, Observations, and Rewards

**Files:**
- Create: `src/garboid_pocketrocks/training/bounds.py`
- Create: `src/garboid_pocketrocks/training/actions.py`
- Create: `src/garboid_pocketrocks/training/observations.py`
- Create: `src/garboid_pocketrocks/training/rewards.py`
- Modify: `src/garboid_pocketrocks/training/__init__.py`
- Create: `tests/training/test_actions.py`
- Create: `tests/training/test_observations.py`
- Create: `tests/training/test_rewards.py`

**Interfaces:**
- Consumes: SDK contexts, `RulesetKnowledge`, engine transitions/results
- Produces: `EnvironmentBounds`, `ActionCodec`, `ObservationEncoder`,
  `RewardConfig`, `RewardBreakdown`, and `RewardTracker`

- [ ] **Step 1: Write failing action-codec tests**

```python
def test_action_codec_round_trips_legal_decisions() -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))
    assert codec.decode(0) == BotDecision.pass_turn()
    assert codec.decode(17) == BotDecision.submit_bid(17)
    assert codec.decode(101) == BotDecision.select_info_to_reveal(0)
    assert codec.encode(BotDecision.select_info_to_reveal(4)) == 105


def test_action_masks_match_sdk_context_legality() -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))
    bid_mask = codec.mask(_bid_context(7))
    assert tuple(index for index, enabled in enumerate(bid_mask) if enabled) == tuple(
        range(8)
    )
    reveal_mask = codec.mask(_reveal_context(3))
    assert tuple(
        index for index, enabled in enumerate(reveal_mask) if enabled
    ) == (0, 101, 102, 103)
```

- [ ] **Step 2: Write failing observation and reward tests**

Assert:

- observation space contains exact keys `phase`, `player_count`, `bot_seat`,
  `starting_cash`, `value_chart`, `active_objectives`, `current_action`,
  `current_resources`, `cash_by_seat`, `priority_seat`, `won_resources`,
  `revealed_info`, `owned_objectives`, `private_hand`, `rules_resource_counts`,
  `rules_action_counts`, `rules_private_cards`, `rules_objective_pool`,
  `rules_active_objective_count`, `rules_objectives_enabled`, and
  `action_mask`;
- `observation_space.contains(encoded)` for bid and reveal contexts;
- changing a hidden opponent hand or deck order does not change encoded output;
- changing public ruleset action counts does change encoded output;
- configuring `hidden_ruleset_fields={"rules_action_counts"}` zeroes only that
  field, so domain randomization can train without selected public knowledge;
- an Auction 1 bid of $4 yields accounting reward `-4 / starting_cash`;
- a Loan 10 bid of $4 also yields `-4 / starting_cash`;
- an Invest 5 bid yields `5 / starting_cash`;
- terminal resource value supplies the residual normalized final-money delta;
- tied winners divide the configured win bonus.

- [ ] **Step 3: Run training unit tests and verify they fail**

Run:

```bash
uv run pytest tests/training/test_actions.py tests/training/test_observations.py tests/training/test_rewards.py -q
```

Expected: collection fails because training modules do not exist.

- [ ] **Step 4: Implement bounds and action codec**

```python
@dataclass(frozen=True, slots=True)
class EnvironmentBounds:
    max_bid: int
    max_hand_size: int

    def __post_init__(self) -> None:
        if self.max_bid < 0 or self.max_hand_size < 0:
            raise ValueError("environment bounds must be nonnegative")


```

`ActionCodec(bounds)` owns a `gymnasium.spaces.Discrete` of size
`1 + max_bid + max_hand_size` and implements
`encode(decision: BotDecision) -> int`,
`decode(action: int) -> BotDecision`, and
`mask(context: DecisionContext) -> np.ndarray`.

Use `np.int8` masks. Reject out-of-bound bids/reveal indices with `ValueError`.
The mask always enables pass. Bid contexts enable `1..legal_max_amount`;
reveal contexts enable offsets for `0..revealable_count - 1`.

- [ ] **Step 5: Implement a bounded Gymnasium `Dict` observation**

Use fixed numeric dtypes:

- `np.int8` for phase, suits, actions, masks, and objective bitsets;
- `np.int16` for value chart, rules counts, cash, and count matrices.

Pad seats to 5, objectives to 30, private hand to `max_hand_size`, and use zero
as padding because SDK IDs start at one. The encoded observation must use only
`DecisionContext` and `RulesetKnowledge`; do not accept `GameState` as an
encoder argument. `ObservationEncoder(bounds, *,
hidden_ruleset_fields: frozenset[str] = frozenset())` validates mask names and
zero-fills only the selected public-ruleset observation fields.

Validate before encoding:

- `legal_max_amount <= max_bid`;
- `revealable_count <= max_hand_size`;
- configured private cards and maximum possible loan cash fit declared bounds.

- [ ] **Step 6: Implement auditable reward potential**

```python
@dataclass(frozen=True, slots=True)
class RewardConfig:
    accounting_weight: float = 1.0
    win_bonus: float = 1.0
    placement_bonuses: tuple[float, ...] = ()
    invalid_action_penalty: float = 0.0
    event_bonuses: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    accounting: float = 0.0
    terminal_resource: float = 0.0
    placement: float = 0.0
    shaping: float = 0.0
    penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.accounting
            + self.terminal_resource
            + self.placement
            + self.shaping
            + self.penalty
        )
```

`RewardTracker` stores the previous public potential by seat. Derive potential
from player cash, investment locks/payouts, loan principals, and owned
objective payouts. At terminal, add final resource value. Divide deltas by that
seat's starting cash. Add placement bonuses only once. Split `win_bonus`
equally among rank-one seats. Apply optional `event_bonuses` by matching exact
`GameEvent.kind` values, keep them in the `shaping` field, and reject duplicate
or unknown event kinds during `RewardConfig` validation.

- [ ] **Step 7: Run training unit and static checks**

Run:

```bash
uv run pytest tests/training/test_actions.py tests/training/test_observations.py tests/training/test_rewards.py -q
uv run ruff check src/garboid_pocketrocks/training tests/training
uv run mypy src/garboid_pocketrocks/training tests/training
```

Expected: action, observation, and reward tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/garboid_pocketrocks/training tests/training
git commit -m "feat: add RL observations actions and rewards"
```

---

### Task 9: PettingZoo Multi-Agent and Gymnasium Single-Agent Environments

**Files:**
- Create: `src/garboid_pocketrocks/training/multi_agent_env.py`
- Create: `src/garboid_pocketrocks/training/single_agent_env.py`
- Modify: `src/garboid_pocketrocks/training/__init__.py`
- Create: `tests/training/test_multi_agent_env.py`
- Create: `tests/training/test_single_agent_env.py`

**Interfaces:**
- Consumes: `GameEngine`, `ActionCodec`, `ObservationEncoder`,
  `RewardTracker`, `BotSpec`, and ruleset samplers
- Produces: `PocketRocksAECEnv`, `PocketRocksEnv`, and
  `InvalidActionMode`

- [ ] **Step 1: Write failing PettingZoo lifecycle tests**

Test a three-player environment with deterministic forced setup:

- `reset(seed=3)` selects all three seats in bidding order;
- the engine state does not change after the first or second sealed bid;
- the third bid resolves the batch;
- only the winner is selected for reveal;
- after reveal, bidding resumes at seat zero;
- observations do not include stored sealed bids;
- every agent receives a terminal score and termination flag.

Run PettingZoo's:

```python
from pettingzoo.test import api_test


def test_aec_contract() -> None:
    api_test(make_small_aec_env(), num_cycles=200, verbose_progress=False)
```

- [ ] **Step 2: Write failing Gymnasium wrapper tests**

Assert that:

- learner seat is reproducible with reset seed;
- fixed learner seat overrides randomization;
- one learner bid causes all opponents to bid and resolves the joint batch;
- opponent-only reveal is auto-played before returning;
- learner reveal is returned as its own step;
- rewards accumulated through internal transitions equal reward breakdown totals;
- invalid masked action raises by default;
- penalty-and-pass mode records the penalty and advances;
- `gymnasium.utils.env_checker.check_env(env)` passes.

- [ ] **Step 3: Run environment tests and verify they fail**

Run:

```bash
uv run pytest tests/training/test_multi_agent_env.py tests/training/test_single_agent_env.py -q
```

Expected: collection fails because environment classes do not exist.

- [ ] **Step 4: Implement the PettingZoo AEC adapter**

Subclass `pettingzoo.AECEnv`. Keep all seats in `possible_agents` for the
episode. During bidding:

1. expose each seat's pre-bid context in ascending seat order;
2. store its decoded action;
3. call `_was_dead_step` only for actually terminated agents;
4. resolve through `GameEngine.step` after the final seat;
5. publish rewards and infos from `RewardTracker`.

During reveal, set `agent_selection` to the winner only. After resolution,
restart bidding order or terminate. Use PettingZoo's `agent_selector` utility
and cumulative-reward conventions. `observe(agent)` must return that seat's
encoded pending context; never expose another seat's private hand. During
construction, validate every ruleset returned by `ruleset_sampler.support()`
against `EnvironmentBounds` and reject incompatible supports before reset.

- [ ] **Step 5: Implement the Gymnasium single-agent wrapper**

```python
class InvalidActionMode(StrEnum):
    RAISE = "raise"
    PENALIZE_AND_PASS = "penalize_and_pass"


```

`PocketRocksEnv` subclasses
`gymnasium.Env[dict[str, np.ndarray], int]`. Its keyword-only constructor takes
`opponent_specs: Sequence[BotSpec]`, `ruleset_sampler: RulesetSampler`,
`player_count: int`, `bounds: EnvironmentBounds`,
`reward_config: RewardConfig = RewardConfig()`,
`learner_seat: int | None = None`, and
`invalid_action_mode: InvalidActionMode = InvalidActionMode.RAISE`.
Construction validates every ruleset returned by `ruleset_sampler.support()`
against the bounds, player count, and observation space before the first reset.

`reset(seed=...)` resolves a ruleset, selects/fixes learner seat, creates fresh
opponent brains, starts the engine, and returns at the learner's first decision.
`step(action)` applies the learner decision, then repeatedly invokes opponent
brains and engine transitions until the learner acts again or the game ends.
Return the accumulated `RewardBreakdown` dictionary in `info`.

No time-limit truncation is used; `truncated` is always false.

- [ ] **Step 6: Run API contract and full training tests**

Run:

```bash
uv run pytest tests/training -q
uv run ruff check src/garboid_pocketrocks/training tests/training
uv run mypy src/garboid_pocketrocks/training tests/training
```

Expected: PettingZoo and Gymnasium checkers pass along with all focused tests.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/training tests/training
git commit -m "feat: add multi-agent and single-agent RL environments"
```

---

### Task 10: CLI, Documentation, Benchmarks, and Full Integration

**Files:**
- Create: `src/garboid_pocketrocks/simulator/cli.py`
- Create: `tests/simulator/test_cli.py`
- Create: `tests/test_integration.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/superpowers/specs/2026-07-28-random-bot-sdk-design.md`
- Modify: `docs/superpowers/plans/2026-07-28-random-bot-sdk.md`

**Interfaces:**
- Consumes: every earlier task
- Produces: `garboid-simulate` command, documented public workflow, benchmark
  result, verified release-quality milestone

- [ ] **Step 1: Write failing CLI and end-to-end tests**

Use `subprocess.run` against the installed command and assert:

```python
def test_simulate_cli_emits_reproducible_json() -> None:
    args = [
        "uv",
        "run",
        "garboid-simulate",
        "--bots",
        "random,random,random",
        "--games",
        "6",
        "--players",
        "3",
        "--seed",
        "42",
        "--format",
        "json",
    ]
    left = subprocess.run(args, check=True, text=True, capture_output=True)
    right = subprocess.run(args, check=True, text=True, capture_output=True)
    assert json.loads(left.stdout) == json.loads(right.stdout)
```

An integration test must run one match for every chart A–E and player count
3–5, then run both RL environments to termination using masked random actions.

- [ ] **Step 2: Run tests and verify the red state**

Run:

```bash
uv run pytest tests/simulator/test_cli.py tests/test_integration.py -q
```

Expected: the command is not registered and tests fail.

- [ ] **Step 3: Implement and register `garboid-simulate`**

Add:

```toml
[project.scripts]
garboid-random-bot = "garboid_pocketrocks.bots.random_bot:main"
garboid-simulate = "garboid_pocketrocks.simulator.cli:main"
```

The CLI supports:

- `--bots` comma-separated registered names;
- `--games` positive integer;
- `--players` one of 3, 4, 5;
- `--seed` integer;
- `--ruleset` one of `live-A` through `live-E`;
- `--workers` positive integer;
- `--format` `table` or `json`;
- `--replay-dir` optional replay output directory.

Register `random` through `BotSpec.from_bot_class(RandomBot)`. Reject unknown
bot names and invalid counts with `argparse` errors. JSON output uses stable key
ordering and contains configuration plus all aggregate metrics. Supplying
`--replay-dir` sets `capture_replays=True`, creates that directory if needed,
and writes each captured replay with `save_replay` using a deterministic
zero-padded game-index filename.

- [ ] **Step 4: Update documentation and stale random-bot records**

README examples must include:

```bash
uv run garboid-simulate \
  --bots random,random,random \
  --games 1000 \
  --players 3 \
  --ruleset live-A \
  --seed 42
```

Document:

- `RandomBot.BOT_ID` as the committed public identity;
- `.env` now needs only the API key and SDK settings;
- synchronous brain versus live wrapper;
- fixed and sampled rulesets;
- replay and deterministic seeds;
- Monte Carlo metrics and workers;
- action encoding/masks;
- accounting-potential rewards;
- PettingZoo and Gymnasium entry points;
- no neural policy or training algorithm yet.

Update the earlier random-bot design and plan so they no longer claim
`RANDOM_BOT_ID` is the final configuration boundary. Preserve the historical
milestone explanation and add a dated note that the simulator phase moved the
ID into `RandomBot.BOT_ID`.

- [ ] **Step 5: Run local throughput benchmarks**

Run:

```bash
for players in 3 4 5; do
  for chart in A E; do
    time uv run garboid-simulate \
      --bots random,random,random,random,random \
      --games 10000 \
      --players "$players" \
      --ruleset "live-$chart" \
      --seed 2026 \
      --format table
  done
done
```

Record chart, player count, games, elapsed time, and calculated games/second
for all six runs in the implementation commit message body or final handoff.
Do not add a CI timing threshold.

- [ ] **Step 6: Run the complete verification gate**

Run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run garboid-random-bot --help
uv run garboid-simulate --help
git diff --check
```

Expected: every command exits zero. The test count must include rule,
conformance, replay, Monte Carlo, Gymnasium, PettingZoo, CLI, and integration
coverage.

- [ ] **Step 7: Commit**

```bash
git add \
  .env.example \
  README.md \
  pyproject.toml \
  uv.lock \
  src \
  tests \
  docs/superpowers/specs/2026-07-28-random-bot-sdk-design.md \
  docs/superpowers/plans/2026-07-28-random-bot-sdk.md
git commit -m "feat: complete simulator Monte Carlo and RL milestone"
```

- [ ] **Step 8: Push and verify CI**

Run:

```bash
git push origin main
gh run list \
  --repo chrisgarber/garboid-pocketrocks \
  --limit 5 \
  --json databaseId,status,conclusion,headSha,url
gh run watch RUN_ID \
  --repo chrisgarber/garboid-pocketrocks \
  --exit-status
```

Expected: the workflow for the final implementation commit completes
successfully. If it fails, reproduce the root cause locally, add or preserve a
regression test, make one focused fix commit, push, and watch the replacement
run.
