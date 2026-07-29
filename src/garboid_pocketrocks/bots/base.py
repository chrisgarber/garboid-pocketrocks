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
