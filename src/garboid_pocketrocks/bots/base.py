from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from pocketrocks import BotDecision, DecisionContext, PocketRocksBot

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    public_history_from_sdk_frame,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge, knowledge_for_context


class BotBrain(Protocol):
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        """Return one synchronous SDK decision."""


@runtime_checkable
class HistoryAwareBotBrain(Protocol):
    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        """Return one decision using exact immutable public history."""


BrainFactory = Callable[[int | None], BotBrain]


class PocketRocksFastBot(PocketRocksBot):
    BOT_ID: ClassVar[str]
    BOT_NAME: ClassVar[str]

    def __init__(
        self,
        *,
        seed: int | None = None,
        brain: BotBrain | None = None,
        **sdk_options: Any,
    ) -> None:
        sdk_options.setdefault("bot_id", self.BOT_ID)
        super().__init__(**sdk_options)
        self._brain = brain if brain is not None else self.build_brain(seed)

    @classmethod
    def build_brain(cls, seed: int | None) -> BotBrain:
        raise NotImplementedError

    def _knowledge_for_context(self, context: DecisionContext) -> RulesetKnowledge:
        return knowledge_for_context(context)

    def choose_decision_sync(self, context: DecisionContext) -> BotDecision:
        knowledge = self._knowledge_for_context(context)
        return self._brain.choose_decision(context, knowledge)

    def choose_decision_with_history_sync(
        self,
        context: DecisionContext,
        history: PublicHistory,
    ) -> BotDecision:
        knowledge = self._knowledge_for_context(context)
        if isinstance(self._brain, HistoryAwareBotBrain):
            return self._brain.choose_decision_with_history(context, knowledge, history)
        return self._brain.choose_decision(context, knowledge)

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
        return self.choose_decision_sync(context)

    async def choose_raw_decision(
        self,
        frame: object,
        context: DecisionContext,
    ) -> BotDecision:
        history = public_history_from_sdk_frame(frame)
        return self.choose_decision_with_history_sync(context, history)


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

    @classmethod
    def for_simulation(
        cls,
        name: str,
        brain_factory: BrainFactory,
    ) -> BotSpec:
        """Create a local-only spec whose simulation identity is its name."""
        return cls(name=name, bot_id=name, brain_factory=brain_factory)

    def make_brain(self, *, seed: int | None = None) -> BotBrain:
        return self.brain_factory(seed)
