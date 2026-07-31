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
