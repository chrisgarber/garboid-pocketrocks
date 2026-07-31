from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotBrain, BotSpec, HistoryAwareBotBrain
from garboid_pocketrocks.knowledge import RulesetKnowledge


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


def initialize_brains(
    lineup: Sequence[BotSpec],
    *,
    seed: int,
    fault_mode: FaultMode,
) -> tuple[tuple[BotBrain | None, ...], tuple[BotFault, ...]]:
    """Construct one independently seeded brain per seat."""

    brain_rng = random.Random(seed)
    brains: list[BotBrain | None] = []
    faults: list[BotFault] = []
    for seat, spec in enumerate(lineup):
        try:
            brains.append(spec.make_brain(seed=brain_rng.randrange(2**63)))
        except Exception as error:
            if fault_mode is FaultMode.RAISE:
                raise
            brains.append(None)
            _append_fault(
                faults,
                turn_index=0,
                seat=seat,
                bot_name=spec.name,
                error=error,
            )
    return tuple(brains), tuple(faults)


def choose_brain_decision(
    *,
    brain: BotBrain | None,
    context: DecisionContext,
    knowledge: RulesetKnowledge,
    history: PublicHistory,
    fault_mode: FaultMode,
    faults: list[BotFault],
    turn_index: int,
    seat: int,
    bot_name: str,
) -> BotDecision:
    """Invoke and validate one brain decision under the configured fault policy."""

    if brain is None:
        return _fallback_decision(context)
    try:
        if isinstance(brain, HistoryAwareBotBrain):
            decision = brain.choose_decision_with_history(
                context,
                knowledge,
                history,
            )
        else:
            decision = brain.choose_decision(context, knowledge)
        context.validate(decision)
        return decision
    except Exception as error:
        if fault_mode is FaultMode.RAISE:
            raise
        _append_fault(
            faults,
            turn_index=turn_index,
            seat=seat,
            bot_name=bot_name,
            error=error,
        )
        return _fallback_decision(context)


def _fallback_decision(context: DecisionContext) -> BotDecision:
    if context.decision_kind == "selectInfoToReveal":
        return BotDecision.select_info_to_reveal(0)
    return BotDecision.submit_bid(0)


def _append_fault(
    faults: list[BotFault],
    *,
    turn_index: int,
    seat: int,
    bot_name: str,
    error: Exception,
) -> None:
    faults.append(
        BotFault(
            turn_index=turn_index,
            seat=seat,
            bot_name=bot_name,
            error_type=type(error).__name__,
            message=str(error),
        )
    )
