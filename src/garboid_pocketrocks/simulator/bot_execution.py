from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotBrain, BotSpec
from garboid_pocketrocks.diagnostics.trace import (
    BotResultMetric,
    DecisionExplanation,
    ExplainedBotDecision,
    FixedObjectiveOverlayV3BidExplanation,
    HeuristicBidExplanation,
    NeuralPolicyExplanation,
    RecordedAction,
    SelectionSource,
    legal_actions_for_context,
)
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


@dataclass(frozen=True, slots=True)
class DecisionExecution:
    """A validated decision and diagnostics produced by the same policy call."""

    decision: BotDecision
    explanation: DecisionExplanation | None
    result_metrics: tuple[BotResultMetric, ...]
    selection_source: SelectionSource


@runtime_checkable
class ExplanationAwareBotBrain(Protocol):
    """Optional interface returning a decision with explanation and result metrics."""

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Return one decision and its optional live-compatible diagnostics."""


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
    """Compatibility wrapper returning only the validated SDK decision."""

    return execute_brain_decision(
        brain=brain,
        context=context,
        knowledge=knowledge,
        history=history,
        fault_mode=fault_mode,
        faults=faults,
        turn_index=turn_index,
        seat=seat,
        bot_name=bot_name,
    ).decision


def execute_brain_decision(
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
    request_explanation: bool = False,
) -> DecisionExecution:
    """Invoke one brain once and retain diagnostics from that same call."""

    if brain is None:
        return DecisionExecution(
            decision=_fallback_decision(context),
            explanation=None,
            result_metrics=(),
            selection_source="fault_fallback",
        )
    try:
        if request_explanation and isinstance(brain, ExplanationAwareBotBrain):
            explained = brain.choose_explained_decision(
                context,
                knowledge,
                history,
            )
            decision = explained.decision
            explanation = explained.explanation
            result_metrics = explained.result_metrics
            _validate_explanation_agrees_with_decision(
                context,
                decision,
                explanation,
            )
        else:
            decision = brain.choose_decision(context, knowledge, history)
            explanation = None
            result_metrics = ()
        context.validate(decision)
        return DecisionExecution(
            decision=decision,
            explanation=explanation,
            result_metrics=result_metrics,
            selection_source="policy",
        )
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
        return DecisionExecution(
            decision=_fallback_decision(context),
            explanation=None,
            result_metrics=(),
            selection_source="fault_fallback",
        )


def _validate_explanation_agrees_with_decision(
    context: DecisionContext,
    decision: BotDecision,
    explanation: DecisionExplanation | None,
) -> None:
    recorded = RecordedAction.from_decision(decision)
    if isinstance(explanation, NeuralPolicyExplanation):
        legal_actions = legal_actions_for_context(context)
        if len(explanation.legal_action_probabilities) != len(legal_actions):
            raise ValueError("neural explanation requires one probability per legal action")
        selected_index = legal_actions.index(recorded)
        if not math.isclose(
            explanation.selected_probability,
            explanation.legal_action_probabilities[selected_index],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("neural selected probability must match the selected legal action")
        return
    if not isinstance(
        explanation,
        (HeuristicBidExplanation, FixedObjectiveOverlayV3BidExplanation),
    ):
        return
    selected_bid = 0 if recorded.action_kind == "pass" else recorded.value
    if recorded.action_kind not in ("pass", "submitBid") or selected_bid != explanation.chosen_bid:
        raise ValueError("heuristic explanation chosen bid disagrees with its decision")


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
