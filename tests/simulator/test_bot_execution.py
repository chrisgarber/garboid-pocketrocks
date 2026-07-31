from __future__ import annotations

import inspect

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.diagnostics.trace import (
    ExplainedBotDecision,
    NeuralPolicyExplanation,
    OpponentAwareHeuristicBidExplanation,
    legal_actions_for_context,
)
from garboid_pocketrocks.heuristics.opponent_bids import (
    OPPONENT_BID_MODEL_NAME,
    CompetitiveBidPoint,
    OpponentBidDistribution,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge, canonical_knowledge
from garboid_pocketrocks.simulator.bot_execution import (
    BotFault,
    DecisionExecution,
    FaultMode,
    execute_brain_decision,
)
from garboid_pocketrocks.simulator.session import SdkGameSession


def _inputs() -> tuple[DecisionContext, RulesetKnowledge, PublicHistory]:
    session = SdkGameSession.start(player_count=3, seed=19)
    context = session.pending.contexts[0][1]
    return context, canonical_knowledge(3), ()


class _OrdinaryBrain:
    def __init__(self) -> None:
        self.calls = 0

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        del context, ruleset, history
        self.calls += 1
        return BotDecision.submit_bid(2)


class _ExplainingBrain(_OrdinaryBrain):
    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        del ruleset, history
        self.calls += 1
        legal_actions = legal_actions_for_context(context)
        selected_index = legal_actions.index(
            legal_actions_for_context(context)[2],
        )
        probabilities = tuple(
            0.75 if index == selected_index else 0.25 / (len(legal_actions) - 1)
            for index in range(len(legal_actions))
        )
        return ExplainedBotDecision(
            decision=BotDecision.submit_bid(2),
            explanation=NeuralPolicyExplanation(
                predicted_value=0.25,
                selected_probability=0.75,
                entropy=0.5,
                legal_action_probabilities=probabilities,
            ),
        )


class _RaisingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        del context, ruleset, history
        raise RuntimeError("broken policy")


class _OpponentExplainingBrain(_OrdinaryBrain):
    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        del ruleset, history
        self.calls += 1
        legal_max = context.legal_max_amount
        assert legal_max is not None
        return ExplainedBotDecision(
            decision=BotDecision.submit_bid(2),
            explanation=OpponentAwareHeuristicBidExplanation(
                resource_value=4.0,
                objective_completion_value=0.0,
                objective_progress_value=0.0,
                terminal_cash_value=0.0,
                liquidity_value=0.0,
                future_cash_value=0.0,
                total_value=4.0,
                reservation_bid=2,
                chosen_bid=2,
                public_game_phase="early",
                model_name=OPPONENT_BID_MODEL_NAME,
                model_config_digest="a" * 64,
                opponent_distributions=tuple(
                    OpponentBidDistribution(
                        opponent_seat=seat,
                        legal_max_amount=0,
                        probabilities_by_amount=(1.0,),
                        prior_only=True,
                        history_round_count=0,
                        effective_history_weight=0.0,
                    )
                    for seat in range(context.player_count)
                    if seat != context.bot_seat
                ),
                competitive_bid_points=tuple(
                    CompetitiveBidPoint(
                        effective_bid=bid,
                        win_probability=1.0,
                        win_delta=2.0 if bid == 2 else 0.0,
                        expected_surplus=2.0 if bid == 2 else 0.0,
                    )
                    for bid in range(legal_max + 1)
                ),
            ),
        )


def _execute(
    brain: object,
    *,
    fault_mode: FaultMode = FaultMode.RAISE,
    request_explanation: bool = False,
) -> tuple[DecisionExecution, list[BotFault]]:
    context, knowledge, history = _inputs()
    faults: list[BotFault] = []
    execution = execute_brain_decision(
        brain=brain,  # type: ignore[arg-type]
        context=context,
        knowledge=knowledge,
        history=history,
        fault_mode=fault_mode,
        faults=faults,
        turn_index=0,
        seat=0,
        bot_name="test",
        request_explanation=request_explanation,
    )
    return execution, faults


def test_ordinary_brain_keeps_its_decision_without_an_explanation() -> None:
    brain = _OrdinaryBrain()

    execution, faults = _execute(brain)

    assert execution.decision == BotDecision.submit_bid(2)
    assert execution.explanation is None
    assert execution.selection_source == "policy"
    assert brain.calls == 1
    assert faults == []


def test_explanation_aware_brain_uses_the_ordinary_protocol_by_default() -> None:
    brain = _ExplainingBrain()

    execution, faults = _execute(brain)

    assert execution.decision == BotDecision.submit_bid(2)
    assert execution.explanation is None
    assert execution.selection_source == "policy"
    assert brain.calls == 1
    assert faults == []


def test_explanation_capture_must_be_explicitly_requested() -> None:
    assert "request_explanation" in inspect.signature(execute_brain_decision).parameters
    brain = _ExplainingBrain()

    execution, faults = _execute(brain, request_explanation=True)

    assert execution.decision == BotDecision.submit_bid(2)
    assert isinstance(execution.explanation, NeuralPolicyExplanation)
    assert execution.explanation.selected_probability == 0.75
    assert brain.calls == 1
    assert faults == []


def test_opponent_aware_explanation_reuses_heuristic_action_agreement() -> None:
    execution, faults = _execute(_OpponentExplainingBrain(), request_explanation=True)

    assert execution.decision == BotDecision.submit_bid(2)
    assert isinstance(execution.explanation, OpponentAwareHeuristicBidExplanation)
    assert execution.explanation.chosen_bid == 2
    assert faults == []


def test_fault_mode_preserves_existing_fallback_and_marks_its_source() -> None:
    execution, faults = _execute(_RaisingBrain(), fault_mode=FaultMode.RECORD_AND_PASS)

    assert execution.decision == BotDecision.submit_bid(0)
    assert execution.explanation is None
    assert execution.selection_source == "fault_fallback"
    assert len(faults) == 1
    assert faults[0].error_type == "RuntimeError"
