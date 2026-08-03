from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from typing import cast

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.diagnostics.trace import (
    BotResultMetric,
    DecisionTrace,
    FixedObjectiveOverlayV3BidExplanation,
    HeuristicBidExplanation,
    NeuralPolicyExplanation,
    PendingDecisionTrace,
    PublicDecisionOutcome,
    RecordedAction,
    decision_trace_from_payload,
    decision_trace_payload,
    legal_actions_for_context,
    public_context_from_sdk,
)


def _context(
    *,
    decision_kind: str = "submitBid",
    hand: tuple[int, ...] = (2, 5),
    metadata: dict[str, object] | None = None,
) -> DecisionContext:
    return DecisionContext(
        request_id="private-request-id",
        deadline_at=123_456,
        received_at=123_000,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(1, 10),
        current_action_id=1,
        current_resource_ids=(3, 0),
        cash_by_seat=(22, 30, 18),
        tiebreak_seat=1,
        won_resource_counts_by_seat=(
            (1, 0, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0),
        ),
        revealed_info_counts_by_seat=(
            (0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        ),
        owned_objective_ids_by_seat=((1,), (), ()),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=7 if decision_kind == "submitBid" else None,
        revealable_count=len(hand),
        metadata={} if metadata is None else metadata,
    )


def _history() -> PublicHistory:
    return (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=30,
            value_chart=(0, 4, 8, 12, 16, 20),
            initial_tiebreak_seat=1,
            objective_ids=(1, 10),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(3, 0),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(7, 3, 0),
        ),
    )


def _pending(
    *,
    selected_action: RecordedAction | None = None,
    explanation: (
        HeuristicBidExplanation
        | FixedObjectiveOverlayV3BidExplanation
        | NeuralPolicyExplanation
        | None
    ) = None,
    result_metrics: tuple[BotResultMetric, ...] = (),
) -> PendingDecisionTrace:
    return PendingDecisionTrace(
        game_index=2,
        chart="A",
        step_index=4,
        turn_index=3,
        seat=0,
        bot_name="balanced",
        bot_id="balanced-id",
        bot_names_by_seat=("balanced", "random-a", "random-b"),
        bot_ids_by_seat=("balanced-id", "random-a-id", "random-b-id"),
        context=public_context_from_sdk(_context()),
        public_history=_history(),
        legal_actions=legal_actions_for_context(_context()),
        selected_action=selected_action or RecordedAction("submitBid", 3),
        explanation=explanation,
        selection_source="policy",
        result_metrics=result_metrics,
    )


def _outcome() -> PublicDecisionOutcome:
    return PublicDecisionOutcome(
        rank=1,
        first_place_tied=True,
        final_money=48,
    )


def test_public_context_is_an_explicit_immutable_allowlist() -> None:
    first = public_context_from_sdk(_context())
    changed_private_inputs = replace(
        _context(),
        request_id="different-request",
        deadline_at=999_999,
        received_at=888_888,
        current_hand_suit_ids=(1, 1),
        metadata={
            "opponent_hands": ((5, 5), (4, 4)),
            "deck_order": (5, 4, 3, 2, 1),
            "engine_rng_state": "secret",
        },
    )

    assert public_context_from_sdk(changed_private_inputs) == first
    assert set(first.to_payload()) == {
        "decision_kind",
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
        "legal_max_amount",
        "revealable_count",
    }
    with pytest.raises(FrozenInstanceError):
        first.bot_seat = 2  # type: ignore[misc]


def test_legal_actions_have_one_canonical_order_and_normalize_zero_bid() -> None:
    assert legal_actions_for_context(_context()) == (
        RecordedAction("pass"),
        *(RecordedAction("submitBid", amount) for amount in range(1, 8)),
    )
    reveal = _context(decision_kind="selectInfoToReveal", hand=(5, 1, 2))
    assert legal_actions_for_context(reveal) == (
        RecordedAction("pass"),
        RecordedAction("selectInfoToReveal", 0),
        RecordedAction("selectInfoToReveal", 1),
        RecordedAction("selectInfoToReveal", 2),
    )
    assert RecordedAction.from_decision(BotDecision.submit_bid(0)) == RecordedAction("pass")


def test_pending_trace_rejects_a_selected_action_outside_the_legal_set() -> None:
    with pytest.raises(ValueError, match="selected action must be legal"):
        _pending(selected_action=RecordedAction("submitBid", 8))


def test_pending_trace_requires_the_complete_canonical_legal_actions() -> None:
    pending = _pending()

    with pytest.raises(ValueError, match="canonical"):
        replace(
            pending,
            legal_actions=(
                RecordedAction("pass"),
                RecordedAction("submitBid", 3),
            ),
        )


def test_pending_trace_requires_a_seat_within_the_lineup() -> None:
    with pytest.raises(ValueError, match="seat|actor"):
        replace(
            _pending(),
            seat=3,
            context=replace(public_context_from_sdk(_context()), bot_seat=3),
        )


def test_pending_trace_requires_the_actor_name_and_id_at_its_seat() -> None:
    with pytest.raises(ValueError, match="actor"):
        replace(_pending(), bot_name="wrong-name")
    with pytest.raises(ValueError, match="actor"):
        replace(_pending(), bot_id="wrong-id")


def test_explanation_types_reject_nonfinite_values_at_construction() -> None:
    with pytest.raises(ValueError, match="finite"):
        HeuristicBidExplanation(
            resource_value=float("nan"),
            objective_completion_value=0.0,
            objective_progress_value=0.0,
            terminal_cash_value=-2.0,
            liquidity_value=-0.5,
            future_cash_value=-0.25,
            total_value=1.0,
            reservation_bid=3,
            chosen_bid=2,
        )
    with pytest.raises(ValueError, match="finite"):
        NeuralPolicyExplanation(
            predicted_value=0.5,
            selected_probability=float("inf"),
            entropy=0.1,
            legal_action_probabilities=(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="finite"):
        NeuralPolicyExplanation(
            predicted_value=True,
            selected_probability=0.5,
            entropy=0.1,
            legal_action_probabilities=(0.5, 0.5),
        )


def test_neural_explanation_schema_includes_ordered_legal_action_probabilities() -> None:
    assert "legal_action_probabilities" in {field.name for field in fields(NeuralPolicyExplanation)}


@pytest.mark.parametrize(
    "probabilities",
    (
        (),
        (float("nan"),),
        (float("inf"),),
        (-0.1, 1.1),
        (0.2, 0.2),
    ),
)
def test_neural_explanation_requires_a_finite_normalized_probability_tuple(
    probabilities: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        NeuralPolicyExplanation(
            predicted_value=0.5,
            selected_probability=0.5,
            entropy=0.1,
            legal_action_probabilities=probabilities,
        )


def _neural_explanation(
    *,
    selected_probability: float = 0.4,
    probabilities: tuple[float, ...] = (0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.05, 0.05),
) -> NeuralPolicyExplanation:
    return NeuralPolicyExplanation(
        predicted_value=0.5,
        selected_probability=selected_probability,
        entropy=0.1,
        legal_action_probabilities=probabilities,
    )


def test_neural_probabilities_must_cover_every_canonical_legal_action() -> None:
    with pytest.raises(ValueError, match="one probability per legal action"):
        _pending(
            explanation=_neural_explanation(
                probabilities=(0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1),
            )
        )


def test_neural_selected_probability_must_match_the_selected_action() -> None:
    with pytest.raises(ValueError, match="selected probability"):
        _pending(explanation=_neural_explanation(selected_probability=0.2))


def test_trace_payload_round_trip_uses_typed_public_history_and_rejects_unknown_fields() -> None:
    trace = DecisionTrace.from_pending(_pending(), _outcome())
    payload = decision_trace_payload(trace)

    assert decision_trace_from_payload(payload) == trace
    assert "root_seed" not in payload
    assert "engine_seed" not in payload
    assert set(cast(dict[str, object], payload["outcome"])) == {
        "rank",
        "first_place_tied",
        "final_money",
    }
    public_history_payload = cast(list[dict[str, object]], payload["public_history"])
    assert [event["kind"] for event in public_history_payload] == [
        "game_setup",
        "turn_opened",
        "auction_resolved",
    ]

    payload_with_hidden_state = {
        **payload,
        "opponent_hands": [[5, 5], [4, 4]],
    }
    with pytest.raises(ValueError, match="unknown decision trace fields"):
        decision_trace_from_payload(payload_with_hidden_state)

    context_payload = dict(cast(dict[str, object], payload["context"]))
    context_payload["current_hand_suit_ids"] = [2, 5]
    with pytest.raises(ValueError, match="unknown public context fields"):
        decision_trace_from_payload({**payload, "context": context_payload})


def test_typed_explanations_round_trip_without_arbitrary_metadata() -> None:
    explanation = HeuristicBidExplanation(
        resource_value=6.5,
        objective_completion_value=0.0,
        objective_progress_value=1.25,
        terminal_cash_value=-3.0,
        liquidity_value=-0.5,
        future_cash_value=-0.75,
        total_value=3.5,
        reservation_bid=4,
        chosen_bid=3,
    )
    payload = decision_trace_payload(
        DecisionTrace.from_pending(_pending(explanation=explanation), _outcome())
    )

    assert decision_trace_from_payload(payload).explanation == explanation
    explanation_payload = dict(cast(dict[str, object], payload["explanation"]))
    explanation_payload["belief_state"] = {"opponent_hidden_slots": 4}
    with pytest.raises(ValueError, match="unknown heuristic explanation fields"):
        decision_trace_from_payload({**payload, "explanation": explanation_payload})


def test_v3_rule_explanation_round_trips_and_requires_the_selected_bid() -> None:
    explanation = FixedObjectiveOverlayV3BidExplanation(
        rule="guaranteed_win",
        planned_bid=5,
        chosen_bid=3,
    )
    metric = BotResultMetric(
        namespace="fixed_objective_overlay_v3_rules",
        path=("rule_counts", "guaranteed_win"),
        aggregation="sum",
        value=1,
    )
    trace = DecisionTrace.from_pending(
        _pending(explanation=explanation, result_metrics=(metric,)),
        _outcome(),
    )
    payload = decision_trace_payload(trace)

    assert decision_trace_from_payload(payload).explanation == explanation
    assert decision_trace_from_payload(payload).result_metrics == (metric,)
    assert payload["schema_version"] == 2
    assert payload["explanation"] == {
        "kind": "fixed_objective_overlay_v3_bid",
        "rule": "guaranteed_win",
        "planned_bid": 5,
        "chosen_bid": 3,
    }
    assert payload["result_metrics"] == [
        {
            "namespace": "fixed_objective_overlay_v3_rules",
            "path": ["rule_counts", "guaranteed_win"],
            "aggregation": "sum",
            "value": 1,
        }
    ]
    with pytest.raises(ValueError, match="agree with selected action"):
        _pending(
            explanation=FixedObjectiveOverlayV3BidExplanation(
                rule="guaranteed_win",
                planned_bid=5,
                chosen_bid=2,
            )
        )


def test_v3_inactive_rule_cannot_change_the_bid() -> None:
    with pytest.raises(ValueError, match="inactive"):
        FixedObjectiveOverlayV3BidExplanation(
            rule="baseline",
            planned_bid=5,
            chosen_bid=4,
        )


def test_bot_result_metrics_round_trip_without_a_bot_specific_explanation() -> None:
    metrics = (
        BotResultMetric("example_policy", ("decisions",), "sum", 1),
        BotResultMetric("example_policy", ("confidence",), "mean", 0.75),
    )
    trace = DecisionTrace.from_pending(
        _pending(result_metrics=metrics),
        _outcome(),
    )

    payload = decision_trace_payload(trace)

    assert payload["schema_version"] == 2
    assert decision_trace_from_payload(payload).result_metrics == metrics
    assert decision_trace_from_payload(payload).explanation is None


@pytest.mark.parametrize(
    "values",
    (
        {"namespace": "Invalid", "path": ("value",), "aggregation": "sum", "value": 1},
        {"namespace": "valid", "path": (), "aggregation": "sum", "value": 1},
        {"namespace": "valid", "path": ("value",), "aggregation": "last", "value": 1},
        {"namespace": "valid", "path": ("value",), "aggregation": "sum", "value": float("nan")},
    ),
)
def test_bot_result_metrics_reject_invalid_public_contract_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="bot result metric"):
        BotResultMetric(**values)  # type: ignore[arg-type]


def test_one_decision_cannot_emit_the_same_result_metric_twice() -> None:
    metric = BotResultMetric("example_policy", ("decisions",), "sum", 1)

    with pytest.raises(ValueError, match="same bot result metric twice"):
        _pending(result_metrics=(metric, metric))


def test_neural_explanation_probabilities_round_trip_in_canonical_action_order() -> None:
    explanation = _neural_explanation()
    payload = decision_trace_payload(
        DecisionTrace.from_pending(_pending(explanation=explanation), _outcome())
    )

    assert decision_trace_from_payload(payload).explanation == explanation
    explanation_payload = cast(dict[str, object], payload["explanation"])
    assert explanation_payload["legal_action_probabilities"] == list(
        explanation.legal_action_probabilities
    )
