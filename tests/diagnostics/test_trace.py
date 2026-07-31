from __future__ import annotations

import hashlib
import json
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
    DecisionTrace,
    HeuristicBidExplanation,
    NeuralPolicyExplanation,
    OpponentAwareHeuristicBidExplanation,
    PendingDecisionTrace,
    PublicDecisionOutcome,
    RecordedAction,
    decision_trace_from_payload,
    decision_trace_payload,
    legal_actions_for_context,
    public_context_from_sdk,
)
from garboid_pocketrocks.heuristics.opponent_bids import (
    OPPONENT_BID_MODEL_NAME,
    CompetitiveBidPoint,
    OpponentBidDistribution,
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
        | OpponentAwareHeuristicBidExplanation
        | NeuralPolicyExplanation
        | None
    ) = None,
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
    )


def _outcome() -> PublicDecisionOutcome:
    return PublicDecisionOutcome(
        rank=1,
        first_place_tied=True,
        final_money=48,
    )


def _opponent_aware_explanation(
    *,
    chosen_bid: int = 3,
    model_config_digest: str = "a" * 64,
) -> OpponentAwareHeuristicBidExplanation:
    win_deltas = (0.0, 2.0, 4.0, 8.0, 6.0, -2.0, -4.0, -6.0)
    return OpponentAwareHeuristicBidExplanation(
        resource_value=6.5,
        objective_completion_value=0.0,
        objective_progress_value=1.25,
        terminal_cash_value=-3.0,
        liquidity_value=-0.5,
        future_cash_value=-0.75,
        total_value=8.0,
        reservation_bid=4,
        chosen_bid=chosen_bid,
        public_game_phase="early",
        model_name=OPPONENT_BID_MODEL_NAME,
        model_config_digest=model_config_digest,
        opponent_distributions=tuple(
            OpponentBidDistribution(
                opponent_seat=seat,
                legal_max_amount=7 - seat,
                probabilities_by_amount=(1.0,) + (0.0,) * (7 - seat),
                prior_only=False,
                history_round_count=2,
                effective_history_weight=3.0,
            )
            for seat in (1, 2)
        ),
        competitive_bid_points=tuple(
            CompetitiveBidPoint(
                effective_bid=bid,
                win_probability=0.5,
                win_delta=win_delta,
                expected_surplus=win_delta * 0.5,
            )
            for bid, win_delta in enumerate(win_deltas)
        ),
    )


def _opponent_pending(
    *,
    selected_action: RecordedAction | None = None,
    explanation: OpponentAwareHeuristicBidExplanation | None = None,
) -> PendingDecisionTrace:
    pending = _pending(selected_action=selected_action)
    return replace(
        pending,
        context=replace(pending.context, cash_by_seat=(7, 6, 5)),
        explanation=explanation or _opponent_aware_explanation(),
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


def test_ordinary_heuristic_trace_schema_v1_canonical_json_bytes_are_unchanged() -> None:
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
    trace = DecisionTrace.from_pending(_pending(explanation=explanation), _outcome())

    canonical_json_line = (
        json.dumps(
            decision_trace_payload(trace),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()

    assert len(canonical_json_line) == 1694
    assert (
        hashlib.sha256(canonical_json_line).hexdigest()
        == "58b445f7e9006c7d1d2b9c521e4409f88059af573a9c7bc9cbfc4f622fc899a5"
    )


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


def test_opponent_aware_bid_uses_strict_trace_schema_v2() -> None:
    explanation = _opponent_aware_explanation()
    trace = DecisionTrace.from_pending(_opponent_pending(explanation=explanation), _outcome())

    payload = decision_trace_payload(trace)

    assert payload["schema_version"] == 2
    explanation_payload = cast(dict[str, object], payload["explanation"])
    assert explanation_payload["kind"] == "opponent_aware_heuristic_bid"
    assert explanation_payload["public_game_phase"] == "early"
    assert explanation_payload["model_name"] == OPPONENT_BID_MODEL_NAME
    assert explanation_payload["model_config_digest"] == "a" * 64
    assert explanation_payload["opponent_distributions"] == [
        {
            "opponent_seat": seat,
            "legal_max_amount": 7 - seat,
            "probabilities_by_amount": [1.0] + [0.0] * (7 - seat),
            "prior_only": False,
            "history_round_count": 2,
            "effective_history_weight": 3.0,
        }
        for seat in (1, 2)
    ]
    assert [
        point["effective_bid"]
        for point in cast(list[dict[str, object]], explanation_payload["competitive_bid_points"])
    ] == list(range(8))
    assert decision_trace_from_payload(payload) == trace


def test_opponent_aware_explanation_requires_canonical_complete_opponents_and_bids() -> None:
    explanation = _opponent_aware_explanation()

    with pytest.raises(ValueError, match="unique and ascending"):
        replace(
            explanation,
            opponent_distributions=tuple(reversed(explanation.opponent_distributions)),
        )
    with pytest.raises(ValueError, match="all non-actor seats"):
        _opponent_pending(
            explanation=replace(
                explanation,
                opponent_distributions=explanation.opponent_distributions[:1],
            )
        )
    with pytest.raises(ValueError, match="every legal bid"):
        _opponent_pending(
            explanation=replace(
                explanation,
                competitive_bid_points=explanation.competitive_bid_points[:-1],
            )
        )


def test_opponent_aware_explanation_requires_phase_argmax_and_selected_action_agreement() -> None:
    explanation = _opponent_aware_explanation()

    with pytest.raises(ValueError, match="trace turn"):
        _opponent_pending(explanation=replace(explanation, public_game_phase="middle"))
    points = list(explanation.competitive_bid_points)
    points[2] = CompetitiveBidPoint(
        effective_bid=2,
        win_probability=0.25,
        win_delta=8.0,
        expected_surplus=2.0,
    )
    with pytest.raises(ValueError, match="maximize expected surplus"):
        _opponent_pending(
            selected_action=RecordedAction("submitBid", 2),
            explanation=replace(
                explanation,
                chosen_bid=2,
                competitive_bid_points=tuple(points),
            ),
        )
    with pytest.raises(ValueError, match="agree with selected action"):
        _opponent_pending(
            selected_action=RecordedAction("submitBid", 2),
            explanation=explanation,
        )


def test_opponent_aware_explanation_reconciles_public_cash_credit_and_ordinary_values() -> None:
    explanation = _opponent_aware_explanation()
    first_distribution = explanation.opponent_distributions[0]
    wrong_support = replace(
        first_distribution,
        legal_max_amount=5,
        probabilities_by_amount=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="public cash and credit"):
        _opponent_pending(
            explanation=replace(
                explanation,
                opponent_distributions=(
                    wrong_support,
                    explanation.opponent_distributions[1],
                ),
            )
        )

    pending = _opponent_pending()
    with pytest.raises(ValueError, match="nonnegative public credit"):
        replace(
            pending,
            context=replace(pending.context, cash_by_seat=(8, 6, 5)),
        )

    points = list(explanation.competitive_bid_points)
    points[3] = CompetitiveBidPoint(
        effective_bid=3,
        win_probability=0.5,
        win_delta=9.0,
        expected_surplus=4.5,
    )
    with pytest.raises(ValueError, match="win delta must agree with total value"):
        _opponent_pending(explanation=replace(explanation, competitive_bid_points=tuple(points)))

    with pytest.raises(ValueError, match="reservation bid"):
        _opponent_pending(explanation=replace(explanation, reservation_bid=3))


def test_opponent_aware_argmax_breaks_expected_surplus_ties_toward_lower_bid() -> None:
    explanation = _opponent_aware_explanation()
    points = list(explanation.competitive_bid_points)
    points[4] = CompetitiveBidPoint(
        effective_bid=4,
        win_probability=0.5,
        win_delta=8.0,
        expected_surplus=4.0,
    )

    assert _opponent_pending(explanation=replace(explanation, competitive_bid_points=tuple(points)))
    with pytest.raises(ValueError, match="maximize expected surplus"):
        _opponent_pending(
            selected_action=RecordedAction("submitBid", 4),
            explanation=replace(
                explanation,
                chosen_bid=4,
                competitive_bid_points=tuple(points),
            ),
        )


def test_opponent_aware_trace_decoder_rejects_nested_tampering() -> None:
    payload = decision_trace_payload(
        DecisionTrace.from_pending(
            _opponent_pending(),
            _outcome(),
        )
    )
    explanation_payload = dict(cast(dict[str, object], payload["explanation"]))
    distributions = [
        dict(item)
        for item in cast(list[dict[str, object]], explanation_payload["opponent_distributions"])
    ]
    distributions[0]["probabilities_by_amount"] = [0.25, 0.5]

    with pytest.raises(ValueError, match="sum to one"):
        decision_trace_from_payload(
            {
                **payload,
                "explanation": {
                    **explanation_payload,
                    "opponent_distributions": distributions,
                },
            }
        )

    points = [
        dict(item)
        for item in cast(list[dict[str, object]], explanation_payload["competitive_bid_points"])
    ]
    points[3]["expected_surplus"] = 4.5
    with pytest.raises(ValueError, match="expected surplus must equal"):
        decision_trace_from_payload(
            {
                **payload,
                "explanation": {
                    **explanation_payload,
                    "competitive_bid_points": points,
                },
            }
        )

    distributions = [
        dict(item)
        for item in cast(list[dict[str, object]], explanation_payload["opponent_distributions"])
    ]
    distributions[0]["hidden_hand"] = [5, 5]
    with pytest.raises(ValueError, match="unknown opponent distribution fields"):
        decision_trace_from_payload(
            {
                **payload,
                "explanation": {
                    **explanation_payload,
                    "opponent_distributions": distributions,
                },
            }
        )


@pytest.mark.parametrize("digest", ("A" * 64, "a" * 63, "g" * 64))
def test_opponent_aware_explanation_requires_lowercase_sha256_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _opponent_aware_explanation(model_config_digest=digest)


def test_opponent_aware_explanation_requires_stable_model_name() -> None:
    with pytest.raises(ValueError, match="model name is unknown"):
        replace(_opponent_aware_explanation(), model_name="public-opponent-bids-v2")


def test_trace_decoder_rejects_v1_v2_explanation_mixing() -> None:
    ordinary_payload = decision_trace_payload(
        DecisionTrace.from_pending(
            _pending(
                explanation=HeuristicBidExplanation(
                    resource_value=6.5,
                    objective_completion_value=0.0,
                    objective_progress_value=1.25,
                    terminal_cash_value=-3.0,
                    liquidity_value=-0.5,
                    future_cash_value=-0.75,
                    total_value=3.5,
                    reservation_bid=4,
                    chosen_bid=3,
                ),
            ),
            _outcome(),
        )
    )
    opponent_payload = decision_trace_payload(
        DecisionTrace.from_pending(
            _opponent_pending(),
            _outcome(),
        )
    )

    with pytest.raises(ValueError, match="schema|explanation"):
        decision_trace_from_payload({**ordinary_payload, "schema_version": 2})
    with pytest.raises(ValueError, match="schema|explanation"):
        decision_trace_from_payload({**opponent_payload, "schema_version": 1})
    with pytest.raises(ValueError, match="schema|explanation"):
        decision_trace_from_payload({**opponent_payload, "schema_version": 3})
    with pytest.raises(ValueError, match="schema|explanation"):
        decision_trace_from_payload({**opponent_payload, "explanation": None})


@pytest.mark.parametrize(
    "pending",
    (
        _pending(),
        replace(_pending(), selected_action=RecordedAction("pass")),
        replace(_pending(), selection_source="fault_fallback"),
    ),
)
def test_null_explanation_paths_remain_trace_schema_v1(
    pending: PendingDecisionTrace,
) -> None:
    payload = decision_trace_payload(DecisionTrace.from_pending(pending, _outcome()))

    assert payload["schema_version"] == 1
    assert payload["explanation"] is None
