"""Immutable, explicitly allowlisted decision-trace records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.game_phase import GamePhase, game_phase_for_turn_index
from garboid_pocketrocks.heuristics.opponent_bids import (
    OPPONENT_BID_MODEL_NAME,
    CompetitiveBidPoint,
    OpponentBidDistribution,
)

type RecordedActionKind = Literal["pass", "submitBid", "selectInfoToReveal"]
type SelectionSource = Literal["policy", "fault_fallback"]

_PUBLIC_CONTEXT_FIELDS = frozenset(
    {
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
)
_TRACE_FIELDS = frozenset(
    {
        "schema_version",
        "game_index",
        "chart",
        "step_index",
        "turn_index",
        "seat",
        "bot_name",
        "bot_id",
        "bot_names_by_seat",
        "bot_ids_by_seat",
        "context",
        "public_history",
        "legal_actions",
        "selected_action",
        "explanation",
        "selection_source",
        "outcome",
    }
)


@dataclass(frozen=True, slots=True)
class PublicDecisionContext:
    """Only the current SDK fields safe to expose in a public diagnostic."""

    decision_kind: Literal["submitBid", "selectInfoToReveal"]
    player_count: int
    starting_cash: int
    value_chart: tuple[int, ...]
    objective_ids: tuple[int, ...]
    current_action_id: int | None
    current_resource_ids: tuple[int, int]
    cash_by_seat: tuple[int, ...]
    tiebreak_seat: int
    won_resource_counts_by_seat: tuple[tuple[int, ...], ...]
    revealed_info_counts_by_seat: tuple[tuple[int, ...], ...]
    owned_objective_ids_by_seat: tuple[tuple[int, ...], ...]
    bot_seat: int
    legal_max_amount: int | None
    revealable_count: int

    def to_payload(self) -> dict[str, object]:
        """Return the explicit JSON-compatible public field allowlist."""

        return {
            "decision_kind": self.decision_kind,
            "player_count": self.player_count,
            "starting_cash": self.starting_cash,
            "value_chart": list(self.value_chart),
            "objective_ids": list(self.objective_ids),
            "current_action_id": self.current_action_id,
            "current_resource_ids": list(self.current_resource_ids),
            "cash_by_seat": list(self.cash_by_seat),
            "tiebreak_seat": self.tiebreak_seat,
            "won_resource_counts_by_seat": [list(row) for row in self.won_resource_counts_by_seat],
            "revealed_info_counts_by_seat": [
                list(row) for row in self.revealed_info_counts_by_seat
            ],
            "owned_objective_ids_by_seat": [list(row) for row in self.owned_objective_ids_by_seat],
            "bot_seat": self.bot_seat,
            "legal_max_amount": self.legal_max_amount,
            "revealable_count": self.revealable_count,
        }


def public_context_from_sdk(context: DecisionContext) -> PublicDecisionContext:
    """Copy the SDK context field-by-field without timing, metadata, or hand data."""

    return PublicDecisionContext(
        decision_kind=context.decision_kind,
        player_count=context.player_count,
        starting_cash=context.starting_cash,
        value_chart=context.value_chart,
        objective_ids=context.objective_ids,
        current_action_id=context.current_action_id,
        current_resource_ids=context.current_resource_ids,
        cash_by_seat=context.cash_by_seat,
        tiebreak_seat=context.tiebreak_seat,
        won_resource_counts_by_seat=context.won_resource_counts_by_seat,
        revealed_info_counts_by_seat=context.revealed_info_counts_by_seat,
        owned_objective_ids_by_seat=context.owned_objective_ids_by_seat,
        bot_seat=context.bot_seat,
        legal_max_amount=context.legal_max_amount,
        revealable_count=context.revealable_count,
    )


@dataclass(frozen=True, slots=True)
class RecordedAction:
    """Canonical public representation of one bot response."""

    action_kind: RecordedActionKind
    value: int | None = None

    def __post_init__(self) -> None:
        if self.action_kind == "pass":
            if self.value is not None:
                raise ValueError("pass action cannot contain a value")
            return
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise ValueError(f"{self.action_kind} action requires an integer value")
        if self.action_kind == "submitBid" and self.value <= 0:
            raise ValueError("recorded bids must be positive; zero is canonical pass")
        if self.action_kind == "selectInfoToReveal" and self.value < 0:
            raise ValueError("recorded reveal indexes must be nonnegative")

    @classmethod
    def from_decision(cls, decision: BotDecision) -> RecordedAction:
        if decision.action_kind == "pass" or (
            decision.action_kind == "submitBid" and decision.value == 0
        ):
            return cls("pass")
        return cls(decision.action_kind, decision.value)

    def to_payload(self) -> dict[str, object]:
        return {"action_kind": self.action_kind, "value": self.value}


def legal_actions_for_context(
    context: DecisionContext | PublicDecisionContext,
) -> tuple[RecordedAction, ...]:
    """Enumerate legal responses in one deterministic, duplicate-free order."""

    if context.decision_kind == "submitBid":
        legal_maximum = context.legal_max_amount
        if (
            legal_maximum is None
            or not isinstance(legal_maximum, int)
            or isinstance(legal_maximum, bool)
            or legal_maximum < 0
        ):
            raise ValueError("bid context requires a nonnegative legal maximum")
        return (
            RecordedAction("pass"),
            *(RecordedAction("submitBid", amount) for amount in range(1, legal_maximum + 1)),
        )
    if context.decision_kind == "selectInfoToReveal":
        if (
            not isinstance(context.revealable_count, int)
            or isinstance(context.revealable_count, bool)
            or context.revealable_count < 0
        ):
            raise ValueError("reveal context requires a nonnegative revealable count")
        return (
            RecordedAction("pass"),
            *(
                RecordedAction("selectInfoToReveal", index)
                for index in range(context.revealable_count)
            ),
        )
    raise ValueError(f"unsupported decision kind {context.decision_kind!r}")


@dataclass(frozen=True, slots=True)
class HeuristicBidExplanation:
    """Closed dollar-equivalent explanation for a heuristic bid."""

    resource_value: float
    objective_completion_value: float
    objective_progress_value: float
    terminal_cash_value: float
    liquidity_value: float
    future_cash_value: float
    total_value: float
    reservation_bid: int
    chosen_bid: int

    def __post_init__(self) -> None:
        _validate_explanation(self)


@dataclass(frozen=True, slots=True)
class OpponentAwareHeuristicBidExplanation(HeuristicBidExplanation):
    """A heuristic bid with its complete public opponent-model audit trail."""

    public_game_phase: GamePhase
    model_name: str
    model_config_digest: str
    opponent_distributions: tuple[OpponentBidDistribution, ...]
    competitive_bid_points: tuple[CompetitiveBidPoint, ...]

    def __post_init__(self) -> None:
        _validate_explanation(self)


@dataclass(frozen=True, slots=True)
class NeuralPolicyExplanation:
    """Closed explanation for one masked neural-policy selection."""

    predicted_value: float
    selected_probability: float
    entropy: float
    legal_action_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_explanation(self)


type DecisionExplanation = (
    HeuristicBidExplanation
    | OpponentAwareHeuristicBidExplanation
    | NeuralPolicyExplanation
)


@dataclass(frozen=True, slots=True)
class ExplainedBotDecision:
    """A selected SDK decision and optional explanation computed in the same call."""

    decision: BotDecision
    explanation: DecisionExplanation | None = None

    def __post_init__(self) -> None:
        if self.explanation is not None:
            _validate_explanation(self.explanation)


@dataclass(frozen=True, slots=True)
class PendingDecisionTrace:
    """One public decision record awaiting the game's public terminal outcome."""

    game_index: int | None
    chart: str
    step_index: int
    turn_index: int
    seat: int
    bot_name: str
    bot_id: str
    bot_names_by_seat: tuple[str, ...]
    bot_ids_by_seat: tuple[str, ...]
    context: PublicDecisionContext
    public_history: PublicHistory
    legal_actions: tuple[RecordedAction, ...]
    selected_action: RecordedAction
    explanation: DecisionExplanation | None
    selection_source: SelectionSource

    def __post_init__(self) -> None:
        _validate_pending(self)


@dataclass(frozen=True, slots=True)
class PublicDecisionOutcome:
    """Terminal public result authoritative in the ordinary game summary."""

    rank: int
    first_place_tied: bool
    final_money: int


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """One complete decision diagnostic with its eventual public outcome."""

    game_index: int | None
    chart: str
    step_index: int
    turn_index: int
    seat: int
    bot_name: str
    bot_id: str
    bot_names_by_seat: tuple[str, ...]
    bot_ids_by_seat: tuple[str, ...]
    context: PublicDecisionContext
    public_history: PublicHistory
    legal_actions: tuple[RecordedAction, ...]
    selected_action: RecordedAction
    explanation: DecisionExplanation | None
    selection_source: SelectionSource
    outcome: PublicDecisionOutcome

    def __post_init__(self) -> None:
        _validate_pending(self)

    @classmethod
    def from_pending(
        cls,
        pending: PendingDecisionTrace,
        outcome: PublicDecisionOutcome,
    ) -> DecisionTrace:
        return cls(
            game_index=pending.game_index,
            chart=pending.chart,
            step_index=pending.step_index,
            turn_index=pending.turn_index,
            seat=pending.seat,
            bot_name=pending.bot_name,
            bot_id=pending.bot_id,
            bot_names_by_seat=pending.bot_names_by_seat,
            bot_ids_by_seat=pending.bot_ids_by_seat,
            context=pending.context,
            public_history=pending.public_history,
            legal_actions=pending.legal_actions,
            selected_action=pending.selected_action,
            explanation=pending.explanation,
            selection_source=pending.selection_source,
            outcome=outcome,
        )


def _validate_pending(trace: PendingDecisionTrace | DecisionTrace) -> None:
    if trace.legal_actions != legal_actions_for_context(trace.context):
        raise ValueError("legal actions must equal the canonical actions for the public context")
    if trace.selected_action not in trace.legal_actions:
        raise ValueError("selected action must be legal")
    if trace.selection_source not in ("policy", "fault_fallback"):
        raise ValueError("selection source is unknown")
    if trace.selection_source == "fault_fallback" and trace.explanation is not None:
        raise ValueError("fault fallback cannot have a policy explanation")
    if trace.explanation is not None:
        _validate_explanation(trace.explanation)
        if isinstance(trace.explanation, HeuristicBidExplanation):
            selected_bid = (
                0 if trace.selected_action.action_kind == "pass" else trace.selected_action.value
            )
            if trace.selected_action.action_kind not in ("pass", "submitBid"):
                raise ValueError("heuristic bid explanation requires a bid action")
            if trace.explanation.chosen_bid != selected_bid:
                raise ValueError("heuristic chosen bid must agree with selected action")
            if isinstance(trace.explanation, OpponentAwareHeuristicBidExplanation):
                _validate_opponent_aware_explanation_against_trace(
                    trace.explanation,
                    trace,
                )
        elif isinstance(trace.explanation, NeuralPolicyExplanation):
            probabilities = trace.explanation.legal_action_probabilities
            if len(probabilities) != len(trace.legal_actions):
                raise ValueError("neural explanation requires one probability per legal action")
            selected_index = trace.legal_actions.index(trace.selected_action)
            if not math.isclose(
                trace.explanation.selected_probability,
                probabilities[selected_index],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("neural selected probability must match the selected legal action")
    if (
        len(trace.bot_names_by_seat) != trace.context.player_count
        or len(trace.bot_ids_by_seat) != trace.context.player_count
    ):
        raise ValueError("lineup identities must match player count")
    if not 0 <= trace.seat < trace.context.player_count:
        raise ValueError("trace seat must be within the lineup")
    if trace.seat != trace.context.bot_seat:
        raise ValueError("trace seat must match public context")
    if (
        trace.bot_name != trace.bot_names_by_seat[trace.seat]
        or trace.bot_id != trace.bot_ids_by_seat[trace.seat]
    ):
        raise ValueError("trace actor identity must match the lineup seat")


def _validate_opponent_aware_explanation_against_trace(
    explanation: OpponentAwareHeuristicBidExplanation,
    trace: PendingDecisionTrace | DecisionTrace,
) -> None:
    context = trace.context
    if context.decision_kind != "submitBid" or context.legal_max_amount is None:
        raise ValueError("opponent-aware explanation requires a bid context")
    if explanation.public_game_phase != game_phase_for_turn_index(trace.turn_index):
        raise ValueError("opponent-aware explanation phase must agree with the trace turn")
    if not 0 <= trace.seat < context.player_count:
        raise ValueError("opponent-aware explanation requires a valid actor seat")
    if len(context.cash_by_seat) != context.player_count:
        raise ValueError("opponent-aware explanation requires cash for every seat")
    actor_cash = context.cash_by_seat[trace.seat]
    additional_credit = context.legal_max_amount - actor_cash
    if additional_credit < 0:
        raise ValueError("opponent-aware explanation requires nonnegative public credit")
    expected_seats = tuple(seat for seat in range(context.player_count) if seat != trace.seat)
    actual_seats = tuple(
        distribution.opponent_seat for distribution in explanation.opponent_distributions
    )
    if actual_seats != expected_seats:
        raise ValueError(
            "opponent-aware distributions must cover all non-actor seats in canonical order"
        )
    for distribution in explanation.opponent_distributions:
        expected_legal_max = context.cash_by_seat[distribution.opponent_seat] + additional_credit
        if distribution.legal_max_amount != expected_legal_max:
            raise ValueError(
                "opponent distribution legal maximum must agree with public cash and credit"
            )
    expected_bids = tuple(range(context.legal_max_amount + 1))
    actual_bids = tuple(point.effective_bid for point in explanation.competitive_bid_points)
    if actual_bids != expected_bids:
        raise ValueError("competitive bid points must enumerate every legal bid from zero")
    if explanation.chosen_bid >= len(explanation.competitive_bid_points):
        raise ValueError("opponent-aware chosen bid must have a competitive bid point")
    chosen_point = explanation.competitive_bid_points[explanation.chosen_bid]
    if not math.isclose(
        chosen_point.win_delta,
        explanation.total_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("chosen competitive bid win delta must agree with total value")
    expected_reservation_bid = max(
        (
            point.effective_bid
            for point in explanation.competitive_bid_points
            if point.win_delta >= 0.0
        ),
        default=0,
    )
    if explanation.reservation_bid != expected_reservation_bid:
        raise ValueError("opponent-aware reservation bid must agree with nonnegative win deltas")
    best_point = max(
        explanation.competitive_bid_points,
        key=lambda point: (point.expected_surplus, -point.effective_bid),
    )
    if explanation.chosen_bid != best_point.effective_bid:
        raise ValueError(
            "opponent-aware chosen bid must maximize expected surplus with lower-bid ties"
        )


def _validate_explanation(explanation: DecisionExplanation) -> None:
    if isinstance(explanation, HeuristicBidExplanation):
        heuristic_values = (
            explanation.resource_value,
            explanation.objective_completion_value,
            explanation.objective_progress_value,
            explanation.terminal_cash_value,
            explanation.liquidity_value,
            explanation.future_cash_value,
            explanation.total_value,
        )
        if not all(_is_finite_number(value) for value in heuristic_values):
            raise ValueError("heuristic explanation values must be finite")
        if (
            not isinstance(explanation.reservation_bid, int)
            or isinstance(explanation.reservation_bid, bool)
            or explanation.reservation_bid < 0
            or not isinstance(explanation.chosen_bid, int)
            or isinstance(explanation.chosen_bid, bool)
            or explanation.chosen_bid < 0
            or explanation.chosen_bid > explanation.reservation_bid
        ):
            raise ValueError("heuristic explanation bids are invalid")
        if isinstance(explanation, OpponentAwareHeuristicBidExplanation):
            if explanation.public_game_phase not in ("early", "middle", "late"):
                raise ValueError("opponent-aware explanation public phase is unknown")
            if explanation.model_name != OPPONENT_BID_MODEL_NAME:
                raise ValueError("opponent-aware explanation model name is unknown")
            if re.fullmatch(r"[0-9a-f]{64}", explanation.model_config_digest) is None:
                raise ValueError(
                    "opponent-aware model config digest must be a lowercase SHA-256 digest"
                )
            if not isinstance(explanation.opponent_distributions, tuple) or not all(
                isinstance(distribution, OpponentBidDistribution)
                for distribution in explanation.opponent_distributions
            ):
                raise ValueError("opponent distributions must be a typed tuple")
            seats = tuple(
                distribution.opponent_seat for distribution in explanation.opponent_distributions
            )
            if seats != tuple(sorted(set(seats))):
                raise ValueError("opponent distribution seats must be unique and ascending")
            if (
                not isinstance(explanation.competitive_bid_points, tuple)
                or not explanation.competitive_bid_points
                or not all(
                    isinstance(point, CompetitiveBidPoint)
                    for point in explanation.competitive_bid_points
                )
            ):
                raise ValueError("competitive bid points must be a nonempty typed tuple")
            bids = tuple(point.effective_bid for point in explanation.competitive_bid_points)
            if bids != tuple(range(len(bids))):
                raise ValueError("competitive bid points must enumerate bids from zero")
        return
    neural_values = (
        explanation.predicted_value,
        explanation.selected_probability,
        explanation.entropy,
    )
    if not all(_is_finite_number(value) for value in neural_values):
        raise ValueError("neural explanation values must be finite")
    if not 0.0 <= explanation.selected_probability <= 1.0:
        raise ValueError("neural selected probability must be between zero and one")
    if explanation.entropy < 0.0:
        raise ValueError("neural entropy must be nonnegative")
    probabilities = explanation.legal_action_probabilities
    if (
        not isinstance(probabilities, tuple)
        or not probabilities
        or not all(
            _is_finite_number(probability) and 0.0 <= probability <= 1.0
            for probability in probabilities
        )
        or not math.isclose(
            math.fsum(probabilities),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("neural legal action probabilities must be a finite normalized tuple")


def decision_trace_payload(trace: DecisionTrace) -> dict[str, object]:
    """Encode a complete trace without recursively inspecting foreign objects."""

    return {
        "schema_version": (
            2 if isinstance(trace.explanation, OpponentAwareHeuristicBidExplanation) else 1
        ),
        "game_index": trace.game_index,
        "chart": trace.chart,
        "step_index": trace.step_index,
        "turn_index": trace.turn_index,
        "seat": trace.seat,
        "bot_name": trace.bot_name,
        "bot_id": trace.bot_id,
        "bot_names_by_seat": list(trace.bot_names_by_seat),
        "bot_ids_by_seat": list(trace.bot_ids_by_seat),
        "context": trace.context.to_payload(),
        "public_history": [_public_event_payload(event) for event in trace.public_history],
        "legal_actions": [action.to_payload() for action in trace.legal_actions],
        "selected_action": trace.selected_action.to_payload(),
        "explanation": _explanation_payload(trace.explanation),
        "selection_source": trace.selection_source,
        "outcome": _outcome_payload(trace.outcome),
    }


def decision_trace_from_payload(payload: Mapping[str, object]) -> DecisionTrace:
    """Strictly decode a trace, rejecting every unrecognized schema field."""

    _require_exact_fields(payload, _TRACE_FIELDS, "decision trace")
    schema_version = _integer(payload["schema_version"], "schema_version")
    if schema_version not in (1, 2):
        raise ValueError("unsupported decision trace schema version")
    context = _public_context_from_payload(_mapping(payload["context"], "context"))
    return DecisionTrace(
        game_index=_optional_integer(payload["game_index"], "game_index"),
        chart=_string(payload["chart"], "chart"),
        step_index=_integer(payload["step_index"], "step_index"),
        turn_index=_integer(payload["turn_index"], "turn_index"),
        seat=_integer(payload["seat"], "seat"),
        bot_name=_string(payload["bot_name"], "bot_name"),
        bot_id=_string(payload["bot_id"], "bot_id"),
        bot_names_by_seat=_string_tuple(payload["bot_names_by_seat"], "bot_names_by_seat"),
        bot_ids_by_seat=_string_tuple(payload["bot_ids_by_seat"], "bot_ids_by_seat"),
        context=context,
        public_history=tuple(
            _public_event_from_payload(_mapping(item, "public history event"))
            for item in _sequence(payload["public_history"], "public_history")
        ),
        legal_actions=tuple(
            _action_from_payload(_mapping(item, "legal action"))
            for item in _sequence(payload["legal_actions"], "legal_actions")
        ),
        selected_action=_action_from_payload(
            _mapping(payload["selected_action"], "selected_action")
        ),
        explanation=_explanation_from_payload(
            payload["explanation"],
            schema_version=schema_version,
        ),
        selection_source=_selection_source(payload["selection_source"]),
        outcome=_outcome_from_payload(_mapping(payload["outcome"], "outcome")),
    )


def _public_context_from_payload(payload: Mapping[str, object]) -> PublicDecisionContext:
    _require_exact_fields(payload, _PUBLIC_CONTEXT_FIELDS, "public context")
    decision_kind = payload["decision_kind"]
    if decision_kind not in ("submitBid", "selectInfoToReveal"):
        raise ValueError("public context decision_kind is unknown")
    resources = _integer_tuple(payload["current_resource_ids"], "current_resource_ids")
    if len(resources) != 2:
        raise ValueError("current_resource_ids must contain two values")
    return PublicDecisionContext(
        decision_kind=decision_kind,
        player_count=_integer(payload["player_count"], "player_count"),
        starting_cash=_integer(payload["starting_cash"], "starting_cash"),
        value_chart=_integer_tuple(payload["value_chart"], "value_chart"),
        objective_ids=_integer_tuple(payload["objective_ids"], "objective_ids"),
        current_action_id=_optional_integer(payload["current_action_id"], "current_action_id"),
        current_resource_ids=(resources[0], resources[1]),
        cash_by_seat=_integer_tuple(payload["cash_by_seat"], "cash_by_seat"),
        tiebreak_seat=_integer(payload["tiebreak_seat"], "tiebreak_seat"),
        won_resource_counts_by_seat=_integer_matrix(
            payload["won_resource_counts_by_seat"],
            "won_resource_counts_by_seat",
        ),
        revealed_info_counts_by_seat=_integer_matrix(
            payload["revealed_info_counts_by_seat"],
            "revealed_info_counts_by_seat",
        ),
        owned_objective_ids_by_seat=_integer_matrix(
            payload["owned_objective_ids_by_seat"],
            "owned_objective_ids_by_seat",
        ),
        bot_seat=_integer(payload["bot_seat"], "bot_seat"),
        legal_max_amount=_optional_integer(payload["legal_max_amount"], "legal_max_amount"),
        revealable_count=_integer(payload["revealable_count"], "revealable_count"),
    )


def _action_from_payload(payload: Mapping[str, object]) -> RecordedAction:
    _require_exact_fields(payload, frozenset({"action_kind", "value"}), "recorded action")
    kind = payload["action_kind"]
    if kind not in ("pass", "submitBid", "selectInfoToReveal"):
        raise ValueError("recorded action kind is unknown")
    return RecordedAction(
        kind,
        _optional_integer(payload["value"], "action value"),
    )


def _public_event_payload(event: PublicEvent) -> dict[str, object]:
    if isinstance(event, PublicGameSetup):
        return {
            "kind": event.kind.value,
            "player_count": event.player_count,
            "starting_cash": event.starting_cash,
            "value_chart": list(event.value_chart),
            "initial_tiebreak_seat": event.initial_tiebreak_seat,
            "objective_ids": list(event.objective_ids),
        }
    if isinstance(event, PublicTurnOpened):
        return {
            "kind": event.kind.value,
            "action_id": event.action_id,
            "resource_ids": list(event.resource_ids),
        }
    if isinstance(event, PublicAuctionResolved):
        return {
            "kind": event.kind.value,
            "bids_by_seat": list(event.bids_by_seat),
        }
    if isinstance(event, PublicInformationRevealed):
        return {
            "kind": event.kind.value,
            "seat": event.seat,
            "suit_id": event.suit_id,
        }
    raise TypeError(f"unsupported public event {type(event).__name__}")


def _public_event_from_payload(payload: Mapping[str, object]) -> PublicEvent:
    kind = payload.get("kind")
    if kind == PublicEventKind.GAME_SETUP.value:
        _require_exact_fields(
            payload,
            frozenset(
                {
                    "kind",
                    "player_count",
                    "starting_cash",
                    "value_chart",
                    "initial_tiebreak_seat",
                    "objective_ids",
                }
            ),
            "game setup event",
        )
        return PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=_integer(payload["player_count"], "player_count"),
            starting_cash=_integer(payload["starting_cash"], "starting_cash"),
            value_chart=_integer_tuple(payload["value_chart"], "value_chart"),
            initial_tiebreak_seat=_integer(
                payload["initial_tiebreak_seat"], "initial_tiebreak_seat"
            ),
            objective_ids=_integer_tuple(payload["objective_ids"], "objective_ids"),
        )
    if kind == PublicEventKind.TURN_OPENED.value:
        _require_exact_fields(
            payload,
            frozenset({"kind", "action_id", "resource_ids"}),
            "turn opened event",
        )
        resources = _integer_tuple(payload["resource_ids"], "resource_ids")
        if len(resources) != 2:
            raise ValueError("turn resource_ids must contain two values")
        return PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=_integer(payload["action_id"], "action_id"),
            resource_ids=(resources[0], resources[1]),
        )
    if kind == PublicEventKind.AUCTION_RESOLVED.value:
        _require_exact_fields(
            payload,
            frozenset({"kind", "bids_by_seat"}),
            "auction resolved event",
        )
        return PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=_integer_tuple(payload["bids_by_seat"], "bids_by_seat"),
        )
    if kind == PublicEventKind.INFORMATION_REVEALED.value:
        _require_exact_fields(
            payload,
            frozenset({"kind", "seat", "suit_id"}),
            "information revealed event",
        )
        return PublicInformationRevealed(
            kind=PublicEventKind.INFORMATION_REVEALED,
            seat=_integer(payload["seat"], "seat"),
            suit_id=_integer(payload["suit_id"], "suit_id"),
        )
    raise ValueError("public history event kind is unknown")


def _explanation_payload(explanation: DecisionExplanation | None) -> dict[str, object] | None:
    if explanation is None:
        return None
    if isinstance(explanation, OpponentAwareHeuristicBidExplanation):
        return {
            "kind": "opponent_aware_heuristic_bid",
            "resource_value": explanation.resource_value,
            "objective_completion_value": explanation.objective_completion_value,
            "objective_progress_value": explanation.objective_progress_value,
            "terminal_cash_value": explanation.terminal_cash_value,
            "liquidity_value": explanation.liquidity_value,
            "future_cash_value": explanation.future_cash_value,
            "total_value": explanation.total_value,
            "reservation_bid": explanation.reservation_bid,
            "chosen_bid": explanation.chosen_bid,
            "public_game_phase": explanation.public_game_phase,
            "model_name": explanation.model_name,
            "model_config_digest": explanation.model_config_digest,
            "opponent_distributions": [
                {
                    "opponent_seat": distribution.opponent_seat,
                    "legal_max_amount": distribution.legal_max_amount,
                    "probabilities_by_amount": list(distribution.probabilities_by_amount),
                    "prior_only": distribution.prior_only,
                    "history_round_count": distribution.history_round_count,
                    "effective_history_weight": distribution.effective_history_weight,
                }
                for distribution in explanation.opponent_distributions
            ],
            "competitive_bid_points": [
                {
                    "effective_bid": point.effective_bid,
                    "win_probability": point.win_probability,
                    "win_delta": point.win_delta,
                    "expected_surplus": point.expected_surplus,
                }
                for point in explanation.competitive_bid_points
            ],
        }
    if isinstance(explanation, HeuristicBidExplanation):
        return {
            "kind": "heuristic_bid",
            "resource_value": explanation.resource_value,
            "objective_completion_value": explanation.objective_completion_value,
            "objective_progress_value": explanation.objective_progress_value,
            "terminal_cash_value": explanation.terminal_cash_value,
            "liquidity_value": explanation.liquidity_value,
            "future_cash_value": explanation.future_cash_value,
            "total_value": explanation.total_value,
            "reservation_bid": explanation.reservation_bid,
            "chosen_bid": explanation.chosen_bid,
        }
    return {
        "kind": "neural_policy",
        "predicted_value": explanation.predicted_value,
        "selected_probability": explanation.selected_probability,
        "entropy": explanation.entropy,
        "legal_action_probabilities": list(explanation.legal_action_probabilities),
    }


def _explanation_from_payload(
    payload: object,
    *,
    schema_version: int,
) -> DecisionExplanation | None:
    if payload is None:
        if schema_version == 2:
            raise ValueError(f"decision trace schema v{schema_version} requires an explanation")
        return None
    explanation = _mapping(payload, "explanation")
    kind = explanation.get("kind")
    if kind == "opponent_aware_heuristic_bid":
        if schema_version != 2:
            raise ValueError("opponent-aware explanation requires decision trace schema v2")
        fields = frozenset(
            {
                "kind",
                "resource_value",
                "objective_completion_value",
                "objective_progress_value",
                "terminal_cash_value",
                "liquidity_value",
                "future_cash_value",
                "total_value",
                "reservation_bid",
                "chosen_bid",
                "public_game_phase",
                "model_name",
                "model_config_digest",
                "opponent_distributions",
                "competitive_bid_points",
            }
        )
        _require_exact_fields(explanation, fields, "opponent-aware heuristic explanation")
        public_game_phase = explanation["public_game_phase"]
        if public_game_phase not in ("early", "middle", "late"):
            raise ValueError("opponent-aware explanation public phase is unknown")
        opponent_aware_explanation = OpponentAwareHeuristicBidExplanation(
            resource_value=_number(explanation["resource_value"], "resource_value"),
            objective_completion_value=_number(
                explanation["objective_completion_value"],
                "objective_completion_value",
            ),
            objective_progress_value=_number(
                explanation["objective_progress_value"],
                "objective_progress_value",
            ),
            terminal_cash_value=_number(
                explanation["terminal_cash_value"],
                "terminal_cash_value",
            ),
            liquidity_value=_number(explanation["liquidity_value"], "liquidity_value"),
            future_cash_value=_number(
                explanation["future_cash_value"],
                "future_cash_value",
            ),
            total_value=_number(explanation["total_value"], "total_value"),
            reservation_bid=_integer(explanation["reservation_bid"], "reservation_bid"),
            chosen_bid=_integer(explanation["chosen_bid"], "chosen_bid"),
            public_game_phase=public_game_phase,
            model_name=_string(explanation["model_name"], "model_name"),
            model_config_digest=_string(
                explanation["model_config_digest"],
                "model_config_digest",
            ),
            opponent_distributions=tuple(
                _opponent_distribution_from_payload(_mapping(item, "opponent distribution"))
                for item in _sequence(
                    explanation["opponent_distributions"],
                    "opponent_distributions",
                )
            ),
            competitive_bid_points=tuple(
                _competitive_bid_point_from_payload(_mapping(item, "competitive bid point"))
                for item in _sequence(
                    explanation["competitive_bid_points"],
                    "competitive_bid_points",
                )
            ),
        )
        _validate_explanation(opponent_aware_explanation)
        return opponent_aware_explanation
    if kind == "heuristic_bid":
        if schema_version != 1:
            raise ValueError("ordinary heuristic explanation requires decision trace schema v1")
        fields = frozenset(
            {
                "kind",
                "resource_value",
                "objective_completion_value",
                "objective_progress_value",
                "terminal_cash_value",
                "liquidity_value",
                "future_cash_value",
                "total_value",
                "reservation_bid",
                "chosen_bid",
            }
        )
        _require_exact_fields(explanation, fields, "heuristic explanation")
        heuristic_explanation = HeuristicBidExplanation(
            resource_value=_number(explanation["resource_value"], "resource_value"),
            objective_completion_value=_number(
                explanation["objective_completion_value"],
                "objective_completion_value",
            ),
            objective_progress_value=_number(
                explanation["objective_progress_value"],
                "objective_progress_value",
            ),
            terminal_cash_value=_number(
                explanation["terminal_cash_value"],
                "terminal_cash_value",
            ),
            liquidity_value=_number(explanation["liquidity_value"], "liquidity_value"),
            future_cash_value=_number(
                explanation["future_cash_value"],
                "future_cash_value",
            ),
            total_value=_number(explanation["total_value"], "total_value"),
            reservation_bid=_integer(explanation["reservation_bid"], "reservation_bid"),
            chosen_bid=_integer(explanation["chosen_bid"], "chosen_bid"),
        )
        _validate_explanation(heuristic_explanation)
        return heuristic_explanation
    if kind == "neural_policy":
        if schema_version != 1:
            raise ValueError("neural explanation requires decision trace schema v1")
        fields = frozenset(
            {
                "kind",
                "predicted_value",
                "selected_probability",
                "entropy",
                "legal_action_probabilities",
            }
        )
        _require_exact_fields(explanation, fields, "neural explanation")
        neural_explanation = NeuralPolicyExplanation(
            predicted_value=_number(explanation["predicted_value"], "predicted_value"),
            selected_probability=_number(
                explanation["selected_probability"],
                "selected_probability",
            ),
            entropy=_number(explanation["entropy"], "entropy"),
            legal_action_probabilities=tuple(
                _number(item, "legal action probability")
                for item in _sequence(
                    explanation["legal_action_probabilities"],
                    "legal_action_probabilities",
                )
            ),
        )
        _validate_explanation(neural_explanation)
        return neural_explanation
    raise ValueError("explanation kind is unknown")


def _opponent_distribution_from_payload(
    payload: Mapping[str, object],
) -> OpponentBidDistribution:
    fields = frozenset(
        {
            "opponent_seat",
            "legal_max_amount",
            "probabilities_by_amount",
            "prior_only",
            "history_round_count",
            "effective_history_weight",
        }
    )
    _require_exact_fields(payload, fields, "opponent distribution")
    return OpponentBidDistribution(
        opponent_seat=_integer(payload["opponent_seat"], "opponent_seat"),
        legal_max_amount=_integer(payload["legal_max_amount"], "legal_max_amount"),
        probabilities_by_amount=tuple(
            _number(item, "opponent probability")
            for item in _sequence(
                payload["probabilities_by_amount"],
                "probabilities_by_amount",
            )
        ),
        prior_only=_boolean(payload["prior_only"], "prior_only"),
        history_round_count=_integer(
            payload["history_round_count"],
            "history_round_count",
        ),
        effective_history_weight=_number(
            payload["effective_history_weight"],
            "effective_history_weight",
        ),
    )


def _competitive_bid_point_from_payload(
    payload: Mapping[str, object],
) -> CompetitiveBidPoint:
    fields = frozenset(
        {
            "effective_bid",
            "win_probability",
            "win_delta",
            "expected_surplus",
        }
    )
    _require_exact_fields(payload, fields, "competitive bid point")
    return CompetitiveBidPoint(
        effective_bid=_integer(payload["effective_bid"], "effective_bid"),
        win_probability=_number(payload["win_probability"], "win_probability"),
        win_delta=_number(payload["win_delta"], "win_delta"),
        expected_surplus=_number(payload["expected_surplus"], "expected_surplus"),
    )


def _outcome_payload(outcome: PublicDecisionOutcome) -> dict[str, object]:
    return {
        "rank": outcome.rank,
        "first_place_tied": outcome.first_place_tied,
        "final_money": outcome.final_money,
    }


def _outcome_from_payload(payload: Mapping[str, object]) -> PublicDecisionOutcome:
    fields = frozenset({"rank", "first_place_tied", "final_money"})
    _require_exact_fields(payload, fields, "public outcome")
    return PublicDecisionOutcome(
        rank=_integer(payload["rank"], "rank"),
        first_place_tied=_boolean(payload["first_place_tied"], "first_place_tied"),
        final_money=_integer(payload["final_money"], "final_money"),
    )


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = frozenset(payload)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")
    if missing:
        raise ValueError(f"missing {name} fields: {missing}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return cast(Sequence[object], value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    return tuple(_integer(item, name) for item in _sequence(value, name))


def _integer_matrix(value: object, name: str) -> tuple[tuple[int, ...], ...]:
    return tuple(_integer_tuple(row, f"{name} row") for row in _sequence(value, name))


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    return tuple(_string(item, name) for item in _sequence(value, name))


def _selection_source(value: object) -> SelectionSource:
    if value not in ("policy", "fault_fallback"):
        raise ValueError("selection_source is unknown")
    return value
