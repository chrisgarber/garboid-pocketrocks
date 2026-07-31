"""Pure opponent-bid forecasts from an immutable public event prefix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from pocketrocks import ActionId, Suit

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.game_phase import GamePhase, game_phase_for_turn_index

_CHART_BUCKET_COUNT = 6
_AUCTION_SIZES = {
    ActionId.AUCTION1: 1,
    ActionId.AUCTION2: 2,
}
_FACE_VALUES = {
    ActionId.LOAN10: 10,
    ActionId.LOAN20: 20,
    ActionId.INVEST5: 5,
    ActionId.INVEST10: 10,
}
_ACTION_CREDIT = {
    ActionId.AUCTION1: 0,
    ActionId.AUCTION2: 0,
    ActionId.LOAN10: 10,
    ActionId.LOAN20: 20,
    ActionId.INVEST5: 0,
    ActionId.INVEST10: 0,
}
_PLAYER_PRESSURE = {
    3: Fraction(1),
    4: Fraction(11, 10),
    5: Fraction(6, 5),
}
_PHASE_PRESSURE = {
    "early": Fraction(3, 4),
    "middle": Fraction(1),
    "late": Fraction(5, 4),
}
OPPONENT_BID_MODEL_NAME = "public-opponent-bids-v1"


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True, slots=True)
class PublicOpponentBidContext:
    """The complete public field allowlist consumed by the model."""

    player_count: int
    starting_cash: int
    value_chart: tuple[int, ...]
    current_action_id: int
    cash_by_seat: tuple[int, ...]
    tiebreak_seat: int
    bot_seat: int
    legal_max_amount: int
    game_phase: GamePhase

    def __post_init__(self) -> None:
        _validate_context(self)


@dataclass(frozen=True, slots=True)
class PublicResolvedBidRound:
    """One completed public turn and its effective bids."""

    turn_index: int
    game_phase: GamePhase
    action_id: int
    resource_ids: tuple[int, int]
    bids_by_seat: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OpponentBidDistribution:
    """A seat's probability mass, indexed by legal effective-bid amount."""

    opponent_seat: int
    legal_max_amount: int
    probabilities_by_amount: tuple[float, ...]
    prior_only: bool
    history_round_count: int
    effective_history_weight: float

    def __post_init__(self) -> None:
        if not _is_integer(self.opponent_seat) or self.opponent_seat < 0:
            raise ValueError("opponent seat must be a nonnegative integer")
        if not _is_integer(self.legal_max_amount) or self.legal_max_amount < 0:
            raise ValueError("opponent legal maximum must be a nonnegative integer")
        _validate_probability_mass(self.probabilities_by_amount)
        if len(self.probabilities_by_amount) != self.legal_max_amount + 1:
            raise ValueError("opponent probability mass must enumerate its legal support")
        if not isinstance(self.prior_only, bool):
            raise ValueError("prior-only marker must be boolean")
        if not _is_integer(self.history_round_count) or self.history_round_count < 0:
            raise ValueError("history round count must be a nonnegative integer")
        if (
            isinstance(self.effective_history_weight, bool)
            or not isinstance(self.effective_history_weight, (int, float))
            or not math.isfinite(self.effective_history_weight)
            or self.effective_history_weight < 0
        ):
            raise ValueError("effective history weight must be finite and nonnegative")
        if self.prior_only and self.effective_history_weight != 0.0:
            raise ValueError("prior-only distributions cannot have effective history weight")


@dataclass(frozen=True, slots=True)
class CompetitiveBidPoint:
    """The expected competitive surplus for one legal effective bid."""

    effective_bid: int
    win_probability: float
    win_delta: float
    expected_surplus: float

    def __post_init__(self) -> None:
        if not _is_integer(self.effective_bid) or self.effective_bid < 0:
            raise ValueError("effective bid must be a nonnegative integer")
        _validate_probability(self.win_probability, name="win probability")
        if (
            isinstance(self.win_delta, bool)
            or not isinstance(self.win_delta, (int, float))
            or not math.isfinite(self.win_delta)
        ):
            raise ValueError("win delta must be finite")
        if (
            isinstance(self.expected_surplus, bool)
            or not isinstance(self.expected_surplus, (int, float))
            or not math.isfinite(self.expected_surplus)
        ):
            raise ValueError("expected surplus must be finite")
        if not math.isclose(
            self.expected_surplus,
            self.win_probability * self.win_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("expected surplus must equal win probability times win delta")


@dataclass(frozen=True, slots=True)
class LegalBidWinningForecast:
    """The predicted chance of winning with one legal effective bid."""

    effective_bid: int
    win_probability: float

    def __post_init__(self) -> None:
        if not _is_integer(self.effective_bid) or self.effective_bid < 0:
            raise ValueError("effective bid must be a nonnegative integer")
        _validate_probability(self.win_probability, name="win probability")


@dataclass(frozen=True, slots=True)
class OpponentBidForecast:
    """All component distributions and tie-aware legal-bid forecasts."""

    opponent_distributions: tuple[OpponentBidDistribution, ...]
    legal_bid_forecasts: tuple[LegalBidWinningForecast, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_distributions, tuple):
            raise ValueError("opponent distributions must be a tuple")
        if not isinstance(self.legal_bid_forecasts, tuple):
            raise ValueError("legal bid forecasts must be a tuple")
        seats = tuple(item.opponent_seat for item in self.opponent_distributions)
        if len(seats) != len(set(seats)):
            raise ValueError("opponent distributions must contain unique seats")
        amounts = tuple(item.effective_bid for item in self.legal_bid_forecasts)
        if amounts != tuple(range(len(amounts))):
            raise ValueError("legal bid forecasts must enumerate amounts from zero")


@dataclass(frozen=True, slots=True)
class OpponentBidModelConfig:
    """Frozen public smoothing strengths for the deterministic model."""

    prior_strength: float = 4.0
    minimum_history_rounds: int = 2
    same_action_phase_weight: float = 4.0
    partial_match_weight: float = 2.0
    fallback_weight: float = 1.0

    def __post_init__(self) -> None:
        if not _is_finite_positive(self.prior_strength):
            raise ValueError("prior strength must be finite and positive")
        if not _is_integer(self.minimum_history_rounds) or self.minimum_history_rounds < 2:
            raise ValueError("minimum history rounds must be an integer of at least two")
        weights = (
            self.same_action_phase_weight,
            self.partial_match_weight,
            self.fallback_weight,
        )
        if not all(_is_finite_positive(weight) for weight in weights):
            raise ValueError("history weights must be finite and positive")
        if not (self.same_action_phase_weight > self.partial_match_weight > self.fallback_weight):
            raise ValueError("history weights must be strictly strongest, medium, fallback")


DEFAULT_OPPONENT_BID_MODEL_CONFIG = OpponentBidModelConfig()


@dataclass(frozen=True, slots=True)
class _ParsedHistory:
    rounds: tuple[PublicResolvedBidRound, ...]
    current_turn: PublicTurnOpened | None
    replayed_tiebreak_seat: int


@dataclass(frozen=True, slots=True)
class _ExactDistribution:
    opponent_seat: int
    probabilities_by_amount: tuple[Fraction, ...]
    prior_only: bool
    history_round_count: int
    effective_history_weight: Fraction


def resolved_bid_rounds_from_public_history(
    history: PublicHistory,
    context: PublicOpponentBidContext,
) -> tuple[PublicResolvedBidRound, ...]:
    """Validate a public prefix and return completed bid rounds only."""

    parsed = _parse_history(history, context)
    _validate_current_turn(
        parsed.current_turn,
        context,
        len(parsed.rounds),
        replayed_tiebreak_seat=parsed.replayed_tiebreak_seat,
    )
    return parsed.rounds


def forecast_opponent_bids(
    history: PublicHistory,
    context: PublicOpponentBidContext,
    config: OpponentBidModelConfig = DEFAULT_OPPONENT_BID_MODEL_CONFIG,
) -> OpponentBidForecast:
    """Forecast every opponent bid and every legal own-bid win probability."""

    if not isinstance(config, OpponentBidModelConfig):
        raise HeuristicInputError("opponent bid model config has the wrong type")
    rounds = resolved_bid_rounds_from_public_history(history, context)
    own_cash = context.cash_by_seat[context.bot_seat]
    additional_credit = context.legal_max_amount - own_cash
    exact_distributions = tuple(
        _distribution_for_opponent(
            opponent_seat=seat,
            legal_max=context.cash_by_seat[seat] + additional_credit,
            rounds=rounds,
            context=context,
            config=config,
        )
        for seat in range(context.player_count)
        if seat != context.bot_seat
    )
    public_distributions = tuple(
        OpponentBidDistribution(
            opponent_seat=item.opponent_seat,
            legal_max_amount=len(item.probabilities_by_amount) - 1,
            probabilities_by_amount=tuple(
                float(probability) for probability in item.probabilities_by_amount
            ),
            prior_only=item.prior_only,
            history_round_count=item.history_round_count,
            effective_history_weight=float(item.effective_history_weight),
        )
        for item in exact_distributions
    )
    legal_bid_forecasts = tuple(
        LegalBidWinningForecast(
            effective_bid=amount,
            win_probability=float(
                _winning_probability(
                    amount,
                    exact_distributions,
                    context=context,
                )
            ),
        )
        for amount in range(context.legal_max_amount + 1)
    )
    return OpponentBidForecast(
        opponent_distributions=public_distributions,
        legal_bid_forecasts=legal_bid_forecasts,
    )


def _parse_history(
    history: PublicHistory,
    context: PublicOpponentBidContext,
) -> _ParsedHistory:
    if not isinstance(history, tuple) or not history:
        raise HeuristicInputError("public history must be a nonempty tuple")
    setup = history[0]
    if not isinstance(setup, PublicGameSetup) or setup.kind is not PublicEventKind.GAME_SETUP:
        raise HeuristicInputError("public history must begin with game setup")
    _validate_setup(setup, context)

    rounds: list[PublicResolvedBidRound] = []
    current_turn: PublicTurnOpened | None = None
    replayed_tiebreak_seat = setup.initial_tiebreak_seat
    expected_reveal_seat: int | None = None
    for index, event in enumerate(history[1:], start=1):
        if isinstance(event, PublicTurnOpened):
            if event.kind is not PublicEventKind.TURN_OPENED:
                raise HeuristicInputError(f"public history event {index} has a contradictory kind")
            if current_turn is not None:
                raise HeuristicInputError("public history contains two unresolved turns")
            _validate_turn(event, index)
            current_turn = event
            expected_reveal_seat = None
            continue
        if isinstance(event, PublicAuctionResolved):
            if event.kind is not PublicEventKind.AUCTION_RESOLVED:
                raise HeuristicInputError(f"public history event {index} has a contradictory kind")
            if current_turn is None:
                raise HeuristicInputError("public history resolves a turn that is not open")
            bids = _validated_bids(event, context.player_count, index)
            replayed_tiebreak_seat = _winning_seat(
                bids,
                tiebreak_seat=replayed_tiebreak_seat,
            )
            turn_index = len(rounds)
            rounds.append(
                PublicResolvedBidRound(
                    turn_index=turn_index,
                    game_phase=game_phase_for_turn_index(turn_index),
                    action_id=current_turn.action_id,
                    resource_ids=current_turn.resource_ids,
                    bids_by_seat=bids,
                )
            )
            current_turn = None
            expected_reveal_seat = replayed_tiebreak_seat
            continue
        if isinstance(event, PublicInformationRevealed):
            if event.kind is not PublicEventKind.INFORMATION_REVEALED:
                raise HeuristicInputError(f"public history event {index} has a contradictory kind")
            if expected_reveal_seat is None or current_turn is not None:
                raise HeuristicInputError("public information reveal is out of sequence")
            _validate_reveal(
                event,
                context.player_count,
                index,
                expected_seat=expected_reveal_seat,
            )
            expected_reveal_seat = None
            continue
        raise HeuristicInputError(f"public history event {index} has an unsupported type")
    return _ParsedHistory(
        rounds=tuple(rounds),
        current_turn=current_turn,
        replayed_tiebreak_seat=replayed_tiebreak_seat,
    )


def _validate_setup(setup: PublicGameSetup, context: PublicOpponentBidContext) -> None:
    if not _is_integer(setup.player_count) or not 3 <= setup.player_count <= 5:
        raise HeuristicInputError("history setup player count must be between three and five")
    if setup.player_count != context.player_count:
        raise HeuristicInputError("history setup player count contradicts current context")
    if not _is_integer(setup.starting_cash) or setup.starting_cash <= 0:
        raise HeuristicInputError("history setup starting cash must be positive")
    if setup.starting_cash != context.starting_cash:
        raise HeuristicInputError("history setup starting cash contradicts current context")
    _validate_integer_tuple("history setup value chart", setup.value_chart, _CHART_BUCKET_COUNT)
    if setup.value_chart != context.value_chart:
        raise HeuristicInputError("history setup value chart contradicts current context")
    if not _is_integer(setup.initial_tiebreak_seat) or not (
        0 <= setup.initial_tiebreak_seat < setup.player_count
    ):
        raise HeuristicInputError("history setup tiebreak seat is outside player count")


def _validate_turn(event: PublicTurnOpened, index: int) -> None:
    _known_action(event.action_id, f"public history event {index}")
    _validate_integer_tuple(f"public history event {index} resources", event.resource_ids, 2)
    if any(not 0 <= resource_id <= len(Suit) for resource_id in event.resource_ids):
        raise HeuristicInputError(f"public history event {index} has an unknown resource ID")
    if event.resource_ids[0] == 0 and event.resource_ids[1] != 0:
        raise HeuristicInputError(f"public history event {index} resources are not zero-padded")


def _validated_bids(
    event: PublicAuctionResolved,
    player_count: int,
    index: int,
) -> tuple[int, ...]:
    _validate_integer_tuple(f"public history event {index} bids", event.bids_by_seat, player_count)
    if any(bid < 0 for bid in event.bids_by_seat):
        raise HeuristicInputError(f"public history event {index} contains a negative bid")
    return event.bids_by_seat


def _validate_reveal(
    event: PublicInformationRevealed,
    player_count: int,
    index: int,
    *,
    expected_seat: int,
) -> None:
    if not _is_integer(event.seat) or not 0 <= event.seat < player_count:
        raise HeuristicInputError(
            f"public history event {index} reveal seat is outside player count"
        )
    if not _is_integer(event.suit_id) or not 1 <= event.suit_id <= len(Suit):
        raise HeuristicInputError(f"public history event {index} has an unknown revealed suit")
    if event.seat != expected_seat:
        raise HeuristicInputError(
            f"public history event {index} reveal seat contradicts the auction winner"
        )


def _validate_current_turn(
    current_turn: PublicTurnOpened | None,
    context: PublicOpponentBidContext,
    completed_round_count: int,
    *,
    replayed_tiebreak_seat: int,
) -> None:
    if current_turn is None:
        raise HeuristicInputError("public history does not contain the current open turn")
    if current_turn.action_id != context.current_action_id:
        raise HeuristicInputError("history current action contradicts current context")
    expected_phase = game_phase_for_turn_index(completed_round_count)
    if context.game_phase != expected_phase:
        raise HeuristicInputError("history current turn contradicts current game phase")
    if context.tiebreak_seat != replayed_tiebreak_seat:
        raise HeuristicInputError("history tiebreak evolution contradicts current context")


def _validate_context(context: PublicOpponentBidContext) -> None:
    if not _is_integer(context.player_count) or not 3 <= context.player_count <= 5:
        raise ValueError("player count must be between three and five")
    if not _is_integer(context.starting_cash) or context.starting_cash <= 0:
        raise ValueError("starting cash must be positive")
    _validate_integer_tuple("value chart", context.value_chart, _CHART_BUCKET_COUNT, ValueError)
    action = _known_action(context.current_action_id, "current context", ValueError)
    _validate_integer_tuple("cash by seat", context.cash_by_seat, context.player_count, ValueError)
    if any(cash < 0 for cash in context.cash_by_seat):
        raise ValueError("cash by seat must be nonnegative")
    if not _is_integer(context.tiebreak_seat) or not (
        0 <= context.tiebreak_seat < context.player_count
    ):
        raise ValueError("tiebreak seat is outside player count")
    if not _is_integer(context.bot_seat) or not 0 <= context.bot_seat < context.player_count:
        raise ValueError("bot seat is outside player count")
    if not _is_integer(context.legal_max_amount) or context.legal_max_amount < 0:
        raise ValueError("legal maximum amount must be a nonnegative integer")
    expected_legal_maximum = context.cash_by_seat[context.bot_seat] + _ACTION_CREDIT[action]
    if context.legal_max_amount != expected_legal_maximum:
        raise ValueError("legal maximum amount contradicts current action credit")
    if context.game_phase not in _PHASE_PRESSURE:
        raise ValueError("game phase must be early, middle, or late")


def _validate_integer_tuple(
    name: str,
    values: tuple[int, ...],
    expected_length: int,
    error_type: type[ValueError] = HeuristicInputError,
) -> None:
    if not isinstance(values, tuple) or len(values) != expected_length:
        raise error_type(f"{name} must be a tuple containing {expected_length} entries")
    if any(not _is_integer(value) for value in values):
        raise error_type(f"{name} must contain only integers")


def _known_action(
    value: int,
    location: str,
    error_type: type[ValueError] = HeuristicInputError,
) -> ActionId:
    if not _is_integer(value):
        raise error_type(f"{location} action ID must be an integer")
    try:
        return ActionId(value)
    except ValueError as error:
        raise error_type(f"{location} action ID is unknown") from error


def _distribution_for_opponent(
    *,
    opponent_seat: int,
    legal_max: int,
    rounds: tuple[PublicResolvedBidRound, ...],
    context: PublicOpponentBidContext,
    config: OpponentBidModelConfig,
) -> _ExactDistribution:
    prior = _public_prior(legal_max=legal_max, context=context)
    prior_strength = _fraction(config.prior_strength)
    weights = [probability * prior_strength for probability in prior]
    prior_only = len(rounds) < config.minimum_history_rounds
    effective_history_weight = Fraction()
    if not prior_only:
        for round_ in rounds:
            observed_amount = min(round_.bids_by_seat[opponent_seat], legal_max)
            history_weight = _history_weight(round_, context, config)
            weights[observed_amount] += history_weight
            effective_history_weight += history_weight
    total = sum(weights, start=Fraction())
    probabilities = tuple(weight / total for weight in weights)
    if sum(probabilities, start=Fraction()) != 1:
        raise AssertionError("exact opponent distribution must normalize")
    return _ExactDistribution(
        opponent_seat=opponent_seat,
        probabilities_by_amount=probabilities,
        prior_only=prior_only,
        history_round_count=len(rounds),
        effective_history_weight=effective_history_weight,
    )


def _public_prior(
    *,
    legal_max: int,
    context: PublicOpponentBidContext,
) -> tuple[Fraction, ...]:
    """Give every legal amount one unit plus a triangle peaking at reference."""

    reference = min(_reference_bid(context), legal_max)
    radius = max(reference, legal_max - reference, 1)
    weights = tuple(
        Fraction(1 + radius + 1 - abs(amount - reference)) for amount in range(legal_max + 1)
    )
    total = sum(weights, start=Fraction())
    return tuple(weight / total for weight in weights)


def _reference_bid(context: PublicOpponentBidContext) -> int:
    action = ActionId(context.current_action_id)
    if action in _AUCTION_SIZES:
        gaps = sorted(
            abs(right - left)
            for left, right in zip(context.value_chart[:-1], context.value_chart[1:], strict=True)
        )
        base = Fraction(gaps[len(gaps) // 2] * _AUCTION_SIZES[action])
    else:
        base = Fraction(_FACE_VALUES[action])
    adjusted = base * _PLAYER_PRESSURE[context.player_count] * _PHASE_PRESSURE[context.game_phase]
    return (adjusted.numerator * 2 + adjusted.denominator) // (2 * adjusted.denominator)


def _history_weight(
    round_: PublicResolvedBidRound,
    context: PublicOpponentBidContext,
    config: OpponentBidModelConfig,
) -> Fraction:
    same_action = round_.action_id == context.current_action_id
    same_phase = round_.game_phase == context.game_phase
    if same_action and same_phase:
        return _fraction(config.same_action_phase_weight)
    if same_action or same_phase:
        return _fraction(config.partial_match_weight)
    return _fraction(config.fallback_weight)


def _winning_seat(bids: tuple[int, ...], *, tiebreak_seat: int) -> int:
    highest_bid = max(bids)
    for offset in range(1, len(bids) + 1):
        seat = (tiebreak_seat + offset) % len(bids)
        if bids[seat] == highest_bid:
            return seat
    raise AssertionError("a nonempty bid tuple always has a winner")


def _winning_probability(
    own_bid: int,
    distributions: tuple[_ExactDistribution, ...],
    *,
    context: PublicOpponentBidContext,
) -> Fraction:
    probability = Fraction(1)
    own_order = _tiebreak_order_index(context.bot_seat, context)
    for distribution in distributions:
        opponent_ahead = _tiebreak_order_index(distribution.opponent_seat, context) < own_order
        defeated_maximum = own_bid - 1 if opponent_ahead else own_bid
        if defeated_maximum < 0:
            return Fraction()
        probability *= sum(
            distribution.probabilities_by_amount[: defeated_maximum + 1],
            start=Fraction(),
        )
    return probability


def _tiebreak_order_index(seat: int, context: PublicOpponentBidContext) -> int:
    first_seat = (context.tiebreak_seat + 1) % context.player_count
    return (seat - first_seat) % context.player_count


def _fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _validate_probability_mass(probabilities: tuple[float, ...]) -> None:
    if not isinstance(probabilities, tuple) or not probabilities:
        raise ValueError("probability mass must be a nonempty tuple")
    for probability in probabilities:
        _validate_probability(probability, name="bid probability")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bid probabilities must sum to one")


def _validate_probability(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
