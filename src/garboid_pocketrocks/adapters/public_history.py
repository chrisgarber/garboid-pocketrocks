from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Literal

from pocketrocks import OBJECTIVES, ActionId, Suit
from pocketrocks.sim.constants import (
    ACTION_DECK,
    ACTION_WIRE_IDS,
    INFO_CARDS_PER_PLAYER,
    INVEST_PAYOUT,
    ITEM_DECK_SUITS,
    LOAN_PRINCIPAL,
    OBJECTIVES_PER_GAME,
    STARTING_CASH,
    VALUE_CHARTS,
    objective_pattern_met,
)

_ACTION_NAME_BY_ID = {wire_id: name for name, wire_id in ACTION_WIRE_IDS.items()}
_ACTION_CREDIT_BY_ID = {
    wire_id: LOAN_PRINCIPAL.get(name, 0) for name, wire_id in ACTION_WIRE_IDS.items()
}
_ACTION_COUNT_BY_ID = {
    wire_id: ACTION_DECK.count(name) for name, wire_id in ACTION_WIRE_IDS.items()
}
_AUCTION_ONE = int(ActionId.AUCTION1)
_AUCTION_TWO = int(ActionId.AUCTION2)
_SUIT_COUNT = len(Suit)
_RESOURCE_COUNT_BY_SUIT = tuple(
    ITEM_DECK_SUITS.count(suit_id) for suit_id in range(1, _SUIT_COUNT + 1)
)


class PublicHistoryCompatibilityError(ValueError):
    """Raised when a raw SDK frame no longer matches the pinned public schema."""


class PublicEventKind(StrEnum):
    GAME_SETUP = "game_setup"
    TURN_OPENED = "turn_opened"
    AUCTION_RESOLVED = "auction_resolved"
    INFORMATION_REVEALED = "information_revealed"


@dataclass(frozen=True, slots=True)
class PublicGameSetup:
    kind: Literal[PublicEventKind.GAME_SETUP]
    player_count: int
    starting_cash: int
    value_chart: tuple[int, ...]
    initial_tiebreak_seat: int
    objective_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublicTurnOpened:
    kind: Literal[PublicEventKind.TURN_OPENED]
    action_id: int
    resource_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PublicAuctionResolved:
    kind: Literal[PublicEventKind.AUCTION_RESOLVED]
    bids_by_seat: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublicInformationRevealed:
    kind: Literal[PublicEventKind.INFORMATION_REVEALED]
    seat: int
    suit_id: int


type PublicEvent = (
    PublicGameSetup | PublicTurnOpened | PublicAuctionResolved | PublicInformationRevealed
)
type PublicHistory = tuple[PublicEvent, ...]
type PublicHistoryPhase = Literal["ready_for_turn", "turn_open", "reveal_pending"]


@dataclass(frozen=True, slots=True)
class ValidatedPublicHistory:
    """Cumulative public state reconstructed from one exact event sequence."""

    setup: PublicGameSetup
    phase: PublicHistoryPhase
    latest_turn: PublicTurnOpened | None
    tiebreak_seat: int
    cash_by_seat: tuple[int, ...]
    won_resource_counts_by_seat: tuple[tuple[int, ...], ...]
    revealed_info_counts_by_seat: tuple[tuple[int, ...], ...]
    owned_objective_ids_by_seat: tuple[tuple[int, ...], ...]
    loan_principal_by_seat: tuple[int, ...]
    investment_value_by_seat: tuple[int, ...]
    legal_max_bid_by_seat: tuple[int, ...] | None
    visible_resource_ids: tuple[int, ...]
    resolved_turn_count: int
    seen_action_counts: tuple[int, ...]


def public_history_from_sdk_events(events: Sequence[object]) -> PublicHistory:
    """Convert canonical SDK sim events through the live-frame parser."""

    return public_history_from_sdk_frame(SimpleNamespace(common_events=tuple(events)))


def public_history_from_sdk_frame(frame: object) -> PublicHistory:
    """Convert one pinned-SDK raw decision frame into immutable public history."""

    common_events = _common_events(frame)
    setup = _parse_setup(common_events[0])
    history: list[PublicEvent] = [setup]
    tiebreak_seat = setup.initial_tiebreak_seat
    turn_open = False
    reveal_available = False

    for index, event in enumerate(common_events[1:], start=1):
        kind = _string_attribute(event, "kind", index)
        if kind == "turnOpened":
            history.append(_parse_turn_opened(event, index))
            turn_open = True
            reveal_available = False
        elif kind == "auctionResolved":
            if not turn_open:
                raise _compatibility_error(index, "auction resolved without an open turn")
            resolved = _parse_auction_resolved(event, index, setup.player_count)
            history.append(resolved)
            tiebreak_seat = _winning_seat(
                resolved.bids_by_seat,
                tiebreak_seat=tiebreak_seat,
            )
            turn_open = False
            reveal_available = True
        elif kind == "infoRevealed":
            if not reveal_available:
                raise _compatibility_error(index, "information revealed before auction resolution")
            suit_id = _integer_attribute(event, "suit_id", index)
            if not 1 <= suit_id <= len(Suit):
                raise _compatibility_error(index, "revealed suit ID is outside the known range")
            history.append(
                PublicInformationRevealed(
                    kind=PublicEventKind.INFORMATION_REVEALED,
                    seat=tiebreak_seat,
                    suit_id=suit_id,
                )
            )
            reveal_available = False
        else:
            raise _compatibility_error(index, f"unsupported public event kind {kind!r}")
    output = tuple(history)
    validate_public_history(output)
    return output


def validate_public_history(history: PublicHistory) -> ValidatedPublicHistory:
    """Replay public events and reject any state the pinned SDK rules cannot produce."""

    if type(history) is not tuple or not history or type(history[0]) is not PublicGameSetup:
        raise PublicHistoryCompatibilityError(
            "public history must be an exact nonempty tuple beginning with game setup"
        )
    setup = history[0]
    _require_event_kind(setup, PublicEventKind.GAME_SETUP, index=0)
    if not _is_integer(setup.player_count) or not 3 <= setup.player_count <= 5:
        raise _compatibility_error(0, "player count must be between three and five")
    if (
        not _is_integer(setup.starting_cash)
        or setup.starting_cash != STARTING_CASH[setup.player_count]
    ):
        raise _compatibility_error(0, "starting cash does not match the SDK player count")
    if not _is_integer_tuple(setup.value_chart) or setup.value_chart not in VALUE_CHARTS.values():
        raise _compatibility_error(0, "value chart must be one canonical SDK chart")
    if not _is_integer(setup.initial_tiebreak_seat) or not (
        0 <= setup.initial_tiebreak_seat < setup.player_count
    ):
        raise _compatibility_error(0, "initial tiebreak seat is outside player count")
    if (
        not _is_integer_tuple(setup.objective_ids)
        or len(set(setup.objective_ids)) != len(setup.objective_ids)
        or len(setup.objective_ids) not in (0, OBJECTIVES_PER_GAME)
        or any(objective_id not in OBJECTIVES for objective_id in setup.objective_ids)
    ):
        raise _compatibility_error(0, "objective IDs do not match a canonical SDK game")

    phase: PublicHistoryPhase = "ready_for_turn"
    latest_turn: PublicTurnOpened | None = None
    previous_resolved_turn: PublicTurnOpened | None = None
    tiebreak_seat = setup.initial_tiebreak_seat
    private_cards_per_player = INFO_CARDS_PER_PLAYER[setup.player_count]
    cash_by_seat = [setup.starting_cash] * setup.player_count
    won_resource_counts_by_seat = [[0] * _SUIT_COUNT for _seat in range(setup.player_count)]
    revealed_info_counts_by_seat = [[0] * _SUIT_COUNT for _seat in range(setup.player_count)]
    owned_objective_ids_by_seat: list[list[int]] = [[] for _seat in range(setup.player_count)]
    loan_principal_by_seat = [0] * setup.player_count
    investment_value_by_seat = [0] * setup.player_count
    seen_action_counts = [0] * len(ActionId)
    resolved_turn_count = 0
    for index, event in enumerate(history[1:], start=1):
        if type(event) is PublicTurnOpened:
            turn = event
            _require_event_kind(turn, PublicEventKind.TURN_OPENED, index=index)
            if phase != "ready_for_turn":
                raise _compatibility_error(index, "turn opened before the prior turn completed")
            if not _is_integer(turn.action_id):
                raise _compatibility_error(index, "action ID must be an integer")
            try:
                ActionId(turn.action_id)
            except ValueError as error:
                raise _compatibility_error(index, "action ID is unknown") from error
            if (
                not _is_integer_tuple(turn.resource_ids)
                or len(turn.resource_ids) != 2
                or any(not 0 <= resource_id <= _SUIT_COUNT for resource_id in turn.resource_ids)
                or (turn.resource_ids[0] == 0 and turn.resource_ids[1] != 0)
                or turn.resource_ids[0] == 0
            ):
                raise _compatibility_error(index, "turn resources are malformed")
            _validate_resource_carry(previous_resolved_turn, turn, index=index)
            seen_action_counts[turn.action_id - 1] += 1
            if seen_action_counts[turn.action_id - 1] > _ACTION_COUNT_BY_ID[turn.action_id]:
                raise _compatibility_error(index, "action appears more often than in the SDK deck")
            latest_turn = turn
            phase = "turn_open"
            _require_known_resources_fit_deck(
                won_resource_counts_by_seat,
                revealed_info_counts_by_seat,
                turn=turn,
                phase=phase,
                index=index,
            )
        elif type(event) is PublicAuctionResolved:
            resolved = event
            _require_event_kind(resolved, PublicEventKind.AUCTION_RESOLVED, index=index)
            if phase != "turn_open":
                raise _compatibility_error(index, "auction resolved without an open turn")
            if (
                not _is_integer_tuple(resolved.bids_by_seat)
                or len(resolved.bids_by_seat) != setup.player_count
                or any(bid < 0 for bid in resolved.bids_by_seat)
            ):
                raise _compatibility_error(index, "resolved bids are malformed")
            if latest_turn is None:
                raise AssertionError("turn-open phase always has a latest turn")
            bid_credit = _ACTION_CREDIT_BY_ID[latest_turn.action_id]
            if any(
                bid > cash_by_seat[seat] + bid_credit
                for seat, bid in enumerate(resolved.bids_by_seat)
            ):
                raise _compatibility_error(
                    index, "resolved bid exceeds its replayed SDK legal maximum"
                )
            tiebreak_seat = _winning_seat(
                resolved.bids_by_seat,
                tiebreak_seat=tiebreak_seat,
            )
            paid = max(resolved.bids_by_seat)
            cash_by_seat[tiebreak_seat] -= paid
            action_name = _ACTION_NAME_BY_ID[latest_turn.action_id]
            if action_name in LOAN_PRINCIPAL:
                principal = LOAN_PRINCIPAL[action_name]
                cash_by_seat[tiebreak_seat] += principal
                loan_principal_by_seat[tiebreak_seat] += principal
            if action_name in INVEST_PAYOUT:
                investment_value_by_seat[tiebreak_seat] += paid + INVEST_PAYOUT[action_name]

            granted_resource_count = (
                1
                if latest_turn.action_id == _AUCTION_ONE
                else 2
                if latest_turn.action_id == _AUCTION_TWO
                else 0
            )
            for suit_id in latest_turn.resource_ids[:granted_resource_count]:
                if suit_id:
                    won_resource_counts_by_seat[tiebreak_seat][suit_id - 1] += 1
            if granted_resource_count:
                for objective_id in setup.objective_ids:
                    if any(
                        objective_id in owned_objectives
                        for owned_objectives in owned_objective_ids_by_seat
                    ):
                        continue
                    if objective_pattern_met(
                        objective_id,
                        won_resource_counts_by_seat[tiebreak_seat],
                    ):
                        owned_objective_ids_by_seat[tiebreak_seat].append(objective_id)

            resolved_turn_count += 1
            previous_resolved_turn = latest_turn
            phase = (
                "reveal_pending"
                if sum(revealed_info_counts_by_seat[tiebreak_seat]) < private_cards_per_player
                else "ready_for_turn"
            )
            _require_known_resources_fit_deck(
                won_resource_counts_by_seat,
                revealed_info_counts_by_seat,
                turn=latest_turn,
                phase=phase,
                index=index,
            )
        elif type(event) is PublicInformationRevealed:
            revealed = event
            _require_event_kind(revealed, PublicEventKind.INFORMATION_REVEALED, index=index)
            if phase != "reveal_pending":
                raise _compatibility_error(index, "information revealed without a resolved auction")
            if not _is_integer(revealed.seat) or not 0 <= revealed.seat < setup.player_count:
                raise _compatibility_error(index, "information reveal seat is outside player count")
            if revealed.seat != tiebreak_seat:
                raise _compatibility_error(
                    index, "information reveal seat is not the auction winner"
                )
            if not _is_integer(revealed.suit_id) or not 1 <= revealed.suit_id <= _SUIT_COUNT:
                raise _compatibility_error(index, "revealed suit ID is outside the known range")
            revealed_info_counts_by_seat[revealed.seat][revealed.suit_id - 1] += 1
            if sum(revealed_info_counts_by_seat[revealed.seat]) > private_cards_per_player:
                raise _compatibility_error(index, "player reveals more than their initial hand")
            phase = "ready_for_turn"
            _require_known_resources_fit_deck(
                won_resource_counts_by_seat,
                revealed_info_counts_by_seat,
                turn=latest_turn,
                phase=phase,
                index=index,
            )
        else:
            raise _compatibility_error(index, "event class is not part of public history")
    legal_max_bid_by_seat = (
        tuple(cash + _ACTION_CREDIT_BY_ID[latest_turn.action_id] for cash in cash_by_seat)
        if phase == "turn_open" and latest_turn is not None
        else None
    )
    return ValidatedPublicHistory(
        setup=setup,
        phase=phase,
        latest_turn=latest_turn,
        tiebreak_seat=tiebreak_seat,
        cash_by_seat=tuple(cash_by_seat),
        won_resource_counts_by_seat=tuple(map(tuple, won_resource_counts_by_seat)),
        revealed_info_counts_by_seat=tuple(map(tuple, revealed_info_counts_by_seat)),
        owned_objective_ids_by_seat=tuple(map(tuple, owned_objective_ids_by_seat)),
        loan_principal_by_seat=tuple(loan_principal_by_seat),
        investment_value_by_seat=tuple(investment_value_by_seat),
        legal_max_bid_by_seat=legal_max_bid_by_seat,
        visible_resource_ids=_visible_resource_ids(latest_turn, phase),
        resolved_turn_count=resolved_turn_count,
        seen_action_counts=tuple(seen_action_counts),
    )


def _validate_resource_carry(
    previous: PublicTurnOpened | None,
    current: PublicTurnOpened,
    *,
    index: int,
) -> None:
    if previous is None:
        return
    if previous.action_id == _AUCTION_TWO and previous.resource_ids[1] == 0:
        raise _compatibility_error(
            index, "terminal one-card Auction2 cannot be followed by another turn"
        )
    if previous.action_id == _AUCTION_ONE and current.resource_ids[0] != previous.resource_ids[1]:
        raise _compatibility_error(index, "resources contradict the one-card auction carry")
    if previous.action_id not in (_AUCTION_ONE, _AUCTION_TWO) and (
        current.resource_ids != previous.resource_ids
    ):
        raise _compatibility_error(index, "resources change after a non-resource action")


def _require_known_resources_fit_deck(
    won_by_seat: list[list[int]],
    revealed_by_seat: list[list[int]],
    *,
    turn: PublicTurnOpened | None,
    phase: PublicHistoryPhase,
    index: int,
) -> None:
    visible_resource_ids = _visible_resource_ids(turn, phase)
    known_by_suit = tuple(
        sum(row[suit_index] for row in won_by_seat)
        + sum(row[suit_index] for row in revealed_by_seat)
        + visible_resource_ids.count(suit_index + 1)
        for suit_index in range(_SUIT_COUNT)
    )
    if any(
        known > available
        for known, available in zip(known_by_suit, _RESOURCE_COUNT_BY_SUIT, strict=True)
    ):
        raise _compatibility_error(index, "public resources exceed the finite SDK deck")


def _visible_resource_ids(
    turn: PublicTurnOpened | None,
    phase: PublicHistoryPhase,
) -> tuple[int, ...]:
    if turn is None:
        return ()
    if phase == "turn_open":
        return tuple(suit_id for suit_id in turn.resource_ids if suit_id)
    if turn.action_id == _AUCTION_ONE:
        return tuple(suit_id for suit_id in turn.resource_ids[1:] if suit_id)
    if turn.action_id == _AUCTION_TWO:
        return ()
    return tuple(suit_id for suit_id in turn.resource_ids if suit_id)


def _common_events(frame: object) -> tuple[object, ...]:
    try:
        value = getattr(frame, "common_events")  # noqa: B009
    except AttributeError as error:
        raise PublicHistoryCompatibilityError(
            "raw decision frame is missing common_events"
        ) from error
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PublicHistoryCompatibilityError("raw decision frame common_events must be a sequence")
    events = tuple(value)
    if not events:
        raise PublicHistoryCompatibilityError("raw decision frame common_events must not be empty")
    return events


def _parse_setup(event: object) -> PublicGameSetup:
    index = 0
    kind = _string_attribute(event, "kind", index)
    if kind != "gameSetup":
        raise _compatibility_error(index, "game setup must be the first public event")
    player_count = _integer_attribute(event, "player_count", index)
    if not 3 <= player_count <= 5:
        raise _compatibility_error(index, "player count must be between three and five")
    starting_cash = _integer_attribute(event, "starting_cash", index)
    if starting_cash <= 0:
        raise _compatibility_error(index, "starting cash must be positive")
    value_chart = _integer_tuple_attribute(event, "value_chart", index)
    if len(value_chart) != 6:
        raise _compatibility_error(index, "value chart must contain six entries")
    tiebreak_seat = _integer_attribute(event, "initial_tiebreak_seat", index)
    if not 0 <= tiebreak_seat < player_count:
        raise _compatibility_error(index, "initial tiebreak seat is outside player count")
    objective_ids = _integer_tuple_attribute(event, "objective_ids", index)
    if len(set(objective_ids)) != len(objective_ids):
        raise _compatibility_error(index, "objective IDs must be unique")
    if any(objective_id not in OBJECTIVES for objective_id in objective_ids):
        raise _compatibility_error(index, "objective ID is unknown")
    return PublicGameSetup(
        kind=PublicEventKind.GAME_SETUP,
        player_count=player_count,
        starting_cash=starting_cash,
        value_chart=value_chart,
        initial_tiebreak_seat=tiebreak_seat,
        objective_ids=objective_ids,
    )


def _parse_turn_opened(event: object, index: int) -> PublicTurnOpened:
    action_id = _integer_attribute(event, "action_id", index)
    try:
        ActionId(action_id)
    except ValueError as error:
        raise _compatibility_error(index, "action ID is unknown") from error
    resource_ids = _integer_tuple_attribute(event, "resource_ids", index)
    if len(resource_ids) != 2:
        raise _compatibility_error(index, "turn resources must contain two entries")
    if any(not 0 <= resource_id <= len(Suit) for resource_id in resource_ids):
        raise _compatibility_error(index, "turn resource ID is outside the known range")
    if resource_ids[0] == 0 and resource_ids[1] != 0:
        raise _compatibility_error(index, "turn resources must be zero-padded at the end")
    return PublicTurnOpened(
        kind=PublicEventKind.TURN_OPENED,
        action_id=action_id,
        resource_ids=(resource_ids[0], resource_ids[1]),
    )


def _parse_auction_resolved(
    event: object,
    index: int,
    player_count: int,
) -> PublicAuctionResolved:
    bids = _integer_tuple_attribute(event, "bids_by_seat", index)
    if len(bids) != player_count:
        raise _compatibility_error(index, "resolved bids must contain one bid per seat")
    if any(bid < 0 for bid in bids):
        raise _compatibility_error(index, "resolved bids must be nonnegative")
    return PublicAuctionResolved(
        kind=PublicEventKind.AUCTION_RESOLVED,
        bids_by_seat=bids,
    )


def _winning_seat(
    bids: tuple[int, ...],
    *,
    tiebreak_seat: int,
) -> int:
    highest = max(bids)
    for offset in range(1, len(bids) + 1):
        seat = (tiebreak_seat + offset) % len(bids)
        if bids[seat] == highest:
            return seat
    raise AssertionError("a nonempty bid tuple always has a winner")


def _string_attribute(event: object, name: str, index: int) -> str:
    value = _attribute(event, name, index)
    if not isinstance(value, str):
        raise _compatibility_error(index, f"{name} must be a string")
    return value


def _integer_attribute(event: object, name: str, index: int) -> int:
    value = _attribute(event, name, index)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _compatibility_error(index, f"{name} must be an integer")
    return value


def _integer_tuple_attribute(
    event: object,
    name: str,
    index: int,
) -> tuple[int, ...]:
    value = _attribute(event, name, index)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _compatibility_error(index, f"{name} must be a sequence")
    output: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise _compatibility_error(index, f"{name} must contain only integers")
        output.append(item)
    return tuple(output)


def _attribute(event: object, name: str, index: int) -> object:
    try:
        return getattr(event, name)
    except AttributeError as error:
        raise _compatibility_error(index, f"missing required field {name!r}") from error


def _require_event_kind(
    event: PublicEvent,
    expected: PublicEventKind,
    *,
    index: int,
) -> None:
    if event.kind is not expected:
        raise _compatibility_error(index, "event class and kind disagree")


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_integer_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(_is_integer(item) for item in value)


def _compatibility_error(index: int, message: str) -> PublicHistoryCompatibilityError:
    return PublicHistoryCompatibilityError(f"common_events[{index}]: {message}")
