from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Literal

from pocketrocks import OBJECTIVES, ActionId, Suit


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
    return tuple(history)


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


def _compatibility_error(index: int, message: str) -> PublicHistoryCompatibilityError:
    return PublicHistoryCompatibilityError(f"common_events[{index}]: {message}")
