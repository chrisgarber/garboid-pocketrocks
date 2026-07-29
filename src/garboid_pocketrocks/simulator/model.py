from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pocketrocks import ActionId, Suit

from garboid_pocketrocks.rules import Ruleset

Seat = int


class Phase(StrEnum):
    BIDDING = "bidding"
    REVEAL = "reveal"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ResourceCard:
    card_id: int
    suit: Suit


def offered_resource_ids(
    action_id: ActionId | None,
    resources: Sequence[ResourceCard],
) -> tuple[int, int]:
    if action_id not in (ActionId.AUCTION1, ActionId.AUCTION2):
        return (0, 0)
    count = 1 if action_id is ActionId.AUCTION1 else 2
    suit_ids = tuple(int(card.suit) for card in resources[:count])
    first = suit_ids[0] if suit_ids else 0
    second = suit_ids[1] if len(suit_ids) > 1 else 0
    return (first, second)


@dataclass(frozen=True, slots=True)
class ActionCard:
    card_id: int
    action_id: ActionId


@dataclass(frozen=True, slots=True)
class LoanPosition:
    principal: int
    winning_bid: int


@dataclass(frozen=True, slots=True)
class InvestmentPosition:
    locked: int
    payout: int


@dataclass(frozen=True, slots=True)
class PlayerState:
    seat: Seat
    cash: int
    private_hand: tuple[ResourceCard, ...] = ()
    revealed_info: tuple[ResourceCard, ...] = ()
    won_resources: tuple[ResourceCard, ...] = ()
    loans: tuple[LoanPosition, ...] = ()
    investments: tuple[InvestmentPosition, ...] = ()
    owned_objective_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    ruleset: Ruleset
    player_count: int
    seed: int
    turn_index: int
    phase: Phase
    players: tuple[PlayerState, ...]
    resource_deck: tuple[ResourceCard, ...]
    action_deck: tuple[ActionCard, ...]
    visible_resources: tuple[ResourceCard, ...]
    current_action: ActionCard | None
    active_objective_ids: tuple[int, ...]
    priority_seat: Seat
    reveal_seat: Seat | None = None
    current_resource_ids: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if len(self.current_resource_ids) != 2:
            raise ValueError("current resource IDs must contain exactly two entries")
        if any(not 0 <= resource_id <= len(Suit) for resource_id in self.current_resource_ids):
            raise ValueError("current resource IDs must be zero or a known suit ID")
        if self.current_resource_ids[0] == 0 and self.current_resource_ids[1] != 0:
            raise ValueError("current resource IDs must be zero-padded at the end")


@dataclass(frozen=True, slots=True)
class Score:
    seat: Seat
    final_money: int
    rank: int


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[Score, ...]
