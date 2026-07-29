from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class Score:
    seat: Seat
    final_money: int
    rank: int


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[Score, ...]
