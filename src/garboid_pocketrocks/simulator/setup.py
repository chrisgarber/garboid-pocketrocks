from __future__ import annotations

import random
from dataclasses import dataclass

from pocketrocks import ActionId, Suit

from garboid_pocketrocks.rules import Ruleset
from garboid_pocketrocks.simulator.events import EventKind, GameEvent
from garboid_pocketrocks.simulator.model import (
    ActionCard,
    GameState,
    Phase,
    PlayerState,
    ResourceCard,
    offered_resource_ids,
)


@dataclass(frozen=True, slots=True)
class SetupResult:
    state: GameState
    events: tuple[GameEvent, ...]


def build_setup(
    ruleset: Ruleset,
    *,
    player_count: int,
    seed: int,
) -> SetupResult:
    player_setup = ruleset.setup_for(player_count)
    resource_cards = [
        ResourceCard(card_id=card_id, suit=suit)
        for card_id, suit in enumerate(
            (
                suit
                for suit, count in zip(Suit, ruleset.resource_counts, strict=True)
                for _ in range(count)
            ),
            start=1,
        )
    ]
    action_cards = [
        ActionCard(card_id=card_id, action_id=action_id)
        for card_id, action_id in enumerate(
            (
                action_id
                for action_id, count in zip(ActionId, ruleset.action_counts, strict=True)
                for _ in range(count)
            ),
            start=1,
        )
    ]

    rng = random.Random(seed)
    rng.shuffle(resource_cards)
    rng.shuffle(action_cards)

    hands: list[list[ResourceCard]] = [[] for _ in range(player_count)]
    next_resource_index = 0
    for _ in range(player_setup.private_cards_per_player):
        for hand in hands:
            hand.append(resource_cards[next_resource_index])
            next_resource_index += 1

    visible_resources = tuple(resource_cards[next_resource_index : next_resource_index + 2])
    next_resource_index += len(visible_resources)
    resource_deck = tuple(resource_cards[next_resource_index:])

    objective_ids = list(ruleset.objective_pool)
    rng.shuffle(objective_ids)
    active_objective_ids = (
        tuple(objective_ids[: ruleset.active_objective_count]) if ruleset.objectives_enabled else ()
    )

    priority_seat = rng.randrange(player_count)
    current_action = action_cards[0]
    action_deck = tuple(action_cards[1:])
    players = tuple(
        PlayerState(
            seat=seat,
            cash=player_setup.starting_cash,
            private_hand=tuple(hand),
        )
        for seat, hand in enumerate(hands)
    )
    state = GameState(
        ruleset=ruleset,
        player_count=player_count,
        seed=seed,
        turn_index=0,
        phase=Phase.BIDDING,
        players=players,
        resource_deck=resource_deck,
        action_deck=action_deck,
        visible_resources=visible_resources,
        current_action=current_action,
        active_objective_ids=active_objective_ids,
        priority_seat=priority_seat,
        current_resource_ids=offered_resource_ids(
            current_action.action_id,
            visible_resources,
        ),
    )
    events = (
        GameEvent(
            EventKind.GAME_SETUP,
            turn_index=0,
            resource_ids=tuple(int(card.suit) for card in visible_resources),
            objective_ids=active_objective_ids,
        ),
        GameEvent(
            EventKind.TURN_OPENED,
            turn_index=0,
            action_id=current_action.action_id,
            resource_ids=tuple(int(card.suit) for card in visible_resources),
        ),
    )
    return SetupResult(state=state, events=events)
