from __future__ import annotations

from dataclasses import dataclass

from pocketrocks import OBJECTIVES, DecisionContext
from pocketrocks.sim.constants import (
    ACTION_DECK,
    ACTION_WIRE_IDS,
    INFO_CARDS_PER_PLAYER,
    ITEM_DECK_SUITS,
    OBJECTIVES_PER_GAME,
    STARTING_CASH,
    VALUE_CHARTS,
)


@dataclass(frozen=True, slots=True)
class RulesetKnowledge:
    """Public rule metadata used by strategies without owning game rules."""

    name: str
    player_count: int
    starting_cash: int
    private_cards_per_player: int
    resource_counts: tuple[int, ...]
    action_counts: tuple[int, ...]
    value_chart: tuple[int, ...]
    objective_pool: tuple[int, ...]
    active_objective_count: int
    objectives_enabled: bool


def _variant_name(value_chart: str, objectives_enabled: bool) -> str:
    suffix = "" if objectives_enabled else "-no-objectives"
    return f"live-{value_chart}{suffix}"


def _resource_counts() -> tuple[int, ...]:
    return tuple(ITEM_DECK_SUITS.count(suit) for suit in range(1, 6))


def _action_counts() -> tuple[int, ...]:
    actions_by_id = tuple(
        action for action, _wire_id in sorted(ACTION_WIRE_IDS.items(), key=lambda item: item[1])
    )
    return tuple(ACTION_DECK.count(action) for action in actions_by_id)


def canonical_knowledge(
    player_count: int,
    *,
    value_chart: str = "A",
    objectives_enabled: bool = True,
) -> RulesetKnowledge:
    """Return immutable strategy metadata for one SDK-supported variant."""

    try:
        starting_cash = STARTING_CASH[player_count]
        private_cards = INFO_CARDS_PER_PLAYER[player_count]
    except KeyError as error:
        raise ValueError("PocketRocks supports 3-5 players") from error
    chart = value_chart.upper()
    try:
        chart_values = VALUE_CHARTS[chart]
    except KeyError as error:
        raise ValueError(f"unknown value chart {value_chart!r} (expected A-E)") from error
    return RulesetKnowledge(
        name=_variant_name(chart, objectives_enabled),
        player_count=player_count,
        starting_cash=starting_cash,
        private_cards_per_player=private_cards,
        resource_counts=_resource_counts(),
        action_counts=_action_counts(),
        value_chart=chart_values,
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=OBJECTIVES_PER_GAME if objectives_enabled else 0,
        objectives_enabled=objectives_enabled,
    )


def knowledge_for_context(context: DecisionContext) -> RulesetKnowledge:
    """Derive strategy metadata entirely from public SDK context fields."""

    try:
        chart = next(name for name, values in VALUE_CHARTS.items() if values == context.value_chart)
    except StopIteration as error:
        raise ValueError("SDK context contains an unknown value chart") from error
    knowledge = canonical_knowledge(
        context.player_count,
        value_chart=chart,
        objectives_enabled=bool(context.objective_ids),
    )
    private_cards = knowledge.private_cards_per_player
    if 0 <= context.bot_seat < len(context.revealed_info_counts_by_seat):
        private_cards = sum(context.revealed_info_counts_by_seat[context.bot_seat]) + len(
            context.current_hand_suit_ids
        )
    return RulesetKnowledge(
        name=knowledge.name,
        player_count=context.player_count,
        starting_cash=context.starting_cash,
        private_cards_per_player=private_cards,
        resource_counts=knowledge.resource_counts,
        action_counts=knowledge.action_counts,
        value_chart=context.value_chart,
        objective_pool=knowledge.objective_pool,
        active_objective_count=len(context.objective_ids),
        objectives_enabled=bool(context.objective_ids),
    )
