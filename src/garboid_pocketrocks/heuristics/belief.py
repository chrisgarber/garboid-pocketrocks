from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pocketrocks import ActionId, DecisionContext, Suit

from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.rules import RulesetKnowledge

_SUIT_COUNT = len(Suit)
_CHART_BUCKET_COUNT = 6
_ACTION_COUNT = len(ActionId)
_AUCTIONS = (ActionId.AUCTION1, ActionId.AUCTION2)


@dataclass(frozen=True, slots=True)
class SuitBelief:
    """Marginal terminal-price belief for one resource suit."""

    suit: Suit
    known_terminal_reveals: int
    unseen_suit_count: int
    unseen_population: int
    opponent_hidden_slots: int
    terminal_price_pmf: tuple[float, ...]
    expected_terminal_price: float


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Public-information belief about prices and remaining auctions."""

    suits: tuple[SuitBelief, ...]
    expected_future_biddable_counts: tuple[float, ...]
    normalized_horizon: float


def _hypergeometric_probability(
    population: int,
    successes: int,
    draws: int,
    selected: int,
) -> float:
    if population < 0 or successes < 0 or successes > population or draws < 0 or draws > population:
        raise ValueError("invalid hypergeometric population")
    if (
        selected < 0
        or selected > successes
        or selected > draws
        or draws - selected > population - successes
    ):
        return 0.0
    return (
        math.comb(successes, selected)
        * math.comb(population - successes, draws - selected)
        / math.comb(population, draws)
    )


def _require_length(name: str, values: Sequence[object], expected: int) -> None:
    if len(values) != expected:
        raise HeuristicInputError(f"{name} must contain {expected} entries")


def _require_nonnegative_counts(name: str, values: Iterable[int]) -> None:
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HeuristicInputError(f"{name} values must be nonnegative integers")


def _require_integer_values(name: str, values: Iterable[int]) -> None:
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise HeuristicInputError(f"{name} values must be integers")


def _validate_knowledge(knowledge: RulesetKnowledge) -> None:
    _require_length("ruleset resource counts", knowledge.resource_counts, _SUIT_COUNT)
    _require_nonnegative_counts("ruleset resource count", knowledge.resource_counts)
    _require_length("ruleset action counts", knowledge.action_counts, _ACTION_COUNT)
    _require_nonnegative_counts("ruleset action count", knowledge.action_counts)
    _require_length("ruleset value chart", knowledge.value_chart, _CHART_BUCKET_COUNT)
    _require_integer_values("ruleset value chart", knowledge.value_chart)
    if not 3 <= knowledge.player_count <= 5:
        raise HeuristicInputError("ruleset player count must be between three and five")
    if knowledge.starting_cash <= 0:
        raise HeuristicInputError("ruleset starting cash must be positive")
    if knowledge.private_cards_per_player < 0:
        raise HeuristicInputError("ruleset private card count must be nonnegative")
    if len(set(knowledge.objective_pool)) != len(knowledge.objective_pool):
        raise HeuristicInputError("ruleset objective pool contains duplicate IDs")
    if not 0 <= knowledge.active_objective_count <= len(knowledge.objective_pool):
        raise HeuristicInputError("ruleset active objective count exceeds its objective pool")
    if not knowledge.objectives_enabled and knowledge.active_objective_count != 0:
        raise HeuristicInputError("disabled objectives require zero active objectives")


def _validate_matrix(
    name: str,
    matrix: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    columns: int | None,
) -> None:
    _require_length(f"{name} matrix", matrix, rows)
    if columns is not None:
        for row in matrix:
            _require_length(f"{name} matrix row", row, columns)
    _require_nonnegative_counts(name, (value for row in matrix for value in row))


def _action_from_context(context: DecisionContext, knowledge: RulesetKnowledge) -> ActionId:
    if context.current_action_id is None:
        raise HeuristicInputError("current action ID is required")
    try:
        action = ActionId(context.current_action_id)
    except (TypeError, ValueError) as error:
        raise HeuristicInputError("current action ID is unknown") from error
    if knowledge.action_counts[int(action) - 1] == 0:
        raise HeuristicInputError("current action contradicts ruleset action counts")
    return action


def _validate_context(
    context: DecisionContext,
    knowledge: RulesetKnowledge,
) -> ActionId:
    if context.player_count != knowledge.player_count:
        raise HeuristicInputError("context player count contradicts ruleset knowledge")
    if context.starting_cash != knowledge.starting_cash:
        raise HeuristicInputError("context starting cash contradicts ruleset knowledge")
    _require_length("context value chart", context.value_chart, _CHART_BUCKET_COUNT)
    _require_integer_values("context value chart", context.value_chart)
    if context.value_chart != knowledge.value_chart:
        raise HeuristicInputError("context value chart contradicts ruleset knowledge")
    if context.decision_kind not in ("submitBid", "selectInfoToReveal"):
        raise HeuristicInputError("decision kind is unsupported")

    player_count = context.player_count
    _require_length("cash by seat", context.cash_by_seat, player_count)
    _require_nonnegative_counts("cash", context.cash_by_seat)
    _validate_matrix(
        "won resource",
        context.won_resource_counts_by_seat,
        rows=player_count,
        columns=_SUIT_COUNT,
    )
    _validate_matrix(
        "revealed information count",
        context.revealed_info_counts_by_seat,
        rows=player_count,
        columns=_SUIT_COUNT,
    )
    _validate_matrix(
        "owned objective",
        context.owned_objective_ids_by_seat,
        rows=player_count,
        columns=None,
    )
    if not 0 <= context.bot_seat < player_count:
        raise HeuristicInputError("bot seat is outside player count")
    if not 0 <= context.tiebreak_seat < player_count:
        raise HeuristicInputError("tiebreak seat is outside player count")

    if len(set(context.objective_ids)) != len(context.objective_ids):
        raise HeuristicInputError("active objective IDs must be unique")
    if len(context.objective_ids) != knowledge.active_objective_count:
        raise HeuristicInputError("active objective count contradicts ruleset knowledge")
    if not set(context.objective_ids) <= set(knowledge.objective_pool):
        raise HeuristicInputError("active objective ID is outside the ruleset pool")
    active_objectives = set(context.objective_ids)
    owned_objectives = [
        objective_id for row in context.owned_objective_ids_by_seat for objective_id in row
    ]
    if not set(owned_objectives) <= active_objectives:
        raise HeuristicInputError("owned objective ID is not active")
    if len(owned_objectives) != len(set(owned_objectives)):
        raise HeuristicInputError("owned objective IDs must be unique across players")

    _require_length("current resource IDs", context.current_resource_ids, 2)
    for resource_id in context.current_resource_ids:
        if not isinstance(resource_id, int) or not 0 <= resource_id <= _SUIT_COUNT:
            raise HeuristicInputError("current resource ID is outside the suit range")
    if context.current_resource_ids[0] == 0 and context.current_resource_ids[1] != 0:
        raise HeuristicInputError("current resource IDs cannot have a missing first card")
    for suit_id in context.current_hand_suit_ids:
        if not isinstance(suit_id, int) or not 1 <= suit_id <= _SUIT_COUNT:
            raise HeuristicInputError("hand suit ID is outside the suit range")
    if context.revealable_count != len(context.current_hand_suit_ids):
        raise HeuristicInputError("revealable count contradicts current hand")

    private_cards = knowledge.private_cards_per_player
    revealed_totals = tuple(sum(row) for row in context.revealed_info_counts_by_seat)
    if any(revealed > private_cards for revealed in revealed_totals):
        raise HeuristicInputError("revealed private card count exceeds ruleset setup")
    if revealed_totals[context.bot_seat] + len(context.current_hand_suit_ids) != private_cards:
        raise HeuristicInputError("own private card count contradicts ruleset setup")

    action = _action_from_context(context, knowledge)
    offered_count = sum(resource_id != 0 for resource_id in context.current_resource_ids)
    if action not in _AUCTIONS and offered_count:
        raise HeuristicInputError("offered resource is invalid for a financial action")
    maximum_offer = 1 if action is ActionId.AUCTION1 else 2
    if action in _AUCTIONS and offered_count > maximum_offer:
        raise HeuristicInputError("offered resource count exceeds auction capacity")
    if context.decision_kind == "submitBid":
        if context.legal_max_amount is None or context.legal_max_amount < 0:
            raise HeuristicInputError("bid request requires a nonnegative legal maximum")
        if action in _AUCTIONS and offered_count == 0:
            raise HeuristicInputError("auction bid request requires an offered resource")
    elif context.legal_max_amount is not None:
        raise HeuristicInputError("reveal request cannot include a legal bid maximum")
    return action


def _offered_counts(
    context: DecisionContext,
    action: ActionId,
) -> tuple[int, ...]:
    counts = [0] * _SUIT_COUNT
    if context.decision_kind == "submitBid" and action in _AUCTIONS:
        for resource_id in context.current_resource_ids:
            if resource_id:
                counts[resource_id - 1] += 1
    return tuple(counts)


def _terminal_price_pmf(
    *,
    known_reveals: int,
    unseen_population: int,
    unseen_suit_count: int,
    opponent_hidden_slots: int,
) -> tuple[float, ...]:
    buckets = [0.0] * _CHART_BUCKET_COUNT
    for selected in range(opponent_hidden_slots + 1):
        probability = _hypergeometric_probability(
            unseen_population,
            unseen_suit_count,
            opponent_hidden_slots,
            selected,
        )
        buckets[min(known_reveals + selected, _CHART_BUCKET_COUNT - 1)] += probability
    total_probability = sum(buckets)
    if total_probability <= 0:
        raise HeuristicInputError("terminal price distribution has no valid outcomes")
    return tuple(probability / total_probability for probability in buckets)


def build_belief(
    context: DecisionContext,
    knowledge: RulesetKnowledge,
) -> BeliefState:
    """Build exact public-information resource beliefs for an SDK decision."""

    _validate_knowledge(knowledge)
    action = _validate_context(context, knowledge)

    revealed_by_suit = tuple(
        sum(row[index] for row in context.revealed_info_counts_by_seat)
        for index in range(_SUIT_COUNT)
    )
    hand_by_suit = tuple(
        sum(suit_id == int(suit) for suit_id in context.current_hand_suit_ids) for suit in Suit
    )
    known_terminal_reveals = tuple(
        revealed + hand for revealed, hand in zip(revealed_by_suit, hand_by_suit, strict=True)
    )
    won_by_suit = tuple(
        sum(row[index] for row in context.won_resource_counts_by_seat)
        for index in range(_SUIT_COUNT)
    )
    offered_by_suit = _offered_counts(context, action)
    unseen_by_suit = tuple(
        total - terminal - won - offered
        for total, terminal, won, offered in zip(
            knowledge.resource_counts,
            known_terminal_reveals,
            won_by_suit,
            offered_by_suit,
            strict=True,
        )
    )
    if any(unseen < 0 for unseen in unseen_by_suit):
        raise HeuristicInputError("known card counts exceed ruleset resources")

    opponent_hidden_slots = sum(
        knowledge.private_cards_per_player - sum(context.revealed_info_counts_by_seat[seat])
        for seat in range(context.player_count)
        if seat != context.bot_seat
    )
    unseen_population = sum(unseen_by_suit)
    total_biddable = sum(knowledge.resource_counts) - (
        context.player_count * knowledge.private_cards_per_player
    )
    if total_biddable <= 0:
        raise HeuristicInputError("ruleset setup must leave a biddable resource")
    won_count = sum(won_by_suit)
    offered_count = sum(offered_by_suit)
    future_biddable = total_biddable - won_count - offered_count
    if future_biddable < 0:
        raise HeuristicInputError("known won and offered cards exceed biddable resources")
    if unseen_population < opponent_hidden_slots:
        raise HeuristicInputError("opponent hidden slots exceed the unseen population")
    if unseen_population - opponent_hidden_slots != future_biddable:
        raise HeuristicInputError("public card counts violate finite-population conservation")

    if unseen_population:
        expected_future_biddable_counts = tuple(
            unseen * future_biddable / unseen_population for unseen in unseen_by_suit
        )
    else:
        expected_future_biddable_counts = (0.0,) * _SUIT_COUNT

    suits: list[SuitBelief] = []
    for suit, known, unseen in zip(
        Suit,
        known_terminal_reveals,
        unseen_by_suit,
        strict=True,
    ):
        terminal_price_pmf = _terminal_price_pmf(
            known_reveals=known,
            unseen_population=unseen_population,
            unseen_suit_count=unseen,
            opponent_hidden_slots=opponent_hidden_slots,
        )
        try:
            expected_terminal_price = sum(
                probability * price
                for probability, price in zip(
                    terminal_price_pmf,
                    context.value_chart,
                    strict=True,
                )
            )
        except OverflowError as error:
            raise HeuristicInputError("expected terminal price is not finite") from error
        if not math.isfinite(expected_terminal_price):
            raise HeuristicInputError("expected terminal price is not finite")
        suits.append(
            SuitBelief(
                suit=suit,
                known_terminal_reveals=known,
                unseen_suit_count=unseen,
                unseen_population=unseen_population,
                opponent_hidden_slots=opponent_hidden_slots,
                terminal_price_pmf=terminal_price_pmf,
                expected_terminal_price=expected_terminal_price,
            )
        )

    return BeliefState(
        suits=tuple(suits),
        expected_future_biddable_counts=expected_future_biddable_counts,
        normalized_horizon=future_biddable / total_biddable,
    )
