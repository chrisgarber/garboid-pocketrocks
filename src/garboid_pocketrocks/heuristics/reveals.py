from __future__ import annotations

import math

from pocketrocks import DecisionContext, Suit

from garboid_pocketrocks.heuristics.belief import build_belief
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.knowledge import RulesetKnowledge

_CHART_BUCKET_COUNT = 6
_SUIT_COUNT = len(Suit)


def _probability(
    *,
    population: int,
    successes: int,
    draws: int,
    selected: int,
) -> float:
    if selected < 0 or selected > successes or selected > draws:
        return 0.0
    failures = population - successes
    unselected = draws - selected
    if unselected > failures:
        return 0.0
    return (
        math.comb(successes, selected)
        * math.comb(failures, unselected)
        / math.comb(population, draws)
    )


def _expected_price(
    *,
    known_reveals: int,
    unseen_suit_count: int,
    unseen_population: int,
    hidden_slots: int,
    value_chart: tuple[int, ...],
) -> float:
    expected = 0.0
    for selected in range(hidden_slots + 1):
        probability = _probability(
            population=unseen_population,
            successes=unseen_suit_count,
            draws=hidden_slots,
            selected=selected,
        )
        expected += (
            probability * value_chart[min(known_reveals + selected, _CHART_BUCKET_COUNT - 1)]
        )
    if not math.isfinite(expected):
        raise HeuristicInputError("observer expected price must be finite")
    return expected


def build_observer_price_vector(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    *,
    revealed_suit: Suit | None = None,
) -> tuple[float, ...]:
    """Return expected prices for an observer who cannot see the actor's hand."""

    if context.decision_kind != "selectInfoToReveal":
        raise HeuristicInputError("observer prices require a reveal decision context")

    # Reuse the canonical public-context and conservation validation. Its returned
    # posterior knows our hand, so the observer posterior below is built separately.
    build_belief(context, ruleset)
    if not context.current_hand_suit_ids:
        raise HeuristicInputError("reveal decision requires a nonempty hand")
    if revealed_suit is not None and int(revealed_suit) not in context.current_hand_suit_ids:
        raise HeuristicInputError("candidate reveal suit is not present in the hand")

    try:
        revealed_by_suit = tuple(
            sum(row[index] for row in context.revealed_info_counts_by_seat)
            for index in range(_SUIT_COUNT)
        )
        won_by_suit = tuple(
            sum(row[index] for row in context.won_resource_counts_by_seat)
            for index in range(_SUIT_COUNT)
        )
        unseen_by_suit = [
            total - won - revealed
            for total, won, revealed in zip(
                ruleset.resource_counts,
                won_by_suit,
                revealed_by_suit,
                strict=True,
            )
        ]
        hidden_slots = sum(
            ruleset.private_cards_per_player - sum(row)
            for row in context.revealed_info_counts_by_seat
        )
        known_reveals = list(revealed_by_suit)

        if revealed_suit is not None:
            candidate_index = int(revealed_suit) - 1
            unseen_by_suit[candidate_index] -= 1
            hidden_slots -= 1
            known_reveals[candidate_index] += 1

        unseen_population = sum(unseen_by_suit)
        if any(count < 0 for count in unseen_by_suit):
            raise HeuristicInputError("known card counts exceed ruleset resources")
        if hidden_slots < 0 or hidden_slots > unseen_population:
            raise HeuristicInputError(
                "hidden private slots contradict the unseen resource population"
            )

        total_biddable = sum(ruleset.resource_counts) - (
            context.player_count * ruleset.private_cards_per_player
        )
        won_count = sum(won_by_suit)
        if unseen_population - hidden_slots != total_biddable - won_count:
            raise HeuristicInputError(
                "public card counts violate observer finite-population conservation"
            )

        prices = tuple(
            _expected_price(
                known_reveals=known,
                unseen_suit_count=unseen,
                unseen_population=unseen_population,
                hidden_slots=hidden_slots,
                value_chart=context.value_chart,
            )
            for known, unseen in zip(
                known_reveals,
                unseen_by_suit,
                strict=True,
            )
        )
    except HeuristicInputError:
        raise
    except (IndexError, OverflowError, TypeError, ValueError, ZeroDivisionError) as error:
        raise HeuristicInputError(str(error)) from error

    if len(prices) != _SUIT_COUNT or not all(math.isfinite(price) for price in prices):
        raise HeuristicInputError("observer price vector must be finite and complete")
    return prices


def choose_reveal(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
) -> int:
    """Choose the legal hand index exposing the least value to opponents."""

    before = build_observer_price_vector(context, ruleset)
    first_index_by_suit: dict[Suit, int] = {}
    for index, suit_id in enumerate(context.current_hand_suit_ids):
        first_index_by_suit.setdefault(Suit(suit_id), index)

    opponent_won_by_suit = tuple(
        sum(
            row[index]
            for seat, row in enumerate(context.won_resource_counts_by_seat)
            if seat != context.bot_seat
        )
        for index in range(_SUIT_COUNT)
    )
    candidates: list[tuple[float, int]] = []
    for suit, first_index in first_index_by_suit.items():
        after = build_observer_price_vector(
            context,
            ruleset,
            revealed_suit=suit,
        )
        influence = sum(
            held * (post_price - pre_price)
            for held, post_price, pre_price in zip(
                opponent_won_by_suit,
                after,
                before,
                strict=True,
            )
        )
        if not math.isfinite(influence):
            raise HeuristicInputError("reveal influence must be finite")
        candidates.append((influence, first_index))

    if not candidates:
        raise HeuristicInputError("reveal decision requires a nonempty hand")
    chosen_index = min(candidates)[1]
    if not 0 <= chosen_index < len(context.current_hand_suit_ids):
        raise HeuristicInputError("reveal policy produced an illegal hand index")
    return chosen_index
