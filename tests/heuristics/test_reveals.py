from __future__ import annotations

import math
from dataclasses import replace

import pytest
from pocketrocks import ActionId, DecisionContext, Suit

from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.profiles import BALANCED_PROFILE
from garboid_pocketrocks.heuristics.reveals import (
    build_observer_price_vector,
    choose_reveal,
)
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.rules import RulesetKnowledge

from .helpers import make_context, make_knowledge


def _reveal_context(
    *,
    hand: tuple[int, ...] = (int(Suit.BRICK), int(Suit.WOOD)),
    won: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    revealed: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    value_chart: tuple[int, ...] = (0, 4, 8, 12, 16, 20),
) -> DecisionContext:
    return make_context(
        decision_kind="selectInfoToReveal",
        action_id=ActionId.AUCTION1,
        current_resources=(int(Suit.BRICK), 0),
        hand=hand,
        won=won,
        revealed=revealed,
        legal_max=None,
        value_chart=value_chart,
    )


def test_empty_hand_cannot_choose_a_reveal() -> None:
    context = _reveal_context(hand=())

    with pytest.raises(HeuristicInputError, match="hand"):
        choose_reveal(context, make_knowledge(private_cards=0))


@pytest.mark.parametrize(
    "hand",
    (
        (int(Suit.BRICK),),
        (int(Suit.BRICK), int(Suit.WOOD)),
        (int(Suit.SHEEP), int(Suit.WOOD), int(Suit.ORE)),
    ),
)
def test_every_returned_index_is_legal(hand: tuple[int, ...]) -> None:
    context = _reveal_context(hand=hand)

    index = choose_reveal(
        context,
        make_knowledge(private_cards=len(hand), resource_counts=(4, 4, 4, 4, 4)),
    )

    assert 0 <= index < len(hand)


def test_repeated_suits_choose_their_lowest_hand_index() -> None:
    hand = (int(Suit.WOOD), int(Suit.WOOD))
    chart = (0, 2, 5, 9, 14, 20)
    context = _reveal_context(hand=hand, value_chart=chart)
    knowledge = make_knowledge(
        private_cards=len(hand),
        resource_counts=(5, 5, 5, 5, 5),
        value_chart=chart,
    )

    assert choose_reveal(context, knowledge) == 0


def test_constant_chart_tie_chooses_index_zero() -> None:
    chart = (7, 7, 7, 7, 7, 7)
    context = _reveal_context(
        hand=(int(Suit.ORE), int(Suit.BRICK)),
        value_chart=chart,
    )

    assert choose_reveal(
        context,
        make_knowledge(
            private_cards=2,
            resource_counts=(4, 4, 4, 4, 4),
            value_chart=chart,
        ),
    ) == 0


def test_influence_sums_price_changes_for_every_suit() -> None:
    hand = (int(Suit.BRICK), int(Suit.WOOD))
    won = (
        (0, 0, 0, 0, 0),
        (3, 4, 0, 0, 3),
        (0, 0, 0, 0, 0),
    )
    context = _reveal_context(
        hand=hand,
        won=won,
        value_chart=(0, 1, 4, 9, 16, 25),
    )
    knowledge = make_knowledge(
        private_cards=2,
        resource_counts=(6, 6, 6, 6, 6),
        value_chart=context.value_chart,
    )
    before = build_observer_price_vector(context, knowledge)
    influences: list[float] = []
    candidate_only_influences: list[float] = []
    opponent_counts = tuple(
        sum(row[suit_index] for seat, row in enumerate(won) if seat != context.bot_seat)
        for suit_index in range(len(Suit))
    )
    for suit_id in hand:
        suit = Suit(suit_id)
        after = build_observer_price_vector(
            context,
            knowledge,
            revealed_suit=suit,
        )
        influences.append(
            sum(
                count * (post - pre)
                for count, post, pre in zip(
                    opponent_counts,
                    after,
                    before,
                    strict=True,
                )
            )
        )
        candidate_only_influences.append(
            opponent_counts[suit_id - 1] * (after[suit_id - 1] - before[suit_id - 1])
        )

    assert min(range(len(hand)), key=lambda index: (influences[index], index)) != min(
        range(len(hand)),
        key=lambda index: (candidate_only_influences[index], index),
    )
    assert choose_reveal(context, knowledge) == min(
        range(len(hand)),
        key=lambda index: (influences[index], index),
    )


def test_valuator_exposes_the_same_reveal_policy() -> None:
    context = _reveal_context()
    knowledge = make_knowledge(
        private_cards=2,
        resource_counts=(4, 4, 4, 4, 4),
    )

    assert HeuristicValuator(BALANCED_PROFILE).choose_reveal(
        context,
        knowledge,
    ) == choose_reveal(context, knowledge)


def test_observer_price_vectors_are_finite() -> None:
    context = _reveal_context()
    knowledge = make_knowledge(
        private_cards=2,
        resource_counts=(4, 4, 4, 4, 4),
    )

    for candidate in (None, Suit(context.current_hand_suit_ids[0])):
        prices = build_observer_price_vector(
            context,
            knowledge,
            revealed_suit=candidate,
        )
        assert len(prices) == len(Suit)
        assert all(math.isfinite(price) for price in prices)


def test_observer_treats_actor_hand_as_hidden_until_candidate_reveal() -> None:
    chart = (0, 10, 20, 30, 40, 50)
    context = _reveal_context(
        hand=(int(Suit.BRICK),),
        value_chart=chart,
    )
    knowledge = make_knowledge(
        private_cards=1,
        resource_counts=(1, 1, 1, 1, 1),
        value_chart=chart,
    )

    before = build_observer_price_vector(context, knowledge)
    after = build_observer_price_vector(
        context,
        knowledge,
        revealed_suit=Suit.BRICK,
    )

    assert before == pytest.approx((6.0, 6.0, 6.0, 6.0, 6.0))
    assert after == pytest.approx((10.0, 5.0, 5.0, 5.0, 5.0))


@pytest.mark.parametrize(
    ("context", "knowledge", "message"),
    (
        (
            make_context(),
            make_knowledge(),
            "reveal",
        ),
        (
            _reveal_context(),
            replace(make_knowledge(private_cards=2), player_count=4),
            "player count",
        ),
        (
            _reveal_context(hand=(int(Suit.BRICK),)),
            make_knowledge(private_cards=1, resource_counts=(0, 2, 2, 2, 2)),
            "known card",
        ),
    ),
)
def test_invalid_public_contexts_raise_heuristic_input_error(
    context: DecisionContext,
    knowledge: RulesetKnowledge,
    message: str,
) -> None:
    with pytest.raises(HeuristicInputError, match=message):
        choose_reveal(context, knowledge)


def test_candidate_must_be_present_in_the_actor_hand() -> None:
    context = _reveal_context(hand=(int(Suit.BRICK),))

    with pytest.raises(HeuristicInputError, match="hand"):
        build_observer_price_vector(
            context,
            make_knowledge(private_cards=1),
            revealed_suit=Suit.WOOD,
        )
