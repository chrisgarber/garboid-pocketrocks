from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.sim.constants import VALUE_CHARTS

from garboid_pocketrocks.heuristics import belief as belief_module
from garboid_pocketrocks.heuristics.belief import build_belief
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.knowledge import RulesetKnowledge, canonical_knowledge
from garboid_pocketrocks.simulator.session import SdkGameSession

from .helpers import make_context, make_knowledge


def test_no_hidden_private_cards_uses_deterministic_chart_bucket() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        value_chart=(7, 7, 7, 7, 7, 7),
    )
    belief = build_belief(
        context,
        make_knowledge(value_chart=context.value_chart),
    )
    brick = belief.suits[int(Suit.BRICK) - 1]
    assert brick.terminal_price_pmf == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert brick.expected_terminal_price == 7.0


def test_exact_hypergeometric_pmf_conditions_on_known_hand_and_offer() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.WOOD),),
    )
    belief = build_belief(context, make_knowledge(private_cards=1))

    brick = belief.suits[int(Suit.BRICK) - 1]
    wood = belief.suits[int(Suit.WOOD) - 1]
    ore = belief.suits[int(Suit.ORE) - 1]
    assert brick.known_terminal_reveals == 0
    assert brick.unseen_suit_count == 1
    assert brick.unseen_population == 8
    assert brick.opponent_hidden_slots == 2
    assert brick.terminal_price_pmf == pytest.approx((0.75, 0.25, 0, 0, 0, 0))
    assert wood.known_terminal_reveals == 1
    assert wood.terminal_price_pmf == pytest.approx((0, 0.75, 0.25, 0, 0, 0))
    assert ore.terminal_price_pmf == pytest.approx((15 / 28, 12 / 28, 1 / 28, 0, 0, 0))


@pytest.mark.parametrize(("chart_name", "chart"), tuple(VALUE_CHARTS.items()))
def test_expected_terminal_prices_follow_each_chart_exactly(
    chart_name: str,
    chart: tuple[int, ...],
) -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.WOOD),),
        value_chart=chart,
    )
    belief = build_belief(
        context,
        make_knowledge(private_cards=1, value_chart=chart),
    )

    brick = belief.suits[int(Suit.BRICK) - 1]
    wood = belief.suits[int(Suit.WOOD) - 1]
    assert brick.expected_terminal_price == pytest.approx(0.75 * chart[0] + 0.25 * chart[1]), (
        chart_name
    )
    assert wood.expected_terminal_price == pytest.approx(0.75 * chart[1] + 0.25 * chart[2]), (
        chart_name
    )


def test_terminal_reveal_probability_is_capped_in_chart_bucket_five() -> None:
    context = make_context(
        current_resources=(int(Suit.WOOD), 0),
        hand=(int(Suit.BRICK),) * 3,
        revealed=((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    belief = build_belief(
        context,
        make_knowledge(private_cards=3, resource_counts=(10, 2, 2, 2, 2)),
    )

    brick = belief.suits[int(Suit.BRICK) - 1]
    assert brick.known_terminal_reveals == 4
    assert brick.terminal_price_pmf[4] > 0
    assert brick.terminal_price_pmf[5] > 0
    assert brick.terminal_price_pmf[:4] == (0.0, 0.0, 0.0, 0.0)
    assert sum(brick.terminal_price_pmf) == pytest.approx(1.0)


def test_expected_future_biddable_counts_and_horizon_conserve_cards() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.WOOD),),
    )
    belief = build_belief(context, make_knowledge(private_cards=1))

    assert belief.expected_future_biddable_counts == pytest.approx((0.75, 0.75, 1.5, 1.5, 1.5))
    assert sum(belief.expected_future_biddable_counts) == pytest.approx(6.0)
    assert belief.normalized_horizon == pytest.approx(6 / 7)


def test_public_card_accounting_is_the_conserved_posterior_boundary() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.WOOD),),
    )
    knowledge = make_knowledge(private_cards=1)

    accounting = belief_module._account_public_cards(
        context,
        knowledge,
        ActionId.AUCTION1,
    )

    assert accounting == belief_module._PublicCardAccounting(
        known_terminal_reveals=(0, 1, 0, 0, 0),
        unseen_by_suit=(1, 1, 2, 2, 2),
        known_future_by_suit=(0, 0, 0, 0, 0),
        opponent_hidden_slots=2,
        unseen_population=8,
        unknown_future_biddable=6,
        future_biddable=6,
        total_biddable=7,
    )


def test_two_card_auction_subtracts_both_offered_cards() -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(int(Suit.BRICK), int(Suit.BRICK)),
    )
    belief = build_belief(context, make_knowledge())

    brick = belief.suits[int(Suit.BRICK) - 1]
    assert brick.unseen_suit_count == 0
    assert sum(suit.unseen_suit_count for suit in belief.suits) == 8
    assert sum(belief.expected_future_biddable_counts) == pytest.approx(8.0)
    assert belief.normalized_horizon == pytest.approx(0.8)


def test_reveal_context_does_not_subtract_preserved_offer_twice() -> None:
    won = ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    context = make_context(
        decision_kind="selectInfoToReveal",
        current_resources=(1, 0),
        won=won,
        hand=(2,),
        legal_max=None,
    )
    belief = build_belief(context, make_knowledge(private_cards=1))

    assert sum(suit.unseen_suit_count for suit in belief.suits) == 8
    assert (
        sum(suit.unseen_suit_count for suit in belief.suits) - belief.suits[0].opponent_hidden_slots
        == 6
    )
    assert sum(belief.expected_future_biddable_counts) == pytest.approx(6.0)
    assert belief.normalized_horizon == pytest.approx(6 / 7)


def test_reveal_context_ignores_the_preserved_offer_identity() -> None:
    won = ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
    context = make_context(
        decision_kind="selectInfoToReveal",
        current_resources=(int(Suit.BRICK), 0),
        won=won,
        hand=(int(Suit.WOOD),),
        legal_max=None,
    )

    with_preserved_offer = build_belief(context, make_knowledge(private_cards=1))
    without_preserved_offer = build_belief(
        replace(context, current_resource_ids=(0, 0)),
        make_knowledge(private_cards=1),
    )
    assert with_preserved_offer == without_preserved_offer


def test_financial_bid_ignores_visible_board_resources() -> None:
    context = make_context(
        action_id=ActionId.INVEST10,
        current_resources=(int(Suit.WOOD), int(Suit.ORE)),
    )
    belief = build_belief(context, make_knowledge())

    assert tuple(suit.unseen_suit_count for suit in belief.suits) == (2, 1, 1, 2, 2)
    assert belief.expected_future_biddable_counts == (2.0, 2.0, 2.0, 2.0, 2.0)
    assert belief.normalized_horizon == 1.0


def test_one_card_auction_ignores_second_visible_resource() -> None:
    context = make_context(
        action_id=ActionId.AUCTION1,
        current_resources=(int(Suit.ORE), int(Suit.WOOD)),
    )
    belief = build_belief(context, make_knowledge())

    assert tuple(suit.unseen_suit_count for suit in belief.suits) == (2, 1, 1, 2, 2)
    assert belief.expected_future_biddable_counts == (2.0, 2.0, 1.0, 2.0, 2.0)
    assert belief.normalized_horizon == 0.9


def test_moving_own_hand_card_to_revealed_information_preserves_belief() -> None:
    context = make_context(
        current_resources=(int(Suit.BRICK), 0),
        hand=(int(Suit.WOOD),),
    )
    revealed = [list(row) for row in context.revealed_info_counts_by_seat]
    revealed[context.bot_seat][int(Suit.WOOD) - 1] += 1
    after_reveal = replace(
        context,
        revealed_info_counts_by_seat=tuple(tuple(row) for row in revealed),
        current_hand_suit_ids=(),
        revealable_count=0,
    )
    knowledge = make_knowledge(private_cards=1)

    assert build_belief(context, knowledge) == build_belief(after_reveal, knowledge)


def test_inconsistent_known_cards_are_rejected() -> None:
    context = make_context(
        won=((3, 0, 0, 0, 0),) + ((0, 0, 0, 0, 0),) * 2,
    )
    with pytest.raises(HeuristicInputError, match="known card"):
        build_belief(context, make_knowledge(resource_counts=(2, 2, 2, 2, 2)))


@pytest.mark.parametrize(
    ("context", "knowledge", "message"),
    (
        (
            make_context(player_count=3),
            make_knowledge(player_count=4),
            "player count",
        ),
        (
            make_context(starting_cash=30),
            make_knowledge(starting_cash=25),
            "starting cash",
        ),
        (
            make_context(value_chart=(0, 1, 2, 3, 4, 5)),
            make_knowledge(value_chart=(5, 4, 3, 2, 1, 0)),
            "value chart",
        ),
        (
            make_context(cash=(30, 30)),
            make_knowledge(),
            "cash",
        ),
        (
            make_context(won=((0, 0, 0, 0),) * 3),
            make_knowledge(),
            "won resource",
        ),
        (
            make_context(revealed=((0, 0, 0, 0, -1),) * 3),
            make_knowledge(),
            "revealed information count",
        ),
        (
            make_context(owned_objectives=((), ())),
            make_knowledge(),
            "owned objective",
        ),
        (
            make_context(bot_seat=3),
            make_knowledge(),
            "bot seat",
        ),
        (
            make_context(hand=(0,)),
            make_knowledge(private_cards=1),
            "hand suit",
        ),
        (
            make_context(hand=()),
            make_knowledge(private_cards=1),
            "private card",
        ),
        (
            make_context(
                hand=(1,),
                revealed=((0, 0, 0, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
            ),
            make_knowledge(private_cards=1),
            "private card",
        ),
    ),
)
def test_invalid_context_or_ruleset_shape_is_rejected(
    context: DecisionContext,
    knowledge: RulesetKnowledge,
    message: str,
) -> None:
    with pytest.raises(HeuristicInputError, match=message):
        build_belief(context, knowledge)


def test_invalid_ruleset_widths_are_rejected() -> None:
    malformed = replace(
        make_knowledge(),
        resource_counts=(2, 2, 2, 2),
        action_counts=(12, 8, 3),
        value_chart=(0, 4),
    )
    with pytest.raises(HeuristicInputError, match="resource counts"):
        build_belief(make_context(), malformed)


@pytest.mark.parametrize("invalid_entry", (1.5, True))
def test_value_chart_entries_must_be_integers(invalid_entry: object) -> None:
    chart = cast(tuple[int, ...], (invalid_entry, 4, 8, 12, 16, 20))
    with pytest.raises(HeuristicInputError, match="value chart.*integers"):
        build_belief(
            make_context(value_chart=chart),
            make_knowledge(value_chart=chart),
        )


def test_overflowing_integer_chart_is_rejected_as_heuristic_input() -> None:
    huge_price = 10**400
    chart = (huge_price,) * 6
    with pytest.raises(HeuristicInputError, match="expected terminal price"):
        build_belief(
            make_context(value_chart=chart),
            make_knowledge(value_chart=chart),
        )


def test_invalid_current_resource_shape_is_rejected() -> None:
    context = replace(
        make_context(),
        current_resource_ids=cast(tuple[int, int], (int(Suit.BRICK),)),
    )
    with pytest.raises(HeuristicInputError, match="current resource"):
        build_belief(context, make_knowledge())


def test_constant_chart_stays_constant() -> None:
    context = make_context(value_chart=(9, 9, 9, 9, 9, 9))
    belief = build_belief(
        context,
        make_knowledge(private_cards=0, value_chart=context.value_chart),
    )
    assert all(suit.expected_terminal_price == 9.0 for suit in belief.suits)


@pytest.mark.parametrize("player_count", (3, 4, 5))
@settings(max_examples=25, deadline=None)
@given(
    chart_name=st.sampled_from(tuple(VALUE_CHARTS)),
    game_seed=st.integers(min_value=0, max_value=2**32 - 1),
    decision_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_engine_generated_contexts_preserve_exact_belief_properties(
    player_count: int,
    chart_name: str,
    game_seed: int,
    decision_seed: int,
) -> None:
    knowledge = canonical_knowledge(player_count, value_chart=chart_name)
    session = SdkGameSession.start(
        player_count=player_count,
        seed=game_seed,
        value_chart=chart_name,
    )
    decision_rng = random.Random(decision_seed)
    total_biddable = sum(knowledge.resource_counts) - (
        player_count * knowledge.private_cards_per_player
    )

    while not session.terminated:
        for _seat, context in session.pending.contexts:
            belief = build_belief(context, knowledge)
            won_count = sum(count for row in context.won_resource_counts_by_seat for count in row)
            if context.decision_kind == "submitBid" and context.current_action_id == int(
                ActionId.AUCTION1
            ):
                offered_count = int(context.current_resource_ids[0] != 0)
            elif context.decision_kind == "submitBid" and context.current_action_id == int(
                ActionId.AUCTION2
            ):
                offered_count = sum(
                    resource_id != 0 for resource_id in context.current_resource_ids
                )
            else:
                offered_count = 0
            visible_count = (
                sum(resource_id != 0 for resource_id in context.current_resource_ids)
                if context.decision_kind == "submitBid"
                else 0
            )
            known_future_count = visible_count - offered_count
            future_biddable = total_biddable - won_count - offered_count
            hidden_slots = sum(
                knowledge.private_cards_per_player - sum(context.revealed_info_counts_by_seat[seat])
                for seat in range(player_count)
                if seat != context.bot_seat
            )

            assert tuple(suit.suit for suit in belief.suits) == tuple(Suit)
            assert len(belief.suits) == len(Suit)
            assert all(suit.opponent_hidden_slots == hidden_slots for suit in belief.suits)
            assert (
                sum(suit.unseen_suit_count for suit in belief.suits)
                - hidden_slots
                + known_future_count
                == future_biddable
            )
            assert sum(belief.expected_future_biddable_counts) == pytest.approx(future_biddable)
            assert belief.normalized_horizon == pytest.approx(future_biddable / total_biddable)
            for suit in belief.suits:
                assert len(suit.terminal_price_pmf) == 6
                assert sum(suit.terminal_price_pmf) == pytest.approx(1.0)
                assert all(0.0 <= probability <= 1.0 for probability in suit.terminal_price_pmf)
                assert 0 <= suit.unseen_suit_count <= suit.unseen_population
                assert (
                    min(context.value_chart)
                    <= suit.expected_terminal_price
                    <= max(context.value_chart)
                )
                expected = sum(
                    probability * price
                    for probability, price in zip(
                        suit.terminal_price_pmf,
                        context.value_chart,
                        strict=True,
                    )
                )
                assert suit.expected_terminal_price == pytest.approx(expected)
                assert math.isfinite(suit.expected_terminal_price)

            if context.current_hand_suit_ids:
                revealed = [list(row) for row in context.revealed_info_counts_by_seat]
                revealed_suit = context.current_hand_suit_ids[0]
                revealed[context.bot_seat][revealed_suit - 1] += 1
                after_reveal = replace(
                    context,
                    revealed_info_counts_by_seat=tuple(tuple(row) for row in revealed),
                    current_hand_suit_ids=context.current_hand_suit_ids[1:],
                    revealable_count=context.revealable_count - 1,
                )
                assert build_belief(after_reveal, knowledge) == belief

        decisions: dict[int, BotDecision] = {}
        for seat, context in session.pending.contexts:
            if context.decision_kind == "submitBid":
                assert context.legal_max_amount is not None
                decisions[seat] = BotDecision.submit_bid(
                    decision_rng.randint(0, context.legal_max_amount)
                )
            elif context.revealable_count:
                decisions[seat] = BotDecision.select_info_to_reveal(
                    decision_rng.randrange(context.revealable_count)
                )
            else:
                decisions[seat] = BotDecision.pass_turn()
        session.step(decisions)
