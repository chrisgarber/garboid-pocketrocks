from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pocketrocks import ActionId, BotDecision, DecisionContext
from pocketrocks.sim.constants import VALUE_CHARTS

from garboid_pocketrocks.heuristics.profiles import (
    AGGRESSIVE_PROFILE,
    BALANCED_PROFILE,
    PASSIVE_PROFILE,
    HeuristicProfile,
)
from garboid_pocketrocks.heuristics.valuation import (
    BidEvaluation,
    HeuristicValuator,
)
from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.simulator.session import SdkGameSession

from .helpers import make_context, make_knowledge

PROFILES = (AGGRESSIVE_PROFILE, BALANCED_PROFILE, PASSIVE_PROFILE)


def _assert_finite_legal_evaluation(
    context: DecisionContext,
    result: BidEvaluation,
) -> None:
    assert context.legal_max_amount is not None
    assert result.points
    assert tuple(point.bid for point in result.points) == tuple(range(context.legal_max_amount + 1))
    assert 0 <= result.chosen_bid <= result.reservation_bid
    assert context.is_legal(BotDecision.submit_bid(result.chosen_bid))
    assert all(context.is_legal(BotDecision.submit_bid(point.bid)) for point in result.points)

    assert math.isfinite(result.belief.normalized_horizon)
    assert all(math.isfinite(count) for count in result.belief.expected_future_biddable_counts)
    for suit in result.belief.suits:
        assert math.isfinite(suit.expected_terminal_price)
        assert all(math.isfinite(probability) for probability in suit.terminal_price_pmf)
    for point in result.points:
        breakdown = point.breakdown
        assert all(
            math.isfinite(value)
            for value in (
                point.win_delta,
                breakdown.resource,
                breakdown.objective_completion,
                breakdown.objective_progress,
                breakdown.terminal_cash,
                breakdown.liquidity,
                breakdown.future_cash,
                breakdown.total,
            )
        )


def _play_and_check_every_bidding_context(
    *,
    chart: str,
    player_count: int,
    seed: int,
) -> None:
    knowledge = canonical_knowledge(player_count, value_chart=chart)
    session = SdkGameSession.start(
        player_count=player_count,
        seed=seed,
        value_chart=chart,
    )
    bidding_context_count = 0

    while not session.terminated:
        decisions: dict[int, BotDecision] = {}
        for seat, context in session.pending.contexts:
            if session.pending.decision_kind == "submitBid":
                bidding_context_count += 1
                for profile in PROFILES:
                    evaluator = HeuristicValuator(profile)
                    first = evaluator.evaluate_bid(context, knowledge)
                    second = evaluator.evaluate_bid(context, knowledge)
                    assert first == second
                    _assert_finite_legal_evaluation(context, first)
            decisions[seat] = BotDecision.pass_turn()
        session.step(decisions)

    assert bidding_context_count > 0


@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("chart", tuple(VALUE_CHARTS))
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_generated_live_games_have_finite_legal_deterministic_valuations(
    player_count: int,
    chart: str,
    seed: int,
) -> None:
    _play_and_check_every_bidding_context(
        chart=chart,
        player_count=player_count,
        seed=seed,
    )


@pytest.mark.parametrize("chart", tuple(VALUE_CHARTS))
@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("profile", PROFILES)
def test_moving_an_own_card_from_hand_to_public_reveal_preserves_resource_beliefs(
    chart: str,
    player_count: int,
    profile: HeuristicProfile,
) -> None:
    knowledge = canonical_knowledge(player_count, value_chart=chart)
    session = SdkGameSession.start(
        player_count=player_count,
        seed=player_count * 100 + ord(chart),
        value_chart=chart,
    )
    context = session.pending.contexts[0][1]
    revealed_suit = context.current_hand_suit_ids[0]
    revealed = [list(row) for row in context.revealed_info_counts_by_seat]
    revealed[context.bot_seat][revealed_suit - 1] += 1
    moved = replace(
        context,
        current_hand_suit_ids=context.current_hand_suit_ids[1:],
        revealed_info_counts_by_seat=tuple(tuple(row) for row in revealed),
        revealable_count=context.revealable_count - 1,
    )
    evaluator = HeuristicValuator(profile)

    original = evaluator.evaluate_bid(context, knowledge)
    transformed = evaluator.evaluate_bid(moved, knowledge)

    assert tuple(
        (
            suit.known_terminal_reveals,
            suit.unseen_suit_count,
            suit.terminal_price_pmf,
            suit.expected_terminal_price,
        )
        for suit in original.belief.suits
    ) == tuple(
        (
            suit.known_terminal_reveals,
            suit.unseen_suit_count,
            suit.terminal_price_pmf,
            suit.expected_terminal_price,
        )
        for suit in transformed.belief.suits
    )
    assert tuple(point.breakdown.resource for point in original.points) == tuple(
        point.breakdown.resource for point in transformed.points
    )


def _financial_context(*, action_id: ActionId, late: bool) -> DecisionContext:
    won = (
        (2, 2, 2, 2, 1) if late else (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    cash = (20, 20, 20)
    principal = 10 if action_id is ActionId.LOAN10 else 0
    return make_context(
        action_id=action_id,
        current_resources=(0, 0),
        cash=cash,
        won=won,
        legal_max=cash[0] + principal,
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_financial_value_moves_in_the_expected_horizon_direction(
    profile: HeuristicProfile,
) -> None:
    knowledge = make_knowledge()
    evaluator = HeuristicValuator(profile)
    early_loan = evaluator.evaluate_bid(
        _financial_context(action_id=ActionId.LOAN10, late=False),
        knowledge,
    )
    late_loan = evaluator.evaluate_bid(
        _financial_context(action_id=ActionId.LOAN10, late=True),
        knowledge,
    )
    early_investment = evaluator.evaluate_bid(
        _financial_context(action_id=ActionId.INVEST5, late=False),
        knowledge,
    )
    late_investment = evaluator.evaluate_bid(
        _financial_context(action_id=ActionId.INVEST5, late=True),
        knowledge,
    )

    assert early_loan.points[5].win_delta > late_loan.points[5].win_delta
    assert early_investment.points[5].win_delta < late_investment.points[5].win_delta


def _revealed_rows_for_first_suit_count(
    first_suit_count: int,
) -> tuple[tuple[int, ...], ...]:
    aggregate = [first_suit_count, 0, 0, 0, 0]
    remaining = 15 - first_suit_count
    for suit_index in range(1, 5):
        count = min(6, remaining)
        aggregate[suit_index] = count
        remaining -= count
    assert remaining == 0

    rows = [[0] * 5 for _ in range(3)]
    for seat in range(3):
        row_remaining = 5
        for suit_index in range(5):
            count = min(aggregate[suit_index], row_remaining)
            rows[seat][suit_index] = count
            aggregate[suit_index] -= count
            row_remaining -= count
        assert row_remaining == 0
    assert aggregate == [0, 0, 0, 0, 0]
    return tuple(tuple(row) for row in rows)


def _deterministic_first_suit_price(chart: str, reveal_count: int) -> float:
    revealed = _revealed_rows_for_first_suit_count(reveal_count)
    aggregate = tuple(sum(row[index] for row in revealed) for index in range(5))
    won_counts = tuple(
        6 - revealed_count - (1 if index == 0 else 0)
        for index, revealed_count in enumerate(aggregate)
    )
    context = make_context(
        current_resources=(1, 0),
        won=(won_counts, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        revealed=revealed,
        hand=(),
        value_chart=VALUE_CHARTS[chart],
    )
    knowledge = make_knowledge(
        private_cards=5,
        resource_counts=(6, 6, 6, 6, 6),
        value_chart=VALUE_CHARTS[chart],
    )
    result = HeuristicValuator(BALANCED_PROFILE).evaluate_bid(context, knowledge)
    return result.belief.suits[0].expected_terminal_price


@pytest.mark.parametrize("chart", ("B", "D"))
def test_decreasing_charts_are_not_treated_as_increasing(chart: str) -> None:
    assert _deterministic_first_suit_price(chart, 0) > (_deterministic_first_suit_price(chart, 5))


def test_nonmonotone_chart_preserves_its_peak_and_decline() -> None:
    assert _deterministic_first_suit_price("E", 3) > (_deterministic_first_suit_price("E", 5))
