from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict, fields, replace
from pathlib import Path

import pytest
from pocketrocks import ActionId

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.game_phase import game_phase_for_turn_index
from garboid_pocketrocks.heuristics.opponent_bids import (
    LegalBidWinningForecast,
    OpponentBidDistribution,
    OpponentBidForecast,
    OpponentBidModelConfig,
    PublicOpponentBidContext,
    PublicResolvedBidRound,
    forecast_opponent_bids,
    resolved_bid_rounds_from_public_history,
)

CHART = (0, 4, 8, 12, 16, 20)


def _setup(
    *,
    player_count: int = 3,
    starting_cash: int = 30,
    value_chart: tuple[int, ...] = CHART,
    objective_ids: tuple[int, ...] = (1, 2, 3, 4),
    initial_tiebreak_seat: int = 0,
) -> PublicGameSetup:
    return PublicGameSetup(
        kind=PublicEventKind.GAME_SETUP,
        player_count=player_count,
        starting_cash=starting_cash,
        value_chart=value_chart,
        initial_tiebreak_seat=initial_tiebreak_seat,
        objective_ids=objective_ids,
    )


def _turn(
    action: ActionId = ActionId.AUCTION1,
    resources: tuple[int, int] | None = None,
) -> PublicTurnOpened:
    if resources is None:
        resources = (1, 0) if action is ActionId.AUCTION1 else (1, 2)
        if action not in (ActionId.AUCTION1, ActionId.AUCTION2):
            resources = (0, 0)
    return PublicTurnOpened(
        kind=PublicEventKind.TURN_OPENED,
        action_id=int(action),
        resource_ids=resources,
    )


def _resolution(bids: tuple[int, ...]) -> PublicAuctionResolved:
    return PublicAuctionResolved(
        kind=PublicEventKind.AUCTION_RESOLVED,
        bids_by_seat=bids,
    )


def _history(
    rounds: tuple[tuple[ActionId, tuple[int, ...]], ...] = (),
    *,
    current_action: ActionId = ActionId.AUCTION1,
    player_count: int = 3,
    starting_cash: int = 30,
    value_chart: tuple[int, ...] = CHART,
    objective_ids: tuple[int, ...] = (1, 2, 3, 4),
    initial_tiebreak_seat: int = 0,
) -> PublicHistory:
    events: list[PublicEvent] = [
        _setup(
            player_count=player_count,
            starting_cash=starting_cash,
            value_chart=value_chart,
            objective_ids=objective_ids,
            initial_tiebreak_seat=initial_tiebreak_seat,
        )
    ]
    for action, bids in rounds:
        events.extend((_turn(action), _resolution(bids)))
    events.append(_turn(current_action))
    return tuple(events)


def _context(
    *,
    player_count: int = 3,
    starting_cash: int = 30,
    value_chart: tuple[int, ...] = CHART,
    action: ActionId = ActionId.AUCTION1,
    cash: tuple[int, ...] = (10, 10, 10),
    tiebreak_seat: int = 0,
    bot_seat: int = 0,
    legal_max: int = 10,
    completed_rounds: int = 0,
) -> PublicOpponentBidContext:
    return PublicOpponentBidContext(
        player_count=player_count,
        starting_cash=starting_cash,
        value_chart=value_chart,
        current_action_id=int(action),
        cash_by_seat=cash,
        tiebreak_seat=tiebreak_seat,
        bot_seat=bot_seat,
        legal_max_amount=legal_max,
        game_phase=game_phase_for_turn_index(completed_rounds),
    )


def _distribution(forecast: OpponentBidForecast, seat: int) -> OpponentBidDistribution:
    return next(item for item in forecast.opponent_distributions if item.opponent_seat == seat)


def _tiebreak_after(
    rounds: tuple[tuple[ActionId, tuple[int, ...]], ...],
    initial_tiebreak_seat: int = 0,
) -> int:
    tiebreak_seat = initial_tiebreak_seat
    for _action, bids in rounds:
        highest_bid = max(bids)
        for offset in range(1, len(bids) + 1):
            seat = (tiebreak_seat + offset) % len(bids)
            if bids[seat] == highest_bid:
                tiebreak_seat = seat
                break
    return tiebreak_seat


def test_public_types_are_closed_frozen_slotted_allowlists() -> None:
    assert {field.name for field in fields(PublicOpponentBidContext)} == {
        "player_count",
        "starting_cash",
        "value_chart",
        "current_action_id",
        "cash_by_seat",
        "tiebreak_seat",
        "bot_seat",
        "legal_max_amount",
        "game_phase",
    }
    assert {field.name for field in fields(PublicResolvedBidRound)} == {
        "turn_index",
        "game_phase",
        "action_id",
        "resource_ids",
        "bids_by_seat",
    }
    assert {field.name for field in fields(OpponentBidDistribution)} == {
        "opponent_seat",
        "legal_max_amount",
        "probabilities_by_amount",
        "prior_only",
        "history_round_count",
        "effective_history_weight",
    }
    assert {field.name for field in fields(LegalBidWinningForecast)} == {
        "effective_bid",
        "win_probability",
    }
    assert {field.name for field in fields(OpponentBidForecast)} == {
        "opponent_distributions",
        "legal_bid_forecasts",
    }
    assert {field.name for field in fields(OpponentBidModelConfig)} == {
        "prior_strength",
        "minimum_history_rounds",
        "same_action_phase_weight",
        "partial_match_weight",
        "fallback_weight",
    }
    context = _context()
    with pytest.raises(FrozenInstanceError):
        context.bot_seat = 2  # type: ignore[misc]
    assert hasattr(PublicOpponentBidContext, "__slots__")


def test_forecast_is_deterministically_equal_and_primitive_serializable() -> None:
    rounds = (
        (ActionId.AUCTION1, (2, 4, 7)),
        (ActionId.LOAN10, (1, 8, 3)),
    )
    context = _context(tiebreak_seat=_tiebreak_after(rounds), completed_rounds=2)
    history = _history(rounds)

    first = forecast_opponent_bids(history, context)
    second = forecast_opponent_bids(history, context)

    assert first == second
    assert json.dumps(asdict(first), sort_keys=True) == json.dumps(asdict(second), sort_keys=True)


def test_absent_and_one_round_history_use_exactly_the_same_prior() -> None:
    absent = forecast_opponent_bids(_history(), _context())
    rounds = ((ActionId.AUCTION1, (10, 10, 10)),)
    one_round = forecast_opponent_bids(
        _history(rounds),
        _context(tiebreak_seat=_tiebreak_after(rounds), completed_rounds=1),
    )

    assert tuple(item.probabilities_by_amount for item in absent.opponent_distributions) == tuple(
        item.probabilities_by_amount for item in one_round.opponent_distributions
    )


def test_sparse_prior_and_history_strength_are_auditable() -> None:
    one = ((ActionId.AUCTION1, (1, 2, 3)),)
    one_round = forecast_opponent_bids(
        _history(one),
        _context(tiebreak_seat=_tiebreak_after(one), completed_rounds=1),
    )
    two = (
        (ActionId.AUCTION1, (1, 2, 3)),
        (ActionId.LOAN10, (3, 2, 1)),
    )
    two_rounds = forecast_opponent_bids(
        _history(two),
        _context(tiebreak_seat=_tiebreak_after(two), completed_rounds=2),
    )

    sparse = _distribution(one_round, 1)
    smoothed = _distribution(two_rounds, 1)
    assert sparse.prior_only is True
    assert sparse.history_round_count == 1
    assert sparse.effective_history_weight == 0.0
    assert smoothed.prior_only is False
    assert smoothed.history_round_count == 2
    assert smoothed.effective_history_weight > 0.0


def test_prior_uses_absolute_chart_gaps_and_chart_shape() -> None:
    ascending = forecast_opponent_bids(_history(), _context())
    descending_chart = (20, 16, 12, 8, 4, 0)
    descending = forecast_opponent_bids(
        _history(value_chart=descending_chart),
        _context(value_chart=descending_chart),
    )
    nonmonotone_chart = (0, 10, 9, 19, 18, 28)
    nonmonotone = forecast_opponent_bids(
        _history(value_chart=nonmonotone_chart),
        _context(value_chart=nonmonotone_chart),
    )

    assert ascending.opponent_distributions == descending.opponent_distributions
    assert ascending.opponent_distributions != nonmonotone.opponent_distributions


def test_prior_is_positive_with_extra_triangular_weight_at_reference() -> None:
    distribution = _distribution(forecast_opponent_bids(_history(), _context()), 1)

    assert distribution.legal_max_amount == 10
    assert all(probability > 0.0 for probability in distribution.probabilities_by_amount)
    assert distribution.probabilities_by_amount[3] == max(distribution.probabilities_by_amount)


def test_prior_changes_with_action_player_pressure_and_phase() -> None:
    prior_only = OpponentBidModelConfig(minimum_history_rounds=99)
    auction_one = forecast_opponent_bids(_history(), _context(), prior_only)
    auction_two = forecast_opponent_bids(
        _history(current_action=ActionId.AUCTION2),
        _context(action=ActionId.AUCTION2),
        prior_only,
    )

    pressure_chart = (0, 8, 16, 24, 32, 40)
    four_cash = (10, 10, 10, 10)
    three_player_pressure = forecast_opponent_bids(
        _history(value_chart=pressure_chart),
        _context(value_chart=pressure_chart),
        prior_only,
    )
    four_player = forecast_opponent_bids(
        _history(player_count=4, value_chart=pressure_chart),
        _context(player_count=4, cash=four_cash, value_chart=pressure_chart),
        prior_only,
    )
    early_three = _distribution(three_player_pressure, 1).probabilities_by_amount
    early_four = _distribution(four_player, 1).probabilities_by_amount

    middle_rounds = tuple((ActionId.LOAN10, (0, 0, 0)) for _ in range(5))
    middle = forecast_opponent_bids(
        _history(middle_rounds),
        _context(tiebreak_seat=_tiebreak_after(middle_rounds), completed_rounds=5),
        prior_only,
    )

    assert auction_one.opponent_distributions != auction_two.opponent_distributions
    assert early_three != early_four
    assert _distribution(auction_one, 1) != _distribution(middle, 1)


def test_history_weights_same_action_and_phase_more_than_partial_match() -> None:
    prefix = tuple((ActionId.LOAN10, (0, 0, 0)) for _ in range(5))
    same = prefix + ((ActionId.AUCTION1, (0, 7, 0)),)
    partial = prefix + ((ActionId.AUCTION2, (0, 7, 0)),)
    context = _context(tiebreak_seat=_tiebreak_after(same), completed_rounds=6)

    same_probability = _distribution(
        forecast_opponent_bids(_history(same), context), 1
    ).probabilities_by_amount[7]
    partial_probability = _distribution(
        forecast_opponent_bids(_history(partial), context), 1
    ).probabilities_by_amount[7]

    assert same_probability > partial_probability


def test_history_weights_partial_match_more_than_fallback() -> None:
    tail = tuple((ActionId.LOAN10, (0, 0, 0)) for _ in range(5))
    partial = ((ActionId.AUCTION1, (0, 7, 0)),) + tail
    fallback = ((ActionId.AUCTION2, (0, 7, 0)),) + tail
    context = _context(completed_rounds=6)

    partial_probability = _distribution(
        forecast_opponent_bids(_history(partial), context), 1
    ).probabilities_by_amount[7]
    fallback_probability = _distribution(
        forecast_opponent_bids(_history(fallback), context), 1
    ).probabilities_by_amount[7]

    assert partial_probability > fallback_probability


def test_each_opponent_learns_a_separate_distribution() -> None:
    rounds = (
        (ActionId.AUCTION1, (0, 1, 8)),
        (ActionId.AUCTION1, (0, 1, 8)),
    )
    forecast = forecast_opponent_bids(
        _history(rounds),
        _context(tiebreak_seat=_tiebreak_after(rounds), completed_rounds=2),
    )

    seat_one = _distribution(forecast, 1).probabilities_by_amount
    seat_two = _distribution(forecast, 2).probabilities_by_amount
    assert seat_one != seat_two
    assert seat_one[1] > seat_one[8]
    assert seat_two[8] > seat_two[1]


def test_cash_and_public_credit_define_each_opponent_support() -> None:
    loan_context = _context(
        action=ActionId.LOAN10,
        cash=(10, 3, 7),
        legal_max=20,
    )
    loan = forecast_opponent_bids(
        _history(current_action=ActionId.LOAN10),
        loan_context,
    )
    no_credit = forecast_opponent_bids(
        _history(),
        _context(cash=(10, 3, 7), legal_max=10),
    )

    assert len(_distribution(loan, 1).probabilities_by_amount) == 14
    assert len(_distribution(loan, 2).probabilities_by_amount) == 18
    assert len(_distribution(no_credit, 1).probabilities_by_amount) == 4
    assert len(_distribution(no_credit, 2).probabilities_by_amount) == 8


def test_old_bids_are_clipped_only_to_current_legal_support() -> None:
    rounds = (
        (ActionId.AUCTION1, (0, 99, 1)),
        (ActionId.AUCTION1, (0, 99, 1)),
    )
    forecast = forecast_opponent_bids(
        _history(rounds),
        _context(
            cash=(10, 3, 10),
            tiebreak_seat=_tiebreak_after(rounds),
            completed_rounds=2,
        ),
    )
    clipped = _distribution(forecast, 1).probabilities_by_amount

    assert len(clipped) == 4
    assert clipped[3] == max(clipped)


def test_every_distribution_is_positive_normalized_and_finite() -> None:
    forecast = forecast_opponent_bids(
        _history(((ActionId.AUCTION1, (3, 4, 5)), (ActionId.LOAN20, (8, 7, 6)))),
        _context(completed_rounds=2),
    )

    for distribution in forecast.opponent_distributions:
        assert math.isclose(sum(distribution.probabilities_by_amount), 1.0, abs_tol=1e-12)
        assert all(
            probability > 0.0 and math.isfinite(probability)
            for probability in distribution.probabilities_by_amount
        )
    assert all(
        math.isfinite(item.win_probability) and 0.0 <= item.win_probability <= 1.0
        for item in forecast.legal_bid_forecasts
    )


def test_bid_zero_can_win_when_bot_is_first_in_tiebreak_order() -> None:
    first = forecast_opponent_bids(
        _history(),
        _context(bot_seat=1, tiebreak_seat=0),
    )
    last = forecast_opponent_bids(
        _history(),
        _context(bot_seat=0, tiebreak_seat=0),
    )

    assert first.legal_bid_forecasts[0].effective_bid == 0
    assert first.legal_bid_forecasts[0].win_probability > 0.0
    assert last.legal_bid_forecasts[0].win_probability == 0.0


def test_ties_lose_to_opponents_ahead_and_beat_opponents_behind() -> None:
    first = forecast_opponent_bids(
        _history(initial_tiebreak_seat=2),
        _context(tiebreak_seat=2),
    )
    last = forecast_opponent_bids(_history(), _context(tiebreak_seat=0))

    assert first.opponent_distributions == last.opponent_distributions
    assert (
        first.legal_bid_forecasts[5].win_probability > last.legal_bid_forecasts[5].win_probability
    )
    assert first.legal_bid_forecasts[-1].win_probability == 1.0


def test_replayed_tiebreak_evolution_changes_zero_and_tie_probabilities() -> None:
    bot_first_rounds = ((ActionId.AUCTION1, (5, 1, 0)),)
    bot_last_rounds = ((ActionId.AUCTION1, (0, 5, 1)),)
    bot_first = forecast_opponent_bids(
        _history(bot_first_rounds),
        _context(bot_seat=1, tiebreak_seat=0, completed_rounds=1),
    )
    bot_last = forecast_opponent_bids(
        _history(bot_last_rounds),
        _context(bot_seat=1, tiebreak_seat=1, completed_rounds=1),
    )

    assert tuple(
        distribution.probabilities_by_amount for distribution in bot_first.opponent_distributions
    ) == tuple(
        distribution.probabilities_by_amount for distribution in bot_last.opponent_distributions
    )
    assert bot_first.legal_bid_forecasts[0].win_probability > 0.0
    assert bot_last.legal_bid_forecasts[0].win_probability == 0.0
    assert (
        bot_first.legal_bid_forecasts[5].win_probability
        > bot_last.legal_bid_forecasts[5].win_probability
    )


def test_wrong_current_tiebreak_fails_closed() -> None:
    rounds = ((ActionId.AUCTION1, (0, 5, 1)),)

    with pytest.raises(HeuristicInputError, match="tiebreak evolution"):
        forecast_opponent_bids(
            _history(rounds),
            _context(tiebreak_seat=0, completed_rounds=1),
        )


def test_reveal_seat_must_equal_the_replayed_auction_winner() -> None:
    history = (
        _setup(),
        _turn(),
        _resolution((0, 5, 1)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 2, 3),
        _turn(),
    )

    with pytest.raises(HeuristicInputError, match="auction winner"):
        forecast_opponent_bids(history, _context(tiebreak_seat=1, completed_rounds=1))


@pytest.mark.parametrize(
    "history",
    (
        (),
        (_turn(),),
        (_setup(), _resolution((1, 2, 3)), _turn()),
        (_setup(), _turn(), _turn()),
        (_setup(), _turn(), _resolution((1, 2)), _turn()),
        (_setup(), _turn(), _resolution((1, -1, 2)), _turn()),
        (
            _setup(),
            _turn(),
            _resolution((1, 2, 3)),
            PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 0, 1),
            PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 0, 2),
            _turn(),
        ),
    ),
)
def test_malformed_public_history_fails_closed(history: PublicHistory) -> None:
    with pytest.raises(HeuristicInputError):
        forecast_opponent_bids(history, _context())


@pytest.mark.parametrize(
    "context",
    (
        _context(player_count=4, cash=(10, 10, 10, 10)),
        _context(starting_cash=31),
        _context(value_chart=(0, 1, 2, 3, 4, 5)),
        _context(action=ActionId.LOAN10, legal_max=20),
        replace(_context(), game_phase="middle"),
    ),
)
def test_setup_and_current_turn_contradictions_fail_closed(
    context: PublicOpponentBidContext,
) -> None:
    with pytest.raises(HeuristicInputError):
        forecast_opponent_bids(_history(), context)


def test_parser_returns_completed_rounds_and_ignores_valid_reveal() -> None:
    history = (
        _setup(),
        _turn(ActionId.AUCTION2),
        _resolution((2, 4, 6)),
        PublicInformationRevealed(PublicEventKind.INFORMATION_REVEALED, 2, 3),
        _turn(ActionId.LOAN10),
    )
    context = _context(
        action=ActionId.LOAN10,
        legal_max=20,
        tiebreak_seat=2,
        completed_rounds=1,
    )

    assert resolved_bid_rounds_from_public_history(history, context) == (
        PublicResolvedBidRound(
            turn_index=0,
            game_phase="early",
            action_id=int(ActionId.AUCTION2),
            resource_ids=(1, 2),
            bids_by_seat=(2, 4, 6),
        ),
    )


def test_setup_extra_ids_do_not_affect_the_model_and_private_fields_are_not_inputs() -> None:
    context = _context()
    first = forecast_opponent_bids(_history(objective_ids=(1, 2)), context)
    changed = forecast_opponent_bids(_history(objective_ids=(99, 100, 101)), context)
    source = Path(forecast_opponent_bids.__code__.co_filename).read_text(encoding="utf-8")

    assert first == changed
    assert {field.name for field in fields(PublicOpponentBidContext)}.isdisjoint(
        {"current_hand_suit_ids", "request_id", "metadata", "seed", "deck_order"}
    )
    for forbidden_import in (
        "pocketrocks.internal",
        "garboid_pocketrocks.simulator",
        "heuristics.objectives",
    ):
        assert forbidden_import not in source


@pytest.mark.parametrize(
    "kwargs",
    (
        {"prior_strength": 0.0},
        {"prior_strength": math.inf},
        {"minimum_history_rounds": 1},
        {"same_action_phase_weight": 1.0},
        {"partial_match_weight": 1.0},
        {"fallback_weight": 0.0},
    ),
)
def test_model_configuration_is_validated(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        OpponentBidModelConfig(**kwargs)  # type: ignore[arg-type]


def test_public_probability_records_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        OpponentBidDistribution(1, 1, (0.2, 0.2), True, 0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        LegalBidWinningForecast(0, math.nan)


@pytest.mark.parametrize(
    ("action", "credit"),
    (
        (ActionId.AUCTION1, 0),
        (ActionId.AUCTION2, 0),
        (ActionId.LOAN10, 10),
        (ActionId.LOAN20, 20),
        (ActionId.INVEST5, 0),
        (ActionId.INVEST10, 0),
    ),
)
def test_action_specific_public_credit_is_exact(action: ActionId, credit: int) -> None:
    assert _context(action=action, legal_max=10 + credit).legal_max_amount == 10 + credit
    with pytest.raises(ValueError, match="action credit"):
        _context(action=action, legal_max=11 + credit)
