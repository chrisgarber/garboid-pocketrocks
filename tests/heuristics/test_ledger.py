"""Public-history factors: ledger reconstruction, standings, denial, deck pressure."""

from __future__ import annotations

import pytest
from pocketrocks import ActionId, Suit
from pocketrocks.internal.bot_wire_v2 import (
    AuctionResolvedEvent,
    GameSetupEvent,
    InfoRevealedEvent,
    TurnOpenedEvent,
)

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    PublicInformationRevealed,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.heuristics.ledger import (
    _winning_seat,
    auction_pressure,
    deficit_to_leader,
    denial_value,
    projected_scores,
    reconstruct_ledger,
)
from garboid_pocketrocks.knowledge import canonical_knowledge

from .helpers import make_context

RULESET = canonical_knowledge(4, value_chart="A", objectives_enabled=True)


def _history(*events: object, players: int = 4, tiebreak: int = 0) -> PublicHistory:
    setup = GameSetupEvent(
        kind="gameSetup",
        player_count=players,
        starting_cash=25,
        value_chart=(0, 4, 8, 12, 16, 20),
        initial_tiebreak_seat=tiebreak,
        objective_ids=(1, 2, 3, 4),
    )
    return public_history_from_sdk_events((setup, *events))


def _turn(action: ActionId, resources: tuple[int, int] = (0, 0)) -> TurnOpenedEvent:
    return TurnOpenedEvent(
        kind="turnOpened",
        action_id=int(action),
        resource_ids=(int(resources[0]), int(resources[1])),
    )


def _resolved(bids: tuple[int, ...]) -> AuctionResolvedEvent:
    return AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=tuple(bids))


# -- winner derivation --------------------------------------------------------


@pytest.mark.parametrize(
    ("bids", "tiebreak", "expected"),
    [
        ((5, 1, 0, 0), 0, 0),  # outright high bid
        ((7, 7, 7, 7), 2, 3),  # all tied -> holder+1
        ((7, 7, 7, 7), 3, 0),  # wraps around
        ((4, 0, 4, 0), 2, 0),  # tie between 0 and 2, scan starts at 3
        ((0, 0, 0, 0), 1, 2),  # a free lot still goes to someone
    ],
)
def test_winner_matches_the_tie_break_rule(
    bids: tuple[int, ...], tiebreak: int, expected: int
) -> None:
    assert _winning_seat(bids, tiebreak) == expected


# -- ledger reconstruction ----------------------------------------------------


def test_loan_debt_is_attributed_to_the_winner() -> None:
    """Loan principal is invisible in DecisionContext but public in history."""
    history = _history(_turn(ActionId.LOAN20), _resolved((0, 3, 0, 0)))
    ledger = reconstruct_ledger(history, RULESET)
    assert ledger.seats[1].loan_debt == 20
    assert [seat.loan_debt for seat in ledger.seats] == [0, 20, 0, 0]


def test_investment_return_includes_the_locked_bid() -> None:
    """An Invest5 returns the locked bid plus 5, both recoverable from history."""
    history = _history(_turn(ActionId.INVEST5), _resolved((0, 0, 6, 0)))
    ledger = reconstruct_ledger(history, RULESET)
    assert ledger.seats[2].investment_return == 6 + 5
    assert ledger.seats[2].paid_total == 6


def test_tiebreak_marker_follows_the_winner_across_turns() -> None:
    """Winning moves the marker, which changes who takes the *next* tie."""
    history = _history(
        _turn(ActionId.AUCTION1, (Suit.BRICK, 0)),
        _resolved((9, 0, 0, 0)),
        _turn(ActionId.LOAN10),
        _resolved((2, 2, 2, 2)),
    )
    ledger = reconstruct_ledger(history, RULESET)
    # Seat 0 wins the first auction outright, so the marker sits on seat 0. The
    # four-way tie then resolves to seat 1.
    assert ledger.seats[0].auctions_won == 1
    assert ledger.seats[1].loan_debt == 10


def test_reconstructed_winner_agrees_with_garboid_reveal_attribution() -> None:
    """garboid credits a reveal to the auction winner; our winner must match it."""
    history = _history(
        _turn(ActionId.AUCTION1, (Suit.ORE, 0)),
        _resolved((0, 0, 5, 5)),
        InfoRevealedEvent(kind="infoRevealed", suit_id=int(Suit.WHEAT)),
    )
    revealed = [event for event in history if isinstance(event, PublicInformationRevealed)]
    assert revealed, "expected a reveal event"
    ledger = reconstruct_ledger(history, RULESET)
    winner = next(seat for seat, entry in enumerate(ledger.seats) if entry.auctions_won == 1)
    assert winner == revealed[0].seat


def test_remaining_action_deck_counts_down() -> None:
    fresh = reconstruct_ledger(_history(), RULESET)
    # 12 Auction1 + 8 Auction2 in the deck.
    assert fresh.remaining_auctions == 20

    history = _history(
        _turn(ActionId.AUCTION1, (Suit.BRICK, 0)),
        _resolved((1, 0, 0, 0)),
        _turn(ActionId.AUCTION2, (Suit.WOOD, Suit.ORE)),
        _resolved((0, 1, 0, 0)),
        _turn(ActionId.LOAN10),
        _resolved((0, 0, 1, 0)),
    )
    used = reconstruct_ledger(history, RULESET)
    assert used.remaining_auctions == 18
    assert used.remaining_actions[int(ActionId.LOAN10)] == 2


def test_auction_pressure_falls_from_one_to_zero() -> None:
    assert auction_pressure(reconstruct_ledger(_history(), RULESET), RULESET) == 1.0
    events: list[object] = []
    for _ in range(20):
        events += [_turn(ActionId.AUCTION1, (Suit.BRICK, 0)), _resolved((1, 0, 0, 0))]
    drained = reconstruct_ledger(_history(*events), RULESET)
    assert auction_pressure(drained, RULESET) == pytest.approx(0.0)


# -- standings ----------------------------------------------------------------


def test_projected_score_subtracts_loan_debt_and_adds_investment_return() -> None:
    history = _history(
        _turn(ActionId.LOAN20),
        _resolved((5, 0, 0, 0)),
        _turn(ActionId.INVEST10),
        _resolved((0, 4, 0, 0)),
    )
    ledger = reconstruct_ledger(history, RULESET)
    ctx = make_context(
        player_count=4,
        cash=(40, 21, 25, 25),
        won=((0, 0, 0, 0, 0),) * 4,
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((), (), (), ()),
    )
    scores = projected_scores(ctx, ledger, (0.0,) * 5)
    # Seat 0 borrowed 20 (already inside its cash) so it owes 20 back.
    assert scores[0] == pytest.approx(40 - 20)
    # Seat 1 locked 4 into an Invest10, returning 4 + 10.
    assert scores[1] == pytest.approx(21 + 14)


def test_projected_score_values_owned_resources_and_objectives() -> None:
    ledger = reconstruct_ledger(_history(), RULESET)
    ctx = make_context(
        player_count=4,
        cash=(10, 10, 10, 10),
        won=((2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((1,), (), (), ()),  # objective 1 pays 5
    )
    scores = projected_scores(ctx, ledger, (8.0, 0.0, 0.0, 0.0, 0.0))
    assert scores[0] == pytest.approx(10 + 2 * 8.0 + 5)


def test_deficit_to_leader_is_negative_when_ahead() -> None:
    assert deficit_to_leader((50.0, 30.0, 20.0), 0) == pytest.approx(-20.0)
    assert deficit_to_leader((10.0, 30.0, 20.0), 0) == pytest.approx(20.0)
    assert deficit_to_leader((10.0,), 0) == 0.0


# -- denial -------------------------------------------------------------------


def test_denial_prices_an_objective_a_rival_would_complete() -> None:
    """Objective 1 is 'any two of a single suit', payout 5."""
    ctx = make_context(
        player_count=4,
        cash=(10, 10, 10, 10),
        won=((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((), (), (), ()),
        objectives=(1,),
    )
    # A single Brick completes seat 1's pair.
    assert denial_value(ctx, (1, 0, 0, 0, 0)) == 5
    # A Wood does nothing for them.
    assert denial_value(ctx, (0, 1, 0, 0, 0)) == 0


def test_denial_ignores_objectives_already_claimed() -> None:
    ctx = make_context(
        player_count=4,
        cash=(10, 10, 10, 10),
        won=((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((1,), (), (), ()),
        objectives=(1,),
    )
    assert denial_value(ctx, (1, 0, 0, 0, 0)) == 0


def test_denial_ignores_our_own_progress() -> None:
    """Our own completions are priced by evaluate_objectives, not here."""
    ctx = make_context(
        player_count=4,
        cash=(10, 10, 10, 10),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((), (), (), ()),
        objectives=(1,),
    )
    assert denial_value(ctx, (1, 0, 0, 0, 0)) == 0


def test_denial_with_no_objectives_in_play() -> None:
    ctx = make_context(
        player_count=4,
        cash=(10, 10, 10, 10),
        won=((0, 0, 0, 0, 0),) * 4,
        revealed=((0, 0, 0, 0, 0),) * 4,
        owned_objectives=((), (), (), ()),
        objectives=(),
    )
    assert denial_value(ctx, (1, 1, 0, 0, 0)) == 0
