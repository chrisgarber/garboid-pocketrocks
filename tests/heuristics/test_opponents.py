"""The opponent model: history parsing and tie priority."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pocketrocks import ActionId, DecisionContext, Suit
from pocketrocks.internal.bot_wire_v2 import (
    AuctionResolvedEvent,
    GameSetupEvent,
    TurnOpenedEvent,
)

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.heuristics.bid_priors import BidPrior
from garboid_pocketrocks.heuristics.opponents import (
    BidSampler,
    observe_bids,
    scan_priority,
    turn_index_from_history,
)

from .helpers import make_context


def _history(*events: object) -> PublicHistory:
    setup = GameSetupEvent(
        kind="gameSetup",
        player_count=4,
        starting_cash=25,
        value_chart=(0, 4, 8, 12, 16, 20),
        initial_tiebreak_seat=0,
        objective_ids=(1, 2, 3, 4),
    )
    return public_history_from_sdk_events((setup, *events))


def test_resource_and_other_bids_are_separated() -> None:
    history = _history(
        TurnOpenedEvent(
            kind="turnOpened", action_id=int(ActionId.AUCTION1), resource_ids=(int(Suit.BRICK), 0)
        ),
        AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=(3, 5, 0, 1)),
        TurnOpenedEvent(kind="turnOpened", action_id=int(ActionId.LOAN10), resource_ids=(0, 0)),
        AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=(2, 0, 0, 7)),
    )
    bids = observe_bids(history, 4)
    assert bids.resource[1] == (5,)
    assert bids.other[3] == (7,)
    # Loans must not pollute the resource distribution -- rivals price them differently.
    assert bids.resource[3] == (1,)


def test_for_action_routes_by_action_class() -> None:
    bids = observe_bids(
        _history(
            TurnOpenedEvent(
                kind="turnOpened",
                action_id=int(ActionId.AUCTION2),
                resource_ids=(int(Suit.ORE), int(Suit.WHEAT)),
            ),
            AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=(1, 2, 3, 4)),
        ),
        4,
    )
    assert bids.for_action(int(ActionId.AUCTION2)) == bids.resource
    assert bids.for_action(int(ActionId.INVEST5)) == bids.other
    assert bids.for_action(None) == bids.other


def _context(
    *,
    players: int,
    tiebreak: int,
    cash: tuple[int, ...] | None = None,
    seat: int = 0,
) -> DecisionContext:
    """A minimal bidding context with a chosen tie-break seat."""

    zeros = ((0, 0, 0, 0, 0),) * players
    context = make_context(
        player_count=players,
        bot_seat=seat,
        cash=cash if cash is not None else tuple([20] * players),
        won=zeros,
        revealed=zeros,
        owned_objectives=tuple(() for _ in range(players)),
    )
    return replace(context, tiebreak_seat=tiebreak)


@pytest.mark.parametrize(
    ("tiebreak", "expected_best_seat"),
    [(0, 1), (1, 2), (2, 3), (3, 0)],
)
def test_priority_starts_after_the_tiebreak_holder(tiebreak: int, expected_best_seat: int) -> None:
    """Ties scan forward from holder+1, so the holder is checked LAST."""
    priority = scan_priority(_context(players=4, tiebreak=tiebreak))
    assert priority[expected_best_seat] == 0
    assert priority[tiebreak] == 3


def test_sampled_bids_never_exceed_what_a_seat_can_afford() -> None:
    ctx = _context(players=4, tiebreak=0, cash=(25, 3, 0, 40))
    bids = observe_bids(_history(), 4)
    sampler = BidSampler(
        ctx,
        bids,
        rng=np.random.default_rng(0),
        prior_weight=1.0,
        prior=BidPrior(samples={}, fallback=(0.9, 1.2)),
    )
    best, _tie = sampler.sample(256)
    # The richest opponent holds 40, so nothing may exceed that.
    assert best.max() <= 40
    assert best.min() >= 0


def test_single_opponentless_table_is_handled() -> None:
    ctx = _context(players=3, tiebreak=0)
    sampler = BidSampler(
        ctx,
        observe_bids(_history(), 3),
        rng=np.random.default_rng(1),
        prior_weight=1.0,
    )
    best, tie = sampler.sample(16)
    assert best.shape == (16,)
    assert tie.shape == (16,)


def test_turn_index_counts_resolved_auctions() -> None:
    """DecisionContext has no turn number, so it is derived from public events."""
    assert turn_index_from_history(_history()) == 0
    history = _history(
        TurnOpenedEvent(
            kind="turnOpened", action_id=int(ActionId.AUCTION1), resource_ids=(int(Suit.BRICK), 0)
        ),
        AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=(1, 0, 0, 0)),
        TurnOpenedEvent(
            kind="turnOpened", action_id=int(ActionId.AUCTION1), resource_ids=(int(Suit.WOOD), 0)
        ),
        AuctionResolvedEvent(kind="auctionResolved", bids_by_seat=(0, 2, 0, 0)),
    )
    assert turn_index_from_history(history) == 2
