"""The fitted opponent-bid prior."""

from __future__ import annotations

import numpy as np
from pocketrocks import ActionId

from garboid_pocketrocks.heuristics.bid_priors import (
    BID_PRIOR_V1,
    UNINFORMED_BID_PRIOR,
    BidPrior,
    action_class,
    phase_for,
)


def test_phase_boundaries_match_garboid_analysis_bands() -> None:
    # One-based turn: <=5 early, <=12 middle, else late.
    assert phase_for(0) == "early"
    assert phase_for(4) == "early"
    assert phase_for(5) == "middle"
    assert phase_for(11) == "middle"
    assert phase_for(12) == "late"


def test_action_class_groups_loans_and_invests() -> None:
    assert action_class(int(ActionId.AUCTION1)) == "auction1"
    assert action_class(int(ActionId.AUCTION2)) == "auction2"
    assert action_class(int(ActionId.LOAN10)) == "loan"
    assert action_class(int(ActionId.LOAN20)) == "loan"
    assert action_class(int(ActionId.INVEST5)) == "invest"
    assert action_class(None) is None


def test_draws_are_capped_by_affordable_cash() -> None:
    prior = BidPrior(samples={"auction1/early": (1.0, 1.4)}, fallback=(1.0,))
    drawn = prior.draw(int(ActionId.AUCTION1), 0, 7, 128, np.random.default_rng(0))
    assert drawn.max() <= 7
    assert drawn.min() >= 0


def test_zero_cap_draws_nothing() -> None:
    drawn = UNINFORMED_BID_PRIOR.draw(int(ActionId.AUCTION1), 0, 0, 16, np.random.default_rng(0))
    assert drawn.tolist() == [0] * 16


def test_unknown_bucket_falls_back() -> None:
    prior = BidPrior(samples={}, fallback=(0.5,))
    drawn = prior.draw(int(ActionId.INVEST10), 20, 10, 8, np.random.default_rng(1))
    assert drawn.tolist() == [5] * 8


def test_shipped_prior_covers_every_bucket() -> None:
    """The committed table is what the bot actually plays with."""
    prior = BID_PRIOR_V1
    for action in ("auction1", "auction2", "loan", "invest"):
        for phase in ("early", "middle", "late"):
            assert f"{action}/{phase}" in prior.samples


def test_shipped_prior_says_auction2_costs_more_than_auction1() -> None:
    """Sanity check against the game: a two-card lot should draw larger bids."""
    prior = BID_PRIOR_V1
    one = prior.mean_fraction(int(ActionId.AUCTION1), 8)
    two = prior.mean_fraction(int(ActionId.AUCTION2), 8)
    assert two > one
