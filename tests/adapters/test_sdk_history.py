from pocketrocks.sim import SimEngine
from pocketrocks.sim.context import build_sim_request

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicInformationRevealed,
    PublicTurnOpened,
    public_history_from_sdk_events,
    public_history_from_sdk_frame,
)


def test_sdk_events_match_equivalent_decision_request_history() -> None:
    engine = SimEngine(3, "sdk-history")
    assert engine.flip_action() is not None
    outcome = engine.resolve((1, 2, 3))
    if outcome.reveal_needed is not None:
        engine.apply_reveal(outcome.winner_seat, 0, auto=outcome.reveal_needed == "auto")

    request = build_sim_request(
        engine,
        0,
        "submitBid",
        budget_ms=60_000,
    )

    actual = public_history_from_sdk_events(engine.events)

    assert actual == public_history_from_sdk_frame(request)
    assert isinstance(actual[1], PublicTurnOpened)
    assert actual[1].resource_ids == engine.history[0].upcoming_before
    assert isinstance(actual[2], PublicAuctionResolved)
    assert actual[2].bids_by_seat == engine.history[0].effective_bids
    assert isinstance(actual[3], PublicInformationRevealed)
    assert engine.history[0].reveal is not None
    assert actual[3].suit_id == engine.history[0].reveal.suit
