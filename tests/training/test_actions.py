from __future__ import annotations

import numpy as np
import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds


def _context(
    *, decision_kind: str, legal_max: int | None, revealable_count: int
) -> DecisionContext:
    return DecisionContext(
        request_id="test",
        deadline_at=0,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=0,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(1, 2, 3, 4, 5),
        legal_max_amount=legal_max,
        revealable_count=revealable_count,
    )


def test_action_codec_round_trips_legal_decisions() -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))

    assert codec.decode(0) == BotDecision.pass_turn()
    assert codec.encode(BotDecision.submit_bid(0)) == 0
    assert codec.decode(17) == BotDecision.submit_bid(17)
    assert codec.decode(101) == BotDecision.select_info_to_reveal(0)
    assert codec.encode(BotDecision.select_info_to_reveal(4)) == 105


def test_action_masks_match_sdk_context_legality() -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))

    bid_mask = codec.mask(_context(decision_kind="submitBid", legal_max=7, revealable_count=5))
    assert bid_mask.dtype == np.int8
    assert tuple(index for index, enabled in enumerate(bid_mask) if enabled) == tuple(range(8))

    reveal_mask = codec.mask(
        _context(decision_kind="selectInfoToReveal", legal_max=None, revealable_count=3)
    )
    assert tuple(index for index, enabled in enumerate(reveal_mask) if enabled) == (
        0,
        101,
        102,
        103,
    )


@pytest.mark.parametrize(
    "action",
    (-1, 106),
)
def test_decode_rejects_actions_outside_the_fixed_space(action: int) -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))

    with pytest.raises(ValueError, match="outside"):
        codec.decode(action)


def test_encode_rejects_decisions_outside_the_fixed_space() -> None:
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))

    with pytest.raises(ValueError, match="bid"):
        codec.encode(BotDecision.submit_bid(101))
    with pytest.raises(ValueError, match="reveal"):
        codec.encode(BotDecision.select_info_to_reveal(5))
