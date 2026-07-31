from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import OBJECTIVES, ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_BOT_SPEC, FixedBidBotBrain
from garboid_pocketrocks.knowledge import RulesetKnowledge


def _knowledge() -> RulesetKnowledge:
    return RulesetKnowledge(
        name="fixed-bid-test",
        player_count=3,
        starting_cash=30,
        private_cards_per_player=0,
        resource_counts=(2, 2, 2, 2, 2),
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def _context(
    *,
    action_id: ActionId | int = ActionId.AUCTION1,
    decision_kind: str = "submitBid",
    legal_max: int | None = 30,
    revealable_count: int = 0,
) -> DecisionContext:
    return DecisionContext(
        request_id="fixed-bid-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(),
        current_action_id=int(action_id),
        current_resource_ids=(0, 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(),
        legal_max_amount=legal_max,
        revealable_count=revealable_count,
    )


@pytest.mark.parametrize(
    ("action_id", "expected"),
    (
        (ActionId.AUCTION1, 6),
        (ActionId.AUCTION2, 12),
        (ActionId.LOAN10, 1),
        (ActionId.LOAN20, 1),
        (ActionId.INVEST5, 7),
        (ActionId.INVEST10, 7),
    ),
)
def test_fixed_bid_brain_uses_action_target(action_id: ActionId, expected: int) -> None:
    context = _context(action_id=action_id)

    decision = FixedBidBotBrain().choose_decision(context, _knowledge())

    assert decision == BotDecision.submit_bid(expected)
    assert context.is_legal(decision)


@pytest.mark.parametrize(
    ("action_id", "legal_max"),
    (
        (ActionId.AUCTION1, 4),
        (ActionId.AUCTION2, 9),
        (ActionId.LOAN10, 0),
        (ActionId.INVEST5, 3),
    ),
)
def test_fixed_bid_brain_caps_target_at_legal_maximum(
    action_id: ActionId,
    legal_max: int,
) -> None:
    context = _context(action_id=action_id, legal_max=legal_max)

    decision = FixedBidBotBrain().choose_decision(context, _knowledge())

    expected = BotDecision.pass_turn() if legal_max == 0 else BotDecision.submit_bid(legal_max)
    assert decision == expected
    assert context.is_legal(decision)


@pytest.mark.parametrize("legal_max", (None, 0, -1))
def test_fixed_bid_brain_passes_without_positive_bid_limit(legal_max: int | None) -> None:
    context = _context(legal_max=legal_max)

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == BotDecision.pass_turn()


def test_fixed_bid_brain_passes_for_unknown_action() -> None:
    context = replace(_context(), current_action_id=999)

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == BotDecision.pass_turn()


@pytest.mark.parametrize(
    ("revealable_count", "expected"),
    (
        (2, BotDecision.select_info_to_reveal(0)),
        (0, BotDecision.pass_turn()),
    ),
)
def test_fixed_bid_brain_reveals_first_option_when_available(
    revealable_count: int,
    expected: BotDecision,
) -> None:
    context = _context(
        decision_kind="selectInfoToReveal",
        legal_max=None,
        revealable_count=revealable_count,
    )

    assert FixedBidBotBrain().choose_decision(context, _knowledge()) == expected


def test_fixed_bid_spec_is_local_deterministic_and_builds_fresh_brains() -> None:
    left = FIXED_BID_BOT_SPEC.make_brain(seed=11)
    right = FIXED_BID_BOT_SPEC.make_brain(seed=999)
    context = _context(action_id=ActionId.AUCTION2)

    assert FIXED_BID_BOT_SPEC.name == "fixed-bid"
    assert FIXED_BID_BOT_SPEC.bot_id == "fixed-bid"
    assert isinstance(left, FixedBidBotBrain)
    assert isinstance(right, FixedBidBotBrain)
    assert left is not right
    assert left.choose_decision(context, _knowledge()) == right.choose_decision(
        context,
        _knowledge(),
    )
