from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import ActionId, BotDecision

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_objective_overlay import (
    FIXED_OBJECTIVE_OVERLAY_V1_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V2_BOT_SPEC,
    FixedObjectiveOverlayV1Brain,
    FixedObjectiveOverlayV2Brain,
    _public_only_expected_prices,
)

from .helpers import make_context, make_knowledge


@pytest.mark.parametrize(
    ("action", "expected"),
    (
        (ActionId.LOAN10, 2),
        (ActionId.LOAN20, 4),
        (ActionId.INVEST5, 4),
        (ActionId.INVEST10, 9),
    ),
)
def test_nonauction_targets_stay_identical_to_fixed_bid(
    action: ActionId,
    expected: int,
) -> None:
    decision = FixedObjectiveOverlayV1Brain().choose_decision(
        make_context(action_id=action),
        make_knowledge(),
    )

    assert decision == BotDecision.submit_bid(expected)


@pytest.mark.parametrize(
    ("action", "expected"),
    (
        (ActionId.LOAN10, 2),
        (ActionId.LOAN20, 5),
        (ActionId.INVEST5, 4),
        (ActionId.INVEST10, 7),
    ),
)
def test_v2_nonauction_targets_use_tuned_fixed_base(
    action: ActionId,
    expected: int,
) -> None:
    decision = FixedObjectiveOverlayV2Brain().choose_decision(
        make_context(action_id=action),
        make_knowledge(),
    )

    assert decision == BotDecision.submit_bid(expected)


def test_v1_and_v2_reuse_identical_auction_overlay_behavior() -> None:
    context = make_context(
        current_resources=(1, 0),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(6,),
    )
    knowledge = replace(
        make_knowledge(),
        active_objective_count=1,
        objectives_enabled=True,
    )

    assert FixedObjectiveOverlayV1Brain().choose_decision(
        context,
        knowledge,
    ) == FixedObjectiveOverlayV2Brain().choose_decision(context, knowledge)


def test_reveal_policy_stays_identical_to_fixed_bid() -> None:
    decision = FixedObjectiveOverlayV1Brain().choose_decision(
        make_context(
            decision_kind="selectInfoToReveal",
            current_resources=(0, 0),
            hand=(2, 4),
            legal_max=None,
        ),
        make_knowledge(private_cards=2),
    )

    assert decision == BotDecision.select_info_to_reveal(0)


def test_completing_unclaimed_objective_adds_at_most_two() -> None:
    context = make_context(
        current_resources=(1, 0),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(6,),
    )
    knowledge = replace(
        make_knowledge(),
        active_objective_count=1,
        objectives_enabled=True,
    )

    assert FixedObjectiveOverlayV1Brain().choose_decision(
        context,
        knowledge,
    ) == BotDecision.submit_bid(7)


def test_claimed_objective_does_not_raise_bid() -> None:
    context = make_context(
        current_resources=(1, 0),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(6,),
        owned_objectives=((), (6,), ()),
    )
    knowledge = replace(
        make_knowledge(),
        active_objective_count=1,
        objectives_enabled=True,
    )

    assert FixedObjectiveOverlayV1Brain().choose_decision(
        context,
        knowledge,
    ) == BotDecision.submit_bid(5)


def test_low_relative_resource_value_reduces_bid() -> None:
    context = make_context(
        current_resources=(1, 0),
        revealed=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)),
    )

    assert FixedObjectiveOverlayV1Brain().choose_decision(
        context,
        make_knowledge(private_cards=1),
    ) == BotDecision.submit_bid(4)


def test_private_hand_moves_bid_without_changing_public_price_belief() -> None:
    knowledge = make_knowledge(private_cards=1, resource_counts=(4, 4, 4, 4, 4))
    matching_hand = make_context(current_resources=(1, 0), hand=(1,))
    other_hand = make_context(current_resources=(1, 0), hand=(5,))
    brain = FixedObjectiveOverlayV1Brain()

    assert _public_only_expected_prices(matching_hand, knowledge) == pytest.approx(
        _public_only_expected_prices(other_hand, knowledge)
    )
    assert brain.choose_decision(matching_hand, knowledge) == BotDecision.submit_bid(7)
    assert brain.choose_decision(other_hand, knowledge) == BotDecision.submit_bid(4)


def test_adjusted_bid_respects_legal_cap() -> None:
    context = make_context(
        current_resources=(1, 0),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(6,),
        legal_max=6,
    )
    knowledge = replace(
        make_knowledge(),
        active_objective_count=1,
        objectives_enabled=True,
    )

    decision = FixedObjectiveOverlayV1Brain().choose_decision(context, knowledge)

    assert decision == BotDecision.submit_bid(6)
    assert context.is_legal(decision)


def test_invalid_valuation_input_falls_back_to_fixed_bid() -> None:
    context = make_context(current_resources=(1, 0), hand=(1,))

    assert FixedObjectiveOverlayV1Brain().choose_decision(
        context,
        make_knowledge(private_cards=0),
    ) == BotDecision.submit_bid(5)


@pytest.mark.parametrize(
    ("spec", "name", "brain_type"),
    (
        (
            FIXED_OBJECTIVE_OVERLAY_V1_BOT_SPEC,
            "fixed-objective-overlay-v1",
            FixedObjectiveOverlayV1Brain,
        ),
        (
            FIXED_OBJECTIVE_OVERLAY_V2_BOT_SPEC,
            "fixed-objective-overlay-v2",
            FixedObjectiveOverlayV2Brain,
        ),
    ),
)
def test_versioned_specs_are_local_deterministic_and_seed_invariant(
    spec: BotSpec,
    name: str,
    brain_type: type[FixedObjectiveOverlayV1Brain] | type[FixedObjectiveOverlayV2Brain],
) -> None:
    left = spec.make_brain(seed=11)
    right = spec.make_brain(seed=999)
    context = make_context(current_resources=(1, 0))
    knowledge = make_knowledge()

    assert spec.name == name
    assert spec.bot_id == name
    assert isinstance(left, brain_type)
    assert isinstance(right, brain_type)
    assert left is not right
    assert left.choose_decision(context, knowledge) == right.choose_decision(context, knowledge)
