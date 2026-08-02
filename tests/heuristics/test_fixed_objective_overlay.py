from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import ActionId, BotDecision

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEvent,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_TUNED_V1_PROFILE
from garboid_pocketrocks.bots.fixed_objective_overlay import (
    FIXED_OBJECTIVE_OVERLAY_V1_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V2_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC,
    FIXED_OBJECTIVE_OVERLAY_V3_PROFILE,
    FixedObjectiveOverlayV1Brain,
    FixedObjectiveOverlayV2Brain,
    FixedObjectiveOverlayV3Brain,
    _public_only_expected_prices,
)
from garboid_pocketrocks.diagnostics.trace import (
    BotResultMetric,
    FixedObjectiveOverlayV3BidExplanation,
)

from .helpers import make_context, make_knowledge


def _late_history(
    *,
    action: ActionId = ActionId.AUCTION2,
    bids: tuple[int, int, int] = (0, 0, 0),
    initial_tiebreak_seat: int = 2,
) -> PublicHistory:
    resources = (1, 0) if action is ActionId.AUCTION1 else (1, 2)
    events: list[PublicEvent] = [
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=30,
            value_chart=(0, 4, 8, 12, 16, 20),
            initial_tiebreak_seat=initial_tiebreak_seat,
            objective_ids=(),
        )
    ]
    for _ in range(12):
        events.extend(
            (
                PublicTurnOpened(
                    kind=PublicEventKind.TURN_OPENED,
                    action_id=int(action),
                    resource_ids=resources,
                ),
                PublicAuctionResolved(
                    kind=PublicEventKind.AUCTION_RESOLVED,
                    bids_by_seat=bids,
                ),
            )
        )
    events.append(
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=int(action),
            resource_ids=resources,
        )
    )
    return tuple(events)


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


def test_v3_profile_pins_the_selected_constrained_search_result() -> None:
    assert FIXED_OBJECTIVE_OVERLAY_V3_PROFILE.target_bids == (5, 10, 3, 6, 4, 7)
    for action in (ActionId.AUCTION1, ActionId.AUCTION2, ActionId.INVEST5, ActionId.INVEST10):
        v3_target = FIXED_OBJECTIVE_OVERLAY_V3_PROFILE.target_bid(action)
        v2_target = FIXED_BID_TUNED_V1_PROFILE.target_bid(action)
        assert v3_target is not None
        assert v2_target is not None
        assert v3_target <= v2_target


@pytest.mark.parametrize(
    ("action", "expected"),
    (
        (ActionId.LOAN10, 3),
        (ActionId.LOAN20, 6),
        (ActionId.INVEST5, 4),
        (ActionId.INVEST10, 7),
    ),
)
def test_v3_uses_selected_nonresource_targets(action: ActionId, expected: int) -> None:
    assert FixedObjectiveOverlayV3Brain().choose_decision(
        make_context(action_id=action),
        make_knowledge(),
    ) == BotDecision.submit_bid(expected)


def test_v3_bids_opponent_maximum_when_it_wins_ties() -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(1, 2),
        cash=(10, 8, 7),
        legal_max=10,
    )

    decision = FixedObjectiveOverlayV3Brain().choose_decision(
        context,
        make_knowledge(),
    )

    assert decision == BotDecision.submit_bid(8)
    assert context.is_legal(decision)


def test_v3_bids_one_above_opponent_maximum_when_it_loses_ties() -> None:
    context = replace(
        make_context(
            action_id=ActionId.AUCTION2,
            current_resources=(1, 2),
            cash=(10, 8, 7),
            legal_max=10,
        ),
        tiebreak_seat=0,
    )

    decision = FixedObjectiveOverlayV3Brain().choose_decision(
        context,
        make_knowledge(),
    )

    assert decision == BotDecision.submit_bid(9)
    assert context.is_legal(decision)


def test_v3_can_pass_to_guarantee_an_all_zero_auction() -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(1, 2),
        cash=(10, 0, 0),
        legal_max=10,
    )

    assert (
        FixedObjectiveOverlayV3Brain().choose_decision(
            context,
            make_knowledge(),
        )
        == BotDecision.pass_turn()
    )


def test_v3_does_not_chase_a_guarantee_above_its_v2_plan() -> None:
    context = replace(
        make_context(
            action_id=ActionId.AUCTION1,
            cash=(10, 8, 7),
            legal_max=10,
        ),
        tiebreak_seat=1,
    )
    assert FixedObjectiveOverlayV3Brain().choose_decision(
        context,
        make_knowledge(),
    ) == FixedObjectiveOverlayV2Brain().choose_decision(context, make_knowledge())


def test_v3_decision_does_not_depend_on_bid_history() -> None:
    context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(1, 2),
        cash=(10, 8, 7),
        legal_max=10,
    )
    brain = FixedObjectiveOverlayV3Brain()

    assert brain.choose_decision(context, make_knowledge(), ()) == brain.choose_decision(
        context,
        make_knowledge(),
        _late_history(bids=(0, 30, 0)),
    )


def test_v3_explanation_classifies_guaranteed_and_inapplicable_bids() -> None:
    guarantee_context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(1, 2),
        cash=(10, 8, 7),
        legal_max=10,
    )
    brain = FixedObjectiveOverlayV3Brain()

    guaranteed = brain.choose_explained_decision(
        guarantee_context,
        make_knowledge(),
        (),
    )
    baseline_context = make_context(
        action_id=ActionId.AUCTION2,
        current_resources=(1, 2),
        cash=(30, 30, 30),
        legal_max=30,
    )
    baseline = brain.choose_explained_decision(baseline_context, make_knowledge(), ())

    assert guaranteed.explanation == FixedObjectiveOverlayV3BidExplanation(
        rule="guaranteed_win",
        planned_bid=10,
        chosen_bid=8,
    )
    assert baseline.explanation == FixedObjectiveOverlayV3BidExplanation(
        rule="baseline",
        planned_bid=10,
        chosen_bid=10,
    )
    namespace = "fixed_objective_overlay_v3_rules"
    assert guaranteed.result_metrics == (
        BotResultMetric(namespace, ("bid_decisions",), "sum", 1),
        BotResultMetric(namespace, ("rule_counts", "guaranteed_win"), "sum", 1),
        BotResultMetric(namespace, ("resource_auction_decisions",), "sum", 1),
        BotResultMetric(namespace, ("rule_application_rate",), "mean", 1),
        BotResultMetric(namespace, ("adjusted_bid_decisions",), "sum", 1),
        BotResultMetric(namespace, ("adjusted_bid_rate",), "mean", 1),
        BotResultMetric(namespace, ("total_bid_reduction",), "sum", 2),
    )
    assert baseline.result_metrics == (
        BotResultMetric(namespace, ("bid_decisions",), "sum", 1),
        BotResultMetric(namespace, ("rule_counts", "baseline"), "sum", 1),
        BotResultMetric(namespace, ("resource_auction_decisions",), "sum", 1),
        BotResultMetric(namespace, ("rule_application_rate",), "mean", 0),
        BotResultMetric(namespace, ("adjusted_bid_decisions",), "sum", 0),
        BotResultMetric(namespace, ("adjusted_bid_rate",), "mean", 0),
        BotResultMetric(namespace, ("total_bid_reduction",), "sum", 0),
    )


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
        (
            FIXED_OBJECTIVE_OVERLAY_V3_BOT_SPEC,
            "fixed-objective-overlay-v3",
            FixedObjectiveOverlayV3Brain,
        ),
    ),
)
def test_versioned_specs_are_local_deterministic_and_seed_invariant(
    spec: BotSpec,
    name: str,
    brain_type: (
        type[FixedObjectiveOverlayV1Brain]
        | type[FixedObjectiveOverlayV2Brain]
        | type[FixedObjectiveOverlayV3Brain]
    ),
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
