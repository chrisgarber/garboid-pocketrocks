from __future__ import annotations

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicTurnOpened,
)
from garboid_pocketrocks.bots.surplus import (
    SURPLUS_BOT_SPEC,
    SURPLUS_V1_BOT_SPEC,
    SURPLUS_V1_POLICY,
    SURPLUS_V2_BOT_SPEC,
    SURPLUS_V2_POLICY,
    SURPLUS_V3_BOT_SPEC,
    SURPLUS_V3_POLICY,
    SURPLUS_V4_BOT_SPEC,
    SURPLUS_V4_POLICY,
    SURPLUS_V5_BOT_SPEC,
    SURPLUS_V5_POLICY,
    SURPLUS_V6_BOT_SPEC,
    SURPLUS_V6_POLICY,
    SURPLUS_V7_BOT_SPEC,
    SURPLUS_V7_POLICY,
    SURPLUS_V8_BOT_SPEC,
    SURPLUS_V8_POLICY,
    SURPLUS_V9_BOT_SPEC,
    SURPLUS_V9_POLICY,
    SURPLUS_V10_BOT_SPEC,
    SURPLUS_V10_POLICY,
    SURPLUS_V11_BOT_SPEC,
    SURPLUS_V11_POLICY,
    SURPLUS_V12_BOT_SPEC,
    SURPLUS_V12_POLICY,
    SurplusBrain,
    SurplusPolicy,
    SurplusV1Brain,
    SurplusV2Brain,
    SurplusV3Brain,
    SurplusV4Brain,
    SurplusV5Brain,
    SurplusV6Brain,
    SurplusV7Brain,
    SurplusV8Brain,
    SurplusV9Brain,
    SurplusV10Brain,
    SurplusV11Brain,
    SurplusV12Brain,
)
from garboid_pocketrocks.knowledge import canonical_knowledge


def _context(
    *,
    decision_kind: str = "submitBid",
    action_id: int | None = 1,
    resources: tuple[int, int] = (1, 2),
    hand: tuple[int, ...] = (1, 1, 3, 4, 5),
    revealed: tuple[tuple[int, ...], ...] | None = None,
    legal_max: int | None = 30,
    cash: tuple[int, ...] = (30, 30, 30),
    revealable_count: int = 5,
    objectives: tuple[int, ...] = (),
    won: tuple[tuple[int, ...], ...] | None = None,
    owned: tuple[tuple[int, ...], ...] = ((), (), ()),
) -> DecisionContext:
    revealed_counts = revealed or ((0, 0, 0, 0, 0),) * 3
    return DecisionContext(
        request_id="test",
        deadline_at=0,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=objectives,
        current_action_id=action_id,
        current_resource_ids=resources,
        cash_by_seat=cash,
        tiebreak_seat=0,
        won_resource_counts_by_seat=won or ((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=revealed_counts,
        owned_objective_ids_by_seat=owned,
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=revealable_count,
    )


def test_v1_bids_shaded_visible_terminal_value() -> None:
    context = _context()

    decision = SurplusV1Brain().choose_decision(
        context,
        canonical_knowledge(3),
    )

    # Two private Brick cards imply $8; no visible Wood information implies $0.
    assert decision == BotDecision.submit_bid(5)


def test_v1_only_values_the_first_resource_in_a_single_resource_auction() -> None:
    context = _context(hand=(1, 1, 2, 2, 3))

    decision = SurplusV1Brain().choose_decision(context, canonical_knowledge(3))

    assert decision == BotDecision.submit_bid(5)


def test_v1_uses_public_reveals_and_respects_legal_maximum() -> None:
    context = _context(
        hand=(1, 3, 4, 5),
        revealed=((1, 2, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        legal_max=4,
    )

    decision = SurplusV1Brain().choose_decision(context, canonical_knowledge(3))

    assert decision == BotDecision.submit_bid(4)


def test_v1_passes_non_resource_actions_and_reveals_first_card() -> None:
    brain = SurplusV1Brain()

    assert (
        brain.choose_decision(
            _context(action_id=5),
            canonical_knowledge(3),
        )
        == BotDecision.pass_turn()
    )
    assert brain.choose_decision(
        _context(decision_kind="selectInfoToReveal", legal_max=None),
        canonical_knowledge(3),
    ) == BotDecision.select_info_to_reveal(0)


def test_v1_identity_is_versioned_local_and_seed_invariant() -> None:
    assert SURPLUS_V1_BOT_SPEC.name == "surplus-v1"
    assert SURPLUS_V1_BOT_SPEC.bot_id == "surplus-v1"
    assert type(SURPLUS_V1_BOT_SPEC.make_brain(seed=1)) is SurplusV1Brain
    assert type(SURPLUS_V1_BOT_SPEC.make_brain(seed=2)) is SurplusV1Brain


def test_v2_prices_unknown_opponent_information_from_the_finite_deck() -> None:
    context = _context()

    v1 = SurplusV1Brain().choose_decision(context, canonical_knowledge(3))
    v2 = SurplusV2Brain().choose_decision(context, canonical_knowledge(3))

    assert v1 == BotDecision.submit_bid(5)
    assert v2 == BotDecision.submit_bid(8)


def test_v2_identity_preserves_v1_and_is_seed_invariant() -> None:
    assert SURPLUS_V2_BOT_SPEC.name == "surplus-v2"
    assert SURPLUS_V2_BOT_SPEC.bot_id == "surplus-v2"
    assert type(SURPLUS_V2_BOT_SPEC.make_brain(seed=1)) is SurplusV2Brain
    assert type(SURPLUS_V2_BOT_SPEC.make_brain(seed=2)) is SurplusV2Brain
    assert type(SURPLUS_V1_BOT_SPEC.make_brain()) is SurplusV1Brain


def test_v3_adds_only_newly_completed_unclaimed_objective_value() -> None:
    known_opponent_hands = (
        (0, 0, 0, 0, 0),
        (0, 1, 1, 1, 2),
        (0, 1, 1, 1, 2),
    )
    context = _context(
        resources=(1, 2),
        revealed=known_opponent_hands,
        objectives=(1,),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )

    v2 = SurplusV2Brain().choose_decision(context, canonical_knowledge(3))
    v3 = SurplusV3Brain().choose_decision(context, canonical_knowledge(3))
    claimed = SurplusV3Brain().choose_decision(
        _context(
            resources=(1, 2),
            revealed=known_opponent_hands,
            objectives=(1,),
            won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
            owned=((), (1,), ()),
        ),
        canonical_knowledge(3),
    )

    assert v2 == BotDecision.submit_bid(5)
    assert v3 == BotDecision.submit_bid(8)
    assert claimed == v2


def test_v3_identity_preserves_earlier_generations() -> None:
    assert SURPLUS_V3_BOT_SPEC.name == "surplus-v3"
    assert SURPLUS_V3_BOT_SPEC.bot_id == "surplus-v3"
    assert type(SURPLUS_V3_BOT_SPEC.make_brain(seed=1)) is SurplusV3Brain
    assert type(SURPLUS_V2_BOT_SPEC.make_brain()) is SurplusV2Brain


def test_v4_bids_the_guaranteed_bonus_for_investments_but_not_loans() -> None:
    brain = SurplusV4Brain()

    assert brain.choose_decision(
        _context(action_id=5, legal_max=30),
        canonical_knowledge(3),
    ) == BotDecision.submit_bid(5)
    assert brain.choose_decision(
        _context(action_id=6, legal_max=7),
        canonical_knowledge(3),
    ) == BotDecision.submit_bid(7)
    assert (
        brain.choose_decision(
            _context(action_id=4, legal_max=50),
            canonical_knowledge(3),
        )
        == BotDecision.pass_turn()
    )


def test_v4_branches_from_v2_and_preserves_all_prior_identities() -> None:
    assert SURPLUS_V4_BOT_SPEC.name == "surplus-v4"
    assert SURPLUS_V4_BOT_SPEC.bot_id == "surplus-v4"
    assert type(SURPLUS_V4_BOT_SPEC.make_brain(seed=1)) is SurplusV4Brain
    assert type(SURPLUS_V3_BOT_SPEC.make_brain()) is SurplusV3Brain


def test_v5_caps_its_value_bid_at_observed_rival_price_plus_one() -> None:
    context = _context(hand=(1, 1, 1, 1, 1))
    history = (
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(2, 3),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(13, 4, 3),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(1, 2),
        ),
    )

    v4 = SurplusV4Brain().choose_decision(context, canonical_knowledge(3), history)
    v5 = SurplusV5Brain().choose_decision(context, canonical_knowledge(3), history)

    assert v4 == BotDecision.submit_bid(13)
    assert v5 == BotDecision.submit_bid(5)


def test_v5_never_raises_a_low_value_bid_to_the_market_price() -> None:
    context = _context(hand=(1, 1, 3, 4, 5))
    history = (
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(2, 3),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(5, 20, 18),
        ),
    )

    assert SurplusV5Brain().choose_decision(
        context,
        canonical_knowledge(3),
        history,
    ) == SurplusV4Brain().choose_decision(context, canonical_knowledge(3), history)


def test_v5_identity_preserves_all_prior_generations() -> None:
    assert SURPLUS_V5_BOT_SPEC.name == "surplus-v5"
    assert SURPLUS_V5_BOT_SPEC.bot_id == "surplus-v5"
    assert type(SURPLUS_V5_BOT_SPEC.make_brain(seed=1)) is SurplusV5Brain
    assert type(SURPLUS_V4_BOT_SPEC.make_brain()) is SurplusV4Brain


def test_v6_uses_objective_value_only_up_to_the_learned_market_cap() -> None:
    context = _context(
        resources=(1, 2),
        revealed=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 2),
            (0, 1, 1, 1, 2),
        ),
        objectives=(1,),
        won=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    history = (
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(2, 3),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(5, 6, 4),
        ),
    )

    v5 = SurplusV5Brain().choose_decision(context, canonical_knowledge(3), history)
    v6 = SurplusV6Brain().choose_decision(context, canonical_knowledge(3), history)

    assert v5 == BotDecision.submit_bid(5)
    assert v6 == BotDecision.submit_bid(7)
    assert v6.value is not None and v6.value <= 7


def test_v6_identity_preserves_v5() -> None:
    assert SURPLUS_V6_BOT_SPEC.name == "surplus-v6"
    assert SURPLUS_V6_BOT_SPEC.bot_id == "surplus-v6"
    assert type(SURPLUS_V6_BOT_SPEC.make_brain(seed=1)) is SurplusV6Brain
    assert type(SURPLUS_V5_BOT_SPEC.make_brain()) is SurplusV5Brain


def test_unversioned_alias_selects_latest_validated_generation() -> None:
    assert SURPLUS_BOT_SPEC.name == "surplus"
    assert SURPLUS_BOT_SPEC.bot_id == "surplus"
    assert type(SURPLUS_BOT_SPEC.make_brain(seed=1)) is SurplusV12Brain


def test_released_generation_policies_remain_exact() -> None:
    assert SURPLUS_V1_POLICY == SurplusPolicy()
    assert SURPLUS_V2_POLICY == SurplusPolicy(use_posterior_values=True)
    assert SURPLUS_V3_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
    )
    assert SURPLUS_V4_POLICY == SurplusPolicy(
        use_posterior_values=True,
        bid_investments=True,
    )
    assert SURPLUS_V5_POLICY == SurplusPolicy(
        use_posterior_values=True,
        bid_investments=True,
        use_market_prices=True,
    )
    assert SURPLUS_V6_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
        bid_investments=True,
        use_market_prices=True,
    )
    assert SURPLUS_V7_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
        bid_investments=True,
        use_market_prices=True,
        resource_value_numerator=13,
        resource_value_denominator=16,
        objective_value_numerator=3,
        objective_value_denominator=8,
    )
    assert SURPLUS_V8_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
        use_opponent_objective_threat=True,
        bid_investments=True,
        use_market_prices=True,
        resource_value_numerator=3,
        resource_value_denominator=4,
        objective_value_numerator=3,
        objective_value_denominator=8,
        opponent_objective_numerator=1,
        opponent_objective_denominator=32,
    )
    assert SURPLUS_V9_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
        use_opponent_objective_threat=True,
        use_objective_progress=True,
        bid_investments=True,
        bid_liquidity_loans=True,
        manage_liquidity=True,
        use_market_prices=True,
        resource_value_numerator=3,
        resource_value_denominator=4,
        objective_value_numerator=3,
        objective_value_denominator=8,
        opponent_objective_numerator=1,
        opponent_objective_denominator=32,
        objective_progress_numerator=1,
        objective_progress_denominator=8,
        resource_reserve_numerator=3,
        resource_reserve_denominator=4,
        investment_reserve_numerator=3,
        investment_reserve_denominator=2,
        objective_reserve_release_numerator=1,
        objective_reserve_release_denominator=2,
        loan_trigger_numerator=5,
        loan_trigger_denominator=1,
        loan_fee_numerator=3,
        loan_fee_denominator=10,
    )
    assert SURPLUS_V10_POLICY == SurplusPolicy(
        use_posterior_values=True,
        use_objective_values=True,
        use_opponent_objective_threat=True,
        use_objective_progress=True,
        bid_investments=True,
        bid_liquidity_loans=True,
        manage_liquidity=True,
        use_action_liquidity_demand=True,
        use_market_prices=True,
        resource_value_numerator=3,
        resource_value_denominator=4,
        objective_value_numerator=3,
        objective_value_denominator=8,
        opponent_objective_numerator=1,
        opponent_objective_denominator=32,
        objective_progress_numerator=1,
        objective_progress_denominator=8,
        resource_reserve_numerator=1,
        resource_reserve_denominator=8,
        investment_reserve_numerator=3,
        investment_reserve_denominator=10,
        objective_reserve_release_numerator=1,
        objective_reserve_release_denominator=2,
        loan_trigger_numerator=3,
        loan_trigger_denominator=4,
        loan_fee_numerator=2,
        loan_fee_denominator=5,
        loan_opening_fee_numerator=7,
        loan_opening_fee_denominator=20,
        auction1_fallback_price=5,
        auction2_fallback_price=10,
    )


def test_policy_rejects_invalid_tuning_ratios() -> None:
    with pytest.raises(ValueError, match="resource value"):
        SurplusPolicy(resource_value_numerator=-1)
    with pytest.raises(ValueError, match="resource value"):
        SurplusPolicy(resource_value_denominator=0)
    with pytest.raises(ValueError, match="objective value"):
        SurplusPolicy(objective_value_numerator=-1)
    with pytest.raises(ValueError, match="objective value"):
        SurplusPolicy(objective_value_denominator=0)
    with pytest.raises(ValueError, match="opponent objective"):
        SurplusPolicy(opponent_objective_numerator=-1)
    with pytest.raises(ValueError, match="opponent objective"):
        SurplusPolicy(opponent_objective_denominator=0)
    with pytest.raises(ValueError, match="market quantile"):
        SurplusPolicy(market_quantile_numerator=5)
    with pytest.raises(ValueError, match="objective progress"):
        SurplusPolicy(objective_progress_denominator=0)
    with pytest.raises(ValueError, match="objective reachability floor"):
        SurplusPolicy(objective_reachability_floor_numerator=2)
    with pytest.raises(ValueError, match="objective race discount"):
        SurplusPolicy(objective_race_discount_denominator=0)
    with pytest.raises(ValueError, match="fallback auction"):
        SurplusPolicy(auction1_fallback_price=-1)
    with pytest.raises(ValueError, match="loan opening fee"):
        SurplusPolicy(loan_opening_fee_denominator=0)


def test_v7_discounts_objectives_and_surrounding_resource_value() -> None:
    context = _context(
        resources=(1, 2),
        revealed=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 2),
            (0, 1, 1, 1, 2),
        ),
        objectives=(2,),
        won=((2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )

    v5 = SurplusV5Brain().choose_decision(context, canonical_knowledge(3))
    v6 = SurplusV6Brain().choose_decision(context, canonical_knowledge(3))
    v7 = SurplusV7Brain().choose_decision(context, canonical_knowledge(3))

    assert v5 == BotDecision.submit_bid(5)
    assert v6 == BotDecision.submit_bid(12)
    assert v7 == BotDecision.submit_bid(6)


def test_v7_identity_preserves_all_prior_generations() -> None:
    assert SURPLUS_V7_BOT_SPEC.name == "surplus-v7"
    assert SURPLUS_V7_BOT_SPEC.bot_id == "surplus-v7"
    assert type(SURPLUS_V7_BOT_SPEC.make_brain(seed=1)) is SurplusV7Brain
    assert type(SURPLUS_V6_BOT_SPEC.make_brain()) is SurplusV6Brain


def test_opponent_objective_threat_uses_public_rival_holdings() -> None:
    context = _context(
        resources=(1, 2),
        revealed=(
            (0, 0, 0, 0, 0),
            (0, 1, 1, 1, 2),
            (0, 1, 1, 1, 2),
        ),
        objectives=(1,),
        won=((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    threat_aware = SurplusBrain(
        SurplusPolicy(
            use_posterior_values=True,
            use_objective_values=True,
            use_opponent_objective_threat=True,
            bid_investments=True,
            use_market_prices=True,
            resource_value_numerator=13,
            resource_value_denominator=16,
            objective_value_numerator=3,
            objective_value_denominator=8,
            opponent_objective_numerator=1,
            opponent_objective_denominator=2,
        )
    )

    v7 = SurplusV7Brain().choose_decision(context, canonical_knowledge(3))
    with_threat = threat_aware.choose_decision(context, canonical_knowledge(3))

    assert v7 == BotDecision.submit_bid(4)
    assert with_threat == BotDecision.submit_bid(6)


def test_v8_uses_a_small_public_objective_denial_value() -> None:
    context = _context(
        resources=(1, 2),
        objectives=(1,),
        won=((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )

    assert SurplusV8Brain._opponent_objective_threat(context, (1,)) == 5
    assert SURPLUS_V8_POLICY.opponent_objective_numerator == 1
    assert SURPLUS_V8_POLICY.opponent_objective_denominator == 32


def test_v8_identity_preserves_all_prior_generations() -> None:
    assert SURPLUS_V8_BOT_SPEC.name == "surplus-v8"
    assert SURPLUS_V8_BOT_SPEC.bot_id == "surplus-v8"
    assert type(SURPLUS_V8_BOT_SPEC.make_brain(seed=1)) is SurplusV8Brain
    assert type(SURPLUS_V7_BOT_SPEC.make_brain()) is SurplusV7Brain


def test_v9_values_partial_objective_progress_without_double_counting_completion() -> None:
    context = _context(
        resources=(1, 2),
        objectives=(2,),
        won=((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )

    assert SurplusV9Brain._objective_progress_value(context, (1,)) > 0
    completing = _context(
        resources=(1, 2),
        objectives=(2,),
        won=((2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    assert SurplusV9Brain._objective_progress_value(completing, (1,)) == 0


def test_v9_preserves_cash_for_future_resources_before_buying_investments() -> None:
    context = _context(action_id=5, legal_max=6, cash=(6, 30, 30))

    assert SurplusV8Brain().choose_decision(
        context,
        canonical_knowledge(3),
    ) == BotDecision.submit_bid(5)
    assert (
        SurplusV9Brain().choose_decision(
            context,
            canonical_knowledge(3),
        )
        == BotDecision.pass_turn()
    )


def test_v9_borrows_only_while_future_resource_demand_exceeds_cash() -> None:
    low_cash = _context(action_id=3, legal_max=50, cash=(5, 30, 30))
    exhausted_resources = _context(
        action_id=3,
        legal_max=50,
        cash=(20, 30, 30),
        won=((6, 6, 3, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )

    assert SurplusV9Brain().choose_decision(
        low_cash,
        canonical_knowledge(3),
    ) == BotDecision.submit_bid(3)
    assert (
        SurplusV9Brain().choose_decision(
            exhausted_resources,
            canonical_knowledge(3),
        )
        == BotDecision.pass_turn()
    )


def test_v9_identity_preserves_all_prior_generations() -> None:
    assert SURPLUS_V9_BOT_SPEC.name == "surplus-v9"
    assert SURPLUS_V9_BOT_SPEC.bot_id == "surplus-v9"
    assert type(SURPLUS_V9_BOT_SPEC.make_brain(seed=1)) is SurplusV9Brain
    assert type(SURPLUS_V8_BOT_SPEC.make_brain()) is SurplusV8Brain


def test_v10_projects_future_spend_from_remaining_action_mix() -> None:
    context = _context(action_id=3, legal_max=50)

    projected = SurplusV10Brain()._projected_resource_spend(
        context,
        canonical_knowledge(3),
        (),
    )

    assert projected == 75


def test_v10_stops_borrowing_when_observed_resource_prices_make_cash_sufficient() -> None:
    context = _context(
        action_id=3,
        legal_max=50,
        cash=(10, 30, 30),
        won=((6, 6, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    history = (
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(1, 2),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(0, 1, 1),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=2,
            resource_ids=(2, 3),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(0, 2, 2),
        ),
    )

    assert SurplusV9Brain().choose_decision(
        context,
        canonical_knowledge(3),
        history,
    ) == BotDecision.submit_bid(3)
    assert (
        SurplusV10Brain().choose_decision(
            context,
            canonical_knowledge(3),
            history,
        )
        == BotDecision.pass_turn()
    )


def test_v10_identity_preserves_all_prior_generations() -> None:
    assert SURPLUS_V10_BOT_SPEC.name == "surplus-v10"
    assert SURPLUS_V10_BOT_SPEC.bot_id == "surplus-v10"
    assert type(SURPLUS_V10_BOT_SPEC.make_brain(seed=1)) is SurplusV10Brain
    assert type(SURPLUS_V9_BOT_SPEC.make_brain()) is SurplusV9Brain


def test_v11_uses_loan_proceeds_to_support_a_bid_with_no_cash() -> None:
    context = _context(action_id=3, legal_max=10, cash=(0, 30, 30))

    assert SurplusV11Brain().choose_decision(
        context,
        canonical_knowledge(3),
    ) == BotDecision.submit_bid(4)


def test_v11_reduces_loan_fee_when_only_a_small_shortfall_remains() -> None:
    brain = SurplusV11Brain()

    assert (
        brain._net_loan_reservation_bid(
            principal=10,
            cash=18,
            target_cash=20,
            legal_max=28,
            fee_cap=4,
        )
        == 1
    )
    assert (
        brain._net_loan_reservation_bid(
            principal=10,
            cash=5,
            target_cash=20,
            legal_max=15,
            fee_cap=4,
        )
        == 4
    )


def test_v11_identity_preserves_v10_and_advances_latest_alias() -> None:
    assert SURPLUS_V11_BOT_SPEC.name == "surplus-v11"
    assert SURPLUS_V11_BOT_SPEC.bot_id == "surplus-v11"
    assert type(SURPLUS_V11_BOT_SPEC.make_brain(seed=1)) is SurplusV11Brain
    assert type(SURPLUS_V10_BOT_SPEC.make_brain()) is SurplusV10Brain
    assert SURPLUS_V11_POLICY.use_net_loan_value is True
    assert SURPLUS_V10_POLICY.use_net_loan_value is False


def test_v12_zeros_progress_when_a_required_suit_cannot_remain_in_the_deck() -> None:
    exhausted = _context(
        resources=(1, 0),
        hand=(2, 2, 2, 2, 2),
        revealed=((0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0)),
        objectives=(21,),
    )
    available = _context(resources=(1, 0), objectives=(21,))
    brain = SurplusBrain(
        SurplusPolicy(
            use_objective_reachability=True,
            objective_reachability_floor_numerator=0,
        )
    )

    assert SurplusV11Brain._objective_progress_value(exhausted, (1,)) > 0
    assert (
        brain._reachable_objective_progress_value(
            exhausted,
            canonical_knowledge(3),
            (1,),
        )
        == 0
    )
    assert (
        brain._reachable_objective_progress_value(
            available,
            canonical_knowledge(3),
            (1,),
        )
        > 0
    )


def test_v12_discounts_progress_when_a_rival_is_already_closer() -> None:
    context = _context(
        resources=(1, 0),
        objectives=(2,),
        won=((0, 0, 0, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    no_race_discount = SurplusBrain(
        SurplusPolicy(
            use_objective_reachability=True,
            objective_reachability_floor_numerator=0,
        )
    )
    race_aware = SurplusBrain(
        SurplusPolicy(
            use_objective_reachability=True,
            objective_reachability_floor_numerator=0,
            objective_race_discount_numerator=1,
        )
    )

    baseline = no_race_discount._reachable_objective_progress_value(
        context,
        canonical_knowledge(3),
        (1,),
    )
    discounted = race_aware._reachable_objective_progress_value(
        context,
        canonical_knowledge(3),
        (1,),
    )

    assert discounted == baseline / 2


def test_v12_identity_preserves_v11_and_advances_latest_alias() -> None:
    assert SURPLUS_V12_BOT_SPEC.name == "surplus-v12"
    assert SURPLUS_V12_BOT_SPEC.bot_id == "surplus-v12"
    assert type(SURPLUS_V12_BOT_SPEC.make_brain(seed=1)) is SurplusV12Brain
    assert type(SURPLUS_V11_BOT_SPEC.make_brain()) is SurplusV11Brain
    assert type(SURPLUS_BOT_SPEC.make_brain()) is SurplusV12Brain
    assert SURPLUS_V12_POLICY.use_objective_reachability is True
    assert SURPLUS_V12_POLICY.objective_reachability_floor_numerator == 1
    assert SURPLUS_V12_POLICY.objective_reachability_floor_denominator == 2
    assert SURPLUS_V12_POLICY.objective_race_discount_numerator == 0
    assert SURPLUS_V12_POLICY.objective_reserve_release_numerator == 5
    assert SURPLUS_V12_POLICY.objective_reserve_release_denominator == 8
    assert SURPLUS_V11_POLICY.use_objective_reachability is False
