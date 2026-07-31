from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from pocketrocks import DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.hybrid.experts import ExpertAvailability, load_promoted_experts
from garboid_pocketrocks.hybrid.selector import (
    LiveSelectorInput,
    SelectorInputRejected,
    choose_promoted_expert,
)
from garboid_pocketrocks.knowledge import knowledge_for_context


def _context() -> DecisionContext:
    return DecisionContext(
        request_id="private-request-id",
        deadline_at=123_456,
        received_at=123_000,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(1, 2, 3, 10),
        current_action_id=1,
        current_resource_ids=(3, 0),
        cash_by_seat=(22, 30, 18),
        tiebreak_seat=1,
        won_resource_counts_by_seat=((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0)),
        revealed_info_counts_by_seat=((0, 1, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0, 0)),
        owned_objective_ids_by_seat=((1,), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(2, 5, 1, 3),
        legal_max_amount=7,
        revealable_count=4,
        metadata={"engine_rng_state": "must not reach selector"},
    )


def _history(
    *,
    starting_cash: int = 30,
    objective_ids: tuple[int, ...] = (1, 2, 3, 10),
) -> PublicHistory:
    return (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=starting_cash,
            value_chart=(0, 4, 8, 12, 16, 20),
            initial_tiebreak_seat=1,
            objective_ids=objective_ids,
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(3, 0),
        ),
    )


def _selector_input() -> LiveSelectorInput:
    context = _context()
    return LiveSelectorInput.from_live_state(context, knowledge_for_context(context), _history())


def _available(*names: str) -> dict[str, ExpertAvailability]:
    return {
        name: ExpertAvailability(expert_name=name, available=True, reason="available")
        for name in names
    }


class _FixedSelector:
    def __init__(self, name: str) -> None:
        self.name = name

    def choose_expert(
        self, selector_input: LiveSelectorInput, eligible_expert_names: tuple[str, ...]
    ) -> str:
        del selector_input, eligible_expert_names
        return self.name


class _RejectingSelector:
    def choose_expert(
        self, selector_input: LiveSelectorInput, eligible_expert_names: tuple[str, ...]
    ) -> str:
        del selector_input, eligible_expert_names
        raise SelectorInputRejected("unsupported public condition")


def test_live_selector_input_copies_only_live_compatible_fields() -> None:
    context = _context()
    first = LiveSelectorInput.from_live_state(context, knowledge_for_context(context), _history())
    private_transport_changed = replace(
        context,
        request_id="other-request",
        deadline_at=999_999,
        received_at=888_888,
        metadata={"opponent_hands": ((5, 5),), "deck_order": (5, 4, 3)},
    )
    same = LiveSelectorInput.from_live_state(
        private_transport_changed,
        knowledge_for_context(private_transport_changed),
        _history(),
    )

    assert same == first
    assert first.own_hand_suit_ids == (2, 5, 1, 3)
    assert not hasattr(first.context, "metadata")
    assert not hasattr(first.context, "request_id")
    assert not hasattr(first.context, "current_hand_suit_ids")
    assert replace(first, own_hand_suit_ids=(1, 5, 1, 3)) != first


def test_live_selector_input_rejects_ruleset_name_or_objective_tampering() -> None:
    context = _context()
    canonical = knowledge_for_context(context)

    with pytest.raises(ValueError, match="canonical live ruleset"):
        LiveSelectorInput.from_live_state(
            context,
            replace(canonical, name="live-B"),
            _history(),
        )

    with pytest.raises(ValueError, match="canonical live ruleset"):
        LiveSelectorInput.from_live_state(
            context,
            replace(canonical, active_objective_count=1),
            _history(),
        )

    unknown_objective = replace(context, objective_ids=(1, 2, 3, 999))
    with pytest.raises(ValueError, match="invalid objective"):
        LiveSelectorInput.from_live_state(
            unknown_objective,
            knowledge_for_context(unknown_objective),
            _history(),
        )


def test_coherent_context_and_ruleset_tampering_cannot_redefine_sdk_constants() -> None:
    context = _context()

    changed_cash = replace(context, starting_cash=999)
    with pytest.raises(ValueError, match="starting cash"):
        LiveSelectorInput.from_live_state(
            changed_cash,
            knowledge_for_context(changed_cash),
            _history(starting_cash=999),
        )

    four_private_cards = replace(
        context,
        current_hand_suit_ids=(2, 5, 1),
        revealable_count=3,
    )
    with pytest.raises(ValueError, match="private-card total"):
        LiveSelectorInput.from_live_state(
            four_private_cards,
            knowledge_for_context(four_private_cards),
            _history(),
        )

    one_objective = replace(
        context,
        objective_ids=(1,),
        owned_objective_ids_by_seat=((1,), (), ()),
    )
    with pytest.raises(ValueError, match="wrong number of active objectives"):
        LiveSelectorInput.from_live_state(
            one_objective,
            knowledge_for_context(one_objective),
            _history(objective_ids=(1,)),
        )


def test_live_selector_input_rejects_invalid_hand_and_public_count_matrices() -> None:
    context = _context()
    invalid_hand = replace(context, current_hand_suit_ids=(2, 5, 1, 6))
    with pytest.raises(ValueError, match="hand contains an invalid suit"):
        LiveSelectorInput.from_live_state(
            invalid_hand,
            knowledge_for_context(invalid_hand),
            _history(),
        )

    negative_public_count = replace(
        context,
        revealed_info_counts_by_seat=(
            context.revealed_info_counts_by_seat[0],
            (-1, 0, 0, 0, 0),
            context.revealed_info_counts_by_seat[2],
        ),
    )
    with pytest.raises(ValueError, match="nonnegative integers"):
        LiveSelectorInput.from_live_state(
            negative_public_count,
            knowledge_for_context(negative_public_count),
            _history(),
        )

    too_many_resources = replace(
        context,
        won_resource_counts_by_seat=(
            (7, 0, 0, 0, 0),
            context.won_resource_counts_by_seat[1],
            context.won_resource_counts_by_seat[2],
        ),
    )
    with pytest.raises(ValueError, match="more resource cards"):
        LiveSelectorInput.from_live_state(
            too_many_resources,
            knowledge_for_context(too_many_resources),
            _history(),
        )


def test_fixed_input_reproduces_the_same_expert_choice() -> None:
    experts = load_promoted_experts()
    availability = _available(*(expert.name for expert in experts))
    selector = _FixedSelector("balanced-v3")

    first = choose_promoted_expert(selector, _selector_input(), experts, availability)
    second = choose_promoted_expert(selector, _selector_input(), experts, availability)

    assert first == second
    assert first.selected_expert_name == "balanced-v3"
    assert first.used_fallback is False
    assert first.fallback_reason == "none"


def test_unavailable_choice_falls_back_in_catalog_order_with_diagnostics() -> None:
    experts = load_promoted_experts()
    availability = _available("balanced-v3", "passive-v3")
    availability["vector_ppo_large_v1_g350k"] = ExpertAvailability(
        expert_name="vector_ppo_large_v1_g350k",
        available=False,
        reason="runtime_dependency_missing",
        detail="torch is not installed",
    )

    result = choose_promoted_expert(
        _FixedSelector("vector_ppo_large_v1_g350k"),
        _selector_input(),
        experts,
        availability,
    )

    assert result.requested_expert_name == "vector_ppo_large_v1_g350k"
    assert result.selected_expert_name == "balanced-v3"
    assert result.used_fallback is True
    assert result.fallback_reason == "requested_expert_unavailable"
    assert tuple(item.expert_name for item in result.unavailable_experts) == (
        "aggressive-v3",
        "vector_ppo_large_v1_g350k",
    )


def test_rejected_or_ineligible_choice_uses_the_same_deterministic_fallback() -> None:
    experts = load_promoted_experts()
    availability = _available("passive-v3")

    rejected = choose_promoted_expert(
        _RejectingSelector(), _selector_input(), experts, availability
    )
    ineligible = choose_promoted_expert(
        _FixedSelector("aggressive"), _selector_input(), experts, availability
    )

    assert rejected.selected_expert_name == "passive-v3"
    assert rejected.fallback_reason == "selector_rejected_input"
    assert ineligible.selected_expert_name == "passive-v3"
    assert ineligible.fallback_reason == "selector_returned_ineligible_expert"


def test_no_available_expert_is_explicit_and_reproducible() -> None:
    experts = load_promoted_experts()

    first = choose_promoted_expert(_FixedSelector("aggressive-v3"), _selector_input(), experts, {})
    second = choose_promoted_expert(_FixedSelector("aggressive-v3"), _selector_input(), experts, {})

    assert first == second
    assert first.selected_expert_name is None
    assert first.used_fallback is True
    assert first.fallback_reason == "no_available_expert"
    assert tuple(item.expert_name for item in first.unavailable_experts) == tuple(
        expert.name for expert in experts
    )
    assert all(item.reason == "availability_not_reported" for item in first.unavailable_experts)


def test_unknown_availability_identity_is_rejected() -> None:
    experts = load_promoted_experts()

    try:
        choose_promoted_expert(
            _FixedSelector("aggressive-v3"),
            _selector_input(),
            experts,
            _available("latest"),
        )
    except ValueError as error:
        assert "unknown expert" in str(error)
    else:
        raise AssertionError("unknown availability identity was accepted")


def test_selection_rejects_an_in_memory_forged_expert_roster() -> None:
    catalog = load_promoted_experts()
    forged = replace(catalog[0], name="aggressive")
    forged_roster = (forged, *tuple(catalog)[1:])

    with pytest.raises(TypeError, match="verified catalog"):
        choose_promoted_expert(
            _FixedSelector("aggressive-v3"),
            _selector_input(),
            cast(Any, forged_roster),
            _available("aggressive-v3"),
        )
