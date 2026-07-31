from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from pocketrocks import ActionId, BotDecision, DecisionContext, Suit
from pocketrocks.internal.bot_wire_v2 import DecisionRequest, decode_frame
from pocketrocks.protocol import build_decision_context
from pocketrocks.testing import scenario

from garboid_pocketrocks.adapters.public_history import (
    PublicGameSetup,
    PublicHistory,
    public_history_from_sdk_events,
    public_history_from_sdk_frame,
)
from garboid_pocketrocks.hybrid.experts import ExpertAvailability, load_promoted_experts
from garboid_pocketrocks.hybrid.selector import (
    LiveSelectorInput,
    SelectorInputRejected,
    choose_promoted_expert,
)
from garboid_pocketrocks.knowledge import knowledge_for_context
from garboid_pocketrocks.simulator.session import SdkGameSession


def _decoded_selector_frame(*, reveal_decision: bool = False) -> DecisionRequest:
    narration = (
        scenario(
            players=3,
            starting_cash=30,
            initial_tiebreak_seat=1,
            objective_ids=(1, 2, 3, 10),
        )
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.WOOD))
        .auction((8, 0, 0))
    )
    if reveal_decision:
        narration.deciding(
            seat=0,
            hand=(Suit.ORE, Suit.SHEEP, Suit.WOOD, Suit.BRICK, Suit.WHEAT),
            kind="selectInfoToReveal",
        )
    else:
        narration.reveal(Suit.ORE).turn(
            ActionId.LOAN10,
            resources=(Suit.WOOD, Suit.WHEAT),
        ).deciding(
            seat=0,
            hand=(Suit.SHEEP, Suit.WOOD, Suit.BRICK, Suit.WHEAT),
        )
    frame = decode_frame(narration.to_bytes(deadline_at=123_456))
    if not isinstance(frame, DecisionRequest):
        raise AssertionError("selector fixture must decode to a decision request")
    return frame


def _context() -> DecisionContext:
    context = build_decision_context(_decoded_selector_frame(), received_at=123_000)
    return replace(context, metadata={"engine_rng_state": "must not reach selector"})


def _history(
    *,
    starting_cash: int = 30,
    objective_ids: tuple[int, ...] = (1, 2, 3, 10),
) -> PublicHistory:
    history = public_history_from_sdk_frame(_decoded_selector_frame())
    setup = history[0]
    if not isinstance(setup, PublicGameSetup):
        raise AssertionError("selector fixture history must begin with game setup")
    return (
        replace(setup, starting_cash=starting_cash, objective_ids=objective_ids),
        *history[1:],
    )


def _selector_input() -> LiveSelectorInput:
    context = _context()
    return LiveSelectorInput.from_live_state(context, knowledge_for_context(context), _history())


def _reveal_context_and_history() -> tuple[DecisionContext, PublicHistory]:
    frame = _decoded_selector_frame(reveal_decision=True)
    return (
        build_decision_context(frame, received_at=123_000),
        public_history_from_sdk_frame(frame),
    )


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
    assert first.own_hand_suit_ids == (4, 2, 1, 5)
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


def test_selector_input_enforces_sdk_phase_fields_and_latest_public_turn() -> None:
    context = _context()
    mismatched_revealable_count = replace(context, revealable_count=3)
    with pytest.raises(ValueError, match="equal the focal hand length"):
        LiveSelectorInput.from_live_state(
            mismatched_revealable_count,
            knowledge_for_context(mismatched_revealable_count),
            _history(),
        )

    bid_without_limit = replace(context, legal_max_amount=None)
    with pytest.raises(ValueError, match="legal bid maximum"):
        LiveSelectorInput.from_live_state(
            bid_without_limit,
            knowledge_for_context(bid_without_limit),
            _history(),
        )

    mismatched_action = replace(context, current_action_id=2)
    with pytest.raises(ValueError, match="latest public turn"):
        LiveSelectorInput.from_live_state(
            mismatched_action,
            knowledge_for_context(mismatched_action),
            _history(),
        )

    malformed_resources = replace(
        context,
        current_resource_ids=(3,),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="current resources"):
        LiveSelectorInput.from_live_state(
            malformed_resources,
            knowledge_for_context(malformed_resources),
            _history(),
        )

    reveal_context, reveal_history = _reveal_context_and_history()
    accepted = LiveSelectorInput.from_live_state(
        reveal_context,
        knowledge_for_context(reveal_context),
        reveal_history,
    )
    assert accepted.context.decision_kind == "selectInfoToReveal"

    reveal_with_bid_limit = replace(reveal_context, legal_max_amount=7)
    with pytest.raises(ValueError, match="reveal decision cannot"):
        LiveSelectorInput.from_live_state(
            reveal_with_bid_limit,
            knowledge_for_context(reveal_with_bid_limit),
            reveal_history,
        )

    bid_after_resolution = replace(
        reveal_context,
        decision_kind="submitBid",
        legal_max_amount=reveal_context.cash_by_seat[reveal_context.bot_seat],
    )
    with pytest.raises(ValueError, match="bid decision requires one unresolved"):
        LiveSelectorInput.from_live_state(
            bid_after_resolution,
            knowledge_for_context(bid_after_resolution),
            reveal_history,
        )


@pytest.mark.parametrize("value_chart", tuple("ABCDE"))
@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("objectives_enabled", (False, True))
def test_real_sdk_opening_contexts_are_accepted_across_supported_games(
    value_chart: str,
    player_count: int,
    objectives_enabled: bool,
) -> None:
    session = SdkGameSession.start(
        player_count=player_count,
        seed=f"hybrid-input-{value_chart}-{player_count}-{objectives_enabled}",
        value_chart=value_chart,
        objectives_enabled=objectives_enabled,
    )
    _seat, context = session.pending.contexts[0]
    history = public_history_from_sdk_events(session.events)

    selector_input = LiveSelectorInput.from_live_state(
        context,
        knowledge_for_context(context),
        history,
    )

    assert selector_input.context.player_count == player_count
    assert selector_input.ruleset_name.startswith(f"live-{value_chart}")
    assert bool(selector_input.context.objective_ids) is objectives_enabled


def test_opening_context_must_match_the_replayed_public_snapshot() -> None:
    session = SdkGameSession.start(player_count=3, seed="selector-opening-mismatch")
    _seat, context = session.pending.contexts[0]
    history = public_history_from_sdk_events(session.events)
    won = [list(row) for row in context.won_resource_counts_by_seat]
    won[1][0] = 1
    revealed = [list(row) for row in context.revealed_info_counts_by_seat]
    revealed[1][0] = 1
    owned = [list(row) for row in context.owned_objective_ids_by_seat]
    owned[1].append(context.objective_ids[0])
    mismatches = (
        (
            "cash",
            replace(
                context,
                cash_by_seat=(context.cash_by_seat[0] - 1, *context.cash_by_seat[1:]),
            ),
        ),
        ("won resources", replace(context, won_resource_counts_by_seat=tuple(map(tuple, won)))),
        (
            "revealed information",
            replace(context, revealed_info_counts_by_seat=tuple(map(tuple, revealed))),
        ),
        (
            "owned objectives",
            replace(context, owned_objective_ids_by_seat=tuple(map(tuple, owned))),
        ),
        (
            "legal bid maximum",
            replace(context, legal_max_amount=cast(int, context.legal_max_amount) - 1),
        ),
    )

    for field_name, malformed_context in mismatches:
        with pytest.raises(ValueError, match=field_name):
            LiveSelectorInput.from_live_state(
                malformed_context,
                knowledge_for_context(malformed_context),
                history,
            )


def test_selector_rejects_correlated_known_resources_that_exceed_one_sdk_deck() -> None:
    frame = decode_frame(
        scenario(players=3, starting_cash=30)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, Suit.BRICK))
        .deciding(seat=0, hand=(Suit.BRICK,) * 5)
        .to_bytes(deadline_at=123_456)
    )
    if not isinstance(frame, DecisionRequest):
        raise AssertionError("correlated resource fixture must be a decision request")
    context = build_decision_context(frame, received_at=123_000)
    history = public_history_from_sdk_frame(frame)

    with pytest.raises(ValueError, match="combined known resources"):
        LiveSelectorInput.from_live_state(context, knowledge_for_context(context), history)


@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_every_live_context_in_complete_sdk_games_matches_public_replay(
    player_count: int,
) -> None:
    session = SdkGameSession.start(
        player_count=player_count,
        seed=f"complete-selector-replay-{player_count}",
    )
    accepted_contexts = 0

    while not session.terminated:
        history = public_history_from_sdk_events(session.events)
        for _seat, context in session.pending.contexts:
            LiveSelectorInput.from_live_state(
                context,
                knowledge_for_context(context),
                history,
            )
            accepted_contexts += 1
        decisions = {
            seat: (
                BotDecision.submit_bid(1)
                if session.pending.decision_kind == "submitBid"
                and seat == 0
                and cast(int, dict(session.pending.contexts)[seat].legal_max_amount) >= 1
                else BotDecision.pass_turn()
            )
            for seat in session.pending.acting_seats
        }
        session.step(decisions)

    assert accepted_contexts > player_count


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
