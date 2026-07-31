from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, fields, replace

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicEventKind,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.knowledge import knowledge_for_context
from garboid_pocketrocks.search.public_belief import (
    LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
    PublicSearchPosition,
    SampledWorld,
    reconstruct_public_search_position,
    sample_compatible_worlds,
)
from garboid_pocketrocks.simulator.session import SdkGameSession


def _reconstruct(
    context: DecisionContext,
    history: PublicHistory,
) -> PublicSearchPosition:
    return reconstruct_public_search_position(
        context,
        knowledge_for_context(context),
        history,
    )


def _first_position(
    *, player_count: int = 3, seed: int = 42
) -> tuple[
    SdkGameSession,
    DecisionContext,
    PublicHistory,
    PublicSearchPosition,
]:
    session = SdkGameSession.start(player_count=player_count, seed=seed)
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    return session, context, history, _reconstruct(context, history)


def _assert_world_conserves(
    context: DecisionContext,
    history: PublicHistory,
    position: PublicSearchPosition,
    world: SampledWorld,
) -> None:
    ruleset = knowledge_for_context(context)
    expected_resources = Counter(
        suit_id
        for suit_id, total in enumerate(ruleset.resource_counts, start=1)
        for _ in range(total)
    )
    visible = (
        [suit_id for suit_id in context.current_resource_ids if suit_id]
        if context.decision_kind == "submitBid"
        else []
    )
    public_cards = (
        [
            suit_id
            for row in context.won_resource_counts_by_seat
            for suit_id, count in enumerate(row, start=1)
            for _ in range(count)
        ]
        + [
            suit_id
            for row in context.revealed_info_counts_by_seat
            for suit_id, count in enumerate(row, start=1)
            for _ in range(count)
        ]
        + visible
    )
    sampled_cards = [suit_id for hand in world.hand_suits_by_seat for suit_id in hand] + list(
        world.future_resource_suits
    )
    assert Counter(public_cards + sampled_cards) == expected_resources

    expected_actions = Counter(
        action_id
        for action_id, total in enumerate(ruleset.action_counts, start=1)
        for _ in range(total)
    )
    public_actions = [event.action_id for event in history if isinstance(event, PublicTurnOpened)]
    assert Counter(public_actions + list(world.future_action_ids)) == expected_actions

    for suit_index, suit_belief in enumerate(position.belief.suits):
        opponent_count = sum(
            hand.count(suit_index + 1)
            for seat, hand in enumerate(world.hand_suits_by_seat)
            if seat != context.bot_seat
        )
        terminal_bucket = min(
            suit_belief.known_terminal_reveals + opponent_count,
            len(suit_belief.terminal_price_pmf) - 1,
        )
        assert suit_belief.terminal_price_pmf[terminal_bucket] > 0.0


def test_candidate_identity_is_new_development_only_and_not_registered() -> None:
    assert LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY.endswith("-dev")
    assert LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY not in BOT_SPECS_BY_NAME
    assert "balanced-v3" in BOT_SPECS_BY_NAME


def test_public_position_is_a_closed_immutable_allowlist() -> None:
    names = {field.name for field in fields(PublicSearchPosition)}

    assert names == {
        "ruleset_name",
        "player_count",
        "starting_cash",
        "value_chart",
        "objective_ids",
        "bot_seat",
        "decision_kind",
        "current_action_id",
        "current_resource_ids",
        "cash_by_seat",
        "tiebreak_seat",
        "won_resource_counts_by_seat",
        "revealed_info_counts_by_seat",
        "owned_objective_ids_by_seat",
        "current_hand_suit_ids",
        "legal_max_amount",
        "loan_principal_by_seat",
        "investment_value_by_seat",
        "resolved_turn_count",
        "remaining_action_counts",
        "unseen_resource_counts",
        "opponent_hidden_slots_by_seat",
        "belief",
        "canonical_input_digest",
    }
    assert not names & {
        "seed",
        "deck",
        "opponent_hands",
        "engine",
        "snapshot",
        "metadata",
        "deadline_at",
        "request_id",
    }
    assert hasattr(PublicSearchPosition, "__slots__")


@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("chart", ("A", "B", "C", "D", "E"))
def test_every_live_sdk_policy_input_reconstructs_for_a_complete_game(
    player_count: int,
    chart: str,
) -> None:
    session = SdkGameSession.start(
        player_count=player_count,
        seed=f"public-belief-{player_count}-{chart}",
        value_chart=chart,
    )
    visited_kinds: set[str] = set()
    while not session.terminated:
        history = public_history_from_sdk_events(session.events)
        for _seat, context in session.pending.contexts:
            position = _reconstruct(context, history)
            visited_kinds.add(position.decision_kind)
            assert position.cash_by_seat == context.cash_by_seat
            assert position.current_hand_suit_ids == context.current_hand_suit_ids
            world = sample_compatible_worlds(
                position,
                candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
                sample_count=1,
            )[0]
            _assert_world_conserves(context, history, position, world)

        if session.pending.decision_kind == "submitBid":
            decisions = {
                seat: BotDecision.submit_bid(
                    min((session.snapshot.turn_index + seat) % 7, context.legal_max_amount or 0)
                )
                for seat, context in session.pending.contexts
            }
        else:
            reveal_seat = session.pending.acting_seats[0]
            decisions = {reveal_seat: BotDecision.select_info_to_reveal(0)}
        session.step(decisions)

    assert "submitBid" in visited_kinds
    assert "selectInfoToReveal" in visited_kinds


def test_samples_exactly_conserve_cards_actions_and_hidden_slots() -> None:
    _session, context, _history, position = _first_position(player_count=5)
    worlds = sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=16,
    )

    for world in worlds:
        assert isinstance(world, SampledWorld)
        _assert_world_conserves(context, _history, position, world)
        assert world.hand_suits_by_seat[context.bot_seat] == context.current_hand_suit_ids
        for seat, slots in enumerate(position.opponent_hidden_slots_by_seat):
            if seat != context.bot_seat:
                assert len(world.hand_suits_by_seat[seat]) == slots


def test_sampling_is_reproducible_prefix_stable_and_identity_bound() -> None:
    _session, _context, _history, position = _first_position()
    short = sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=3,
    )
    long = sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=8,
    )

    assert short == long[:3]
    assert short == sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=3,
    )
    assert len({world.hand_suits_by_seat for world in long}) > 1
    assert tuple(world.sample_index for world in long) == tuple(range(8))
    assert (
        position.canonical_input_digest
        == "bc24b6b9a3176b47d1ecaa6da832b11998b536a5941b89c780b698725d732183"
    )
    encoded_world = json.dumps(asdict(short[0]), sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(encoded_world).hexdigest()
        == "4fef75ef0026a162c12c2cff68ac6db38bba41c6429c98167a66bdac0d4b153e"
    )
    with pytest.raises(ValueError, match="explicit development candidate identity"):
        sample_compatible_worlds(
            position,
            candidate_identity="balanced-v3",
            sample_count=1,
        )


def test_sampled_terminal_counts_match_existing_bayesian_marginals() -> None:
    _session, context, _history, position = _first_position(seed=8_191)
    worlds = sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=8_192,
    )

    for suit_index, suit_belief in enumerate(position.belief.suits):
        observed_buckets = Counter(
            min(
                suit_belief.known_terminal_reveals
                + sum(
                    hand.count(suit_index + 1)
                    for seat, hand in enumerate(world.hand_suits_by_seat)
                    if seat != context.bot_seat
                ),
                len(suit_belief.terminal_price_pmf) - 1,
            )
            for world in worlds
        )
        empirical = tuple(
            observed_buckets[bucket] / len(worlds)
            for bucket in range(len(suit_belief.terminal_price_pmf))
        )
        assert empirical == pytest.approx(suit_belief.terminal_price_pmf, abs=0.025)


def test_transport_bookkeeping_and_metadata_cannot_change_samples() -> None:
    _session, context, history, position = _first_position()
    changed = replace(
        context,
        request_id="different-request",
        deadline_at=123,
        received_at=122,
        metadata={"private_seed": "must-not-enter-search"},
    )
    changed_position = _reconstruct(changed, history)

    assert changed_position.canonical_input_digest == position.canonical_input_digest
    assert sample_compatible_worlds(
        changed_position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=4,
    ) == sample_compatible_worlds(
        position,
        candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
        sample_count=4,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("cash_by_seat", (29, 30, 30), "cash"),
        ("tiebreak_seat", 99, "tiebreak"),
        ("current_resource_ids", (5, 5), "resources"),
        (
            "won_resource_counts_by_seat",
            ((1, 0, 0, 0, 0),) + ((0, 0, 0, 0, 0),) * 2,
            "won resources",
        ),
    ),
)
def test_context_that_contradicts_public_history_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    _session, context, history, _position = _first_position()

    with pytest.raises(HeuristicInputError, match=message):
        _reconstruct(replace(context, **{field: value}), history)  # type: ignore[arg-type]


def test_history_that_skips_or_changes_a_required_reveal_is_rejected() -> None:
    session = SdkGameSession.start(player_count=3, seed=91)
    session.step({seat: BotDecision.pass_turn() for seat in session.pending.acting_seats})
    assert session.pending.decision_kind == "selectInfoToReveal"
    winner = session.pending.acting_seats[0]
    session.step({winner: BotDecision.select_info_to_reveal(0)})
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    reveal_index = next(
        index for index, event in enumerate(history) if isinstance(event, PublicInformationRevealed)
    )
    skipped = history[:reveal_index] + history[reveal_index + 1 :]
    reveal = history[reveal_index]
    assert isinstance(reveal, PublicInformationRevealed)
    changed = (
        history[:reveal_index]
        + (
            PublicInformationRevealed(
                kind=PublicEventKind.INFORMATION_REVEALED,
                seat=(reveal.seat + 1) % context.player_count,
                suit_id=reveal.suit_id,
            ),
        )
        + history[reveal_index + 1 :]
    )

    with pytest.raises(HeuristicInputError, match="required information reveal"):
        _reconstruct(context, skipped)
    with pytest.raises(HeuristicInputError, match="auction winner"):
        _reconstruct(context, changed)


def test_noncanonical_ruleset_and_invalid_sample_counts_fail_closed() -> None:
    _session, context, history, position = _first_position()
    ruleset = knowledge_for_context(context)
    with pytest.raises(HeuristicInputError, match="not canonical"):
        reconstruct_public_search_position(
            context,
            replace(ruleset, resource_counts=(7, 6, 6, 6, 5)),
            history,
        )
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            sample_compatible_worlds(
                position,
                candidate_identity=LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY,
                sample_count=invalid,  # type: ignore[arg-type]
            )
