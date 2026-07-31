"""Reconstruct and sample late-game worlds from deployable bot inputs only.

This module deliberately does not advance sampled worlds. The pinned SDK owns
game transitions, but does not yet expose a supported restore/fork API. A
future search policy can consume these samples only after that SDK boundary
exists; strategy callbacks must continue to receive seat-specific public
inputs rather than ``SampledWorld`` objects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pocketrocks import ActionId, DecisionContext, Suit
from pocketrocks.sim.constants import (
    ACTION_WIRE_IDS,
    INVEST_PAYOUT,
    LOAN_PRINCIPAL,
    objective_pattern_met,
)

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.belief import BeliefState, build_belief
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.knowledge import RulesetKnowledge, knowledge_for_context

LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY = "late-game-public-belief-v1-dev"
"""Development-only identity; it is intentionally absent from bot registries."""

_SAMPLING_ALGORITHM = "sha256-ranked-finite-population-v1"
_FIXED_SEARCH_SEED = 0
_SUIT_COUNT = len(Suit)
_ACTION_COUNT = len(ActionId)
_AUCTION_ONE = int(ActionId.AUCTION1)
_AUCTION_TWO = int(ActionId.AUCTION2)
_ACTION_NAME_BY_ID = {wire_id: name for name, wire_id in ACTION_WIRE_IDS.items()}
_ACTION_CREDIT_BY_ID = {
    wire_id: LOAN_PRINCIPAL.get(name, 0) for name, wire_id in ACTION_WIRE_IDS.items()
}


@dataclass(frozen=True, slots=True)
class PublicSearchPosition:
    """A canonical public position proven consistent with its public history."""

    ruleset_name: str
    player_count: int
    starting_cash: int
    value_chart: tuple[int, ...]
    objective_ids: tuple[int, ...]
    bot_seat: int
    decision_kind: Literal["submitBid", "selectInfoToReveal"]
    current_action_id: int
    current_resource_ids: tuple[int, int]
    cash_by_seat: tuple[int, ...]
    tiebreak_seat: int
    won_resource_counts_by_seat: tuple[tuple[int, ...], ...]
    revealed_info_counts_by_seat: tuple[tuple[int, ...], ...]
    owned_objective_ids_by_seat: tuple[tuple[int, ...], ...]
    current_hand_suit_ids: tuple[int, ...]
    legal_max_amount: int | None
    loan_principal_by_seat: tuple[int, ...]
    investment_value_by_seat: tuple[int, ...]
    resolved_turn_count: int
    remaining_action_counts: tuple[int, ...]
    unseen_resource_counts: tuple[int, ...]
    opponent_hidden_slots_by_seat: tuple[int, ...]
    belief: BeliefState
    canonical_input_digest: str


@dataclass(frozen=True, slots=True)
class SampledWorld:
    """One deterministic completion of information hidden from a live bot.

    This is search-backend data, never a policy callback input. The focal row
    contains the bot's real hand; every other hand and both future orders are
    sampled solely from the canonical public position.
    """

    candidate_identity: str
    canonical_input_digest: str
    sampling_algorithm: str
    search_seed: int
    sample_index: int
    hand_suits_by_seat: tuple[tuple[int, ...], ...]
    future_resource_suits: tuple[int, ...]
    future_action_ids: tuple[int, ...]


@dataclass(slots=True)
class _ReplayState:
    cash: list[int]
    won: list[list[int]]
    revealed: list[list[int]]
    owned: list[list[int]]
    loans: list[int]
    investments: list[int]
    tiebreak_seat: int
    current_turn: PublicTurnOpened | None = None
    latest_turn: PublicTurnOpened | None = None
    required_reveal_seat: int | None = None
    resolved_turn_count: int = 0
    previous_resources: tuple[int, int] | None = None
    previous_action_id: int | None = None


def reconstruct_public_search_position(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    history: PublicHistory,
) -> PublicSearchPosition:
    """Validate and canonicalize exactly the information a live bot can know."""

    try:
        canonical_ruleset = knowledge_for_context(context)
    except ValueError as error:
        raise HeuristicInputError(str(error)) from error
    if ruleset != canonical_ruleset:
        raise HeuristicInputError("ruleset knowledge is not canonical for the public context")
    belief = build_belief(context, ruleset)
    setup, replay, seen_actions = _replay_history(history, ruleset)
    _validate_replay_against_context(context, replay, setup, ruleset)

    remaining_action_counts = tuple(
        total - seen for total, seen in zip(ruleset.action_counts, seen_actions, strict=True)
    )
    if any(count < 0 for count in remaining_action_counts):
        raise HeuristicInputError("public history uses more actions than the ruleset contains")

    # During a reveal request the latest turn's resources have already moved
    # into the winner's public holdings. They remain in the SDK context as the
    # latest turn description, so count them as visible only before bids resolve.
    visible_counts = (
        _count_suits(context.current_resource_ids)
        if context.decision_kind == "submitBid"
        else (0,) * _SUIT_COUNT
    )
    known_counts = tuple(
        sum(row[suit_index] for row in context.won_resource_counts_by_seat)
        + sum(row[suit_index] for row in context.revealed_info_counts_by_seat)
        + context.current_hand_suit_ids.count(suit_index + 1)
        + visible_counts[suit_index]
        for suit_index in range(_SUIT_COUNT)
    )
    unseen_resource_counts = tuple(
        total - known for total, known in zip(ruleset.resource_counts, known_counts, strict=True)
    )
    if any(count < 0 for count in unseen_resource_counts):
        raise HeuristicInputError("publicly known cards exceed the finite resource deck")

    opponent_hidden_slots_by_seat = tuple(
        0
        if seat == context.bot_seat
        else ruleset.private_cards_per_player - sum(context.revealed_info_counts_by_seat[seat])
        for seat in range(context.player_count)
    )
    if any(slots < 0 for slots in opponent_hidden_slots_by_seat):
        raise HeuristicInputError("public reveals exceed an opponent's initial hand")
    opponent_slots = sum(opponent_hidden_slots_by_seat)
    if opponent_slots > sum(unseen_resource_counts):
        raise HeuristicInputError("opponent hidden slots exceed the unseen resource population")

    decision_kind = context.decision_kind
    canonical_fields: dict[str, object] = {
        "ruleset_name": ruleset.name,
        "player_count": context.player_count,
        "starting_cash": context.starting_cash,
        "value_chart": context.value_chart,
        "objective_ids": context.objective_ids,
        "bot_seat": context.bot_seat,
        "decision_kind": decision_kind,
        "current_action_id": context.current_action_id,
        "current_resource_ids": context.current_resource_ids,
        "cash_by_seat": context.cash_by_seat,
        "tiebreak_seat": context.tiebreak_seat,
        "won_resource_counts_by_seat": context.won_resource_counts_by_seat,
        "revealed_info_counts_by_seat": context.revealed_info_counts_by_seat,
        "owned_objective_ids_by_seat": context.owned_objective_ids_by_seat,
        "current_hand_suit_ids": context.current_hand_suit_ids,
        "legal_max_amount": context.legal_max_amount,
        "loan_principal_by_seat": tuple(replay.loans),
        "investment_value_by_seat": tuple(replay.investments),
        "resolved_turn_count": replay.resolved_turn_count,
        "remaining_action_counts": remaining_action_counts,
        "unseen_resource_counts": unseen_resource_counts,
        "opponent_hidden_slots_by_seat": opponent_hidden_slots_by_seat,
    }
    digest = _canonical_digest(canonical_fields, history)
    assert context.current_action_id is not None
    return PublicSearchPosition(
        ruleset_name=ruleset.name,
        player_count=context.player_count,
        starting_cash=context.starting_cash,
        value_chart=context.value_chart,
        objective_ids=context.objective_ids,
        bot_seat=context.bot_seat,
        decision_kind=decision_kind,
        current_action_id=context.current_action_id,
        current_resource_ids=context.current_resource_ids,
        cash_by_seat=context.cash_by_seat,
        tiebreak_seat=context.tiebreak_seat,
        won_resource_counts_by_seat=context.won_resource_counts_by_seat,
        revealed_info_counts_by_seat=context.revealed_info_counts_by_seat,
        owned_objective_ids_by_seat=context.owned_objective_ids_by_seat,
        current_hand_suit_ids=context.current_hand_suit_ids,
        legal_max_amount=context.legal_max_amount,
        loan_principal_by_seat=tuple(replay.loans),
        investment_value_by_seat=tuple(replay.investments),
        resolved_turn_count=replay.resolved_turn_count,
        remaining_action_counts=remaining_action_counts,
        unseen_resource_counts=unseen_resource_counts,
        opponent_hidden_slots_by_seat=opponent_hidden_slots_by_seat,
        belief=belief,
        canonical_input_digest=digest,
    )


def sample_compatible_worlds(
    position: PublicSearchPosition,
    *,
    candidate_identity: str,
    sample_count: int,
) -> tuple[SampledWorld, ...]:
    """Return a deterministic, prefix-stable sequence of compatible worlds."""

    if candidate_identity != LATE_GAME_PUBLIC_BELIEF_V1_DEV_IDENTITY:
        raise ValueError(
            "public-belief samples require the explicit development candidate identity"
        )
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
        raise ValueError("sample count must be a nonnegative integer")

    unseen_resources = tuple(
        suit_id
        for suit_id, count in enumerate(position.unseen_resource_counts, start=1)
        for _ in range(count)
    )
    remaining_actions = tuple(
        action_id
        for action_id, count in enumerate(position.remaining_action_counts, start=1)
        for _ in range(count)
    )
    opponent_slot_count = sum(position.opponent_hidden_slots_by_seat)
    worlds: list[SampledWorld] = []
    for sample_index in range(sample_count):
        resource_order = _deterministic_permutation(
            unseen_resources,
            candidate_identity=candidate_identity,
            canonical_input_digest=position.canonical_input_digest,
            sample_index=sample_index,
            domain="resources",
        )
        action_order = _deterministic_permutation(
            remaining_actions,
            candidate_identity=candidate_identity,
            canonical_input_digest=position.canonical_input_digest,
            sample_index=sample_index,
            domain="actions",
        )
        cursor = 0
        hands: list[tuple[int, ...]] = []
        for seat, slots in enumerate(position.opponent_hidden_slots_by_seat):
            if seat == position.bot_seat:
                hands.append(position.current_hand_suit_ids)
                continue
            hands.append(tuple(resource_order[cursor : cursor + slots]))
            cursor += slots
        if cursor != opponent_slot_count:
            raise AssertionError("all opponent hidden slots must be filled exactly once")
        worlds.append(
            SampledWorld(
                candidate_identity=candidate_identity,
                canonical_input_digest=position.canonical_input_digest,
                sampling_algorithm=_SAMPLING_ALGORITHM,
                search_seed=_FIXED_SEARCH_SEED,
                sample_index=sample_index,
                hand_suits_by_seat=tuple(hands),
                future_resource_suits=tuple(resource_order[cursor:]),
                future_action_ids=action_order,
            )
        )
    return tuple(worlds)


def _replay_history(
    history: PublicHistory,
    ruleset: RulesetKnowledge,
) -> tuple[PublicGameSetup, _ReplayState, tuple[int, ...]]:
    if not isinstance(history, tuple) or not history:
        raise HeuristicInputError("public history must be a nonempty tuple")
    setup = history[0]
    if not isinstance(setup, PublicGameSetup) or setup.kind is not PublicEventKind.GAME_SETUP:
        raise HeuristicInputError("public history must begin with game setup")
    _validate_setup(setup, ruleset)
    state = _ReplayState(
        cash=[setup.starting_cash] * setup.player_count,
        won=[[0] * _SUIT_COUNT for _ in range(setup.player_count)],
        revealed=[[0] * _SUIT_COUNT for _ in range(setup.player_count)],
        owned=[[] for _ in range(setup.player_count)],
        loans=[0] * setup.player_count,
        investments=[0] * setup.player_count,
        tiebreak_seat=setup.initial_tiebreak_seat,
    )
    seen_actions = [0] * _ACTION_COUNT
    for index, event in enumerate(history[1:], start=1):
        if isinstance(event, PublicTurnOpened):
            _open_turn(state, event, index, ruleset, seen_actions)
        elif isinstance(event, PublicAuctionResolved):
            _resolve_turn(state, event, index, setup, ruleset)
        elif isinstance(event, PublicInformationRevealed):
            _reveal_information(state, event, index, ruleset)
        else:
            raise HeuristicInputError(f"public history event {index} has an unsupported type")
    return setup, state, tuple(seen_actions)


def _validate_setup(setup: PublicGameSetup, ruleset: RulesetKnowledge) -> None:
    if setup.player_count != ruleset.player_count:
        raise HeuristicInputError("history setup player count contradicts ruleset knowledge")
    if setup.starting_cash != ruleset.starting_cash:
        raise HeuristicInputError("history setup starting cash contradicts ruleset knowledge")
    if setup.value_chart != ruleset.value_chart:
        raise HeuristicInputError("history setup value chart contradicts ruleset knowledge")
    if len(set(setup.objective_ids)) != len(setup.objective_ids):
        raise HeuristicInputError("history setup objective IDs must be unique")
    if len(setup.objective_ids) != ruleset.active_objective_count:
        raise HeuristicInputError("history setup objective count contradicts ruleset knowledge")
    if not set(setup.objective_ids) <= set(ruleset.objective_pool):
        raise HeuristicInputError("history setup contains an unknown objective ID")
    if not 0 <= setup.initial_tiebreak_seat < setup.player_count:
        raise HeuristicInputError("history setup tiebreak seat is outside player count")


def _open_turn(
    state: _ReplayState,
    event: PublicTurnOpened,
    index: int,
    ruleset: RulesetKnowledge,
    seen_actions: list[int],
) -> None:
    if event.kind is not PublicEventKind.TURN_OPENED:
        raise HeuristicInputError(f"public history event {index} has a contradictory kind")
    if state.current_turn is not None:
        raise HeuristicInputError("public history contains two unresolved turns")
    if state.required_reveal_seat is not None:
        raise HeuristicInputError("public history skips a required information reveal")
    action_id = _known_action_id(event.action_id, index)
    resources = _validated_resources(event.resource_ids, index)
    _validate_resource_carry(state, resources)
    seen_actions[action_id - 1] += 1
    if seen_actions[action_id - 1] > ruleset.action_counts[action_id - 1]:
        raise HeuristicInputError("public history uses more actions than the ruleset contains")
    state.current_turn = event
    state.latest_turn = event


def _resolve_turn(
    state: _ReplayState,
    event: PublicAuctionResolved,
    index: int,
    setup: PublicGameSetup,
    ruleset: RulesetKnowledge,
) -> None:
    if event.kind is not PublicEventKind.AUCTION_RESOLVED:
        raise HeuristicInputError(f"public history event {index} has a contradictory kind")
    turn = state.current_turn
    if turn is None:
        raise HeuristicInputError("public history resolves a turn that is not open")
    if len(event.bids_by_seat) != setup.player_count:
        raise HeuristicInputError("public resolution must contain one bid per seat")
    credit = _ACTION_CREDIT_BY_ID[turn.action_id]
    for seat, bid in enumerate(event.bids_by_seat):
        if not isinstance(bid, int) or isinstance(bid, bool) or bid < 0:
            raise HeuristicInputError("public resolution bids must be nonnegative integers")
        if bid > state.cash[seat] + credit:
            raise HeuristicInputError("public resolution bid exceeds its historical legal maximum")
    winner = _winning_seat(event.bids_by_seat, state.tiebreak_seat)
    paid = max(event.bids_by_seat)
    state.cash[winner] -= paid
    action_name = _ACTION_NAME_BY_ID[turn.action_id]
    if action_name in LOAN_PRINCIPAL:
        principal = LOAN_PRINCIPAL[action_name]
        state.cash[winner] += principal
        state.loans[winner] += principal
    if action_name in INVEST_PAYOUT:
        state.investments[winner] += paid + INVEST_PAYOUT[action_name]

    grant_count = (
        1 if turn.action_id == _AUCTION_ONE else 2 if turn.action_id == _AUCTION_TWO else 0
    )
    granted = tuple(suit_id for suit_id in turn.resource_ids[:grant_count] if suit_id)
    if grant_count and not granted:
        raise HeuristicInputError("public auction resolves without an offered resource")
    for suit_id in granted:
        state.won[winner][suit_id - 1] += 1
    if grant_count:
        for objective_id in setup.objective_ids:
            if any(objective_id in row for row in state.owned):
                continue
            if objective_pattern_met(objective_id, state.won[winner]):
                state.owned[winner].append(objective_id)

    state.tiebreak_seat = winner
    state.resolved_turn_count += 1
    state.required_reveal_seat = (
        winner if sum(state.revealed[winner]) < ruleset.private_cards_per_player else None
    )
    state.previous_resources = turn.resource_ids
    state.previous_action_id = turn.action_id
    state.current_turn = None


def _reveal_information(
    state: _ReplayState,
    event: PublicInformationRevealed,
    index: int,
    ruleset: RulesetKnowledge,
) -> None:
    if event.kind is not PublicEventKind.INFORMATION_REVEALED:
        raise HeuristicInputError(f"public history event {index} has a contradictory kind")
    if state.current_turn is not None or state.required_reveal_seat is None:
        raise HeuristicInputError("public information reveal is out of sequence")
    if event.seat != state.required_reveal_seat:
        raise HeuristicInputError("public information reveal seat contradicts the auction winner")
    if not isinstance(event.suit_id, int) or not 1 <= event.suit_id <= _SUIT_COUNT:
        raise HeuristicInputError("public information reveal contains an unknown suit")
    state.revealed[event.seat][event.suit_id - 1] += 1
    if sum(state.revealed[event.seat]) > ruleset.private_cards_per_player:
        raise HeuristicInputError("public reveals exceed the player's initial hand")
    state.required_reveal_seat = None


def _validate_resource_carry(
    state: _ReplayState,
    resources: tuple[int, int],
) -> None:
    previous = state.previous_resources
    previous_action = state.previous_action_id
    if previous is None or previous_action is None:
        return
    if previous_action == _AUCTION_ONE and resources[0] != previous[1]:
        raise HeuristicInputError("public resources contradict the one-card auction carry")
    if previous_action not in (_AUCTION_ONE, _AUCTION_TWO) and resources != previous:
        raise HeuristicInputError("public resources change after a non-resource action")


def _validate_replay_against_context(
    context: DecisionContext,
    state: _ReplayState,
    setup: PublicGameSetup,
    ruleset: RulesetKnowledge,
) -> None:
    if context.objective_ids != setup.objective_ids:
        raise HeuristicInputError("context objective IDs contradict public history")
    expected = (
        ("cash", context.cash_by_seat, tuple(state.cash)),
        ("won resources", context.won_resource_counts_by_seat, tuple(map(tuple, state.won))),
        (
            "revealed information",
            context.revealed_info_counts_by_seat,
            tuple(map(tuple, state.revealed)),
        ),
        (
            "owned objectives",
            context.owned_objective_ids_by_seat,
            tuple(map(tuple, state.owned)),
        ),
    )
    for name, actual, replayed in expected:
        if actual != replayed:
            raise HeuristicInputError(f"context {name} contradict public history")
    if context.tiebreak_seat != state.tiebreak_seat:
        raise HeuristicInputError("context tiebreak seat contradicts public history")
    latest = state.latest_turn
    if latest is None:
        raise HeuristicInputError("public history must contain at least one turn")
    if context.current_action_id != latest.action_id:
        raise HeuristicInputError("context current action contradicts public history")
    if context.current_resource_ids != latest.resource_ids:
        raise HeuristicInputError("context current resources contradict public history")

    if context.decision_kind == "submitBid":
        if state.current_turn is None:
            raise HeuristicInputError("bid context requires a current unresolved public turn")
        if state.required_reveal_seat is not None:
            raise HeuristicInputError("bid context cannot skip a required public reveal")
        assert context.current_action_id is not None
        legal_max = state.cash[context.bot_seat] + _ACTION_CREDIT_BY_ID[context.current_action_id]
        if context.legal_max_amount != legal_max:
            raise HeuristicInputError("context legal maximum contradicts replayed public cash")
    elif context.decision_kind == "selectInfoToReveal":
        if state.current_turn is not None or state.required_reveal_seat != context.bot_seat:
            raise HeuristicInputError("reveal context does not belong to the pending public winner")
        if context.legal_max_amount is not None:
            raise HeuristicInputError("reveal context cannot include a legal bid maximum")
        if len(context.current_hand_suit_ids) <= 1:
            raise HeuristicInputError("choice reveal requires at least two cards in the bot hand")
    else:
        raise HeuristicInputError("decision kind is unsupported")

    if sum(state.revealed[context.bot_seat]) + len(context.current_hand_suit_ids) != (
        ruleset.private_cards_per_player
    ):
        raise HeuristicInputError("bot hand contradicts replayed public reveals")


def _known_action_id(value: int, index: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise HeuristicInputError(f"public history event {index} action ID must be an integer")
    try:
        return int(ActionId(value))
    except ValueError as error:
        raise HeuristicInputError(f"public history event {index} action ID is unknown") from error


def _validated_resources(values: tuple[int, int], index: int) -> tuple[int, int]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise HeuristicInputError(
            f"public history event {index} resources must contain two entries"
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _SUIT_COUNT
        for value in values
    ):
        raise HeuristicInputError(f"public history event {index} resource ID is unknown")
    if values[0] == 0 and values[1] != 0:
        raise HeuristicInputError(f"public history event {index} resources are not zero-padded")
    return values


def _winning_seat(bids: tuple[int, ...], tiebreak_seat: int) -> int:
    highest = max(bids)
    for offset in range(1, len(bids) + 1):
        seat = (tiebreak_seat + offset) % len(bids)
        if bids[seat] == highest:
            return seat
    raise AssertionError("a nonempty bid tuple always has a winner")


def _count_suits(suit_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(suit_ids.count(suit_id) for suit_id in range(1, _SUIT_COUNT + 1))


def _canonical_digest(fields: dict[str, object], history: PublicHistory) -> str:
    payload = {
        **fields,
        "history": [_public_event_payload(event) for event in history],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _public_event_payload(event: object) -> dict[str, object]:
    if isinstance(event, PublicGameSetup):
        return {
            "kind": event.kind,
            "player_count": event.player_count,
            "starting_cash": event.starting_cash,
            "value_chart": event.value_chart,
            "initial_tiebreak_seat": event.initial_tiebreak_seat,
            "objective_ids": event.objective_ids,
        }
    if isinstance(event, PublicTurnOpened):
        return {
            "kind": event.kind,
            "action_id": event.action_id,
            "resource_ids": event.resource_ids,
        }
    if isinstance(event, PublicAuctionResolved):
        return {"kind": event.kind, "bids_by_seat": event.bids_by_seat}
    if isinstance(event, PublicInformationRevealed):
        return {"kind": event.kind, "seat": event.seat, "suit_id": event.suit_id}
    raise HeuristicInputError("public history contains an unsupported event")


def _deterministic_permutation(
    values: tuple[int, ...],
    *,
    candidate_identity: str,
    canonical_input_digest: str,
    sample_index: int,
    domain: str,
) -> tuple[int, ...]:
    key = hashlib.sha256(
        b"\0".join(
            (
                _SAMPLING_ALGORITHM.encode(),
                candidate_identity.encode(),
                canonical_input_digest.encode(),
                str(_FIXED_SEARCH_SEED).encode(),
                str(sample_index).encode(),
                domain.encode(),
            )
        )
    ).digest()
    ranked = sorted(
        enumerate(values),
        key=lambda item: hashlib.sha256(
            key + item[0].to_bytes(8, byteorder="big") + item[1].to_bytes(2, byteorder="big")
        ).digest(),
    )
    return tuple(value for _index, value in ranked)
