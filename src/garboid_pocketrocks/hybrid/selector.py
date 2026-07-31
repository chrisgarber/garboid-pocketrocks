"""Deterministic identity selection and fallback diagnostics for future hybrids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pocketrocks import DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicHistory,
    ValidatedPublicHistory,
    validate_public_history,
)
from garboid_pocketrocks.diagnostics.trace import (
    PublicDecisionContext,
    public_context_from_sdk,
)
from garboid_pocketrocks.hybrid.experts import (
    ExpertAvailability,
    VerifiedPromotedExpertCatalog,
    _require_verified_catalog,
)
from garboid_pocketrocks.knowledge import (
    RulesetKnowledge,
    canonical_knowledge,
    knowledge_for_context,
    value_chart_from_ruleset_name,
)

type FallbackReason = Literal[
    "none",
    "selector_rejected_input",
    "selector_returned_ineligible_expert",
    "requested_expert_unavailable",
    "no_available_expert",
]


@dataclass(frozen=True, slots=True)
class LiveSelectorInput:
    """An explicit copy of state available to a live bot at decision time."""

    context: PublicDecisionContext
    own_hand_suit_ids: tuple[int, ...]
    ruleset_name: str
    public_history: PublicHistory

    @classmethod
    def from_live_state(
        cls,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        public_history: PublicHistory,
    ) -> LiveSelectorInput:
        """Copy allowlisted context, the bot's own hand, and public history."""

        derived = knowledge_for_context(context)
        canonical = canonical_knowledge(
            context.player_count,
            value_chart=value_chart_from_ruleset_name(derived.name),
            objectives_enabled=bool(context.objective_ids),
        )
        _validate_live_context(context, canonical)
        history_state = validate_public_history(public_history)
        _bind_history_to_context(history_state, context=context, canonical=canonical)
        if derived != canonical:
            raise ValueError("live context does not match the canonical PocketRocks ruleset")
        if ruleset != canonical:
            raise ValueError("provided ruleset knowledge does not match the canonical live ruleset")
        return cls(
            context=public_context_from_sdk(context),
            own_hand_suit_ids=tuple(context.current_hand_suit_ids),
            ruleset_name=canonical.name,
            public_history=tuple(public_history),
        )


def _validate_live_context(
    context: DecisionContext,
    canonical: RulesetKnowledge,
) -> None:
    player_count = canonical.player_count
    if context.starting_cash != canonical.starting_cash:
        raise ValueError("live context starting cash does not match the canonical ruleset")
    if not 0 <= context.bot_seat < player_count:
        raise ValueError("live context bot seat is outside the player count")
    if not 0 <= context.tiebreak_seat < player_count:
        raise ValueError("live context tiebreak seat is outside the player count")
    if len(context.objective_ids) != canonical.active_objective_count:
        raise ValueError("live context has the wrong number of active objectives")
    if len(set(context.objective_ids)) != len(context.objective_ids) or any(
        objective_id not in canonical.objective_pool for objective_id in context.objective_ids
    ):
        raise ValueError("live context contains invalid objective identities")
    _require_nonnegative_vector("cash by seat", context.cash_by_seat, length=player_count)
    won = _require_count_matrix(
        "won resource counts",
        context.won_resource_counts_by_seat,
        rows=player_count,
        columns=len(canonical.resource_counts),
    )
    revealed = _require_count_matrix(
        "revealed information counts",
        context.revealed_info_counts_by_seat,
        rows=player_count,
        columns=len(canonical.resource_counts),
    )
    _require_objective_matrix(
        context.owned_objective_ids_by_seat,
        player_count=player_count,
        active_objectives=frozenset(context.objective_ids),
    )
    hand = tuple(context.current_hand_suit_ids)
    if any(not _is_suit_id(suit_id) for suit_id in hand):
        raise ValueError("live context hand contains an invalid suit ID")
    if sum(revealed[context.bot_seat]) + len(hand) != canonical.private_cards_per_player:
        raise ValueError(
            "live context focal private-card total does not match the canonical ruleset"
        )
    if any(sum(row) > canonical.private_cards_per_player for row in revealed):
        raise ValueError("live context reveals more private cards than a player owns")
    visible_information_by_suit = tuple(
        sum(row[suit_index] for row in revealed)
        + sum(suit_id == suit_index + 1 for suit_id in hand)
        for suit_index in range(len(canonical.resource_counts))
    )
    if any(
        visible > available
        for visible, available in zip(
            visible_information_by_suit,
            canonical.resource_counts,
            strict=True,
        )
    ):
        raise ValueError("live context exposes more information cards than the canonical deck")
    won_by_suit = tuple(sum(row[index] for row in won) for index in range(len(won[0])))
    if any(
        total > available
        for total, available in zip(won_by_suit, canonical.resource_counts, strict=True)
    ):
        raise ValueError("live context awards more resource cards than the canonical deck")
    resources = tuple(context.current_resource_ids)
    if len(resources) != 2 or any(not _is_resource_id(resource_id) for resource_id in resources):
        raise ValueError("live context current resources contain an invalid suit ID")
    if resources[0] == 0 and resources[1] != 0:
        raise ValueError("live context current resources are not zero-padded")
    action_id = context.current_action_id
    if action_id is not None and (
        not _is_integer(action_id) or not 1 <= action_id <= len(canonical.action_counts)
    ):
        raise ValueError("live context current action ID is invalid")
    if context.legal_max_amount is not None and (
        not _is_integer(context.legal_max_amount) or context.legal_max_amount < 0
    ):
        raise ValueError("live context legal bid maximum is invalid")
    if (
        not _is_integer(context.revealable_count)
        or context.revealable_count < 0
        or context.revealable_count != len(hand)
    ):
        raise ValueError("live context revealable count must equal the focal hand length")


def _bind_history_to_context(
    history: ValidatedPublicHistory,
    *,
    context: DecisionContext,
    canonical: RulesetKnowledge,
) -> None:
    setup = history.setup
    if (
        setup.player_count != canonical.player_count
        or setup.starting_cash != canonical.starting_cash
        or setup.value_chart != canonical.value_chart
        or setup.objective_ids != context.objective_ids
    ):
        raise ValueError("public history setup does not match the canonical live context")
    replayed_fields = (
        ("cash", context.cash_by_seat, history.cash_by_seat),
        (
            "won resources",
            context.won_resource_counts_by_seat,
            history.won_resource_counts_by_seat,
        ),
        (
            "revealed information",
            context.revealed_info_counts_by_seat,
            history.revealed_info_counts_by_seat,
        ),
        (
            "owned objectives",
            context.owned_objective_ids_by_seat,
            history.owned_objective_ids_by_seat,
        ),
    )
    for name, live_value, replayed_value in replayed_fields:
        if live_value != replayed_value:
            raise ValueError(f"live context {name} contradict public history")
    combined_known_resources_by_suit = tuple(
        sum(row[suit_index] for row in history.won_resource_counts_by_seat)
        + sum(row[suit_index] for row in history.revealed_info_counts_by_seat)
        + context.current_hand_suit_ids.count(suit_index + 1)
        + history.visible_resource_ids.count(suit_index + 1)
        for suit_index in range(len(canonical.resource_counts))
    )
    if any(
        known > available
        for known, available in zip(
            combined_known_resources_by_suit,
            canonical.resource_counts,
            strict=True,
        )
    ):
        raise ValueError("live context combined known resources exceed the canonical deck")
    turn = history.latest_turn
    if turn is None:
        raise ValueError("public history has no current turn")
    if (
        context.current_action_id != turn.action_id
        or context.current_resource_ids != turn.resource_ids
    ):
        raise ValueError("live context action and resources do not match the latest public turn")
    if context.tiebreak_seat != history.tiebreak_seat:
        raise ValueError("live context tiebreak seat does not match public history")
    if context.decision_kind == "submitBid":
        if history.phase != "turn_open":
            raise ValueError("bid decision requires one unresolved public turn")
        if history.legal_max_bid_by_seat is None:
            raise AssertionError("turn-open public history always has legal bid maxima")
        expected_legal_max = history.legal_max_bid_by_seat[context.bot_seat]
        if context.legal_max_amount != expected_legal_max:
            raise ValueError("live context legal bid maximum does not match public history")
    elif context.decision_kind == "selectInfoToReveal":
        if history.phase != "reveal_pending":
            raise ValueError("reveal decision requires one resolved auction awaiting reveal")
        if context.legal_max_amount is not None:
            raise ValueError("reveal decision cannot contain a legal bid maximum")
        if context.bot_seat != history.tiebreak_seat:
            raise ValueError("reveal decision must belong to the public auction winner")
        if context.revealable_count <= 1:
            raise ValueError("choice reveal decision requires at least two cards")
    else:
        raise ValueError("live context decision kind is unsupported")


def _require_nonnegative_vector(
    name: str,
    values: tuple[int, ...],
    *,
    length: int,
) -> tuple[int, ...]:
    output = tuple(values)
    if len(output) != length or any(not _is_integer(value) or value < 0 for value in output):
        raise ValueError(f"live context {name} must contain {length} nonnegative integers")
    return output


def _require_count_matrix(
    name: str,
    values: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    columns: int,
) -> tuple[tuple[int, ...], ...]:
    output = tuple(tuple(row) for row in values)
    if len(output) != rows:
        raise ValueError(f"live context {name} must contain one row per player")
    for row in output:
        _require_nonnegative_vector(name, row, length=columns)
    return output


def _require_objective_matrix(
    values: tuple[tuple[int, ...], ...],
    *,
    player_count: int,
    active_objectives: frozenset[int],
) -> None:
    rows = tuple(tuple(row) for row in values)
    if len(rows) != player_count:
        raise ValueError("live context owned objectives must contain one row per player")
    flattened = tuple(objective_id for row in rows for objective_id in row)
    if len(set(flattened)) != len(flattened) or any(
        not _is_integer(value) or value not in active_objectives for value in flattened
    ):
        raise ValueError("live context owned objectives contain invalid or duplicate identities")


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_suit_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5


def _is_resource_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 5


class SelectorInputRejected(ValueError):
    """A deterministic selector could not classify one valid live input."""


class DeterministicExpertSelector(Protocol):
    """Choose one identity without running it or observing hidden state."""

    def choose_expert(
        self,
        selector_input: LiveSelectorInput,
        eligible_expert_names: tuple[str, ...],
    ) -> str:
        """Return one explicit eligible identity or reject the input."""


@dataclass(frozen=True, slots=True)
class ExpertSelection:
    """Chosen identity plus stable fallback evidence."""

    requested_expert_name: str | None
    selected_expert_name: str | None
    used_fallback: bool
    fallback_reason: FallbackReason
    unavailable_experts: tuple[ExpertAvailability, ...]


def choose_promoted_expert(
    selector: DeterministicExpertSelector,
    selector_input: LiveSelectorInput,
    catalog: VerifiedPromotedExpertCatalog,
    availability_by_name: Mapping[str, ExpertAvailability],
) -> ExpertSelection:
    """Resolve one eligible, available identity with catalog-order fallback."""

    experts = _require_verified_catalog(catalog)
    names = tuple(expert.name for expert in experts)
    unknown_availability = set(availability_by_name).difference(names)
    if unknown_availability:
        raise ValueError(
            f"availability contains unknown expert names {sorted(unknown_availability)!r}"
        )
    diagnostics = tuple(_availability_for(name, availability_by_name) for name in names)
    available_names = frozenset(
        diagnostic.expert_name for diagnostic in diagnostics if diagnostic.available
    )
    unavailable = tuple(diagnostic for diagnostic in diagnostics if not diagnostic.available)

    try:
        requested = selector.choose_expert(selector_input, names)
    except SelectorInputRejected:
        return _fallback_selection(
            names,
            available_names,
            unavailable,
            requested=None,
            reason="selector_rejected_input",
        )
    if not isinstance(requested, str) or requested not in names:
        return _fallback_selection(
            names,
            available_names,
            unavailable,
            requested=requested if isinstance(requested, str) else None,
            reason="selector_returned_ineligible_expert",
        )
    if requested not in available_names:
        return _fallback_selection(
            names,
            available_names,
            unavailable,
            requested=requested,
            reason="requested_expert_unavailable",
        )
    return ExpertSelection(
        requested_expert_name=requested,
        selected_expert_name=requested,
        used_fallback=False,
        fallback_reason="none",
        unavailable_experts=unavailable,
    )


def _fallback_selection(
    names: tuple[str, ...],
    available_names: frozenset[str],
    unavailable: tuple[ExpertAvailability, ...],
    *,
    requested: str | None,
    reason: FallbackReason,
) -> ExpertSelection:
    selected = next((name for name in names if name in available_names), None)
    return ExpertSelection(
        requested_expert_name=requested,
        selected_expert_name=selected,
        used_fallback=True,
        fallback_reason=reason if selected is not None else "no_available_expert",
        unavailable_experts=unavailable,
    )


def _availability_for(
    name: str,
    availability_by_name: Mapping[str, ExpertAvailability],
) -> ExpertAvailability:
    diagnostic = availability_by_name.get(name)
    if diagnostic is None:
        return ExpertAvailability(
            expert_name=name,
            available=False,
            reason="availability_not_reported",
            detail="Runtime availability was not checked.",
        )
    if diagnostic.expert_name != name:
        raise ValueError(f"availability key {name!r} does not match its diagnostic")
    return diagnostic
