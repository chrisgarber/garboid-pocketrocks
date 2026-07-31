"""Deterministic identity selection and fallback diagnostics for future hybrids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pocketrocks import DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.diagnostics.trace import (
    PublicDecisionContext,
    public_context_from_sdk,
)
from garboid_pocketrocks.hybrid.experts import (
    ExpertAvailability,
    VerifiedPromotedExpertCatalog,
    _require_verified_catalog,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge, knowledge_for_context

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
        if len(set(context.objective_ids)) != len(context.objective_ids) or any(
            objective_id not in derived.objective_pool for objective_id in context.objective_ids
        ):
            raise ValueError("live context contains invalid objective identities")
        if ruleset != derived:
            raise ValueError("provided ruleset knowledge does not match the canonical live ruleset")
        return cls(
            context=public_context_from_sdk(context),
            own_hand_suit_ids=tuple(context.current_hand_suit_ids),
            ruleset_name=derived.name,
            public_history=tuple(public_history),
        )


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
