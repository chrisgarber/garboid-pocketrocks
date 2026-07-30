from __future__ import annotations

from functools import cached_property
from importlib.resources import files
from typing import Protocol

from pocketrocks import (
    OBJECTIVES,
    ActionId,
    DecisionContext,
    Suit,
    describe_action,
    describe_objective,
    describe_suit,
    objective_payout,
)

from garboid_pocketrocks.rules import RulesetKnowledge

_ACTION_LABELS: dict[ActionId, str] = {
    ActionId.AUCTION1: "Auction 1",
    ActionId.AUCTION2: "Auction 2",
    ActionId.LOAN10: "Loan 10",
    ActionId.LOAN20: "Loan 20",
    ActionId.INVEST5: "Invest 5",
    ActionId.INVEST10: "Invest 10",
}


class PromptSkill(Protocol):
    def render(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        correction: str | None = None,
    ) -> str:
        """Render complete instructions for one independent LLM decision."""


class PocketRocksPromptSkill:
    @cached_property
    def instructions(self) -> str:
        skill = files("garboid_pocketrocks.bots.llm").joinpath(
            "skills",
            "pocketrocks",
            "SKILL.md",
        )
        return skill.read_text(encoding="utf-8").strip()

    def render(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        *,
        correction: str | None = None,
    ) -> str:
        lines = [
            self.instructions,
            "",
            "## Current game state",
            f"Ruleset: {ruleset.name}",
            (
                f"Players: {context.player_count}; you are seat {context.bot_seat}; "
                f"priority seat: {context.tiebreak_seat}"
            ),
            f"Starting cash: ${context.starting_cash}",
            f"Private cards per player: {ruleset.private_cards_per_player}",
            f"Resource deck counts: {_resource_counts(ruleset.resource_counts)}",
            f"Action deck counts: {_action_counts(ruleset.action_counts)}",
            f"Value chart: {_value_chart(context.value_chart)}",
            "Active objectives:",
            *_active_objectives(context.objective_ids),
            f"Current action: {_current_action(context.current_action_id)}",
            f"Offered resources: {_current_resources(context.current_resource_ids)}",
            "Seats:",
            *_seat_states(context),
            f"Your private hand: [{_indexed_hand(context.current_hand_suit_ids)}]",
            (
                "Prior bids and current loan/investment positions are not exposed by "
                "this SDK snapshot."
            ),
        ]
        if context.decision_kind == "selectInfoToReveal":
            lines.append(f"Reveal choices: {_reveal_choices(context.current_hand_suit_ids)}")
        lines.extend(("", "## Required response"))
        if correction is not None:
            lines.append(f"Correction: {correction}")
        lines.append(_response_contract(context))
        return "\n".join(lines)


def _resource_counts(counts: tuple[int, ...]) -> str:
    return ", ".join(f"{suit.label}={count}" for suit, count in zip(Suit, counts, strict=True))


def _action_counts(counts: tuple[int, ...]) -> str:
    return ", ".join(
        f"{_ACTION_LABELS[action]}={count}" for action, count in zip(ActionId, counts, strict=True)
    )


def _value_chart(chart: tuple[int, ...]) -> str:
    labels = tuple(str(index) for index in range(len(chart) - 1)) + (f"{len(chart) - 1}+",)
    entries = ", ".join(f"{label}=${value}" for label, value in zip(labels, chart, strict=True))
    return f"revealed {entries}"


def _active_objectives(objective_ids: tuple[int, ...]) -> list[str]:
    if not objective_ids:
        return ["- none"]
    return [
        (
            f"- Objective {objective_id}: {describe_objective(objective_id)}; "
            f"payout=${objective_payout(objective_id)}"
        )
        for objective_id in objective_ids
    ]


def _current_action(action_id: int | None) -> str:
    if action_id is None:
        return "none"
    try:
        label = _ACTION_LABELS[ActionId(action_id)]
    except KeyError, ValueError:
        label = f"Action {action_id}"
    return f"{label} — {describe_action(action_id)}"


def _current_resources(resource_ids: tuple[int, int]) -> str:
    names = [describe_suit(resource_id) for resource_id in resource_ids if resource_id]
    return ", ".join(names) if names else "none"


def _seat_states(context: DecisionContext) -> list[str]:
    states = []
    for seat in range(context.player_count):
        marker = " (YOU)" if seat == context.bot_seat else ""
        objectives = ", ".join(
            f"{objective_id}: {OBJECTIVES[objective_id].description} "
            f"(${OBJECTIVES[objective_id].payout})"
            for objective_id in context.owned_objective_ids_by_seat[seat]
            if objective_id in OBJECTIVES
        )
        states.append(
            f"- Seat {seat}{marker}: cash=${context.cash_by_seat[seat]}; "
            f"won={_suit_mapping(context.won_resource_counts_by_seat[seat])}; "
            f"revealed={_suit_mapping(context.revealed_info_counts_by_seat[seat])}; "
            f"objectives=[{objectives}]"
        )
    return states


def _suit_mapping(counts: tuple[int, ...]) -> str:
    entries = ", ".join(f"{suit.label}: {count}" for suit, count in zip(Suit, counts, strict=True))
    return f"{{{entries}}}"


def _indexed_hand(hand: tuple[int, ...]) -> str:
    return ", ".join(f"{index}: {describe_suit(suit_id)}" for index, suit_id in enumerate(hand))


def _reveal_choices(hand: tuple[int, ...]) -> str:
    return "; ".join(f"{index}: {describe_suit(suit_id)}" for index, suit_id in enumerate(hand))


def _response_contract(context: DecisionContext) -> str:
    if context.decision_kind == "submitBid":
        if context.legal_max_amount is None or context.legal_max_amount < 0:
            raise ValueError("bid prompt requires a nonnegative legal maximum")
        return (
            "Return exactly one base-10 integer from 0 through "
            f"{context.legal_max_amount}. 0 means bid zero/pass."
        )
    if context.revealable_count <= 0:
        raise ValueError("reveal prompt requires at least one revealable card")
    return (
        "Return exactly one base-10 integer from 0 through "
        f"{context.revealable_count - 1}: the card index to reveal."
    )
