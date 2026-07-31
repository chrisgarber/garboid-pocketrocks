"""Select heuristic experts from the publicly known resource horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pocketrocks import ActionId

from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.knowledge import RulesetKnowledge

type HeuristicPhase = Literal["early", "middle", "late"]


class _PublicResourceContext(Protocol):
    """The public context fields needed to count remaining resources."""

    @property
    def decision_kind(self) -> str: ...

    @property
    def player_count(self) -> int: ...

    @property
    def current_action_id(self) -> int | None: ...

    @property
    def current_resource_ids(self) -> tuple[int, int]: ...

    @property
    def won_resource_counts_by_seat(self) -> tuple[tuple[int, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class PublicResourceHorizon:
    """Public counts used to choose one phase-specific heuristic expert."""

    total_biddable_resources: int
    future_biddable_resources: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.total_biddable_resources, int)
            or isinstance(self.total_biddable_resources, bool)
            or self.total_biddable_resources <= 0
        ):
            raise ValueError("total biddable resources must be a positive integer")
        if (
            not isinstance(self.future_biddable_resources, int)
            or isinstance(self.future_biddable_resources, bool)
            or not 0 <= self.future_biddable_resources <= self.total_biddable_resources
        ):
            raise ValueError("future biddable resources must be between zero and the total")


def public_resource_horizon(
    context: _PublicResourceContext,
    ruleset: RulesetKnowledge,
) -> PublicResourceHorizon:
    """Count resources still available to bid on from public information only."""

    _validate_player_counts(context, ruleset)
    if context.decision_kind not in ("submitBid", "selectInfoToReveal"):
        raise HeuristicInputError("decision kind is unsupported")
    resource_type_count = len(ruleset.resource_counts)
    _require_nonnegative_integers("ruleset resource counts", ruleset.resource_counts)
    if (
        not isinstance(ruleset.private_cards_per_player, int)
        or isinstance(ruleset.private_cards_per_player, bool)
        or ruleset.private_cards_per_player < 0
    ):
        raise HeuristicInputError("ruleset private cards per player must be a nonnegative integer")
    total = sum(ruleset.resource_counts) - (context.player_count * ruleset.private_cards_per_player)
    if total <= 0:
        raise HeuristicInputError("total biddable resources must be positive")

    won = context.won_resource_counts_by_seat
    if len(won) != context.player_count:
        raise HeuristicInputError("public won-resource rows must match the player count")
    for row in won:
        if len(row) != resource_type_count:
            raise HeuristicInputError(
                "public won-resource row width must match ruleset resource counts"
            )
        _require_nonnegative_integers("public won-resource counts", row)

    already_won = sum(sum(row) for row in won)
    currently_offered = _resources_awarded_by_current_action(
        context,
        resource_type_count=resource_type_count,
    )
    future = total - already_won - currently_offered
    try:
        return PublicResourceHorizon(
            total_biddable_resources=total,
            future_biddable_resources=future,
        )
    except ValueError as error:
        raise HeuristicInputError(str(error)) from error


def select_expert_phase(horizon: PublicResourceHorizon) -> HeuristicPhase:
    """Return the inclusive resource third containing the public horizon."""

    future = horizon.future_biddable_resources
    total = horizon.total_biddable_resources
    if 3 * future >= 2 * total:
        return "early"
    if 3 * future >= total:
        return "middle"
    return "late"


def _validate_player_counts(
    context: _PublicResourceContext,
    ruleset: RulesetKnowledge,
) -> None:
    if (
        not isinstance(context.player_count, int)
        or isinstance(context.player_count, bool)
        or not isinstance(ruleset.player_count, int)
        or isinstance(ruleset.player_count, bool)
    ):
        raise HeuristicInputError("player counts must be integers")
    if context.player_count != ruleset.player_count:
        raise HeuristicInputError("context player count contradicts ruleset knowledge")


def _resources_awarded_by_current_action(
    context: _PublicResourceContext,
    *,
    resource_type_count: int,
) -> int:
    if context.decision_kind != "submitBid":
        return 0
    if len(context.current_resource_ids) != 2:
        raise HeuristicInputError("current resources must contain two slots")
    _require_resource_ids(context.current_resource_ids, resource_type_count)
    action_id = context.current_action_id
    if not isinstance(action_id, int) or isinstance(action_id, bool):
        raise HeuristicInputError("current action ID is unknown")
    try:
        action = ActionId(action_id)
    except ValueError as error:
        raise HeuristicInputError("current action ID is unknown") from error
    offered_resources: tuple[int, ...]
    if action is ActionId.AUCTION1:
        offered_resources = context.current_resource_ids[:1]
    elif action is ActionId.AUCTION2:
        offered_resources = context.current_resource_ids
    else:
        return 0
    offered_count = sum(resource_id != 0 for resource_id in offered_resources)
    if offered_count == 0:
        raise HeuristicInputError("auction is missing an offered resource")
    return offered_count


def _require_nonnegative_integers(
    name: str,
    values: tuple[int, ...],
) -> None:
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise HeuristicInputError(f"{name} must be nonnegative integers")


def _require_resource_ids(
    resource_ids: tuple[int, ...],
    resource_type_count: int,
) -> None:
    if any(
        not isinstance(resource_id, int)
        or isinstance(resource_id, bool)
        or not 0 <= resource_id <= resource_type_count
        for resource_id in resource_ids
    ):
        raise HeuristicInputError("current resource IDs must identify known public resources")
