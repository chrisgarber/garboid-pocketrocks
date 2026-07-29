from __future__ import annotations

import gymnasium as gym
import numpy as np
from pocketrocks import ActionId, DecisionContext, Suit

from garboid_pocketrocks.rules import RulesetKnowledge
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds

_MAX_SEATS = 5
_MAX_OBJECTIVES = 30
_RULESET_FIELDS = frozenset(
    {
        "rules_resource_counts",
        "rules_action_counts",
        "rules_private_cards",
        "rules_objective_pool",
        "rules_active_objective_count",
        "rules_objectives_enabled",
    }
)


class ObservationEncoder:
    """Encodes only public SDK context and public ruleset knowledge."""

    def __init__(
        self,
        bounds: EnvironmentBounds,
        *,
        hidden_ruleset_fields: frozenset[str] = frozenset(),
    ) -> None:
        unknown_fields = hidden_ruleset_fields - _RULESET_FIELDS
        if unknown_fields:
            raise ValueError(f"unknown hidden ruleset fields: {sorted(unknown_fields)!r}")
        self.bounds = bounds
        self.hidden_ruleset_fields = hidden_ruleset_fields
        self.action_codec = ActionCodec(bounds)
        self.observation_space = gym.spaces.Dict(
            {
                "phase": _box(0, 1, (1,), np.int8),
                "player_count": _box(0, _MAX_SEATS, (1,), np.int8),
                "bot_seat": _box(0, _MAX_SEATS - 1, (1,), np.int8),
                "starting_cash": _box(0, int(np.iinfo(np.int16).max), (1,), np.int16),
                "value_chart": _box(0, int(np.iinfo(np.int16).max), (6,), np.int16),
                "active_objectives": _box(0, 1, (_MAX_OBJECTIVES,), np.int8),
                "current_action": _box(0, len(ActionId), (1,), np.int8),
                "current_resources": _box(0, len(Suit), (2,), np.int8),
                "cash_by_seat": _box(0, int(np.iinfo(np.int16).max), (_MAX_SEATS,), np.int16),
                "priority_seat": _box(0, _MAX_SEATS - 1, (1,), np.int8),
                "won_resources": _box(
                    0, int(np.iinfo(np.int16).max), (_MAX_SEATS, len(Suit)), np.int16
                ),
                "revealed_info": _box(
                    0, int(np.iinfo(np.int16).max), (_MAX_SEATS, len(Suit)), np.int16
                ),
                "owned_objectives": _box(0, 1, (_MAX_SEATS, _MAX_OBJECTIVES), np.int8),
                "private_hand": _box(0, len(Suit), (bounds.max_hand_size,), np.int8),
                "rules_resource_counts": _box(
                    0, int(np.iinfo(np.int16).max), (len(Suit),), np.int16
                ),
                "rules_action_counts": _box(
                    0, int(np.iinfo(np.int16).max), (len(ActionId),), np.int16
                ),
                "rules_private_cards": _box(0, int(np.iinfo(np.int16).max), (1,), np.int16),
                "rules_objective_pool": _box(0, 1, (_MAX_OBJECTIVES,), np.int8),
                "rules_active_objective_count": _box(0, _MAX_OBJECTIVES, (1,), np.int8),
                "rules_objectives_enabled": _box(0, 1, (1,), np.int8),
                "action_mask": _box(0, 1, (self.action_codec.size,), np.int8),
            }
        )

    def encode(
        self,
        context: DecisionContext,
        knowledge: RulesetKnowledge,
    ) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        self._validate(context, knowledge)
        encoded: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]] = {
            "phase": np.array([int(context.decision_kind == "selectInfoToReveal")], dtype=np.int8),
            "player_count": np.array([context.player_count], dtype=np.int8),
            "bot_seat": np.array([context.bot_seat], dtype=np.int8),
            "starting_cash": np.array([context.starting_cash], dtype=np.int16),
            "value_chart": np.asarray(context.value_chart, dtype=np.int16),
            "active_objectives": _bitset(context.objective_ids),
            "current_action": np.array([context.current_action_id or 0], dtype=np.int8),
            "current_resources": np.asarray(context.current_resource_ids, dtype=np.int8),
            "cash_by_seat": _padded_vector(context.cash_by_seat, _MAX_SEATS, np.int16),
            "priority_seat": np.array([context.tiebreak_seat], dtype=np.int8),
            "won_resources": _padded_matrix(
                context.won_resource_counts_by_seat, _MAX_SEATS, len(Suit), np.int16
            ),
            "revealed_info": _padded_matrix(
                context.revealed_info_counts_by_seat, _MAX_SEATS, len(Suit), np.int16
            ),
            "owned_objectives": _objective_matrix(context.owned_objective_ids_by_seat),
            "private_hand": _padded_vector(
                context.current_hand_suit_ids, self.bounds.max_hand_size, np.int8
            ),
            "rules_resource_counts": np.asarray(knowledge.resource_counts, dtype=np.int16),
            "rules_action_counts": np.asarray(knowledge.action_counts, dtype=np.int16),
            "rules_private_cards": np.array([knowledge.private_cards_per_player], dtype=np.int16),
            "rules_objective_pool": _bitset(knowledge.objective_pool),
            "rules_active_objective_count": np.array(
                [knowledge.active_objective_count], dtype=np.int8
            ),
            "rules_objectives_enabled": np.array([knowledge.objectives_enabled], dtype=np.int8),
            "action_mask": self.action_codec.mask(context),
        }
        for field in self.hidden_ruleset_fields:
            encoded[field] = np.zeros_like(encoded[field])
        return encoded

    def _validate(self, context: DecisionContext, knowledge: RulesetKnowledge) -> None:
        if not 1 <= context.player_count <= _MAX_SEATS:
            raise ValueError("player count exceeds observation bounds")
        if not 0 <= context.bot_seat < context.player_count:
            raise ValueError("bot seat is outside player count")
        if context.starting_cash > np.iinfo(np.int16).max:
            raise ValueError("starting cash exceeds observation bounds")
        if knowledge.private_cards_per_player > self.bounds.max_hand_size:
            raise ValueError("private cards exceed environment hand bounds")
        maximum_possible_cash = (
            knowledge.starting_cash
            + (knowledge.action_counts[ActionId.LOAN10 - 1] * 10)
            + (knowledge.action_counts[ActionId.LOAN20 - 1] * 20)
        )
        if maximum_possible_cash > self.bounds.max_bid:
            raise ValueError("maximum possible loan cash exceeds environment bid bounds")
        if context.legal_max_amount is not None and context.legal_max_amount > self.bounds.max_bid:
            raise ValueError("legal maximum exceeds environment bounds")
        if context.revealable_count > self.bounds.max_hand_size:
            raise ValueError("revealable count exceeds environment bounds")


def _box(
    low: int,
    high: int,
    shape: tuple[int, ...],
    dtype: type[np.int8] | type[np.int16],
) -> gym.spaces.Box:
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _padded_vector(
    values: tuple[int, ...],
    length: int,
    dtype: type[np.generic],
) -> np.ndarray[tuple[int], np.dtype[np.generic]]:
    output = np.zeros(length, dtype=dtype)
    output[: len(values)] = values
    return output


def _padded_matrix(
    values: tuple[tuple[int, ...], ...],
    rows: int,
    columns: int,
    dtype: type[np.generic],
) -> np.ndarray[tuple[int, int], np.dtype[np.generic]]:
    output = np.zeros((rows, columns), dtype=dtype)
    for row, value in enumerate(values):
        output[row, : len(value)] = value
    return output


def _bitset(ids: tuple[int, ...]) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
    output = np.zeros(_MAX_OBJECTIVES, dtype=np.int8)
    for identifier in ids:
        output[identifier - 1] = 1
    return output


def _objective_matrix(
    ownership: tuple[tuple[int, ...], ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.int8]]:
    output = np.zeros((_MAX_SEATS, _MAX_OBJECTIVES), dtype=np.int8)
    for seat, objective_ids in enumerate(ownership):
        output[seat] = _bitset(objective_ids)
    return output
