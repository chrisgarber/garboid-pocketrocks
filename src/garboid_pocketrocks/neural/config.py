"""JSON-safe configuration for neural observation encoding and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from garboid_pocketrocks.knowledge import canonical_knowledge

ModelProfile = Literal["small", "medium", "large"]


@dataclass(frozen=True, slots=True)
class NeuralEncoderConfig:
    """Checkpointed support and normalization limits for an encoder."""

    schema_version: int
    supported_ruleset_names: tuple[str, ...]
    supported_player_counts: tuple[int, ...]
    max_bid: int
    max_hand_size: int
    max_history_events: int
    max_cash: int
    max_abs_chart: int
    max_resource_cards: int
    max_action_cards: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_ruleset_names",
            tuple(self.supported_ruleset_names),
        )
        object.__setattr__(
            self,
            "supported_player_counts",
            tuple(self.supported_player_counts),
        )
        if self.schema_version != 1:
            raise ValueError("unsupported neural encoder schema version")
        if (
            not self.supported_ruleset_names
            or any(not name for name in self.supported_ruleset_names)
            or len(set(self.supported_ruleset_names)) != len(self.supported_ruleset_names)
        ):
            raise ValueError("supported ruleset names must be unique and nonempty")
        if (
            not self.supported_player_counts
            or any(not 3 <= count <= 5 for count in self.supported_player_counts)
            or len(set(self.supported_player_counts)) != len(self.supported_player_counts)
        ):
            raise ValueError("supported player counts must be unique values from three to five")
        if self.max_bid < 0:
            raise ValueError("maximum bid must be nonnegative")
        if self.max_hand_size <= 0:
            raise ValueError("maximum hand size must be positive")
        for name in (
            "max_history_events",
            "max_cash",
            "max_abs_chart",
            "max_resource_cards",
            "max_action_cards",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    """Checkpointed widths for the policy/value network."""

    categorical_embedding_size: int
    suit_embedding_size: int
    seat_hidden_size: int
    event_embedding_size: int
    gru_hidden_size: int
    snapshot_hidden_size: int
    trunk_hidden_size: int

    def __post_init__(self) -> None:
        for name in (
            "categorical_embedding_size",
            "suit_embedding_size",
            "seat_hidden_size",
            "event_embedding_size",
            "gru_hidden_size",
            "snapshot_hidden_size",
            "trunk_hidden_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def training_model_config(profile: ModelProfile) -> NeuralModelConfig:
    """Return a checkpoint-stable capacity profile for self-play training."""

    if profile == "small":
        return NeuralModelConfig(
            categorical_embedding_size=8,
            suit_embedding_size=4,
            seat_hidden_size=32,
            event_embedding_size=64,
            gru_hidden_size=64,
            snapshot_hidden_size=128,
            trunk_hidden_size=128,
        )
    if profile == "medium":
        return NeuralModelConfig(
            categorical_embedding_size=16,
            suit_embedding_size=8,
            seat_hidden_size=64,
            event_embedding_size=128,
            gru_hidden_size=128,
            snapshot_hidden_size=256,
            trunk_hidden_size=256,
        )
    if profile == "large":
        return NeuralModelConfig(
            categorical_embedding_size=32,
            suit_embedding_size=16,
            seat_hidden_size=128,
            event_embedding_size=256,
            gru_hidden_size=256,
            snapshot_hidden_size=512,
            trunk_hidden_size=512,
        )
    raise ValueError(f"unknown model profile {profile!r}")


def training_encoder_config() -> NeuralEncoderConfig:
    """Return the finite live-chart, three-to-five-player training contract."""

    player_counts = (3, 4, 5)
    variants = tuple(
        canonical_knowledge(player_count, value_chart=chart)
        for chart in "ABCDE"
        for player_count in player_counts
    )
    max_history_events = max(
        1
        + (2 * sum(knowledge.action_counts))
        + (knowledge.player_count * knowledge.private_cards_per_player)
        for knowledge in variants
    )
    return NeuralEncoderConfig(
        schema_version=1,
        supported_ruleset_names=tuple(f"live-{chart}" for chart in "ABCDE"),
        supported_player_counts=player_counts,
        max_bid=100,
        max_hand_size=5,
        max_history_events=max_history_events,
        max_cash=100,
        max_abs_chart=20,
        max_resource_cards=max(sum(knowledge.resource_counts) for knowledge in variants),
        max_action_cards=max(sum(knowledge.action_counts) for knowledge in variants),
    )
