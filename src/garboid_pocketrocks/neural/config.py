"""JSON-safe configuration for neural observation encoding and models."""

from __future__ import annotations

from dataclasses import dataclass

from garboid_pocketrocks.rules import LIVE_RULESET


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
    """Checkpointed widths for the Stage 1 policy/value network."""

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


def stage1_encoder_config() -> NeuralEncoderConfig:
    """Return the exact live-A, three-player Stage 1 encoder contract."""

    player_count = 3
    setup = LIVE_RULESET.setup_for(player_count)
    history_bound = (
        1 + (2 * sum(LIVE_RULESET.action_counts)) + (player_count * setup.private_cards_per_player)
    )
    return NeuralEncoderConfig(
        schema_version=1,
        supported_ruleset_names=(LIVE_RULESET.name,),
        supported_player_counts=(player_count,),
        max_bid=100,
        max_hand_size=5,
        max_history_events=history_bound,
        max_cash=100,
        max_abs_chart=20,
        max_resource_cards=30,
        max_action_cards=30,
    )


def stage1_model_config() -> NeuralModelConfig:
    """Return the exact Stage 1 neural-network widths."""

    return NeuralModelConfig(
        categorical_embedding_size=8,
        suit_embedding_size=4,
        seat_hidden_size=32,
        event_embedding_size=64,
        gru_hidden_size=64,
        snapshot_hidden_size=128,
        trunk_hidden_size=128,
    )
