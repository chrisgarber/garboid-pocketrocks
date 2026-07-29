"""Stateless full-history recurrent policy/value model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from garboid_pocketrocks.neural.config import (
    NeuralEncoderConfig,
    NeuralModelConfig,
)
from garboid_pocketrocks.neural.encoding import NeuralBatch

_MAX_SEATS = 5
_SEAT_NUMERIC_SIZE = 41
_GLOBAL_NUMERIC_SIZE = 21
_OBJECTIVE_BIT_SIZE = 60
_HISTORY_NUMERIC_SIZE = 42


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Raw phase-specific policy heads and scalar values."""

    bid_logits: Tensor
    reveal_logits: Tensor
    value: Tensor


class NeuralPolicy(nn.Module):
    """Encode a complete observation from a new zero GRU state on every call."""

    def __init__(
        self,
        encoder_config: NeuralEncoderConfig,
        model_config: NeuralModelConfig,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.model_config = model_config

        categorical_size = model_config.categorical_embedding_size
        suit_size = model_config.suit_embedding_size

        self.phase_embedding = nn.Embedding(3, categorical_size)
        self.player_count_embedding = nn.Embedding(6, categorical_size)
        self.current_action_embedding = nn.Embedding(
            7,
            categorical_size,
            padding_idx=0,
        )
        self.current_resource_0_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )
        self.current_resource_1_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )
        self.priority_embedding = nn.Embedding(5, categorical_size)
        self.private_hand_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )

        snapshot_input_size = (
            (4 * categorical_size)
            + (2 * suit_size)
            + _GLOBAL_NUMERIC_SIZE
            + _OBJECTIVE_BIT_SIZE
            + (encoder_config.max_hand_size * suit_size)
        )
        self.snapshot_encoder = nn.Sequential(
            nn.Linear(snapshot_input_size, model_config.snapshot_hidden_size),
            nn.Tanh(),
        )

        self.seat_encoder = nn.Sequential(
            nn.Linear(_SEAT_NUMERIC_SIZE, model_config.seat_hidden_size),
            nn.Tanh(),
        )

        self.event_kind_embedding = nn.Embedding(
            5,
            categorical_size,
            padding_idx=0,
        )
        self.event_action_embedding = nn.Embedding(
            7,
            categorical_size,
            padding_idx=0,
        )
        self.event_resource_0_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )
        self.event_resource_1_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )
        self.event_actor_embedding = nn.Embedding(
            6,
            categorical_size,
            padding_idx=0,
        )
        self.event_revealed_suit_embedding = nn.Embedding(
            6,
            suit_size,
            padding_idx=0,
        )
        event_input_size = (3 * categorical_size) + (3 * suit_size) + _HISTORY_NUMERIC_SIZE
        self.event_encoder = nn.Sequential(
            nn.Linear(event_input_size, model_config.event_embedding_size),
            nn.Tanh(),
        )
        self.history_gru = nn.GRU(
            input_size=model_config.event_embedding_size,
            hidden_size=model_config.gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        trunk_input_size = (
            model_config.snapshot_hidden_size
            + (_MAX_SEATS * model_config.seat_hidden_size)
            + model_config.gru_hidden_size
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_size, model_config.trunk_hidden_size),
            nn.Tanh(),
            nn.Linear(model_config.trunk_hidden_size, model_config.trunk_hidden_size),
            nn.Tanh(),
        )
        self.bid_head = nn.Linear(
            model_config.trunk_hidden_size,
            encoder_config.max_bid + 1,
        )
        self.reveal_head = nn.Linear(
            model_config.trunk_hidden_size,
            encoder_config.max_hand_size + 1,
        )
        self.value_head = nn.Linear(model_config.trunk_hidden_size, 1)

    def forward(self, batch: NeuralBatch) -> PolicyValueOutput:
        """Replay the entire padded history from an internal zero state."""

        snapshot = self._encode_snapshot(batch)
        seats = self.seat_encoder(batch.seat_numeric)
        seats = seats * batch.seat_valid.unsqueeze(-1).to(seats.dtype)
        seats = seats.flatten(start_dim=1)
        history = self._encode_history(batch)
        trunk = self.trunk(torch.cat((snapshot, seats, history), dim=-1))
        return PolicyValueOutput(
            bid_logits=self.bid_head(trunk),
            reveal_logits=self.reveal_head(trunk),
            value=self.value_head(trunk).squeeze(-1),
        )

    def _encode_snapshot(self, batch: NeuralBatch) -> Tensor:
        hand = self.private_hand_embedding(batch.private_hand_ids)
        hand = hand * batch.hand_valid.unsqueeze(-1).to(hand.dtype)
        categorical = (
            self.phase_embedding(batch.global_ids[:, 0]),
            self.player_count_embedding(batch.global_ids[:, 1]),
            self.current_action_embedding(batch.global_ids[:, 2]),
            self.current_resource_0_embedding(batch.global_ids[:, 3]),
            self.current_resource_1_embedding(batch.global_ids[:, 4]),
            self.priority_embedding(batch.global_ids[:, 5]),
        )
        encoded = self.snapshot_encoder(
            torch.cat(
                (
                    *categorical,
                    batch.global_numeric,
                    batch.objective_bits,
                    hand.flatten(start_dim=1),
                ),
                dim=-1,
            )
        )
        return cast(Tensor, encoded)

    def _encode_history(self, batch: NeuralBatch) -> Tensor:
        event_inputs = torch.cat(
            (
                self.event_kind_embedding(batch.history_ids[:, :, 0]),
                self.event_action_embedding(batch.history_ids[:, :, 1]),
                self.event_resource_0_embedding(batch.history_ids[:, :, 2]),
                self.event_resource_1_embedding(batch.history_ids[:, :, 3]),
                self.event_actor_embedding(batch.history_ids[:, :, 4]),
                self.event_revealed_suit_embedding(batch.history_ids[:, :, 5]),
                batch.history_numeric,
            ),
            dim=-1,
        )
        events = self.event_encoder(event_inputs)
        batch_size = events.shape[0]
        zero_hidden = torch.zeros(
            (1, batch_size, self.model_config.gru_hidden_size),
            dtype=events.dtype,
            device=events.device,
        )
        history_outputs, _ = self.history_gru(events, zero_hidden)
        lengths = batch.history_valid.to(torch.int64).sum(dim=-1)
        if not torch.all(lengths > 0).item():
            raise ValueError("every neural observation requires public history")
        rows = torch.arange(batch_size, device=events.device)
        return cast(Tensor, history_outputs[rows, lengths - 1])
