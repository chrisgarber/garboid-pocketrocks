"""Canonical self-play rollout data structures and packing for PPO."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from pocketrocks import DecisionContext

from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.neural.encoding import (
    NeuralBatch,
    NeuralObservation,
)
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.simulator.session import SessionResult
from garboid_pocketrocks.training.rewards import RewardBreakdown


class RolloutError(ValueError):
    """Raised when a rollout cannot satisfy the training contract."""


@dataclass(frozen=True, slots=True)
class RolloutMetadata:
    """Public, immutable provenance attached to every learner transition."""

    ruleset_name: str
    player_count: int
    learner_seat: int
    opponent_names: tuple[str, ...]
    environment_seed: int
    opponent_seed: int
    policy_seed: int


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    """One learner decision and the old policy quantities needed by PPO."""

    observation: NeuralObservation
    context: DecisionContext
    action: int
    old_log_probability: float
    old_value: float
    reward: float
    reward_breakdown: RewardBreakdown
    terminated: bool
    truncated: bool
    bid_logits: tuple[float, ...]
    reveal_logits: tuple[float, ...]
    masked_logits: tuple[float, ...]
    illegal_probability: float
    metadata: RolloutMetadata


@dataclass(frozen=True, slots=True)
class SeatTrajectory:
    """One policy identity's complete trajectory for one game seat."""

    seat: int
    policy_identity: str
    trainable: bool
    transitions: tuple[RolloutTransition, ...]


@dataclass(frozen=True, slots=True)
class MultiSeatEpisode:
    """One complete self-play game with a separate trajectory per seat."""

    plan: SelfPlayEpisodePlan
    trajectories: tuple[SeatTrajectory, ...]
    result: SessionResult


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Complete episodes collected under one frozen policy."""

    episodes: tuple[MultiSeatEpisode, ...]

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        return tuple(
            transition
            for episode in self.episodes
            for trajectory in episode.trajectories
            if trajectory.trainable
            for transition in trajectory.transitions
        )

    @classmethod
    def from_multi_seat(
        cls,
        episodes: Iterable[MultiSeatEpisode],
    ) -> RolloutBatch:
        """Build a rollout whose PPO-visible transitions are trainable seats."""

        collected = tuple(episodes)
        if not collected:
            raise RolloutError("multi-seat rollout requires at least one episode")
        if not any(
            trajectory.trainable for episode in collected for trajectory in episode.trajectories
        ):
            raise RolloutError("multi-seat rollout contains no trainable trajectory")
        return cls(episodes=collected)


@dataclass(frozen=True, slots=True)
class PackedRollout:
    """One compact array per observation and training field."""

    global_ids: NDArray[np.int64]
    global_numeric: NDArray[np.float32]
    objective_bits: NDArray[np.float32]
    seat_numeric: NDArray[np.float32]
    seat_valid: NDArray[np.bool_]
    private_hand_ids: NDArray[np.int64]
    hand_valid: NDArray[np.bool_]
    history_ids: NDArray[np.int64]
    history_numeric: NDArray[np.float32]
    history_valid: NDArray[np.bool_]
    action_masks: NDArray[np.bool_]
    actions: NDArray[np.int64]
    old_log_probabilities: NDArray[np.float32]
    old_values: NDArray[np.float32]
    rewards: NDArray[np.float32]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    episode_indices: NDArray[np.int64]
    seats: NDArray[np.int64]
    chart_indices: NDArray[np.int64]
    player_counts: NDArray[np.int64]
    phase_buckets: NDArray[np.int64]
    trajectory_ranges: tuple[tuple[int, int], ...]

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def from_batch(cls, rollout: RolloutBatch) -> PackedRollout:
        """Stack a rollout exactly once while retaining trajectory boundaries."""

        trajectories: list[tuple[int, tuple[RolloutTransition, ...]]] = []
        for episode in rollout.episodes:
            for seat_trajectory in episode.trajectories:
                if seat_trajectory.trainable:
                    trajectories.append(
                        (
                            episode.plan.episode_index,
                            seat_trajectory.transitions,
                        )
                    )
        if not trajectories or any(not transitions for _, transitions in trajectories):
            raise RolloutError("packed rollout requires nonempty trajectories")

        transitions: list[RolloutTransition] = []
        episode_indices: list[int] = []
        ranges: list[tuple[int, int]] = []
        for episode_index, trajectory in trajectories:
            start = len(transitions)
            transitions.extend(trajectory)
            episode_indices.extend((episode_index,) * len(trajectory))
            ranges.append((start, len(transitions)))
            if trajectory[-1].terminated is not True or any(item.truncated for item in trajectory):
                raise RolloutError("packed rollout requires complete terminal trajectories")

        observations = tuple(item.observation for item in transitions)
        packed = cls(
            global_ids=np.stack([item.global_ids for item in observations]),
            global_numeric=np.stack([item.global_numeric for item in observations]),
            objective_bits=np.stack([item.objective_bits for item in observations]),
            seat_numeric=np.stack([item.seat_numeric for item in observations]),
            seat_valid=np.stack([item.seat_valid for item in observations]),
            private_hand_ids=np.stack([item.private_hand_ids for item in observations]),
            hand_valid=np.stack([item.hand_valid for item in observations]),
            history_ids=np.stack([item.history_ids for item in observations]),
            history_numeric=np.stack([item.history_numeric for item in observations]),
            history_valid=np.stack([item.history_valid for item in observations]),
            action_masks=np.stack([item.action_mask for item in observations]),
            actions=np.asarray([item.action for item in transitions], dtype=np.int64),
            old_log_probabilities=np.asarray(
                [item.old_log_probability for item in transitions],
                dtype=np.float32,
            ),
            old_values=np.asarray(
                [item.old_value for item in transitions],
                dtype=np.float32,
            ),
            rewards=np.asarray(
                [item.reward for item in transitions],
                dtype=np.float32,
            ),
            terminated=np.asarray(
                [item.terminated for item in transitions],
                dtype=np.bool_,
            ),
            truncated=np.asarray(
                [item.truncated for item in transitions],
                dtype=np.bool_,
            ),
            episode_indices=np.asarray(episode_indices, dtype=np.int64),
            seats=np.asarray(
                [item.metadata.learner_seat for item in transitions],
                dtype=np.int64,
            ),
            chart_indices=np.asarray(
                [ord(item.metadata.ruleset_name[-1]) - ord("A") for item in transitions],
                dtype=np.int64,
            ),
            player_counts=np.asarray(
                [item.metadata.player_count for item in transitions],
                dtype=np.int64,
            ),
            phase_buckets=np.asarray(
                [_phase_bucket(item) for item in transitions],
                dtype=np.int64,
            ),
            trajectory_ranges=tuple(ranges),
        )
        for value in (
            packed.global_ids,
            packed.global_numeric,
            packed.objective_bits,
            packed.seat_numeric,
            packed.seat_valid,
            packed.private_hand_ids,
            packed.hand_valid,
            packed.history_ids,
            packed.history_numeric,
            packed.history_valid,
            packed.action_masks,
            packed.actions,
            packed.old_log_probabilities,
            packed.old_values,
            packed.rewards,
            packed.terminated,
            packed.truncated,
            packed.episode_indices,
            packed.seats,
            packed.chart_indices,
            packed.player_counts,
            packed.phase_buckets,
        ):
            value.flags.writeable = False
        return packed

    def observation(self, index: int) -> NeuralObservation:
        """Return one observation as array views into packed storage."""

        if not 0 <= index < len(self):
            raise IndexError(index)
        return NeuralObservation(
            global_ids=self.global_ids[index],
            global_numeric=self.global_numeric[index],
            objective_bits=self.objective_bits[index],
            seat_numeric=self.seat_numeric[index],
            seat_valid=self.seat_valid[index],
            private_hand_ids=self.private_hand_ids[index],
            hand_valid=self.hand_valid[index],
            history_ids=self.history_ids[index],
            history_numeric=self.history_numeric[index],
            history_valid=self.history_valid[index],
            action_mask=self.action_masks[index],
        )

    def batch(
        self,
        indices: NDArray[np.int64],
        device: torch.device,
    ) -> NeuralBatch:
        """Gather one minibatch and transfer it to the learner device."""

        return NeuralBatch(
            global_ids=torch.as_tensor(self.global_ids[indices], device=device),
            global_numeric=torch.as_tensor(self.global_numeric[indices], device=device),
            objective_bits=torch.as_tensor(self.objective_bits[indices], device=device),
            seat_numeric=torch.as_tensor(self.seat_numeric[indices], device=device),
            seat_valid=torch.as_tensor(self.seat_valid[indices], device=device),
            private_hand_ids=torch.as_tensor(self.private_hand_ids[indices], device=device),
            hand_valid=torch.as_tensor(self.hand_valid[indices], device=device),
            history_ids=torch.as_tensor(self.history_ids[indices], device=device),
            history_numeric=torch.as_tensor(self.history_numeric[indices], device=device),
            history_valid=torch.as_tensor(self.history_valid[indices], device=device),
            action_mask=torch.as_tensor(self.action_masks[indices], device=device),
        )


def _phase_bucket(transition: RolloutTransition) -> int:
    valid_ids = transition.observation.history_ids[transition.observation.history_valid]
    opened_turns = int(np.count_nonzero(valid_ids[:, 0] == 2))
    knowledge = canonical_knowledge(
        transition.metadata.player_count,
        value_chart=transition.metadata.ruleset_name.removeprefix("live-"),
    )
    total_turns = sum(knowledge.action_counts) - (
        transition.metadata.player_count * knowledge.private_cards_per_player
    )
    if total_turns <= 0:
        return 0
    return min(2, ((opened_turns - 1) * 3) // total_turns)


def _immutable_observation(observation: NeuralObservation) -> NeuralObservation:
    return NeuralObservation(
        global_ids=_immutable_array(observation.global_ids),
        global_numeric=_immutable_array(observation.global_numeric),
        objective_bits=_immutable_array(observation.objective_bits),
        seat_numeric=_immutable_array(observation.seat_numeric),
        seat_valid=_immutable_array(observation.seat_valid),
        private_hand_ids=_immutable_array(observation.private_hand_ids),
        hand_valid=_immutable_array(observation.hand_valid),
        history_ids=_immutable_array(observation.history_ids),
        history_numeric=_immutable_array(observation.history_numeric),
        history_valid=_immutable_array(observation.history_valid),
        action_mask=_immutable_array(observation.action_mask),
    )


def _immutable_array[Scalar: np.generic](array: NDArray[Scalar]) -> NDArray[Scalar]:
    copied = array.copy()
    copied.flags.writeable = False
    return copied
