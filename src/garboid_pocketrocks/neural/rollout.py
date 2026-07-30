"""Deterministic live-A rollout collection for Stage 1 PPO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from pocketrocks import DecisionContext

from garboid_pocketrocks.bots.heuristic import (
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
)
from garboid_pocketrocks.neural.config import stage1_encoder_config
from garboid_pocketrocks.neural.encoding import (
    NeuralObservation,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.neural.policy import evaluate_masked_policy
from garboid_pocketrocks.neural.seeding import EpisodePlan
from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.model import GameResult
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.rewards import RewardBreakdown
from garboid_pocketrocks.training.single_agent_env import PocketRocksEnv

_STAGE1_PLAYER_COUNT = 3
_STAGE1_BOUNDS = EnvironmentBounds(max_bid=100, max_hand_size=5)
_STAGE1_OPPONENTS = (
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
)


class RolloutError(ValueError):
    """Raised when a requested rollout is outside the Stage 1 contract."""


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
class RolloutEpisode:
    """One complete game, retaining its trajectory boundary for GAE."""

    plan: EpisodePlan
    opponent_names: tuple[str, ...]
    transitions: tuple[RolloutTransition, ...]
    result: GameResult
    terminated: bool
    truncated: bool
    final_money: int
    rank: int
    outright_first: bool
    tied_first: bool
    reward_breakdown: RewardBreakdown


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
    result: GameResult


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Complete episodes collected under one frozen policy."""

    episodes: tuple[RolloutEpisode, ...]
    multi_seat_episodes: tuple[MultiSeatEpisode, ...] = ()

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        stage1 = tuple(
            transition
            for episode in self.episodes
            for transition in episode.transitions
        )
        self_play = tuple(
            transition
            for episode in self.multi_seat_episodes
            for trajectory in episode.trajectories
            if trajectory.trainable
            for transition in trajectory.transitions
        )
        return (*stage1, *self_play)

    @classmethod
    def from_multi_seat(
        cls,
        episodes: Sequence[MultiSeatEpisode],
    ) -> RolloutBatch:
        """Build a rollout whose PPO-visible transitions are trainable seats."""

        collected = tuple(episodes)
        if not collected:
            raise RolloutError("multi-seat rollout requires at least one episode")
        if not any(
            trajectory.trainable
            for episode in collected
            for trajectory in episode.trajectories
        ):
            raise RolloutError("multi-seat rollout contains no trainable trajectory")
        return cls(episodes=(), multi_seat_episodes=collected)


def collect_rollout(
    model: NeuralPolicy,
    plans: Sequence[EpisodePlan],
) -> RolloutBatch:
    """Collect complete live-A learner trajectories without updating the model."""

    if model.encoder_config != stage1_encoder_config():
        raise RolloutError("Stage 1 rollout requires the exact live-A/3 encoder config")
    if not plans:
        raise RolloutError("rollout requires at least one episode plan")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise RolloutError("rollout model has no parameters") from error
    if device.type != "cpu":
        raise RolloutError("Stage 1 rollout is fixed to CPU")

    encoder = NeuralObservationEncoder(model.encoder_config, _STAGE1_BOUNDS)
    was_training = model.training
    model.eval()
    try:
        episodes = tuple(_collect_episode(model, encoder, plan) for plan in plans)
    finally:
        model.train(was_training)
    return RolloutBatch(episodes=episodes)


def _collect_episode(
    model: NeuralPolicy,
    encoder: NeuralObservationEncoder,
    plan: EpisodePlan,
) -> RolloutEpisode:
    if not 0 <= plan.learner_seat < _STAGE1_PLAYER_COUNT:
        raise RolloutError("episode learner seat is outside Stage 1 player count")

    opponent_names = tuple(spec.name for spec in _STAGE1_OPPONENTS)
    metadata = RolloutMetadata(
        ruleset_name=LIVE_RULESET.name,
        player_count=_STAGE1_PLAYER_COUNT,
        learner_seat=plan.learner_seat,
        opponent_names=opponent_names,
        environment_seed=plan.environment_seed,
        opponent_seed=plan.opponent_seed,
        policy_seed=plan.policy_seed,
    )
    env = PocketRocksEnv(
        opponent_specs=_STAGE1_OPPONENTS,
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        player_count=_STAGE1_PLAYER_COUNT,
        bounds=_STAGE1_BOUNDS,
        learner_seat=plan.learner_seat,
    )
    env.reset(
        seed=plan.environment_seed,
        options={"opponent_seed": plan.opponent_seed},
    )
    generator = torch.Generator(device="cpu").manual_seed(plan.policy_seed)
    transitions: list[RolloutTransition] = []

    terminated = False
    truncated = False
    while not terminated and not truncated:
        context = env.learner_context
        observation = encoder.encode(
            context,
            env.ruleset_knowledge,
            env.public_history,
        )
        batch = batch_observations((observation,), torch.device("cpu"))
        with torch.no_grad():
            output = model(batch)
            selection = evaluate_masked_policy(
                output,
                batch,
                generator=generator,
                deterministic=False,
            )
        action = int(selection.actions[0].item())
        illegal_probability = float(selection.probabilities[0][~batch.action_mask[0]].sum().item())
        _, reward, terminated, truncated, info = env.step(action)
        breakdown = _reward_breakdown(info)
        transitions.append(
            RolloutTransition(
                observation=_immutable_observation(observation),
                context=context,
                action=action,
                old_log_probability=float(selection.log_probability[0].item()),
                old_value=float(selection.value[0].item()),
                reward=float(reward),
                reward_breakdown=breakdown,
                terminated=terminated,
                truncated=truncated,
                bid_logits=_tensor_row(output.bid_logits),
                reveal_logits=_tensor_row(output.reveal_logits),
                masked_logits=_tensor_row(selection.masked_logits),
                illegal_probability=illegal_probability,
                metadata=metadata,
            )
        )

    if truncated:
        raise RolloutError("Stage 1 environment unexpectedly truncated an episode")
    if env.transition is None or env.transition.result is None:
        raise RolloutError("Stage 1 environment stopped without a game result")
    result = env.transition.result
    scores_by_seat = {score.seat: score for score in result.scores}
    score = scores_by_seat[plan.learner_seat]
    first_place_count = sum(result_score.rank == 1 for result_score in result.scores)
    return RolloutEpisode(
        plan=plan,
        opponent_names=opponent_names,
        transitions=tuple(transitions),
        result=result,
        terminated=terminated,
        truncated=truncated,
        final_money=score.final_money,
        rank=score.rank,
        outright_first=score.rank == 1 and first_place_count == 1,
        tied_first=score.rank == 1 and first_place_count > 1,
        reward_breakdown=_sum_breakdowns(transitions),
    )


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


def _tensor_row(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor[0].detach().cpu().tolist())


def _reward_breakdown(info: dict[str, object]) -> RewardBreakdown:
    raw = info.get("reward_breakdown")
    if not isinstance(raw, dict):
        raise RolloutError("environment omitted its reward breakdown")
    try:
        return RewardBreakdown(
            accounting=float(raw["accounting"]),
            terminal_resource=float(raw["terminal_resource"]),
            placement=float(raw["placement"]),
            shaping=float(raw["shaping"]),
            penalty=float(raw["penalty"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RolloutError("environment returned an invalid reward breakdown") from error


def _sum_breakdowns(
    transitions: Sequence[RolloutTransition],
) -> RewardBreakdown:
    return RewardBreakdown(
        accounting=sum(item.reward_breakdown.accounting for item in transitions),
        terminal_resource=sum(item.reward_breakdown.terminal_resource for item in transitions),
        placement=sum(item.reward_breakdown.placement for item in transitions),
        shaping=sum(item.reward_breakdown.shaping for item in transitions),
        penalty=sum(item.reward_breakdown.penalty for item in transitions),
    )
