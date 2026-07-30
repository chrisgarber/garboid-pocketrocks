"""Batched, schedule-independent in-process neural self-play collection."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.neural.encoding import batch_observations
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import SelfPlayEpisodePlan
from garboid_pocketrocks.neural.policy import evaluate_row_seeded_policy
from garboid_pocketrocks.neural.rollout import MultiSeatEpisode, RolloutBatch
from garboid_pocketrocks.neural.self_play import (
    PendingPolicyRequest,
    PolicyResponse,
    SelfPlayGame,
)
from garboid_pocketrocks.training.rewards import RewardConfig


class CollectorError(ValueError):
    """Raised when a self-play collection request is inconsistent."""


@dataclass(frozen=True, slots=True)
class CollectorMetrics:
    """Throughput and batching measurements for one collection."""

    games: int
    decisions: int
    elapsed_seconds: float
    inference_seconds: float
    inference_batches: int
    inference_batch_sizes: tuple[int, ...]
    cell_games: tuple[tuple[str, int, int], ...]

    @property
    def games_per_second(self) -> float:
        return self.games / self.elapsed_seconds

    @property
    def decisions_per_second(self) -> float:
        return self.decisions / self.elapsed_seconds

    @property
    def mean_inference_batch_size(self) -> float:
        return self.decisions / self.inference_batches


def collect_self_play(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    active_games: int,
    max_inference_batch: int,
) -> tuple[RolloutBatch, CollectorMetrics]:
    """Collect complete all-seat games under frozen policies."""

    collected_plans = tuple(plans)
    _validate_collection(
        policies,
        collected_plans,
        encoder_config=encoder_config,
        device=device,
        active_games=active_games,
        max_inference_batch=max_inference_batch,
    )
    start = time.perf_counter()
    prior_modes = _freeze_policies(policies)
    remaining_index = 0
    active: dict[int, SelfPlayGame] = {}
    completed: list[MultiSeatEpisode] = []
    decisions = 0
    inference_seconds = 0.0
    inference_batch_sizes: list[int] = []

    def fill_active() -> None:
        nonlocal remaining_index
        while (
            len(active) < active_games
            and remaining_index < len(collected_plans)
        ):
            plan = collected_plans[remaining_index]
            remaining_index += 1
            active[plan.episode_index] = SelfPlayGame.start(
                plan,
                encoder_config=encoder_config,
                reward_config=reward_config,
            )

    try:
        fill_active()
        while active:
            requests = sorted(
                (
                    request
                    for game in active.values()
                    for request in game.pending_requests()
                ),
                key=lambda request: (
                    request.policy_identity,
                    request.episode_index,
                    request.seat,
                    request.decision_index,
                ),
            )
            decisions += len(requests)
            responses_by_episode: dict[int, list[PolicyResponse]] = defaultdict(list)
            by_policy: dict[str, list[PendingPolicyRequest]] = defaultdict(list)
            for request in requests:
                by_policy[request.policy_identity].append(request)

            for identity in sorted(by_policy):
                model = policies[identity]
                policy_requests = by_policy[identity]
                for offset in range(0, len(policy_requests), max_inference_batch):
                    chunk = policy_requests[offset : offset + max_inference_batch]
                    inference_start = time.perf_counter()
                    batch = batch_observations(
                        tuple(request.observation for request in chunk),
                        device,
                    )
                    with torch.no_grad():
                        output = model(batch)
                        selection = evaluate_row_seeded_policy(
                            output,
                            batch,
                            row_seeds=tuple(
                                request.sampling_seed for request in chunk
                            ),
                        )
                    inference_seconds += time.perf_counter() - inference_start
                    inference_batch_sizes.append(len(chunk))
                    for row, request in enumerate(chunk):
                        responses_by_episode[request.episode_index].append(
                            PolicyResponse(
                                episode_index=request.episode_index,
                                seat=request.seat,
                                decision_index=request.decision_index,
                                action=int(selection.actions[row].item()),
                                old_log_probability=float(
                                    selection.log_probability[row].item()
                                ),
                                old_value=float(selection.value[row].item()),
                            )
                        )

            for episode_index in tuple(sorted(active)):
                game = active[episode_index]
                game.apply(responses_by_episode[episode_index])
                if game.terminated:
                    completed.append(game.episode())
                    del active[episode_index]
            fill_active()
    finally:
        _restore_policy_modes(prior_modes)

    elapsed_seconds = time.perf_counter() - start
    completed.sort(key=lambda episode: episode.plan.episode_index)
    cell_counts = Counter(
        (episode.plan.ruleset_name, episode.plan.player_count)
        for episode in completed
    )
    metrics = CollectorMetrics(
        games=len(completed),
        decisions=decisions,
        elapsed_seconds=elapsed_seconds,
        inference_seconds=inference_seconds,
        inference_batches=len(inference_batch_sizes),
        inference_batch_sizes=tuple(inference_batch_sizes),
        cell_games=tuple(
            (ruleset_name, player_count, count)
            for (ruleset_name, player_count), count in sorted(cell_counts.items())
        ),
    )
    return RolloutBatch.from_multi_seat(completed), metrics


def _validate_collection(
    policies: Mapping[str, NeuralPolicy],
    plans: tuple[SelfPlayEpisodePlan, ...],
    *,
    encoder_config: NeuralEncoderConfig,
    device: torch.device,
    active_games: int,
    max_inference_batch: int,
) -> None:
    if not plans:
        raise CollectorError("self-play collection requires at least one plan")
    if (
        not isinstance(active_games, int)
        or isinstance(active_games, bool)
        or active_games <= 0
    ):
        raise CollectorError("active_games must be a positive integer")
    if (
        not isinstance(max_inference_batch, int)
        or isinstance(max_inference_batch, bool)
        or max_inference_batch <= 0
    ):
        raise CollectorError("max_inference_batch must be a positive integer")
    episode_indices = tuple(plan.episode_index for plan in plans)
    if len(set(episode_indices)) != len(episode_indices):
        raise CollectorError("episode indices must be unique within a collection")
    required_identities = {
        assignment.identity
        for plan in plans
        for assignment in plan.seat_policies
    }
    missing = required_identities - policies.keys()
    if missing:
        raise CollectorError(f"missing policies for identities: {sorted(missing)!r}")
    for identity in required_identities:
        model = policies[identity]
        if model.encoder_config != encoder_config:
            raise CollectorError(f"policy {identity!r} has an incompatible encoder")
        try:
            model_device = next(model.parameters()).device
        except StopIteration as error:
            raise CollectorError(f"policy {identity!r} has no parameters") from error
        if model_device != device:
            raise CollectorError(
                f"policy {identity!r} is on {model_device}, expected {device}"
            )


def _freeze_policies(
    policies: Mapping[str, NeuralPolicy],
) -> tuple[tuple[NeuralPolicy, bool], ...]:
    unique: dict[int, NeuralPolicy] = {}
    for model in policies.values():
        unique[id(model)] = model
    prior_modes = tuple((model, model.training) for model in unique.values())
    for model, _ in prior_modes:
        model.eval()
    return prior_modes


def _restore_policy_modes(
    prior_modes: tuple[tuple[NeuralPolicy, bool], ...],
) -> None:
    for model, was_training in prior_modes:
        model.train(was_training)
