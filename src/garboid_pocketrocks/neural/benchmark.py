"""Balanced calibration candidates and measured throughput records."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from garboid_pocketrocks.neural.config import (
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.devices import (
    available_devices,
    resolve_device,
    synchronize_device,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.ppo import PPOTrainer
from garboid_pocketrocks.neural.run_config import TrainingRunConfig
from garboid_pocketrocks.neural.vector_collector import (
    collect_self_play_vectorized,
)
from garboid_pocketrocks.neural.vector_parallel import (
    collect_self_play_vectorized_parallel,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    device: str
    workers: int
    active_games_per_worker: int
    max_inference_batch: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    candidate: BenchmarkCandidate
    games: int
    decisions: int
    elapsed_seconds: float
    ppo_seconds: float
    total_seconds: float
    games_per_second: float
    decisions_per_second: float
    inference_batch_p50: float
    inference_batch_p95: float
    peak_rss_bytes: int | None


def calibration_plans(
    *,
    root_seed: int,
    games_per_cell: int = 64,
) -> tuple[SelfPlayEpisodePlan, ...]:
    return plan_mirror_episodes(
        root_seed=root_seed,
        update_index=0,
        games_per_cell=games_per_cell,
        policy_identity="current",
    )


def calibration_candidates(
    devices: Sequence[str],
) -> tuple[BenchmarkCandidate, ...]:
    """Return the bounded vector-engine/device calibration matrix."""

    candidates: list[BenchmarkCandidate] = []
    for device in devices:
        if device == "cpu":
            candidates.extend(
                (
                    BenchmarkCandidate(
                        device="cpu",
                        workers=1,
                        active_games_per_worker=128,
                        max_inference_batch=1024,
                    ),
                    BenchmarkCandidate(
                        device="cpu",
                        workers=8,
                        active_games_per_worker=64,
                        max_inference_batch=1024,
                    ),
                    BenchmarkCandidate(
                        device="cpu",
                        workers=8,
                        active_games_per_worker=128,
                        max_inference_batch=1024,
                    ),
                )
            )
        else:
            candidates.extend(
                BenchmarkCandidate(
                    device=device,
                    workers=1,
                    active_games_per_worker=engine_batch_size,
                    max_inference_batch=1024,
                )
                for engine_batch_size in (64, 128)
            )
    return tuple(candidates)


def choose_candidate(
    results: tuple[BenchmarkResult, ...],
) -> BenchmarkCandidate:
    if not results:
        raise ValueError("calibration requires a successful candidate")
    return min(
        results,
        key=lambda result: (
            -result.decisions_per_second,
            result.total_seconds,
            result.candidate.workers,
            result.candidate.device,
        ),
    ).candidate


def calibrate(
    config: TrainingRunConfig,
    *,
    plans: Sequence[SelfPlayEpisodePlan],
) -> tuple[BenchmarkCandidate, tuple[BenchmarkResult, ...]]:
    """Measure complete collection-plus-PPO paths and select the fastest."""

    devices = (
        available_devices()
        if config.device == "auto"
        else (resolve_device(config.device).type,)
    )
    candidates = calibration_candidates(devices)
    results: list[BenchmarkResult] = []
    ppo_seconds_by_device: dict[str, float] = {}
    encoder_config = training_encoder_config()
    for candidate in candidates:
        device = resolve_device(candidate.device)
        torch.manual_seed(config.root_seed)
        model = NeuralPolicy(
            encoder_config,
            training_model_config(config.model_profile),
        ).to(device)
        try:
            if candidate.workers == 1:
                rollout, collection = collect_self_play_vectorized(
                    {"current": model},
                    plans,
                    encoder_config=encoder_config,
                    reward_config=config.reward,
                    device=device,
                    engine_batch_size=candidate.active_games_per_worker,
                    max_inference_batch=candidate.max_inference_batch,
                )
            else:
                rollout, collection = (
                    collect_self_play_vectorized_parallel(
                        {"current": model},
                        plans,
                        encoder_config=encoder_config,
                        reward_config=config.reward,
                        workers=candidate.workers,
                        engine_batch_size=(
                            candidate.active_games_per_worker
                        ),
                        max_inference_batch=(
                            candidate.max_inference_batch
                        ),
                    )
                )
            ppo_seconds = ppo_seconds_by_device.get(candidate.device)
            if ppo_seconds is None:
                synchronize_device(device)
                ppo_started = time.perf_counter()
                PPOTrainer(
                    model,
                    config.ppo,
                ).update(rollout, update_seed=config.root_seed)
                synchronize_device(device)
                ppo_seconds = time.perf_counter() - ppo_started
                ppo_seconds_by_device[candidate.device] = ppo_seconds
        except (RuntimeError, ValueError):
            continue
        total = collection.elapsed_seconds + ppo_seconds
        results.append(
            BenchmarkResult(
                candidate=candidate,
                games=collection.games,
                decisions=collection.decisions,
                elapsed_seconds=collection.elapsed_seconds,
                ppo_seconds=ppo_seconds,
                total_seconds=total,
                games_per_second=collection.games / total,
                decisions_per_second=collection.decisions / total,
                inference_batch_p50=collection.inference_batch_p50,
                inference_batch_p95=collection.inference_batch_p95,
                peak_rss_bytes=None,
            )
        )
    selected = choose_candidate(tuple(results))
    return selected, tuple(results)
