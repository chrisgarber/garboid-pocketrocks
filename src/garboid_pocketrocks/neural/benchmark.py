"""Balanced calibration candidates and measured throughput records."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from garboid_pocketrocks.neural.config import (
    stage1_model_config,
    training_encoder_config,
)
from garboid_pocketrocks.neural.devices import available_devices, resolve_device
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.parallel import collect_self_play_parallel
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.ppo import PPOTrainer
from garboid_pocketrocks.neural.run_config import TrainingRunConfig


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
    games_per_cell: int = 2,
) -> tuple[SelfPlayEpisodePlan, ...]:
    return plan_mirror_episodes(
        root_seed=root_seed,
        update_index=0,
        games_per_cell=games_per_cell,
        policy_identity="current",
    )


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
    worker_values = tuple(
        sorted({1, 2, 4, min(os.cpu_count() or 1, 8)})
    )
    candidates = tuple(
        BenchmarkCandidate(device, workers, active, batch)
        for device in devices
        for workers in worker_values
        for active in (2, 4)
        for batch in (64, 256)
    )
    results: list[BenchmarkResult] = []
    encoder_config = training_encoder_config()
    for candidate in candidates:
        device = resolve_device(candidate.device)
        torch.manual_seed(config.root_seed)
        model = NeuralPolicy(
            encoder_config,
            stage1_model_config(),
        ).to(device)
        try:
            rollout, collection = collect_self_play_parallel(
                {"current": model},
                plans,
                encoder_config=encoder_config,
                reward_config=config.reward,
                device=device,
                workers=candidate.workers,
                active_games_per_worker=candidate.active_games_per_worker,
                max_inference_batch=candidate.max_inference_batch,
                max_queue_delay_ms=config.parallel.max_queue_delay_ms,
            )
            ppo_started = time.perf_counter()
            PPOTrainer(
                model,
                replace(config.ppo, epochs=1),
            ).update(rollout, update_seed=config.root_seed)
            ppo_seconds = time.perf_counter() - ppo_started
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
