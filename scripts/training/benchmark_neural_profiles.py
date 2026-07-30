"""Benchmark neural model capacities on real balanced self-play rollouts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from garboid_pocketrocks.neural.config import (
    ModelProfile,
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.devices import (
    available_devices,
    resolve_device,
    synchronize_device,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.planning import plan_mirror_episodes
from garboid_pocketrocks.neural.ppo import PPOConfig, PPOTrainer
from garboid_pocketrocks.neural.seeding import configure_torch_runtime
from garboid_pocketrocks.neural.vector_collector import (
    collect_self_play_vectorized,
)
from garboid_pocketrocks.training.rewards import RewardConfig


@dataclass(frozen=True, slots=True)
class ProfileBenchmark:
    profile: ModelProfile
    device: str
    parameters: int
    games: int
    decisions: int
    collection_seconds: float
    collection_games_per_second: float
    collection_decisions_per_second: float
    ppo_seconds: float


def benchmark(
    *,
    profiles: Sequence[ModelProfile],
    devices: Sequence[str],
    games_per_cell: int,
    root_seed: int,
    engine_batch_size: int,
) -> tuple[ProfileBenchmark, ...]:
    """Measure one real collection and PPO epoch for every requested pair."""

    configure_torch_runtime(root_seed, deterministic_algorithms=False)
    encoder_config = training_encoder_config()
    plans = plan_mirror_episodes(
        root_seed=root_seed,
        update_index=0,
        games_per_cell=games_per_cell,
        policy_identity="current",
    )
    results: list[ProfileBenchmark] = []
    for profile in profiles:
        for device_name in devices:
            device = resolve_device(device_name)
            torch.manual_seed(root_seed)
            model = NeuralPolicy(
                encoder_config,
                training_model_config(profile),
            ).to(device)
            rollout, collection = collect_self_play_vectorized(
                {"current": model},
                plans,
                encoder_config=encoder_config,
                reward_config=RewardConfig(),
                device=device,
                engine_batch_size=engine_batch_size,
                max_inference_batch=1024,
            )
            synchronize_device(device)
            ppo_started = time.perf_counter()
            PPOTrainer(model, PPOConfig(epochs=1)).update(
                rollout,
                update_seed=root_seed,
            )
            synchronize_device(device)
            results.append(
                ProfileBenchmark(
                    profile=profile,
                    device=device_name,
                    parameters=sum(parameter.numel() for parameter in model.parameters()),
                    games=collection.games,
                    decisions=collection.decisions,
                    collection_seconds=collection.elapsed_seconds,
                    collection_games_per_second=collection.games_per_second,
                    collection_decisions_per_second=(collection.decisions_per_second),
                    ppo_seconds=time.perf_counter() - ppo_started,
                )
            )
            print(json.dumps(asdict(results[-1]), sort_keys=True), flush=True)
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("small", "medium", "large"),
        default=("small", "medium", "large"),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "cuda", "mps"),
        default=available_devices(),
    )
    parser.add_argument("--games-per-cell", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine-batch-size", type=int, default=128)
    arguments = parser.parse_args()
    benchmark(
        profiles=arguments.profiles,
        devices=arguments.devices,
        games_per_cell=arguments.games_per_cell,
        root_seed=arguments.seed,
        engine_batch_size=arguments.engine_batch_size,
    )


if __name__ == "__main__":
    main()
