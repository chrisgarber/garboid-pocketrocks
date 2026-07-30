"""Durable update-boundary orchestration for neural self-play training."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch

from garboid_pocketrocks.neural.benchmark import (
    BenchmarkCandidate,
    BenchmarkResult,
    calibrate,
    calibration_plans,
)
from garboid_pocketrocks.neural.collector import (
    CollectorMetrics,
    collect_self_play,
)
from garboid_pocketrocks.neural.config import (
    stage1_model_config,
    training_encoder_config,
)
from garboid_pocketrocks.neural.devices import resolve_device
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.parallel import collect_self_play_parallel
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
)
from garboid_pocketrocks.neural.ppo import PPOTrainer, PPOUpdateMetrics
from garboid_pocketrocks.neural.rollout import RolloutBatch
from garboid_pocketrocks.neural.run_config import (
    DeviceName,
    ParallelConfig,
    TrainingRunConfig,
)
from garboid_pocketrocks.neural.seeding import (
    configure_deterministic_torch,
    derive_seed,
)
from garboid_pocketrocks.neural.training_checkpoint import (
    TrainingCheckpointManifest,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TrainerError(ValueError):
    """Raised when a durable training invocation is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_dir: Path
    final_checkpoint: Path
    completed_updates: int
    completed_episodes: int
    completed_decisions: int
    elapsed_seconds: float


def train(
    config: TrainingRunConfig,
    output_dir: Path,
) -> TrainingRunResult:
    """Start a new self-play lineage and stop only at an update boundary."""

    run_dir = _prepare_run_dir(output_dir)
    configure_deterministic_torch(config.root_seed)
    candidate, benchmarks = _resolve_candidate(config)
    resolved = _resolved_config(config, candidate)
    _write_json(run_dir / "resolved-config.json", resolved.to_json_dict())
    _write_json(
        run_dir / "benchmark.json",
        {
            "selected": asdict(candidate),
            "results": [asdict(result) for result in benchmarks],
        },
    )
    device = resolve_device(candidate.device)
    torch.manual_seed(derive_seed(config.root_seed, "model"))
    model = NeuralPolicy(
        training_encoder_config(),
        stage1_model_config(),
    ).to(device)
    trainer = PPOTrainer(model, resolved.ppo)
    selected_result = next(
        (
            result
            for result in benchmarks
            if result.candidate == candidate
        ),
        None,
    )
    estimated_decisions_per_game = (
        selected_result.decisions / selected_result.games
        if selected_result is not None
        else 20.0
    )
    return _run_updates(
        resolved,
        run_dir,
        model=model,
        trainer=trainer,
        initial_progress=TrainingProgress(0, 0, 0, ()),
        lineage=(),
        max_updates_this_run=resolved.max_updates,
        games_per_cell=resolve_games_per_cell(
            resolved,
            estimated_decisions_per_game=estimated_decisions_per_game,
        ),
    )


def resume(
    checkpoint: Path,
    output_dir: Path,
    *,
    max_additional_updates: int | None = None,
) -> TrainingRunResult:
    """Resume exact model and optimizer state into a new run directory."""

    run_dir = _prepare_run_dir(output_dir)
    loaded = load_training_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    config = loaded.manifest.run_config
    device = resolve_device(config.device)
    if device.type != "cpu":
        loaded = load_training_checkpoint(checkpoint, device=device)
    _write_json(run_dir / "resolved-config.json", config.to_json_dict())
    _write_json(
        run_dir / "benchmark.json",
        {"resumed_from": str(checkpoint.resolve())},
    )
    trainer = PPOTrainer(loaded.model, config.ppo)
    trainer.optimizer = loaded.optimizer
    limit = max_additional_updates
    if limit is not None:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
        ):
            raise TrainerError("max_additional_updates must be positive")
        limit += loaded.manifest.progress.next_update_index
    return _run_updates(
        config,
        run_dir,
        model=loaded.model,
        trainer=trainer,
        initial_progress=loaded.manifest.progress,
        lineage=(
            *loaded.manifest.lineage,
            str(checkpoint.resolve()),
        ),
        max_updates_this_run=limit,
        games_per_cell=resolve_games_per_cell(
            config,
            estimated_decisions_per_game=(
                loaded.manifest.progress.completed_decisions
                / loaded.manifest.progress.completed_episodes
            ),
        ),
    )


def inspect_checkpoint(checkpoint: Path) -> dict[str, object]:
    """Return the concise durable progress and support contract."""

    loaded = load_training_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    manifest = loaded.manifest
    return {
        "repository_commit": manifest.repository_commit,
        "next_update_index": manifest.progress.next_update_index,
        "completed_updates": manifest.progress.next_update_index,
        "completed_episodes": manifest.progress.completed_episodes,
        "completed_decisions": manifest.progress.completed_decisions,
        "cell_games": manifest.progress.cell_games,
        "device": manifest.run_config.device,
        "supported_ruleset_names": manifest.encoder_config.supported_ruleset_names,
        "supported_player_counts": manifest.encoder_config.supported_player_counts,
        "parameter_digest": manifest.parameter_digest,
        "lineage": manifest.lineage,
    }


def _run_updates(
    config: TrainingRunConfig,
    run_dir: Path,
    *,
    model: NeuralPolicy,
    trainer: PPOTrainer,
    initial_progress: TrainingProgress,
    lineage: tuple[str, ...],
    max_updates_this_run: int | None,
    games_per_cell: int,
) -> TrainingRunResult:
    started = time.perf_counter()
    update_index = initial_progress.next_update_index
    completed_episodes = initial_progress.completed_episodes
    completed_decisions = initial_progress.completed_decisions
    cells = Counter(
        {
            (ruleset, players): games
            for ruleset, players, games in initial_progress.cell_games
        }
    )
    durations: list[float] = []
    checkpoint = run_dir / "checkpoints" / "latest"

    while True:
        if max_updates_this_run is not None and update_index >= max_updates_this_run:
            break
        elapsed = time.perf_counter() - started
        if config.max_wall_seconds is not None:
            remaining = config.max_wall_seconds - elapsed
            if remaining <= 0.0:
                break
            if durations and remaining < max(durations[-5:]):
                break
        update_started = time.perf_counter()
        plans = plan_mirror_episodes(
            root_seed=config.root_seed,
            update_index=update_index,
            games_per_cell=games_per_cell,
            policy_identity="current",
        )
        snapshot = NeuralPolicy(
            model.encoder_config,
            model.model_config,
        ).to(trainer.device)
        snapshot.load_state_dict(model.state_dict())
        snapshot.eval()
        collection = _collect(config, snapshot, plans)
        ppo = trainer.update(
            collection[0],
            update_seed=derive_seed(
                config.root_seed,
                "minibatch",
                update_index,
            ),
        )
        completed_episodes += collection[1].games
        completed_decisions += collection[1].decisions
        for ruleset, players, games in collection[1].cell_games:
            cells[(ruleset, players)] += games
        update_index += 1
        duration = time.perf_counter() - update_started
        durations.append(duration)
        progress = TrainingProgress(
            update_index,
            completed_episodes,
            completed_decisions,
            tuple(
                (ruleset, players, games)
                for (ruleset, players), games in sorted(cells.items())
            ),
        )
        metrics = _update_metrics(
            update_index - 1,
            games_per_cell,
            duration,
            collection[1],
            ppo,
        )
        _append_json_line(run_dir / "metrics.jsonl", metrics)
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=trainer.optimizer,
            manifest=TrainingCheckpointManifest(
                schema_version=1,
                repository_commit=_repository_commit(),
                encoder_config=model.encoder_config,
                model_config=model.model_config,
                run_config=config,
                progress=progress,
                lineage=lineage,
                champion_identity=None,
                league_identities=(),
            ),
            generator_states={"torch": torch.get_rng_state()},
            metrics=metrics,
        )

    if update_index == initial_progress.next_update_index:
        raise TrainerError("training budget did not permit one complete update")
    return TrainingRunResult(
        run_dir=run_dir,
        final_checkpoint=checkpoint,
        completed_updates=update_index,
        completed_episodes=completed_episodes,
        completed_decisions=completed_decisions,
        elapsed_seconds=time.perf_counter() - started,
    )


def _collect(
    config: TrainingRunConfig,
    model: NeuralPolicy,
    plans: tuple[SelfPlayEpisodePlan, ...],
) -> tuple[RolloutBatch, CollectorMetrics]:
    # Kept in one function so all training updates share the same selection rule.
    device = next(model.parameters()).device
    workers = config.parallel.workers
    if workers == 1:
        return collect_self_play(
            {"current": model},
            plans,
            encoder_config=model.encoder_config,
            reward_config=config.reward,
            device=device,
            active_games=config.parallel.active_games_per_worker,
            max_inference_batch=config.parallel.max_inference_batch,
        )
    if workers == "auto":
        raise TrainerError("training workers must be resolved before collection")
    return collect_self_play_parallel(
        {"current": model},
        plans,
        encoder_config=model.encoder_config,
        reward_config=config.reward,
        device=device,
        workers=workers,
        active_games_per_worker=config.parallel.active_games_per_worker,
        max_inference_batch=config.parallel.max_inference_batch,
        max_queue_delay_ms=config.parallel.max_queue_delay_ms,
    )


def _resolve_candidate(
    config: TrainingRunConfig,
) -> tuple[BenchmarkCandidate, tuple[BenchmarkResult, ...]]:
    if config.device != "auto" and config.parallel.workers != "auto":
        resolve_device(config.device)
        return (
            BenchmarkCandidate(
                config.device,
                config.parallel.workers,
                config.parallel.active_games_per_worker,
                config.parallel.max_inference_batch,
            ),
            (),
        )
    return calibrate(
        config,
        plans=calibration_plans(root_seed=config.root_seed),
    )


def _resolved_config(
    config: TrainingRunConfig,
    candidate: BenchmarkCandidate,
) -> TrainingRunConfig:
    return replace(
        config,
        device=cast(DeviceName, candidate.device),
        parallel=ParallelConfig(
            workers=candidate.workers,
            active_games_per_worker=candidate.active_games_per_worker,
            max_inference_batch=candidate.max_inference_batch,
            max_queue_delay_ms=config.parallel.max_queue_delay_ms,
        ),
    )


def resolve_games_per_cell(
    config: TrainingRunConfig,
    *,
    estimated_decisions_per_game: float,
) -> int:
    """Resolve balanced games from a measured complete-path decision rate."""

    if config.games_per_cell is not None:
        return config.games_per_cell
    if (
        not math.isfinite(estimated_decisions_per_game)
        or estimated_decisions_per_game <= 0.0
    ):
        raise TrainerError("estimated_decisions_per_game must be positive")
    assert config.target_decisions_per_update is not None
    return max(
        1,
        math.ceil(
            config.target_decisions_per_update
            / (15.0 * estimated_decisions_per_game)
        ),
    )


def _update_metrics(
    update_index: int,
    games_per_cell: int,
    duration: float,
    collection: CollectorMetrics,
    ppo: PPOUpdateMetrics,
) -> dict[str, object]:
    collection_payload = asdict(collection)
    collection_payload["games_per_second"] = collection.games_per_second
    collection_payload["decisions_per_second"] = (
        collection.decisions_per_second
    )
    collection_payload["mean_inference_batch_size"] = (
        collection.mean_inference_batch_size
    )
    return {
        "update_index": update_index,
        "games_per_cell": games_per_cell,
        "duration_seconds": duration,
        "collection": collection_payload,
        "ppo": asdict(ppo),
    }


def _prepare_run_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and (
        not resolved.is_dir() or any(resolved.iterdir())
    ):
        raise TrainerError("training output directory must be empty")
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "checkpoints").mkdir()
    return resolved


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
        )


def _repository_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
