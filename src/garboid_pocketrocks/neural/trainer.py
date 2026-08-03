"""Durable update-boundary orchestration for neural self-play training."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import cast

import torch

from garboid_pocketrocks.neural.benchmark import (
    BenchmarkCandidate,
    BenchmarkResult,
    calibrate,
    calibration_plans,
)
from garboid_pocketrocks.neural.collector import CollectorMetrics
from garboid_pocketrocks.neural.config import (
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.devices import resolve_device
from garboid_pocketrocks.neural.identity import trained_neural_bot_id
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.parallel import collect_self_play_parallel
from garboid_pocketrocks.neural.planning import (
    SelfPlayEpisodePlan,
    plan_mirror_episodes,
    plan_strong_field_episodes,
)
from garboid_pocketrocks.neural.ppo import (
    PolicyShiftMetrics,
    PPOTrainer,
    PPOUpdateMetrics,
)
from garboid_pocketrocks.neural.rollout import RolloutBatch
from garboid_pocketrocks.neural.run_config import (
    DeviceName,
    ParallelConfig,
    TrainingRunConfig,
    validate_runtime_support,
)
from garboid_pocketrocks.neural.seeding import (
    configure_torch_runtime,
    derive_seed,
)
from garboid_pocketrocks.neural.training_checkpoint import (
    TrainingCheckpointManifest,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from garboid_pocketrocks.neural.vector_collector import (
    collect_self_play_vectorized,
)
from garboid_pocketrocks.neural.vector_parallel import (
    collect_self_play_vectorized_parallel,
)
from garboid_pocketrocks.neural.vector_pool import VectorActorPool


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


@dataclass(frozen=True, slots=True)
class _PendingCollection:
    update_index: int
    policy_version: int
    launched_at: float
    future: Future[tuple[RolloutBatch, CollectorMetrics]]


def train(
    config: TrainingRunConfig,
    output_dir: Path,
) -> TrainingRunResult:
    """Start a new self-play lineage and stop only at an update boundary."""

    validate_runtime_support(config)
    run_dir = _prepare_run_dir(output_dir)
    configure_torch_runtime(
        config.root_seed,
        deterministic_algorithms=config.deterministic_algorithms,
    )
    torch.set_num_threads(config.learner_threads)
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
        training_model_config(config.model_profile),
    ).to(device)
    trainer = PPOTrainer(model, resolved.ppo)
    selected_result = next(
        (result for result in benchmarks if result.candidate == candidate),
        None,
    )
    estimated_decisions_per_game = (
        selected_result.decisions / selected_result.games if selected_result is not None else 20.0
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
    config_override: TrainingRunConfig | None = None,
) -> TrainingRunResult:
    """Resume exact model and optimizer state into a new run directory."""

    loaded = load_training_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    checkpoint_config = loaded.manifest.run_config
    config = config_override or checkpoint_config
    if config.root_seed != checkpoint_config.root_seed:
        raise TrainerError("resume config cannot change the root seed")
    if training_model_config(config.model_profile) != loaded.model.model_config:
        raise TrainerError("resume config cannot change the model profile")
    if config.bot_generation != checkpoint_config.bot_generation:
        raise TrainerError("resume config cannot change the bot generation")
    validate_runtime_support(config)
    run_dir = _prepare_run_dir(output_dir)
    configure_torch_runtime(
        config.root_seed,
        deterministic_algorithms=config.deterministic_algorithms,
    )
    benchmarks: tuple[BenchmarkResult, ...] = ()
    if config.device == "auto" or config.parallel.workers == "auto":
        candidate, benchmarks = _resolve_candidate(config)
        config = _resolved_config(config, candidate)
    torch.set_num_threads(config.learner_threads)
    device = resolve_device(config.device)
    if device.type != "cpu":
        loaded = load_training_checkpoint(checkpoint, device=device)
    _write_json(run_dir / "resolved-config.json", config.to_json_dict())
    _write_json(
        run_dir / "benchmark.json",
        {
            "resumed_from": str(checkpoint.resolve()),
            "results": [asdict(result) for result in benchmarks],
        },
    )
    trainer = PPOTrainer(loaded.model, config.ppo)
    trainer.optimizer = loaded.optimizer
    limit = max_additional_updates
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
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
        "bot_id": manifest.champion_identity
        or trained_neural_bot_id(
            manifest.run_config.model_profile,
            manifest.progress.completed_episodes,
            generation=manifest.run_config.bot_generation,
        ),
        "repository_commit": manifest.repository_commit,
        "next_update_index": manifest.progress.next_update_index,
        "completed_updates": manifest.progress.next_update_index,
        "completed_episodes": manifest.progress.completed_episodes,
        "completed_decisions": manifest.progress.completed_decisions,
        "cell_games": manifest.progress.cell_games,
        "device": manifest.run_config.device,
        "model_profile": manifest.run_config.model_profile,
        "bot_generation": manifest.run_config.bot_generation,
        "learner_threads": manifest.run_config.learner_threads,
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
    vector_pool: VectorActorPool | None = None
    workers = config.parallel.workers
    model_device = next(model.parameters()).device
    if model_device.type in ("cpu", "mps") and isinstance(workers, int) and workers > 1:
        vector_pool = VectorActorPool(
            encoder_config=model.encoder_config,
            reward_config=config.reward,
            workers=workers,
            engine_batch_size=config.parallel.active_games_per_worker,
            max_inference_batch=config.parallel.max_inference_batch,
        )
    try:
        return _run_updates_with_pool(
            config,
            run_dir,
            model=model,
            trainer=trainer,
            initial_progress=initial_progress,
            lineage=lineage,
            max_updates_this_run=max_updates_this_run,
            games_per_cell=games_per_cell,
            vector_pool=vector_pool,
        )
    finally:
        if vector_pool is not None:
            vector_pool.close()


def _run_updates_with_pool(
    config: TrainingRunConfig,
    run_dir: Path,
    *,
    model: NeuralPolicy,
    trainer: PPOTrainer,
    initial_progress: TrainingProgress,
    lineage: tuple[str, ...],
    max_updates_this_run: int | None,
    games_per_cell: int,
    vector_pool: VectorActorPool | None,
) -> TrainingRunResult:
    if vector_pool is not None and trainer.device.type == "mps":
        return _run_pipelined_updates(
            config,
            run_dir,
            model=model,
            trainer=trainer,
            initial_progress=initial_progress,
            lineage=lineage,
            max_updates_this_run=max_updates_this_run,
            games_per_cell=games_per_cell,
            vector_pool=vector_pool,
        )
    started = time.perf_counter()
    update_index = initial_progress.next_update_index
    completed_episodes = initial_progress.completed_episodes
    completed_decisions = initial_progress.completed_decisions
    cells = Counter(
        {(ruleset, players): games for ruleset, players, games in initial_progress.cell_games}
    )
    durations: list[float] = []
    checkpoint = run_dir / "checkpoints" / "latest"
    next_periodic_at = config.checkpoint_interval_seconds

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
        plans = _training_plans(
            config,
            update_index=update_index,
            games_per_cell=games_per_cell,
        )
        snapshot = NeuralPolicy(
            model.encoder_config,
            model.model_config,
        ).to(trainer.device)
        snapshot.load_state_dict(model.state_dict())
        snapshot.eval()
        collection = _collect(
            config,
            snapshot,
            plans,
            vector_pool=vector_pool,
        )
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
            tuple((ruleset, players, games) for (ruleset, players), games in sorted(cells.items())),
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
                champion_identity=trained_neural_bot_id(
                    config.model_profile,
                    completed_episodes,
                    generation=config.bot_generation,
                ),
                league_identities=(),
            ),
            generator_states={"torch": torch.get_rng_state()},
            metrics=metrics,
        )
        next_periodic_at = _maybe_keep_periodic_checkpoint(
            config,
            checkpoint=checkpoint,
            update_index=update_index - 1,
            elapsed_seconds=time.perf_counter() - started,
            next_periodic_at=next_periodic_at,
        )
        del collection, ppo

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


def _run_pipelined_updates(
    config: TrainingRunConfig,
    run_dir: Path,
    *,
    model: NeuralPolicy,
    trainer: PPOTrainer,
    initial_progress: TrainingProgress,
    lineage: tuple[str, ...],
    max_updates_this_run: int | None,
    games_per_cell: int,
    vector_pool: VectorActorPool,
) -> TrainingRunResult:
    """Overlap bounded CPU collection with one-update-stale MPS learning."""

    started = time.perf_counter()
    update_index = initial_progress.next_update_index
    completed_episodes = initial_progress.completed_episodes
    completed_decisions = initial_progress.completed_decisions
    cells = Counter(
        {(ruleset, players): games for ruleset, players, games in initial_progress.cell_games}
    )
    durations: list[float] = []
    checkpoint = run_dir / "checkpoints" / "latest"
    pending: _PendingCollection | None = None
    next_periodic_at = config.checkpoint_interval_seconds

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-prefetch") as executor:
        while True:
            if pending is None:
                if not _budget_allows_new_collection(
                    config,
                    started=started,
                    update_index=update_index,
                    max_updates_this_run=max_updates_this_run,
                    durations=durations,
                ):
                    break
                pending = _launch_collection(
                    executor,
                    config,
                    vector_pool,
                    model=model,
                    update_index=update_index,
                    policy_version=update_index,
                    games_per_cell=games_per_cell,
                )

            update_started = time.perf_counter()
            wait_started = time.perf_counter()
            collection = pending.future.result()
            collection_wait_seconds = time.perf_counter() - wait_started
            collection_age_seconds = time.perf_counter() - pending.launched_at
            policy_lag = update_index - pending.policy_version
            if pending.update_index != update_index or policy_lag not in (0, 1):
                raise TrainerError("pipelined rollout exceeded the one-update policy lag")
            pending = None

            shift_started = time.perf_counter()
            stale_shift = (
                trainer.measure_policy_shift(collection[0])
                if policy_lag == 1
                else PolicyShiftMetrics(0.0, 0.0, len(collection[0].transitions))
            )
            shift_measurement_seconds = time.perf_counter() - shift_started
            discarded_stale_collection_seconds = 0.0
            if not _stale_rollout_is_acceptable(config, stale_shift):
                discarded_stale_collection_seconds = collection[1].elapsed_seconds
                fresh = _launch_collection(
                    executor,
                    config,
                    vector_pool,
                    model=model,
                    update_index=update_index,
                    policy_version=update_index,
                    games_per_cell=games_per_cell,
                )
                fresh_wait_started = time.perf_counter()
                collection = fresh.future.result()
                collection_wait_seconds += time.perf_counter() - fresh_wait_started
                collection_age_seconds = time.perf_counter() - fresh.launched_at
                policy_lag = 0

            if _should_prefetch_next(
                config,
                started=started,
                update_index=update_index,
                max_updates_this_run=max_updates_this_run,
                durations=durations,
                collection=collection[1],
            ):
                pending = _launch_collection(
                    executor,
                    config,
                    vector_pool,
                    model=model,
                    update_index=update_index + 1,
                    policy_version=update_index,
                    games_per_cell=games_per_cell,
                )

            learner_started = time.perf_counter()
            ppo = trainer.update(
                collection[0],
                update_seed=derive_seed(config.root_seed, "minibatch", update_index),
            )
            learner_seconds = time.perf_counter() - learner_started
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
                pipeline={
                    "collection_age_seconds": collection_age_seconds,
                    "collection_wait_seconds": collection_wait_seconds,
                    "learner_seconds": learner_seconds,
                    "overlap_seconds": max(
                        0.0,
                        collection[1].elapsed_seconds - collection_wait_seconds,
                    ),
                    "policy_lag_updates": policy_lag,
                    "policy_version": update_index - 1 - policy_lag,
                    "shift_measurement_seconds": shift_measurement_seconds,
                    "stale_approximate_kl": stale_shift.approximate_kl,
                    "stale_clip_fraction": stale_shift.clip_fraction,
                    "stale_rollout_rejected": discarded_stale_collection_seconds > 0.0,
                    "discarded_stale_collection_seconds": (
                        discarded_stale_collection_seconds
                    ),
                },
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
                    champion_identity=trained_neural_bot_id(
                        config.model_profile,
                        completed_episodes,
                        generation=config.bot_generation,
                    ),
                    league_identities=(),
                ),
                generator_states={"torch": torch.get_rng_state()},
                metrics=metrics,
            )
            next_periodic_at = _maybe_keep_periodic_checkpoint(
                config,
                checkpoint=checkpoint,
                update_index=update_index - 1,
                elapsed_seconds=time.perf_counter() - started,
                next_periodic_at=next_periodic_at,
            )
            del collection, ppo

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


def _budget_allows_new_collection(
    config: TrainingRunConfig,
    *,
    started: float,
    update_index: int,
    max_updates_this_run: int | None,
    durations: list[float],
) -> bool:
    if max_updates_this_run is not None and update_index >= max_updates_this_run:
        return False
    if config.max_wall_seconds is None:
        return True
    remaining = config.max_wall_seconds - (time.perf_counter() - started)
    if remaining <= 0.0:
        return False
    return not durations or remaining >= max(durations[-5:])


def _maybe_keep_periodic_checkpoint(
    config: TrainingRunConfig,
    *,
    checkpoint: Path,
    update_index: int,
    elapsed_seconds: float,
    next_periodic_at: float | None,
) -> float | None:
    if next_periodic_at is None or elapsed_seconds < next_periodic_at:
        return next_periodic_at
    periodic_root = checkpoint.parent / "periodic"
    periodic_root.mkdir(exist_ok=True)
    destination = periodic_root / f"update-{update_index:08d}"
    temporary = periodic_root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    retained = sorted(path for path in periodic_root.iterdir() if path.is_dir())
    for expired in retained[: -config.keep_periodic_checkpoints]:
        shutil.rmtree(expired)
    assert config.checkpoint_interval_seconds is not None
    elapsed_intervals = math.floor(
        (elapsed_seconds - next_periodic_at) / config.checkpoint_interval_seconds
    )
    next_periodic_at += (elapsed_intervals + 1) * config.checkpoint_interval_seconds
    return next_periodic_at


def _stale_rollout_is_acceptable(
    config: TrainingRunConfig,
    shift: PolicyShiftMetrics,
) -> bool:
    return (
        shift.approximate_kl <= config.parallel.max_stale_approximate_kl
        and shift.clip_fraction <= config.parallel.max_stale_clip_fraction
    )


def _should_prefetch_next(
    config: TrainingRunConfig,
    *,
    started: float,
    update_index: int,
    max_updates_this_run: int | None,
    durations: list[float],
    collection: CollectorMetrics,
) -> bool:
    next_update = update_index + 1
    if max_updates_this_run is not None and next_update >= max_updates_this_run:
        return False
    if config.max_wall_seconds is None:
        return True
    remaining = config.max_wall_seconds - (time.perf_counter() - started)
    estimate = max(durations[-5:], default=collection.elapsed_seconds + 15.0)
    return remaining >= estimate


def _launch_collection(
    executor: ThreadPoolExecutor,
    config: TrainingRunConfig,
    vector_pool: VectorActorPool,
    *,
    model: NeuralPolicy,
    update_index: int,
    policy_version: int,
    games_per_cell: int,
) -> _PendingCollection:
    plans = _training_plans(
        config,
        update_index=update_index,
        games_per_cell=games_per_cell,
    )
    snapshot = NeuralPolicy(model.encoder_config, model.model_config)
    snapshot.load_state_dict(model.state_dict())
    snapshot.eval()
    launched_at = time.perf_counter()
    return _PendingCollection(
        update_index=update_index,
        policy_version=policy_version,
        launched_at=launched_at,
        future=executor.submit(
            _collect,
            config,
            snapshot,
            plans,
            vector_pool=vector_pool,
        ),
    )


def _collect(
    config: TrainingRunConfig,
    model: NeuralPolicy,
    plans: tuple[SelfPlayEpisodePlan, ...],
    *,
    vector_pool: VectorActorPool | None = None,
) -> tuple[RolloutBatch, CollectorMetrics]:
    # Kept in one function so all training updates share the same selection rule.
    device = next(model.parameters()).device
    workers = config.parallel.workers
    if vector_pool is not None:
        return vector_pool.collect({"current": model}, plans)
    if workers == 1:
        return collect_self_play_vectorized(
            {"current": model},
            plans,
            encoder_config=model.encoder_config,
            reward_config=config.reward,
            device=device,
            engine_batch_size=config.parallel.active_games_per_worker,
            max_inference_batch=config.parallel.max_inference_batch,
        )
    if workers == "auto":
        raise TrainerError("training workers must be resolved before collection")
    if device.type == "cpu":
        return collect_self_play_vectorized_parallel(
            {"current": model},
            plans,
            encoder_config=model.encoder_config,
            reward_config=config.reward,
            workers=workers,
            engine_batch_size=config.parallel.active_games_per_worker,
            max_inference_batch=config.parallel.max_inference_batch,
        )
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


def _training_plans(
    config: TrainingRunConfig,
    *,
    update_index: int,
    games_per_cell: int,
) -> tuple[SelfPlayEpisodePlan, ...]:
    if config.opponent_training == "mirror-self-play":
        return plan_mirror_episodes(
            root_seed=config.root_seed,
            update_index=update_index,
            games_per_cell=games_per_cell,
            policy_identity="current",
        )
    share = {
        "strong-field-v1-75": Fraction(3, 4),
        "strong-field-v1-50": Fraction(1, 2),
        "strong-field-v1-25": Fraction(1, 4),
    }[config.opponent_training]
    return plan_strong_field_episodes(
        root_seed=config.root_seed,
        update_index=update_index,
        games_per_cell=games_per_cell,
        policy_identity="current",
        fixed_opponent_share=share,
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
            max_stale_approximate_kl=config.parallel.max_stale_approximate_kl,
            max_stale_clip_fraction=config.parallel.max_stale_clip_fraction,
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
    if not math.isfinite(estimated_decisions_per_game) or estimated_decisions_per_game <= 0.0:
        raise TrainerError("estimated_decisions_per_game must be positive")
    assert config.target_decisions_per_update is not None
    return max(
        1,
        math.ceil(config.target_decisions_per_update / (15.0 * estimated_decisions_per_game)),
    )


def _update_metrics(
    update_index: int,
    games_per_cell: int,
    duration: float,
    collection: CollectorMetrics,
    ppo: PPOUpdateMetrics,
    pipeline: dict[str, object] | None = None,
) -> dict[str, object]:
    collection_payload = asdict(collection)
    collection_payload["games_per_second"] = collection.games_per_second
    collection_payload["decisions_per_second"] = collection.decisions_per_second
    collection_payload["mean_inference_batch_size"] = collection.mean_inference_batch_size
    ppo_payload: dict[str, object] = {
        "epochs": ppo.epochs,
        "optimizer_steps": ppo.optimizer_steps,
        "transition_count": ppo.transition_count,
        "total_loss": ppo.total_loss,
        "policy_loss": ppo.policy_loss,
        "value_loss": ppo.value_loss,
        "entropy": ppo.entropy,
        "approximate_kl": ppo.approximate_kl,
        "clip_fraction": ppo.clip_fraction,
        "value": asdict(ppo.value),
        "value_slices": [asdict(value_slice) for value_slice in ppo.value_slices],
        "distributions": {
            "advantages": _distribution_summary(ppo.advantages),
            "ratios": _distribution_summary(ppo.ratios),
            "values": _distribution_summary(ppo.values),
            "entropies": _distribution_summary(ppo.entropies),
            "pre_clip_gradient_norms": _distribution_summary(
                ppo.pre_clip_gradient_norms
            ),
            "post_clip_gradient_norms": _distribution_summary(
                ppo.post_clip_gradient_norms
            ),
        },
    }
    payload: dict[str, object] = {
        "update_index": update_index,
        "games_per_cell": games_per_cell,
        "duration_seconds": duration,
        "collection": collection_payload,
        "ppo": ppo_payload,
    }
    if pipeline is not None:
        payload["pipeline"] = pipeline
    return payload


def _distribution_summary(values: tuple[float, ...]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": math.fsum(ordered) / len(ordered),
        "min": ordered[0],
        "p05": _ordered_percentile(ordered, 0.05),
        "p50": _ordered_percentile(ordered, 0.50),
        "p95": _ordered_percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _ordered_percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _prepare_run_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
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
        file.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")


def _repository_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"
