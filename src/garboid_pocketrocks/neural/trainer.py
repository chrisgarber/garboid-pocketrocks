"""Durable update-boundary orchestration for neural self-play training."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch

from garboid_pocketrocks.neural.behavior_cloning import (
    BehaviorCloningMetrics,
    BehaviorCloningTrainer,
    BehaviorCloningUpdateMetrics,
    behavior_cloning_game_shards,
    collect_behavior_cloning_dataset,
    plan_behavior_cloning_games,
)
from garboid_pocketrocks.neural.benchmark import (
    BenchmarkCandidate,
    BenchmarkResult,
    calibrate,
    calibration_plans,
)
from garboid_pocketrocks.neural.checkpoint import parameter_digest
from garboid_pocketrocks.neural.collector import CollectorMetrics
from garboid_pocketrocks.neural.config import (
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.devices import resolve_device
from garboid_pocketrocks.neural.heuristic_bootstrap import (
    EXPERIMENT_GAMES_PER_CELL,
    TRAINING_CELL_COUNT,
    HeuristicBootstrapArm,
    bootstrap_strategy,
    validate_fixed_compute_arm,
)
from garboid_pocketrocks.neural.heuristic_curriculum import (
    FOCAL_SEAT_CONTROL_V1,
    HEURISTIC_OPPONENT_CURRICULUM_V1,
    plan_heuristic_curriculum_episodes,
)
from garboid_pocketrocks.neural.identity import (
    experimental_neural_bot_id,
    trained_neural_bot_id,
)
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
class _OfficialResumeArtifacts:
    metrics_jsonl: bytes
    behavior_cloning_json: bytes | None


@dataclass(frozen=True, slots=True)
class _OfficialExperimentRun:
    arm: HeuristicBootstrapArm
    repository_commit: str


def train(
    config: TrainingRunConfig,
    output_dir: Path,
) -> TrainingRunResult:
    """Start a new self-play lineage and stop only at an update boundary."""

    validate_runtime_support(config)
    official_experiment = _validate_official_experiment(config)
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
    _run_behavior_cloning_if_configured(resolved, run_dir, model)
    trainer = PPOTrainer(
        model,
        resolved.ppo,
        heuristic_auxiliary_config=resolved.heuristic_auxiliary,
    )
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
        official_repository_commit=(
            None if official_experiment is None else official_experiment.repository_commit
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
    if checkpoint_config.experiment_arm is not None and config != checkpoint_config:
        raise TrainerError("official experiment resume config must exactly match its checkpoint")
    validate_runtime_support(config)
    official_experiment = _validate_official_experiment(config)
    if official_experiment is not None:
        if loaded.manifest.repository_commit != official_experiment.repository_commit:
            raise TrainerError(
                "official experiment resume must use the checkpoint's exact source commit"
            )
        _validate_official_checkpoint_progress(
            loaded.manifest.progress,
            official_experiment.arm,
        )
    limit = _resume_update_limit(
        loaded.manifest.progress.next_update_index,
        max_additional_updates=max_additional_updates,
        official_arm=(None if official_experiment is None else official_experiment.arm),
    )
    official_artifacts = (
        _load_official_resume_artifacts(checkpoint, loaded.metrics, config)
        if official_experiment is not None
        else None
    )
    run_dir = _prepare_run_dir(output_dir)
    if official_artifacts is not None:
        (run_dir / "metrics.jsonl").write_bytes(official_artifacts.metrics_jsonl)
        if official_artifacts.behavior_cloning_json is not None:
            (run_dir / "behavior-cloning.json").write_bytes(
                official_artifacts.behavior_cloning_json
            )
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
    trainer = PPOTrainer(
        loaded.model,
        config.ppo,
        heuristic_auxiliary_config=config.heuristic_auxiliary,
    )
    trainer.optimizer = loaded.optimizer
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
        official_repository_commit=(
            None if official_experiment is None else official_experiment.repository_commit
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
        ),
        "repository_commit": manifest.repository_commit,
        "next_update_index": manifest.progress.next_update_index,
        "completed_updates": manifest.progress.next_update_index,
        "completed_episodes": manifest.progress.completed_episodes,
        "completed_decisions": manifest.progress.completed_decisions,
        "cell_games": manifest.progress.cell_games,
        "device": manifest.run_config.device,
        "model_profile": manifest.run_config.model_profile,
        "learner_threads": manifest.run_config.learner_threads,
        "supported_ruleset_names": manifest.encoder_config.supported_ruleset_names,
        "supported_player_counts": manifest.encoder_config.supported_player_counts,
        "parameter_digest": manifest.parameter_digest,
        "lineage": manifest.lineage,
    }


def _load_official_resume_artifacts(
    checkpoint: Path,
    checkpoint_metrics: dict[str, object],
    config: TrainingRunConfig,
) -> _OfficialResumeArtifacts:
    """Validate the official aggregate prefix that a resumed run must preserve."""

    resolved_checkpoint = checkpoint.resolve()
    if resolved_checkpoint.name != "latest" or resolved_checkpoint.parent.name != "checkpoints":
        raise TrainerError("official resume checkpoint must use run/checkpoints/latest layout")
    source_run = resolved_checkpoint.parent.parent
    metrics_path = source_run / "metrics.jsonl"
    try:
        metrics_bytes = metrics_path.read_bytes()
        metrics_text = metrics_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TrainerError("official resume requires readable prior metrics.jsonl") from error
    lines = metrics_text.splitlines()
    if metrics_bytes and not metrics_bytes.endswith(b"\n"):
        raise TrainerError("official resume metrics.jsonl lacks its terminal newline")
    expected_count = config.max_updates
    completed = (
        _json_int(checkpoint_metrics.get("update_index"), "checkpoint metrics update_index") + 1
    )
    if expected_count is None or completed > expected_count or len(lines) != completed:
        raise TrainerError("official resume metrics.jsonl is not the complete checkpoint prefix")
    parsed: list[dict[str, object]] = []
    for expected_index, line in enumerate(lines):
        if not line.strip():
            raise TrainerError("official resume metrics.jsonl contains a blank row")
        try:
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid constant {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise TrainerError("official resume metrics.jsonl is invalid JSON") from error
        if not isinstance(value, dict) or value.get("update_index") != expected_index:
            raise TrainerError("official resume metrics.jsonl update indices are not contiguous")
        parsed.append(cast(dict[str, object], value))
    if not parsed or parsed[-1] != checkpoint_metrics:
        raise TrainerError("official resume metrics.jsonl does not match checkpoint metrics")

    behavior_path = source_run / "behavior-cloning.json"
    behavior_bytes: bytes | None = None
    if config.behavior_cloning is not None:
        try:
            behavior_bytes = behavior_path.read_bytes()
            behavior_value = json.loads(
                behavior_bytes.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid constant {token}")
                ),
            )
            from garboid_pocketrocks.neural.bootstrap_reporting import (
                validate_behavior_cloning_payload,
            )

            validate_behavior_cloning_payload(behavior_value, config.behavior_cloning)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise TrainerError(
                "official cloning resume requires valid prior behavior-cloning.json"
            ) from error
    elif behavior_path.exists():
        raise TrainerError("non-cloning official resume found unexpected behavior-cloning.json")
    return _OfficialResumeArtifacts(metrics_bytes, behavior_bytes)


def _json_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainerError(f"{name} must be an integer")
    return value


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
    official_repository_commit: str | None = None,
) -> TrainingRunResult:
    vector_pool: VectorActorPool | None = None
    try:
        workers = config.parallel.workers
        if (
            next(model.parameters()).device.type == "cpu"
            and isinstance(workers, int)
            and workers > 1
        ):
            if official_repository_commit is not None:
                _require_official_source_unchanged(official_repository_commit)
            vector_pool = VectorActorPool(
                encoder_config=model.encoder_config,
                reward_config=config.reward,
                workers=workers,
                engine_batch_size=config.parallel.active_games_per_worker,
                max_inference_batch=config.parallel.max_inference_batch,
                expected_repository_commit=official_repository_commit,
            )
            if official_repository_commit is not None:
                _require_official_source_unchanged(official_repository_commit)
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
            official_repository_commit=official_repository_commit,
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
    official_repository_commit: str | None = None,
) -> TrainingRunResult:
    started = time.perf_counter()
    update_index = initial_progress.next_update_index
    completed_episodes = initial_progress.completed_episodes
    completed_decisions = initial_progress.completed_decisions
    cells = Counter(
        {(ruleset, players): games for ruleset, players, games in initial_progress.cell_games}
    )
    durations: list[float] = []
    checkpoint = run_dir / "checkpoints" / "latest"

    while True:
        if official_repository_commit is not None:
            _require_official_source_unchanged(official_repository_commit)
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
        if official_repository_commit is not None:
            _require_official_source_unchanged(official_repository_commit)
        checkpoint_repository_commit = official_repository_commit or _repository_commit()
        _append_json_line(run_dir / "metrics.jsonl", metrics)
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=trainer.optimizer,
            manifest=TrainingCheckpointManifest(
                schema_version=1,
                repository_commit=checkpoint_repository_commit,
                encoder_config=model.encoder_config,
                model_config=model.model_config,
                run_config=config,
                progress=progress,
                lineage=lineage,
                champion_identity=_checkpoint_identity(
                    config,
                    completed_episodes,
                    model,
                    repository_commit=checkpoint_repository_commit,
                ),
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


def _run_behavior_cloning_if_configured(
    config: TrainingRunConfig,
    run_dir: Path,
    model: NeuralPolicy,
) -> None:
    cloning = config.behavior_cloning
    if cloning is None:
        return
    started = time.perf_counter()
    game_shards = behavior_cloning_game_shards(
        plan_behavior_cloning_games(cloning),
        games_per_shard=cloning.games_per_shard,
    )
    cloning_trainer = BehaviorCloningTrainer(model, cloning)
    combined_digest = hashlib.sha256()
    combined_updates: list[BehaviorCloningUpdateMetrics] = []
    combined_cell_games: Counter[tuple[str, int]] = Counter()
    example_count = 0
    game_count = 0
    shard_records: list[dict[str, object]] = []
    for shard_index, game_plans in enumerate(game_shards):
        dataset = collect_behavior_cloning_dataset(
            game_plans,
            encoder_config=model.encoder_config,
        )
        shard_metrics = cloning_trainer.train(dataset, shard_index=shard_index)
        shard_records.append(
            {
                "shard_index": shard_index,
                "game_count": dataset.game_count,
                "example_count": len(dataset.examples),
                "dataset_digest": dataset.dataset_digest,
            }
        )
        combined_digest.update(shard_index.to_bytes(8, "big"))
        combined_digest.update(bytes.fromhex(dataset.dataset_digest))
        combined_updates.extend(shard_metrics.updates)
        example_count += len(dataset.examples)
        game_count += dataset.game_count
        for ruleset, players, games in dataset.cell_game_counts:
            combined_cell_games[(ruleset, players)] += games
    metrics = BehaviorCloningMetrics(
        config_digest=cloning.config_digest,
        dataset_digest=combined_digest.hexdigest(),
        example_count=example_count,
        epochs=cloning.epochs,
        optimizer_steps=len(combined_updates),
        updates=tuple(combined_updates),
    )
    _write_json(
        run_dir / "behavior-cloning.json",
        {
            "config": cloning.to_json_dict(),
            "dataset": {
                "cell_game_counts": tuple(
                    (ruleset, players, games)
                    for (ruleset, players), games in sorted(combined_cell_games.items())
                ),
                "dataset_digest": metrics.dataset_digest,
                "example_count": example_count,
                "game_count": game_count,
                "shard_count": len(game_shards),
                "shards": shard_records,
                "teacher_identity": cloning.teacher_identity,
                "teacher_profile_digest": cloning.teacher_profile_digest,
            },
            "training": {
                **asdict(metrics),
                "elapsed_seconds": time.perf_counter() - started,
            },
        },
    )


def _training_plans(
    config: TrainingRunConfig,
    *,
    update_index: int,
    games_per_cell: int,
) -> tuple[SelfPlayEpisodePlan, ...]:
    if config.opponent_training in (
        "focal-seat-control-v1",
        "heuristic-opponent-curriculum-v1",
    ):
        curriculum = (
            FOCAL_SEAT_CONTROL_V1
            if config.opponent_training == "focal-seat-control-v1"
            else HEURISTIC_OPPONENT_CURRICULUM_V1
        )
        return plan_heuristic_curriculum_episodes(
            root_seed=config.root_seed,
            update_index=update_index,
            games_per_cell=games_per_cell,
            learner_policy_identity="current",
            curriculum=curriculum,
        ).plans
    return plan_mirror_episodes(
        root_seed=config.root_seed,
        update_index=update_index,
        games_per_cell=games_per_cell,
        policy_identity="current",
    )


def _checkpoint_identity(
    config: TrainingRunConfig,
    completed_ppo_games: int,
    model: NeuralPolicy,
    *,
    repository_commit: str,
) -> str:
    try:
        validate_fixed_compute_arm(config)
    except ValueError:
        return trained_neural_bot_id(config.model_profile, completed_ppo_games)
    canonical = json.dumps(
        config.to_json_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    demonstration_games = (
        config.behavior_cloning.rounds * config.behavior_cloning.games_per_cell * 15
        if config.behavior_cloning is not None
        else 0
    )
    return experimental_neural_bot_id(
        config.model_profile,
        strategy=bootstrap_strategy(config),
        root_seed=config.root_seed,
        completed_games=completed_ppo_games + demonstration_games,
        config_digest=hashlib.sha256(canonical).hexdigest(),
        parameter_digest=parameter_digest(model.state_dict()),
        repository_commit=repository_commit,
    )


def _validate_official_experiment(
    config: TrainingRunConfig,
) -> _OfficialExperimentRun | None:
    if config.experiment_arm is None:
        return None
    try:
        arm = validate_fixed_compute_arm(config)
    except ValueError as error:
        raise TrainerError(str(error)) from error
    repository_commit_before_status = _repository_commit()
    dirty = _repository_status()
    repository_commit_after_status = _repository_commit()
    if not _is_exact_repository_commit(
        repository_commit_before_status
    ) or not _is_exact_repository_commit(repository_commit_after_status):
        raise TrainerError("official experiment requires an exact Git source commit")
    if dirty:
        raise TrainerError("official experiment requires a clean source tree")
    if repository_commit_before_status != repository_commit_after_status:
        raise TrainerError("official experiment source changed while reading Git provenance")
    return _OfficialExperimentRun(
        arm=arm,
        repository_commit=repository_commit_before_status,
    )


def _is_exact_repository_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _require_official_source_unchanged(repository_commit: str) -> None:
    """Stop an official run before source drift can rebrand a checkpoint."""

    if _repository_commit() != repository_commit or _repository_status():
        raise TrainerError("official experiment source changed during training")


def _repository_status() -> str:
    try:
        return subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainerError("official experiment requires readable Git provenance") from error


def _validate_official_checkpoint_progress(
    progress: TrainingProgress,
    arm: HeuristicBootstrapArm,
) -> None:
    """Require an official checkpoint to be an exact complete-update prefix."""

    update_index = progress.next_update_index
    games_in_cell = update_index * EXPERIMENT_GAMES_PER_CELL
    expected_cells = {
        (f"live-{chart}", player_count): games_in_cell
        for chart in "ABCDE"
        for player_count in (3, 4, 5)
    }
    actual_cells = {
        (ruleset, player_count): games for ruleset, player_count, games in progress.cell_games
    }
    if (
        update_index >= arm.ppo_updates
        or progress.completed_episodes != games_in_cell * TRAINING_CELL_COUNT
        or len(actual_cells) != len(progress.cell_games)
        or actual_cells != expected_cells
    ):
        raise TrainerError(
            "official experiment checkpoint is not an exact resumable fixed-budget prefix"
        )


def _resume_update_limit(
    completed_updates: int,
    *,
    max_additional_updates: int | None,
    official_arm: HeuristicBootstrapArm | None,
) -> int | None:
    """Resolve a bounded official resume without changing generic semantics."""

    if max_additional_updates is not None and (
        not isinstance(max_additional_updates, int)
        or isinstance(max_additional_updates, bool)
        or max_additional_updates <= 0
    ):
        raise TrainerError("max_additional_updates must be positive")
    if official_arm is None:
        return (
            None if max_additional_updates is None else completed_updates + max_additional_updates
        )
    terminal_update = official_arm.ppo_updates
    requested_terminal = (
        terminal_update
        if max_additional_updates is None
        else completed_updates + max_additional_updates
    )
    if requested_terminal > terminal_update:
        raise TrainerError("official experiment resume exceeds its fixed update budget")
    return requested_terminal


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
) -> dict[str, object]:
    collection_payload = asdict(collection)
    collection_payload["games_per_second"] = collection.games_per_second
    collection_payload["decisions_per_second"] = collection.decisions_per_second
    collection_payload["mean_inference_batch_size"] = collection.mean_inference_batch_size
    return {
        "update_index": update_index,
        "games_per_cell": games_per_cell,
        "duration_seconds": duration,
        "collection": collection_payload,
        "ppo": asdict(ppo),
    }


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
