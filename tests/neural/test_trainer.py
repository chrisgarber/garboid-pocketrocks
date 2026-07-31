from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural import trainer as trainer_module  # noqa: E402
from garboid_pocketrocks.neural.collector import CollectorMetrics  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.heuristic_bootstrap import (  # noqa: E402
    EXPERIMENT_GAMES_PER_CELL,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.ppo import PPOTrainer  # noqa: E402
from garboid_pocketrocks.neural.rollout import RolloutBatch  # noqa: E402
from garboid_pocketrocks.neural.run_config import (  # noqa: E402
    ParallelConfig,
    TrainingRunConfig,
    validate_runtime_support,
)
from garboid_pocketrocks.neural.trainer import (  # noqa: E402
    TrainerError,
    TrainingRunResult,
    inspect_checkpoint,
    resolve_games_per_cell,
    resume,
    train,
)
from garboid_pocketrocks.neural.training_checkpoint import (  # noqa: E402
    TrainingCheckpointManifest,
    TrainingProgress,
    save_training_checkpoint,
)


def _official_checkpoint(
    path: Path,
    *,
    completed_updates: int = 195,
    episode_offset: int = 0,
    repository_commit: str = "0123456789abcdef0123456789abcdef01234567",
) -> Path:
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    torch.manual_seed(812)
    model = NeuralPolicy(training_encoder_config(), training_model_config("large"))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.ppo.learning_rate,
        foreach=False,
    )
    games_in_cell = completed_updates * EXPERIMENT_GAMES_PER_CELL
    completed_episodes = (games_in_cell * 15) + episode_offset
    checkpoint = save_training_checkpoint(
        path / "checkpoints" / "latest",
        model=model,
        optimizer=optimizer,
        manifest=TrainingCheckpointManifest(
            schema_version=1,
            repository_commit=repository_commit,
            encoder_config=model.encoder_config,
            model_config=model.model_config,
            run_config=config,
            progress=TrainingProgress(
                next_update_index=completed_updates,
                completed_episodes=completed_episodes,
                completed_decisions=completed_episodes * 20,
                cell_games=tuple(
                    (f"live-{chart}", players, games_in_cell)
                    for chart in "ABCDE"
                    for players in (3, 4, 5)
                ),
            ),
            lineage=(),
            champion_identity=None,
            league_identities=(),
        ),
        generator_states={"torch": torch.get_rng_state()},
        metrics={"update_index": completed_updates - 1},
    )
    (path / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"update_index": update_index}) + "\n"
            for update_index in range(completed_updates)
        ),
        encoding="utf-8",
    )
    return checkpoint


def _pretend_source_tree_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    def clean_git_result(
        args: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        stdout = (
            "0123456789abcdef0123456789abcdef01234567\n"
            if args[-2:] == ("rev-parse", "HEAD")
            else ""
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        subprocess,
        "run",
        clean_git_result,
    )


def test_low_volume_train_resume_and_inspect(tmp_path: Path) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        games_per_cell=1,
        max_updates=1,
        max_wall_seconds=None,
        parallel=ParallelConfig(
            workers=1,
            active_games_per_worker=4,
            max_inference_batch=32,
        ),
    )

    first = train(config, tmp_path / "run")
    inspected = inspect_checkpoint(first.final_checkpoint)
    resumed = resume(
        first.final_checkpoint,
        tmp_path / "resumed",
        max_additional_updates=1,
        config_override=replace(
            config,
            learner_threads=2,
            max_updates=None,
        ),
    )

    assert first.completed_updates == 1
    assert first.completed_episodes == 15
    assert inspected["completed_episodes"] == 15
    assert inspected["bot_id"] == "vector_ppo_small_v1_g15"
    assert resumed.completed_updates == 2
    assert resumed.completed_episodes == 30
    assert inspect_checkpoint(resumed.final_checkpoint)["learner_threads"] == 2
    assert (first.run_dir / "metrics.jsonl").is_file()
    assert (first.run_dir / "resolved-config.json").is_file()


def test_committed_profiles_have_exact_wall_envelopes() -> None:
    smoke = TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))
    initial = TrainingRunConfig.from_json(Path("configs/neural/initial-10m.json"))
    long = TrainingRunConfig.from_json(Path("configs/neural/long-8h.json"))

    assert smoke.games_per_cell == 100
    assert smoke.max_updates == 1
    assert smoke.device == "cpu"
    assert smoke.parallel.workers == 8
    assert initial.max_wall_seconds == 600.0
    assert initial.model_profile == "medium"
    assert initial.target_decisions_per_update == 131_072
    assert long.max_wall_seconds == 28_800.0
    assert long.model_profile == "large"
    assert long.target_decisions_per_update == 131_072
    assert long.learner_threads == 4
    for config in (initial, long):
        assert config.checkpoint_interval_seconds is None
        assert config.keep_periodic_checkpoints == 4
        assert config.evaluation_interval_seconds is None
        assert config.evaluation_games_per_seat_cell == 2
        assert not config.evaluate_at_start
        assert not config.evaluate_at_end
        assert config.league_fraction == 0.0
        validate_runtime_support(config)


@pytest.mark.parametrize(
    ("config", "field"),
    (
        (
            replace(TrainingRunConfig(), checkpoint_interval_seconds=60.0),
            "checkpoint_interval_seconds",
        ),
        (
            replace(TrainingRunConfig(), keep_periodic_checkpoints=2),
            "keep_periodic_checkpoints",
        ),
        (
            replace(TrainingRunConfig(), evaluation_interval_seconds=60.0),
            "evaluation_interval_seconds",
        ),
        (
            replace(TrainingRunConfig(), evaluation_games_per_seat_cell=4),
            "evaluation_games_per_seat_cell",
        ),
        (replace(TrainingRunConfig(), evaluate_at_start=True), "evaluate_at_start"),
        (replace(TrainingRunConfig(), evaluate_at_end=True), "evaluate_at_end"),
        (replace(TrainingRunConfig(), league_fraction=0.2), "league_fraction"),
    ),
)
def test_runtime_support_rejects_ignored_controls(
    config: TrainingRunConfig,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_runtime_support(config)


def test_train_rejects_unsupported_controls_before_creating_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="league_fraction"):
        train(
            replace(
                TrainingRunConfig(),
                league_fraction=0.2,
            ),
            output,
        )

    assert not output.exists()


def test_official_train_rejects_arm_drift_before_creating_output(
    tmp_path: Path,
) -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    output = tmp_path / "must-not-exist"

    with pytest.raises(TrainerError, match="does not match fixed arm"):
        train(replace(config, max_updates=195), output)

    assert not output.exists()


def test_official_train_rejects_missing_source_commit_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(TrainerError, match="exact Git source commit"):
        train(config, output)

    assert not output.exists()


def test_official_train_rejects_commit_change_during_provenance_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    output = tmp_path / "must-not-exist"
    commits = iter(
        (
            "0123456789abcdef0123456789abcdef01234567\n",
            "fedcba9876543210fedcba9876543210fedcba98\n",
        )
    )

    def changing_git_result(
        args: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        stdout = next(commits) if args[-2:] == ("rev-parse", "HEAD") else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", changing_git_result)

    with pytest.raises(TrainerError, match="changed while reading Git provenance"):
        train(config, output)

    assert not output.exists()


def test_official_train_passes_captured_commit_to_update_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    output = tmp_path / "official"
    captured: dict[str, object] = {}
    _pretend_source_tree_is_clean(monkeypatch)

    def fake_run_updates(
        run_config: TrainingRunConfig,
        run_dir: Path,
        **kwargs: object,
    ) -> TrainingRunResult:
        captured["config"] = run_config
        captured["official_repository_commit"] = kwargs["official_repository_commit"]
        return TrainingRunResult(
            run_dir=run_dir,
            final_checkpoint=run_dir / "checkpoints" / "latest",
            completed_updates=0,
            completed_episodes=0,
            completed_decisions=0,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(trainer_module, "_run_updates", fake_run_updates)

    result = train(config, output)

    assert result.run_dir == output.resolve()
    assert captured["config"] == config
    assert captured["official_repository_commit"] == ("0123456789abcdef0123456789abcdef01234567")


def test_resume_rejects_historical_unsupported_controls_before_creating_output(
    tmp_path: Path,
) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        games_per_cell=1,
        max_updates=1,
        parallel=ParallelConfig(
            workers=1,
            active_games_per_worker=4,
            max_inference_batch=32,
        ),
    )
    checkpoint = train(config, tmp_path / "source").final_checkpoint
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_config"].update(
        {
            "checkpoint_interval_seconds": 60.0,
            "keep_periodic_checkpoints": 2,
            "evaluation_interval_seconds": 120.0,
            "evaluation_games_per_seat_cell": 4,
            "evaluate_at_start": True,
            "evaluate_at_end": True,
            "league_fraction": 0.2,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="checkpoint_interval_seconds"):
        resume(
            checkpoint,
            output,
            max_additional_updates=1,
        )

    assert not output.exists()


def test_official_resume_rejects_override_and_budget_overrun_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(tmp_path / "official")
    config = TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))
    _pretend_source_tree_is_clean(monkeypatch)

    override_output = tmp_path / "override-must-not-exist"
    with pytest.raises(TrainerError, match="exactly match"):
        resume(
            checkpoint,
            override_output,
            max_additional_updates=1,
            config_override=replace(config, learner_threads=3),
        )
    assert not override_output.exists()

    budget_output = tmp_path / "budget-must-not-exist"
    with pytest.raises(TrainerError, match="exceeds its fixed update budget"):
        resume(
            checkpoint,
            budget_output,
            max_additional_updates=2,
        )
    assert not budget_output.exists()


def test_official_resume_rejects_progress_drift_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(
        tmp_path / "official",
        episode_offset=-1,
    )
    output = tmp_path / "must-not-exist"
    _pretend_source_tree_is_clean(monkeypatch)

    with pytest.raises(TrainerError, match="exact resumable fixed-budget prefix"):
        resume(checkpoint, output, max_additional_updates=1)

    assert not output.exists()


def test_official_resume_rejects_source_commit_drift_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(
        tmp_path / "official",
        repository_commit="fedcba9876543210fedcba9876543210fedcba98",
    )
    output = tmp_path / "must-not-exist"
    _pretend_source_tree_is_clean(monkeypatch)

    with pytest.raises(TrainerError, match="exact source commit"):
        resume(checkpoint, output, max_additional_updates=1)

    assert not output.exists()


def test_official_resume_without_explicit_limit_runs_only_to_terminal_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(tmp_path / "official")
    output = tmp_path / "resumed"
    captured: dict[str, object] = {}
    _pretend_source_tree_is_clean(monkeypatch)

    def fake_run_updates(
        config: TrainingRunConfig,
        run_dir: Path,
        **kwargs: object,
    ) -> TrainingRunResult:
        captured["config"] = config
        captured["limit"] = kwargs["max_updates_this_run"]
        captured["official_repository_commit"] = kwargs["official_repository_commit"]
        progress = kwargs["initial_progress"]
        assert isinstance(progress, TrainingProgress)
        return TrainingRunResult(
            run_dir=run_dir,
            final_checkpoint=checkpoint,
            completed_updates=progress.next_update_index,
            completed_episodes=progress.completed_episodes,
            completed_decisions=progress.completed_decisions,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(trainer_module, "_run_updates", fake_run_updates)

    result = resume(checkpoint, output)

    assert result.run_dir == output.resolve()
    assert captured["limit"] == 196
    assert captured["official_repository_commit"] == ("0123456789abcdef0123456789abcdef01234567")
    assert output.is_dir()
    assert (output / "metrics.jsonl").read_bytes() == (
        tmp_path / "official" / "metrics.jsonl"
    ).read_bytes()


def test_official_run_rejects_source_drift_during_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pretend_source_tree_is_clean(monkeypatch)

    with pytest.raises(TrainerError, match="source changed during training"):
        trainer_module._require_official_source_unchanged(  # noqa: SLF001
            "fedcba9876543210fedcba9876543210fedcba98"
        )


def test_official_update_rejects_source_drift_before_writing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        games_per_cell=1,
        max_updates=1,
        parallel=ParallelConfig(
            workers=1,
            active_games_per_worker=1,
            max_inference_batch=16,
        ),
    )
    model = NeuralPolicy(training_encoder_config(), training_model_config("small"))
    trainer = PPOTrainer(model, config.ppo)
    source_checks = 0
    written_artifacts: list[str] = []

    def require_unchanged(repository_commit: str) -> None:
        nonlocal source_checks
        assert repository_commit == "0123456789abcdef0123456789abcdef01234567"
        source_checks += 1
        if source_checks == 2:
            raise TrainerError("official experiment source changed during training")

    def collect_one_update(
        *args: object, **kwargs: object
    ) -> tuple[RolloutBatch, CollectorMetrics]:
        del args, kwargs
        return (
            cast(RolloutBatch, object()),
            CollectorMetrics(
                games=1,
                decisions=1,
                elapsed_seconds=1.0,
                inference_seconds=0.0,
                inference_batches=1,
                inference_batch_sizes=(1,),
                cell_games=(("live-A", 3, 1),),
            ),
        )

    monkeypatch.setattr(trainer_module, "_require_official_source_unchanged", require_unchanged)
    monkeypatch.setattr(trainer_module, "_collect", collect_one_update)
    monkeypatch.setattr(trainer, "update", lambda *args, **kwargs: object())
    monkeypatch.setattr(trainer_module, "_update_metrics", lambda *args: {"update_index": 0})
    monkeypatch.setattr(
        trainer_module,
        "_append_json_line",
        lambda *args: written_artifacts.append("metrics"),
    )
    monkeypatch.setattr(
        trainer_module,
        "save_training_checkpoint",
        lambda *args, **kwargs: written_artifacts.append("checkpoint"),
    )

    with pytest.raises(TrainerError, match="source changed during training"):
        trainer_module._run_updates_with_pool(  # noqa: SLF001
            config,
            tmp_path,
            model=model,
            trainer=trainer,
            initial_progress=TrainingProgress(0, 0, 0, ()),
            lineage=(),
            max_updates_this_run=1,
            games_per_cell=1,
            vector_pool=None,
            official_repository_commit=("0123456789abcdef0123456789abcdef01234567"),
        )

    assert source_checks == 2
    assert written_artifacts == []


def test_official_resume_rejects_missing_metrics_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(tmp_path / "official")
    (tmp_path / "official" / "metrics.jsonl").unlink()
    output = tmp_path / "must-not-exist"
    _pretend_source_tree_is_clean(monkeypatch)

    with pytest.raises(TrainerError, match="requires readable prior metrics"):
        resume(checkpoint, output, max_additional_updates=1)

    assert not output.exists()


def test_official_resume_rejects_metrics_gap_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _official_checkpoint(tmp_path / "official")
    metrics_path = tmp_path / "official" / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    lines[12] = json.dumps({"update_index": 13})
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    _pretend_source_tree_is_clean(monkeypatch)

    with pytest.raises(TrainerError, match="indices are not contiguous"):
        resume(checkpoint, output, max_additional_updates=1)

    assert not output.exists()


def test_target_decisions_use_measured_decisions_per_game() -> None:
    config = TrainingRunConfig(
        games_per_cell=None,
        target_decisions_per_update=8192,
    )

    assert (
        resolve_games_per_cell(
            config,
            estimated_decisions_per_game=76.0,
        )
        == 8
    )
