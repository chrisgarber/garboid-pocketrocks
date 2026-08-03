from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.run_config import (  # noqa: E402
    ParallelConfig,
    TrainingRunConfig,
    validate_runtime_support,
)
from garboid_pocketrocks.neural.trainer import (  # noqa: E402
    inspect_checkpoint,
    resolve_games_per_cell,
    resume,
    train,
)


def test_low_volume_train_resume_and_inspect(tmp_path: Path) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        bot_generation=2,
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
    assert inspected["bot_id"] == "vector_ppo_small_v2_g15"
    assert inspected["bot_generation"] == 2
    assert resumed.completed_updates == 2
    assert resumed.completed_episodes == 30
    assert inspect_checkpoint(resumed.final_checkpoint)["learner_threads"] == 2
    assert (first.run_dir / "metrics.jsonl").is_file()
    assert (first.run_dir / "resolved-config.json").is_file()


def test_committed_profiles_have_exact_wall_envelopes() -> None:
    smoke = TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))
    initial = TrainingRunConfig.from_json(Path("configs/neural/initial-10m.json"))
    long = TrainingRunConfig.from_json(Path("configs/neural/long-8h.json"))
    preflight = TrainingRunConfig.from_json(
        Path("configs/neural/cold-mixed-mps-15m-preflight-v2.json")
    )

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
    assert preflight.bot_generation == 2
    assert preflight.max_wall_seconds == 900.0
    assert preflight.games_per_cell == 128
    assert preflight.ppo.epochs == 2
    assert preflight.ppo.minibatch_size == 8192
    assert preflight.checkpoint_interval_seconds == 300.0
    assert preflight.keep_periodic_checkpoints == 2
    validate_runtime_support(preflight)
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


def test_periodic_checkpoints_are_rotated_and_metrics_are_compact(
    tmp_path: Path,
) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        bot_generation=2,
        games_per_cell=1,
        max_updates=3,
        max_wall_seconds=None,
        checkpoint_interval_seconds=1e-9,
        keep_periodic_checkpoints=2,
        parallel=ParallelConfig(
            workers=1,
            active_games_per_worker=4,
            max_inference_batch=32,
        ),
    )

    result = train(config, tmp_path / "run")
    periodic = sorted((result.run_dir / "checkpoints" / "periodic").iterdir())
    metric_lines = (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    latest_metric = json.loads(metric_lines[-1])
    ppo = latest_metric["ppo"]

    assert [path.name for path in periodic] == [
        "update-00000001",
        "update-00000002",
    ]
    assert all(inspect_checkpoint(path)["bot_generation"] == 2 for path in periodic)
    assert len(metric_lines) == 3
    assert "advantages" not in ppo
    assert "ratios" not in ppo
    assert "values" not in ppo
    assert "entropies" not in ppo
    assert ppo["distributions"]["advantages"]["count"] == ppo["transition_count"]
    assert (result.run_dir / "metrics.jsonl").stat().st_size < 100_000


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

    with pytest.raises(ValueError, match="evaluation_interval_seconds"):
        resume(
            checkpoint,
            output,
            max_additional_updates=1,
        )

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
