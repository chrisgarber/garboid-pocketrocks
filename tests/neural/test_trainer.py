from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from garboid_pocketrocks.neural.run_config import (
    ParallelConfig,
    TrainingRunConfig,
)
from garboid_pocketrocks.neural.trainer import (
    inspect_checkpoint,
    resolve_games_per_cell,
    resume,
    train,
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
    assert initial.checkpoint_interval_seconds == 120.0
    assert long.max_wall_seconds == 28_800.0
    assert long.model_profile == "large"
    assert long.target_decisions_per_update == 131_072
    assert long.checkpoint_interval_seconds == 900.0
    assert long.evaluation_interval_seconds == 1_800.0
    assert long.learner_threads == 4


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
