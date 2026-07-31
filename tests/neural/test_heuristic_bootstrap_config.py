from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.behavior_cloning import (  # noqa: E402
    BehaviorCloningConfig,
)
from garboid_pocketrocks.neural.heuristic_auxiliary import (  # noqa: E402
    HeuristicAuxiliaryValueConfig,
)
from garboid_pocketrocks.neural.ppo import PPOConfig  # noqa: E402
from garboid_pocketrocks.neural.run_config import (
    ParallelConfig,  # noqa: E402
    TrainingRunConfig,  # noqa: E402
)
from garboid_pocketrocks.neural.trainer import train  # noqa: E402


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_old_training_configs_default_to_no_heuristic_bootstrap() -> None:
    config = TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))

    assert config.behavior_cloning is None
    assert config.heuristic_auxiliary == HeuristicAuxiliaryValueConfig()
    assert config.opponent_training == "mirror-self-play"


@pytest.mark.parametrize(
    "config",
    (
        TrainingRunConfig(
            root_seed=42,
            games_per_cell=119,
            max_updates=192,
            behavior_cloning=BehaviorCloningConfig(
                root_seed=42,
                rounds=4,
                games_per_cell=119,
            ),
        ),
        TrainingRunConfig(
            root_seed=42,
            games_per_cell=119,
            max_updates=196,
            heuristic_auxiliary=HeuristicAuxiliaryValueConfig.balanced_v3(),
        ),
        TrainingRunConfig(
            root_seed=42,
            games_per_cell=119,
            max_updates=196,
            opponent_training="heuristic-opponent-curriculum-v1",
        ),
    ),
)
def test_each_heuristic_training_input_round_trips_separately(
    tmp_path: Path,
    config: TrainingRunConfig,
) -> None:

    loaded = TrainingRunConfig.from_json(
        _write_config(tmp_path / "config.json", config.to_json_dict())
    )

    assert loaded == config


def test_heuristic_strategies_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="separate ablations"):
        TrainingRunConfig(
            root_seed=42,
            behavior_cloning=BehaviorCloningConfig(
                root_seed=42,
                rounds=1,
                games_per_cell=1,
            ),
            heuristic_auxiliary=HeuristicAuxiliaryValueConfig.balanced_v3(),
        )


def test_behavior_cloning_seed_must_match_the_training_lineage() -> None:
    with pytest.raises(ValueError, match="root seeds must match"):
        TrainingRunConfig(
            root_seed=42,
            behavior_cloning=BehaviorCloningConfig(
                root_seed=43,
                rounds=1,
                games_per_cell=1,
            ),
        )


def test_unknown_opponent_schedule_fails_closed(tmp_path: Path) -> None:
    payload = TrainingRunConfig().to_json_dict()
    payload["opponent_training"] = "latest-heuristic"

    with pytest.raises(ValueError, match="immutable schedule"):
        TrainingRunConfig.from_json(_write_config(tmp_path / "bad.json", payload))


@pytest.mark.parametrize(
    ("strategy", "changes"),
    (
        (
            "behavior-cloning",
            {
                "behavior_cloning": BehaviorCloningConfig(
                    root_seed=42,
                    rounds=1,
                    games_per_cell=1,
                    epochs=1,
                    minibatch_size=64,
                )
            },
        ),
        (
            "heuristic-auxiliary",
            {"heuristic_auxiliary": HeuristicAuxiliaryValueConfig.balanced_v3()},
        ),
        (
            "heuristic-curriculum",
            {"opponent_training": "heuristic-opponent-curriculum-v1"},
        ),
    ),
)
def test_each_strategy_completes_a_real_training_update(
    tmp_path: Path,
    strategy: str,
    changes: dict[str, object],
) -> None:
    config_values: dict[str, object] = {
        "root_seed": 42,
        "device": "cpu",
        "deterministic_algorithms": True,
        "model_profile": "small",
        "learner_threads": 1,
        "games_per_cell": 1,
        "max_updates": 1,
        "parallel": ParallelConfig(
            workers=1,
            active_games_per_worker=8,
            max_inference_batch=64,
        ),
        "ppo": PPOConfig(epochs=1, minibatch_size=64),
    }
    config_values.update(changes)

    result = train(
        TrainingRunConfig(**config_values),  # type: ignore[arg-type]
        tmp_path / strategy,
    )

    assert result.completed_updates == 1
    assert result.completed_episodes == 15
    assert result.final_checkpoint.is_dir()
    if strategy == "behavior-cloning":
        assert (result.run_dir / "behavior-cloning.json").is_file()
    metrics = json.loads((result.run_dir / "metrics.jsonl").read_text(encoding="utf-8"))
    included = metrics["ppo"]["heuristic_auxiliary"]["included_count"]
    assert (included > 0) == (strategy == "heuristic-auxiliary")
