from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from torch.optim import Adam  # noqa: E402

from garboid_pocketrocks.neural.checkpoint import (  # noqa: E402
    load_inference_checkpoint,
    parameter_digest,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.ppo import PPOConfig  # noqa: E402
from garboid_pocketrocks.neural.run_config import TrainingRunConfig  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    CheckpointNeuralBrain,
    checkpoint_bot_spec,
)
from garboid_pocketrocks.neural.training_checkpoint import (  # noqa: E402
    TrainingCheckpointError,
    TrainingCheckpointManifest,
    TrainingProgress,
    export_inference_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _components() -> tuple[
    NeuralPolicy,
    Adam,
    TrainingCheckpointManifest,
]:
    torch.manual_seed(81)
    model = NeuralPolicy(training_encoder_config(), training_model_config("small"))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    loss = torch.stack(tuple(parameter.square().mean() for parameter in model.parameters())).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    run_config = TrainingRunConfig(
        games_per_cell=100,
        max_updates=1,
        ppo=PPOConfig(epochs=2),
    )
    manifest = TrainingCheckpointManifest(
        schema_version=1,
        repository_commit="0123456789abcdef",
        encoder_config=model.encoder_config,
        model_config=model.model_config,
        run_config=run_config,
        progress=TrainingProgress(
            next_update_index=3,
            completed_episodes=1_500,
            completed_decisions=27_000,
            cell_games=(("live-A", 3, 100),),
        ),
        lineage=("root",),
        champion_identity=None,
        league_identities=(),
    )
    return model, optimizer, manifest


def _save(path: Path) -> Path:
    model, optimizer, manifest = _components()
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        generator_states={"policy": torch.Generator(device="cpu").manual_seed(7).get_state()},
        metrics={"games_per_second": 123.5, "updates": [1, 2]},
    )


def test_training_checkpoint_contains_exact_validated_files(
    tmp_path: Path,
) -> None:
    saved = _save(tmp_path / "checkpoint")

    assert {item.name for item in saved.iterdir()} == {
        "manifest.json",
        "model.pt",
        "optimizer.pt",
        "rng.pt",
        "metrics.json",
    }
    loaded = load_training_checkpoint(saved, device=torch.device("cpu"))
    assert loaded.manifest.progress.next_update_index == 3
    assert parameter_digest(loaded.model.state_dict()) == loaded.manifest.parameter_digest
    assert loaded.optimizer.state_dict()["state"]
    assert loaded.generator_states["policy"].dtype == torch.uint8
    assert loaded.metrics["games_per_second"] == 123.5


def test_training_checkpoint_loads_historical_unsupported_run_controls(
    tmp_path: Path,
) -> None:
    model, optimizer, manifest = _components()
    historical_config = replace(
        manifest.run_config,
        checkpoint_interval_seconds=60.0,
        keep_periodic_checkpoints=2,
        evaluation_interval_seconds=120.0,
        evaluation_games_per_seat_cell=4,
        evaluate_at_start=True,
        evaluate_at_end=True,
        league_fraction=0.2,
    )
    checkpoint = save_training_checkpoint(
        tmp_path / "historical-checkpoint",
        model=model,
        optimizer=optimizer,
        manifest=replace(manifest, run_config=historical_config),
        generator_states={"policy": torch.Generator(device="cpu").manual_seed(7).get_state()},
        metrics={},
    )

    loaded = load_training_checkpoint(checkpoint, device=torch.device("cpu"))

    assert loaded.manifest.run_config == historical_config


def test_training_checkpoint_defaults_legacy_bot_generation_to_one(
    tmp_path: Path,
) -> None:
    checkpoint = _save(tmp_path / "legacy-checkpoint")
    manifest_path = checkpoint / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["run_config"]["bot_generation"]
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_training_checkpoint(checkpoint, device=torch.device("cpu"))

    assert loaded.manifest.run_config.bot_generation == 1


def test_exported_training_checkpoint_builds_tournament_bot(
    tmp_path: Path,
) -> None:
    checkpoint = _save(tmp_path / "training-checkpoint")
    inference = export_inference_checkpoint(
        checkpoint,
        tmp_path / "inference-checkpoint",
        device=torch.device("cpu"),
    )

    spec = checkpoint_bot_spec("candidate-v2", inference)
    brain = spec.make_brain(seed=9)

    assert spec.name == "candidate-v2"
    assert spec.bot_id == "candidate-v2"
    assert isinstance(brain, CheckpointNeuralBrain)


def test_failed_save_does_not_replace_last_valid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _save(tmp_path / "checkpoint")
    original = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))[
        "parameter_digest"
    ]
    monkeypatch.setattr(torch, "save", Mock(side_effect=OSError("disk full")))

    model, optimizer, manifest = _components()
    with pytest.raises(TrainingCheckpointError, match="disk full"):
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            generator_states={},
            metrics={},
        )

    current = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))[
        "parameter_digest"
    ]
    assert current == original
    load_training_checkpoint(checkpoint, device=torch.device("cpu"))


def test_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path / "checkpoint")
    optimizer_path = checkpoint / "optimizer.pt"
    optimizer_path.write_bytes(optimizer_path.read_bytes() + b"corrupt")

    with pytest.raises(TrainingCheckpointError, match="checksum"):
        load_training_checkpoint(checkpoint, device=torch.device("cpu"))


def test_update_boundary_resume_matches_uninterrupted_optimizer_step(
    tmp_path: Path,
) -> None:
    model, optimizer, manifest = _components()
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        generator_states={},
        metrics={},
    )
    resumed = load_training_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )

    for candidate, candidate_optimizer in (
        (model, optimizer),
        (resumed.model, resumed.optimizer),
    ):
        loss = torch.stack(
            tuple(
                (index + 1) * parameter.square().mean()
                for index, parameter in enumerate(candidate.parameters())
            )
        ).sum()
        loss.backward()
        candidate_optimizer.step()
        candidate_optimizer.zero_grad(set_to_none=True)

    assert parameter_digest(model.state_dict()) == parameter_digest(resumed.model.state_dict())
    original_state = optimizer.state_dict()
    resumed_state = resumed.optimizer.state_dict()
    assert original_state["param_groups"] == resumed_state["param_groups"]
    for parameter_id, state in original_state["state"].items():
        for name, value in state.items():
            other = resumed_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other


def test_training_checkpoint_exports_portable_inference_bundle(
    tmp_path: Path,
) -> None:
    training = _save(tmp_path / "training")
    inference = export_inference_checkpoint(
        training,
        tmp_path / "inference",
        device=torch.device("cpu"),
    )

    loaded = load_inference_checkpoint(inference, device=torch.device("cpu"))
    assert loaded.manifest.supported_ruleset_names == (
        "live-A",
        "live-B",
        "live-C",
        "live-D",
        "live-E",
    )
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
