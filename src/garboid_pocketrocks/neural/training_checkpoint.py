"""Atomic, checksummed, resumable neural training checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from garboid_pocketrocks.neural.checkpoint import (
    InferenceManifest,
    _file_sha256,
    _read_encoder_config,
    _read_model_config,
    _require_state_dict,
    _validate_state_schema,
    parameter_digest,
    save_inference_checkpoint,
)
from garboid_pocketrocks.neural.config import (
    ModelProfile,
    NeuralEncoderConfig,
    NeuralModelConfig,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.neural.ppo import PPOConfig
from garboid_pocketrocks.neural.run_config import (
    DeviceName,
    ParallelConfig,
    TrainingRunConfig,
    WorkerSetting,
)
from garboid_pocketrocks.training.rewards import RewardConfig

_FILES = {
    "manifest.json",
    "model.pt",
    "optimizer.pt",
    "rng.pt",
    "metrics.json",
}
_PAYLOAD_FILES = ("metrics.json", "model.pt", "optimizer.pt", "rng.pt")


class TrainingCheckpointError(ValueError):
    """Raised when a training checkpoint is invalid or cannot be persisted."""


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Update-boundary counters needed to continue a run."""

    next_update_index: int
    completed_episodes: int
    completed_decisions: int
    cell_games: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_games", tuple(self.cell_games))
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (
                self.next_update_index,
                self.completed_episodes,
                self.completed_decisions,
            )
        ):
            raise TrainingCheckpointError("training progress counters must be nonnegative integers")
        if any(
            not ruleset or player_count not in (3, 4, 5) or games < 0
            for ruleset, player_count, games in self.cell_games
        ):
            raise TrainingCheckpointError("training progress cells are invalid")


@dataclass(frozen=True, slots=True)
class TrainingCheckpointManifest:
    """Versioned metadata and integrity declarations for a training bundle."""

    schema_version: int
    repository_commit: str
    encoder_config: NeuralEncoderConfig
    model_config: NeuralModelConfig
    run_config: TrainingRunConfig
    progress: TrainingProgress
    lineage: tuple[str, ...]
    champion_identity: str | None
    league_identities: tuple[str, ...]
    parameter_digest: str = ""
    file_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", tuple(self.lineage))
        object.__setattr__(
            self,
            "league_identities",
            tuple(self.league_identities),
        )
        object.__setattr__(self, "file_sha256", tuple(self.file_sha256))


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    """Validated model, optimizer, random generators, and progress."""

    model: NeuralPolicy
    optimizer: torch.optim.Adam
    manifest: TrainingCheckpointManifest
    generator_states: dict[str, Tensor]
    metrics: dict[str, object]


def save_training_checkpoint(
    path: Path,
    *,
    model: NeuralPolicy,
    optimizer: torch.optim.Optimizer,
    manifest: TrainingCheckpointManifest,
    generator_states: Mapping[str, Tensor],
    metrics: Mapping[str, object],
) -> Path:
    """Atomically replace a checkpoint only after validating the new bundle."""

    _validate_manifest(manifest, require_integrity=False)
    if (
        model.encoder_config != manifest.encoder_config
        or model.model_config != manifest.model_config
    ):
        raise TrainingCheckpointError("manifest configs do not match model")
    if any(not torch.isfinite(item).all().item() for item in model.state_dict().values()):
        raise TrainingCheckpointError("cannot save nonfinite model weights")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    backup = parent / f".{path.name}.backup-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        torch.save(model.state_dict(), temporary / "model.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(
            {name: state.detach().cpu() for name, state in generator_states.items()},
            temporary / "rng.pt",
        )
        _write_json(temporary / "metrics.json", dict(metrics))
        checksums = tuple((name, _file_sha256(temporary / name)) for name in _PAYLOAD_FILES)
        persisted = replace(
            manifest,
            parameter_digest=parameter_digest(model.state_dict()),
            file_sha256=checksums,
        )
        _validate_manifest(persisted, require_integrity=True)
        _write_json(temporary / "manifest.json", asdict(persisted))
        _sync_directory(temporary)
        load_training_checkpoint(
            temporary,
            device=next(model.parameters()).device,
        )
        if path.exists():
            os.replace(path, backup)
        os.replace(temporary, path)
        _sync_directory(parent)
        if backup.exists():
            shutil.rmtree(backup)
        return path
    except Exception as error:
        if backup.exists() and not path.exists():
            os.replace(backup, path)
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(error, TrainingCheckpointError):
            raise
        raise TrainingCheckpointError(f"training checkpoint save failed: {error}") from error


def load_training_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedTrainingCheckpoint:
    """Verify every declared byte and reconstruct update-boundary state."""

    if not path.is_dir():
        raise TrainingCheckpointError("checkpoint path must be a directory")
    if {item.name for item in path.iterdir()} != _FILES:
        raise TrainingCheckpointError("training checkpoint files are not exact")
    manifest = _read_manifest(path / "manifest.json")
    _validate_manifest(manifest, require_integrity=True)
    for name, expected in manifest.file_sha256:
        if _file_sha256(path / name) != expected:
            raise TrainingCheckpointError(f"checkpoint checksum mismatch for {name}")
    try:
        model_value: object = torch.load(
            path / "model.pt",
            map_location=device,
            weights_only=True,
        )
        optimizer_value: object = torch.load(
            path / "optimizer.pt",
            map_location=device,
            weights_only=True,
        )
        rng_value: object = torch.load(
            path / "rng.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        metrics_value: object = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise TrainingCheckpointError("training checkpoint payload could not be loaded") from error

    state_dict = _require_state_dict(model_value)
    model = NeuralPolicy(
        manifest.encoder_config,
        manifest.model_config,
    ).to(device)
    _validate_state_schema(state_dict, model.state_dict())
    if any(not torch.isfinite(item).all().item() for item in state_dict.values()):
        raise TrainingCheckpointError("checkpoint model contains nonfinite values")
    if parameter_digest(state_dict) != manifest.parameter_digest:
        raise TrainingCheckpointError("checkpoint parameter digest mismatch")
    model.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=manifest.run_config.ppo.learning_rate,
        foreach=False,
    )
    if not isinstance(optimizer_value, dict):
        raise TrainingCheckpointError("optimizer payload must be a dictionary")
    try:
        optimizer.load_state_dict(optimizer_value)
    except (KeyError, RuntimeError, ValueError) as error:
        raise TrainingCheckpointError("optimizer state is incompatible") from error
    if not _nested_tensors_finite(optimizer.state_dict()):
        raise TrainingCheckpointError("optimizer contains nonfinite state")

    generator_states = _generator_states(rng_value)
    if not isinstance(metrics_value, dict) or any(
        not isinstance(key, str) for key in metrics_value
    ):
        raise TrainingCheckpointError("metrics must be a JSON object")
    return LoadedTrainingCheckpoint(
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        generator_states=generator_states,
        metrics=cast(dict[str, object], metrics_value),
    )


def export_inference_checkpoint(
    training_checkpoint: Path,
    output_path: Path,
    *,
    device: torch.device,
) -> Path:
    """Export a portable model-only bundle from a validated training bundle."""

    loaded = load_training_checkpoint(training_checkpoint, device=device)
    manifest = InferenceManifest.create(
        loaded.model,
        repository_commit=loaded.manifest.repository_commit,
        root_seed=loaded.manifest.run_config.root_seed,
        completed_episodes=loaded.manifest.progress.completed_episodes,
        completed_updates=loaded.manifest.progress.next_update_index,
    )
    save_inference_checkpoint(output_path, loaded.model, manifest)
    return output_path


def _validate_manifest(
    manifest: TrainingCheckpointManifest,
    *,
    require_integrity: bool,
) -> None:
    if manifest.schema_version != 1:
        raise TrainingCheckpointError("unsupported training checkpoint schema")
    if not manifest.repository_commit:
        raise TrainingCheckpointError("repository commit must be nonempty")
    if manifest.encoder_config.schema_version != 1:
        raise TrainingCheckpointError("unsupported encoder schema")
    if any(not identity for identity in (*manifest.lineage, *manifest.league_identities)):
        raise TrainingCheckpointError("checkpoint identities must be nonempty")
    if manifest.champion_identity is not None and not manifest.champion_identity:
        raise TrainingCheckpointError("champion identity must be nonempty")
    if require_integrity:
        if len(manifest.parameter_digest) != 64:
            raise TrainingCheckpointError("parameter digest is invalid")
        if tuple(name for name, _ in manifest.file_sha256) != _PAYLOAD_FILES:
            raise TrainingCheckpointError("checkpoint checksum declarations are invalid")
        if any(len(digest) != 64 for _, digest in manifest.file_sha256):
            raise TrainingCheckpointError("checkpoint checksum is invalid")


def _read_manifest(path: Path) -> TrainingCheckpointManifest:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
        payload = _object(value, "manifest")
        _exact(
            payload,
            {
                "schema_version",
                "repository_commit",
                "encoder_config",
                "model_config",
                "run_config",
                "progress",
                "lineage",
                "champion_identity",
                "league_identities",
                "parameter_digest",
                "file_sha256",
            },
            "manifest",
        )
        progress_payload = _object(payload["progress"], "progress")
        _exact(
            progress_payload,
            {
                "next_update_index",
                "completed_episodes",
                "completed_decisions",
                "cell_games",
            },
            "progress",
        )
        return TrainingCheckpointManifest(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            repository_commit=_string(payload["repository_commit"], "repository_commit"),
            encoder_config=_read_encoder_config(payload["encoder_config"]),
            model_config=_read_model_config(payload["model_config"]),
            run_config=_run_config(payload["run_config"]),
            progress=TrainingProgress(
                next_update_index=_integer(
                    progress_payload["next_update_index"],
                    "next_update_index",
                ),
                completed_episodes=_integer(
                    progress_payload["completed_episodes"],
                    "completed_episodes",
                ),
                completed_decisions=_integer(
                    progress_payload["completed_decisions"],
                    "completed_decisions",
                ),
                cell_games=tuple(
                    (
                        _string(item[0], "cell ruleset"),
                        _integer(item[1], "cell player count"),
                        _integer(item[2], "cell games"),
                    )
                    for item in _triples(
                        progress_payload["cell_games"],
                        "cell_games",
                    )
                ),
            ),
            lineage=_strings(payload["lineage"], "lineage"),
            champion_identity=_optional_string(
                payload["champion_identity"],
                "champion_identity",
            ),
            league_identities=_strings(
                payload["league_identities"],
                "league_identities",
            ),
            parameter_digest=_string(
                payload["parameter_digest"],
                "parameter_digest",
            ),
            file_sha256=tuple(
                (
                    _string(item[0], "checksum filename"),
                    _string(item[1], "checksum"),
                )
                for item in _pairs(payload["file_sha256"], "file_sha256")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, TrainingCheckpointError):
            raise
        raise TrainingCheckpointError("manifest values are invalid") from error


def _run_config(value: object) -> TrainingRunConfig:
    payload = _object(value, "run_config")
    expected = set(TrainingRunConfig().to_json_dict())
    legacy_optional = {
        "learner_threads",
        "deterministic_algorithms",
        "model_profile",
        "bot_generation",
        "opponent_training",
    }
    if not expected - legacy_optional <= set(payload) <= expected:
        raise TrainingCheckpointError("run_config fields do not match schema")
    parallel = _object(payload["parallel"], "parallel")
    ppo = _object(payload["ppo"], "ppo")
    reward = _object(payload["reward"], "reward")
    return TrainingRunConfig(
        root_seed=_integer(payload["root_seed"], "root_seed"),
        device=_device(payload["device"]),
        deterministic_algorithms=_boolean(
            payload.get("deterministic_algorithms", True),
            "deterministic_algorithms",
        ),
        model_profile=_model_profile(
            payload.get("model_profile", "small"),
        ),
        bot_generation=_integer(
            payload.get("bot_generation", 1),
            "bot_generation",
        ),
        learner_threads=_integer(
            payload.get("learner_threads", 1),
            "learner_threads",
        ),
        games_per_cell=_optional_integer(payload["games_per_cell"], "games_per_cell"),
        max_updates=_optional_integer(payload["max_updates"], "max_updates"),
        max_wall_seconds=_optional_number(payload["max_wall_seconds"], "max_wall_seconds"),
        target_decisions_per_update=_optional_integer(
            payload["target_decisions_per_update"],
            "target_decisions_per_update",
        ),
        checkpoint_interval_seconds=_optional_number(
            payload["checkpoint_interval_seconds"],
            "checkpoint_interval_seconds",
        ),
        evaluation_interval_seconds=_optional_number(
            payload["evaluation_interval_seconds"],
            "evaluation_interval_seconds",
        ),
        evaluation_games_per_seat_cell=_integer(
            payload["evaluation_games_per_seat_cell"],
            "evaluation_games_per_seat_cell",
        ),
        evaluate_at_start=_boolean(payload["evaluate_at_start"], "evaluate_at_start"),
        evaluate_at_end=_boolean(payload["evaluate_at_end"], "evaluate_at_end"),
        league_fraction=_number(payload["league_fraction"], "league_fraction"),
        keep_periodic_checkpoints=_integer(
            payload["keep_periodic_checkpoints"],
            "keep_periodic_checkpoints",
        ),
        opponent_training=payload.get("opponent_training", "mirror-self-play"),  # type: ignore[arg-type]
        parallel=ParallelConfig(
            workers=_workers(parallel["workers"]),
            active_games_per_worker=_integer(
                parallel["active_games_per_worker"],
                "active_games_per_worker",
            ),
            max_inference_batch=_integer(
                parallel["max_inference_batch"],
                "max_inference_batch",
            ),
            max_queue_delay_ms=_number(
                parallel["max_queue_delay_ms"],
                "max_queue_delay_ms",
            ),
            max_stale_approximate_kl=_number(
                parallel.get("max_stale_approximate_kl", 0.01),
                "max_stale_approximate_kl",
            ),
            max_stale_clip_fraction=_number(
                parallel.get("max_stale_clip_fraction", 0.15),
                "max_stale_clip_fraction",
            ),
        ),
        ppo=PPOConfig(
            gamma=_number(ppo["gamma"], "gamma"),
            gae_lambda=_number(ppo["gae_lambda"], "gae_lambda"),
            clip_ratio=_number(ppo["clip_ratio"], "clip_ratio"),
            value_loss_coefficient=_number(
                ppo["value_loss_coefficient"],
                "value_loss_coefficient",
            ),
            entropy_coefficient=_number(
                ppo["entropy_coefficient"],
                "entropy_coefficient",
            ),
            max_gradient_norm=_number(
                ppo["max_gradient_norm"],
                "max_gradient_norm",
            ),
            learning_rate=_number(ppo["learning_rate"], "learning_rate"),
            epochs=_integer(ppo["epochs"], "epochs"),
            minibatch_size=_integer(ppo["minibatch_size"], "minibatch_size"),
        ),
        reward=RewardConfig(
            accounting_weight=_number(
                reward["accounting_weight"],
                "accounting_weight",
            ),
            win_bonus=_number(reward["win_bonus"], "win_bonus"),
            placement_bonuses=tuple(
                _number(item, "placement bonus")
                for item in _array(
                    reward["placement_bonuses"],
                    "placement_bonuses",
                )
            ),
            invalid_action_penalty=_number(
                reward["invalid_action_penalty"],
                "invalid_action_penalty",
            ),
            event_bonuses=tuple(
                (
                    _string(item[0], "event kind"),
                    _number(item[1], "event bonus"),
                )
                for item in _pairs(reward["event_bonuses"], "event_bonuses")
            ),
        ),
    )


def _write_json(path: Path, value: object) -> None:
    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        path.write_text(encoded, encoding="utf-8")
        with path.open("rb") as file:
            os.fsync(file.fileno())
    except (OSError, TypeError, ValueError) as error:
        raise TrainingCheckpointError(f"JSON payload could not be saved: {error}") from error


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _generator_states(value: object) -> dict[str, Tensor]:
    if not isinstance(value, Mapping):
        raise TrainingCheckpointError("RNG payload must be a mapping")
    output: dict[str, Tensor] = {}
    for name, state in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(state, Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
        ):
            raise TrainingCheckpointError("RNG payload contains an invalid state")
        output[name] = state
    return output


def _nested_tensors_finite(value: object) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_nested_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_tensors_finite(item) for item in value)
    return True


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrainingCheckpointError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _exact(payload: Mapping[str, object], fields: set[str], name: str) -> None:
    if set(payload) != fields:
        raise TrainingCheckpointError(f"{name} fields do not match schema")


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TrainingCheckpointError(f"{name} must be an array")
    return cast(list[object], value)


def _pairs(value: object, name: str) -> list[list[object]]:
    items = _array(value, name)
    if any(not isinstance(item, list) or len(item) != 2 for item in items):
        raise TrainingCheckpointError(f"{name} must contain pairs")
    return cast(list[list[object]], items)


def _triples(value: object, name: str) -> list[list[object]]:
    items = _array(value, name)
    if any(not isinstance(item, list) or len(item) != 3 for item in items):
        raise TrainingCheckpointError(f"{name} must contain triples")
    return cast(list[list[object]], items)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TrainingCheckpointError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _strings(value: object, name: str) -> tuple[str, ...]:
    items = _array(value, name)
    if any(not isinstance(item, str) for item in items):
        raise TrainingCheckpointError(f"{name} must contain strings")
    return tuple(cast(list[str], items))


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingCheckpointError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingCheckpointError(f"{name} must be a number")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingCheckpointError(f"{name} must be a boolean")
    return value


def _device(value: object) -> DeviceName:
    if value not in ("auto", "cpu", "cuda", "mps"):
        raise TrainingCheckpointError("device is invalid")
    return value


def _model_profile(value: object) -> ModelProfile:
    if value not in ("small", "medium", "large"):
        raise TrainingCheckpointError("model profile is invalid")
    return value


def _workers(value: object) -> WorkerSetting:
    if value == "auto":
        return "auto"
    return _integer(value, "workers")
