"""Minimal, versioned inference checkpoints for Stage 1 neural policies."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from garboid_pocketrocks.neural.config import (
    NeuralEncoderConfig,
    NeuralModelConfig,
)
from garboid_pocketrocks.neural.model import NeuralPolicy
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds

_CHECKPOINT_SCHEMA_VERSION = 1
_STAGE1_RULESETS = ("live-A",)
_STAGE1_PLAYER_COUNTS = (3,)
_ENCODER_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "supported_ruleset_names",
        "supported_player_counts",
        "max_bid",
        "max_hand_size",
        "max_history_events",
        "max_cash",
        "max_abs_chart",
        "max_resource_cards",
        "max_action_cards",
    }
)
_MODEL_CONFIG_FIELDS = frozenset(
    {
        "categorical_embedding_size",
        "suit_embedding_size",
        "seat_hidden_size",
        "event_embedding_size",
        "gru_hidden_size",
        "snapshot_hidden_size",
        "trunk_hidden_size",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "encoder_schema_version",
        "repository_commit",
        "python_version",
        "torch_version",
        "numpy_version",
        "sdk_version",
        "encoder_config",
        "model_config",
        "action_space_size",
        "action_space_hash",
        "supported_ruleset_names",
        "supported_player_counts",
        "root_seed",
        "completed_episodes",
        "completed_updates",
        "model_sha256",
        "parameter_digest",
    }
)


class CheckpointError(ValueError):
    """Raised when an inference checkpoint is unsafe or incompatible."""


@dataclass(frozen=True, slots=True)
class InferenceManifest:
    """JSON-safe metadata required to reconstruct one inference policy."""

    schema_version: int
    encoder_schema_version: int
    repository_commit: str
    python_version: str
    torch_version: str
    numpy_version: str
    sdk_version: str
    encoder_config: NeuralEncoderConfig
    model_config: NeuralModelConfig
    action_space_size: int
    action_space_hash: str
    supported_ruleset_names: tuple[str, ...]
    supported_player_counts: tuple[int, ...]
    root_seed: int
    completed_episodes: int
    completed_updates: int
    model_sha256: str = ""
    parameter_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_ruleset_names",
            tuple(self.supported_ruleset_names),
        )
        object.__setattr__(
            self,
            "supported_player_counts",
            tuple(self.supported_player_counts),
        )

    @classmethod
    def create(
        cls,
        model: NeuralPolicy,
        *,
        repository_commit: str,
        root_seed: int,
        completed_episodes: int,
        completed_updates: int,
    ) -> InferenceManifest:
        """Create metadata whose integrity fields will be filled during save."""

        action_space_size, action_space_hash = _action_space_contract(model.encoder_config)
        return cls(
            schema_version=_CHECKPOINT_SCHEMA_VERSION,
            encoder_schema_version=model.encoder_config.schema_version,
            repository_commit=repository_commit,
            python_version=sys.version.split()[0],
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            sdk_version=version("pocketrocks-python-sdk"),
            encoder_config=model.encoder_config,
            model_config=model.model_config,
            action_space_size=action_space_size,
            action_space_hash=action_space_hash,
            supported_ruleset_names=model.encoder_config.supported_ruleset_names,
            supported_player_counts=model.encoder_config.supported_player_counts,
            root_seed=root_seed,
            completed_episodes=completed_episodes,
            completed_updates=completed_updates,
        )


@dataclass(frozen=True, slots=True)
class LoadedInferenceCheckpoint:
    """A validated eval-mode policy and its persisted manifest."""

    model: NeuralPolicy
    manifest: InferenceManifest


def parameter_digest(state_dict: Mapping[str, Tensor]) -> str:
    """Hash sorted parameter names, dtypes, shapes, and contiguous CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        _update_digest_part(digest, name.encode("utf-8"))
        _update_digest_part(digest, str(tensor.dtype).encode("ascii"))
        shape = json.dumps(tuple(tensor.shape), separators=(",", ":")).encode("ascii")
        _update_digest_part(digest, shape)
        raw_bytes = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).tobytes()
        _update_digest_part(digest, raw_bytes)
    return digest.hexdigest()


def save_inference_checkpoint(
    path: Path,
    model: NeuralPolicy,
    manifest: InferenceManifest,
) -> None:
    """Save only a model state dictionary and its validated JSON manifest."""

    _prepare_empty_destination(path)
    _validate_manifest(manifest, require_integrity=False)
    if (
        manifest.encoder_config != model.encoder_config
        or manifest.model_config != model.model_config
    ):
        raise CheckpointError("manifest configs do not match the model")

    state_dict = model.state_dict()
    with torch.random.fork_rng(devices=[]):
        canonical_state = NeuralPolicy(
            manifest.encoder_config,
            manifest.model_config,
        ).state_dict()
    _validate_state_schema(state_dict, canonical_state)
    if any(not torch.isfinite(tensor).all().item() for tensor in state_dict.values()):
        raise CheckpointError("cannot save nonfinite model weights")
    model_path = path / "model.pt"
    torch.save(state_dict, model_path)
    persisted = replace(
        manifest,
        model_sha256=_file_sha256(model_path),
        parameter_digest=parameter_digest(state_dict),
    )
    _validate_manifest(persisted, require_integrity=True)
    (path / "manifest.json").write_text(
        json.dumps(asdict(persisted), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_inference_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedInferenceCheckpoint:
    """Load and validate a trusted local Stage 1 inference checkpoint."""

    if not path.is_dir():
        raise CheckpointError("checkpoint path must be a directory")
    if {item.name for item in path.iterdir()} != {"manifest.json", "model.pt"}:
        raise CheckpointError("checkpoint bundle must contain only manifest.json and model.pt")

    manifest = _read_manifest(path / "manifest.json")
    _validate_manifest(manifest, require_integrity=True)
    model_path = path / "model.pt"
    if _file_sha256(model_path) != manifest.model_sha256:
        raise CheckpointError("model checksum mismatch")

    try:
        loaded: object = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError("model state could not be loaded") from error
    state_dict = _require_state_dict(loaded)
    model = NeuralPolicy(
        manifest.encoder_config,
        manifest.model_config,
    ).to(device)
    _validate_state_schema(state_dict, model.state_dict())
    if any(not torch.isfinite(tensor).all().item() for tensor in state_dict.values()):
        raise CheckpointError("checkpoint contains nonfinite model weights")
    if parameter_digest(state_dict) != manifest.parameter_digest:
        raise CheckpointError("model parameter digest mismatch")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointError("model state is incompatible with checkpoint config") from error
    if parameter_digest(model.state_dict()) != manifest.parameter_digest:
        raise CheckpointError("loaded model parameter digest mismatch")
    model.eval()
    return LoadedInferenceCheckpoint(model=model, manifest=manifest)


def _prepare_empty_destination(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise CheckpointError("checkpoint destination must be an empty directory")
        return
    try:
        path.mkdir(parents=True)
    except OSError as error:
        raise CheckpointError("checkpoint destination could not be created") from error


def _validate_manifest(
    manifest: InferenceManifest,
    *,
    require_integrity: bool,
) -> None:
    if manifest.schema_version != _CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema")
    if (
        manifest.encoder_schema_version != 1
        or manifest.encoder_config.schema_version != manifest.encoder_schema_version
    ):
        raise CheckpointError("unsupported encoder schema")
    if (
        manifest.supported_ruleset_names != _STAGE1_RULESETS
        or manifest.supported_player_counts != _STAGE1_PLAYER_COUNTS
        or manifest.supported_ruleset_names != manifest.encoder_config.supported_ruleset_names
        or manifest.supported_player_counts != manifest.encoder_config.supported_player_counts
    ):
        raise CheckpointError("checkpoint support is not live-A with three players")
    expected_size, expected_hash = _action_space_contract(manifest.encoder_config)
    if manifest.action_space_size != expected_size or manifest.action_space_hash != expected_hash:
        raise CheckpointError("checkpoint action-space contract is incompatible")
    if not all(
        (
            manifest.repository_commit,
            manifest.python_version,
            manifest.torch_version,
            manifest.numpy_version,
            manifest.sdk_version,
        )
    ):
        raise CheckpointError("manifest version metadata must be nonempty")
    if manifest.completed_episodes < 0 or manifest.completed_updates < 0:
        raise CheckpointError("manifest completion counters must be nonnegative")
    if require_integrity and (
        not _is_sha256(manifest.model_sha256) or not _is_sha256(manifest.parameter_digest)
    ):
        raise CheckpointError("manifest integrity digests are invalid")


def _read_manifest(path: Path) -> InferenceManifest:
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError("manifest JSON could not be read") from error
    payload = _require_object_dict(parsed, "manifest")
    if set(payload) != _MANIFEST_FIELDS:
        raise CheckpointError("manifest fields do not match checkpoint schema")
    try:
        encoder_config = _read_encoder_config(payload["encoder_config"])
        model_config = _read_model_config(payload["model_config"])
        return InferenceManifest(
            schema_version=_require_int(payload["schema_version"], "schema_version"),
            encoder_schema_version=_require_int(
                payload["encoder_schema_version"],
                "encoder_schema_version",
            ),
            repository_commit=_require_string(
                payload["repository_commit"],
                "repository_commit",
            ),
            python_version=_require_string(
                payload["python_version"],
                "python_version",
            ),
            torch_version=_require_string(payload["torch_version"], "torch_version"),
            numpy_version=_require_string(payload["numpy_version"], "numpy_version"),
            sdk_version=_require_string(payload["sdk_version"], "sdk_version"),
            encoder_config=encoder_config,
            model_config=model_config,
            action_space_size=_require_int(
                payload["action_space_size"],
                "action_space_size",
            ),
            action_space_hash=_require_string(
                payload["action_space_hash"],
                "action_space_hash",
            ),
            supported_ruleset_names=_require_string_tuple(
                payload["supported_ruleset_names"],
                "supported_ruleset_names",
            ),
            supported_player_counts=_require_int_tuple(
                payload["supported_player_counts"],
                "supported_player_counts",
            ),
            root_seed=_require_int(payload["root_seed"], "root_seed"),
            completed_episodes=_require_int(
                payload["completed_episodes"],
                "completed_episodes",
            ),
            completed_updates=_require_int(
                payload["completed_updates"],
                "completed_updates",
            ),
            model_sha256=_require_string(payload["model_sha256"], "model_sha256"),
            parameter_digest=_require_string(
                payload["parameter_digest"],
                "parameter_digest",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError("manifest values do not match checkpoint schema") from error


def _read_encoder_config(value: object) -> NeuralEncoderConfig:
    payload = _require_object_dict(value, "encoder_config")
    _require_exact_fields(payload, _ENCODER_CONFIG_FIELDS, "encoder_config")
    return NeuralEncoderConfig(
        schema_version=_require_int(payload.get("schema_version"), "schema_version"),
        supported_ruleset_names=_require_string_tuple(
            payload.get("supported_ruleset_names"),
            "supported_ruleset_names",
        ),
        supported_player_counts=_require_int_tuple(
            payload.get("supported_player_counts"),
            "supported_player_counts",
        ),
        max_bid=_require_int(payload.get("max_bid"), "max_bid"),
        max_hand_size=_require_int(payload.get("max_hand_size"), "max_hand_size"),
        max_history_events=_require_int(
            payload.get("max_history_events"),
            "max_history_events",
        ),
        max_cash=_require_int(payload.get("max_cash"), "max_cash"),
        max_abs_chart=_require_int(payload.get("max_abs_chart"), "max_abs_chart"),
        max_resource_cards=_require_int(
            payload.get("max_resource_cards"),
            "max_resource_cards",
        ),
        max_action_cards=_require_int(
            payload.get("max_action_cards"),
            "max_action_cards",
        ),
    )


def _read_model_config(value: object) -> NeuralModelConfig:
    payload = _require_object_dict(value, "model_config")
    _require_exact_fields(payload, _MODEL_CONFIG_FIELDS, "model_config")
    return NeuralModelConfig(
        categorical_embedding_size=_require_int(
            payload.get("categorical_embedding_size"),
            "categorical_embedding_size",
        ),
        suit_embedding_size=_require_int(
            payload.get("suit_embedding_size"),
            "suit_embedding_size",
        ),
        seat_hidden_size=_require_int(
            payload.get("seat_hidden_size"),
            "seat_hidden_size",
        ),
        event_embedding_size=_require_int(
            payload.get("event_embedding_size"),
            "event_embedding_size",
        ),
        gru_hidden_size=_require_int(
            payload.get("gru_hidden_size"),
            "gru_hidden_size",
        ),
        snapshot_hidden_size=_require_int(
            payload.get("snapshot_hidden_size"),
            "snapshot_hidden_size",
        ),
        trunk_hidden_size=_require_int(
            payload.get("trunk_hidden_size"),
            "trunk_hidden_size",
        ),
    )


def _require_state_dict(value: object) -> dict[str, Tensor]:
    if not isinstance(value, Mapping):
        raise CheckpointError("model file must contain a state dictionary")
    output: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise CheckpointError("model state dictionary contains an invalid entry")
        output[name] = tensor
    return output


def _validate_state_schema(
    state_dict: Mapping[str, Tensor],
    expected_state: Mapping[str, Tensor],
) -> None:
    if set(state_dict) != set(expected_state):
        raise CheckpointError("model parameter names do not match checkpoint config")
    for name, expected in expected_state.items():
        actual = state_dict[name]
        if actual.dtype != expected.dtype:
            raise CheckpointError(f"model parameter {name!r} has an incompatible dtype")
        if actual.shape != expected.shape:
            raise CheckpointError(f"model parameter {name!r} has an incompatible shape")


def _action_space_contract(
    config: NeuralEncoderConfig,
) -> tuple[int, str]:
    bounds = EnvironmentBounds(config.max_bid, config.max_hand_size)
    codec = ActionCodec(bounds)
    descriptor = {
        "schema_version": 1,
        "size": codec.size,
        "pass_index": 0,
        "bid_indices": list(range(1, bounds.max_bid + 1)),
        "reveal_indices": list(
            range(
                bounds.max_bid + 1,
                bounds.max_bid + 1 + bounds.max_hand_size,
            )
        ),
    }
    encoded = json.dumps(
        descriptor,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return codec.size, hashlib.sha256(encoded).hexdigest()


def _update_digest_part(digest: object, value: bytes) -> None:
    if not hasattr(digest, "update"):
        raise TypeError("digest must support update")
    length = len(value).to_bytes(8, byteorder="big", signed=False)
    digest.update(length)
    digest.update(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise CheckpointError("checkpoint file could not be read") from error
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_object_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CheckpointError(f"{name} must be a JSON object")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise CheckpointError(f"{name} fields do not match checkpoint schema")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CheckpointError(f"{name} must be a string")
    return value


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CheckpointError(f"{name} must be an integer")
    return value


def _require_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CheckpointError(f"{name} must be a string array")
    return tuple(value)


def _require_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise CheckpointError(f"{name} must be an integer array")
    return tuple(value)
