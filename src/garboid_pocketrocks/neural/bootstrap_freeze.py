"""Development-only freezing for completed heuristic-bootstrap experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from garboid_pocketrocks.neural.behavior_cloning import (
    BALANCED_V3_PROFILE_DIGEST,
    BALANCED_V3_TEACHER_IDENTITY,
    BEHAVIOR_CLONING_OPTIMIZATION_ORDER,
)
from garboid_pocketrocks.neural.checkpoint import (
    LoadedInferenceCheckpoint,
    load_inference_checkpoint,
)
from garboid_pocketrocks.neural.config import ModelProfile, training_model_config
from garboid_pocketrocks.neural.heuristic_bootstrap import (
    EXPERIMENT_GAMES_PER_CELL,
    HEURISTIC_BOOTSTRAP_ARMS,
    REFERENCE_TRAINING_GAMES,
    TRAINING_CELL_COUNT,
    HeuristicBootstrapStrategy,
    bootstrap_strategy,
    validate_fixed_compute_arm,
)
from garboid_pocketrocks.neural.identity import experimental_neural_bot_id
from garboid_pocketrocks.neural.training_checkpoint import (
    export_inference_checkpoint,
    load_training_checkpoint,
)

_SUMMARY_NAME = "bootstrap-summary.json"
_FILES = {"manifest.json", "inference", _SUMMARY_NAME}
_MANIFEST_FIELDS = {
    "schema_version",
    "identity",
    "strategy",
    "source_commit",
    "training_config_digest",
    "arm_digest",
    "development_corpus_name",
    "development_corpus_digest",
    "summary_digest",
    "metrics_digest",
    "parameter_digest",
    "inference_manifest_digest",
    "inference_model_digest",
    "model_profile",
    "root_seed",
    "demonstration_games",
    "ppo_games",
    "total_training_games",
    "ppo_updates",
    "games_per_cell",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")


class _UnspecifiedBehaviorCloningDigest:
    pass


_UNSPECIFIED_BEHAVIOR_CLONING_DIGEST = _UnspecifiedBehaviorCloningDigest()


class BootstrapFreezeError(ValueError):
    """Raised when an experimental checkpoint cannot be frozen safely."""


@dataclass(frozen=True, slots=True)
class BootstrapFreezeManifest:
    """Exact provenance and fixed-compute contract for a frozen candidate."""

    schema_version: int
    identity: str
    strategy: HeuristicBootstrapStrategy
    source_commit: str
    training_config_digest: str
    arm_digest: str
    development_corpus_name: str
    development_corpus_digest: str
    summary_digest: str
    metrics_digest: str
    parameter_digest: str
    inference_manifest_digest: str
    inference_model_digest: str
    model_profile: ModelProfile
    root_seed: int
    demonstration_games: int
    ppo_games: int
    total_training_games: int
    ppo_updates: int
    games_per_cell: int

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BootstrapFreezeError("unsupported bootstrap freeze schema")
        if not self.identity:
            raise BootstrapFreezeError("experimental identity must be nonempty")
        if self.strategy not in {arm.strategy for arm in HEURISTIC_BOOTSTRAP_ARMS}:
            raise BootstrapFreezeError("bootstrap strategy is unknown")
        if _SOURCE_COMMIT.fullmatch(self.source_commit) is None:
            raise BootstrapFreezeError("source commit must be a lowercase Git digest")
        for name in (
            "training_config_digest",
            "arm_digest",
            "development_corpus_digest",
            "summary_digest",
            "metrics_digest",
            "parameter_digest",
            "inference_manifest_digest",
            "inference_model_digest",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise BootstrapFreezeError(f"{name} must be a lowercase SHA-256 digest")
        if (
            not self.development_corpus_name.startswith("development-")
            or "held-out" in self.development_corpus_name
            or "held_out" in self.development_corpus_name
        ):
            raise BootstrapFreezeError("freeze requires an explicitly development corpus")
        if self.model_profile not in ("small", "medium", "large"):
            raise BootstrapFreezeError("model profile is unknown")
        for name in (
            "root_seed",
            "demonstration_games",
            "ppo_games",
            "total_training_games",
            "ppo_updates",
            "games_per_cell",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BootstrapFreezeError(f"{name} must be a nonnegative integer")
        if self.total_training_games != self.demonstration_games + self.ppo_games:
            raise BootstrapFreezeError("frozen training game counts are inconsistent")

    def to_json_dict(self) -> dict[str, object]:
        """Return the complete exact-key JSON representation."""

        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "strategy": self.strategy,
            "source_commit": self.source_commit,
            "training_config_digest": self.training_config_digest,
            "arm_digest": self.arm_digest,
            "development_corpus_name": self.development_corpus_name,
            "development_corpus_digest": self.development_corpus_digest,
            "summary_digest": self.summary_digest,
            "metrics_digest": self.metrics_digest,
            "parameter_digest": self.parameter_digest,
            "inference_manifest_digest": self.inference_manifest_digest,
            "inference_model_digest": self.inference_model_digest,
            "model_profile": self.model_profile,
            "root_seed": self.root_seed,
            "demonstration_games": self.demonstration_games,
            "ppo_games": self.ppo_games,
            "total_training_games": self.total_training_games,
            "ppo_updates": self.ppo_updates,
            "games_per_cell": self.games_per_cell,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> BootstrapFreezeManifest:
        """Read a manifest while rejecting missing, extra, or mistyped fields."""

        payload = _object(value, "bootstrap freeze manifest")
        if set(payload) != _MANIFEST_FIELDS:
            raise BootstrapFreezeError("bootstrap freeze manifest fields are not exact")
        return cls(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            identity=_string(payload["identity"], "identity"),
            strategy=cast(
                HeuristicBootstrapStrategy,
                _string(payload["strategy"], "strategy"),
            ),
            source_commit=_string(payload["source_commit"], "source_commit"),
            training_config_digest=_string(
                payload["training_config_digest"],
                "training_config_digest",
            ),
            arm_digest=_string(payload["arm_digest"], "arm_digest"),
            development_corpus_name=_string(
                payload["development_corpus_name"],
                "development_corpus_name",
            ),
            development_corpus_digest=_string(
                payload["development_corpus_digest"],
                "development_corpus_digest",
            ),
            summary_digest=_string(payload["summary_digest"], "summary_digest"),
            metrics_digest=_string(payload["metrics_digest"], "metrics_digest"),
            parameter_digest=_string(
                payload["parameter_digest"],
                "parameter_digest",
            ),
            inference_manifest_digest=_string(
                payload["inference_manifest_digest"],
                "inference_manifest_digest",
            ),
            inference_model_digest=_string(
                payload["inference_model_digest"],
                "inference_model_digest",
            ),
            model_profile=cast(
                ModelProfile,
                _string(payload["model_profile"], "model_profile"),
            ),
            root_seed=_integer(payload["root_seed"], "root_seed"),
            demonstration_games=_integer(
                payload["demonstration_games"],
                "demonstration_games",
            ),
            ppo_games=_integer(payload["ppo_games"], "ppo_games"),
            total_training_games=_integer(
                payload["total_training_games"],
                "total_training_games",
            ),
            ppo_updates=_integer(payload["ppo_updates"], "ppo_updates"),
            games_per_cell=_integer(
                payload["games_per_cell"],
                "games_per_cell",
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenBootstrapCandidate:
    """Verified manifest and ordinary inference checkpoint."""

    manifest: BootstrapFreezeManifest
    inference: LoadedInferenceCheckpoint


@dataclass(frozen=True, slots=True)
class _ValidatedBootstrapSummary:
    """Exact validated report bytes and the selected arm's metrics binding."""

    content: bytes
    summary_digest: str
    metrics_digest: str


def freeze_bootstrap_candidate(
    training_checkpoint: Path,
    destination: Path,
    *,
    development_corpus_name: str,
    development_corpus_digest: str,
    bootstrap_summary_path: Path,
) -> FrozenBootstrapCandidate:
    """Atomically freeze one final fixed-compute development candidate."""

    _require_empty_destination(destination)
    loaded = load_training_checkpoint(training_checkpoint, device=torch.device("cpu"))
    config = loaded.manifest.run_config
    arm = validate_fixed_compute_arm(config)
    _validate_final_progress(loaded.manifest.progress, arm.strategy, arm.ppo_updates)
    if config.model_profile != "large" or loaded.model.model_config != training_model_config(
        "large"
    ):
        raise BootstrapFreezeError("issue 14 freeze requires the fixed large model profile")
    source_commit = loaded.manifest.repository_commit
    if _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise BootstrapFreezeError("training checkpoint has no exact source commit")
    config_digest = _json_digest(config.to_json_dict())
    strategy = bootstrap_strategy(config)
    summary = _read_and_validate_bootstrap_summary(
        bootstrap_summary_path,
        expected_strategy=strategy,
        expected_config_digest=config_digest,
        expected_behavior_cloning_config_digest=(
            None if config.behavior_cloning is None else config.behavior_cloning.config_digest
        ),
    )
    identity = experimental_neural_bot_id(
        config.model_profile,
        strategy=strategy,
        root_seed=config.root_seed,
        completed_games=arm.total_training_games,
        config_digest=config_digest,
        parameter_digest=loaded.manifest.parameter_digest,
        repository_commit=source_commit,
    )
    parent = destination.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / _SUMMARY_NAME).write_bytes(summary.content)
        inference_path = temporary / "inference"
        export_inference_checkpoint(
            training_checkpoint,
            inference_path,
            device=torch.device("cpu"),
        )
        inference = load_inference_checkpoint(inference_path, device=torch.device("cpu"))
        if inference.manifest.parameter_digest != loaded.manifest.parameter_digest:
            raise BootstrapFreezeError("export changed the training checkpoint parameters")
        if (
            inference.manifest.repository_commit != source_commit
            or inference.manifest.root_seed != config.root_seed
            or inference.manifest.completed_episodes != arm.ppo_games
            or inference.manifest.completed_updates != arm.ppo_updates
        ):
            raise BootstrapFreezeError("exported inference provenance does not match training")

        manifest = BootstrapFreezeManifest(
            schema_version=1,
            identity=identity,
            strategy=strategy,
            source_commit=source_commit,
            training_config_digest=config_digest,
            arm_digest=arm.digest,
            development_corpus_name=development_corpus_name,
            development_corpus_digest=development_corpus_digest,
            summary_digest=summary.summary_digest,
            metrics_digest=summary.metrics_digest,
            parameter_digest=loaded.manifest.parameter_digest,
            inference_manifest_digest=_file_digest(inference_path / "manifest.json"),
            inference_model_digest=_file_digest(inference_path / "model.pt"),
            model_profile=config.model_profile,
            root_seed=config.root_seed,
            demonstration_games=arm.demonstration_games,
            ppo_games=arm.ppo_games,
            total_training_games=arm.total_training_games,
            ppo_updates=arm.ppo_updates,
            games_per_cell=EXPERIMENT_GAMES_PER_CELL,
        )
        _write_json(temporary / "manifest.json", manifest.to_json_dict())
        _sync_directory(inference_path)
        _sync_directory(temporary)
        _install_empty_destination(temporary, destination)
        return load_frozen_bootstrap_candidate(destination)
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(error, BootstrapFreezeError):
            raise
        raise BootstrapFreezeError(f"bootstrap freeze failed: {error}") from error


def load_frozen_bootstrap_candidate(path: Path) -> FrozenBootstrapCandidate:
    """Reload and verify every frozen identity, provenance, and payload binding."""

    if not path.is_dir() or {item.name for item in path.iterdir()} != _FILES:
        raise BootstrapFreezeError("frozen candidate files are not exact")
    try:
        value: object = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapFreezeError("freeze manifest could not be read") from error
    manifest = BootstrapFreezeManifest.from_json_dict(value)
    summary = _read_and_validate_bootstrap_summary(
        path / _SUMMARY_NAME,
        expected_strategy=manifest.strategy,
        expected_config_digest=manifest.training_config_digest,
    )
    if summary.summary_digest != manifest.summary_digest:
        raise BootstrapFreezeError("bootstrap summary digest does not match freeze manifest")
    if summary.metrics_digest != manifest.metrics_digest:
        raise BootstrapFreezeError("bootstrap metrics digest does not match freeze manifest")
    inference_path = path / "inference"
    if _file_digest(inference_path / "manifest.json") != manifest.inference_manifest_digest:
        raise BootstrapFreezeError("inference manifest digest does not match freeze manifest")
    if _file_digest(inference_path / "model.pt") != manifest.inference_model_digest:
        raise BootstrapFreezeError("inference model digest does not match freeze manifest")
    try:
        inference = load_inference_checkpoint(inference_path, device=torch.device("cpu"))
    except ValueError as error:
        raise BootstrapFreezeError("frozen inference checkpoint is invalid") from error
    if (
        inference.manifest.parameter_digest != manifest.parameter_digest
        or inference.manifest.repository_commit != manifest.source_commit
        or inference.manifest.root_seed != manifest.root_seed
        or inference.manifest.completed_episodes != manifest.ppo_games
        or inference.manifest.completed_updates != manifest.ppo_updates
    ):
        raise BootstrapFreezeError("inference checkpoint does not match freeze provenance")
    if manifest.model_profile != "large" or inference.model.model_config != training_model_config(
        "large"
    ):
        raise BootstrapFreezeError("frozen candidate does not use the fixed large model profile")
    expected_identity = experimental_neural_bot_id(
        manifest.model_profile,
        strategy=manifest.strategy,
        root_seed=manifest.root_seed,
        completed_games=manifest.total_training_games,
        config_digest=manifest.training_config_digest,
        parameter_digest=manifest.parameter_digest,
        repository_commit=manifest.source_commit,
    )
    if manifest.identity != expected_identity:
        raise BootstrapFreezeError("experimental identity does not match freeze provenance")
    arm = next(
        (item for item in HEURISTIC_BOOTSTRAP_ARMS if item.strategy == manifest.strategy),
        None,
    )
    if arm is None or (
        manifest.arm_digest != arm.digest
        or manifest.demonstration_games != arm.demonstration_games
        or manifest.ppo_games != arm.ppo_games
        or manifest.total_training_games != REFERENCE_TRAINING_GAMES
        or manifest.ppo_updates != arm.ppo_updates
        or manifest.games_per_cell != EXPERIMENT_GAMES_PER_CELL
    ):
        raise BootstrapFreezeError("freeze manifest does not match fixed-compute arm")
    return FrozenBootstrapCandidate(manifest, inference)


def _read_and_validate_bootstrap_summary(
    path: Path,
    *,
    expected_strategy: HeuristicBootstrapStrategy,
    expected_config_digest: str,
    expected_behavior_cloning_config_digest: (
        str | None | _UnspecifiedBehaviorCloningDigest
    ) = _UNSPECIFIED_BEHAVIOR_CLONING_DIGEST,
) -> _ValidatedBootstrapSummary:
    try:
        content = path.read_bytes()
        value: object = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, BootstrapFreezeError) as error:
        raise BootstrapFreezeError("bootstrap summary could not be read") from error
    _reject_held_out_content(value)
    _reject_private_training_content(value)
    payload = _object(value, "bootstrap summary")
    expected_fields = {
        "schema_version",
        "report_kind",
        "held_out_loaded",
        "learning_curves_digest",
        "official_arm_contract",
        "reported_arm_count",
        "all_official_arms_present",
        "arms",
    }
    if set(payload) != expected_fields:
        raise BootstrapFreezeError("bootstrap summary fields are not exact")
    if (
        _integer(payload["schema_version"], "summary schema_version") != 1
        or _string(payload["report_kind"], "report_kind") != "heuristic_bootstrap_training"
        or payload["held_out_loaded"] is not False
    ):
        raise BootstrapFreezeError("bootstrap summary is not development-only schema 1")
    _digest(payload["learning_curves_digest"], "learning_curves_digest")

    expected_arms = {arm.strategy: arm for arm in HEURISTIC_BOOTSTRAP_ARMS}
    contract = _object(payload["official_arm_contract"], "official arm contract")
    if set(contract) != {"arm_count", "arms"} or _integer(
        contract.get("arm_count"), "official arm count"
    ) != len(expected_arms):
        raise BootstrapFreezeError("official arm contract is not exact")
    contract_rows = _array(contract["arms"], "official arm contract rows")
    parsed_contract: dict[str, str] = {}
    for value_row in contract_rows:
        row = _object(value_row, "official arm contract row")
        if set(row) != {"arm", "arm_digest"}:
            raise BootstrapFreezeError("official arm contract row fields are not exact")
        name = _string(row["arm"], "official arm name")
        digest = _digest(row["arm_digest"], "official arm digest")
        if name in parsed_contract:
            raise BootstrapFreezeError("official arm contract contains duplicates")
        parsed_contract[name] = digest
    if parsed_contract != {name: arm.digest for name, arm in expected_arms.items()}:
        raise BootstrapFreezeError("official arm names or digests do not match the fixed contract")

    if (
        _integer(payload["reported_arm_count"], "reported arm count") != len(expected_arms)
        or payload["all_official_arms_present"] is not True
    ):
        raise BootstrapFreezeError("bootstrap summary does not declare all official arms")
    arm_rows = _array(payload["arms"], "bootstrap arm rows")
    parsed_arms: dict[str, dict[str, object]] = {}
    for value_row in arm_rows:
        row = _validate_summary_arm(value_row, set(expected_arms))
        name = _string(row["arm"], "bootstrap arm name")
        if name in parsed_arms:
            raise BootstrapFreezeError("bootstrap summary contains duplicate arms")
        parsed_arms[name] = row
    if set(parsed_arms) != set(expected_arms):
        raise BootstrapFreezeError("bootstrap summary must contain all five official arms")

    selected = parsed_arms[expected_strategy]
    if _digest(selected["config_digest"], "selected config digest") != expected_config_digest:
        raise BootstrapFreezeError("bootstrap summary config does not match training checkpoint")
    if not isinstance(
        expected_behavior_cloning_config_digest,
        _UnspecifiedBehaviorCloningDigest,
    ):
        selected_cloning = selected["behavior_cloning"]
        if expected_behavior_cloning_config_digest is None:
            if selected_cloning is not None:
                raise BootstrapFreezeError(
                    "bootstrap cloning provenance contradicts training checkpoint"
                )
        else:
            cloning = _object(selected_cloning, "selected cloning provenance")
            if (
                _digest(cloning["config_digest"], "selected cloning config digest")
                != expected_behavior_cloning_config_digest
            ):
                raise BootstrapFreezeError(
                    "bootstrap cloning config does not match training checkpoint"
                )
    metrics_digest = _digest(selected["metrics_digest"], "selected metrics digest")
    return _ValidatedBootstrapSummary(
        content=content,
        summary_digest=hashlib.sha256(content).hexdigest(),
        metrics_digest=metrics_digest,
    )


def _validate_summary_arm(
    value: object,
    expected_arm_names: set[HeuristicBootstrapStrategy],
) -> dict[str, object]:
    row = _object(value, "bootstrap arm")
    expected_fields = {
        "arm",
        "arm_digest",
        "config_digest",
        "metrics_digest",
        "experiment_root_seed",
        "model_profile",
        "complete",
        "configuration",
        "compute",
        "final_learning_metrics",
        "behavior_cloning",
    }
    if set(row) != expected_fields:
        raise BootstrapFreezeError("bootstrap arm fields are not exact")
    name = _string(row["arm"], "bootstrap arm name")
    if name not in expected_arm_names:
        raise BootstrapFreezeError("bootstrap summary contains an unknown arm")
    arm = next(item for item in HEURISTIC_BOOTSTRAP_ARMS if item.strategy == name)
    if _digest(row["arm_digest"], "bootstrap arm digest") != arm.digest:
        raise BootstrapFreezeError("bootstrap arm digest does not match fixed contract")
    _digest(row["config_digest"], "bootstrap config digest")
    _digest(row["metrics_digest"], "bootstrap metrics digest")
    if (
        _integer(row["experiment_root_seed"], "experiment root seed") != 42
        or _string(row["model_profile"], "bootstrap model profile") != "large"
        or row["complete"] is not True
    ):
        raise BootstrapFreezeError("bootstrap arm is not a complete official run")

    configuration = _object(row["configuration"], "bootstrap arm configuration")
    if set(configuration) != {
        "games_per_cell",
        "max_updates",
        "ppo",
        "reward",
        "heuristic_auxiliary",
        "opponent_training",
    }:
        raise BootstrapFreezeError("bootstrap arm configuration fields are not exact")
    if (
        _integer(configuration["games_per_cell"], "configured games per cell")
        != EXPERIMENT_GAMES_PER_CELL
        or _integer(configuration["max_updates"], "configured updates") != arm.ppo_updates
    ):
        raise BootstrapFreezeError("bootstrap arm configuration has drifted fixed compute")
    for field in ("ppo", "reward", "heuristic_auxiliary"):
        _object(configuration[field], f"bootstrap {field}")
    _string(configuration["opponent_training"], "opponent training")

    compute = _object(row["compute"], "bootstrap arm compute")
    if set(compute) != {
        "completed_updates",
        "demonstration_games",
        "ppo_games",
        "total_training_games",
        "decisions",
        "optimizer_steps",
        "duration_seconds",
    }:
        raise BootstrapFreezeError("bootstrap arm compute fields are not exact")
    expected_counts = {
        "completed_updates": arm.ppo_updates,
        "demonstration_games": arm.demonstration_games,
        "ppo_games": arm.ppo_games,
        "total_training_games": arm.total_training_games,
    }
    if any(
        _integer(compute[field], field) != expected for field, expected in expected_counts.items()
    ):
        raise BootstrapFreezeError("bootstrap arm compute is incomplete or inconsistent")
    if _integer(compute["decisions"], "training decisions") <= 0:
        raise BootstrapFreezeError("bootstrap arm aggregate compute must be positive")
    steps = _object(compute["optimizer_steps"], "optimizer step breakdown")
    durations = _object(compute["duration_seconds"], "training duration breakdown")
    if set(steps) != {"behavior_cloning", "ppo", "total"} or set(durations) != {
        "behavior_cloning",
        "ppo",
        "total",
    }:
        raise BootstrapFreezeError("bootstrap compute breakdown fields are not exact")
    cloning_steps = _integer(steps["behavior_cloning"], "cloning optimizer steps")
    ppo_steps = _integer(steps["ppo"], "PPO optimizer steps")
    total_steps = _integer(steps["total"], "total optimizer steps")
    cloning_duration = _number(durations["behavior_cloning"], "cloning duration")
    ppo_duration = _number(durations["ppo"], "PPO duration")
    total_duration = _number(durations["total"], "total duration")
    expects_cloning = arm.demonstration_games > 0
    if (
        cloning_steps < 0
        or cloning_duration < 0.0
        or total_steps < 0
        or total_duration < 0.0
        or ppo_steps <= 0
        or ppo_duration <= 0.0
        or total_steps != cloning_steps + ppo_steps
        or not math.isclose(
            total_duration,
            cloning_duration + ppo_duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or (cloning_steps > 0) is not expects_cloning
        or (cloning_duration > 0.0) is not expects_cloning
    ):
        raise BootstrapFreezeError("bootstrap arm compute breakdown is inconsistent")
    _validate_behavior_cloning_provenance(
        row["behavior_cloning"],
        demonstration_games=arm.demonstration_games,
        optimizer_steps=cloning_steps,
        duration_seconds=cloning_duration,
    )
    _object(row["final_learning_metrics"], "final learning metrics")
    return row


def _validate_behavior_cloning_provenance(
    value: object,
    *,
    demonstration_games: int,
    optimizer_steps: int,
    duration_seconds: float,
) -> None:
    if demonstration_games == 0:
        if value is not None:
            raise BootstrapFreezeError("non-cloning arm contains cloning provenance")
        return
    provenance = _object(value, "behavior cloning public provenance")
    expected_fields = {
        "schema_version",
        "method",
        "config_digest",
        "teacher_identity",
        "teacher_profile_digest",
        "shard_count",
        "demonstration_games",
        "demonstration_examples",
        "cell_game_counts",
        "epochs",
        "optimization_order",
        "optimizer_steps",
        "elapsed_seconds",
        "provenance_digest",
    }
    if set(provenance) != expected_fields:
        raise BootstrapFreezeError("behavior cloning public provenance fields are not exact")
    _digest(provenance["config_digest"], "cloning config digest")
    if (
        _integer(provenance["schema_version"], "cloning provenance schema") != 1
        or _string(provenance["method"], "cloning method") != "behavior_cloning"
        or _string(provenance["teacher_identity"], "cloning teacher")
        != BALANCED_V3_TEACHER_IDENTITY
        or _digest(provenance["teacher_profile_digest"], "cloning teacher digest")
        != BALANCED_V3_PROFILE_DIGEST
        or _integer(provenance["shard_count"], "cloning shard count") <= 0
        or _integer(provenance["demonstration_games"], "cloning games") != demonstration_games
        or _integer(provenance["demonstration_examples"], "cloning examples") < demonstration_games
        or _integer(provenance["epochs"], "cloning epochs") <= 0
        or _string(provenance["optimization_order"], "cloning optimization order")
        != BEHAVIOR_CLONING_OPTIMIZATION_ORDER
        or _integer(provenance["optimizer_steps"], "cloning optimizer steps") != optimizer_steps
        or not math.isclose(
            _number(provenance["elapsed_seconds"], "cloning elapsed seconds"),
            duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise BootstrapFreezeError("behavior cloning public provenance is inconsistent")
    cells = _array(provenance["cell_game_counts"], "cloning cell counts")
    parsed_cells: set[tuple[str, int]] = set()
    cell_games = 0
    for value_row in cells:
        row = _array(value_row, "cloning cell row")
        if len(row) != 3:
            raise BootstrapFreezeError("behavior cloning cell row is malformed")
        ruleset = _string(row[0], "cloning cell ruleset")
        players = _integer(row[1], "cloning cell players")
        games = _integer(row[2], "cloning cell games")
        if (
            ruleset not in {f"live-{chart}" for chart in "ABCDE"}
            or players not in (3, 4, 5)
            or games <= 0
        ):
            raise BootstrapFreezeError("behavior cloning cell is outside A-E/3-5")
        parsed_cells.add((ruleset, players))
        cell_games += games
    if len(cells) != 15 or len(parsed_cells) != 15 or cell_games != demonstration_games:
        raise BootstrapFreezeError("behavior cloning cells do not reconcile")
    expected_digest = _json_digest(
        {key: nested for key, nested in provenance.items() if key != "provenance_digest"}
    )
    if _digest(provenance["provenance_digest"], "cloning provenance digest") != expected_digest:
        raise BootstrapFreezeError("behavior cloning provenance digest is invalid")


def _validate_final_progress(
    progress: object,
    strategy: HeuristicBootstrapStrategy,
    expected_updates: int,
) -> None:
    from garboid_pocketrocks.neural.training_checkpoint import TrainingProgress

    if not isinstance(progress, TrainingProgress):
        raise BootstrapFreezeError("training checkpoint progress has the wrong type")
    arm = next(item for item in HEURISTIC_BOOTSTRAP_ARMS if item.strategy == strategy)
    expected_cell_games = expected_updates * EXPERIMENT_GAMES_PER_CELL
    expected_cells = {
        (f"live-{chart}", player_count): expected_cell_games
        for chart in "ABCDE"
        for player_count in (3, 4, 5)
    }
    actual_cells = {(ruleset, players): games for ruleset, players, games in progress.cell_games}
    if (
        progress.next_update_index != expected_updates
        or progress.completed_episodes != arm.ppo_games
        or len(actual_cells) != len(progress.cell_games)
        or actual_cells != expected_cells
        or len(actual_cells) != TRAINING_CELL_COUNT
    ):
        raise BootstrapFreezeError("training checkpoint is not the final fixed-compute arm")


def _require_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise BootstrapFreezeError("freeze destination must be empty")


def _install_empty_destination(temporary: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise BootstrapFreezeError("freeze destination changed before installation")
        destination.rmdir()
    os.replace(temporary, destination)
    _sync_directory(destination.parent)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BootstrapFreezeError(f"could not hash {path.name}") from error
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BootstrapFreezeError(f"{name} must be a JSON object")
    return dict(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise BootstrapFreezeError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BootstrapFreezeError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise BootstrapFreezeError(f"{name} must be a finite number")
    return float(value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise BootstrapFreezeError(f"{name} must be a JSON array")
    return list(value)


def _digest(value: object, name: str) -> str:
    digest = _string(value, name)
    if _SHA256.fullmatch(digest) is None:
        raise BootstrapFreezeError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _reject_json_constant(value: str) -> object:
    raise BootstrapFreezeError(f"non-finite JSON constant {value} is forbidden")


def _reject_held_out_content(value: object) -> None:
    """Reject any held-out reference beyond the required false declaration."""

    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if ("held_out" in normalized or "heldout" in normalized) and key != "held_out_loaded":
                raise BootstrapFreezeError("bootstrap summary contains held-out content")
            _reject_held_out_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_held_out_content(nested)
    elif isinstance(value, str):
        normalized = value.lower().replace("-", "_")
        if "held_out" in normalized or "heldout" in normalized:
            raise BootstrapFreezeError("bootstrap summary contains held-out content")


def _reject_private_training_content(value: object) -> None:
    """Keep seed-linkable cloning and game details out of the public freeze."""

    forbidden = {
        "aggregate_dataset_digest",
        "dataset_digest",
        "engine_seed",
        "opponent_seed",
        "policy_seed",
        "shard_digests",
        "shard_index",
        "shards",
        "updates",
    }
    if isinstance(value, dict):
        if any(key in forbidden for key in value):
            raise BootstrapFreezeError("bootstrap summary contains private training provenance")
        for nested in value.values():
            _reject_private_training_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_training_content(nested)
