from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.behavior_cloning import (  # noqa: E402
    BALANCED_V3_PROFILE_DIGEST,
    BALANCED_V3_TEACHER_IDENTITY,
    BEHAVIOR_CLONING_OPTIMIZATION_ORDER,
)
from garboid_pocketrocks.neural.bootstrap_freeze import (  # noqa: E402
    BootstrapFreezeError,
    FrozenBootstrapCandidate,
    _read_and_validate_bootstrap_summary,
    freeze_bootstrap_candidate,
    load_frozen_bootstrap_candidate,
)
from garboid_pocketrocks.neural.checkpoint import parameter_digest  # noqa: E402
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.heuristic_bootstrap import (  # noqa: E402
    EXPERIMENT_GAMES_PER_CELL,
    HEURISTIC_BOOTSTRAP_ARMS,
    REFERENCE_TRAINING_GAMES,
    REFERENCE_UPDATES,
)
from garboid_pocketrocks.neural.identity import experimental_neural_bot_id  # noqa: E402
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.run_config import TrainingRunConfig  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    FrozenBootstrapCandidateBrain,
    frozen_bootstrap_bot_spec,
)
from garboid_pocketrocks.neural.training_checkpoint import (  # noqa: E402
    TrainingCheckpointManifest,
    TrainingProgress,
    save_training_checkpoint,
)

_SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_DEVELOPMENT_DIGEST = "d" * 64
_METRICS_DIGEST = "e" * 64


def _config() -> TrainingRunConfig:
    return TrainingRunConfig.from_json(Path("configs/neural/heuristic-bootstrap-control-v1.json"))


def _checkpoint(
    path: Path,
    *,
    updates: int = REFERENCE_UPDATES,
    source_commit: str = _SOURCE_COMMIT,
) -> Path:
    config = _config()
    torch.manual_seed(701)
    model = NeuralPolicy(training_encoder_config(), training_model_config("large"))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate, foreach=False)
    games_in_cell = updates * EXPERIMENT_GAMES_PER_CELL
    completed_games = games_in_cell * 15
    config_digest = hashlib.sha256(
        json.dumps(
            config.to_json_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = TrainingCheckpointManifest(
        schema_version=1,
        repository_commit=source_commit,
        encoder_config=model.encoder_config,
        model_config=model.model_config,
        run_config=config,
        progress=TrainingProgress(
            next_update_index=updates,
            completed_episodes=completed_games,
            completed_decisions=completed_games * 20,
            cell_games=tuple(
                (f"live-{chart}", players, games_in_cell)
                for chart in "ABCDE"
                for players in (3, 4, 5)
            ),
        ),
        lineage=(),
        champion_identity=experimental_neural_bot_id(
            "large",
            strategy="fixed-compute-control-v1",
            root_seed=config.root_seed,
            completed_games=completed_games,
            config_digest=config_digest,
            parameter_digest=parameter_digest(model.state_dict()),
            repository_commit=_SOURCE_COMMIT,
        ),
        league_identities=(),
    )
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        generator_states={"torch": torch.get_rng_state()},
        metrics={"update_index": updates - 1},
    )


def _summary_payload() -> dict[str, object]:
    config_digest = hashlib.sha256(
        json.dumps(
            _config().to_json_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    arms = []
    for arm in HEURISTIC_BOOTSTRAP_ARMS:
        cloning: dict[str, object] | None = None
        if arm.demonstration_games:
            cloning = {
                "schema_version": 1,
                "method": "behavior_cloning",
                "config_digest": "b" * 64,
                "teacher_identity": BALANCED_V3_TEACHER_IDENTITY,
                "teacher_profile_digest": BALANCED_V3_PROFILE_DIGEST,
                "shard_count": 112,
                "demonstration_games": arm.demonstration_games,
                "demonstration_examples": arm.demonstration_games * 20,
                "cell_game_counts": [
                    [f"live-{chart}", players, arm.demonstration_games // 15]
                    for chart in "ABCDE"
                    for players in (3, 4, 5)
                ],
                "epochs": 3,
                "optimization_order": BEHAVIOR_CLONING_OPTIMIZATION_ORDER,
                "optimizer_steps": 2,
                "elapsed_seconds": 2.0,
            }
            cloning["provenance_digest"] = hashlib.sha256(
                json.dumps(
                    cloning,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        arms.append(
            {
                "arm": arm.strategy,
                "arm_digest": arm.digest,
                "config_digest": (
                    config_digest
                    if arm.strategy == "fixed-compute-control-v1"
                    else hashlib.sha256(arm.strategy.encode()).hexdigest()
                ),
                "metrics_digest": (
                    _METRICS_DIGEST
                    if arm.strategy == "fixed-compute-control-v1"
                    else hashlib.sha256(f"metrics:{arm.strategy}".encode()).hexdigest()
                ),
                "experiment_root_seed": 42,
                "model_profile": "large",
                "complete": True,
                "configuration": {
                    "games_per_cell": EXPERIMENT_GAMES_PER_CELL,
                    "max_updates": arm.ppo_updates,
                    "ppo": {},
                    "reward": {},
                    "heuristic_auxiliary": {},
                    "opponent_training": arm.opponent_training,
                },
                "compute": {
                    "completed_updates": arm.ppo_updates,
                    "demonstration_games": arm.demonstration_games,
                    "ppo_games": arm.ppo_games,
                    "total_training_games": arm.total_training_games,
                    "decisions": 1,
                    "optimizer_steps": {
                        "behavior_cloning": 2 if arm.demonstration_games else 0,
                        "ppo": 1,
                        "total": 3 if arm.demonstration_games else 1,
                    },
                    "duration_seconds": {
                        "behavior_cloning": 2.0 if arm.demonstration_games else 0.0,
                        "ppo": 1.0,
                        "total": 3.0 if arm.demonstration_games else 1.0,
                    },
                },
                "final_learning_metrics": {},
                "behavior_cloning": cloning,
            }
        )
    return {
        "schema_version": 1,
        "report_kind": "heuristic_bootstrap_training",
        "held_out_loaded": False,
        "learning_curves_digest": "c" * 64,
        "official_arm_contract": {
            "arm_count": len(HEURISTIC_BOOTSTRAP_ARMS),
            "arms": [
                {"arm": arm.strategy, "arm_digest": arm.digest} for arm in HEURISTIC_BOOTSTRAP_ARMS
            ],
        },
        "reported_arm_count": len(HEURISTIC_BOOTSTRAP_ARMS),
        "all_official_arms_present": True,
        "arms": arms,
    }


def _write_summary(path: Path, payload: dict[str, object] | None = None) -> Path:
    path.write_text(
        json.dumps(payload or _summary_payload(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _freeze(
    tmp_path: Path,
    *,
    updates: int = REFERENCE_UPDATES,
) -> FrozenBootstrapCandidate:
    training = _checkpoint(tmp_path / "training", updates=updates)
    summary = _write_summary(tmp_path / "bootstrap-summary.json")
    destination = tmp_path / "frozen"
    destination.mkdir()
    return freeze_bootstrap_candidate(
        training,
        destination,
        development_corpus_name="development-v1",
        development_corpus_digest=_DEVELOPMENT_DIGEST,
        bootstrap_summary_path=summary,
    )


def test_freeze_exports_exact_identity_and_binds_all_development_provenance(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path)
    manifest = frozen.manifest

    assert manifest.identity == experimental_neural_bot_id(
        "large",
        strategy="fixed-compute-control-v1",
        root_seed=42,
        completed_games=REFERENCE_TRAINING_GAMES,
        config_digest=manifest.training_config_digest,
        parameter_digest=manifest.parameter_digest,
        repository_commit=manifest.source_commit,
    )
    assert manifest.source_commit == _SOURCE_COMMIT
    assert manifest.development_corpus_name == "development-v1"
    assert manifest.development_corpus_digest == _DEVELOPMENT_DIGEST
    assert manifest.metrics_digest == _METRICS_DIGEST
    assert (
        manifest.summary_digest
        == hashlib.sha256((tmp_path / "bootstrap-summary.json").read_bytes()).hexdigest()
    )
    assert manifest.parameter_digest == frozen.inference.manifest.parameter_digest
    assert manifest.total_training_games == REFERENCE_TRAINING_GAMES
    assert manifest.ppo_updates == REFERENCE_UPDATES
    assert set((tmp_path / "frozen").iterdir()) == {
        tmp_path / "frozen" / "manifest.json",
        tmp_path / "frozen" / "inference",
        tmp_path / "frozen" / "bootstrap-summary.json",
    }
    payload = json.loads((tmp_path / "frozen" / "manifest.json").read_text())
    assert "held_out" not in json.dumps(payload)
    assert "held-out" not in json.dumps(payload)
    assert load_frozen_bootstrap_candidate(tmp_path / "frozen").manifest == manifest


def test_freeze_rejects_incomplete_compute_before_writing_destination(tmp_path: Path) -> None:
    training = _checkpoint(tmp_path / "training", updates=REFERENCE_UPDATES - 1)
    summary = _write_summary(tmp_path / "bootstrap-summary.json")
    destination = tmp_path / "frozen"
    destination.mkdir()

    with pytest.raises(BootstrapFreezeError, match="final fixed-compute"):
        freeze_bootstrap_candidate(
            training,
            destination,
            development_corpus_name="development-v1",
            development_corpus_digest=_DEVELOPMENT_DIGEST,
            bootstrap_summary_path=summary,
        )

    assert list(destination.iterdir()) == []


def test_freeze_rejects_held_out_labels_and_nonempty_destinations(tmp_path: Path) -> None:
    training = _checkpoint(tmp_path / "training")
    summary = _write_summary(tmp_path / "bootstrap-summary.json")
    destination = tmp_path / "frozen"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(BootstrapFreezeError, match="destination must be empty"):
        freeze_bootstrap_candidate(
            training,
            destination,
            development_corpus_name="held-out-v1",
            development_corpus_digest=_DEVELOPMENT_DIGEST,
            bootstrap_summary_path=summary,
        )

    assert marker.read_text(encoding="utf-8") == "keep"

    empty_destination = tmp_path / "empty-frozen"
    empty_destination.mkdir()
    with pytest.raises(BootstrapFreezeError, match="development corpus"):
        freeze_bootstrap_candidate(
            training,
            empty_destination,
            development_corpus_name="held-out-v1",
            development_corpus_digest=_DEVELOPMENT_DIGEST,
            bootstrap_summary_path=summary,
        )
    assert list(empty_destination.iterdir()) == []


def test_reload_fails_closed_after_inference_tampering(tmp_path: Path) -> None:
    _freeze(tmp_path)
    model_path = tmp_path / "frozen" / "inference" / "model.pt"
    with model_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(BootstrapFreezeError, match="model digest"):
        load_frozen_bootstrap_candidate(tmp_path / "frozen")


def test_reload_fails_closed_after_bootstrap_summary_tampering(tmp_path: Path) -> None:
    _freeze(tmp_path)
    summary_path = tmp_path / "frozen" / "bootstrap-summary.json"
    with summary_path.open("ab") as file:
        file.write(b"\n")

    with pytest.raises(BootstrapFreezeError, match="summary digest"):
        load_frozen_bootstrap_candidate(tmp_path / "frozen")


def test_freeze_rejects_semantically_incomplete_or_mismatched_summaries(
    tmp_path: Path,
) -> None:
    training = _checkpoint(tmp_path / "training")

    missing_arm = _summary_payload()
    missing_arm_rows = missing_arm["arms"]
    assert isinstance(missing_arm_rows, list)
    missing_arm["arms"] = missing_arm_rows[:-1]

    incomplete_compute = _summary_payload()
    incomplete_rows = incomplete_compute["arms"]
    assert isinstance(incomplete_rows, list)
    incomplete_control = incomplete_rows[0]
    assert isinstance(incomplete_control, dict)
    incomplete_counts = incomplete_control["compute"]
    assert isinstance(incomplete_counts, dict)
    incomplete_counts["completed_updates"] = REFERENCE_UPDATES - 1

    wrong_config = _summary_payload()
    wrong_config_rows = wrong_config["arms"]
    assert isinstance(wrong_config_rows, list)
    wrong_control = wrong_config_rows[0]
    assert isinstance(wrong_control, dict)
    wrong_control["config_digest"] = "0" * 64

    held_out_content = _summary_payload()
    held_out_rows = held_out_content["arms"]
    assert isinstance(held_out_rows, list)
    held_out_control = held_out_rows[0]
    assert isinstance(held_out_control, dict)
    final_metrics = held_out_control["final_learning_metrics"]
    assert isinstance(final_metrics, dict)
    final_metrics["held-out-score"] = 1.0

    private_provenance = _summary_payload()
    private_rows = private_provenance["arms"]
    assert isinstance(private_rows, list)
    private_control = private_rows[0]
    assert isinstance(private_control, dict)
    private_metrics = private_control["final_learning_metrics"]
    assert isinstance(private_metrics, dict)
    private_metrics["dataset_digest"] = "a" * 64

    cases = (
        (missing_arm, "all five official arms"),
        (incomplete_compute, "compute is incomplete"),
        (wrong_config, "config does not match"),
        (held_out_content, "held-out content"),
        (private_provenance, "private training provenance"),
    )
    for index, (payload, message) in enumerate(cases):
        summary = _write_summary(tmp_path / f"summary-{index}.json", payload)
        destination = tmp_path / f"frozen-{index}"
        destination.mkdir()
        with pytest.raises(BootstrapFreezeError, match=message):
            freeze_bootstrap_candidate(
                training,
                destination,
                development_corpus_name="development-v1",
                development_corpus_digest=_DEVELOPMENT_DIGEST,
                bootstrap_summary_path=summary,
            )
        assert list(destination.iterdir()) == []


def test_cloning_summary_uses_redacted_provenance_and_exact_checkpoint_config(
    tmp_path: Path,
) -> None:
    payload = _summary_payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    cloning_arm = next(
        row
        for row in arms
        if isinstance(row, dict) and row["arm"] == "behavior-cloning-balanced-v3-v1"
    )
    provenance = cloning_arm["behavior_cloning"]
    assert isinstance(provenance, dict)
    assert provenance["demonstration_examples"] > provenance["demonstration_games"]
    assert "aggregate_dataset_digest" not in provenance
    assert provenance["optimization_order"] == BEHAVIOR_CLONING_OPTIMIZATION_ORDER
    assert provenance["elapsed_seconds"] == 2.0
    summary = _write_summary(tmp_path / "bootstrap-summary.json", payload)

    with pytest.raises(BootstrapFreezeError, match="cloning config does not match"):
        _read_and_validate_bootstrap_summary(
            summary,
            expected_strategy="behavior-cloning-balanced-v3-v1",
            expected_config_digest=str(cloning_arm["config_digest"]),
            expected_behavior_cloning_config_digest="0" * 64,
        )


def test_freeze_requires_an_exact_40_character_source_commit(tmp_path: Path) -> None:
    training = _checkpoint(tmp_path / "training", source_commit="a" * 64)
    summary = _write_summary(tmp_path / "bootstrap-summary.json")
    destination = tmp_path / "frozen"
    destination.mkdir()

    with pytest.raises(BootstrapFreezeError, match="exact source commit"):
        freeze_bootstrap_candidate(
            training,
            destination,
            development_corpus_name="development-v1",
            development_corpus_digest=_DEVELOPMENT_DIGEST,
            bootstrap_summary_path=summary,
        )
    assert list(destination.iterdir()) == []


def test_verified_freeze_builds_an_identity_preserving_picklable_bot_spec(
    tmp_path: Path,
) -> None:
    frozen = _freeze(tmp_path)

    spec = frozen_bootstrap_bot_spec(tmp_path / "frozen")
    restored = pickle.loads(pickle.dumps(spec))

    assert spec.name == frozen.manifest.identity
    assert spec.bot_id == frozen.manifest.identity
    assert restored == spec
    brain = restored.make_brain()
    assert isinstance(brain, FrozenBootstrapCandidateBrain)


def test_frozen_bot_spec_rechecks_payload_before_constructing_each_brain(
    tmp_path: Path,
) -> None:
    _freeze(tmp_path)
    candidate_path = tmp_path / "frozen"
    spec = frozen_bootstrap_bot_spec(candidate_path)
    with (candidate_path / "inference" / "model.pt").open("ab") as file:
        file.write(b"tampered-after-spec-creation")

    with pytest.raises(BootstrapFreezeError, match="model digest"):
        spec.make_brain()
    with pytest.raises(BootstrapFreezeError, match="model digest"):
        frozen_bootstrap_bot_spec(candidate_path)
