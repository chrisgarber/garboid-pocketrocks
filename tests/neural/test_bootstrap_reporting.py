from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.bootstrap_reporting import (  # noqa: E402
    BootstrapReportingError,
    write_bootstrap_report,
)


def _value_metrics(count: int) -> dict[str, object]:
    return {
        "count": count,
        "mean_prediction": 0.1,
        "mean_target": 0.25,
        "mae": 0.2,
        "rmse": 0.3,
        "bias": -0.15,
        "explained_variance": 0.1,
        "correlation": 0.2,
        "calibration": [],
    }


def _metric(update_index: int, *, auxiliary: bool = False) -> dict[str, object]:
    transitions = 4
    epochs = 3
    optimizer_steps = 3
    included = 6 if auxiliary else 0
    return {
        "update_index": update_index,
        "games_per_cell": 119,
        "duration_seconds": 10.0 + update_index,
        "collection": {
            "games": 1785,
            "decisions": 4,
            "elapsed_seconds": 8.0,
            "inference_seconds": 2.0,
            "inference_batches": 2,
            "inference_batch_sizes": [2, 2],
            "cell_games": [
                [f"live-{chart}", players, 119] for chart in "ABCDE" for players in (3, 4, 5)
            ],
            "queue_wait_seconds": 0.0,
            "ipc_seconds": 0.0,
            "worker_busy_seconds": 8.0,
            "inference_batch_p50": 2.0,
            "inference_batch_p95": 2.0,
            "games_per_second": 223.125,
            "decisions_per_second": 0.5,
            "mean_inference_batch_size": 2.0,
        },
        "ppo": {
            "epochs": epochs,
            "optimizer_steps": optimizer_steps,
            "transition_count": transitions,
            "total_loss": 1.0,
            "policy_loss": 0.4,
            "value_loss": 0.5,
            "entropy": 0.6,
            "advantages": [0.0] * transitions,
            "ratios": [1.0] * (transitions * epochs),
            "values": [0.1] * (transitions * epochs),
            "entropies": [0.6] * (transitions * epochs),
            "pre_clip_gradient_norms": [0.4] * optimizer_steps,
            "post_clip_gradient_norms": [0.3] * optimizer_steps,
            "approximate_kl": 0.01,
            "clip_fraction": 0.02,
            "value": _value_metrics(transitions),
            "value_slices": [],
            "heuristic_auxiliary_weighted_loss": 0.02 if auxiliary else 0.0,
            "heuristic_auxiliary": {
                "included_count": included,
                "total_count": transitions * epochs,
                "included_fraction": included / (transitions * epochs),
                "mean_prediction": 0.1 if auxiliary else None,
                "mean_target": 0.3 if auxiliary else None,
                "mean_absolute_error": 0.2 if auxiliary else None,
                "smooth_l1_loss": 0.2 if auxiliary else 0.0,
            },
        },
    }


def _write_run(
    root: Path,
    config_name: str,
    *,
    auxiliary: bool = False,
    cloning: bool = False,
) -> Path:
    run = root / config_name.removesuffix(".json")
    run.mkdir(parents=True)
    source = Path("configs/neural") / config_name
    config = json.loads(source.read_text(encoding="utf-8"))
    (run / "resolved-config.json").write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(_metric(index, auxiliary=auxiliary)) for index in range(2)) + "\n",
        encoding="utf-8",
    )
    if cloning:
        cloning_config = config["behavior_cloning"]
        from garboid_pocketrocks.neural.behavior_cloning import BehaviorCloningConfig

        parsed = BehaviorCloningConfig.from_json_dict(cloning_config)
        game_count = parsed.rounds * parsed.games_per_cell * 15
        shard_count = (game_count + parsed.games_per_shard - 1) // parsed.games_per_shard
        shard_digests = [
            hashlib.sha256(f"shard-{index}".encode()).hexdigest() for index in range(shard_count)
        ]
        shard_game_counts = [
            min(parsed.games_per_shard, game_count - index * parsed.games_per_shard)
            for index in range(shard_count)
        ]
        combined = hashlib.sha256()
        for shard_index, shard_digest in enumerate(shard_digests):
            combined.update(shard_index.to_bytes(8, "big"))
            combined.update(bytes.fromhex(shard_digest))
        digest = combined.hexdigest()
        update_coordinates = [
            (shard_index, epoch_index)
            for shard_index in range(shard_count)
            for epoch_index in range(parsed.epochs)
        ]
        updates = [
            {
                "shard_index": shard_index,
                "epoch_index": epoch_index,
                "minibatch_index": 0,
                "example_count": shard_game_counts[shard_index],
                "negative_log_likelihood": 0.5,
                "teacher_agreement": 0.5,
                "entropy": 0.25,
                "pre_clip_gradient_norm": 0.4,
                "post_clip_gradient_norm": 0.3,
            }
            for shard_index, epoch_index in update_coordinates
        ]
        (run / "behavior-cloning.json").write_text(
            json.dumps(
                {
                    "config": cloning_config,
                    "dataset": {
                        "cell_game_counts": [
                            [
                                f"live-{chart}",
                                players,
                                parsed.rounds * parsed.games_per_cell,
                            ]
                            for chart in "ABCDE"
                            for players in (3, 4, 5)
                        ],
                        "dataset_digest": digest,
                        "example_count": game_count,
                        "game_count": game_count,
                        "shard_count": shard_count,
                        "shards": [
                            {
                                "shard_index": index,
                                "game_count": shard_game_counts[index],
                                "example_count": shard_game_counts[index],
                                "dataset_digest": shard_digests[index],
                            }
                            for index in range(shard_count)
                        ],
                        "teacher_identity": parsed.teacher_identity,
                        "teacher_profile_digest": parsed.teacher_profile_digest,
                    },
                    "training": {
                        "config_digest": parsed.config_digest,
                        "dataset_digest": digest,
                        "example_count": game_count,
                        "epochs": parsed.epochs,
                        "elapsed_seconds": 123.5,
                        "optimizer_steps": len(updates),
                        "updates": updates,
                    },
                }
            ),
            encoding="utf-8",
        )
    return run


def test_report_writes_update_curves_and_redacted_aggregate_provenance(tmp_path: Path) -> None:
    control = _write_run(tmp_path, "heuristic-bootstrap-control-v1.json")
    auxiliary = _write_run(
        tmp_path,
        "heuristic-auxiliary-value-v1.json",
        auxiliary=True,
    )
    (control / "held-out.json").write_text("not json and must not be read", encoding="utf-8")

    paths = write_bootstrap_report([auxiliary, control], tmp_path / "report")

    with paths.learning_curves_csv.open(encoding="utf-8", newline="") as file:
        curves = list(csv.DictReader(file))
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert len(curves) == 4
    assert curves[1]["cumulative_games"] == "3570"
    assert curves[1]["cumulative_decisions"] == "8"
    assert curves[1]["cumulative_optimizer_steps"] == "6"
    assert curves[0]["mean_training_return"] == "0.25"
    assert summary["held_out_loaded"] is False
    assert summary["report_kind"] == "heuristic_bootstrap_training"
    assert summary["official_arm_contract"]["arm_count"] == 5
    assert summary["reported_arm_count"] == 2
    assert summary["all_official_arms_present"] is False
    assert summary["learning_curves_digest"] == paths.learning_curves_digest
    assert (
        paths.learning_curves_digest
        == hashlib.sha256(paths.learning_curves_csv.read_bytes()).hexdigest()
    )
    assert paths.summary_digest == hashlib.sha256(paths.summary_json.read_bytes()).hexdigest()
    assert [arm["arm"] for arm in summary["arms"]] == sorted(
        ("fixed-compute-control-v1", "auxiliary-value-balanced-v3-v1")
    )
    assert summary["arms"][0]["compute"]["completed_updates"] == 2
    serialized = paths.learning_curves_csv.read_text(encoding="utf-8") + json.dumps(summary)
    for private_field in (
        "engine_seed",
        "policy_seed",
        "opponent_seed",
        "advantages",
        "ratios",
        '"values"',
        "value_slices",
        str(tmp_path),
    ):
        assert private_field not in serialized


def test_report_summarizes_behavior_cloning_without_optimizer_rows(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "heuristic-behavior-cloning-v1.json",
        cloning=True,
    )

    paths = write_bootstrap_report([run], tmp_path / "report")
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    cloning = summary["arms"][0]["behavior_cloning"]

    assert cloning["method"] == "behavior_cloning"
    assert cloning["demonstration_games"] == 7140
    assert cloning["demonstration_examples"] == 7140
    assert cloning["optimizer_steps"] == cloning["shard_count"] * cloning["epochs"]
    assert len(cloning["cell_game_counts"]) == 15
    assert len(cloning["provenance_digest"]) == 64
    assert cloning["optimization_order"] == "sequential-shard-major-epochs-v1"
    assert "aggregate_dataset_digest" not in cloning
    assert "shard_digests" not in cloning
    assert "shard_index" not in cloning
    assert "updates" not in cloning
    compute = summary["arms"][0]["compute"]
    assert compute["total_training_games"] == 7140 + 3570
    assert compute["optimizer_steps"]["total"] == (
        compute["optimizer_steps"]["behavior_cloning"] + compute["optimizer_steps"]["ppo"]
    )
    assert compute["duration_seconds"]["behavior_cloning"] == 123.5

    source = json.loads((run / "behavior-cloning.json").read_text(encoding="utf-8"))
    assert source["dataset"]["shards"][0]["dataset_digest"] not in json.dumps(summary)


def test_report_rejects_tampered_cloning_shard_aggregate(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "heuristic-behavior-cloning-v1.json",
        cloning=True,
    )
    path = run / "behavior-cloning.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset"]["shards"][0]["dataset_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BootstrapReportingError, match="aggregate dataset digest"):
        write_bootstrap_report([run], tmp_path / "report")


@pytest.mark.parametrize("mutation", ("duplicate", "gap", "wrong_examples"))
def test_report_rejects_nonexact_cloning_minibatch_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    run = _write_run(tmp_path, "heuristic-behavior-cloning-v1.json", cloning=True)
    path = run / "behavior-cloning.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    updates = payload["training"]["updates"]
    if mutation == "duplicate":
        updates[1] = dict(updates[0])
    elif mutation == "gap":
        updates[0]["minibatch_index"] = 1
    else:
        updates[0]["example_count"] -= 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BootstrapReportingError, match="optimizer"):
        write_bootstrap_report([run], tmp_path / "report")


def test_report_recognizes_the_complete_official_five_arm_contract(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, "heuristic-bootstrap-control-v1.json"),
        _write_run(tmp_path, "heuristic-opponent-control-v1.json"),
        _write_run(
            tmp_path,
            "heuristic-behavior-cloning-v1.json",
            cloning=True,
        ),
        _write_run(
            tmp_path,
            "heuristic-auxiliary-value-v1.json",
            auxiliary=True,
        ),
        _write_run(tmp_path, "heuristic-opponent-curriculum-v1.json"),
    ]

    paths = write_bootstrap_report(runs, tmp_path / "report")
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))

    assert summary["all_official_arms_present"] is True
    assert summary["reported_arm_count"] == 5
    assert {arm["arm"] for arm in summary["arms"]} == {
        contract["arm"] for contract in summary["official_arm_contract"]["arms"]
    }


@pytest.mark.parametrize(
    "mutation",
    ("gap", "unknown_key", "nonfinite", "bad_auxiliary_count"),
)
def test_report_rejects_malformed_metrics_jsonl(tmp_path: Path, mutation: str) -> None:
    run = _write_run(tmp_path, "heuristic-bootstrap-control-v1.json")
    first = _metric(0)
    second = _metric(1)
    if mutation == "gap":
        second["update_index"] = 2
    elif mutation == "unknown_key":
        second["engine_seed"] = 7
    elif mutation == "nonfinite":
        second["duration_seconds"] = float("nan")
    else:
        second["ppo"]["heuristic_auxiliary"]["total_count"] = 11  # type: ignore[index]
    (run / "metrics.jsonl").write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BootstrapReportingError):
        write_bootstrap_report([run], tmp_path / "report")


def test_report_rejects_duplicate_arms_and_nonempty_destination(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "first", "heuristic-bootstrap-control-v1.json")
    second = _write_run(tmp_path / "second", "heuristic-bootstrap-control-v1.json")

    with pytest.raises(BootstrapReportingError, match="distinct bootstrap arms"):
        write_bootstrap_report([first, second], tmp_path / "duplicate-report")

    destination = tmp_path / "existing-report"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(BootstrapReportingError, match="must be empty"):
        write_bootstrap_report([first], destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
