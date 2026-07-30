from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from garboid_pocketrocks.promotion.analysis import (
    PromotionAnalysis,
    PromotionFailure,
    RatingDifferenceInterval,
)
from garboid_pocketrocks.promotion.corpus import (
    CorpusPurpose,
    PromotionCorpus,
    PromotionCorpusRecipe,
    corpus_snapshot_payload,
)
from garboid_pocketrocks.promotion.reporting import (
    PromotionReport,
    build_promotion_report,
    promotion_report_payload,
    validate_artifact_output_dir,
    write_promotion_artifacts,
)
from garboid_pocketrocks.simulator.monte_carlo import GameSummary

from .helpers import promotion_plan, result_for_plan

_ARTIFACT_NAMES = (
    "promotion-report.json",
    "paired-games.jsonl",
    "corpus-snapshot.json",
)


def _corpus(
    *,
    purpose: CorpusPurpose,
    name: str,
    root_seed: int,
    engine_seed_offset: int,
) -> PromotionCorpus:
    plan = promotion_plan(pair_count=2)
    cases = tuple(
        replace(
            pair.case,
            case_id=pair.case.case_id.replace("fixture-held-out-v1", name),
            engine_seed=pair.case.engine_seed + engine_seed_offset,
        )
        for pair in plan.pairs
    )
    return PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name=name,
            purpose=purpose,
            root_seed=root_seed,
            repetitions_per_seat_cell=1,
            charts=tuple(case.chart for case in cases),
            player_counts=(3,),
            opponent_names=("opponent-a", "opponent-b"),
        ),
        cases=cases,
        digest=("d" if purpose == "development" else "e") * 64,
    )


def _report_inputs() -> tuple[
    PromotionReport,
    tuple[GameSummary, ...],
    PromotionCorpus,
    PromotionCorpus,
]:
    plan = promotion_plan(pair_count=2)
    development = _corpus(
        purpose="development",
        name="fixture-development-v1",
        root_seed=8_001,
        engine_seed_offset=1_000,
    )
    held_out = _corpus(
        purpose="held_out",
        name="fixture-held-out-v1",
        root_seed=90_001,
        engine_seed_offset=0,
    )
    analysis = PromotionAnalysis(
        requested_pairs=2,
        completed_pairs=2,
        requested_games=4,
        completed_games=4,
        rating_difference=125.25,
        interval=RatingDifferenceInterval(lower=10.5, upper=240.0),
        bootstrap_requested=1_000,
        bootstrap_converged=998,
        faults_by_identity=(("opponent-a", 2),),
        warnings=("Two bootstrap fits did not converge and were excluded.",),
        failures=(
            PromotionFailure(
                code="bot_fault",
                message="At least one bot faulted, so this run cannot promote the candidate.",
            ),
        ),
        promoted=False,
    )
    report = build_promotion_report(
        repository_commit="0123456789abcdef",
        candidate=plan.candidate,
        incumbent=plan.incumbent,
        opponents=plan.opponents,
        development=development,
        held_out=held_out,
        bootstrap_samples=1_000,
        bootstrap_seed=42,
        workers=2,
        batch_size=32,
        analysis=analysis,
    )
    summaries = result_for_plan(plan).game_summaries
    return report, summaries, development, held_out


def test_report_payload_has_complete_explicit_schema() -> None:
    report, _, development, held_out = _report_inputs()

    payload = promotion_report_payload(report)

    assert set(payload) == {
        "schema_version",
        "repository_commit",
        "candidate",
        "incumbent",
        "opponents",
        "execution",
        "corpora",
        "coverage",
        "rating_difference",
        "confidence_interval_95",
        "bootstrap",
        "faults",
        "warnings",
        "failures",
        "promoted",
        "artifacts",
    }
    assert payload["candidate"] == {"name": "candidate", "bot_id": "candidate"}
    assert payload["incumbent"] == {"name": "incumbent", "bot_id": "incumbent"}
    assert payload["opponents"] == [
        {"name": "opponent-a", "bot_id": "opponent-a"},
        {"name": "opponent-b", "bot_id": "opponent-b"},
    ]
    assert payload["execution"] == {"workers": 2, "batch_size": 32}
    assert payload["coverage"] == {
        "requested_pairs": 2,
        "completed_pairs": 2,
        "requested_games": 4,
        "completed_games": 4,
    }
    assert payload["rating_difference"] == 125.25
    assert payload["confidence_interval_95"] == {"lower": 10.5, "upper": 240.0}
    assert payload["bootstrap"] == {
        "requested": 1_000,
        "converged": 998,
        "seed": 42,
    }
    assert payload["faults"] == {
        "total": 2,
        "by_identity": [{"bot_id": "opponent-a", "count": 2}],
    }
    assert payload["warnings"] == ["Two bootstrap fits did not converge and were excluded."]
    assert payload["failures"] == [
        {
            "code": "bot_fault",
            "message": "At least one bot faulted, so this run cannot promote the candidate.",
        }
    ]
    assert payload["promoted"] is False
    assert payload["artifacts"] == list(_ARTIFACT_NAMES)

    corpora = payload["corpora"]
    assert isinstance(corpora, dict)
    for label, corpus in (("development", development), ("held_out", held_out)):
        corpus_payload = corpora[label]
        assert corpus_payload == {
            "name": corpus.recipe.name,
            "digest": corpus.digest,
            "purpose": corpus.recipe.purpose,
            "root_seed": corpus.recipe.root_seed,
            "engine_seeds": list(corpus.engine_seeds),
        }

    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "brain_factory" not in encoded
    assert "build_brain" not in encoded


def test_writes_byte_identical_sorted_newline_terminated_artifacts(
    tmp_path: Path,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = write_promotion_artifacts(
        first_dir,
        report=report,
        game_summaries=tuple(reversed(summaries)),
        development=development,
        held_out=held_out,
    )
    second = write_promotion_artifacts(
        second_dir,
        report=report,
        game_summaries=tuple(reversed(summaries)),
        development=development,
        held_out=held_out,
    )

    first_paths = (
        first.report_json,
        first.paired_games_jsonl,
        first.corpus_snapshot_json,
    )
    second_paths = (
        second.report_json,
        second.paired_games_jsonl,
        second.corpus_snapshot_json,
    )
    assert tuple(path.name for path in first_paths) == _ARTIFACT_NAMES
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    assert first.report_json.read_bytes().endswith(b"\n")
    assert first.paired_games_jsonl.read_bytes().endswith(b"\n")
    assert first.corpus_snapshot_json.read_bytes().endswith(b"\n")

    game_payloads = [
        json.loads(line)
        for line in first.paired_games_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["game_index"] for item in game_payloads] == sorted(
        summary.game_index for summary in summaries
    )
    assert game_payloads[0]["scores"] == [
        {"seat": 0, "final_money": 100, "rank": 1},
        {"seat": 1, "final_money": 29, "rank": 2},
        {"seat": 2, "final_money": 28, "rank": 3},
    ]

    snapshot = json.loads(first.corpus_snapshot_json.read_text(encoding="utf-8"))
    assert snapshot == {
        "schema_version": 1,
        "development": corpus_snapshot_payload(development),
        "held_out": corpus_snapshot_payload(held_out),
    }


def test_nonfinite_value_fails_before_any_final_artifact_is_written(
    tmp_path: Path,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    report = replace(
        report,
        analysis=replace(report.analysis, rating_difference=math.nan),
    )

    with pytest.raises(ValueError, match="JSON"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
        )

    assert not any((tmp_path / name).exists() for name in _ARTIFACT_NAMES)


@pytest.mark.parametrize("corpus_label", ("development", "held_out"))
def test_rejects_snapshot_corpus_that_does_not_match_the_report(
    tmp_path: Path,
    corpus_label: str,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    mismatched = replace(
        development if corpus_label == "development" else held_out,
        digest="f" * 64,
    )

    with pytest.raises(ValueError, match=f"{corpus_label.replace('_', '-')} corpus"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=mismatched if corpus_label == "development" else development,
            held_out=mismatched if corpus_label == "held_out" else held_out,
        )

    assert not any((tmp_path / name).exists() for name in _ARTIFACT_NAMES)


def test_nonempty_output_directory_requires_explicit_overwrite(tmp_path: Path) -> None:
    report, summaries, development, held_out = _report_inputs()
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
        )

    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert not any((tmp_path / name).exists() for name in _ARTIFACT_NAMES)


def test_output_directory_preflight_is_public_and_defaults_to_safe(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    validate_artifact_output_dir(missing)
    missing.mkdir()
    validate_artifact_output_dir(missing)

    (missing / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        validate_artifact_output_dir(missing)
    validate_artifact_output_dir(missing, overwrite=True)

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("plain file", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_artifact_output_dir(file_path)


def test_overwrite_replaces_only_known_artifacts_and_preserves_unrelated_files(
    tmp_path: Path,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    artifacts = write_promotion_artifacts(
        tmp_path,
        report=report,
        game_summaries=summaries,
        development=development,
        held_out=held_out,
    )
    old_report = artifacts.report_json.read_bytes()
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    updated = write_promotion_artifacts(
        tmp_path,
        report=replace(report, repository_commit="fedcba9876543210"),
        game_summaries=summaries,
        development=development,
        held_out=held_out,
        overwrite=True,
    )

    assert updated.report_json.read_bytes() != old_report
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert {path.name for path in tmp_path.iterdir()} == {
        *_ARTIFACT_NAMES,
        "notes.txt",
    }


@pytest.mark.parametrize("failure_position", (1, 2, 3))
@pytest.mark.parametrize("existing_generation", (False, True))
def test_replace_failure_rolls_back_the_complete_artifact_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_position: int,
    existing_generation: bool,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    previous_bytes: dict[str, bytes] = {}
    unrelated = tmp_path / "notes.txt"
    if existing_generation:
        artifacts = write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
        )
        previous_bytes = {
            path.name: path.read_bytes()
            for path in (
                artifacts.report_json,
                artifacts.paired_games_jsonl,
                artifacts.corpus_snapshot_json,
            )
        }
        unrelated.write_text("keep me", encoding="utf-8")

    changed_development = replace(development, digest="f" * 64)
    changed_report = replace(
        report,
        repository_commit="new-commit",
        development=changed_development,
    )
    changed_summaries = tuple(
        replace(
            summary,
            decision_counts=tuple(count + 1 for count in summary.decision_counts),
        )
        for summary in summaries
    )
    real_replace = os.replace
    artifact_replacements = 0

    def fail_selected_artifact_replacement(source: Any, destination: Any) -> None:
        nonlocal artifact_replacements
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name in _ARTIFACT_NAMES and ".backup." not in source_path.name:
            artifact_replacements += 1
            if artifact_replacements == failure_position:
                raise OSError(f"simulated failure replacing artifact {failure_position}")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.reporting.os.replace",
        fail_selected_artifact_replacement,
    )

    with pytest.raises(
        OSError,
        match=f"simulated failure replacing artifact {failure_position}",
    ):
        write_promotion_artifacts(
            tmp_path,
            report=changed_report,
            game_summaries=changed_summaries,
            development=changed_development,
            held_out=held_out,
            overwrite=existing_generation,
        )

    if existing_generation:
        assert {name: (tmp_path / name).read_bytes() for name in _ARTIFACT_NAMES} == previous_bytes
        assert unrelated.read_text(encoding="utf-8") == "keep me"
    else:
        assert not any((tmp_path / name).exists() for name in _ARTIFACT_NAMES)
    assert not tuple(path for path in tmp_path.iterdir() if path.name.startswith("."))
