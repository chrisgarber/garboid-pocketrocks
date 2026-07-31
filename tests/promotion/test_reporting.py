from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec, RandomBot
from garboid_pocketrocks.heuristics.frozen import FROZEN_CANDIDATES_BY_NAME
from garboid_pocketrocks.promotion.analysis import (
    PromotionAnalysis,
    PromotionFailure,
    RatingDifferenceInterval,
)
from garboid_pocketrocks.promotion.candidates import (
    PromotionCandidateError,
    resolve_promotion_candidate,
)
from garboid_pocketrocks.promotion.corpus import (
    CorpusPurpose,
    PromotionCorpus,
    PromotionCorpusRecipe,
    corpus_snapshot_payload,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.promotion.reporting import (
    PromotionReport,
    build_promotion_report,
    promotion_report_payload,
    validate_artifact_output_dir,
    write_promotion_artifacts,
)
from garboid_pocketrocks.simulator.monte_carlo import GameSummary

from .helpers import (
    EvilFactory,
    EvilString,
    FrozenCandidateFixture,
    evil_provenance,
    frozen_candidate_fixture,
    promotion_plan,
    result_for_plan,
)

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
        unattributed_faults=3,
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
        opponent_pool=plan.opponent_pool,
        plan=plan,
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


def _frozen_report_inputs() -> tuple[
    PromotionReport,
    tuple[GameSummary, ...],
    PromotionCorpus,
    PromotionCorpus,
    dict[str, FrozenCandidateFixture],
]:
    report, summaries, development, held_out = _report_inputs()
    report = replace(
        report,
        incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
        opponents=(
            BOT_SPECS_BY_NAME["random"],
            BOT_SPECS_BY_NAME["aggressive-v1"],
        ),
    )
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=tuple(opponent.name for opponent in report.opponents),
        ),
    )
    report = replace(report, held_out=held_out)
    development = replace(
        development,
        digest=recompute_promotion_corpus_digest(development),
    )
    frozen = frozen_candidate_fixture(
        bot_spec=report.candidate,
        predecessor_name=report.incumbent.name,
        development=development,
    )
    catalog = {report.candidate.name: frozen}
    resolved = resolve_promotion_candidate(
        report.candidate.name,
        registry={},
        frozen_candidates=catalog,
    )
    return (
        replace(
            report,
            development=development,
            candidate_provenance=resolved.frozen_provenance,
        ),
        summaries,
        development,
        held_out,
        catalog,
    )


def test_report_payload_has_complete_explicit_schema() -> None:
    report, _, development, held_out = _report_inputs()

    payload = promotion_report_payload(report)

    assert set(payload) == {
        "schema_version",
        "repository_commit",
        "candidate",
        "incumbent",
        "opponents",
        "opponent_pool",
        "effective_plan",
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
        {"name": "opponent-b", "bot_id": "opponent-b"},
        {"name": "opponent-a", "bot_id": "opponent-a"},
    ]
    assert payload["opponent_pool"] == {
        "configured": [
            {"name": "opponent-a", "bot_id": "opponent-a"},
            {"name": "opponent-b", "bot_id": "opponent-b"},
        ],
        "exclusions": [],
        "remaining": [
            {"name": "opponent-a", "bot_id": "opponent-a"},
            {"name": "opponent-b", "bot_id": "opponent-b"},
        ],
    }
    effective_plan = payload["effective_plan"]
    assert isinstance(effective_plan, dict)
    assert report.plan is not None
    assert effective_plan["digest"] == report.plan.digest
    assert len(effective_plan["pairs"]) == 2
    assert payload["execution"] == {
        "bot_ids": ["candidate", "incumbent", "opponent-b", "opponent-a"],
        "games": 4,
        "player_counts": [3],
        "value_charts": [case.chart for case in held_out.cases],
        "root_seed": held_out.recipe.root_seed,
        "objectives_enabled": [True],
        "fault_mode": "record_and_pass",
        "capture_replays": False,
        "workers": 2,
        "batch_size": 32,
    }
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
        "total": 5,
        "unattributed": 3,
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
            "repetitions_per_seat_cell": corpus.recipe.repetitions_per_seat_cell,
            "charts": list(corpus.recipe.charts),
            "player_counts": list(corpus.recipe.player_counts),
            "opponent_names": list(corpus.recipe.opponent_names),
            "engine_seeds": list(corpus.engine_seeds),
        }

    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "brain_factory" not in encoded
    assert "build_brain" not in encoded


def test_frozen_candidate_report_records_complete_search_provenance() -> None:
    report, _, _, _, _ = _frozen_report_inputs()

    payload = promotion_report_payload(report)

    assert payload["candidate_provenance"] == {
        "kind": "frozen_heuristic_candidate",
        "candidate_name": report.candidate.name,
        "candidate_bot_id": report.candidate.bot_id,
        "predecessor_name": report.incumbent.name,
        "development_corpus_name": report.development.recipe.name,
        "development_corpus_digest": report.development.digest,
        "search_name": "fixture-v3-search-v1",
        "repository_commit": "1" * 40,
        "freeze_digest": "a" * 64,
        "profile_digest": "b" * 64,
        "manifest_digest": "c" * 64,
        "search_report_digest": "d" * 64,
        "candidate_evaluations_digest": "e" * 64,
    }


def test_writer_does_not_accept_an_alternate_frozen_candidate_catalog() -> None:
    fake_catalog = {"forged-candidate": object()}

    with pytest.raises(TypeError, match="unexpected keyword"):
        inspect.signature(write_promotion_artifacts).bind_partial(
            frozen_candidates=fake_catalog,
        )


def test_writer_rejects_a_forged_frozen_identity_in_the_caller_registry(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    forged = BotSpec.for_simulation(frozen.bot_spec.name, RandomBot.build_brain)
    report, summaries, development, held_out = _report_inputs()

    with pytest.raises(PromotionCandidateError, match="provenance"):
        write_promotion_artifacts(
            tmp_path,
            report=replace(report, candidate=forged),
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={forged.name: forged},
        )

    assert not tuple(tmp_path.iterdir())


def test_writer_rejects_a_forged_predecessor_for_a_real_frozen_candidate(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged_predecessor = BotSpec.for_simulation(
        frozen.predecessor_name,
        RandomBot.build_brain,
    )
    report, summaries, development, held_out = _report_inputs()

    with pytest.raises(PromotionCandidateError, match="exact canonical predecessor"):
        write_promotion_artifacts(
            tmp_path,
            report=replace(
                report,
                candidate=frozen.bot_spec,
                incumbent=forged_predecessor,
                candidate_provenance=resolved.frozen_provenance,
            ),
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={forged_predecessor.name: forged_predecessor},
        )

    assert not tuple(tmp_path.iterdir())


def test_writer_rejects_a_different_equal_frozen_bot_spec(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged = replace(frozen.bot_spec, brain_factory=EvilFactory())
    development = load_promotion_corpus(
        Path(
            f"configs/promotion/development-{frozen.personality}-v3-broad-v1.json"
        ),
        registry=BOT_SPECS_BY_NAME,
    )
    report, summaries, _, held_out = _report_inputs()
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    report = replace(
        report,
        candidate=forged,
        incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
        opponents=(
            BOT_SPECS_BY_NAME["random"],
            BOT_SPECS_BY_NAME["aggressive-v1"],
        ),
        development=development,
        held_out=held_out,
        candidate_provenance=resolved.frozen_provenance,
    )
    assert forged == frozen.bot_spec

    with pytest.raises(PromotionCandidateError, match="trusted frozen candidate record"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry=BOT_SPECS_BY_NAME,
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("search_name", "forged-search-name"),
        ("profile_digest", "f" * 64),
    ),
)
def test_writer_rejects_lying_provenance_string_subclasses(
    tmp_path: Path,
    field_name: str,
    forged_value: str,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    forged_provenance = replace(
        resolved.frozen_provenance,
        **{field_name: EvilString(forged_value)},
    )
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    report, summaries, _, held_out = _report_inputs()
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    report = replace(
        report,
        candidate=frozen.bot_spec,
        incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
        opponents=(
            BOT_SPECS_BY_NAME["random"],
            BOT_SPECS_BY_NAME["aggressive-v1"],
        ),
        development=development,
        held_out=held_out,
        candidate_provenance=forged_provenance,
    )

    with pytest.raises(PromotionCandidateError, match="built-in strings"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry=BOT_SPECS_BY_NAME,
        )

    assert not tuple(tmp_path.iterdir())


def test_writer_rejects_a_lying_provenance_subclass(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    forged_provenance = evil_provenance(resolved.frozen_provenance)
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    report, summaries, _, held_out = _report_inputs()
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    report = replace(
        report,
        candidate=frozen.bot_spec,
        incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
        opponents=(
            BOT_SPECS_BY_NAME["random"],
            BOT_SPECS_BY_NAME["aggressive-v1"],
        ),
        development=development,
        held_out=held_out,
        candidate_provenance=forged_provenance,
    )

    with pytest.raises(PromotionCandidateError, match="exact provenance type"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry=BOT_SPECS_BY_NAME,
        )

    assert not tuple(tmp_path.iterdir())


def test_frozen_writer_rejects_a_forged_canonical_name_opponent(
    tmp_path: Path,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    development = load_promotion_corpus(
        Path(
            f"configs/promotion/development-{frozen.personality}-v3-broad-v1.json"
        ),
        registry=BOT_SPECS_BY_NAME,
    )
    forged_opponent = BotSpec.for_simulation("random", RandomBot.build_brain)
    report, summaries, _, held_out = _report_inputs()
    held_out = replace(
        held_out,
        recipe=replace(
            held_out.recipe,
            opponent_names=("random", "aggressive-v1"),
        ),
    )
    report = replace(
        report,
        candidate=frozen.bot_spec,
        incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
        opponents=(forged_opponent, BOT_SPECS_BY_NAME["aggressive-v1"]),
        development=development,
        held_out=held_out,
        candidate_provenance=resolved.frozen_provenance,
    )

    with pytest.raises(
        PromotionCandidateError,
        match="exact canonical released opponent",
    ):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={forged_opponent.name: forged_opponent},
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("reported_opponents", ("empty", "subset", "extra"))
def test_frozen_writer_requires_exactly_the_held_out_opponent_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_opponents: str,
) -> None:
    report, summaries, development, held_out, catalog = _frozen_report_inputs()
    required_names = tuple(opponent.name for opponent in report.opponents)
    held_out = replace(
        held_out,
        recipe=replace(held_out.recipe, opponent_names=required_names),
    )
    if reported_opponents == "empty":
        opponents: tuple[BotSpec, ...] = ()
    elif reported_opponents == "subset":
        opponents = report.opponents[:1]
    else:
        opponents = (*report.opponents, BOT_SPECS_BY_NAME["passive-v1"])
    report = replace(
        report,
        opponents=opponents,
        held_out=held_out,
    )
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )

    with pytest.raises(
        PromotionCandidateError,
        match="exactly cover the held-out opponent names",
    ):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={},
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "tampered_field",
    (
        "predecessor_name",
        "development_corpus_digest",
        "freeze_digest",
    ),
)
def test_tampered_frozen_provenance_is_rejected_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_field: str,
) -> None:
    report, summaries, development, held_out, catalog = _frozen_report_inputs()
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )
    assert report.candidate_provenance is not None
    if tampered_field == "predecessor_name":
        provenance = replace(
            report.candidate_provenance,
            predecessor_name="other-incumbent",
        )
    elif tampered_field == "development_corpus_digest":
        provenance = replace(
            report.candidate_provenance,
            development_corpus_digest="f" * 64,
        )
    else:
        provenance = replace(
            report.candidate_provenance,
            freeze_digest="not-a-digest",
        )

    with pytest.raises(PromotionCandidateError, match="frozen candidate"):
        write_promotion_artifacts(
            tmp_path,
            report=replace(report, candidate_provenance=provenance),
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={},
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("tampering", ("candidate", "provenance"))
def test_writer_rejects_identity_swap_and_fabricated_frozen_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampering: str,
) -> None:
    report, summaries, development, held_out, catalog = _frozen_report_inputs()
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )
    assert report.candidate_provenance is not None
    provenance = report.candidate_provenance
    if tampering == "candidate":
        report = replace(
            report,
            candidate=BotSpec.for_simulation(
                report.candidate.name,
                lambda seed: RandomBot.build_brain(seed),
            ),
        )
    else:
        report = replace(
            report,
            candidate_provenance=replace(
                provenance,
                manifest_digest="f" * 64,
            ),
        )

    with pytest.raises(PromotionCandidateError, match="trusted frozen candidate"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={},
        )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "corpus_tampering",
    ("name", "stored_digest", "content"),
)
def test_writer_recomputes_and_rebinds_the_development_corpus_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corpus_tampering: str,
) -> None:
    report, summaries, development, held_out, catalog = _frozen_report_inputs()
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: catalog,
    )
    if corpus_tampering == "name":
        development = replace(
            development,
            recipe=replace(development.recipe, name="other-development"),
        )
        development = replace(
            development,
            digest=recompute_promotion_corpus_digest(development),
        )
    elif corpus_tampering == "stored_digest":
        development = replace(development, digest="f" * 64)
    else:
        development = replace(development, cases=())
    report = replace(report, development=development)

    with pytest.raises(PromotionCandidateError, match="development corpus"):
        write_promotion_artifacts(
            tmp_path,
            report=report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            registry={},
        )

    assert not tuple(tmp_path.iterdir())


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


def test_failed_rollback_preserves_and_reports_the_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, summaries, development, held_out = _report_inputs()
    artifacts = write_promotion_artifacts(
        tmp_path,
        report=report,
        game_summaries=summaries,
        development=development,
        held_out=held_out,
    )
    previous_report = artifacts.report_json.read_bytes()
    changed_report = replace(report, repository_commit="new-commit")
    real_replace = os.replace

    def fail_forward_and_rollback_replacements(source: Any, destination: Any) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if ".staged." in source_path.name and destination_path.name == "paired-games.jsonl":
            raise OSError("simulated forward replacement failure")
        if ".backup." in source_path.name and destination_path.name == "promotion-report.json":
            raise OSError("simulated rollback restoration failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.reporting.os.replace",
        fail_forward_and_rollback_replacements,
    )

    with pytest.raises(RuntimeError, match="could not be fully restored") as captured:
        write_promotion_artifacts(
            tmp_path,
            report=changed_report,
            game_summaries=summaries,
            development=development,
            held_out=held_out,
            overwrite=True,
        )

    recovery_backups = tuple(tmp_path.glob(".promotion-report.json.backup.*"))
    assert len(recovery_backups) == 1
    assert recovery_backups[0].read_bytes() == previous_report
    assert str(recovery_backups[0]) in str(captured.value)
    assert not tuple(tmp_path.glob(".*.staged.*"))
    assert not tuple(tmp_path.glob(".paired-games.jsonl.backup.*"))
    assert not tuple(tmp_path.glob(".corpus-snapshot.json.backup.*"))
