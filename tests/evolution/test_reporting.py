from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import garboid_pocketrocks.evolution.diagnostics as diagnostics_module
import garboid_pocketrocks.evolution.reporting as reporting_module
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution.candidates import candidate_bot_spec
from garboid_pocketrocks.evolution.diagnostics import (
    WINNER_DECISION_SLICES_NAME,
    WINNER_DIAGNOSTICS_JSON_NAME,
    WINNER_DIAGNOSTICS_MARKDOWN_NAME,
    WinnerDiagnostics,
    run_winner_diagnostics,
)
from garboid_pocketrocks.evolution.manifest import PhaseSearchManifest
from garboid_pocketrocks.evolution.planning import DevelopmentPlan
from garboid_pocketrocks.evolution.reporting import (
    SearchReport,
    build_search_report,
    repository_commit,
    search_report_payload,
    validate_search_output_dir,
    write_search_artifacts,
)
from garboid_pocketrocks.evolution.runner import SearchFailure, SearchRun, run_search
from garboid_pocketrocks.promotion.corpus import (
    corpus_snapshot_payload,
    recompute_promotion_corpus_digest,
)

from .test_runner import (
    _phase_run_with_recomputed_positive_evidence,
    _run_small_phase_search,
    _small_inputs,
    _small_phase_inputs,
)

_REQUIRED_ARTIFACTS = (
    "search-manifest.json",
    "search-report.json",
    "candidate-evaluations.jsonl",
    "selection-log.jsonl",
    "development-games.jsonl",
    "development-corpus-snapshot.json",
)
_ALL_ARTIFACTS = (*_REQUIRED_ARTIFACTS, "frozen-candidate.json")
_ALL_PHASE_ARTIFACTS = (
    *_ALL_ARTIFACTS,
    WINNER_DECISION_SLICES_NAME,
    WINNER_DIAGNOSTICS_JSON_NAME,
    WINNER_DIAGNOSTICS_MARKDOWN_NAME,
)


@pytest.fixture(scope="module")
def search_run() -> SearchRun:
    manifest, corpus = _small_inputs(generations=1, population=2, elites=1, cases=1)
    return run_search(
        manifest,
        corpus,
        registry=BOT_SPECS_BY_NAME,
        workers=1,
        batch_size=1,
    )


@pytest.fixture(scope="module")
def phase_search_runs() -> tuple[SearchRun, SearchRun, SearchRun]:
    manifest, corpus = _small_phase_inputs(generations=1, cases=1)
    return (
        _run_small_phase_search(
            manifest,
            corpus,
            workers=1,
            batch_size=None,
        ),
        _run_small_phase_search(
            manifest,
            corpus,
            workers=1,
            batch_size=1,
        ),
        _run_small_phase_search(
            manifest,
            corpus,
            workers=2,
            batch_size=1,
        ),
    )


@pytest.fixture(scope="module")
def phase_search_run(
    phase_search_runs: tuple[SearchRun, SearchRun, SearchRun],
) -> SearchRun:
    return phase_search_runs[0]


@pytest.fixture(scope="module")
def frozen_phase_search_run() -> SearchRun:
    manifest, corpus = _small_phase_inputs(generations=1, cases=3)
    run = _run_small_phase_search(
        manifest,
        corpus,
        workers=1,
        batch_size=1,
    )
    return _phase_run_with_recomputed_positive_evidence(run)


@pytest.fixture(scope="module")
def canonical_winner_diagnostics(
    frozen_phase_search_run: SearchRun,
) -> WinnerDiagnostics:
    return _canonical_winner_diagnostics(frozen_phase_search_run)


def _report(run: SearchRun, *, frozen: bool = False) -> SearchReport:
    selected = run.selected_candidate
    assert selected is not None
    normalized_run = replace(
        run,
        frozen_candidate=selected if frozen else None,
    )
    return build_search_report(
        normalized_run,
        repository_commit="0123456789abcdef",
        workers=1,
        batch_size=1,
    )


def _phase_report(
    run: SearchRun,
    *,
    repository_commit: str = "0123456789abcdef",
    workers: int = 1,
    batch_size: int = 1,
    winner_diagnostics: WinnerDiagnostics | None = None,
) -> SearchReport:
    """Build a tiny schema-v2 fixture while bypassing only the fixed-budget contract."""

    with (
        patch.object(
            reporting_module,
            "_validate_fixed_phase_manifest_for_report",
            return_value=None,
        ),
        patch.object(
            diagnostics_module,
            "_require_exact_winner_case_count",
            return_value=None,
        ),
    ):
        return build_search_report(
            run,
            repository_commit=repository_commit,
            workers=workers,
            batch_size=batch_size,
            winner_diagnostics=winner_diagnostics,
        )


def _canonical_winner_diagnostics(run: SearchRun) -> WinnerDiagnostics:
    with (
        patch.object(
            reporting_module,
            "_validate_fixed_phase_manifest_for_report",
            return_value=None,
        ),
        patch.object(
            diagnostics_module,
            "_require_exact_winner_case_count",
            return_value=None,
        ),
    ):
        return run_winner_diagnostics(
            run,
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            batch_size=1,
        )


def _replace_first_phase_plan(run: SearchRun, plan: DevelopmentPlan) -> SearchRun:
    return replace(
        run,
        candidate_runs=(
            replace(
                run.candidate_runs[0],
                plan=plan,
            ),
            *run.candidate_runs[1:],
        ),
    )


def test_report_payload_owns_complete_normalized_search_evidence(
    search_run: SearchRun,
) -> None:
    report = _report(search_run, frozen=True)
    assert report.run.selected_candidate is not None
    assert report.run.frozen_candidate is not None

    payload = search_report_payload(report)

    assert set(payload) == {
        "schema_version",
        "repository_commit",
        "status",
        "search",
        "development_corpus",
        "execution",
        "coverage",
        "best_result",
        "selected_candidate_identity",
        "frozen_candidate_identity",
        "failures",
        "artifacts",
    }
    assert payload["repository_commit"] == "0123456789abcdef"
    assert payload["status"] == "frozen_improvement"
    assert payload["search"] == {
        "name": report.run.manifest.name,
        "manifest_digest": report.run.manifest.digest,
        "personality": report.run.manifest.personality,
        "predecessor_name": report.run.manifest.predecessor_name,
        "generation_count": 1,
        "population_size": 2,
        "elite_count": 1,
    }
    assert payload["development_corpus"] == {
        "name": report.run.development_corpus.recipe.name,
        "digest": report.run.development_corpus.digest,
        "cases": 1,
    }
    assert payload["execution"] == {"workers": 1, "batch_size": 1}
    assert payload["coverage"] == {
        "proposed_candidates": 2,
        "evaluated_candidates": 2,
        "completed_generations": 1,
        "requested_baseline_games": 1,
        "completed_baseline_games": 1,
        "requested_candidate_games": 2,
        "completed_candidate_games": 2,
    }
    assert payload["best_result"] == {
        "candidate_identity": report.run.selected_candidate.identity,
        "generation": report.run.selected_candidate.generation,
        "slot": report.run.selected_candidate.slot,
        "rating_delta": next(
            item.evaluation.rating_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
        "normalized_finish_delta": next(
            item.evaluation.normalized_finish_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
        "final_money_delta": next(
            item.evaluation.final_money_delta
            for item in report.run.candidate_runs
            if item.candidate == report.run.selected_candidate
        ),
    }
    assert payload["selected_candidate_identity"] == report.run.selected_candidate.identity
    assert payload["frozen_candidate_identity"] == report.run.frozen_candidate.identity
    assert payload["failures"] == []
    assert payload["artifacts"] == list(_ALL_ARTIFACTS)


def test_writes_byte_identical_complete_artifact_generations(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    report = _report(search_run, frozen=True)
    assert report.run.frozen_candidate is not None

    first = write_search_artifacts(tmp_path / "first", report=report)
    second = write_search_artifacts(tmp_path / "second", report=report)

    first_paths = tuple(path for path in first.paths)
    second_paths = tuple(path for path in second.paths)
    assert tuple(path.name for path in first_paths) == _ALL_ARTIFACTS
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]
    assert all(path.read_bytes().endswith(b"\n") for path in first_paths)

    manifest = json.loads(first.search_manifest_json.read_text())
    assert manifest["digest"] == report.run.manifest.digest
    assert json.loads(first.development_corpus_snapshot_json.read_text()) == (
        corpus_snapshot_payload(report.run.development_corpus)
    )

    evaluations = [
        json.loads(line) for line in first.candidate_evaluations_jsonl.read_text().splitlines()
    ]
    assert [item["candidate"]["identity"] for item in evaluations] == [
        item.candidate.identity for item in report.run.candidate_runs
    ]
    assert evaluations[0]["ranking_key"]["fields"] == [
        "negative_rating_delta",
        "negative_normalized_finish_delta",
        "negative_final_money_delta",
        "coefficient_values",
        "candidate_identity",
    ]

    games = [json.loads(line) for line in first.development_games_jsonl.read_text().splitlines()]
    assert [item["evidence"] for item in games] == [
        "baseline",
        "candidate",
        "candidate",
    ]
    assert [item["case_index"] for item in games] == [0, 0, 0]
    assert [item["candidate_identity"] for item in games] == [
        None,
        report.run.candidate_runs[0].candidate.identity,
        report.run.candidate_runs[1].candidate.identity,
    ]

    assert first.frozen_candidate_json is not None
    frozen = json.loads(first.frozen_candidate_json.read_text())
    assert frozen["identity"] == report.run.frozen_candidate.identity
    assert frozen["repository_commit"] == report.repository_commit
    assert frozen["source_evidence"]["search_report_sha256"]
    assert frozen["source_evidence"]["candidate_evaluations_sha256"]


def test_schema_v1_complete_artifact_hashes_are_frozen(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    artifacts = write_search_artifacts(
        tmp_path,
        report=_report(search_run, frozen=True),
    )
    expected = {
        "search-manifest.json": (
            1131,
            "60679253fa068cbaaa49ba1609f33724bb1d42ff436f3e2bcc9ac3c3914fad8b",
        ),
        "search-report.json": (
            1504,
            "79887bd72be2f2cba892edfb8ea865f75702aef8a0e50dd9d6ab3354bdad3bf5",
        ),
        "candidate-evaluations.jsonl": (
            1833,
            "79379a1093538cf5b56a8dde405d611682e6bb876a8ffa1c92ab2ba17e7267fe",
        ),
        "selection-log.jsonl": (
            1090,
            "df81b9f4e619ad516a6f5980c6f01e4951b5588ff9c8f51310b52d3cb1646167",
        ),
        "development-games.jsonl": (
            1940,
            "033eb766de10aa08753b2932b4efa4fe1e8e9b06306e2d1b4e1ac2293e3203f6",
        ),
        "development-corpus-snapshot.json": (
            778,
            "0460d0975a6286bdc9676054df911bd21cf24f02a483d63fdef916793bebcf38",
        ),
        "frozen-candidate.json": (
            1136,
            "2bcddd2d923ce1e0253344b3cf16f7f99b57451466fd309a9bd1d9e192993cba",
        ),
    }

    assert set(path.name for path in artifacts.paths) == set(expected)
    for path in artifacts.paths:
        content = path.read_bytes()
        expected_size, expected_digest = expected[path.name]
        assert len(content) == expected_size
        assert hashlib.sha256(content).hexdigest() == expected_digest


def test_phase_report_normalizes_selector_evidence_and_all_twelve_coefficients(
    tmp_path: Path,
    phase_search_run: SearchRun,
) -> None:
    report = _phase_report(
        phase_search_run,
        repository_commit="0123456789abcdef",
        workers=1,
        batch_size=1,
    )
    first = write_search_artifacts(tmp_path / "first", report=report)
    second = write_search_artifacts(tmp_path / "second", report=report)

    assert report.schema_version == 2
    assert isinstance(phase_search_run.manifest, PhaseSearchManifest)
    assert [path.read_bytes() for path in first.paths] == [
        path.read_bytes() for path in second.paths
    ]
    expected_artifacts = {
        "search-manifest.json": (
            3057,
            "a6ac63169b4c8dcdb8f5e9a6bdf08ba9524987897d6141e3a10f55fbd9e4428b",
        ),
        "search-report.json": (
            3106,
            "97615c85dc8f55993588ce209068b095c23649dd5f77f84be774df71e9e42dac",
        ),
        "candidate-evaluations.jsonl": (
            21587,
            "72be5e37b50f4ce56d9c61426e9838694d253ce887a30fcaedcb5fd443430267",
        ),
        "selection-log.jsonl": (
            8526,
            "568468bbe4156f374ed3fa0ddcef06c53fc7e43d276cdbf7a1ed065d9e3bffdc",
        ),
        "development-games.jsonl": (
            11479,
            "2567af9b6a30cb6a8c80a821bf6a2e59ba9473033b0e383317fb2b0b1d9f2f86",
        ),
        "development-corpus-snapshot.json": (
            778,
            "0460d0975a6286bdc9676054df911bd21cf24f02a483d63fdef916793bebcf38",
        ),
    }
    assert {path.name for path in first.paths} == set(expected_artifacts)
    for path in first.paths:
        content = path.read_bytes()
        expected_size, expected_digest = expected_artifacts[path.name]
        assert len(content) == expected_size
        assert hashlib.sha256(content).hexdigest() == expected_digest
    search_payload = json.loads(first.search_report_json.read_text())
    assert search_payload["schema_version"] == 2
    assert search_payload["search"]["phase_selector"] == {
        "kind": "public-resource-horizon-v1",
        "early": "3*future>=2*total",
        "middle": "3*future>=total",
        "late": "otherwise",
    }
    assert search_payload["search"]["boundary_evidence"] == {
        "report_path": phase_search_run.manifest.boundary_evidence.report_path,
        "report_digest": phase_search_run.manifest.boundary_evidence.report_digest,
        "slices_path": phase_search_run.manifest.boundary_evidence.slices_path,
        "slices_digest": phase_search_run.manifest.boundary_evidence.slices_digest,
    }
    evaluations = [
        json.loads(line) for line in first.candidate_evaluations_jsonl.read_text().splitlines()
    ]
    assert phase_search_run.selected_candidate is not None
    assert (
        phase_search_run.selected_candidate.identity
        == "balanced-v4-candidate-g000-s008-d4f1f74b21cf"
    )
    selected_evaluation = next(
        item
        for item in evaluations
        if item["candidate"]["identity"] == phase_search_run.selected_candidate.identity
    )
    assert selected_evaluation["candidate"]["profile_digest"] == (
        "d4f1f74b21cf64da670e62c9482d45cc9df9b9630f4589a21b209da862618a83"
    )
    assert selected_evaluation["scores"] == {
        "rating_delta": 0.0,
        "normalized_finish_delta": 0.0,
        "final_money_delta": 2,
    }
    first_candidate = evaluations[0]["candidate"]
    assert first_candidate["phase_selector"] == search_payload["search"]["phase_selector"]
    assert tuple(first_candidate["experts"]) == ("early", "late", "middle")
    assert sum(len(values) for values in first_candidate["experts"].values()) == 12
    expected_expert = {
        "liquidity_strength": "0.25",
        "future_cash_weight": "1.55",
        "objective_progress_weight": "0.3",
        "bid_shading": "0.35",
    }
    assert first_candidate["experts"] == {
        "early": expected_expert,
        "middle": expected_expert,
        "late": expected_expert,
    }
    assert evaluations[0]["ranking_key"]["fields"] == [
        "negative_rating_delta",
        "negative_normalized_finish_delta",
        "negative_final_money_delta",
        "coefficient_values",
        "candidate_identity",
    ]
    assert len(evaluations[0]["ranking_key"]["values"][3]) == 12


def test_phase_artifacts_are_byte_identical_across_execution_modes(
    tmp_path: Path,
    phase_search_runs: tuple[SearchRun, SearchRun, SearchRun],
) -> None:
    rendered: list[dict[str, bytes]] = []
    for index, run in enumerate(phase_search_runs):
        report = _phase_report(
            run,
            repository_commit="0123456789abcdef",
            workers=2,
            batch_size=1,
        )
        artifacts = write_search_artifacts(tmp_path / str(index), report=report)
        rendered.append({path.name: path.read_bytes() for path in artifacts.paths})

    assert rendered[0] == rendered[1] == rendered[2]


def test_phase_execution_modes_have_identical_run_outcomes(
    phase_search_runs: tuple[SearchRun, SearchRun, SearchRun],
) -> None:
    reference = phase_search_runs[0]
    for run in phase_search_runs[1:]:
        assert run.baseline_result == reference.baseline_result
        assert tuple(item.candidate for item in run.candidate_runs) == tuple(
            item.candidate for item in reference.candidate_runs
        )
        assert tuple(item.evaluation for item in run.candidate_runs) == tuple(
            item.evaluation for item in reference.candidate_runs
        )
        assert tuple(item.result for item in run.candidate_runs) == tuple(
            item.result for item in reference.candidate_runs
        )
        assert run.selections == reference.selections
        assert run.selected_candidate == reference.selected_candidate
        assert run.frozen_candidate == reference.frozen_candidate


def test_phase_report_revalidates_the_fixed_manifest_contract(
    phase_search_run: SearchRun,
) -> None:
    with pytest.raises(ValueError, match="exact committed v4 development-search contract"):
        build_search_report(
            phase_search_run,
            repository_commit="0123456789abcdef",
            workers=1,
            batch_size=1,
        )


def test_phase_report_revalidates_corpus_digest_and_manifest_binding(
    phase_search_run: SearchRun,
) -> None:
    stale_corpus = replace(
        phase_search_run.development_corpus,
        digest="0" * 64,
    )
    renamed_corpus = replace(
        phase_search_run.development_corpus,
        recipe=replace(
            phase_search_run.development_corpus.recipe,
            name="development-forged",
        ),
    )
    renamed_corpus = replace(
        renamed_corpus,
        digest=recompute_promotion_corpus_digest(renamed_corpus),
    )
    for corpus, expected_message in (
        (stale_corpus, "corpus digest"),
        (renamed_corpus, "manifest binding"),
    ):
        with pytest.raises(ValueError, match=expected_message):
            _phase_report(
                replace(phase_search_run, development_corpus=corpus),
            )


@pytest.mark.parametrize("corpus_field", ("name", "digest"))
def test_phase_report_rejects_forged_plan_corpus_binding(
    phase_search_run: SearchRun,
    corpus_field: str,
) -> None:
    plan = phase_search_run.candidate_runs[0].plan
    if corpus_field == "name":
        forged_corpus = replace(
            plan.corpus,
            recipe=replace(plan.corpus.recipe, name="development-forged"),
        )
    else:
        forged_corpus = replace(plan.corpus, digest="0" * 64)
    forged = _replace_first_phase_plan(
        phase_search_run,
        replace(plan, corpus=forged_corpus),
    )

    with pytest.raises(ValueError, match="plan corpus"):
        _phase_report(forged)


def test_phase_report_rejects_forged_candidate_factory_profile(
    phase_search_run: SearchRun,
) -> None:
    candidate_run = phase_search_run.candidate_runs[0]
    other_spec = candidate_bot_spec(phase_search_run.candidate_runs[1].candidate)
    forged_spec = replace(
        other_spec,
        name=candidate_run.candidate.identity,
        bot_id=candidate_run.candidate.identity,
    )
    forged = _replace_first_phase_plan(
        phase_search_run,
        replace(candidate_run.plan, candidate=forged_spec),
    )

    with pytest.raises(ValueError, match="candidate identity and profile"):
        _phase_report(forged)


@pytest.mark.parametrize("binding", ("incumbent", "opponents"))
def test_phase_report_rejects_forged_incumbent_or_opponents(
    phase_search_run: SearchRun,
    binding: str,
) -> None:
    plan = phase_search_run.candidate_runs[0].plan
    forged_plan = (
        replace(plan, incumbent=BOT_SPECS_BY_NAME["balanced-v2"])
        if binding == "incumbent"
        else replace(plan, opponents=tuple(reversed(plan.opponents)))
    )
    forged = _replace_first_phase_plan(phase_search_run, forged_plan)

    with pytest.raises(ValueError, match=binding):
        _phase_report(forged)


@pytest.mark.parametrize(
    "evidence",
    ("baseline_config", "candidate_config", "candidate_job", "bool_as_int"),
)
def test_phase_report_rejects_forged_plan_config_or_jobs(
    phase_search_run: SearchRun,
    evidence: str,
) -> None:
    plan = phase_search_run.candidate_runs[0].plan
    if evidence == "baseline_config":
        forged_plan = replace(
            plan,
            baseline_config=replace(
                plan.baseline_config,
                root_seed=plan.baseline_config.root_seed + 1,
            ),
        )
    elif evidence == "candidate_config":
        forged_plan = replace(
            plan,
            candidate_config=replace(
                plan.candidate_config,
                capture_replays=True,
            ),
        )
    else:
        candidate_job = plan.candidate_jobs[0]
        forged_job = (
            replace(candidate_job, seed=candidate_job.seed + 1)
            if evidence == "candidate_job"
            else replace(
                candidate_job,
                objectives_enabled=cast(Any, 1),
            )
        )
        forged_plan = replace(
            plan,
            candidate_jobs=(forged_job, *plan.candidate_jobs[1:]),
        )
    forged = _replace_first_phase_plan(phase_search_run, forged_plan)

    with pytest.raises(ValueError, match="config and jobs"):
        _phase_report(forged)


def test_phase_freeze_requires_complete_fault_free_positive_evidence(
    phase_search_run: SearchRun,
) -> None:
    selected = phase_search_run.selected_candidate
    assert selected is not None
    forged = replace(phase_search_run, frozen_candidate=selected)

    with pytest.raises(ValueError, match="complete fault-free positive evidence"):
        _phase_report(
            forged,
            repository_commit="0123456789abcdef",
            workers=1,
            batch_size=1,
        )


@pytest.mark.parametrize("partial_field", ("candidate_runs", "selections"))
def test_phase_freeze_rejects_partial_search_evidence(
    phase_search_run: SearchRun,
    partial_field: str,
) -> None:
    frozen_run = _phase_run_with_recomputed_positive_evidence(phase_search_run)
    partial = replace(
        frozen_run,
        **{partial_field: getattr(frozen_run, partial_field)[:-1]},
    )

    with pytest.raises(ValueError, match="complete"):
        _phase_report(
            partial,
            repository_commit="0123456789abcdef",
            workers=1,
            batch_size=1,
        )


def test_phase_freeze_rejects_forged_positive_stored_evaluation(
    phase_search_run: SearchRun,
) -> None:
    selected = phase_search_run.selected_candidate
    assert selected is not None
    candidate_runs = tuple(
        replace(
            candidate_run,
            evaluation=replace(
                candidate_run.evaluation,
                rating_delta=1.0,
                normalized_finish_delta=1.0,
                final_money_delta=1,
                valid=True,
                eligible=True,
            ),
        )
        if candidate_run.candidate == selected
        else candidate_run
        for candidate_run in phase_search_run.candidate_runs
    )
    forged = replace(
        phase_search_run,
        candidate_runs=candidate_runs,
        frozen_candidate=selected,
    )

    with pytest.raises(ValueError, match="evaluations recomputed"):
        _phase_report(
            forged,
            repository_commit="0123456789abcdef",
            workers=1,
            batch_size=1,
        )


def test_phase_freeze_rejects_coherently_forged_final_selection_and_record(
    phase_search_run: SearchRun,
) -> None:
    selection = phase_search_run.selections[0]
    original_winner = selection.elites[0]
    forged_winner = next(
        item for item in selection.ranked_pool if item.candidate != original_winner.candidate
    )
    forged_ranked_pool = (
        forged_winner,
        *(item for item in selection.ranked_pool if item != forged_winner),
    )
    forged_elites = (
        forged_winner,
        *(item for item in selection.elites if item != forged_winner),
    )[: phase_search_run.manifest.algorithm.elite_count]
    forged_record = replace(
        selection.record,
        ranked_pool_identities=tuple(item.candidate.identity for item in forged_ranked_pool),
        elite_identities=tuple(item.candidate.identity for item in forged_elites),
        ranking_keys=tuple(
            (item.candidate.identity, item.ranking_key) for item in forged_ranked_pool
        ),
    )
    forged_selection = replace(
        selection,
        ranked_pool=forged_ranked_pool,
        elites=forged_elites,
        record=forged_record,
    )
    forged = replace(
        phase_search_run,
        selections=(forged_selection,),
        selected_candidate=forged_winner.candidate,
        frozen_candidate=forged_winner.candidate,
    )

    with pytest.raises(ValueError, match="selections recomputed"):
        _phase_report(
            forged,
            repository_commit="0123456789abcdef",
            workers=1,
            batch_size=1,
        )


def test_phase_report_rejects_candidate_family_personality_and_selector_tampering(
    phase_search_run: SearchRun,
    search_run: SearchRun,
) -> None:
    phase_candidate = phase_search_run.candidate_runs[0].candidate
    scalar_candidate = search_run.candidate_runs[0].candidate
    wrong_personality = replace(phase_candidate, personality="aggressive")
    wrong_selector = deepcopy(phase_candidate)
    object.__setattr__(wrong_selector.genome, "phase_selector", "forged-selector")
    for forged_candidate, expected_message in (
        (scalar_candidate, "phase-aware candidates"),
        (wrong_personality, "personality"),
        (wrong_selector, "selector"),
    ):
        forged_runs = (
            replace(
                phase_search_run.candidate_runs[0],
                candidate=forged_candidate,
            ),
            *phase_search_run.candidate_runs[1:],
        )
        forged = replace(phase_search_run, candidate_runs=forged_runs)

        with pytest.raises(ValueError, match=expected_message):
            _phase_report(
                forged,
                repository_commit="0123456789abcdef",
                workers=1,
                batch_size=1,
            )


def test_valid_phase_frozen_payload_binds_selector_and_boundary_evidence(
    tmp_path: Path,
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
) -> None:
    assert isinstance(frozen_phase_search_run.manifest, PhaseSearchManifest)
    report = _phase_report(
        frozen_phase_search_run,
        repository_commit="0123456789abcdef",
        workers=1,
        batch_size=1,
        winner_diagnostics=canonical_winner_diagnostics,
    )

    artifacts = write_search_artifacts(tmp_path, report=report)

    assert artifacts.frozen_candidate_json is not None
    payload = json.loads(artifacts.frozen_candidate_json.read_text())
    assert payload["schema_version"] == 2
    assert payload["phase_selector"]["kind"] == "public-resource-horizon-v1"
    assert sum(len(values) for values in payload["experts"].values()) == 12
    assert payload["boundary_evidence"] == {
        "report_path": frozen_phase_search_run.manifest.boundary_evidence.report_path,
        "report_digest": frozen_phase_search_run.manifest.boundary_evidence.report_digest,
        "slices_path": frozen_phase_search_run.manifest.boundary_evidence.slices_path,
        "slices_digest": frozen_phase_search_run.manifest.boundary_evidence.slices_digest,
    }
    expected_leaf_digests = dict(canonical_winner_diagnostics.artifact_digests)
    report_payload = json.loads(artifacts.search_report_json.read_text())
    assert {
        item["name"]: item["sha256"] for item in report_payload["winner_diagnostics"]["artifacts"]
    } == expected_leaf_digests
    assert payload["source_evidence"]["winner_diagnostics"] == expected_leaf_digests
    assert (
        payload["source_evidence"]["selection_log_sha256"]
        == hashlib.sha256(artifacts.selection_log_jsonl.read_bytes()).hexdigest()
    )
    assert (
        payload["source_evidence"]["development_games_sha256"]
        == hashlib.sha256(artifacts.development_games_jsonl.read_bytes()).hexdigest()
    )
    assert set(path.name for path in artifacts.paths) == set(_ALL_PHASE_ARTIFACTS)


def test_phase_report_rejects_self_hashed_noncanonical_diagnostic_bytes(
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
) -> None:
    forged_json = '{"schema_version":2,"private_snapshot":"forged"}\n'
    forged = replace(
        canonical_winner_diagnostics,
        diagnostics_json=forged_json,
    )
    forged = replace(
        forged,
        artifact_digests=tuple(
            (name, hashlib.sha256(content.encode()).hexdigest())
            for name, content in forged.named_contents()
        ),
    )

    with pytest.raises(ValueError, match="not canonical"):
        _phase_report(
            frozen_phase_search_run,
            winner_diagnostics=forged,
        )


@pytest.mark.parametrize(
    "evidence",
    ("diagnostic_plan", "diagnostic_result", "decision_report"),
)
def test_phase_report_recomputes_typed_winner_diagnostic_evidence(
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
    evidence: str,
) -> None:
    if evidence == "diagnostic_plan":
        forged = replace(
            canonical_winner_diagnostics,
            diagnostic_plan=replace(
                canonical_winner_diagnostics.diagnostic_plan,
                candidate_config=replace(
                    canonical_winner_diagnostics.diagnostic_plan.candidate_config,
                    capture_decision_traces=False,
                ),
            ),
        )
    elif evidence == "diagnostic_result":
        forged = replace(
            canonical_winner_diagnostics,
            diagnostic_result=replace(
                canonical_winner_diagnostics.diagnostic_result,
                game_summaries=canonical_winner_diagnostics.diagnostic_result.game_summaries[:-1],
            ),
        )
    else:
        forged = replace(
            canonical_winner_diagnostics,
            decision_report=replace(
                canonical_winner_diagnostics.decision_report,
                reconciliation=replace(
                    canonical_winner_diagnostics.decision_report.reconciliation,
                    selected_expert_decision_count=(
                        canonical_winner_diagnostics.decision_report.reconciliation.selected_expert_decision_count
                        + 1
                    ),
                ),
            ),
        )

    with pytest.raises(ValueError):
        _phase_report(
            frozen_phase_search_run,
            winner_diagnostics=forged,
        )


@pytest.mark.parametrize(
    "metadata",
    (
        "corpus_digest",
        "corpus_bool_root_seed",
        "candidate_spec",
        "incumbent",
        "opponents",
    ),
)
def test_phase_report_rejects_forged_diagnostic_plan_metadata(
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
    metadata: str,
) -> None:
    plan = canonical_winner_diagnostics.diagnostic_plan
    if metadata == "corpus_digest":
        forged_plan = replace(
            plan,
            corpus=replace(plan.corpus, digest="0" * 64),
        )
    elif metadata == "corpus_bool_root_seed":
        forged_plan = replace(
            plan,
            corpus=replace(
                plan.corpus,
                recipe=replace(
                    plan.corpus.recipe,
                    root_seed=cast(Any, True),
                ),
            ),
        )
    elif metadata == "candidate_spec":
        forged_plan = replace(
            plan,
            candidate=replace(
                plan.candidate,
                name=f"{plan.candidate.name}-forged",
            ),
        )
    elif metadata == "incumbent":
        forged_plan = replace(plan, incumbent=plan.opponents[0])
    else:
        forged_plan = replace(
            plan,
            opponents=(plan.incumbent, *plan.opponents[1:]),
        )
    forged = replace(
        canonical_winner_diagnostics,
        diagnostic_plan=forged_plan,
    )

    with pytest.raises(ValueError, match="may differ only by trace capture"):
        _phase_report(
            frozen_phase_search_run,
            winner_diagnostics=forged,
        )


def test_phase_winner_artifacts_retain_only_aggregate_diagnostics(
    tmp_path: Path,
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
) -> None:
    artifacts = write_search_artifacts(
        tmp_path,
        report=_phase_report(
            frozen_phase_search_run,
            winner_diagnostics=canonical_winner_diagnostics,
        ),
    )
    assert artifacts.winner_decision_slices_csv is not None
    assert artifacts.winner_diagnostics_json is not None
    assert artifacts.winner_diagnostics_markdown is not None
    retained = "\n".join(
        path.read_text()
        for path in (
            artifacts.winner_decision_slices_csv,
            artifacts.winner_diagnostics_json,
            artifacts.winner_diagnostics_markdown,
        )
    )

    assert all(
        sentinel not in retained
        for sentinel in (
            "decision_traces",
            "game_summaries",
            "candidate_jobs",
            "baseline_jobs",
            "root_seed",
            "engine_seed",
            "private_hand",
            "snapshot",
        )
    )


def test_nonfinite_evidence_fails_before_creating_output(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    bad_evaluation = replace(
        search_run.candidate_runs[0].evaluation,
        rating_delta=math.nan,
    )
    bad_run = replace(
        search_run,
        candidate_runs=(
            replace(search_run.candidate_runs[0], evaluation=bad_evaluation),
            *search_run.candidate_runs[1:],
        ),
        frozen_candidate=None,
    )

    with pytest.raises(ValueError, match="finite JSON"):
        write_search_artifacts(tmp_path, report=_report(bad_run))

    assert not any((tmp_path / name).exists() for name in _ALL_ARTIFACTS)


def test_output_preflight_requires_overwrite_for_nonempty_directory(
    tmp_path: Path,
) -> None:
    validate_search_output_dir(tmp_path / "missing")
    notes = tmp_path / "notes.txt"
    notes.write_text("keep me")
    with pytest.raises(FileExistsError, match="not empty"):
        validate_search_output_dir(tmp_path)
    validate_search_output_dir(tmp_path, overwrite=True)

    plain_file = tmp_path / "file"
    plain_file.write_text("plain")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_search_output_dir(plain_file)


def test_repository_commit_comes_from_git_rev_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ("git", "rev-parse", "HEAD")
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n")

    monkeypatch.setattr(
        "garboid_pocketrocks.evolution.reporting.subprocess.run",
        completed,
    )

    assert repository_commit() == "abc123"


def test_overwrite_preserves_unrelated_files_and_removes_stale_freeze(
    tmp_path: Path,
    search_run: SearchRun,
) -> None:
    write_search_artifacts(tmp_path, report=_report(search_run, frozen=True))
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")

    artifacts = write_search_artifacts(
        tmp_path,
        report=_report(search_run, frozen=False),
        overwrite=True,
    )

    assert artifacts.frozen_candidate_json is None
    assert not (tmp_path / "frozen-candidate.json").exists()
    assert unrelated.read_text() == "keep me"
    assert {path.name for path in tmp_path.iterdir()} == {
        *_REQUIRED_ARTIFACTS,
        "notes.txt",
    }


@pytest.mark.parametrize("failure_position", range(1, 8))
def test_replace_failure_rolls_back_replacement_and_stale_freeze_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    search_run: SearchRun,
    failure_position: int,
) -> None:
    write_search_artifacts(tmp_path, report=_report(search_run, frozen=True))
    previous = {name: (tmp_path / name).read_bytes() for name in _ALL_ARTIFACTS}
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")
    real_replace = os.replace
    operations = 0

    def fail_selected_operation(source: Any, destination: Any) -> None:
        nonlocal operations
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name in _REQUIRED_ARTIFACTS and ".backup." not in source_path.name:
            operations += 1
            if operations == failure_position:
                raise OSError(f"simulated failure {failure_position}")
        real_replace(source, destination)

    monkeypatch.setattr(
        "garboid_pocketrocks.evolution.reporting.os.replace",
        fail_selected_operation,
    )
    if failure_position == 7:
        real_unlink = Path.unlink

        def fail_stale_freeze_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name == "frozen-candidate.json" and ".backup." not in path.name:
                raise OSError("simulated failure 7")
            real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_stale_freeze_unlink)

    with pytest.raises(OSError, match=f"simulated failure {failure_position}"):
        write_search_artifacts(
            tmp_path,
            report=replace(_report(search_run), repository_commit=f"commit-{failure_position}"),
            overwrite=True,
        )

    assert {name: (tmp_path / name).read_bytes() for name in _ALL_ARTIFACTS} == previous
    assert unrelated.read_text() == "keep me"
    assert not tuple(path for path in tmp_path.iterdir() if path.name.startswith("."))


def test_phase_failed_overwrite_transactionally_removes_all_winner_artifacts(
    tmp_path: Path,
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
) -> None:
    write_search_artifacts(
        tmp_path,
        report=_phase_report(
            frozen_phase_search_run,
            winner_diagnostics=canonical_winner_diagnostics,
        ),
    )
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")
    failed_run = replace(
        frozen_phase_search_run,
        frozen_candidate=None,
        failures=(SearchFailure("diagnostic_failure", "Diagnostics failed closed."),),
    )

    artifacts = write_search_artifacts(
        tmp_path,
        report=_phase_report(failed_run),
        overwrite=True,
    )

    assert artifacts.frozen_candidate_json is None
    assert artifacts.winner_decision_slices_csv is None
    assert set(path.name for path in tmp_path.iterdir()) == {
        *_REQUIRED_ARTIFACTS,
        "notes.txt",
    }
    assert unrelated.read_text() == "keep me"


@pytest.mark.parametrize("failure_position", range(1, 11))
def test_phase_failed_overwrite_rolls_back_all_ten_artifact_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_phase_search_run: SearchRun,
    canonical_winner_diagnostics: WinnerDiagnostics,
    failure_position: int,
) -> None:
    winner_report = _phase_report(
        frozen_phase_search_run,
        winner_diagnostics=canonical_winner_diagnostics,
    )
    write_search_artifacts(tmp_path, report=winner_report)
    previous = {name: (tmp_path / name).read_bytes() for name in _ALL_PHASE_ARTIFACTS}
    failed_run = replace(
        frozen_phase_search_run,
        frozen_candidate=None,
        failures=(SearchFailure("diagnostic_failure", "Diagnostics failed closed."),),
    )
    failed_report = _phase_report(failed_run)
    real_replace = os.replace
    real_unlink = Path.unlink
    operations = 0

    def fail_selected_replace(source: Any, destination: Any) -> None:
        nonlocal operations
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path.parent == tmp_path
            and destination_path.name in _REQUIRED_ARTIFACTS
            and ".staged." in source_path.name
        ):
            operations += 1
            if operations == failure_position:
                raise OSError(f"simulated phase failure {failure_position}")
        real_replace(source, destination)

    def fail_selected_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal operations
        if path.parent == tmp_path and path.name in _ALL_PHASE_ARTIFACTS[6:]:
            operations += 1
            if operations == failure_position:
                raise OSError(f"simulated phase failure {failure_position}")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        "garboid_pocketrocks.evolution.reporting.os.replace",
        fail_selected_replace,
    )
    monkeypatch.setattr(Path, "unlink", fail_selected_unlink)

    with pytest.raises(
        OSError,
        match=f"simulated phase failure {failure_position}",
    ):
        write_search_artifacts(
            tmp_path,
            report=failed_report,
            overwrite=True,
        )

    assert {name: (tmp_path / name).read_bytes() for name in _ALL_PHASE_ARTIFACTS} == previous
    assert not tuple(path for path in tmp_path.iterdir() if path.name.startswith("."))
