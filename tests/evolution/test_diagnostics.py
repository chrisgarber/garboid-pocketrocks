from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import pytest

import garboid_pocketrocks.evolution.diagnostics as diagnostics_module
import garboid_pocketrocks.evolution.reporting as reporting_module
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.diagnostics.analysis import DecisionReport
from garboid_pocketrocks.evolution.diagnostics import (
    WinnerDiagnostics,
    WinnerDiagnosticsError,
    run_winner_diagnostics,
)
from garboid_pocketrocks.evolution.runner import SearchRun
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)

from .test_runner import (
    _phase_run_with_recomputed_positive_evidence,
    _run_small_phase_search,
    _small_phase_inputs,
)


@pytest.fixture(scope="module")
def frozen_phase_run() -> SearchRun:
    manifest, corpus = _small_phase_inputs(generations=1, cases=3)
    run = _run_small_phase_search(
        manifest,
        corpus,
        workers=1,
        batch_size=1,
    )
    return _phase_run_with_recomputed_positive_evidence(run)


@contextmanager
def _tiny_diagnostics_contract() -> Iterator[None]:
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
        yield


def _run_tiny_diagnostics(
    run: SearchRun,
    *,
    workers: int = 1,
    batch_size: int | None = 1,
) -> WinnerDiagnostics:
    with _tiny_diagnostics_contract():
        return run_winner_diagnostics(
            run,
            registry=BOT_SPECS_BY_NAME,
            workers=workers,
            batch_size=batch_size,
        )


def test_winner_diagnostics_rejects_a_run_without_a_frozen_winner() -> None:
    manifest, corpus = _small_phase_inputs(generations=1, cases=1)
    run = _run_small_phase_search(
        manifest,
        corpus,
        workers=1,
        batch_size=1,
    )

    with pytest.raises(WinnerDiagnosticsError, match="complete frozen") as captured:
        run_winner_diagnostics(
            run,
            registry=BOT_SPECS_BY_NAME,
            workers=1,
            batch_size=1,
        )

    assert captured.value.code == "incomplete_frozen_run"


def test_traces_exactly_one_winner_candidate_run_and_builds_one_report(
    frozen_phase_run: SearchRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run_jobs = MonteCarloRunner.run_jobs
    real_build_report = diagnostics_module._build_winner_decision_report
    calls: list[tuple[MonteCarloConfig, tuple[GameJob, ...], int, int | None]] = []
    report_calls = 0

    def record_run_jobs(
        config: MonteCarloConfig,
        jobs: tuple[GameJob, ...],
        *,
        workers: int,
        batch_size: int | None,
    ) -> MonteCarloResult:
        calls.append((config, jobs, workers, batch_size))
        return real_run_jobs(
            config,
            jobs,
            workers=workers,
            batch_size=batch_size,
        )

    def record_report(result: MonteCarloResult) -> DecisionReport:
        nonlocal report_calls
        report_calls += 1
        return real_build_report(result)

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", record_run_jobs)
    monkeypatch.setattr(
        diagnostics_module,
        "_build_winner_decision_report",
        record_report,
    )

    diagnostics = _run_tiny_diagnostics(frozen_phase_run)

    assert len(calls) == 1
    config, jobs, workers, batch_size = calls[0]
    winner = frozen_phase_run.frozen_candidate
    assert winner is not None
    assert workers == 1
    assert batch_size == 1
    assert config.capture_decision_traces is True
    assert len(jobs) == len(frozen_phase_run.development_corpus.cases)
    assert all(job.capture_decision_traces is True for job in jobs)
    assert all(
        job.lineup[frozen_phase_run.development_corpus.cases[index].focal_seat].name
        == winner.identity
        for index, job in enumerate(jobs)
    )
    assert all(
        "candidate-" not in spec.name or spec.name == winner.identity for spec in config.bot_specs
    )
    assert report_calls == 1
    assert diagnostics.decision_report.schema_version == 2
    assert diagnostics.decision_report.reconciliation.game_count == len(jobs)
    assert diagnostics.decision_report.reconciliation.selected_expert_decision_count > 0
    assert {
        outcome.selected_expert_phase for outcome in diagnostics.decision_report.phase_outcomes
    } == {"early", "middle", "late"}


def test_serial_batched_and_worker_diagnostics_are_byte_identical(
    frozen_phase_run: SearchRun,
) -> None:
    serial = _run_tiny_diagnostics(
        frozen_phase_run,
        workers=1,
        batch_size=None,
    )
    batched = _run_tiny_diagnostics(
        frozen_phase_run,
        workers=1,
        batch_size=1,
    )
    worker = _run_tiny_diagnostics(
        frozen_phase_run,
        workers=2,
        batch_size=1,
    )

    assert serial.named_contents() == batched.named_contents() == worker.named_contents()
    assert serial.artifact_digests == batched.artifact_digests == worker.artifact_digests
    assert serial.decision_report == batched.decision_report == worker.decision_report


def test_rejects_winner_plan_tampering_before_trace_execution(
    frozen_phase_run: SearchRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner = frozen_phase_run.frozen_candidate
    assert winner is not None
    winner_run = next(item for item in frozen_phase_run.candidate_runs if item.candidate == winner)
    job = winner_run.plan.candidate_jobs[0]
    forged_plan = replace(
        winner_run.plan,
        candidate_jobs=(
            replace(job, seed=job.seed + 1),
            *winner_run.plan.candidate_jobs[1:],
        ),
    )
    forged = replace(
        frozen_phase_run,
        candidate_runs=tuple(
            replace(item, plan=forged_plan) if item.candidate == winner else item
            for item in frozen_phase_run.candidate_runs
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("trace simulation started with a forged winner plan")

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", forbidden)

    with _tiny_diagnostics_contract():
        with pytest.raises(WinnerDiagnosticsError) as captured:
            run_winner_diagnostics(
                forged,
                registry=BOT_SPECS_BY_NAME,
                workers=1,
                batch_size=1,
            )

    assert captured.value.code == "invalid_frozen_run"


def test_rejects_traced_result_that_changes_ordinary_winner_evidence(
    frozen_phase_run: SearchRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner = frozen_phase_run.frozen_candidate
    assert winner is not None
    winner_result = next(
        item.result for item in frozen_phase_run.candidate_runs if item.candidate == winner
    )
    monkeypatch.setattr(
        MonteCarloRunner,
        "run_jobs",
        lambda *args, **kwargs: replace(
            winner_result,
            game_summaries=winner_result.game_summaries[:-1],
        ),
    )

    with _tiny_diagnostics_contract():
        with pytest.raises(WinnerDiagnosticsError) as captured:
            run_winner_diagnostics(
                frozen_phase_run,
                registry=BOT_SPECS_BY_NAME,
                workers=1,
                batch_size=1,
            )

    assert captured.value.code == "diagnostic_result_mismatch"


def test_operational_error_detail_is_chained_but_not_exposed(
    frozen_phase_run: SearchRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_message = "PRIVATE_SENTINEL hidden worker detail"

    def fail_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(private_message)

    monkeypatch.setattr(MonteCarloRunner, "run_jobs", fail_run)

    with _tiny_diagnostics_contract():
        with pytest.raises(WinnerDiagnosticsError) as captured:
            run_winner_diagnostics(
                frozen_phase_run,
                registry=BOT_SPECS_BY_NAME,
                workers=1,
                batch_size=1,
            )

    assert captured.value.code == "diagnostic_execution_failed"
    assert private_message not in str(captured.value)
    assert captured.value.__cause__ is not None
    assert private_message in str(captured.value.__cause__)


def test_retained_diagnostics_are_aggregate_only_and_digest_bound(
    frozen_phase_run: SearchRun,
) -> None:
    diagnostics = _run_tiny_diagnostics(frozen_phase_run)
    retained = "\n".join(content for _, content in diagnostics.named_contents())
    forbidden_private_sentinels = (
        "decision_traces",
        "game_summaries",
        "candidate_jobs",
        "baseline_jobs",
        "root_seed",
        "engine_seed",
        "private_hand",
        "snapshot",
    )

    assert all(sentinel not in retained for sentinel in forbidden_private_sentinels)
    assert tuple(name for name, _ in diagnostics.named_contents()) == (
        "winner-decision-slices.csv",
        "winner-diagnostics.json",
        "winner-diagnostics.md",
    )
    assert diagnostics.digests_by_name == dict(diagnostics.artifact_digests)
    assert len(diagnostics.digests_by_name) == 3
