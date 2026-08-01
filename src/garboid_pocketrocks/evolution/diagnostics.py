"""Run and retain privacy-safe diagnostics for one frozen phase-aware winner."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.diagnostics.analysis import (
    DecisionReport,
    build_decision_report,
)
from garboid_pocketrocks.diagnostics.trace import (
    PhaseAwareHeuristicBidExplanation,
)
from garboid_pocketrocks.evolution.candidates import (
    PhaseAwareHeuristicCandidate,
    candidate_bot_spec,
)
from garboid_pocketrocks.evolution.manifest import PhaseSearchManifest
from garboid_pocketrocks.evolution.planning import (
    DevelopmentPlan,
    plan_development_games,
)
from garboid_pocketrocks.evolution.runner import CandidateRun, SearchRun
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.tournament.analysis import analyze_tournament
from garboid_pocketrocks.tournament.rating import (
    fit_plackett_luce,
    observations_from_games,
)

_MIN_SAFE_CONTRIBUTING_GAMES = 30
_PHASE_OUTCOME_FIELDS = (
    "selected_expert_phase",
    "contributing_game_count",
    "decision_count",
    "eventual_final_money_sum",
    "eventual_normalized_finish_sum",
    "outright_win_decision_count",
    "tied_first_decision_count",
    "decisions_from_faulted_game_seat",
)
WINNER_DECISION_SLICES_NAME = "winner-decision-slices.csv"
WINNER_DIAGNOSTICS_JSON_NAME = "winner-diagnostics.json"
WINNER_DIAGNOSTICS_MARKDOWN_NAME = "winner-diagnostics.md"
WINNER_DIAGNOSTIC_NAMES = (
    WINNER_DECISION_SLICES_NAME,
    WINNER_DIAGNOSTICS_JSON_NAME,
    WINNER_DIAGNOSTICS_MARKDOWN_NAME,
)


class WinnerDiagnosticsError(ValueError):
    """Explain why winner-only diagnostic evidence cannot be trusted."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WinnerDiagnostics:
    """Validated in-memory diagnostics and the three retained aggregate artifacts."""

    winner_identity: str
    diagnostic_plan: DevelopmentPlan
    diagnostic_result: MonteCarloResult
    decision_report: DecisionReport
    decision_slices_csv: str
    diagnostics_json: str
    diagnostics_markdown: str
    artifact_digests: tuple[tuple[str, str], ...]

    def named_contents(self) -> tuple[tuple[str, str], ...]:
        """Return retained aggregate artifacts in canonical order."""

        return (
            (WINNER_DECISION_SLICES_NAME, self.decision_slices_csv),
            (WINNER_DIAGNOSTICS_JSON_NAME, self.diagnostics_json),
            (WINNER_DIAGNOSTICS_MARKDOWN_NAME, self.diagnostics_markdown),
        )

    @property
    def digests_by_name(self) -> dict[str, str]:
        """Return retained artifact content digests by canonical name."""

        return dict(self.artifact_digests)


def run_winner_diagnostics(
    run: SearchRun,
    *,
    registry: Mapping[str, BotSpec],
    workers: int,
    batch_size: int | None,
) -> WinnerDiagnostics:
    """Trace only the frozen schema-v2 winner on its exact candidate jobs."""

    if workers <= 0 or (batch_size is not None and batch_size <= 0):
        raise WinnerDiagnosticsError(
            "invalid_diagnostics_execution",
            "winner diagnostics execution controls must be positive",
        )
    if (
        not isinstance(run.manifest, PhaseSearchManifest)
        or not isinstance(run.frozen_candidate, PhaseAwareHeuristicCandidate)
        or run.selected_candidate != run.frozen_candidate
        or run.failures
    ):
        raise WinnerDiagnosticsError(
            "incomplete_frozen_run",
            "winner diagnostics require one complete frozen schema-v2 search run",
        )
    _validate_winner_diagnostics_fixed_contract(run)
    _validate_registry(run, registry=registry)
    winner_run = _winner_candidate_run(run)
    rebuilt_plan = plan_development_games(
        run.development_corpus,
        candidate=candidate_bot_spec(run.frozen_candidate),
        incumbent=run.incumbent,
        registry=BOT_SPECS_BY_NAME,
    )
    from garboid_pocketrocks.evolution.reporting import _plan_evidence_matches

    if not _plan_evidence_matches(winner_run.plan, rebuilt_plan):
        raise WinnerDiagnosticsError(
            "winner_plan_mismatch",
            "frozen winner plan does not match the report-bound development plan",
        )
    if winner_run.result.decision_traces:
        raise WinnerDiagnosticsError(
            "unexpected_search_traces",
            "search evidence must not already contain decision traces",
        )

    diagnostic_plan = _traced_candidate_plan(rebuilt_plan)
    _validate_trace_only_plan(rebuilt_plan, diagnostic_plan)
    try:
        diagnostic_result = MonteCarloRunner.run_jobs(
            diagnostic_plan.candidate_config,
            diagnostic_plan.candidate_jobs,
            workers=workers,
            batch_size=batch_size,
        )
    except Exception as error:
        raise WinnerDiagnosticsError(
            "diagnostic_execution_failed",
            "winner diagnostic simulation failed",
        ) from error
    if replace(diagnostic_result, decision_traces=(), game_details=()) != winner_run.result:
        raise WinnerDiagnosticsError(
            "diagnostic_result_mismatch",
            "winner diagnostics changed the winner's ordinary simulation result",
        )
    try:
        decision_report = _build_winner_decision_report(diagnostic_result)
        _validate_winner_decision_report(
            decision_report,
            expected_games=len(diagnostic_plan.candidate_jobs),
        )
    except WinnerDiagnosticsError:
        raise
    except Exception as error:
        raise WinnerDiagnosticsError(
            "decision_report_failed",
            "winner decision report generation failed",
        ) from error
    contents = _retained_diagnostic_contents(
        run,
        decision_report=decision_report,
    )
    return WinnerDiagnostics(
        winner_identity=run.frozen_candidate.identity,
        diagnostic_plan=diagnostic_plan,
        diagnostic_result=diagnostic_result,
        decision_report=decision_report,
        decision_slices_csv=contents[0][1],
        diagnostics_json=contents[1][1],
        diagnostics_markdown=contents[2][1],
        artifact_digests=tuple((name, _sha256(content)) for name, content in contents),
    )


def validate_winner_diagnostics_evidence(
    run: SearchRun,
    diagnostics: WinnerDiagnostics,
    *,
    registry: Mapping[str, BotSpec],
) -> None:
    """Recompute every report-bound diagnostic value without rerunning games."""

    _validate_winner_diagnostics_fixed_contract(run)
    _validate_registry(run, registry=registry)
    winner_run = _winner_candidate_run(run)
    assert run.frozen_candidate is not None
    rebuilt_plan = plan_development_games(
        run.development_corpus,
        candidate=candidate_bot_spec(run.frozen_candidate),
        incumbent=run.incumbent,
        registry=BOT_SPECS_BY_NAME,
    )
    from garboid_pocketrocks.evolution.reporting import _plan_evidence_matches

    if not _plan_evidence_matches(winner_run.plan, rebuilt_plan):
        raise WinnerDiagnosticsError(
            "winner_plan_mismatch",
            "frozen winner plan does not match the report-bound development plan",
        )
    _validate_trace_only_plan(rebuilt_plan, diagnostics.diagnostic_plan)
    if (
        diagnostics.winner_identity != run.frozen_candidate.identity
        or replace(
            diagnostics.diagnostic_result,
            decision_traces=(),
            game_details=(),
        )
        != winner_run.result
    ):
        raise WinnerDiagnosticsError(
            "diagnostic_result_mismatch",
            "winner diagnostics do not reproduce the frozen winner result",
        )
    try:
        canonical_report = _build_winner_decision_report(diagnostics.diagnostic_result)
        _validate_winner_decision_report(
            canonical_report,
            expected_games=len(rebuilt_plan.candidate_jobs),
        )
        canonical_contents = _retained_diagnostic_contents(
            run,
            decision_report=canonical_report,
        )
    except WinnerDiagnosticsError:
        raise
    except Exception as error:
        raise WinnerDiagnosticsError(
            "decision_report_failed",
            "winner decision report generation failed",
        ) from error
    canonical_digests = tuple((name, _sha256(content)) for name, content in canonical_contents)
    if (
        diagnostics.decision_report != canonical_report
        or diagnostics.named_contents() != canonical_contents
        or diagnostics.artifact_digests != canonical_digests
    ):
        raise WinnerDiagnosticsError(
            "diagnostic_artifact_mismatch",
            "winner diagnostic report and retained artifacts are not canonical",
        )


def _validate_winner_diagnostics_fixed_contract(run: SearchRun) -> None:
    from garboid_pocketrocks.evolution.reporting import (
        _validate_phase_candidates,
        _validate_phase_freeze,
        _validate_phase_report_contract,
    )

    try:
        _validate_phase_report_contract(run)
        _validate_phase_candidates(run)
        _validate_phase_freeze(run)
        _require_exact_winner_case_count(run)
    except WinnerDiagnosticsError:
        raise
    except ValueError as error:
        raise WinnerDiagnosticsError(
            "invalid_frozen_run",
            "winner diagnostics rejected the frozen search evidence",
        ) from error


def _require_exact_winner_case_count(run: SearchRun) -> None:
    if len(run.development_corpus.cases) != 240:
        raise WinnerDiagnosticsError(
            "wrong_diagnostics_case_count",
            "winner diagnostics require the exact 240-case development corpus",
        )


def _validate_registry(
    run: SearchRun,
    *,
    registry: Mapping[str, BotSpec],
) -> None:
    names = (
        run.manifest.predecessor_name,
        *run.development_corpus.recipe.opponent_names,
    )
    if any(registry.get(name) is not BOT_SPECS_BY_NAME.get(name) for name in names):
        raise WinnerDiagnosticsError(
            "noncanonical_diagnostics_registry",
            "winner diagnostics require canonical incumbent and opponent specs",
        )


def _winner_candidate_run(run: SearchRun) -> CandidateRun:
    assert run.frozen_candidate is not None
    matches = tuple(item for item in run.candidate_runs if item.candidate == run.frozen_candidate)
    if len(matches) != 1:
        raise WinnerDiagnosticsError(
            "winner_run_count_mismatch",
            "frozen winner must have exactly one candidate plan and result",
        )
    return matches[0]


def _traced_candidate_plan(plan: DevelopmentPlan) -> DevelopmentPlan:
    return replace(
        plan,
        candidate_config=replace(
            plan.candidate_config,
            capture_decision_traces=True,
        ),
        candidate_jobs=tuple(
            replace(job, capture_decision_traces=True) for job in plan.candidate_jobs
        ),
    )


def _validate_trace_only_plan(
    ordinary: DevelopmentPlan,
    diagnostic: DevelopmentPlan,
) -> None:
    from garboid_pocketrocks.evolution.reporting import (
        _bot_spec_matches,
        _plan_evidence_matches,
    )

    stripped = replace(
        diagnostic,
        candidate_config=replace(
            diagnostic.candidate_config,
            capture_decision_traces=False,
        ),
        candidate_jobs=tuple(
            replace(job, capture_decision_traces=False) for job in diagnostic.candidate_jobs
        ),
    )
    if (
        type(diagnostic.candidate_config.capture_decision_traces) is not bool
        or diagnostic.candidate_config.capture_decision_traces is not True
        or any(
            type(job.capture_decision_traces) is not bool or job.capture_decision_traces is not True
            for job in diagnostic.candidate_jobs
        )
        or diagnostic.corpus is not ordinary.corpus
        or not _bot_spec_matches(diagnostic.candidate, ordinary.candidate)
        or diagnostic.incumbent is not ordinary.incumbent
        or len(diagnostic.opponents) != len(ordinary.opponents)
        or any(
            actual is not expected
            for actual, expected in zip(
                diagnostic.opponents,
                ordinary.opponents,
                strict=True,
            )
        )
        or not _plan_evidence_matches(stripped, ordinary)
    ):
        raise WinnerDiagnosticsError(
            "diagnostic_plan_mismatch",
            "winner diagnostic config and jobs may differ only by trace capture",
        )


def _build_winner_decision_report(result: MonteCarloResult) -> DecisionReport:
    bot_ids = tuple(
        dict.fromkeys(bot_id for summary in result.game_summaries for bot_id in summary.bot_ids)
    )
    observed_bot_ids = set(bot_ids)
    bot_statistics = tuple(
        statistic
        for statistic in result.bot_statistics
        if statistic.bot_id in observed_bot_ids
    )
    fit = fit_plackett_luce(
        observations_from_games(result.game_summaries),
        bot_ids,
    )
    analysis = analyze_tournament(result, fit)
    return build_decision_report(
        result.decision_traces,
        game_summaries=result.game_summaries,
        bot_statistics=bot_statistics,
        tournament_analysis=analysis,
    )


def _validate_winner_decision_report(
    report: DecisionReport,
    *,
    expected_games: int,
) -> None:
    reconciliation = report.reconciliation
    phase_counts = {
        outcome.selected_expert_phase: outcome.decision_count for outcome in report.phase_outcomes
    }
    if (
        report.schema_version != 2
        or reconciliation.game_count != expected_games
        or reconciliation.trace_decision_count != reconciliation.game_summary_decision_count
        or reconciliation.trace_decision_count != reconciliation.slice_decision_count
        or reconciliation.selected_expert_decision_count <= 0
        or set(phase_counts) != {"early", "middle", "late"}
        or any(count <= 0 for count in phase_counts.values())
    ):
        raise WinnerDiagnosticsError(
            "invalid_decision_reconciliation",
            "winner decision diagnostics require full schema-v2 reconciliation "
            "and nonzero early, middle, and late selections",
        )


def _diagnostics_payload(
    run: SearchRun,
    *,
    decision_report: DecisionReport,
    contributing_game_counts: Mapping[str, int],
) -> dict[str, object]:
    reconciliation = decision_report.reconciliation
    return {
        "schema_version": 3,
        "aggregation": {
            "unit": "selected_expert_phase",
            "minimum_contributing_games": _MIN_SAFE_CONTRIBUTING_GAMES,
        },
        "winner_identity": run.frozen_candidate.identity
        if run.frozen_candidate is not None
        else None,
        "manifest_digest": run.manifest.digest,
        "development_corpus": {
            "name": run.development_corpus.recipe.name,
            "digest": run.development_corpus.digest,
            "cases": len(run.development_corpus.cases),
        },
        "reconciliation": {
            "game_count": reconciliation.game_count,
            "game_seat_count": reconciliation.game_seat_count,
            "trace_decision_count": reconciliation.trace_decision_count,
            "game_summary_decision_count": reconciliation.game_summary_decision_count,
            "slice_decision_count": reconciliation.slice_decision_count,
            "selected_expert_decision_count": (reconciliation.selected_expert_decision_count),
        },
        "phase_outcomes": [
            {
                "selected_expert_phase": outcome.selected_expert_phase,
                "contributing_game_count": contributing_game_counts[outcome.selected_expert_phase],
                "decision_count": outcome.decision_count,
                "eventual_final_money_sum": outcome.eventual_final_money_sum,
                "eventual_normalized_finish_sum": (outcome.eventual_normalized_finish_sum),
                "outright_win_decision_count": outcome.outright_win_decision_count,
                "tied_first_decision_count": outcome.tied_first_decision_count,
                "decisions_from_faulted_game_seat": (outcome.decisions_from_faulted_game_seat),
            }
            for outcome in decision_report.phase_outcomes
        ],
    }


def _retained_diagnostic_contents(
    run: SearchRun,
    *,
    decision_report: DecisionReport,
) -> tuple[tuple[str, str], ...]:
    contributing_game_counts = _phase_contributing_game_counts(decision_report)
    phase_outcomes_csv = _render_phase_outcomes_csv(
        decision_report,
        contributing_game_counts=contributing_game_counts,
    )
    summary_json = (
        json.dumps(
            _diagnostics_payload(
                run,
                decision_report=decision_report,
                contributing_game_counts=contributing_game_counts,
            ),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return (
        # The external filename is retained for frozen-provenance compatibility.
        (WINNER_DECISION_SLICES_NAME, phase_outcomes_csv),
        (WINNER_DIAGNOSTICS_JSON_NAME, summary_json),
        (
            WINNER_DIAGNOSTICS_MARKDOWN_NAME,
            _diagnostics_markdown(
                run,
                decision_report=decision_report,
                contributing_game_counts=contributing_game_counts,
            ),
        ),
    )


def _phase_contributing_game_counts(
    decision_report: DecisionReport,
) -> dict[str, int]:
    game_indexes_by_phase: dict[str, set[int]] = {
        outcome.selected_expert_phase: set() for outcome in decision_report.phase_outcomes
    }
    for trace in decision_report.decision_traces:
        explanation = trace.explanation
        if not isinstance(explanation, PhaseAwareHeuristicBidExplanation):
            continue
        if trace.game_index is None:
            raise WinnerDiagnosticsError(
                "invalid_decision_reconciliation",
                "phase-aware winner traces must identify their contributing game",
            )
        game_indexes_by_phase[explanation.selected_expert_phase].add(trace.game_index)

    contributing_game_counts = {
        phase: len(game_indexes) for phase, game_indexes in game_indexes_by_phase.items()
    }
    unsafe_phases = tuple(
        phase
        for phase, count in contributing_game_counts.items()
        if count < _MIN_SAFE_CONTRIBUTING_GAMES
    )
    if unsafe_phases:
        raise WinnerDiagnosticsError(
            "unsafe_diagnostic_aggregation",
            "winner diagnostic phase outcomes require at least "
            f"{_MIN_SAFE_CONTRIBUTING_GAMES} contributing games per phase",
        )
    return contributing_game_counts


def _render_phase_outcomes_csv(
    decision_report: DecisionReport,
    *,
    contributing_game_counts: Mapping[str, int],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_PHASE_OUTCOME_FIELDS)
    writer.writeheader()
    for outcome in decision_report.phase_outcomes:
        writer.writerow(
            {
                "selected_expert_phase": outcome.selected_expert_phase,
                "contributing_game_count": contributing_game_counts[outcome.selected_expert_phase],
                "decision_count": outcome.decision_count,
                "eventual_final_money_sum": outcome.eventual_final_money_sum,
                "eventual_normalized_finish_sum": (outcome.eventual_normalized_finish_sum),
                "outright_win_decision_count": outcome.outright_win_decision_count,
                "tied_first_decision_count": outcome.tied_first_decision_count,
                "decisions_from_faulted_game_seat": (outcome.decisions_from_faulted_game_seat),
            }
        )
    return stream.getvalue()


def _diagnostics_markdown(
    run: SearchRun,
    *,
    decision_report: DecisionReport,
    contributing_game_counts: Mapping[str, int],
) -> str:
    reconciliation = decision_report.reconciliation
    rows = "\n".join(
        "| "
        f"{outcome.selected_expert_phase} | "
        f"{contributing_game_counts[outcome.selected_expert_phase]} | "
        f"{outcome.decision_count} | "
        f"{outcome.eventual_final_money_sum} | "
        f"{outcome.eventual_normalized_finish_sum:.12g} | "
        f"{outcome.outright_win_decision_count} | "
        f"{outcome.tied_first_decision_count} | "
        f"{outcome.decisions_from_faulted_game_seat} |"
        for outcome in decision_report.phase_outcomes
    )
    assert run.frozen_candidate is not None
    return (
        "# Winner diagnostics\n\n"
        f"- Winner: `{run.frozen_candidate.identity}`\n"
        f"- Development corpus: `{run.development_corpus.recipe.name}` "
        f"(`{run.development_corpus.digest}`)\n"
        f"- Games: {reconciliation.game_count}\n"
        f"- Reconciled decisions: {reconciliation.trace_decision_count}\n"
        f"- Expert selections: {reconciliation.selected_expert_decision_count}\n\n"
        "| Expert phase | Contributing games | Decisions | Final-money sum | "
        "Normalized-finish sum | "
        "Wins | Tied first | Faulted-seat decisions |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"{rows}\n"
    )


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
