"""Validate development evidence and score one heuristic candidate."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import garboid_pocketrocks.tournament.rating as rating_module
from garboid_pocketrocks.evolution.planning import DevelopmentPlan
from garboid_pocketrocks.knowledge import ruleset_name
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloResult,
)
from garboid_pocketrocks.tournament.rating import (
    TournamentRatingError,
    observations_from_games,
)

fit_plackett_luce = rating_module.fit_plackett_luce


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One stable evidence or eligibility failure."""

    code: str
    message: str
    invalidates_run: bool


@dataclass(frozen=True, slots=True)
class ChallengerFinishDelta:
    """Matched focal-finish improvement in lineups containing one challenger."""

    opponent_identity: str
    shared_cases: int
    normalized_finish_delta: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Complete validated development fitness for one candidate identity."""

    candidate_identity: str
    incumbent_identity: str
    requested_cases: int
    completed_baseline_games: int
    completed_candidate_games: int
    rating_delta: float | None
    normalized_finish_delta: float | None
    final_money_delta: int | None
    worst_challenger_finish_delta: float | None
    challenger_finish_deltas: tuple[ChallengerFinishDelta, ...]
    candidate_faults: int
    incumbent_faults: int
    opponent_faults: int
    unattributed_faults: int
    faults_by_identity: tuple[tuple[str, int], ...]
    failures: tuple[EvaluationFailure, ...]
    valid: bool
    eligible: bool


@dataclass(frozen=True, slots=True)
class _ValidatedEvidence:
    summaries: tuple[GameSummary, ...]
    completed_games: int
    candidate_faults: int
    incumbent_faults: int
    opponent_faults: int
    unattributed_faults: int
    faults_by_identity: tuple[tuple[str, int], ...]
    failures: tuple[EvaluationFailure, ...]


def evaluate_candidate(
    plan: DevelopmentPlan,
    baseline_result: MonteCarloResult,
    candidate_result: MonteCarloResult,
) -> CandidateEvaluation:
    """Validate matched evidence and compute candidate-minus-incumbent fitness."""

    baseline = _validate_evidence(
        plan,
        plan.baseline_jobs,
        baseline_result,
        evidence_name="baseline",
    )
    candidate = _validate_evidence(
        plan,
        plan.candidate_jobs,
        candidate_result,
        evidence_name="candidate",
    )
    failures = list(baseline.failures) + list(candidate.failures)
    candidate_faults = baseline.candidate_faults + candidate.candidate_faults
    incumbent_faults = baseline.incumbent_faults + candidate.incumbent_faults
    opponent_faults = baseline.opponent_faults + candidate.opponent_faults
    unattributed_faults = baseline.unattributed_faults + candidate.unattributed_faults
    faults_by_identity = _merge_faults(
        baseline.faults_by_identity,
        candidate.faults_by_identity,
    )

    if candidate_faults:
        failures.append(
            EvaluationFailure(
                "candidate_fault",
                f"Candidate {plan.candidate.name!r} faulted {candidate_faults} time(s) "
                "and is ineligible for selection.",
                False,
            )
        )
    if incumbent_faults:
        failures.append(
            EvaluationFailure(
                "incumbent_fault",
                f"Incumbent {plan.incumbent.name!r} faulted {incumbent_faults} time(s); "
                "the reusable baseline is invalid.",
                True,
            )
        )
    if opponent_faults:
        failures.append(
            EvaluationFailure(
                "opponent_fault",
                f"Development opponents faulted {opponent_faults} time(s); "
                "the matched evidence is invalid.",
                True,
            )
        )
    if unattributed_faults:
        failures.append(
            EvaluationFailure(
                "unattributed_fault",
                f"{unattributed_faults} fault(s) could not be assigned to a planned identity.",
                True,
            )
        )

    rating_delta: float | None = None
    normalized_finish_delta: float | None = None
    final_money_delta: int | None = None
    worst_challenger_finish_delta: float | None = None
    challenger_finish_deltas: tuple[ChallengerFinishDelta, ...] = ()
    if not any(failure.invalidates_run for failure in failures):
        (
            rating_delta,
            normalized_finish_delta,
            final_money_delta,
            score_failure,
        ) = _score_evidence(plan, baseline.summaries, candidate.summaries)
        if score_failure is not None:
            failures.append(score_failure)
        else:
            challenger_finish_deltas = _challenger_finish_deltas(
                plan,
                baseline.summaries,
                candidate.summaries,
            )
            if not challenger_finish_deltas:
                failures.append(
                    EvaluationFailure(
                        "missing_challenger_coverage",
                        "Development evidence did not contain any configured challenger.",
                        True,
                    )
                )
            else:
                worst_challenger_finish_delta = min(
                    item.normalized_finish_delta for item in challenger_finish_deltas
                )

    ordered_failures = _ordered_failures(failures)
    valid = not any(failure.invalidates_run for failure in ordered_failures)
    if not valid:
        rating_delta = None
        normalized_finish_delta = None
        final_money_delta = None
        worst_challenger_finish_delta = None
        challenger_finish_deltas = ()
    return CandidateEvaluation(
        candidate_identity=plan.candidate.name,
        incumbent_identity=plan.incumbent.name,
        requested_cases=len(plan.corpus.cases),
        completed_baseline_games=baseline.completed_games,
        completed_candidate_games=candidate.completed_games,
        rating_delta=rating_delta,
        normalized_finish_delta=normalized_finish_delta,
        final_money_delta=final_money_delta,
        worst_challenger_finish_delta=worst_challenger_finish_delta,
        challenger_finish_deltas=challenger_finish_deltas,
        candidate_faults=candidate_faults,
        incumbent_faults=incumbent_faults,
        opponent_faults=opponent_faults,
        unattributed_faults=unattributed_faults,
        faults_by_identity=faults_by_identity,
        failures=ordered_failures,
        valid=valid,
        eligible=valid and candidate_faults == 0,
    )


def _validate_evidence(
    plan: DevelopmentPlan,
    jobs: tuple[GameJob, ...],
    result: MonteCarloResult,
    *,
    evidence_name: str,
) -> _ValidatedEvidence:
    expected_by_index = {job.game_index: job for job in jobs}
    summaries_by_index: defaultdict[int, list[GameSummary]] = defaultdict(list)
    for summary in result.game_summaries:
        summaries_by_index[summary.game_index].append(summary)

    failures: list[EvaluationFailure] = []
    exact_by_index: dict[int, GameSummary] = {}
    faults: defaultdict[str, int] = defaultdict(int)
    candidate_faults = 0
    incumbent_faults = 0
    opponent_faults = 0
    unattributed_faults = 0
    for game_index, summaries in summaries_by_index.items():
        reported_faults = sum(
            _fault_magnitude(count) for summary in summaries for count in summary.fault_counts
        )
        job = expected_by_index.get(game_index)
        if job is None:
            unattributed_faults += reported_faults
            failures.append(
                EvaluationFailure(
                    f"unexpected_{evidence_name}_game",
                    f"The {evidence_name} result includes unexpected game index {game_index}.",
                    True,
                )
            )
            continue
        if len(summaries) != 1:
            unattributed_faults += reported_faults
            failures.append(
                EvaluationFailure(
                    f"duplicate_{evidence_name}_game",
                    f"Expected one {evidence_name} result for game index {game_index}, "
                    f"but received {len(summaries)}.",
                    True,
                )
            )
            continue

        summary = summaries[0]
        summary_failures = _summary_failures(job, summary, evidence_name=evidence_name)
        failures.extend(summary_failures)
        if _has_trusted_seat_identities(job, summary):
            case = plan.corpus.cases[game_index]
            for seat, (bot_id, count) in enumerate(
                zip(summary.bot_ids, summary.fault_counts, strict=True)
            ):
                magnitude = _fault_magnitude(count)
                faults[bot_id] += magnitude
                if not magnitude:
                    continue
                if seat == case.focal_seat:
                    if evidence_name == "candidate":
                        candidate_faults += magnitude
                    else:
                        incumbent_faults += magnitude
                else:
                    opponent_faults += magnitude
        else:
            unattributed_faults += reported_faults
        if not summary_failures:
            exact_by_index[game_index] = summary

    missing_code = f"missing_{evidence_name}_game"
    for game_index in expected_by_index.keys() - summaries_by_index.keys():
        failures.append(
            EvaluationFailure(
                missing_code,
                f"The result is missing planned {evidence_name} game index {game_index}.",
                True,
            )
        )

    return _ValidatedEvidence(
        summaries=tuple(exact_by_index[index] for index in sorted(exact_by_index)),
        completed_games=len(exact_by_index),
        candidate_faults=candidate_faults,
        incumbent_faults=incumbent_faults,
        opponent_faults=opponent_faults,
        unattributed_faults=unattributed_faults,
        faults_by_identity=tuple(
            sorted((bot_id, count) for bot_id, count in faults.items() if count)
        ),
        failures=_ordered_failures(failures),
    )


def _summary_failures(
    job: GameJob,
    summary: GameSummary,
    *,
    evidence_name: str,
) -> tuple[EvaluationFailure, ...]:
    failures: list[EvaluationFailure] = []
    label = f"{evidence_name.capitalize()} game index {job.game_index}"
    if summary.root_seed != job.root_seed or summary.seed != job.seed:
        failures.append(
            EvaluationFailure(
                "seed_mismatch",
                f"{label} does not preserve its planned root and engine seeds.",
                True,
            )
        )
    expected_ruleset = ruleset_name(job.value_chart, job.objectives_enabled)
    if summary.ruleset_name != expected_ruleset:
        failures.append(
            EvaluationFailure(
                "ruleset_mismatch",
                f"{label} reports ruleset {summary.ruleset_name!r}, "
                f"not planned ruleset {expected_ruleset!r}.",
                True,
            )
        )
    expected_names = tuple(spec.name for spec in job.lineup)
    expected_ids = tuple(spec.bot_id for spec in job.lineup)
    if summary.bot_names != expected_names or summary.bot_ids != expected_ids:
        failures.append(
            EvaluationFailure(
                "identity_mismatch",
                f"{label} does not preserve the planned identity at every seat.",
                True,
            )
        )

    expected_seats = set(range(job.player_count))
    score_seats = {score.seat for score in summary.scores}
    if (
        summary.player_count != job.player_count
        or len(summary.bot_names) != job.player_count
        or len(summary.bot_ids) != job.player_count
        or len(summary.scores) != job.player_count
        or score_seats != expected_seats
        or len(summary.decision_counts) != job.player_count
        or len(summary.fault_counts) != job.player_count
    ):
        failures.append(
            EvaluationFailure(
                "player_count_mismatch",
                f"{label} does not contain one complete record for every planned seat.",
                True,
            )
        )
    elif (
        any(
            isinstance(score.rank, bool)
            or not isinstance(score.rank, int)
            or not 1 <= score.rank <= job.player_count
            or isinstance(score.final_money, bool)
            or not isinstance(score.final_money, int)
            for score in summary.scores
        )
        or not any(score.rank == 1 for score in summary.scores)
        or not _ranks_match_final_money(summary)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in (*summary.decision_counts, *summary.fault_counts)
        )
    ):
        failures.append(
            EvaluationFailure(
                "invalid_game_evidence",
                f"{label} contains invalid scores, decisions, or fault counts.",
                True,
            )
        )
    return _ordered_failures(failures)


def _has_trusted_seat_identities(job: GameJob, summary: GameSummary) -> bool:
    return (
        summary.bot_names == tuple(spec.name for spec in job.lineup)
        and summary.bot_ids == tuple(spec.bot_id for spec in job.lineup)
        and len(summary.fault_counts) == job.player_count
    )


def _score_evidence(
    plan: DevelopmentPlan,
    baseline: tuple[GameSummary, ...],
    candidate: tuple[GameSummary, ...],
) -> tuple[float | None, float | None, int | None, EvaluationFailure | None]:
    games = (*baseline, *candidate)
    bot_ids = (
        plan.candidate.bot_id,
        plan.incumbent.bot_id,
        *(opponent.bot_id for opponent in plan.opponents),
    )
    try:
        observations = observations_from_games(games)
    except ValueError as error:
        return (
            None,
            None,
            None,
            EvaluationFailure(
                "rating_observation_failed",
                f"Development games could not form valid ranking observations: {error}",
                True,
            ),
        )
    try:
        fit = fit_plackett_luce(observations, bot_ids)
    except TournamentRatingError as error:
        return (
            None,
            None,
            None,
            EvaluationFailure(
                "rating_fit_failed",
                f"The rating model could not fit complete development evidence: {error}",
                True,
            ),
        )

    rating_delta = (
        fit.ratings_by_id[plan.candidate.bot_id].rating
        - fit.ratings_by_id[plan.incumbent.bot_id].rating
    )
    baseline_finish, baseline_money = _focal_totals(plan, baseline)
    candidate_finish, candidate_money = _focal_totals(plan, candidate)
    normalized_finish_delta = candidate_finish - baseline_finish
    final_money_delta = candidate_money - baseline_money
    if not math.isfinite(rating_delta) or not math.isfinite(normalized_finish_delta):
        return (
            None,
            None,
            None,
            EvaluationFailure(
                "nonfinite_evaluation",
                "Development fitness contained a nonfinite score.",
                True,
            ),
        )
    return rating_delta, normalized_finish_delta, final_money_delta, None


def _ranks_match_final_money(summary: GameSummary) -> bool:
    return all(
        score.rank
        == 1
        + sum(
            other.final_money > score.final_money
            for other in summary.scores
            if other.seat != score.seat
        )
        for score in summary.scores
    )


def _focal_totals(
    plan: DevelopmentPlan,
    summaries: tuple[GameSummary, ...],
) -> tuple[float, int]:
    normalized_finishes: list[float] = []
    final_money = 0
    for case, summary in zip(plan.corpus.cases, summaries, strict=True):
        scores_by_seat = {score.seat: score for score in summary.scores}
        score = scores_by_seat[case.focal_seat]
        normalized_finishes.append((case.player_count - score.rank) / (case.player_count - 1))
        final_money += score.final_money
    return math.fsum(normalized_finishes), final_money


def _challenger_finish_deltas(
    plan: DevelopmentPlan,
    baseline: tuple[GameSummary, ...],
    candidate: tuple[GameSummary, ...],
) -> tuple[ChallengerFinishDelta, ...]:
    """Measure matched focal improvement for every challenger lineup slice."""

    deltas_by_identity: defaultdict[str, list[float]] = defaultdict(list)
    for case, baseline_summary, candidate_summary, candidate_job in zip(
        plan.corpus.cases,
        baseline,
        candidate,
        plan.candidate_jobs,
        strict=True,
    ):
        baseline_scores = {score.seat: score for score in baseline_summary.scores}
        candidate_scores = {score.seat: score for score in candidate_summary.scores}
        denominator = case.player_count - 1
        baseline_finish = (case.player_count - baseline_scores[case.focal_seat].rank) / denominator
        candidate_finish = (
            case.player_count - candidate_scores[case.focal_seat].rank
        ) / denominator
        matched_delta = candidate_finish - baseline_finish
        for seat, opponent in enumerate(candidate_job.lineup):
            if seat != case.focal_seat:
                deltas_by_identity[opponent.bot_id].append(matched_delta)

    results = tuple(
        ChallengerFinishDelta(
            opponent_identity=identity,
            shared_cases=len(deltas),
            normalized_finish_delta=math.fsum(deltas) / len(deltas),
        )
        for identity, deltas in sorted(deltas_by_identity.items())
        if deltas
    )
    if not all(
        math.isfinite(item.normalized_finish_delta) and item.shared_cases > 0 for item in results
    ):
        raise ValueError("challenger finish deltas must be finite and covered")
    return results


def _merge_faults(
    *collections: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    merged: defaultdict[str, int] = defaultdict(int)
    for collection in collections:
        for bot_id, count in collection:
            merged[bot_id] += count
    return tuple(sorted(merged.items()))


def _fault_magnitude(count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, int):
        return 0
    return abs(count)


def _ordered_failures(
    failures: Sequence[EvaluationFailure],
) -> tuple[EvaluationFailure, ...]:
    return tuple(
        sorted(
            set(failures),
            key=lambda failure: (
                failure.code,
                failure.message,
                failure.invalidates_run,
            ),
        )
    )
