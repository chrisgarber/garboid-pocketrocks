"""Validate matched promotion games and measure the candidate's advantage."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

import garboid_pocketrocks.tournament.rating as rating_module
from garboid_pocketrocks.knowledge import ruleset_name
from garboid_pocketrocks.promotion.planning import PromotionPlan
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloResult,
)
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.tournament.rating import (
    RankingObservation,
    TournamentRatingError,
    observations_from_games,
)

fit_plackett_luce = rating_module.fit_plackett_luce


@dataclass(frozen=True, slots=True)
class PromotionFailure:
    """One stable reason that a candidate cannot be promoted."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RatingDifferenceInterval:
    """A 95 percent uncertainty range for candidate minus incumbent rating."""

    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PromotionAnalysis:
    """The complete evidence and decision derived from one promotion run."""

    requested_pairs: int
    completed_pairs: int
    requested_games: int
    completed_games: int
    rating_difference: float | None
    interval: RatingDifferenceInterval | None
    bootstrap_requested: int
    bootstrap_converged: int
    unattributed_faults: int
    faults_by_identity: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]
    failures: tuple[PromotionFailure, ...]
    promoted: bool


@dataclass(frozen=True, slots=True)
class _ValidatedResults:
    pairs: tuple[tuple[GameSummary, GameSummary], ...]
    completed_games: int
    unattributed_faults: int
    faults_by_identity: tuple[tuple[str, int], ...]
    failures: tuple[PromotionFailure, ...]


_BOOTSTRAP_PAIR_OBSERVATIONS: tuple[tuple[RankingObservation, RankingObservation], ...] = ()
_BOOTSTRAP_BOT_IDS: tuple[str, ...] = ()
_BOOTSTRAP_CANDIDATE_ID = ""
_BOOTSTRAP_INCUMBENT_ID = ""
_BOOTSTRAP_ROOT_SEED = 0


def analyze_promotion(
    plan: PromotionPlan,
    result: MonteCarloResult,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    workers: int = 1,
) -> PromotionAnalysis:
    """Validate one matched run and fail closed unless its interval is positive."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if workers <= 0:
        raise ValueError("bootstrap workers must be positive")

    validated = _validate_results(plan, result)
    failures = list(validated.failures)
    if failures:
        return _analysis_from_results(
            plan,
            validated,
            rating_difference=None,
            interval=None,
            bootstrap_requested=bootstrap_samples,
            bootstrap_converged=0,
            warnings=(),
            failures=failures,
        )

    games = tuple(game for pair in validated.pairs for game in pair)
    bot_ids = tuple(spec.bot_id for spec in plan.monte_carlo_config.bot_specs)
    try:
        fit = fit_plackett_luce(observations_from_games(games), bot_ids)
    except TournamentRatingError as error:
        failures.append(
            PromotionFailure(
                "rating_fit_failed",
                f"The rating model could not fit the completed promotion games: {error}",
            )
        )
        return _analysis_from_results(
            plan,
            validated,
            rating_difference=None,
            interval=None,
            bootstrap_requested=bootstrap_samples,
            bootstrap_converged=0,
            warnings=(),
            failures=failures,
        )

    rating_difference = (
        fit.ratings_by_id[plan.candidate.bot_id].rating
        - fit.ratings_by_id[plan.incumbent.bot_id].rating
    )
    if not math.isfinite(rating_difference):
        failures.append(
            PromotionFailure(
                "nonfinite_analysis",
                "The fitted candidate-minus-incumbent rating was not a finite number.",
            )
        )
        return _analysis_from_results(
            plan,
            validated,
            rating_difference=None,
            interval=None,
            bootstrap_requested=bootstrap_samples,
            bootstrap_converged=0,
            warnings=(),
            failures=failures,
        )

    bootstrap_values, bootstrap_converged, warnings = bootstrap_paired_rating_differences(
        validated.pairs,
        bot_ids,
        candidate_id=plan.candidate.bot_id,
        incumbent_id=plan.incumbent.bot_id,
        samples=bootstrap_samples,
        root_seed=bootstrap_seed,
        workers=workers,
    )
    interval: RatingDifferenceInterval | None = None
    required_convergence = math.ceil(bootstrap_samples * 0.9)
    if bootstrap_converged < required_convergence:
        failures.append(
            PromotionFailure(
                "bootstrap_incomplete",
                f"Only {bootstrap_converged} of {bootstrap_samples} bootstrap fits "
                f"converged; at least {required_convergence} are required.",
            )
        )
    elif not all(math.isfinite(value) for value in bootstrap_values):
        failures.append(
            PromotionFailure(
                "nonfinite_analysis",
                "The bootstrap uncertainty range contained a nonfinite number.",
            )
        )
    else:
        lower = float(np.quantile(bootstrap_values, 0.025, method="linear"))
        upper = float(np.quantile(bootstrap_values, 0.975, method="linear"))
        if not math.isfinite(lower) or not math.isfinite(upper):
            failures.append(
                PromotionFailure(
                    "nonfinite_analysis",
                    "The bootstrap uncertainty range contained a nonfinite number.",
                )
            )
        else:
            interval = RatingDifferenceInterval(lower=lower, upper=upper)
            if lower <= 0.0:
                failures.append(
                    PromotionFailure(
                        "interval_includes_zero",
                        "The 95 percent uncertainty range does not show a reliably "
                        "positive candidate advantage.",
                    )
                )

    return _analysis_from_results(
        plan,
        validated,
        rating_difference=rating_difference,
        interval=interval,
        bootstrap_requested=bootstrap_samples,
        bootstrap_converged=bootstrap_converged,
        warnings=warnings,
        failures=failures,
    )


def bootstrap_paired_rating_differences(
    pairs: Sequence[tuple[GameSummary, GameSummary]],
    bot_ids: tuple[str, ...],
    *,
    candidate_id: str,
    incumbent_id: str,
    samples: int,
    root_seed: int,
    workers: int = 1,
) -> tuple[tuple[float, ...], int, tuple[str, ...]]:
    """Refit ratings after resampling complete candidate/incumbent game pairs."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if workers <= 0:
        raise ValueError("bootstrap workers must be positive")
    if not pairs:
        raise ValueError("paired bootstrap requires at least one completed pair")
    if not bot_ids or len(set(bot_ids)) != len(bot_ids):
        raise ValueError("bootstrap bot IDs must be nonempty and unique")
    if candidate_id == incumbent_id or candidate_id not in bot_ids or incumbent_id not in bot_ids:
        raise ValueError("candidate and incumbent must be distinct fitted bot IDs")

    pair_observations = tuple(
        (
            observations_from_games((candidate_game,))[0],
            observations_from_games((incumbent_game,))[0],
        )
        for candidate_game, incumbent_game in pairs
    )
    warnings: tuple[str, ...] = ()
    if workers == 1:
        replicate_results = tuple(
            _fit_bootstrap_replicate(
                pair_observations,
                bot_ids,
                candidate_id,
                incumbent_id,
                root_seed,
                replicate,
            )
            for replicate in range(samples)
        )
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers, samples),
                initializer=_initialize_bootstrap_worker,
                initargs=(
                    pair_observations,
                    bot_ids,
                    candidate_id,
                    incumbent_id,
                    root_seed,
                ),
            ) as executor:
                replicate_results = tuple(executor.map(_fit_bootstrap_worker, range(samples)))
        except Exception as error:
            warnings = (
                f"Parallel bootstrap failed with {type(error).__name__}; "
                "all samples were retried serially.",
            )
            replicate_results = tuple(
                _fit_bootstrap_replicate(
                    pair_observations,
                    bot_ids,
                    candidate_id,
                    incumbent_id,
                    root_seed,
                    replicate,
                )
                for replicate in range(samples)
            )

    values = tuple(value for value in replicate_results if value is not None)
    converged = len(values)
    if converged < samples:
        warnings += (
            f"{samples - converged} of {samples} bootstrap fits did not converge "
            "and were excluded.",
        )
    return values, converged, warnings


def _validate_results(
    plan: PromotionPlan,
    result: MonteCarloResult,
) -> _ValidatedResults:
    expected_by_index = {job.game_index: job for job in plan.jobs}
    summaries_by_index: defaultdict[int, list[GameSummary]] = defaultdict(list)
    for summary in result.game_summaries:
        summaries_by_index[summary.game_index].append(summary)

    failures: list[PromotionFailure] = []
    exact_by_index: dict[int, GameSummary] = {}
    faults: defaultdict[str, int] = defaultdict(int)
    unattributed_faults = 0
    saw_any_fault = any(
        count != 0 for summary in result.game_summaries for count in summary.fault_counts
    )
    for game_index, summaries in summaries_by_index.items():
        fault_count_for_summaries = sum(
            abs(count) for summary in summaries for count in summary.fault_counts
        )
        if game_index not in expected_by_index:
            unattributed_faults += fault_count_for_summaries
            failures.append(
                PromotionFailure(
                    "unexpected_game",
                    f"The result includes unexpected game index {game_index}.",
                )
            )
            continue
        if len(summaries) != 1:
            unattributed_faults += fault_count_for_summaries
            failures.append(
                PromotionFailure(
                    "unexpected_game",
                    f"Expected one result for game index {game_index}, "
                    f"but received {len(summaries)}.",
                )
            )
            continue
        summary = summaries[0]
        job = expected_by_index[game_index]
        summary_failures = _summary_failures(job, summary)
        failures.extend(summary_failures)
        if _has_trusted_seat_identities(job, summary):
            for bot_id, count in zip(
                summary.bot_ids,
                summary.fault_counts,
                strict=True,
            ):
                faults[bot_id] += abs(count)
        else:
            unattributed_faults += fault_count_for_summaries
        if not summary_failures:
            exact_by_index[game_index] = summary

    for game_index in expected_by_index.keys() - summaries_by_index.keys():
        failures.append(
            PromotionFailure(
                "missing_paired_game",
                f"The result is missing planned game index {game_index}.",
            )
        )

    nonzero_faults = tuple(sorted((bot_id, count) for bot_id, count in faults.items() if count))
    if saw_any_fault:
        failures.append(
            PromotionFailure(
                "bot_fault",
                "At least one bot faulted during the promotion games; "
                "a faulted run cannot promote a candidate.",
            )
        )

    validated_pairs: list[tuple[GameSummary, GameSummary]] = []
    for pair in plan.pairs:
        candidate_summary = exact_by_index.get(pair.candidate_game.game_index)
        incumbent_summary = exact_by_index.get(pair.incumbent_game.game_index)
        if candidate_summary is not None and incumbent_summary is not None:
            validated_pairs.append((candidate_summary, incumbent_summary))

    return _ValidatedResults(
        pairs=tuple(validated_pairs),
        completed_games=len(exact_by_index),
        unattributed_faults=unattributed_faults,
        faults_by_identity=nonzero_faults,
        failures=_ordered_failures(failures),
    )


def _has_trusted_seat_identities(
    job: GameJob,
    summary: GameSummary,
) -> bool:
    """Return whether each reported fault can be assigned to its planned bot."""

    return (
        summary.bot_names == tuple(spec.name for spec in job.lineup)
        and summary.bot_ids == tuple(spec.bot_id for spec in job.lineup)
        and len(summary.fault_counts) == job.player_count
    )


def _summary_failures(
    job: GameJob,
    summary: GameSummary,
) -> tuple[PromotionFailure, ...]:
    failures: list[PromotionFailure] = []
    game_label = f"Game index {job.game_index}"
    if summary.root_seed != job.root_seed or summary.seed != job.seed:
        failures.append(
            PromotionFailure(
                "seed_mismatch",
                f"{game_label} does not use its planned root and engine seeds.",
            )
        )
    expected_ruleset = ruleset_name(job.value_chart, job.objectives_enabled)
    if summary.ruleset_name != expected_ruleset:
        failures.append(
            PromotionFailure(
                "ruleset_mismatch",
                f"{game_label} reports ruleset {summary.ruleset_name!r}, "
                f"not planned ruleset {expected_ruleset!r}.",
            )
        )
    expected_names = tuple(spec.name for spec in job.lineup)
    expected_ids = tuple(spec.bot_id for spec in job.lineup)
    if summary.bot_names != expected_names or summary.bot_ids != expected_ids:
        failures.append(
            PromotionFailure(
                "identity_mismatch",
                f"{game_label} does not preserve the planned bot identity at every seat.",
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
            PromotionFailure(
                "player_count_mismatch",
                f"{game_label} does not contain exactly one complete record "
                f"for each of its {job.player_count} planned seats.",
            )
        )
    return _ordered_failures(failures)


def _initialize_bootstrap_worker(
    pair_observations: tuple[tuple[RankingObservation, RankingObservation], ...],
    bot_ids: tuple[str, ...],
    candidate_id: str,
    incumbent_id: str,
    root_seed: int,
) -> None:
    global _BOOTSTRAP_PAIR_OBSERVATIONS
    global _BOOTSTRAP_BOT_IDS
    global _BOOTSTRAP_CANDIDATE_ID
    global _BOOTSTRAP_INCUMBENT_ID
    global _BOOTSTRAP_ROOT_SEED
    _BOOTSTRAP_PAIR_OBSERVATIONS = pair_observations
    _BOOTSTRAP_BOT_IDS = bot_ids
    _BOOTSTRAP_CANDIDATE_ID = candidate_id
    _BOOTSTRAP_INCUMBENT_ID = incumbent_id
    _BOOTSTRAP_ROOT_SEED = root_seed


def _fit_bootstrap_worker(replicate: int) -> float | None:
    if not _BOOTSTRAP_PAIR_OBSERVATIONS or not _BOOTSTRAP_BOT_IDS:
        raise RuntimeError("promotion bootstrap worker was not initialized")
    return _fit_bootstrap_replicate(
        _BOOTSTRAP_PAIR_OBSERVATIONS,
        _BOOTSTRAP_BOT_IDS,
        _BOOTSTRAP_CANDIDATE_ID,
        _BOOTSTRAP_INCUMBENT_ID,
        _BOOTSTRAP_ROOT_SEED,
        replicate,
    )


def _fit_bootstrap_replicate(
    pair_observations: tuple[tuple[RankingObservation, RankingObservation], ...],
    bot_ids: tuple[str, ...],
    candidate_id: str,
    incumbent_id: str,
    root_seed: int,
    replicate: int,
) -> float | None:
    rng = random.Random(derive_seed(root_seed, "promotion-bootstrap", replicate))
    selected_pair_indices = tuple(rng.randrange(len(pair_observations)) for _ in pair_observations)
    observations = tuple(
        observation
        for pair_index in selected_pair_indices
        for observation in pair_observations[pair_index]
    )
    try:
        fit = fit_plackett_luce(observations, bot_ids)
    except TournamentRatingError:
        return None
    return fit.ratings_by_id[candidate_id].rating - fit.ratings_by_id[incumbent_id].rating


def _ordered_failures(
    failures: Sequence[PromotionFailure],
) -> tuple[PromotionFailure, ...]:
    return tuple(
        sorted(
            set(failures),
            key=lambda failure: (failure.code, failure.message),
        )
    )


def _analysis_from_results(
    plan: PromotionPlan,
    validated: _ValidatedResults,
    *,
    rating_difference: float | None,
    interval: RatingDifferenceInterval | None,
    bootstrap_requested: int,
    bootstrap_converged: int,
    warnings: tuple[str, ...],
    failures: Sequence[PromotionFailure],
) -> PromotionAnalysis:
    ordered_failures = _ordered_failures(failures)
    promoted = not ordered_failures and interval is not None and interval.lower > 0.0
    return PromotionAnalysis(
        requested_pairs=len(plan.pairs),
        completed_pairs=len(validated.pairs),
        requested_games=len(plan.jobs),
        completed_games=validated.completed_games,
        rating_difference=rating_difference,
        interval=interval,
        bootstrap_requested=bootstrap_requested,
        bootstrap_converged=bootstrap_converged,
        unattributed_faults=validated.unattributed_faults,
        faults_by_identity=validated.faults_by_identity,
        warnings=warnings,
        failures=ordered_failures,
        promoted=promoted,
    )
