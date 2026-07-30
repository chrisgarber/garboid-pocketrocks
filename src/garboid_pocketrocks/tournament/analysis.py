from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from garboid_pocketrocks.simulator.monte_carlo import (
    GameSummary,
    MonteCarloResult,
)
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.tournament.rating import (
    PlackettLuceFit,
    RankingObservation,
    TournamentRatingError,
    fit_plackett_luce,
    observations_from_games,
)


@dataclass(frozen=True, slots=True)
class RatingInterval:
    bot_id: str
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    requested: int
    converged: int
    intervals: tuple[RatingInterval, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TournamentBotRow:
    rank: int
    bot_name: str
    bot_id: str
    worth: float
    log_worth: float
    pl_rating: float
    games: int
    outright_wins: int
    first_place_ties: int
    mean_normalized_finish: float
    mean_final_money: float
    mean_winning_money: float | None
    faults: int


@dataclass(frozen=True, slots=True)
class ConditionStatistics:
    chart: str
    player_count: int
    bot_id: str
    games: int
    outright_wins: int
    first_place_ties: int
    mean_normalized_finish: float
    mean_final_money: float


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower_probability: float
    upper_probability: float
    count: int
    mean_prediction: float
    observed_score: float


@dataclass(frozen=True, slots=True)
class TournamentAnalysis:
    rows: tuple[TournamentBotRow, ...]
    condition_statistics: tuple[ConditionStatistics, ...]
    calibration: tuple[CalibrationBin, ...]
    pair_outcomes: int

    @property
    def rows_by_id(self) -> dict[str, TournamentBotRow]:
        return {row.bot_id: row for row in self.rows}


@dataclass(slots=True)
class _Accumulator:
    games: int = 0
    outright_wins: int = 0
    first_place_ties: int = 0
    normalized_finishes: list[float] = field(default_factory=list)
    final_money: list[int] = field(default_factory=list)
    winning_money: list[int] = field(default_factory=list)
    faults: int = 0

    def record(
        self,
        *,
        player_count: int,
        rank: int,
        final_money: int,
        first_place_count: int,
        faults: int,
    ) -> None:
        self.games += 1
        self.outright_wins += int(rank == 1 and first_place_count == 1)
        self.first_place_ties += int(rank == 1 and first_place_count > 1)
        self.normalized_finishes.append((player_count - rank) / (player_count - 1))
        self.final_money.append(final_money)
        if rank == 1:
            self.winning_money.append(final_money)
        self.faults += faults


_BOOTSTRAP_OBSERVATIONS: tuple[RankingObservation, ...] = ()
_BOOTSTRAP_BOT_IDS: tuple[str, ...] = ()
_BOOTSTRAP_ROOT_SEED = 0


def analyze_tournament(
    result: MonteCarloResult,
    fit: PlackettLuceFit,
) -> TournamentAnalysis:
    rating_by_id = fit.ratings_by_id
    totals: dict[str, _Accumulator] = defaultdict(_Accumulator)
    conditions: dict[tuple[str, int, str], _Accumulator] = defaultdict(_Accumulator)
    names: dict[str, str] = {}
    calibration_samples: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    pair_outcomes = 0

    for game in result.game_summaries:
        scores_by_seat = {score.seat: score for score in game.scores}
        first_place_count = sum(score.rank == 1 for score in game.scores)
        chart = game.ruleset_name.removeprefix("live-")
        for seat, (bot_id, bot_name) in enumerate(zip(game.bot_ids, game.bot_names, strict=True)):
            names.setdefault(bot_id, bot_name)
            score = scores_by_seat[seat]
            record_options = {
                "player_count": game.player_count,
                "rank": score.rank,
                "final_money": score.final_money,
                "first_place_count": first_place_count,
                "faults": game.fault_counts[seat],
            }
            totals[bot_id].record(**record_options)
            conditions[chart, game.player_count, bot_id].record(**record_options)

        for first_seat in range(game.player_count):
            for second_seat in range(first_seat + 1, game.player_count):
                first_id = game.bot_ids[first_seat]
                second_id = game.bot_ids[second_seat]
                if second_id < first_id:
                    first_id, second_id = second_id, first_id
                    first_seat_oriented, second_seat_oriented = second_seat, first_seat
                else:
                    first_seat_oriented, second_seat_oriented = first_seat, second_seat
                first_score = scores_by_seat[first_seat_oriented]
                second_score = scores_by_seat[second_seat_oriented]
                if first_score.rank < second_score.rank:
                    observed = 1.0
                elif first_score.rank > second_score.rank:
                    observed = 0.0
                else:
                    observed = 0.5
                first_worth = rating_by_id[first_id].worth
                second_worth = rating_by_id[second_id].worth
                predicted = first_worth / (first_worth + second_worth)
                bin_index = min(int(predicted * 10), 9)
                calibration_samples[bin_index].append((predicted, observed))
                pair_outcomes += 1

    missing = set(rating_by_id) - set(totals)
    if missing:
        raise TournamentRatingError(
            f"fitted bots have no tournament games: {', '.join(sorted(missing))}"
        )

    rows = tuple(
        TournamentBotRow(
            rank=rank,
            bot_name=names[rating.bot_id],
            bot_id=rating.bot_id,
            worth=rating.worth,
            log_worth=rating.log_worth,
            pl_rating=rating.rating,
            games=totals[rating.bot_id].games,
            outright_wins=totals[rating.bot_id].outright_wins,
            first_place_ties=totals[rating.bot_id].first_place_ties,
            mean_normalized_finish=float(
                statistics.mean(totals[rating.bot_id].normalized_finishes)
            ),
            mean_final_money=float(statistics.mean(totals[rating.bot_id].final_money)),
            mean_winning_money=(
                float(statistics.mean(totals[rating.bot_id].winning_money))
                if totals[rating.bot_id].winning_money
                else None
            ),
            faults=totals[rating.bot_id].faults,
        )
        for rank, rating in enumerate(fit.ratings, start=1)
    )
    condition_statistics = tuple(
        ConditionStatistics(
            chart=chart,
            player_count=player_count,
            bot_id=bot_id,
            games=accumulator.games,
            outright_wins=accumulator.outright_wins,
            first_place_ties=accumulator.first_place_ties,
            mean_normalized_finish=float(statistics.mean(accumulator.normalized_finishes)),
            mean_final_money=float(statistics.mean(accumulator.final_money)),
        )
        for (chart, player_count, bot_id), accumulator in sorted(conditions.items())
    )
    calibration = tuple(
        CalibrationBin(
            lower_probability=index / 10,
            upper_probability=(index + 1) / 10,
            count=len(samples),
            mean_prediction=float(statistics.mean(item[0] for item in samples)),
            observed_score=float(statistics.mean(item[1] for item in samples)),
        )
        for index, samples in enumerate(calibration_samples)
        if samples
    )
    return TournamentAnalysis(
        rows=rows,
        condition_statistics=condition_statistics,
        calibration=calibration,
        pair_outcomes=pair_outcomes,
    )


def bootstrap_rating_intervals(
    games: tuple[GameSummary, ...],
    bot_ids: tuple[str, ...],
    *,
    samples: int,
    root_seed: int,
    workers: int = 1,
) -> BootstrapSummary:
    if samples < 0:
        raise ValueError("bootstrap samples must be nonnegative")
    if workers <= 0:
        raise ValueError("bootstrap workers must be positive")
    if not bot_ids or len(set(bot_ids)) != len(bot_ids):
        raise ValueError("bootstrap bot IDs must be nonempty and unique")
    if samples == 0:
        return BootstrapSummary(0, 0, (), ())
    if not games:
        raise ValueError("bootstrap requires at least one game")

    observations = observations_from_games(games)
    ratings_by_bot: dict[str, list[float]] = {bot_id: [] for bot_id in bot_ids}
    converged = 0
    execution_warnings: tuple[str, ...] = ()
    if workers == 1:
        replicate_results = tuple(
            _fit_bootstrap_replicate(observations, bot_ids, root_seed, replicate)
            for replicate in range(samples)
        )
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers, samples),
                initializer=_initialize_bootstrap_worker,
                initargs=(observations, bot_ids, root_seed),
            ) as executor:
                replicate_results = tuple(executor.map(_fit_bootstrap_worker, range(samples)))
        except Exception as error:
            execution_warnings = (
                f"parallel bootstrap failed with {type(error).__name__}; retried serially",
            )
            replicate_results = tuple(
                _fit_bootstrap_replicate(observations, bot_ids, root_seed, replicate)
                for replicate in range(samples)
            )
    for ratings in replicate_results:
        if ratings is None:
            continue
        for bot_id, rating in ratings:
            ratings_by_bot[bot_id].append(rating)
        converged += 1

    required = math.ceil(samples * 0.9)
    if converged < required:
        return BootstrapSummary(
            requested=samples,
            converged=converged,
            intervals=(),
            warnings=execution_warnings
            + (
                f"only {converged} of {samples} bootstrap fits converged; "
                "confidence intervals are unavailable",
            ),
        )
    intervals = tuple(
        RatingInterval(
            bot_id=bot_id,
            lower=float(np.quantile(ratings_by_bot[bot_id], 0.025, method="linear")),
            upper=float(np.quantile(ratings_by_bot[bot_id], 0.975, method="linear")),
        )
        for bot_id in bot_ids
    )
    convergence_warnings = (
        (f"{samples - converged} of {samples} bootstrap fits failed and were excluded",)
        if converged < samples
        else ()
    )
    return BootstrapSummary(
        requested=samples,
        converged=converged,
        intervals=intervals,
        warnings=execution_warnings + convergence_warnings,
    )


def _initialize_bootstrap_worker(
    observations: tuple[RankingObservation, ...],
    bot_ids: tuple[str, ...],
    root_seed: int,
) -> None:
    global _BOOTSTRAP_OBSERVATIONS, _BOOTSTRAP_BOT_IDS, _BOOTSTRAP_ROOT_SEED
    _BOOTSTRAP_OBSERVATIONS = observations
    _BOOTSTRAP_BOT_IDS = bot_ids
    _BOOTSTRAP_ROOT_SEED = root_seed


def _fit_bootstrap_worker(replicate: int) -> tuple[tuple[str, float], ...] | None:
    if not _BOOTSTRAP_OBSERVATIONS or not _BOOTSTRAP_BOT_IDS:
        raise RuntimeError("bootstrap worker was not initialized")
    return _fit_bootstrap_replicate(
        _BOOTSTRAP_OBSERVATIONS,
        _BOOTSTRAP_BOT_IDS,
        _BOOTSTRAP_ROOT_SEED,
        replicate,
    )


def _fit_bootstrap_replicate(
    observations: tuple[RankingObservation, ...],
    bot_ids: tuple[str, ...],
    root_seed: int,
    replicate: int,
) -> tuple[tuple[str, float], ...] | None:
    rng = random.Random(derive_seed(root_seed, "bootstrap", replicate))
    counts: dict[int, int] = {}
    for _ in observations:
        index = rng.randrange(len(observations))
        counts[index] = counts.get(index, 0) + 1
    resampled = tuple(
        RankingObservation(
            observations[index].rank_groups,
            weight=observations[index].weight * count,
        )
        for index, count in counts.items()
    )
    try:
        fit = fit_plackett_luce(resampled, bot_ids)
    except TournamentRatingError:
        return None
    return tuple((rating.bot_id, rating.rating) for rating in fit.ratings)
