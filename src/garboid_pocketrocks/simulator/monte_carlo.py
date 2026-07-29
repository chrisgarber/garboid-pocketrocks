from __future__ import annotations

import importlib
import pickle
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.rules import Ruleset
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.model import Score
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.runner import (
    FaultMode,
    MatchResult,
    MatchRunner,
)
from garboid_pocketrocks.simulator.sampling import (
    RulesetSampler,
    derive_seed,
)


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    bot_specs: tuple[BotSpec, ...]
    games: int
    player_counts: tuple[int, ...]
    ruleset_sampler: RulesetSampler
    root_seed: int
    fault_mode: FaultMode = FaultMode.RAISE
    capture_replays: bool = False

    def __post_init__(self) -> None:
        if self.games < 0:
            raise ValueError("games must be nonnegative")
        if not self.player_counts:
            raise ValueError("player_counts must be nonempty")
        if not self.bot_specs:
            raise ValueError("bot_specs must be nonempty")
        if any(player_count < 3 or player_count > 5 for player_count in self.player_counts):
            raise ValueError("player counts must be between 3 and 5")
        if len(self.bot_specs) < max(self.player_counts):
            raise ValueError("not enough bot specs for the requested player counts")
        support = self.ruleset_sampler.support()
        if not support:
            raise ValueError("ruleset sampler support must be nonempty")
        rulesets_by_name: dict[str, Ruleset] = {}
        for ruleset in support:
            existing = rulesets_by_name.get(ruleset.name)
            if existing is not None and existing != ruleset:
                raise ValueError(f"different rulesets must not use the same name {ruleset.name!r}")
            rulesets_by_name[ruleset.name] = ruleset
            for player_count in self.player_counts:
                ruleset.setup_for(player_count)


@dataclass(frozen=True, slots=True)
class GameJob:
    game_index: int
    root_seed: int
    seed: int
    player_count: int
    ruleset: Ruleset
    lineup: tuple[BotSpec, ...]
    fault_mode: FaultMode


@dataclass(frozen=True, slots=True)
class GameSummary:
    game_index: int
    root_seed: int
    seed: int
    player_count: int
    ruleset_name: str
    bot_names: tuple[str, ...]
    bot_ids: tuple[str, ...]
    scores: tuple[Score, ...]
    decision_counts: tuple[int, ...]
    fault_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SeatStatistics:
    seat: int
    games: int
    outright_wins: int
    first_place_ties: int
    rank_counts: tuple[int, ...]
    final_money_samples: tuple[int, ...]
    score_margins: tuple[int, ...]
    decision_counts: tuple[int, ...]
    faults: int

    @property
    def decision_count(self) -> int:
        return sum(self.decision_counts)


@dataclass(frozen=True, slots=True)
class RulesetStatistics:
    ruleset_name: str
    games: int
    outright_wins: int
    first_place_ties: int
    rank_counts: tuple[int, ...]
    final_money_samples: tuple[int, ...]
    score_margins: tuple[int, ...]
    decision_counts: tuple[int, ...]
    faults: int

    @property
    def decision_count(self) -> int:
        return sum(self.decision_counts)


@dataclass(frozen=True, slots=True)
class BotStatistics:
    bot_name: str
    bot_id: str
    games: int
    outright_wins: int
    first_place_ties: int
    rank_counts: tuple[int, ...]
    final_money_samples: tuple[int, ...]
    score_margins: tuple[int, ...]
    per_seat: tuple[SeatStatistics, ...]
    per_ruleset: tuple[RulesetStatistics, ...]
    decision_counts: tuple[int, ...]
    faults: int

    @property
    def decision_count(self) -> int:
        return sum(self.decision_counts)

    def mean_final_money(self) -> float:
        return self.mean()

    def median_final_money(self) -> float:
        return self.median()

    def final_money_population_spread(self) -> float:
        return self.population_spread()

    def final_money_quantile(self, probability: float) -> float:
        return self.quantile(probability)

    def mean(self) -> float:
        if not self.final_money_samples:
            return 0.0
        return float(statistics.mean(self.final_money_samples))

    def median(self) -> float:
        if not self.final_money_samples:
            return 0.0
        return float(statistics.median(self.final_money_samples))

    def population_spread(self) -> float:
        if not self.final_money_samples:
            return 0.0
        return float(statistics.pstdev(self.final_money_samples))

    def mean_rank(self) -> float:
        if not self.games:
            return 0.0
        rank_total = sum(rank * count for rank, count in enumerate(self.rank_counts, start=1))
        return rank_total / self.games

    def quantile(self, probability: float) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("quantile probability must be between zero and one")
        if not self.final_money_samples:
            return 0.0
        ordered = sorted(self.final_money_samples)
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction))

    def quantiles(
        self,
        probabilities: tuple[float, ...] = (0.25, 0.5, 0.75),
    ) -> tuple[float, ...]:
        return tuple(self.quantile(probability) for probability in probabilities)


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    game_summaries: tuple[GameSummary, ...]
    bot_statistics: tuple[BotStatistics, ...]
    replays: tuple[MatchReplay, ...]

    @property
    def games(self) -> tuple[GameSummary, ...]:
        return self.game_summaries

    @property
    def statistics(self) -> tuple[BotStatistics, ...]:
        return self.bot_statistics


@dataclass(frozen=True, slots=True)
class _CompletedGame:
    job: GameJob
    match: MatchResult


@dataclass(slots=True)
class _StatisticsAccumulator:
    games: int = 0
    outright_wins: int = 0
    first_place_ties: int = 0
    rank_counts: list[int] = field(default_factory=list)
    final_money_samples: list[int] = field(default_factory=list)
    score_margins: list[int] = field(default_factory=list)
    decision_counts: list[int] = field(default_factory=list)
    faults: int = 0

    @classmethod
    def with_rank_count(cls, rank_count: int) -> _StatisticsAccumulator:
        return cls(rank_counts=[0] * rank_count)

    def record(
        self,
        *,
        score: Score,
        outright_win: bool,
        first_place_tie: bool,
        score_margin: int,
        decision_count: int,
        faults: int,
    ) -> None:
        self.games += 1
        self.outright_wins += int(outright_win)
        self.first_place_ties += int(first_place_tie)
        self.rank_counts[score.rank - 1] += 1
        self.final_money_samples.append(score.final_money)
        self.score_margins.append(score_margin)
        self.decision_counts.append(decision_count)
        self.faults += faults


class MonteCarloRunner:
    @staticmethod
    def plan(config: MonteCarloConfig) -> tuple[GameJob, ...]:
        base_lineups: dict[int, tuple[BotSpec, ...]] = {}
        for player_count in set(config.player_counts):
            if len(config.bot_specs) == player_count:
                shuffled = list(config.bot_specs)
                random.Random(derive_seed(config.root_seed, "lineup", player_count)).shuffle(
                    shuffled
                )
                base_lineups[player_count] = tuple(shuffled)

        jobs: list[GameJob] = []
        for game_index in range(config.games):
            player_count = random.Random(
                derive_seed(config.root_seed, "player_count", game_index)
            ).choice(config.player_counts)
            ruleset = config.ruleset_sampler.sample(
                root_seed=config.root_seed,
                game_index=game_index,
            )
            ruleset.setup_for(player_count)
            if player_count in base_lineups:
                selected = list(base_lineups[player_count])
            else:
                lineup_rng = random.Random(derive_seed(config.root_seed, "lineup", game_index))
                selected = lineup_rng.sample(config.bot_specs, player_count)
                lineup_rng.shuffle(selected)
            offset = game_index % player_count
            lineup = tuple(selected[offset:] + selected[:offset])
            jobs.append(
                GameJob(
                    game_index=game_index,
                    root_seed=config.root_seed,
                    seed=derive_seed(config.root_seed, "game", game_index),
                    player_count=player_count,
                    ruleset=ruleset,
                    lineup=lineup,
                    fault_mode=config.fault_mode,
                )
            )
        return tuple(jobs)

    @staticmethod
    def run(
        config: MonteCarloConfig,
        *,
        workers: int = 1,
    ) -> MonteCarloResult:
        if workers < 1:
            raise ValueError("workers must be positive")
        jobs = MonteCarloRunner.plan(config)
        if workers == 1:
            completed = tuple(_execute_job(job) for job in jobs)
        else:
            _validate_picklable_bot_specs(config.bot_specs)
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    completed = tuple(executor.map(_execute_job, jobs))
            except (BrokenProcessPool, pickle.PicklingError) as error:
                names = ", ".join(spec.name for spec in config.bot_specs)
                raise SimulationError(
                    f"process workers failed to load bot specs: {names}"
                ) from error
        return _aggregate(config, completed)


def _validate_picklable_bot_specs(bot_specs: tuple[BotSpec, ...]) -> None:
    for spec in bot_specs:
        try:
            pickle.dumps(spec)
        except Exception as error:
            raise SimulationError(
                f"bot spec {spec.name!r} ({spec.bot_id}) is not picklable for process workers"
            ) from error
        factory = spec.brain_factory
        module_name = getattr(factory, "__module__", None)
        qualified_name = getattr(factory, "__qualname__", None)
        if module_name is None or qualified_name is None:
            continue
        if module_name == "__main__" or "<locals>" in qualified_name:
            raise SimulationError(
                f"bot spec {spec.name!r} ({spec.bot_id}) factory is not "
                "importable by spawn process workers"
            )
        try:
            resolved: object = importlib.import_module(module_name)
            for part in qualified_name.split("."):
                resolved = getattr(resolved, part)
        except (ImportError, AttributeError) as error:
            raise SimulationError(
                f"bot spec {spec.name!r} ({spec.bot_id}) factory is not "
                "importable by spawn process workers"
            ) from error
        if resolved != factory:
            raise SimulationError(
                f"bot spec {spec.name!r} ({spec.bot_id}) factory is not "
                "importable by spawn process workers"
            )


def _execute_job(job: GameJob) -> _CompletedGame:
    return _CompletedGame(
        job=job,
        match=MatchRunner.run(
            job.lineup,
            ruleset=job.ruleset,
            player_count=job.player_count,
            seed=job.seed,
            fault_mode=job.fault_mode,
        ),
    )


def _aggregate(
    config: MonteCarloConfig,
    completed_games: tuple[_CompletedGame, ...],
) -> MonteCarloResult:
    rank_count = max(config.player_counts)
    bot_order: list[str] = []
    bot_names: dict[str, str] = {}
    total: dict[str, _StatisticsAccumulator] = {}
    per_seat: dict[str, dict[int, _StatisticsAccumulator]] = {}
    per_ruleset: dict[str, dict[str, _StatisticsAccumulator]] = {}
    for spec in config.bot_specs:
        if spec.bot_id in total:
            continue
        bot_order.append(spec.bot_id)
        bot_names[spec.bot_id] = spec.name
        total[spec.bot_id] = _StatisticsAccumulator.with_rank_count(rank_count)
        per_seat[spec.bot_id] = {
            seat: _StatisticsAccumulator.with_rank_count(rank_count) for seat in range(rank_count)
        }
        per_ruleset[spec.bot_id] = {
            ruleset.name: _StatisticsAccumulator.with_rank_count(rank_count)
            for ruleset in config.ruleset_sampler.support()
        }

    summaries: list[GameSummary] = []
    replays: list[MatchReplay] = []
    for completed in sorted(completed_games, key=lambda item: item.job.game_index):
        job = completed.job
        match = completed.match
        decisions_by_seat = tuple(
            sum(
                decision_seat == seat
                for _, decisions in match.replay.decisions
                for decision_seat, _ in decisions
            )
            for seat in range(job.player_count)
        )
        faults_by_seat = tuple(
            sum(fault.seat == seat for fault in match.faults) for seat in range(job.player_count)
        )
        summaries.append(
            GameSummary(
                game_index=job.game_index,
                root_seed=job.root_seed,
                seed=job.seed,
                player_count=job.player_count,
                ruleset_name=job.ruleset.name,
                bot_names=tuple(spec.name for spec in job.lineup),
                bot_ids=tuple(spec.bot_id for spec in job.lineup),
                scores=match.result.scores,
                decision_counts=decisions_by_seat,
                fault_counts=faults_by_seat,
            )
        )
        if config.capture_replays:
            replays.append(
                replace(
                    match.replay,
                    root_seed=job.root_seed,
                    game_index=job.game_index,
                )
            )

        scores_by_seat = {score.seat: score for score in match.result.scores}
        first_place_count = sum(score.rank == 1 for score in match.result.scores)
        first_place_money = max(score.final_money for score in match.result.scores)
        for seat, spec in enumerate(job.lineup):
            score = scores_by_seat[seat]
            for accumulator in (
                total[spec.bot_id],
                per_seat[spec.bot_id][seat],
                per_ruleset[spec.bot_id][job.ruleset.name],
            ):
                accumulator.record(
                    score=score,
                    outright_win=score.rank == 1 and first_place_count == 1,
                    first_place_tie=score.rank == 1 and first_place_count > 1,
                    score_margin=score.final_money - first_place_money,
                    decision_count=decisions_by_seat[seat],
                    faults=faults_by_seat[seat],
                )

    statistics_by_bot = tuple(
        _freeze_bot_statistics(
            bot_id=bot_id,
            bot_name=bot_names[bot_id],
            total=total[bot_id],
            per_seat=per_seat[bot_id],
            per_ruleset=per_ruleset[bot_id],
        )
        for bot_id in bot_order
    )
    return MonteCarloResult(
        game_summaries=tuple(summaries),
        bot_statistics=statistics_by_bot,
        replays=tuple(replays),
    )


def _freeze_bot_statistics(
    *,
    bot_id: str,
    bot_name: str,
    total: _StatisticsAccumulator,
    per_seat: dict[int, _StatisticsAccumulator],
    per_ruleset: dict[str, _StatisticsAccumulator],
) -> BotStatistics:
    return BotStatistics(
        bot_name=bot_name,
        bot_id=bot_id,
        games=total.games,
        outright_wins=total.outright_wins,
        first_place_ties=total.first_place_ties,
        rank_counts=tuple(total.rank_counts),
        final_money_samples=tuple(total.final_money_samples),
        score_margins=tuple(total.score_margins),
        per_seat=tuple(
            SeatStatistics(
                seat=seat,
                games=bucket.games,
                outright_wins=bucket.outright_wins,
                first_place_ties=bucket.first_place_ties,
                rank_counts=tuple(bucket.rank_counts),
                final_money_samples=tuple(bucket.final_money_samples),
                score_margins=tuple(bucket.score_margins),
                decision_counts=tuple(bucket.decision_counts),
                faults=bucket.faults,
            )
            for seat, bucket in sorted(per_seat.items())
        ),
        per_ruleset=tuple(
            RulesetStatistics(
                ruleset_name=ruleset_name,
                games=bucket.games,
                outright_wins=bucket.outright_wins,
                first_place_ties=bucket.first_place_ties,
                rank_counts=tuple(bucket.rank_counts),
                final_money_samples=tuple(bucket.final_money_samples),
                score_margins=tuple(bucket.score_margins),
                decision_counts=tuple(bucket.decision_counts),
                faults=bucket.faults,
            )
            for ruleset_name, bucket in per_ruleset.items()
        ),
        decision_counts=tuple(total.decision_counts),
        faults=total.faults,
    )
