from __future__ import annotations

import importlib
import pickle
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace

from pocketrocks.sim.constants import ACTION_WIRE_IDS, VALUE_CHARTS

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.replay import MatchReplay
from garboid_pocketrocks.simulator.runner import (
    FaultMode,
    MatchResult,
    MatchRunner,
)
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.simulator.session import SessionScore


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    bot_specs: tuple[BotSpec, ...]
    games: int
    player_counts: tuple[int, ...]
    value_charts: tuple[str, ...]
    root_seed: int
    objectives_enabled: tuple[bool, ...] = (True,)
    fault_mode: FaultMode = FaultMode.RAISE
    capture_replays: bool = False

    def __post_init__(self) -> None:
        if self.games < 0:
            raise ValueError("games must be nonnegative")
        if not self.player_counts:
            raise ValueError("player_counts must be nonempty")
        if not self.bot_specs:
            raise ValueError("bot_specs must be nonempty")
        if not self.value_charts:
            raise ValueError("value_charts must be nonempty")
        normalized_charts = tuple(chart.upper() for chart in self.value_charts)
        if any(chart not in VALUE_CHARTS for chart in normalized_charts):
            raise ValueError("value charts must use SDK chart names A-E")
        object.__setattr__(self, "value_charts", normalized_charts)
        if not self.objectives_enabled:
            raise ValueError("objectives_enabled must be nonempty")
        if any(not isinstance(enabled, bool) for enabled in self.objectives_enabled):
            raise ValueError("objectives_enabled values must be booleans")
        if any(player_count < 3 or player_count > 5 for player_count in self.player_counts):
            raise ValueError("player counts must be between 3 and 5")
        if len(self.bot_specs) < max(self.player_counts):
            raise ValueError("not enough bot specs for the requested player counts")


@dataclass(frozen=True, slots=True)
class GameJob:
    game_index: int
    root_seed: int
    seed: int
    player_count: int
    value_chart: str
    objectives_enabled: bool
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
    scores: tuple[SessionScore, ...]
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
class BehaviorStatistics:
    bidding_requests: int
    passes: int
    nonzero_bids: tuple[int, ...]
    reveal_choices: tuple[int, ...]
    wins_by_action: tuple[int, ...]
    resource_cards_won: int
    objectives_claimed: int

    def pass_rate(self) -> float:
        return self.passes / self.bidding_requests if self.bidding_requests else 0.0

    def mean_nonzero_bid(self) -> float:
        return float(statistics.mean(self.nonzero_bids)) if self.nonzero_bids else 0.0


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
    behavior: BehaviorStatistics

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
        score: SessionScore,
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


@dataclass(slots=True)
class _BehaviorAccumulator:
    bidding_requests: int = 0
    passes: int = 0
    nonzero_bids: list[int] = field(default_factory=list)
    reveal_choices: list[int] = field(default_factory=list)
    wins_by_action: list[int] = field(default_factory=lambda: [0] * 6)
    resource_cards_won: int = 0
    objectives_claimed: int = 0


def _variant_name(value_chart: str, objectives_enabled: bool) -> str:
    suffix = "" if objectives_enabled else "-no-objectives"
    return f"live-{value_chart}{suffix}"


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
        variants = tuple(
            (chart, objectives_enabled)
            for chart in config.value_charts
            for objectives_enabled in config.objectives_enabled
        )
        for game_index in range(config.games):
            player_count = random.Random(
                derive_seed(config.root_seed, "player_count", game_index)
            ).choice(config.player_counts)
            value_chart, objectives_enabled = random.Random(
                derive_seed(config.root_seed, "variant", game_index)
            ).choice(variants)
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
                    value_chart=value_chart,
                    objectives_enabled=objectives_enabled,
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
        jobs = MonteCarloRunner.plan(config)
        return MonteCarloRunner.run_jobs(config, jobs, workers=workers)

    @staticmethod
    def run_jobs(
        config: MonteCarloConfig,
        jobs: tuple[GameJob, ...],
        *,
        workers: int = 1,
    ) -> MonteCarloResult:
        if workers < 1:
            raise ValueError("workers must be positive")
        _validate_jobs(config, jobs)
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


def _validate_jobs(
    config: MonteCarloConfig,
    jobs: tuple[GameJob, ...],
) -> None:
    if len(jobs) != config.games:
        raise ValueError(f"job count {len(jobs)} does not match configured games {config.games}")
    if tuple(job.game_index for job in jobs) != tuple(range(config.games)):
        raise ValueError("game indices must be contiguous and start at zero")
    if any(job.root_seed != config.root_seed for job in jobs):
        raise ValueError("every job root seed must match the configuration")

    supported_variants = {
        (value_chart, objectives_enabled)
        for value_chart in config.value_charts
        for objectives_enabled in config.objectives_enabled
    }
    configured_bot_ids = {spec.bot_id for spec in config.bot_specs}
    for job in jobs:
        if job.player_count not in config.player_counts:
            raise ValueError(f"job {job.game_index} uses an unconfigured player count")
        if (job.value_chart, job.objectives_enabled) not in supported_variants:
            raise ValueError(f"job {job.game_index} uses an unsupported SDK variant")
        if len(job.lineup) != job.player_count:
            raise ValueError(f"job {job.game_index} lineup length does not match player count")
        if any(spec.bot_id not in configured_bot_ids for spec in job.lineup):
            raise ValueError(f"job {job.game_index} uses an unconfigured bot identity")
        if job.fault_mode is not config.fault_mode:
            raise ValueError(f"job {job.game_index} fault mode does not match the configuration")


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
            player_count=job.player_count,
            seed=job.seed,
            value_chart=job.value_chart,
            objectives_enabled=job.objectives_enabled,
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
    behavior: dict[str, _BehaviorAccumulator] = {}
    variant_names = tuple(
        _variant_name(chart, enabled)
        for chart in config.value_charts
        for enabled in config.objectives_enabled
    )
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
            variant_name: _StatisticsAccumulator.with_rank_count(rank_count)
            for variant_name in variant_names
        }
        behavior[spec.bot_id] = _BehaviorAccumulator()

    summaries: list[GameSummary] = []
    replays: list[MatchReplay] = []
    for completed in sorted(completed_games, key=lambda item: item.job.game_index):
        job = completed.job
        match = completed.match
        variant_name = _variant_name(job.value_chart, job.objectives_enabled)
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
                ruleset_name=variant_name,
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

        _record_behavior(
            match.replay,
            bot_ids_by_seat=tuple(spec.bot_id for spec in job.lineup),
            accumulators=behavior,
        )
        scores_by_seat = {score.seat: score for score in match.result.scores}
        first_place_count = sum(score.rank == 1 for score in match.result.scores)
        first_place_money = max(score.final_money for score in match.result.scores)
        for seat, spec in enumerate(job.lineup):
            score = scores_by_seat[seat]
            for accumulator in (
                total[spec.bot_id],
                per_seat[spec.bot_id][seat],
                per_ruleset[spec.bot_id][variant_name],
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
            behavior=behavior[bot_id],
        )
        for bot_id in bot_order
    )
    return MonteCarloResult(
        game_summaries=tuple(summaries),
        bot_statistics=statistics_by_bot,
        replays=tuple(replays),
    )


def _record_behavior(
    replay: MatchReplay,
    *,
    bot_ids_by_seat: tuple[str, ...],
    accumulators: dict[str, _BehaviorAccumulator],
) -> None:
    for _, decisions in replay.decisions:
        is_bidding_batch = len(decisions) == replay.player_count
        for seat, decision in decisions:
            accumulator = accumulators[bot_ids_by_seat[seat]]
            if is_bidding_batch:
                accumulator.bidding_requests += 1
                if decision.action_kind == "pass" or (
                    decision.action_kind == "submitBid" and decision.value == 0
                ):
                    accumulator.passes += 1
                elif decision.action_kind == "submitBid":
                    assert decision.value is not None
                    accumulator.nonzero_bids.append(decision.value)
            elif decision.action_kind == "selectInfoToReveal":
                assert decision.value is not None
                accumulator.reveal_choices.append(decision.value)

    for turn in replay.turns:
        accumulator = accumulators[bot_ids_by_seat[turn.winner_seat]]
        accumulator.wins_by_action[ACTION_WIRE_IDS[turn.action] - 1] += 1
        accumulator.resource_cards_won += len(turn.bundle_suits)
        accumulator.objectives_claimed += len(turn.claimed_objective_wire_ids)


def _freeze_bot_statistics(
    *,
    bot_id: str,
    bot_name: str,
    total: _StatisticsAccumulator,
    per_seat: dict[int, _StatisticsAccumulator],
    per_ruleset: dict[str, _StatisticsAccumulator],
    behavior: _BehaviorAccumulator,
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
        behavior=BehaviorStatistics(
            bidding_requests=behavior.bidding_requests,
            passes=behavior.passes,
            nonzero_bids=tuple(sorted(behavior.nonzero_bids)),
            reveal_choices=tuple(sorted(behavior.reveal_choices)),
            wins_by_action=tuple(behavior.wins_by_action),
            resource_cards_won=behavior.resource_cards_won,
            objectives_claimed=behavior.objectives_claimed,
        ),
    )
