from __future__ import annotations

import itertools
from collections import Counter, deque
from dataclasses import dataclass, replace

from pocketrocks.sim.constants import VALUE_CHARTS

from garboid_pocketrocks.bots import BotSpec
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
)
from garboid_pocketrocks.simulator.runner import FaultMode
from garboid_pocketrocks.simulator.seeding import derive_seed


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    bot_specs: tuple[BotSpec, ...]
    games: int = 10_000
    player_counts: tuple[int, ...] = (3, 4, 5)
    charts: tuple[str, ...] = tuple(VALUE_CHARTS)
    root_seed: int = 0
    fault_mode: FaultMode = FaultMode.RECORD_AND_PASS
    batch_size: int = 256
    bootstrap_samples: int = 200

    def __post_init__(self) -> None:
        if not self.bot_specs:
            raise ValueError("tournament requires at least one bot")
        if not self.player_counts:
            raise ValueError("tournament player counts must be nonempty")
        if any(player_count not in (3, 4, 5) for player_count in self.player_counts):
            raise ValueError("tournament player counts must be 3, 4, or 5")
        if len(set(self.player_counts)) != len(self.player_counts):
            raise ValueError("tournament player counts must be unique")
        if not self.charts:
            raise ValueError("tournament charts must be nonempty")
        if any(chart not in VALUE_CHARTS for chart in self.charts):
            raise ValueError("tournament charts must be known uppercase live chart names")
        if len(set(self.charts)) != len(self.charts):
            raise ValueError("tournament charts must be unique")
        if self.games < len(self.player_counts) * len(self.charts):
            raise ValueError("tournament games must cover every chart/player-count cell")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap samples must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch size must be positive")

        names = tuple(spec.name for spec in self.bot_specs)
        bot_ids = tuple(spec.bot_id for spec in self.bot_specs)
        if len(set(names)) != len(names):
            raise ValueError("tournament bot names must be unique")
        if len(set(bot_ids)) != len(bot_ids):
            raise ValueError("tournament bot IDs must be unique")
        required = max(self.player_counts)
        if len(self.bot_specs) < required:
            raise ValueError(
                f"{required}-player games require {required} distinct bots; "
                f"received {len(self.bot_specs)}"
            )


@dataclass(frozen=True, slots=True)
class ConditionQuota:
    chart: str
    player_count: int
    games: int


@dataclass(frozen=True, slots=True)
class PairExposure:
    first_bot_id: str
    second_bot_id: str
    games: int


@dataclass(frozen=True, slots=True)
class TournamentPlan:
    monte_carlo_config: MonteCarloConfig
    jobs: tuple[GameJob, ...]
    quotas: tuple[ConditionQuota, ...]
    pair_exposures: tuple[PairExposure, ...]


class TournamentPlanner:
    @staticmethod
    def plan(config: TournamentConfig) -> TournamentPlan:
        quotas = _allocate_quotas(config)
        condition_appearances: Counter[tuple[str, int, str]] = Counter()
        pair_appearances: Counter[tuple[str, str]] = Counter()
        global_appearances: Counter[str] = Counter()
        seat_appearances: Counter[tuple[int, str, int]] = Counter()
        jobs: list[GameJob] = []

        for quota in quotas:
            for _ in range(quota.games):
                game_index = len(jobs)
                selected = _select_lineup(
                    config,
                    chart=quota.chart,
                    player_count=quota.player_count,
                    game_index=game_index,
                    condition_appearances=condition_appearances,
                    pair_appearances=pair_appearances,
                    global_appearances=global_appearances,
                )
                lineup = _assign_seats(
                    selected,
                    player_count=quota.player_count,
                    game_index=game_index,
                    root_seed=config.root_seed,
                    seat_appearances=seat_appearances,
                )
                jobs.append(
                    GameJob(
                        game_index=game_index,
                        root_seed=config.root_seed,
                        seed=derive_seed(config.root_seed, "tournament-game", game_index),
                        player_count=quota.player_count,
                        value_chart=quota.chart,
                        objectives_enabled=True,
                        lineup=lineup,
                        fault_mode=config.fault_mode,
                    )
                )
                _record_lineup(
                    lineup,
                    chart=quota.chart,
                    player_count=quota.player_count,
                    condition_appearances=condition_appearances,
                    pair_appearances=pair_appearances,
                    global_appearances=global_appearances,
                    seat_appearances=seat_appearances,
                )

        jobs = list(_rebalance_seats(tuple(jobs)))
        monte_carlo_config = MonteCarloConfig(
            bot_specs=config.bot_specs,
            games=config.games,
            player_counts=config.player_counts,
            value_charts=config.charts,
            root_seed=config.root_seed,
            fault_mode=config.fault_mode,
        )
        pair_exposures = tuple(
            PairExposure(first, second, pair_appearances[first, second])
            for first, second in itertools.combinations(
                sorted(spec.bot_id for spec in config.bot_specs),
                2,
            )
        )
        return TournamentPlan(
            monte_carlo_config=monte_carlo_config,
            jobs=tuple(jobs),
            quotas=quotas,
            pair_exposures=pair_exposures,
        )


def _allocate_quotas(config: TournamentConfig) -> tuple[ConditionQuota, ...]:
    cells = tuple(
        (chart, player_count) for chart in config.charts for player_count in config.player_counts
    )
    games_per_cell, remainder = divmod(config.games, len(cells))
    return tuple(
        ConditionQuota(
            chart=chart,
            player_count=player_count,
            games=games_per_cell + int(index < remainder),
        )
        for index, (chart, player_count) in enumerate(cells)
    )


def _select_lineup(
    config: TournamentConfig,
    *,
    chart: str,
    player_count: int,
    game_index: int,
    condition_appearances: Counter[tuple[str, int, str]],
    pair_appearances: Counter[tuple[str, str]],
    global_appearances: Counter[str],
) -> tuple[BotSpec, ...]:
    selected: list[BotSpec] = []
    while len(selected) < player_count:
        selected_ids = {spec.bot_id for spec in selected}

        def candidate_key(spec: BotSpec) -> tuple[int, int, int, int]:
            pair_count = sum(
                pair_appearances[_pair(spec.bot_id, selected_spec.bot_id)]
                for selected_spec in selected
            )
            namespace = f"tournament-lineup:{chart}:{player_count}:{len(selected)}:{spec.bot_id}"
            return (
                condition_appearances[chart, player_count, spec.bot_id],
                pair_count,
                global_appearances[spec.bot_id],
                derive_seed(config.root_seed, namespace, game_index),
            )

        candidates = (spec for spec in config.bot_specs if spec.bot_id not in selected_ids)
        selected.append(min(candidates, key=candidate_key))
    return tuple(selected)


def _assign_seats(
    selected: tuple[BotSpec, ...],
    *,
    player_count: int,
    game_index: int,
    root_seed: int,
    seat_appearances: Counter[tuple[int, str, int]],
) -> tuple[BotSpec, ...]:
    def permutation_key(lineup: tuple[BotSpec, ...]) -> tuple[int, int, int]:
        spreads: list[int] = []
        square_total = 0
        for seat, spec in enumerate(lineup):
            counts = [
                seat_appearances[player_count, spec.bot_id, candidate_seat]
                + int(candidate_seat == seat)
                for candidate_seat in range(player_count)
            ]
            spreads.append(max(counts) - min(counts))
            square_total += sum(count * count for count in counts)
        namespace = "tournament-seats:" + ",".join(spec.bot_id for spec in lineup)
        return (
            max(spreads, default=0),
            square_total,
            derive_seed(root_seed, namespace, game_index),
        )

    return min(itertools.permutations(selected), key=permutation_key)


def _record_lineup(
    lineup: tuple[BotSpec, ...],
    *,
    chart: str,
    player_count: int,
    condition_appearances: Counter[tuple[str, int, str]],
    pair_appearances: Counter[tuple[str, str]],
    global_appearances: Counter[str],
    seat_appearances: Counter[tuple[int, str, int]],
) -> None:
    for seat, spec in enumerate(lineup):
        condition_appearances[chart, player_count, spec.bot_id] += 1
        global_appearances[spec.bot_id] += 1
        seat_appearances[player_count, spec.bot_id, seat] += 1
    for first, second in itertools.combinations(sorted(spec.bot_id for spec in lineup), 2):
        pair_appearances[first, second] += 1


def _rebalance_seats(jobs: tuple[GameJob, ...]) -> tuple[GameJob, ...]:
    balanced = list(jobs)
    for player_count in (3, 4, 5):
        indexes = [index for index, job in enumerate(balanced) if job.player_count == player_count]
        counts: Counter[tuple[str, int]] = Counter(
            (spec.bot_id, seat)
            for index in indexes
            for seat, spec in enumerate(balanced[index].lineup)
        )
        while True:
            best: tuple[int, int, int, int] | None = None
            for index in indexes:
                lineup = balanced[index].lineup
                for first_seat, second_seat in itertools.combinations(range(player_count), 2):
                    first = lineup[first_seat].bot_id
                    second = lineup[second_seat].bot_id
                    before = (
                        counts[first, first_seat] ** 2
                        + counts[first, second_seat] ** 2
                        + counts[second, first_seat] ** 2
                        + counts[second, second_seat] ** 2
                    )
                    after = (
                        (counts[first, first_seat] - 1) ** 2
                        + (counts[first, second_seat] + 1) ** 2
                        + (counts[second, first_seat] + 1) ** 2
                        + (counts[second, second_seat] - 1) ** 2
                    )
                    change = after - before
                    candidate = (change, index, first_seat, second_seat)
                    if change < 0 and (best is None or candidate < best):
                        best = candidate
            if best is None:
                break
            _, index, first_seat, second_seat = best
            mutable_lineup = list(balanced[index].lineup)
            first = mutable_lineup[first_seat].bot_id
            second = mutable_lineup[second_seat].bot_id
            counts[first, first_seat] -= 1
            counts[first, second_seat] += 1
            counts[second, first_seat] += 1
            counts[second, second_seat] -= 1
            mutable_lineup[first_seat], mutable_lineup[second_seat] = (
                mutable_lineup[second_seat],
                mutable_lineup[first_seat],
            )
            balanced[index] = replace(balanced[index], lineup=tuple(mutable_lineup))
        _repair_seat_spreads(
            balanced,
            indexes=indexes,
            player_count=player_count,
            counts=counts,
        )
    return tuple(balanced)


def _repair_seat_spreads(
    jobs: list[GameJob],
    *,
    indexes: list[int],
    player_count: int,
    counts: Counter[tuple[str, int]],
) -> None:
    bot_ids = sorted({spec.bot_id for index in indexes for spec in jobs[index].lineup})
    while True:
        repair: tuple[str, int, int] | None = None
        for bot_id in bot_ids:
            values = [counts[bot_id, seat] for seat in range(player_count)]
            if max(values) - min(values) > 1:
                repair = (bot_id, values.index(max(values)), values.index(min(values)))
                break
        if repair is None:
            return
        start, surplus_seat, deficit_seat = repair
        targets = {
            bot_id
            for bot_id in bot_ids
            if counts[bot_id, deficit_seat] > counts[bot_id, surplus_seat]
        }
        previous: dict[str, tuple[str, int] | None] = {start: None}
        queue = deque((start,))
        target: str | None = None
        while queue and target is None:
            current = queue.popleft()
            for index in indexes:
                lineup = jobs[index].lineup
                if lineup[surplus_seat].bot_id != current:
                    continue
                displaced = lineup[deficit_seat].bot_id
                if displaced in previous:
                    continue
                previous[displaced] = (current, index)
                if displaced in targets:
                    target = displaced
                    break
                queue.append(displaced)
        if target is None:
            return

        path: list[int] = []
        current = target
        while True:
            edge = previous[current]
            if edge is None:
                break
            parent, index = edge
            path.append(index)
            current = parent
        for index in reversed(path):
            mutable_lineup = list(jobs[index].lineup)
            first = mutable_lineup[surplus_seat].bot_id
            second = mutable_lineup[deficit_seat].bot_id
            counts[first, surplus_seat] -= 1
            counts[first, deficit_seat] += 1
            counts[second, surplus_seat] += 1
            counts[second, deficit_seat] -= 1
            mutable_lineup[surplus_seat], mutable_lineup[deficit_seat] = (
                mutable_lineup[deficit_seat],
                mutable_lineup[surplus_seat],
            )
            jobs[index] = replace(jobs[index], lineup=tuple(mutable_lineup))


def _pair(first_bot_id: str, second_bot_id: str) -> tuple[str, str]:
    if first_bot_id < second_bot_id:
        return (first_bot_id, second_bot_id)
    return (second_bot_id, first_bot_id)
