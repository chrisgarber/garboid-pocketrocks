"""Reproduce game-length, terminal-resource, cash, and bid-constraint datasets."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import batched

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    BotSpec,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.model import Phase
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    MonteCarloConfig,
    MonteCarloRunner,
    _execute_job,
)
from garboid_pocketrocks.simulator.sampling import WeightedRulesetSampler

BOT_CLASSES = (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
BOT_NAMES = tuple(bot.BOT_NAME for bot in BOT_CLASSES)
GAMES = 100_000
ROOT_SEED = 20260729
CHUNK_SIZE = 50


@dataclass
class Histogram:
    count: int = 0
    total: int = 0
    values: Counter[int] = field(default_factory=Counter)

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.values[value] += 1

    def merge(self, other: Histogram) -> None:
        self.count += other.count
        self.total += other.total
        self.values.update(other.values)

    def quantile(self, probability: float) -> int:
        target = math.ceil(probability * self.count)
        seen = 0
        for value, count in sorted(self.values.items()):
            seen += count
            if seen >= target:
                return value
        return 0

    def frequency_at_least(self, threshold: int) -> int:
        return sum(count for value, count in self.values.items() if value >= threshold)

    def frequency_equal(self, target: int) -> int:
        return self.values[target]

    def freeze(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean": round(self.total / self.count if self.count else 0.0, 4),
            "p10": self.quantile(0.10),
            "p25": self.quantile(0.25),
            "median": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "min": min(self.values) if self.values else 0,
            "max": max(self.values) if self.values else 0,
            "histogram": dict(sorted(self.values.items())),
        }


@dataclass
class BidConstraints:
    requests: int = 0
    passes: int = 0
    cash_zero_requests: int = 0
    cash_zero_passes: int = 0
    hard_constrained_requests: int = 0
    hard_constrained_passes: int = 0
    cash_positive_passes: int = 0
    bot_games_cash_zero: int = 0
    bot_games_hard_constrained: int = 0

    def merge(self, other: BidConstraints) -> None:
        self.requests += other.requests
        self.passes += other.passes
        self.cash_zero_requests += other.cash_zero_requests
        self.cash_zero_passes += other.cash_zero_passes
        self.hard_constrained_requests += other.hard_constrained_requests
        self.hard_constrained_passes += other.hard_constrained_passes
        self.cash_positive_passes += other.cash_positive_passes
        self.bot_games_cash_zero += other.bot_games_cash_zero
        self.bot_games_hard_constrained += other.bot_games_hard_constrained

    def freeze(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "passes": self.passes,
            "pass_rate": round(100 * self.passes / self.requests, 4),
            "cash_zero_requests": self.cash_zero_requests,
            "cash_zero_request_rate": round(
                100 * self.cash_zero_requests / self.requests,
                4,
            ),
            "cash_zero_passes": self.cash_zero_passes,
            "cash_zero_share_of_passes": round(
                100 * self.cash_zero_passes / self.passes,
                4,
            ),
            "hard_constrained_requests": self.hard_constrained_requests,
            "hard_constrained_request_rate": round(
                100 * self.hard_constrained_requests / self.requests,
                4,
            ),
            "hard_constrained_share_of_passes": round(
                100 * self.hard_constrained_passes / self.passes,
                4,
            ),
            "cash_positive_passes": self.cash_positive_passes,
            "voluntary_share_of_passes": round(
                100 * self.cash_positive_passes / self.passes,
                4,
            ),
            "bot_games_cash_zero": self.bot_games_cash_zero,
            "bot_games_cash_zero_rate": round(
                100 * self.bot_games_cash_zero / GAMES,
                4,
            ),
            "bot_games_hard_constrained": self.bot_games_hard_constrained,
            "bot_games_hard_constrained_rate": round(
                100 * self.bot_games_hard_constrained / GAMES,
                4,
            ),
        }


@dataclass
class ChunkSummary:
    turns: Histogram
    resources: dict[str, Histogram]
    cash: dict[str, Histogram]
    net_cash_after_debt: dict[str, Histogram]
    constraints: dict[str, BidConstraints]


def _analyze_chunk(jobs: tuple[GameJob, ...]) -> ChunkSummary:
    turns = Histogram()
    resources: defaultdict[str, Histogram] = defaultdict(Histogram)
    cash: defaultdict[str, Histogram] = defaultdict(Histogram)
    net_cash_after_debt: defaultdict[str, Histogram] = defaultdict(Histogram)
    constraints: defaultdict[str, BidConstraints] = defaultdict(BidConstraints)

    for job in jobs:
        completed = _execute_job(job)
        replay = completed.match.replay
        transition = GameEngine.start(
            job.ruleset,
            player_count=job.player_count,
            seed=job.seed,
        )
        saw_cash_zero = {bot: False for bot in BOT_NAMES}
        saw_hard_constraint = {bot: False for bot in BOT_NAMES}

        for _, recorded_decisions in replay.decisions:
            assert transition.pending is not None
            decisions = dict(recorded_decisions)
            if transition.state.phase is Phase.BIDDING:
                contexts = transition.pending.contexts_by_seat
                for seat in range(job.player_count):
                    bot = job.lineup[seat].name
                    context = contexts[seat]
                    decision = decisions[seat]
                    bid = decision.value or 0
                    bucket = constraints[bot]
                    bucket.requests += 1
                    if bid == 0:
                        bucket.passes += 1
                    cash_before = context.cash_by_seat[seat]
                    if cash_before == 0:
                        bucket.cash_zero_requests += 1
                        saw_cash_zero[bot] = True
                        if bid == 0:
                            bucket.cash_zero_passes += 1
                    elif bid == 0:
                        bucket.cash_positive_passes += 1
                    if context.legal_max_amount == 0:
                        bucket.hard_constrained_requests += 1
                        saw_hard_constraint[bot] = True
                        if bid == 0:
                            bucket.hard_constrained_passes += 1
            transition = GameEngine.step(transition.state, decisions)

        turns.add(transition.state.turn_index)
        for seat, player in enumerate(transition.state.players):
            bot = job.lineup[seat].name
            resources[bot].add(len(player.won_resources))
            cash[bot].add(player.cash)
            debt = sum(loan.principal for loan in player.loans)
            net_cash_after_debt[bot].add(player.cash - debt)
            constraints[bot].bot_games_cash_zero += int(saw_cash_zero[bot])
            constraints[bot].bot_games_hard_constrained += int(saw_hard_constraint[bot])

    return ChunkSummary(
        turns=turns,
        resources=dict(resources),
        cash=dict(cash),
        net_cash_after_debt=dict(net_cash_after_debt),
        constraints=dict(constraints),
    )


def main() -> None:
    charts = tuple(live_ruleset(chart) for chart in "ABCDE")
    config = MonteCarloConfig(
        bot_specs=tuple(BotSpec.from_bot_class(bot) for bot in BOT_CLASSES),
        games=GAMES,
        player_counts=(3,),
        ruleset_sampler=WeightedRulesetSampler(tuple((ruleset, 1) for ruleset in charts)),
        root_seed=ROOT_SEED,
    )
    jobs = MonteCarloRunner.plan(config)
    chunks = tuple(tuple(chunk) for chunk in batched(jobs, CHUNK_SIZE, strict=False))

    turns = Histogram()
    resources: defaultdict[str, Histogram] = defaultdict(Histogram)
    cash: defaultdict[str, Histogram] = defaultdict(Histogram)
    net_cash_after_debt: defaultdict[str, Histogram] = defaultdict(Histogram)
    constraints: defaultdict[str, BidConstraints] = defaultdict(BidConstraints)

    with ProcessPoolExecutor(max_workers=16) as executor:
        for summary in executor.map(_analyze_chunk, chunks):
            turns.merge(summary.turns)
            for bot, bucket in summary.resources.items():
                resources[bot].merge(bucket)
            for bot, bucket in summary.cash.items():
                cash[bot].merge(bucket)
            for bot, bucket in summary.net_cash_after_debt.items():
                net_cash_after_debt[bot].merge(bucket)
            for bot, bucket in summary.constraints.items():
                constraints[bot].merge(bucket)

    payload = {
        "configuration": {
            "games": GAMES,
            "players": 3,
            "bots": BOT_NAMES,
            "charts": tuple("ABCDE"),
            "root_seed": ROOT_SEED,
            "workers": 16,
        },
        "turns_per_game": turns.freeze(),
        "bots": {
            bot: {
                "resources_won": resources[bot].freeze(),
                "terminal_liquid_cash": {
                    **cash[bot].freeze(),
                    "positive_count": cash[bot].frequency_at_least(1),
                    "positive_rate": round(
                        100 * cash[bot].frequency_at_least(1) / GAMES,
                        4,
                    ),
                    "at_least_5_count": cash[bot].frequency_at_least(5),
                    "at_least_5_rate": round(
                        100 * cash[bot].frequency_at_least(5) / GAMES,
                        4,
                    ),
                    "at_least_10_count": cash[bot].frequency_at_least(10),
                    "at_least_10_rate": round(
                        100 * cash[bot].frequency_at_least(10) / GAMES,
                        4,
                    ),
                    "zero_count": cash[bot].frequency_equal(0),
                    "zero_rate": round(
                        100 * cash[bot].frequency_equal(0) / GAMES,
                        4,
                    ),
                },
                "terminal_cash_after_loan_debt": net_cash_after_debt[bot].freeze(),
                "bid_constraints": constraints[bot].freeze(),
            }
            for bot in BOT_NAMES
        },
        "checks": {
            "mean_total_resources": round(
                sum(resources[bot].total for bot in BOT_NAMES) / GAMES,
                4,
            ),
        },
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
