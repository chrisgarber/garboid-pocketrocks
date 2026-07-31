from __future__ import annotations

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    BotStatistics,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)


def _benchmark_config(*, games: int = 100) -> MonteCarloConfig:
    return MonteCarloConfig(
        bot_specs=(
            BotSpec.from_bot_class(AggressiveHeuristicBot),
            BotSpec.from_bot_class(BalancedHeuristicBot),
            BotSpec.from_bot_class(PassiveHeuristicBot),
        ),
        games=games,
        player_counts=(3,),
        value_charts=("A",),
        root_seed=42,
    )


def _statistics_by_name(result: MonteCarloResult) -> dict[str, BotStatistics]:
    return {statistics.bot_name: statistics for statistics in result.bot_statistics}


def _behavior_snapshot(by_name: dict[str, BotStatistics]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "bidding_requests": statistics.behavior.bidding_requests,
            "passes": statistics.behavior.passes,
            "nonzero_bid_count": len(statistics.behavior.nonzero_bids),
            "nonzero_bid_total": sum(statistics.behavior.nonzero_bids),
            "resource_cards_won": statistics.behavior.resource_cards_won,
            "objectives_claimed": statistics.behavior.objectives_claimed,
        }
        for name, statistics in by_name.items()
    }


def test_seed_42_live_a_heuristic_tournament_matches_promoted_v3_behavior() -> None:
    config = _benchmark_config()

    result = MonteCarloRunner.run(config, workers=1)

    assert all(statistics.faults == 0 for statistics in result.bot_statistics)
    for statistics in result.bot_statistics:
        seat_games = tuple(bucket.games for bucket in statistics.per_seat)
        assert sum(seat_games) == config.games
        assert max(seat_games) - min(seat_games) <= 1

    by_name = _statistics_by_name(result)
    assert set(by_name) == {"aggressive", "balanced", "passive"}
    assert _behavior_snapshot(by_name) == {
        "aggressive": {
            "bidding_requests": 1587,
            "passes": 139,
            "nonzero_bid_count": 1448,
            "nonzero_bid_total": 7039,
            "resource_cards_won": 490,
            "objectives_claimed": 90,
        },
        "balanced": {
            "bidding_requests": 1587,
            "passes": 199,
            "nonzero_bid_count": 1388,
            "nonzero_bid_total": 6692,
            "resource_cards_won": 563,
            "objectives_claimed": 158,
        },
        "passive": {
            "bidding_requests": 1587,
            "passes": 89,
            "nonzero_bid_count": 1498,
            "nonzero_bid_total": 7666,
            "resource_cards_won": 447,
            "objectives_claimed": 106,
        },
    }
