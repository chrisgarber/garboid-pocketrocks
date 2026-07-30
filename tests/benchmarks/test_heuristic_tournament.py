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


def _benchmark_config(*, games: int = 1_000) -> MonteCarloConfig:
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


def _behavior_summary(by_name: dict[str, BotStatistics]) -> str:
    return ", ".join(
        (
            f"{name}: pass_rate={statistics.behavior.pass_rate():.6f}, "
            f"mean_nonzero_bid={statistics.behavior.mean_nonzero_bid():.6f}, "
            f"resource_cards_won={statistics.behavior.resource_cards_won}"
        )
        for name, statistics in sorted(by_name.items())
    )


def test_seed_42_live_a_heuristic_tournament_is_reproducible_and_distinct() -> None:
    config = _benchmark_config()

    serial = MonteCarloRunner.run(config, workers=1)
    parallel = MonteCarloRunner.run(config, workers=2)

    assert serial == parallel
    assert all(statistics.faults == 0 for statistics in serial.bot_statistics)
    for statistics in serial.bot_statistics:
        seat_games = tuple(bucket.games for bucket in statistics.per_seat)
        assert sum(seat_games) == config.games
        assert max(seat_games) - min(seat_games) <= 1

    by_name = _statistics_by_name(serial)
    assert set(by_name) == {"aggressive", "balanced", "passive"}
    summary = _behavior_summary(by_name)
    assert (
        by_name["aggressive"].behavior.mean_nonzero_bid()
        > by_name["balanced"].behavior.mean_nonzero_bid()
        > by_name["passive"].behavior.mean_nonzero_bid()
    ), summary
    assert (
        by_name["aggressive"].behavior.resource_cards_won
        > by_name["balanced"].behavior.resource_cards_won
        > by_name["passive"].behavior.resource_cards_won
    ), summary
