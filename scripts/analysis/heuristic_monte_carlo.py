"""Reproduce the 100,000-game heuristic tournament summary dataset."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    BotSpec,
    PassiveHeuristicBot,
)
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)

BOT_CLASSES = (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
)
BOT_NAMES = tuple(bot.BOT_NAME for bot in BOT_CLASSES)
ACTION_NAMES = ("Auction 1", "Auction 2", "Loan $10", "Loan $20", "Invest $5", "Invest $10")
ROOT_SEED = 20260729
GAMES = 100_000
TREND_BIN_SIZE = 1_000


def quantile(values: list[int] | tuple[int, ...], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[int] | tuple[int, ...]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p05": round(quantile(values, 0.05), 2),
        "p25": round(quantile(values, 0.25), 2),
        "median": round(quantile(values, 0.50), 2),
        "mean": round(statistics.mean(values), 3),
        "p75": round(quantile(values, 0.75), 2),
        "p95": round(quantile(values, 0.95), 2),
        "max": max(values),
        "stddev": round(statistics.pstdev(values), 3),
    }


def main() -> None:
    charts = tuple("ABCDE")
    config = MonteCarloConfig(
        bot_specs=tuple(BotSpec.from_bot_class(bot) for bot in BOT_CLASSES),
        games=GAMES,
        player_counts=(3,),
        value_charts=charts,
        root_seed=ROOT_SEED,
    )
    result = MonteCarloRunner.run(config, workers=16)

    stats_by_name = {stat.bot_name: stat for stat in result.bot_statistics}
    bot_summaries: dict[str, object] = {}
    for name in BOT_NAMES:
        stat = stats_by_name[name]
        behavior = stat.behavior
        bot_summaries[name] = {
            "games": stat.games,
            "outright_wins": stat.outright_wins,
            "win_rate": round(100 * stat.outright_wins / stat.games, 4),
            "first_place_ties": stat.first_place_ties,
            "tied_first_rate": round(100 * stat.first_place_ties / stat.games, 4),
            "rank_counts": list(stat.rank_counts),
            "mean_rank": round(stat.mean_rank(), 4),
            "score": distribution(stat.final_money_samples),
            "pass_rate": round(100 * behavior.pass_rate(), 4),
            "mean_nonzero_bid": round(behavior.mean_nonzero_bid(), 4),
            "objectives_claimed": behavior.objectives_claimed,
            "objectives_per_100_games": round(100 * behavior.objectives_claimed / stat.games, 3),
            "resource_cards_won": behavior.resource_cards_won,
            "resource_cards_per_game": round(behavior.resource_cards_won / stat.games, 4),
            "action_wins": dict(zip(ACTION_NAMES, behavior.wins_by_action, strict=True)),
            "action_wins_per_100_games": {
                action: round(100 * count / stat.games, 3)
                for action, count in zip(ACTION_NAMES, behavior.wins_by_action, strict=True)
            },
            "faults": stat.faults,
        }

    per_chart: dict[str, object] = {}
    for chart in charts:
        label = f"live-{chart}"
        games = sum(1 for summary in result.game_summaries if summary.ruleset_name == label)
        winning_scores = [
            max(score.final_money for score in summary.scores)
            for summary in result.game_summaries
            if summary.ruleset_name == label
        ]
        bots: dict[str, object] = {}
        for name in BOT_NAMES:
            bucket = next(
                item for item in stats_by_name[name].per_ruleset if item.ruleset_name == label
            )
            bots[name] = {
                "games": bucket.games,
                "outright_wins": bucket.outright_wins,
                "win_rate": round(100 * bucket.outright_wins / bucket.games, 4),
                "ties": bucket.first_place_ties,
                "mean_score": round(statistics.mean(bucket.final_money_samples), 3),
                "median_score": round(statistics.median(bucket.final_money_samples), 3),
                "mean_rank": round(
                    sum(rank * count for rank, count in enumerate(bucket.rank_counts, start=1))
                    / bucket.games,
                    4,
                ),
            }
        per_chart[label] = {
            "games": games,
            "winning_score": distribution(winning_scores),
            "bots": bots,
        }

    trend_bins: list[dict[str, object]] = []
    cumulative_wins = defaultdict(int)
    cumulative_games = 0
    for start in range(0, GAMES, TREND_BIN_SIZE):
        games = result.game_summaries[start : start + TREND_BIN_SIZE]
        winning_scores = [max(score.final_money for score in game.scores) for game in games]
        wins = {name: 0 for name in BOT_NAMES}
        ties = 0
        for game in games:
            first = [score.seat for score in game.scores if score.rank == 1]
            if len(first) == 1:
                name = game.bot_names[first[0]]
                wins[name] += 1
                cumulative_wins[name] += 1
            else:
                ties += 1
        cumulative_games += len(games)
        trend_bins.append(
            {
                "through_games": cumulative_games,
                "mean_winning_score": round(statistics.mean(winning_scores), 3),
                "median_winning_score": round(statistics.median(winning_scores), 3),
                "wins": wins,
                "win_rates": {name: round(100 * wins[name] / len(games), 3) for name in BOT_NAMES},
                "cumulative_win_rates": {
                    name: round(100 * cumulative_wins[name] / cumulative_games, 3)
                    for name in BOT_NAMES
                },
                "ties": ties,
            }
        )

    winning_scores = [
        max(score.final_money for score in summary.scores) for summary in result.game_summaries
    ]
    all_scores = [
        score.final_money for summary in result.game_summaries for score in summary.scores
    ]
    summary = {
        "configuration": {
            "games": GAMES,
            "players": 3,
            "bots": list(BOT_NAMES),
            "charts": list("ABCDE"),
            "chart_sampling": "equal-weight deterministic sampling",
            "root_seed": ROOT_SEED,
            "workers": 16,
            "trend_bin_size": TREND_BIN_SIZE,
        },
        "overall": {
            "winning_score": distribution(winning_scores),
            "all_scores": distribution(all_scores),
            "outright_games": sum(bot_summaries[name]["outright_wins"] for name in BOT_NAMES),
            "tied_games": sum(
                1
                for game in result.game_summaries
                if sum(score.rank == 1 for score in game.scores) > 1
            ),
        },
        "bots": bot_summaries,
        "per_chart": per_chart,
        "trends": trend_bins,
    }
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
