from __future__ import annotations

import argparse
import json
import math
import statistics
import time

from _fixed_finalists_bot import FixedBidDiverseV1Brain, FixedBidTunedV1Brain

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.registry import BOT_SPECS_BY_NAME
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner
from garboid_pocketrocks.tournament.analysis import analyze_tournament
from garboid_pocketrocks.tournament.rating import fit_plackett_luce, observations_from_games
from garboid_pocketrocks.tournament.schedule import TournamentConfig, TournamentPlanner

FINALIST_SPECS = (
    BotSpec.for_simulation("fixed-bid-tuned-v1", FixedBidTunedV1Brain),
    BotSpec.for_simulation("fixed-bid-diverse-v1", FixedBidDiverseV1Brain),
)
FIXED_IDS = ("fixed-bid", "fixed-bid-tuned-v1", "fixed-bid-diverse-v1")
PRIOR_DEFAULT_NAMES = (
    "random",
    "fixed-bid",
    "aggressive-v1",
    "balanced-v1",
    "passive-v1",
    "aggressive-v2",
    "balanced-v2",
    "passive-v2",
    "sdk-greedy-value-v1",
    "vector_ppo_small_v1_g1500",
    "vector_ppo_large_v1_g350k",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    field = tuple(BOT_SPECS_BY_NAME[name] for name in PRIOR_DEFAULT_NAMES) + FINALIST_SPECS
    assert len(field) == 13
    assert 4 * len(FIXED_IDS) <= len(field)
    config = TournamentConfig(
        bot_specs=field,
        games=args.games,
        root_seed=args.seed,
        batch_size=args.batch_size,
        bootstrap_samples=0,
    )
    started = time.perf_counter()
    plan = TournamentPlanner.plan(config)
    result = MonteCarloRunner.run_jobs(
        plan.monte_carlo_config,
        plan.jobs,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    fit = fit_plackett_luce(
        observations_from_games(result.game_summaries),
        tuple(spec.bot_id for spec in field),
    )
    analysis = analyze_tournament(result, fit)
    rows = analysis.rows_by_id

    pair_results: dict[str, dict[str, float | int]] = {}
    for first_index, first in enumerate(FIXED_IDS):
        for second in FIXED_IDS[first_index + 1 :]:
            scores: list[float] = []
            for game in result.game_summaries:
                if first not in game.bot_ids or second not in game.bot_ids:
                    continue
                rank_by_id = {game.bot_ids[score.seat]: score.rank for score in game.scores}
                if rank_by_id[first] < rank_by_id[second]:
                    scores.append(1.0)
                elif rank_by_id[first] == rank_by_id[second]:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            mean = statistics.mean(scores)
            se = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
            pair_results[f"{first}__vs__{second}"] = {
                "games": len(scores),
                "first_score": mean,
                "se": se,
                "ci95_lower": mean - 1.96 * se,
                "ci95_upper": mean + 1.96 * se,
            }

    payload = {
        "games": args.games,
        "seed": args.seed,
        "field_size": len(field),
        "fixed_count": len(FIXED_IDS),
        "fixed_fraction": len(FIXED_IDS) / len(field),
        "runtime_seconds": time.perf_counter() - started,
        "fixed_rows": {
            bot_id: {
                "rank": rows[bot_id].rank,
                "rating": rows[bot_id].pl_rating,
                "games": rows[bot_id].games,
                "wins": rows[bot_id].outright_wins,
                "finish": rows[bot_id].mean_normalized_finish,
                "faults": rows[bot_id].faults,
            }
            for bot_id in FIXED_IDS
        },
        "pairs": pair_results,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
