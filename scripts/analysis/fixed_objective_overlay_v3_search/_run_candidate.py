from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

from _search_bot import SearchFixedObjectiveOverlayV3Brain

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.registry import BOT_SPECS_BY_NAME
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner
from garboid_pocketrocks.tournament.analysis import analyze_tournament
from garboid_pocketrocks.tournament.rating import fit_plackett_luce, observations_from_games
from garboid_pocketrocks.tournament.schedule import TournamentConfig, TournamentPlanner

_CANDIDATE_ID = "fixed-objective-overlay-v3-search"
_PREDECESSOR_ID = "fixed-objective-overlay-v2"
_FIELD_NAMES = (
    "fixed-objective-overlay-v2",
    "fixed-objective-overlay-v1",
    "fixed-bid-tuned-v1",
    "aggressive-v2",
    "fixed-bid-diverse-v1",
    "balanced-v2",
    "fixed-bid",
    "passive-v2",
    "passive-v1",
    "aggressive-v3",
    "balanced-v3",
    "passive-v3",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("values", help="six comma-separated fixed action targets")
    parser.add_argument("--games", type=int, default=1_200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    values = tuple(int(value) for value in args.values.split(","))
    if len(values) != 6 or any(value < 0 for value in values):
        raise SystemExit("values must be six nonnegative integers")
    os.environ["GARBOID_FIXED_OBJECTIVE_OVERLAY_V3_SEARCH_VALUES"] = ",".join(map(str, values))
    candidate_spec = BotSpec.for_simulation(
        _CANDIDATE_ID,
        SearchFixedObjectiveOverlayV3Brain,
    )
    field = (candidate_spec, *(BOT_SPECS_BY_NAME[name] for name in _FIELD_NAMES))
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
    candidate = analysis.rows_by_id[_CANDIDATE_ID]
    predecessor = analysis.rows_by_id[_PREDECESSOR_ID]

    paired_scores: list[float] = []
    for game in result.game_summaries:
        if _CANDIDATE_ID not in game.bot_ids or _PREDECESSOR_ID not in game.bot_ids:
            continue
        rank_by_id = {game.bot_ids[score.seat]: score.rank for score in game.scores}
        candidate_rank = rank_by_id[_CANDIDATE_ID]
        predecessor_rank = rank_by_id[_PREDECESSOR_ID]
        if candidate_rank < predecessor_rank:
            paired_scores.append(1.0)
        elif candidate_rank == predecessor_rank:
            paired_scores.append(0.5)
        else:
            paired_scores.append(0.0)
    paired_score = statistics.mean(paired_scores)
    paired_se = (
        statistics.stdev(paired_scores) / math.sqrt(len(paired_scores))
        if len(paired_scores) > 1
        else 0.0
    )
    print(
        json.dumps(
            {
                "values": values,
                "games": args.games,
                "seed": args.seed,
                "candidate_rating": candidate.pl_rating,
                "predecessor_rating": predecessor.pl_rating,
                "rating_delta": candidate.pl_rating - predecessor.pl_rating,
                "candidate_rank": candidate.rank,
                "predecessor_rank": predecessor.rank,
                "candidate_appearances": candidate.games,
                "candidate_win_rate": candidate.outright_wins / candidate.games,
                "candidate_mean_finish": candidate.mean_normalized_finish,
                "candidate_mean_money": candidate.mean_final_money,
                "predecessor_win_rate": predecessor.outright_wins / predecessor.games,
                "predecessor_mean_finish": predecessor.mean_normalized_finish,
                "predecessor_mean_money": predecessor.mean_final_money,
                "paired_games": len(paired_scores),
                "paired_score": paired_score,
                "paired_se": paired_se,
                "faults": candidate.faults + predecessor.faults,
                "runtime_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
