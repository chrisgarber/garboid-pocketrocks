from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time

from _fixed_search_bot import SearchFixedBrain

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_BOT_SPEC
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
)
from garboid_pocketrocks.bots.random_bot import RandomBot
from garboid_pocketrocks.bots.sdk_samples import SDK_GREEDY_VALUE_V1_BOT_SPEC
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner
from garboid_pocketrocks.tournament.analysis import analyze_tournament
from garboid_pocketrocks.tournament.rating import fit_plackett_luce, observations_from_games
from garboid_pocketrocks.tournament.schedule import TournamentConfig, TournamentPlanner


def _field() -> tuple[BotSpec, ...]:
    return (
        BotSpec.from_bot_class(RandomBot),
        FIXED_BID_BOT_SPEC,
        BotSpec.for_simulation("fixed-search", SearchFixedBrain),
        AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
        BALANCED_HEURISTIC_V1_BOT_SPEC,
        PASSIVE_HEURISTIC_V1_BOT_SPEC,
        AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
        BALANCED_HEURISTIC_V2_BOT_SPEC,
        PASSIVE_HEURISTIC_V2_BOT_SPEC,
        SDK_GREEDY_VALUE_V1_BOT_SPEC,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("values", help="six comma-separated bids")
    parser.add_argument("--games", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=24_701)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    values = tuple(int(value) for value in args.values.split(","))
    if len(values) != 6 or any(value < 0 for value in values):
        raise SystemExit("values must be six nonnegative integers")
    os.environ["GARBOID_FIXED_SEARCH_VALUES"] = ",".join(map(str, values))

    field = _field()
    fixed_count = sum(spec.bot_id in {"fixed-bid", "fixed-search"} for spec in field)
    assert 4 * fixed_count <= len(field)
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

    paired_scores: list[float] = []
    for game in result.game_summaries:
        if "fixed-bid" not in game.bot_ids or "fixed-search" not in game.bot_ids:
            continue
        rank_by_id = {game.bot_ids[score.seat]: score.rank for score in game.scores}
        candidate_rank = rank_by_id["fixed-search"]
        baseline_rank = rank_by_id["fixed-bid"]
        if candidate_rank < baseline_rank:
            paired_scores.append(1.0)
        elif candidate_rank == baseline_rank:
            paired_scores.append(0.5)
        else:
            paired_scores.append(0.0)
    paired_mean = statistics.mean(paired_scores)
    paired_se = (
        statistics.stdev(paired_scores) / math.sqrt(len(paired_scores))
        if len(paired_scores) > 1
        else 0.0
    )
    candidate = rows["fixed-search"]
    baseline = rows["fixed-bid"]
    payload = {
        "values": values,
        "games": args.games,
        "seed": args.seed,
        "field_size": len(field),
        "fixed_count": fixed_count,
        "fixed_fraction": fixed_count / len(field),
        "candidate_rating": candidate.pl_rating,
        "baseline_rating": baseline.pl_rating,
        "rating_delta": candidate.pl_rating - baseline.pl_rating,
        "candidate_rank": candidate.rank,
        "baseline_rank": baseline.rank,
        "candidate_games": candidate.games,
        "baseline_games": baseline.games,
        "candidate_finish": candidate.mean_normalized_finish,
        "baseline_finish": baseline.mean_normalized_finish,
        "candidate_wins": candidate.outright_wins,
        "baseline_wins": baseline.outright_wins,
        "paired_games": len(paired_scores),
        "paired_score": paired_mean,
        "paired_se": paired_se,
        "faults": candidate.faults + baseline.faults,
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
