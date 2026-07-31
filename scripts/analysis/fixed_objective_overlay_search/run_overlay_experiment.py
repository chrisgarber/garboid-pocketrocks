from __future__ import annotations

import argparse
import json
from pathlib import Path

from fixed_objective_overlay_v1 import (
    FixedObjectiveOverlayNarrowV1Brain,
    FixedObjectiveOverlayPrivateHeavyV1Brain,
    FixedObjectiveOverlayStandardV1Brain,
)

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.fixed_bid import FIXED_BID_BOT_SPEC
from garboid_pocketrocks.bots.heuristic import (
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    AggressiveHeuristicV2Brain,
    BalancedHeuristicV2Brain,
    PassiveHeuristicV2Brain,
)
from garboid_pocketrocks.bots.random_bot import RandomBot
from garboid_pocketrocks.bots.sdk_samples import SDK_GREEDY_VALUE_V1_BOT_SPEC
from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig

NARROW_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-narrow-v1", FixedObjectiveOverlayNarrowV1Brain
)
STANDARD_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-v1", FixedObjectiveOverlayStandardV1Brain
)
PRIVATE_HEAVY_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-private-heavy-v1", FixedObjectiveOverlayPrivateHeavyV1Brain
)
FROZEN_SPEC = BotSpec.for_simulation(
    "fixed-objective-overlay-v1", FixedObjectiveOverlayPrivateHeavyV1Brain
)

NON_FIXED = (
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    SDK_GREEDY_VALUE_V1_BOT_SPEC,
    BotSpec.from_bot_class(RandomBot),
)

SCREEN_ONLY_NON_FIXED = (
    BotSpec.for_simulation("aggressive-v2-field-b", AggressiveHeuristicV2Brain),
    BotSpec.for_simulation("balanced-v2-field-b", BalancedHeuristicV2Brain),
    BotSpec.for_simulation("passive-v2-field-b", PassiveHeuristicV2Brain),
    BotSpec.for_simulation("balanced-v2-field-c", BalancedHeuristicV2Brain),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "validate"), required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = (
        (NARROW_SPEC, STANDARD_SPEC, PRIVATE_HEAVY_SPEC)
        if args.phase == "screen"
        else (FROZEN_SPEC,)
    )
    specs = (
        (FIXED_BID_BOT_SPEC, *candidates, *NON_FIXED, *SCREEN_ONLY_NON_FIXED)
        if args.phase == "screen"
        else (FIXED_BID_BOT_SPEC, *candidates, *NON_FIXED)
    )
    run = TournamentRunner.run(
        TournamentConfig(
            bot_specs=specs,
            games=args.games,
            player_counts=(3, 4, 5),
            charts=("A", "B", "C", "D", "E"),
            root_seed=args.seed,
            batch_size=64,
            bootstrap_samples=200,
        ),
        workers=args.workers,
        output_dir=args.output,
        overwrite=True,
    )
    intervals = {interval.bot_id: interval for interval in run.bootstrap.intervals}
    output = []
    stats = {item.bot_id: item for item in run.monte_carlo_result.bot_statistics}
    for row in run.analysis.rows:
        stat = stats[row.bot_id]
        interval = intervals.get(row.bot_id)
        output.append(
            {
                "rank": row.rank,
                "bot": row.bot_id,
                "rating": row.pl_rating,
                "rating_95": None if interval is None else [interval.lower, interval.upper],
                "games": row.games,
                "wins": row.outright_wins,
                "win_rate": row.outright_wins / row.games,
                "mean_rank": stat.mean_rank(),
                "mean_money": row.mean_final_money,
                "mean_bid": stat.behavior.mean_nonzero_bid(),
                "pass_rate": stat.behavior.pass_rate(),
                "objectives": stat.behavior.objectives_claimed,
                "faults": row.faults,
            }
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
