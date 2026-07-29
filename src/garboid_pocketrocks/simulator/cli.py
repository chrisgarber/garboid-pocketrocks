from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    BotSpec,
    PassiveHeuristicBot,
    RandomBot,
)
from garboid_pocketrocks.rules import live_ruleset
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.replay import save_replay
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler

_BOT_REGISTRY = {
    RandomBot.BOT_NAME: BotSpec.from_bot_class(RandomBot),
    AggressiveHeuristicBot.BOT_NAME: BotSpec.from_bot_class(AggressiveHeuristicBot),
    BalancedHeuristicBot.BOT_NAME: BotSpec.from_bot_class(BalancedHeuristicBot),
    PassiveHeuristicBot.BOT_NAME: BotSpec.from_bot_class(PassiveHeuristicBot),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _bot_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one bot name is required")
    unknown = tuple(name for name in names if name not in _BOT_REGISTRY)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown bot name(s): {', '.join(sorted(set(unknown)))}")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic local PocketRocks Monte Carlo matches"
    )
    parser.add_argument(
        "--bots",
        required=True,
        type=_bot_names,
        help="comma-separated registered bot names (random, aggressive, balanced, passive)",
    )
    parser.add_argument("--games", required=True, type=_positive_int)
    parser.add_argument("--players", required=True, type=int, choices=(3, 4, 5))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ruleset",
        choices=tuple(f"live-{chart}" for chart in "ABCDE"),
        default="live-A",
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--replay-dir",
        type=Path,
        help="optional directory for deterministic per-game replay JSON",
    )
    return parser


def _json_payload(
    args: argparse.Namespace,
    result: MonteCarloResult,
) -> dict[str, Any]:
    return {
        "configuration": {
            "bots": list(args.bots),
            "games": args.games,
            "players": args.players,
            "seed": args.seed,
            "ruleset": args.ruleset,
            "workers": args.workers,
        },
        "result": asdict(result),
    }


def _table(result: MonteCarloResult) -> str:
    headings = (
        "bot",
        "games",
        "wins",
        "ties",
        "mean_rank",
        "mean_money",
        "pass_rate",
        "mean_bid",
        "resource_wins",
        "objectives",
    )
    rows = [
        (
            statistics.bot_name,
            str(statistics.games),
            str(statistics.outright_wins),
            str(statistics.first_place_ties),
            f"{statistics.mean_rank():.3f}",
            f"{statistics.mean_final_money():.3f}",
            f"{statistics.behavior.pass_rate():.3f}",
            f"{statistics.behavior.mean_nonzero_bid():.3f}",
            str(statistics.behavior.resource_cards_won),
            str(statistics.behavior.objectives_claimed),
        )
        for statistics in result.bot_statistics
    ]
    widths = tuple(
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    )
    lines = [
        "  ".join(heading.ljust(widths[index]) for index, heading in enumerate(headings)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if len(args.bots) < args.players:
        parser.error(
            f"--bots supplies {len(args.bots)} entries but --players requires "
            f"at least {args.players}"
        )
    ruleset = live_ruleset(args.ruleset.removeprefix("live-"))
    config = MonteCarloConfig(
        bot_specs=tuple(_BOT_REGISTRY[name] for name in args.bots),
        games=args.games,
        player_counts=(args.players,),
        ruleset_sampler=FixedRulesetSampler(ruleset),
        root_seed=args.seed,
        capture_replays=args.replay_dir is not None,
    )
    result = MonteCarloRunner.run(config, workers=args.workers)
    if args.replay_dir is not None:
        args.replay_dir.mkdir(parents=True, exist_ok=True)
        for replay in result.replays:
            assert replay.game_index is not None
            save_replay(
                replay,
                args.replay_dir / f"game-{replay.game_index:06d}.json",
            )
    if args.format == "json":
        print(json.dumps(_json_payload(args, result), sort_keys=True))
    else:
        print(_table(result))
