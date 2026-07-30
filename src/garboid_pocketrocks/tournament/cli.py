from __future__ import annotations

import argparse
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

from garboid_pocketrocks.bots import (
    BOT_SPECS_BY_NAME,
    DEFAULT_TOURNAMENT_BOT_SPECS,
    BotSpec,
)
from garboid_pocketrocks.tournament.runner import TournamentRun, TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("comma-separated values must be unique")
    return values


def _players(value: str) -> tuple[int, ...]:
    try:
        players = tuple(int(item) for item in _csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("players must be comma-separated integers") from error
    if any(player_count not in (3, 4, 5) for player_count in players):
        raise argparse.ArgumentTypeError("players must contain only 3, 4, and 5")
    return players


def _charts(value: str) -> tuple[str, ...]:
    charts = tuple(item.upper() for item in _csv(value))
    if any(chart not in "ABCDE" for chart in charts):
        raise argparse.ArgumentTypeError("charts must contain only A, B, C, D, and E")
    return charts


def _secure_randbits(width: int) -> int:
    return secrets.randbits(width)


def _resolve_root_seed(
    requested_seed: int | None,
    *,
    decision_reports: bool,
) -> int:
    if requested_seed is not None:
        return requested_seed
    if decision_reports:
        return _secure_randbits(63)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank registered PocketRocks bots with a multiplayer Plackett-Luce model"
    )
    parser.add_argument("--games", type=_positive_int, default=15_000)
    parser.add_argument("--players", type=_players, default=(3, 4, 5))
    parser.add_argument("--charts", type=_charts, default=("A", "B", "C", "D", "E"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--bootstrap-samples", type=_nonnegative_int, default=200)
    parser.add_argument("--bots", type=_csv)
    parser.add_argument("--exclude-bots", type=_csv, default=())
    parser.add_argument("--output-dir", type=Path, default=Path("tournament-results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decision-reports", action="store_true")
    return parser


def _resolve_bot_specs(
    *,
    include: tuple[str, ...] | None,
    exclude: tuple[str, ...],
    registry: Mapping[str, BotSpec],
    defaults: tuple[BotSpec, ...] | None = None,
) -> tuple[BotSpec, ...]:
    requested = (
        tuple(spec.name for spec in defaults)
        if include is None and defaults is not None
        else tuple(registry)
        if include is None
        else include
    )
    unknown = (set(requested) | set(exclude)) - set(registry)
    if unknown:
        raise ValueError(f"unknown bot name(s): {', '.join(sorted(unknown))}")
    excluded = set(exclude)
    selected = tuple(registry[name] for name in requested if name not in excluded)
    if not selected:
        raise ValueError("at least one bot must remain after filtering")
    return selected


def _leaderboard(run: TournamentRun) -> str:
    intervals = {item.bot_id: item for item in run.bootstrap.intervals}
    headings = (
        "rank",
        "bot",
        "PL_rating",
        "95%_interval",
        "games",
        "win_rate",
        "mean_money",
        "faults",
    )
    rows = []
    for row in run.analysis.rows:
        interval = intervals.get(row.bot_id)
        interval_text = f"{interval.lower:.1f}..{interval.upper:.1f}" if interval else "n/a"
        rows.append(
            (
                str(row.rank),
                row.bot_name,
                f"{row.pl_rating:.2f}",
                interval_text,
                str(row.games),
                f"{row.outright_wins / row.games:.3f}",
                f"{row.mean_final_money:.2f}",
                str(row.faults),
            )
        )
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
    try:
        bot_specs = _resolve_bot_specs(
            include=args.bots,
            exclude=args.exclude_bots,
            registry=BOT_SPECS_BY_NAME,
            defaults=DEFAULT_TOURNAMENT_BOT_SPECS,
        )
        config = TournamentConfig(
            bot_specs=bot_specs,
            games=args.games,
            player_counts=args.players,
            charts=args.charts,
            root_seed=_resolve_root_seed(
                args.seed,
                decision_reports=args.decision_reports,
            ),
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
            decision_reports=args.decision_reports,
        )
        run = TournamentRunner.run(
            config,
            workers=args.workers,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(_leaderboard(run))
    print(f"\nArtifacts: {run.artifacts.report_html.parent}")
    if run.config.decision_reports:
        for path in (
            run.artifacts.game_summaries_jsonl,
            run.artifacts.decision_traces_jsonl,
            run.artifacts.decision_slices_csv,
        ):
            if path is not None:
                print(path)
