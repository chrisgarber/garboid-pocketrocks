"""Command-line entry point for tournament insight reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from garboid_pocketrocks.visualizer.analysis import (
    BotInsightsEngine,
    TournamentInsightsEngine,
)
from garboid_pocketrocks.visualizer.loading import TournamentDataset, TournamentDatasetError
from garboid_pocketrocks.visualizer.reporting import (
    render_insights_html,
    write_insights_html,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="garboid-visualize",
        description="Build tournament-wide and single-bot insight visualizations.",
    )
    parser.add_argument("tournament_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: TOURNAMENT_DIR/insights.html).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        dataset = TournamentDataset.load(args.tournament_dir)
        tournament = TournamentInsightsEngine(dataset).build()
        bots = BotInsightsEngine(dataset).build_all()
        output = args.output or dataset.source_dir / "insights.html"
        write_insights_html(
            output,
            render_insights_html(tournament, bots),
            overwrite=args.overwrite,
        )
    except (FileExistsError, TournamentDatasetError, ValueError) as error:
        parser.error(str(error))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
