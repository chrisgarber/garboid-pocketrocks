"""Command-line entry point for the Stage 1 neural mechanics smoke."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from garboid_pocketrocks.neural.smoke import SmokeConfig, SmokeError, run_smoke


def main(argv: Sequence[str] | None = None) -> int:
    """Run the only Stage 1 neural command."""

    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    try:
        config = SmokeConfig(
            root_seed=arguments.seed,
            updates=arguments.updates,
            games_per_update=arguments.games_per_update,
            device=arguments.device,
        )
        result = run_smoke(config, arguments.output_dir)
    except SmokeError as error:
        parser.print_usage(sys.stderr)
        print(f"garboid-train: error: {error}", file=sys.stderr)
        return 2
    games = result.config.updates * result.config.games_per_update
    print(
        f"completed {result.config.updates} updates and {games} games on CPU; "
        f"checkpoint: {result.output_dir / 'checkpoint'}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garboid-train")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke",
        help="run deterministic Stage 1 PPO mechanics on CPU",
    )
    smoke.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty artifact directory",
    )
    smoke.add_argument(
        "--seed",
        type=int,
        default=42,
        help="root seed (default: 42)",
    )
    smoke.add_argument(
        "--updates",
        type=_positive_int,
        default=2,
        help="PPO updates (default: 2)",
    )
    smoke.add_argument(
        "--games-per-update",
        type=_positive_int,
        default=16,
        help="complete games per update (default: 16)",
    )
    smoke.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="fixed Stage 1 device (default: cpu)",
    )
    return parser


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
