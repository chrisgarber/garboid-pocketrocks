"""Command-line entry point for neural smoke, train, and checkpoint tools."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from garboid_pocketrocks.neural.run_config import TrainingRunConfig
from garboid_pocketrocks.neural.smoke import run_smoke, smoke_run_config
from garboid_pocketrocks.neural.trainer import (
    inspect_checkpoint,
    resume,
    train,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one neural training command and return a process exit code."""

    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    try:
        if arguments.command == "smoke":
            return _smoke(arguments)
        if arguments.command == "train":
            result = train(
                TrainingRunConfig.from_json(arguments.config),
                arguments.output_dir,
            )
            print(
                f"completed {result.completed_updates} updates and "
                f"{result.completed_episodes} games; "
                f"checkpoint: {result.final_checkpoint}"
            )
            return 0
        if arguments.command == "resume":
            result = resume(
                arguments.checkpoint,
                arguments.output_dir,
                max_additional_updates=arguments.max_additional_updates,
                config_override=(
                    None
                    if arguments.config is None
                    else TrainingRunConfig.from_json(arguments.config)
                ),
            )
            print(
                f"resumed through update {result.completed_updates}; "
                f"checkpoint: {result.final_checkpoint}"
            )
            return 0
        if arguments.command == "inspect":
            payload = inspect_checkpoint(arguments.checkpoint)
            if arguments.format == "json":
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                for key, value in payload.items():
                    print(f"{key}: {value}")
            return 0
        raise AssertionError("argparse returned an unknown command")
    except (OSError, RuntimeError, ValueError) as error:
        parser.print_usage(sys.stderr)
        print(f"garboid-train: error: {error}", file=sys.stderr)
        return 2


def _smoke(arguments: argparse.Namespace) -> int:
    config = smoke_run_config()
    workers = config.parallel.workers if arguments.workers is None else arguments.workers
    resolved = replace(
        config,
        root_seed=arguments.seed,
        device=arguments.device or config.device,
        games_per_cell=arguments.games_per_cell or config.games_per_cell,
        parallel=replace(config.parallel, workers=workers),
    )
    result = run_smoke(resolved, arguments.output_dir)
    print(
        f"completed {result.completed_episodes} games "
        f"({result.games_per_second:.2f} games/s, "
        f"{result.decisions_per_second:.2f} decisions/s); "
        f"checkpoint: {arguments.output_dir / 'checkpoints/latest'}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garboid-train")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke",
        help="run the balanced A-E / 3-5-player self-play smoke",
    )
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=42, help="root seed (default: 42)")
    smoke.add_argument(
        "--games-per-cell",
        type=_positive_int,
        help="games per ruleset/player-count cell",
    )
    smoke.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    smoke.add_argument(
        "--workers",
        type=_workers,
        help="collector worker count or auto",
    )

    train_parser = commands.add_parser("train", help="start a durable run")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)

    resume_parser = commands.add_parser("resume", help="resume a training checkpoint")
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--output-dir", type=Path, required=True)
    resume_parser.add_argument("--config", type=Path)
    resume_parser.add_argument(
        "--max-additional-updates",
        type=_positive_int,
    )

    inspect = commands.add_parser("inspect", help="inspect a training checkpoint")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _workers(raw: str) -> int | str:
    if raw == "auto":
        return raw
    return _positive_int(raw)


if __name__ == "__main__":
    raise SystemExit(main())
