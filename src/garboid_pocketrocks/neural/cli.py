"""Command-line entry point for neural smoke, train, and checkpoint tools."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from garboid_pocketrocks.neural.run_config import TrainingRunConfig
from garboid_pocketrocks.neural.smoke import (
    SmokeConfig,
    run_self_play_smoke,
    run_smoke,
    smoke_run_config,
)
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
        if arguments.command == "evaluate":
            payload = inspect_checkpoint(arguments.checkpoint)
            payload["evaluation_config"] = str(arguments.config)
            arguments.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote checkpoint evaluation metadata: {arguments.output}")
            return 0
        raise AssertionError("argparse returned an unknown command")
    except (OSError, RuntimeError, ValueError) as error:
        parser.print_usage(sys.stderr)
        print(f"garboid-train: error: {error}", file=sys.stderr)
        return 2


def _smoke(arguments: argparse.Namespace) -> int:
    if arguments.updates is not None or arguments.games_per_update is not None:
        legacy_config = SmokeConfig(
            root_seed=arguments.seed,
            updates=arguments.updates or 2,
            games_per_update=arguments.games_per_update or 16,
            device=arguments.device or "cpu",
        )
        legacy_result = run_smoke(legacy_config, arguments.output_dir)
        games = legacy_config.updates * legacy_config.games_per_update
        print(
            f"completed {legacy_config.updates} legacy updates and {games} games; "
            f"checkpoint: {legacy_result.output_dir / 'checkpoint'}"
        )
        return 0
    self_play_config = smoke_run_config()
    workers = self_play_config.parallel.workers if arguments.workers is None else arguments.workers
    resolved = replace(
        self_play_config,
        root_seed=arguments.seed,
        device=arguments.device or self_play_config.device,
        games_per_cell=arguments.games_per_cell or self_play_config.games_per_cell,
        parallel=replace(self_play_config.parallel, workers=workers),
    )
    self_play_result = run_self_play_smoke(resolved, arguments.output_dir)
    print(
        f"completed {self_play_result.completed_episodes} games "
        f"({self_play_result.games_per_second:.2f} games/s, "
        f"{self_play_result.decisions_per_second:.2f} decisions/s); "
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
    smoke.add_argument("--games-per-cell", type=_positive_int)
    smoke.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    smoke.add_argument("--workers", type=_workers)
    smoke.add_argument(
        "--updates",
        type=_positive_int,
        help="legacy Stage 1 updates (legacy default: 2)",
    )
    smoke.add_argument(
        "--games-per-update",
        type=_positive_int,
        help="legacy Stage 1 games (legacy default: 16)",
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

    evaluate = commands.add_parser("evaluate", help="write evaluation metadata")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

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
