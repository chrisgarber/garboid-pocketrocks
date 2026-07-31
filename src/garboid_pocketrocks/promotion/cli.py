"""Command-line entry point for the held-out bot promotion gate."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpusError,
    load_promotion_corpus,
)
from garboid_pocketrocks.promotion.runner import (
    PromotionRunConfig,
    PromotionRunner,
)

_BOT_REGISTRY = BOT_SPECS_BY_NAME
_DEFAULT_DEVELOPMENT_CORPUS = Path("configs/promotion/development-v1.json")
_DEFAULT_HELD_OUT_CORPUS = Path("configs/promotion/held-out-v1.json")
_DEFAULT_OUTPUT_DIR = Path("promotion-results")


class _InvocationError(ValueError):
    """An invalid command line that should produce usage and exit code two."""


class _PromotionArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _InvocationError(message)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = _PromotionArgumentParser(
        description=(
            "Compare a candidate bot with an incumbent using paired held-out games "
            "that were not used for tuning and a deterministic bootstrap 95% interval."
        )
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="registered candidate bot that may replace the incumbent",
    )
    parser.add_argument(
        "--incumbent",
        required=True,
        help="registered incumbent bot the candidate must outperform",
    )
    parser.add_argument(
        "--development-corpus",
        type=Path,
        default=_DEFAULT_DEVELOPMENT_CORPUS,
        help="development corpus used for tuning, checked for separation from held-out games",
    )
    parser.add_argument(
        "--held-out-corpus",
        type=Path,
        default=_DEFAULT_HELD_OUT_CORPUS,
        help="held-out corpus of final-exam games that were not used for tuning",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=1_000,
        help="number of paired resamples used to calculate the bootstrap 95%% interval",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="seed that makes the bootstrap interval repeatable",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="positive number of worker processes",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=64,
        help="positive number of games grouped into each simulator batch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="output directory for promotion-report.json and supporting evidence",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a prior promotion artifact generation in the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the promotion command and return its stable process exit code."""

    parser = _parser()
    report_path = _requested_output_dir(argv) / "promotion-report.json"
    try:
        args = parser.parse_args(argv)
        report_path = args.output_dir / "promotion-report.json"
        candidate = _registered_bot(args.candidate)
        incumbent = _registered_bot(args.incumbent)
        _require_distinct_compared_bots(candidate, incumbent)
        development = load_promotion_corpus(
            args.development_corpus,
            registry=_BOT_REGISTRY,
        )
        held_out = load_promotion_corpus(
            args.held_out_corpus,
            registry=_BOT_REGISTRY,
        )
        run = PromotionRunner.run(
            PromotionRunConfig(
                candidate=candidate,
                incumbent=incumbent,
                development=development,
                held_out=held_out,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                batch_size=args.batch_size,
            ),
            registry=_BOT_REGISTRY,
            workers=args.workers,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (_InvocationError, PromotionCorpusError, OSError, RuntimeError, ValueError) as error:
        parser.print_usage(file=sys.stderr)
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 2

    if run.report.analysis.promoted:
        print(f"{candidate.name} passed the held-out final exam.")
        interval = run.report.analysis.interval
        assert interval is not None
        print(
            f"95% uncertainty interval: {interval.lower:.2f} to {interval.upper:.2f} rating points."
        )
        print(f"Report: {run.artifacts.report_json}")
        return 0
    print(f"{candidate.name} did not pass the held-out final exam.")
    for failure in run.report.analysis.failures:
        print(f"- {failure.message}")
    print(f"Report: {run.artifacts.report_json}")
    return 1


def _registered_bot(name: str) -> BotSpec:
    try:
        return _BOT_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown bot name: {name}") from error


def _require_distinct_compared_bots(candidate: BotSpec, incumbent: BotSpec) -> None:
    if candidate.name == incumbent.name or candidate.bot_id == incumbent.bot_id:
        raise ValueError("candidate and incumbent must have different names and bot IDs")


def _requested_output_dir(argv: Sequence[str] | None) -> Path:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    output_dir = _DEFAULT_OUTPUT_DIR
    for index, value in enumerate(arguments):
        if value == "--output-dir":
            if index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
                output_dir = Path(arguments[index + 1])
        elif value.startswith("--output-dir="):
            output_dir = Path(value.partition("=")[2])
    return output_dir
