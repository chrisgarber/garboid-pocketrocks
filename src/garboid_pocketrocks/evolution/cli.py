"""Execution-only command for deterministic heuristic evolution."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME
from garboid_pocketrocks.evolution.manifest import (
    SearchManifestError,
    load_search_manifest,
)
from garboid_pocketrocks.evolution.reporting import (
    build_search_report,
    repository_commit,
    validate_search_output_dir,
    write_search_artifacts,
)
from garboid_pocketrocks.evolution.runner import SearchRunError, run_search
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpusError,
    load_promotion_corpus,
)

_BOT_REGISTRY = BOT_SPECS_BY_NAME
_DEFAULT_DEVELOPMENT_CORPUS = Path("configs/promotion/historical/development-v1-17c01635.json")
_DEFAULT_OUTPUT_DIR = Path("artifacts/evolution")


class _InvocationError(ValueError):
    """An invalid command line that should return exit code two."""


class _EvolutionArgumentParser(argparse.ArgumentParser):
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
    parser = _EvolutionArgumentParser(
        description=(
            "Execute a fixed heuristic evolution manifest on development games. "
            "A positive winner is frozen only for a later, separate held-out "
            "promotion evaluation."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="versioned search manifest containing every strategy and evolution control",
    )
    parser.add_argument(
        "--development-corpus",
        type=Path,
        default=_DEFAULT_DEVELOPMENT_CORPUS,
        help="development corpus used for tuning; held-out games are forbidden",
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
        help=(
            "local output directory for search-report.json and supporting evidence; "
            "defaults under the gitignored artifacts tree"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace a prior search artifact generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one manifest and return the stable search process exit code."""

    parser = _parser()
    report_path = _requested_output_dir(argv) / "search-report.json"
    try:
        args = parser.parse_args(argv)
        report_path = args.output_dir / "search-report.json"
        validate_search_output_dir(args.output_dir, overwrite=args.overwrite)
        corpus = load_promotion_corpus(
            args.development_corpus,
            registry=_BOT_REGISTRY,
        )
        manifest = load_search_manifest(
            args.manifest,
            development_corpus=corpus,
        )
        commit = repository_commit()
        run = run_search(
            manifest,
            corpus,
            registry=_BOT_REGISTRY,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        report = build_search_report(
            run,
            repository_commit=commit,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        artifacts = write_search_artifacts(
            args.output_dir,
            report=report,
            overwrite=args.overwrite,
        )
    except (
        _InvocationError,
        PromotionCorpusError,
        SearchManifestError,
        SearchRunError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.print_usage(file=sys.stderr)
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 2

    if run.failures:
        print("Search evidence was invalid; no candidate was frozen.")
        for failure in run.failures:
            print(f"- {failure.message}")
        print(f"Report: {artifacts.search_report_json}")
        return 2
    if run.frozen_candidate is not None:
        print(
            f"{run.frozen_candidate.identity} showed a positive development improvement "
            "and was frozen for held-out promotion evaluation."
        )
        print(f"Report: {artifacts.search_report_json}")
        return 0
    print("Search completed without a positive development improvement; no candidate was frozen.")
    print(f"Report: {artifacts.search_report_json}")
    return 1


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
