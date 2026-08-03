"""Export and tournament-test one durable neural training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from garboid_pocketrocks.neural.opponent_pool import STRONG_FIELD_POOL_V1
from garboid_pocketrocks.neural.tournament_bot import checkpoint_bot_spec
from garboid_pocketrocks.neural.trainer import inspect_checkpoint
from garboid_pocketrocks.neural.training_checkpoint import export_inference_checkpoint
from garboid_pocketrocks.tournament.runner import TournamentRunner
from garboid_pocketrocks.tournament.schedule import TournamentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--decision-reports", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()

    inspected = inspect_checkpoint(arguments.checkpoint)
    bot_id = inspected["bot_id"]
    if not isinstance(bot_id, str):
        raise TypeError("checkpoint bot ID must be a string")
    inference = export_inference_checkpoint(
        arguments.checkpoint,
        arguments.inference_checkpoint,
        device=torch.device("cpu"),
    )
    unique_opponents = {spec.name: spec for spec in STRONG_FIELD_POOL_V1}
    candidate = checkpoint_bot_spec(bot_id, inference)
    run = TournamentRunner.run(
        TournamentConfig(
            bot_specs=(candidate, *unique_opponents.values()),
            games=arguments.games,
            root_seed=arguments.seed,
            batch_size=64,
            bootstrap_samples=arguments.bootstrap_samples,
            decision_reports=arguments.decision_reports,
        ),
        workers=arguments.workers,
        output_dir=arguments.output_dir,
    )
    for row in run.analysis.rows:
        print(
            f"{row.rank}\t{row.bot_name}\t{row.pl_rating:.2f}\t"
            f"{row.outright_wins / row.games:.3f}\t{row.mean_final_money:.2f}\t"
            f"faults={row.faults}",
            flush=True,
        )
    print(f"report={run.artifacts.report_html.resolve()}", flush=True)


if __name__ == "__main__":
    main()
