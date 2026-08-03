from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

type Values = tuple[int, int, int, int, int, int]

_BASELINE: Values = (5, 10, 2, 5, 4, 7)
_RANGES: tuple[tuple[int, ...], ...] = (
    (3, 4, 5),
    (7, 8, 9, 10),
    (0, 1, 2, 3, 4),
    (3, 4, 5, 6, 7),
    (2, 3, 4),
    (4, 5, 6, 7),
)
_SEEDS = (20_260_804_01, 20_260_804_02)


def _neighbors(values: Values) -> tuple[Values, ...]:
    candidates: set[Values] = {values}
    for index, choices in enumerate(_RANGES):
        for choice in choices:
            candidate = list(values)
            candidate[index] = choice
            candidates.add(tuple(candidate))  # type: ignore[arg-type]
    return tuple(sorted(candidates))


def _run(values: Values, *, games: int, workers: int, seed: int) -> dict[str, object]:
    runner = Path(__file__).with_name("_run_candidate.py")
    command = (
        sys.executable,
        str(runner),
        ",".join(map(str, values)),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _aggregate(values: Values, runs: list[dict[str, object]]) -> dict[str, object]:
    matching = [row for row in runs if tuple(row["values"]) == values]
    paired_games = sum(int(row["paired_games"]) for row in matching)
    return {
        "values": values,
        "mean_rating_delta": sum(float(row["rating_delta"]) for row in matching) / len(matching),
        "paired_games": paired_games,
        "paired_score": sum(
            float(row["paired_score"]) * int(row["paired_games"]) for row in matching
        )
        / paired_games,
        "mean_candidate_win_rate": sum(float(row["candidate_win_rate"]) for row in matching)
        / len(matching),
        "mean_candidate_money": sum(float(row["candidate_mean_money"]) for row in matching)
        / len(matching),
        "faults": sum(int(row["faults"]) for row in matching),
    }


def _evaluate(
    candidates: tuple[Values, ...],
    *,
    games: int,
    workers: int,
    label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs: list[dict[str, object]] = []
    total = len(candidates) * len(_SEEDS)
    for index, values in enumerate(candidates, start=1):
        for seed_index, seed in enumerate(_SEEDS, start=1):
            row = _run(values, games=games, workers=workers, seed=seed)
            runs.append(row)
            completed = ((index - 1) * len(_SEEDS)) + seed_index
            print(
                f"{label} {completed:03d}/{total} values={values} seed={seed} "
                f"delta={float(row['rating_delta']):+.2f} "
                f"paired={float(row['paired_score']):.3f}",
                flush=True,
            )
    aggregates = [_aggregate(values, runs) for values in candidates]
    aggregates.sort(
        key=lambda row: (
            float(row["mean_rating_delta"]),
            float(row["paired_score"]),
        ),
        reverse=True,
    )
    return runs, aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1_200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tuning/fixed-objective-overlay-v3-fixed/search.json"),
    )
    args = parser.parse_args()

    initial_candidates = _neighbors(_BASELINE)
    initial_runs, initial_aggregates = _evaluate(
        initial_candidates,
        games=args.games,
        workers=args.workers,
        label="coordinate",
    )
    beam = tuple(tuple(row["values"]) for row in initial_aggregates[: args.beam])
    evaluated = set(initial_candidates)
    combination_candidates = tuple(
        sorted(
            {
                candidate
                for elite in beam
                for candidate in _neighbors(elite)  # type: ignore[arg-type]
                if candidate not in evaluated
            }
        )
    )
    combination_runs, combination_aggregates = _evaluate(
        combination_candidates,
        games=args.games,
        workers=args.workers,
        label="combination",
    )
    all_aggregates = sorted(
        (*initial_aggregates, *combination_aggregates),
        key=lambda row: (
            float(row["mean_rating_delta"]),
            float(row["paired_score"]),
        ),
        reverse=True,
    )
    payload = {
        "constraints": {
            "baseline": _BASELINE,
            "ranges": _RANGES,
            "resource_and_investment_targets_never_increase": True,
            "loan_targets_may_move_both_directions": True,
        },
        "games_per_run": args.games,
        "seeds": _SEEDS,
        "initial_runs": initial_runs,
        "initial_aggregates": initial_aggregates,
        "combination_runs": combination_runs,
        "combination_aggregates": combination_aggregates,
        "aggregates": all_aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("TOP CANDIDATES")
    for row in all_aggregates[:10]:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
