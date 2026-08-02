from __future__ import annotations

import argparse
import json
from pathlib import Path

from _drive_search import _RANGES, Values, _aggregate, _run

_SEEDS = (20_260_804_11, 20_260_804_12)
_BASELINE: Values = (5, 10, 2, 5, 4, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/tuning/fixed-objective-overlay-v3-fixed/search.json"),
    )
    parser.add_argument("--games", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tuning/fixed-objective-overlay-v3-fixed/refinement.json"),
    )
    args = parser.parse_args()

    search = json.loads(args.input.read_text())
    winner = tuple(search["aggregates"][0]["values"])
    fixed_reduction_neighbors: list[Values] = []
    for index in (0, 1, 4, 5):
        for choice in _RANGES[index]:
            candidate = list(winner)
            candidate[index] = choice
            fixed_reduction_neighbors.append(tuple(candidate))  # type: ignore[arg-type]
    candidates = tuple(
        dict.fromkeys(
            (
                *(tuple(row["values"]) for row in search["aggregates"][: args.finalists]),
                *fixed_reduction_neighbors,
                _BASELINE,
            )
        )
    )
    runs: list[dict[str, object]] = []
    total = len(candidates) * len(_SEEDS)
    for candidate_index, values in enumerate(candidates, start=1):
        for seed_index, seed in enumerate(_SEEDS, start=1):
            row = _run(values, games=args.games, workers=args.workers, seed=seed)
            runs.append(row)
            completed = ((candidate_index - 1) * len(_SEEDS)) + seed_index
            print(
                f"refine {completed:02d}/{total} values={values} seed={seed} "
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
    payload = {
        "games_per_run": args.games,
        "seeds": _SEEDS,
        "runs": runs,
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("REFINEMENT RESULTS")
    for row in aggregates:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
