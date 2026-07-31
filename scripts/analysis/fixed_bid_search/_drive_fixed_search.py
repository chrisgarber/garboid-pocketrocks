from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

BASELINE = (5, 10, 2, 4, 4, 9)
BOUNDS = ((3, 8), (7, 14), (1, 4), (2, 7), (2, 7), (6, 13))


def _screen_candidates() -> tuple[tuple[int, ...], ...]:
    candidates = {BASELINE}
    coordinate_values = (
        (3, 4, 6, 7, 8),
        (7, 8, 9, 11, 12, 13, 14),
        (1, 3, 4),
        (2, 3, 5, 6, 7),
        (2, 3, 5, 6, 7),
        (6, 7, 8, 10, 11, 12, 13),
    )
    for index, values in enumerate(coordinate_values):
        for value in values:
            candidate = list(BASELINE)
            candidate[index] = value
            candidates.add(tuple(candidate))

    coupled_offsets = (
        (-2, -3, -1, -2, -2, -3),
        (-1, -2, -1, -1, -1, -2),
        (1, 2, 1, 1, 1, 2),
        (2, 3, 2, 2, 2, 3),
        (-1, -2, 1, 2, 1, 2),
        (1, 2, -1, -2, -1, -2),
        (-1, 1, 0, 1, -1, 1),
        (1, -1, 1, -1, 1, -1),
    )
    for offsets in coupled_offsets:
        candidates.add(
            tuple(
                min(high, max(low, base + offset))
                for base, offset, (low, high) in zip(BASELINE, offsets, BOUNDS, strict=True)
            )
        )

    rng = random.Random(20_260_731)
    while len(candidates) < 64:
        candidate = tuple(
            min(high, max(low, base + rng.choice(offsets)))
            for base, offsets, (low, high) in zip(
                BASELINE,
                (
                    (-2, -1, 0, 1, 2, 3),
                    (-3, -2, -1, 0, 1, 2, 3, 4),
                    (-1, 0, 1, 2),
                    (-2, -1, 0, 1, 2, 3),
                    (-2, -1, 0, 1, 2, 3),
                    (-3, -2, -1, 0, 1, 2, 3, 4),
                ),
                BOUNDS,
                strict=True,
            )
        )
        if sum(left != right for left, right in zip(candidate, BASELINE, strict=True)) >= 2:
            candidates.add(candidate)
    return tuple(sorted(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=24_701)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fixed-screen-results.json"))
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    candidates = _screen_candidates()
    for index, candidate in enumerate(candidates, start=1):
        command = (
            sys.executable,
            "_run_fixed_candidate.py",
            ",".join(map(str, candidate)),
            "--games",
            str(args.games),
            "--seed",
            str(args.seed),
            "--workers",
            str(args.workers),
        )
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        results.append(result)
        print(
            f"{index:02d}/{len(candidates)} {candidate} "
            f"delta={result['rating_delta']:+.1f} paired={result['paired_score']:.3f}",
            flush=True,
        )
    ordered = sorted(results, key=lambda item: float(item["rating_delta"]), reverse=True)
    args.output.write_text(json.dumps(ordered, indent=2) + "\n")
    print("TOP")
    for item in ordered[:15]:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
