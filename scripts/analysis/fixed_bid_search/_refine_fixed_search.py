from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CANDIDATES = (
    (5, 10, 2, 4, 4, 9),
    (5, 10, 2, 5, 4, 9),
    (5, 10, 2, 4, 4, 7),
    (5, 10, 2, 4, 4, 8),
    (5, 10, 2, 4, 4, 6),
    (4, 8, 3, 6, 5, 11),
    (5, 10, 2, 4, 3, 9),
    (5, 10, 2, 5, 4, 7),
    (5, 10, 2, 5, 3, 7),
    (5, 10, 2, 6, 4, 7),
    (4, 9, 2, 5, 4, 7),
)
SEEDS = (9_929, 424_242)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1_800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fixed-refine-results.json"))
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    total = len(CANDIDATES) * len(SEEDS)
    completed_count = 0
    for candidate in CANDIDATES:
        for seed in SEEDS:
            command = (
                sys.executable,
                "_run_fixed_candidate.py",
                ",".join(map(str, candidate)),
                "--games",
                str(args.games),
                "--seed",
                str(seed),
                "--workers",
                str(args.workers),
            )
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout)
            results.append(result)
            completed_count += 1
            print(
                f"{completed_count:02d}/{total} {candidate} seed={seed} "
                f"delta={result['rating_delta']:+.1f} paired={result['paired_score']:.3f}",
                flush=True,
            )

    aggregates: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        rows = [row for row in results if tuple(row["values"]) == candidate]
        aggregates.append(
            {
                "values": candidate,
                "rating_deltas": [row["rating_delta"] for row in rows],
                "mean_rating_delta": sum(float(row["rating_delta"]) for row in rows) / len(rows),
                "paired_games": sum(int(row["paired_games"]) for row in rows),
                "paired_score": sum(
                    float(row["paired_score"]) * int(row["paired_games"]) for row in rows
                )
                / sum(int(row["paired_games"]) for row in rows),
                "faults": sum(int(row["faults"]) for row in rows),
                "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
            }
        )
    aggregates.sort(key=lambda item: float(item["mean_rating_delta"]), reverse=True)
    payload = {"runs": results, "aggregates": aggregates}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("AGGREGATES")
    for item in aggregates:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
