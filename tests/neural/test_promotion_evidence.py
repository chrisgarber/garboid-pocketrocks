from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

EVIDENCE_DIRECTORY = Path(
    "docs/benchmarks/promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500"
)
REPORT_PATH = EVIDENCE_DIRECTORY / "promotion-report.json"
PAIRED_GAMES_PATH = EVIDENCE_DIRECTORY / "paired-games.jsonl"
EVIDENCE_NOTE_PATH = Path("docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-promotion.md")
NEURAL_README_PATH = Path("src/garboid_pocketrocks/neural/README.md")

CANDIDATE_IDENTITY = "vector_ppo_large_v1_g350k"
INCUMBENT_IDENTITY = "vector_ppo_small_v1_g1500"


def _parse_json_object(contents: str) -> dict[str, Any]:
    parsed: object = json.loads(contents)
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def _load_json(path: Path) -> dict[str, Any]:
    return _parse_json_object(path.read_text(encoding="utf-8"))


def _load_game_summaries() -> list[dict[str, Any]]:
    return [
        _parse_json_object(line)
        for line in PAIRED_GAMES_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _without_focal_seat(values: list[Any], focal_seat: int) -> list[Any]:
    return values[:focal_seat] + values[focal_seat + 1 :]


def test_promotion_report_pins_the_held_out_neural_decision() -> None:
    report = _load_json(REPORT_PATH)

    assert report["repository_commit"] == "5852176ff3c28b3f469a85a349be40ce41c05aa8"
    assert report["candidate"] == {
        "bot_id": CANDIDATE_IDENTITY,
        "name": CANDIDATE_IDENTITY,
    }
    assert report["incumbent"] == {
        "bot_id": INCUMBENT_IDENTITY,
        "name": INCUMBENT_IDENTITY,
    }
    assert report["corpora"]["development"]["digest"] == (
        "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"
    )
    assert report["corpora"]["held_out"]["digest"] == (
        "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da"
    )
    assert report["execution"] == {
        "batch_size": 64,
        "bot_ids": [
            CANDIDATE_IDENTITY,
            INCUMBENT_IDENTITY,
            "passive-v1",
            "bot_e0e2c541-1615-4f47-983c-224e7d888d89",
            "aggressive-v1",
            "balanced-v1",
        ],
        "capture_replays": False,
        "fault_mode": "record_and_pass",
        "games": 960,
        "objectives_enabled": [True],
        "player_counts": [3, 4, 5],
        "root_seed": 90001,
        "value_charts": ["A", "B", "C", "D", "E"],
        "workers": 8,
    }
    assert report["coverage"] == {
        "completed_games": 960,
        "completed_pairs": 480,
        "requested_games": 960,
        "requested_pairs": 480,
    }
    assert report["rating_difference"] == 380.4651404425133
    assert report["confidence_interval_95"] == {
        "lower": 333.59571588851003,
        "upper": 429.92008317501353,
    }
    assert report["bootstrap"] == {
        "converged": 1000,
        "requested": 1000,
        "seed": 0,
    }
    assert report["faults"] == {
        "by_identity": [],
        "total": 0,
        "unattributed": 0,
    }
    assert report["failures"] == []
    assert report["warnings"] == []
    assert report["promoted"] is True


def test_paired_game_evidence_contains_480_ordered_fault_free_twins() -> None:
    summaries = _load_game_summaries()

    assert len(summaries) == 960
    for pair_index in range(480):
        candidate_game = summaries[2 * pair_index]
        incumbent_game = summaries[2 * pair_index + 1]

        assert candidate_game["game_index"] == 2 * pair_index
        assert incumbent_game["game_index"] == 2 * pair_index + 1

        candidate_focal_seat = candidate_game["bot_ids"].index(CANDIDATE_IDENTITY)
        incumbent_focal_seat = incumbent_game["bot_ids"].index(INCUMBENT_IDENTITY)
        assert candidate_focal_seat == incumbent_focal_seat
        assert candidate_game["bot_names"][candidate_focal_seat] == CANDIDATE_IDENTITY
        assert incumbent_game["bot_names"][incumbent_focal_seat] == INCUMBENT_IDENTITY

        for field in ("seed", "root_seed", "ruleset_name", "player_count"):
            assert candidate_game[field] == incumbent_game[field]
        for field in ("bot_ids", "bot_names"):
            assert _without_focal_seat(
                candidate_game[field],
                candidate_focal_seat,
            ) == _without_focal_seat(
                incumbent_game[field],
                incumbent_focal_seat,
            )

        assert candidate_game["fault_counts"] == [0] * candidate_game["player_count"]
        assert incumbent_game["fault_counts"] == [0] * incumbent_game["player_count"]


def test_neural_readme_links_the_dated_promotion_evidence() -> None:
    readme = NEURAL_README_PATH.read_text(encoding="utf-8")

    assert EVIDENCE_NOTE_PATH.name in readme
    assert EVIDENCE_DIRECTORY.name in readme
