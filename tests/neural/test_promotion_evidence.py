from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict, cast

EVIDENCE_DIRECTORY = Path(
    "docs/benchmarks/promotions/2026-07-30-vector-ppo-large-v1-g350k-vs-small-v1-g1500"
)
REPORT_PATH = EVIDENCE_DIRECTORY / "promotion-report.json"
PAIRED_GAMES_PATH = EVIDENCE_DIRECTORY / "paired-games.jsonl"
EVIDENCE_NOTE_PATH = Path("docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-promotion.md")
NEURAL_README_PATH = Path("src/garboid_pocketrocks/neural/README.md")

CANDIDATE_IDENTITY = "vector_ppo_large_v1_g350k"
INCUMBENT_IDENTITY = "vector_ppo_small_v1_g1500"


class GameSummary(TypedDict):
    bot_ids: list[str]
    bot_names: list[str]
    fault_counts: list[int]
    game_index: int
    player_count: int
    root_seed: int
    ruleset_name: str
    seed: int


def _parse_json_object(contents: str) -> dict[str, Any]:
    parsed: object = json.loads(contents)
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def _load_json(path: Path) -> dict[str, Any]:
    return _parse_json_object(path.read_text(encoding="utf-8"))


def _load_game_summaries() -> list[GameSummary]:
    return [
        cast(GameSummary, _parse_json_object(line))
        for line in PAIRED_GAMES_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _without_focal_seat(values: list[str], focal_seat: int) -> list[str]:
    return values[:focal_seat] + values[focal_seat + 1 :]


def test_promotion_report_pins_the_held_out_neural_decision() -> None:
    report = _load_json(REPORT_PATH)

    assert report["artifacts"] == ["promotion-report.json", "paired-games.jsonl"]
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


def test_report_and_games_cover_every_requested_seat_with_distinct_seeds() -> None:
    report = _load_json(REPORT_PATH)
    held_out = report["corpora"]["held_out"]
    summaries = _load_game_summaries()
    candidate_games = summaries[::2]

    assert {
        field: held_out[field]
        for field in (
            "name",
            "purpose",
            "root_seed",
            "repetitions_per_seat_cell",
            "charts",
            "player_counts",
            "opponent_names",
        )
    } == {
        "name": "held-out-v1",
        "purpose": "held_out",
        "root_seed": 90001,
        "repetitions_per_seat_cell": 8,
        "charts": ["A", "B", "C", "D", "E"],
        "player_counts": [3, 4, 5],
        "opponent_names": ["random", "aggressive-v1", "balanced-v1", "passive-v1"],
    }
    assert len(candidate_games) == 480
    assert len(set(held_out["engine_seeds"])) == 480
    assert {game["seed"] for game in candidate_games} == set(held_out["engine_seeds"])

    coverage = Counter(
        (
            game["ruleset_name"].removeprefix("live-"),
            game["player_count"],
            game["bot_ids"].index(CANDIDATE_IDENTITY),
        )
        for game in candidate_games
    )
    requested_seats = {
        (chart, player_count, focal_seat)
        for chart in "ABCDE"
        for player_count in (3, 4, 5)
        for focal_seat in range(player_count)
    }
    assert set(coverage) == requested_seats
    assert set(coverage.values()) == {8}


def test_paired_game_evidence_contains_480_ordered_fault_free_twins() -> None:
    summaries = _load_game_summaries()

    assert len(summaries) == 960
    for pair_index in range(480):
        candidate_game = summaries[2 * pair_index]
        incumbent_game = summaries[2 * pair_index + 1]

        assert candidate_game["game_index"] == 2 * pair_index
        assert incumbent_game["game_index"] == 2 * pair_index + 1

        focal_seat = candidate_game["bot_ids"].index(CANDIDATE_IDENTITY)
        assert candidate_game["bot_ids"][focal_seat] == CANDIDATE_IDENTITY
        assert incumbent_game["bot_ids"][focal_seat] == INCUMBENT_IDENTITY
        assert candidate_game["bot_names"][focal_seat] == CANDIDATE_IDENTITY
        assert incumbent_game["bot_names"][focal_seat] == INCUMBENT_IDENTITY

        assert candidate_game["seed"] == incumbent_game["seed"]
        assert candidate_game["root_seed"] == incumbent_game["root_seed"] == 90001
        assert candidate_game["ruleset_name"] == incumbent_game["ruleset_name"]
        assert candidate_game["player_count"] == incumbent_game["player_count"]
        for field in ("bot_ids", "bot_names"):
            assert _without_focal_seat(
                candidate_game[field],
                focal_seat,
            ) == _without_focal_seat(
                incumbent_game[field],
                focal_seat,
            )

        assert candidate_game["fault_counts"] == [0] * candidate_game["player_count"]
        assert incumbent_game["fault_counts"] == [0] * incumbent_game["player_count"]


def test_neural_readme_links_the_dated_promotion_evidence() -> None:
    readme = NEURAL_README_PATH.read_text(encoding="utf-8")

    assert EVIDENCE_NOTE_PATH.name in readme
    assert EVIDENCE_DIRECTORY.name in readme


def test_benchmark_note_records_complete_reproduction_constraints() -> None:
    note = EVIDENCE_NOTE_PATH.read_text(encoding="utf-8")
    normalized_note = " ".join(note.split())

    assert "--bootstrap-seed 0" in note
    assert "--batch-size 64" in note
    assert "5852176ff3c28b3f469a85a349be40ce41c05aa8" in note
    assert "empty output directory" in normalized_note
