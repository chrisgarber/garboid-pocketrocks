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
CORPUS_SNAPSHOT_PATH = EVIDENCE_DIRECTORY / "corpus-snapshot.json"
EVIDENCE_NOTE_PATH = Path("docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-promotion.md")
NEURAL_README_PATH = Path("src/garboid_pocketrocks/neural/README.md")

CANDIDATE_IDENTITY = "vector_ppo_large_v1_g350k"
INCUMBENT_IDENTITY = "vector_ppo_small_v1_g1500"


class CorpusRecipe(TypedDict):
    schema_version: int
    name: str
    purpose: str
    root_seed: int
    repetitions_per_seat_cell: int
    charts: list[str]
    player_counts: list[int]
    opponent_names: list[str]


class PromotionCase(TypedDict):
    case_id: str
    chart: str
    player_count: int
    focal_seat: int
    engine_seed: int
    opponent_names_by_seat: list[str | None]


class CorpusSnapshotEntry(TypedDict):
    digest: str
    recipe: CorpusRecipe
    cases: list[PromotionCase]


class CorpusSnapshot(TypedDict):
    development: CorpusSnapshotEntry
    held_out: CorpusSnapshotEntry


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


def _load_corpus_snapshot() -> CorpusSnapshot:
    return cast(
        CorpusSnapshot,
        _parse_json_object(CORPUS_SNAPSHOT_PATH.read_text(encoding="utf-8")),
    )


def _load_game_summaries() -> list[GameSummary]:
    return [
        cast(GameSummary, _parse_json_object(line))
        for line in PAIRED_GAMES_PATH.read_text(encoding="utf-8").splitlines()
    ]


def _without_focal_seat(values: list[str], focal_seat: int) -> list[str]:
    return values[:focal_seat] + values[focal_seat + 1 :]


def _report_corpus_metadata(snapshot: CorpusSnapshotEntry) -> dict[str, object]:
    recipe = snapshot["recipe"]
    return {
        "charts": recipe["charts"],
        "digest": snapshot["digest"],
        "engine_seeds": [case["engine_seed"] for case in snapshot["cases"]],
        "name": recipe["name"],
        "opponent_names": recipe["opponent_names"],
        "player_counts": recipe["player_counts"],
        "purpose": recipe["purpose"],
        "repetitions_per_seat_cell": recipe["repetitions_per_seat_cell"],
        "root_seed": recipe["root_seed"],
    }


def _bot_names_for_case(case: PromotionCase, focal_identity: str) -> list[str]:
    return [
        focal_identity if opponent_name is None else opponent_name
        for opponent_name in case["opponent_names_by_seat"]
    ]


def test_promotion_report_pins_the_held_out_neural_decision() -> None:
    report = _load_json(REPORT_PATH)
    snapshot = _load_corpus_snapshot()

    assert report["repository_commit"] == "5852176ff3c28b3f469a85a349be40ce41c05aa8"
    assert report["candidate"] == {
        "bot_id": CANDIDATE_IDENTITY,
        "name": CANDIDATE_IDENTITY,
    }
    assert report["incumbent"] == {
        "bot_id": INCUMBENT_IDENTITY,
        "name": INCUMBENT_IDENTITY,
    }
    assert snapshot["development"]["digest"] == (
        "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"
    )
    assert snapshot["held_out"]["digest"] == (
        "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da"
    )
    assert report["corpora"]["development"] == _report_corpus_metadata(snapshot["development"])
    assert report["corpora"]["held_out"] == _report_corpus_metadata(snapshot["held_out"])
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


def test_held_out_snapshot_covers_every_requested_seat_with_distinct_seeds() -> None:
    held_out = _load_corpus_snapshot()["held_out"]
    recipe = held_out["recipe"]
    cases = held_out["cases"]

    assert recipe == {
        "schema_version": 1,
        "name": "held-out-v1",
        "purpose": "held_out",
        "root_seed": 90001,
        "repetitions_per_seat_cell": 8,
        "charts": ["A", "B", "C", "D", "E"],
        "player_counts": [3, 4, 5],
        "opponent_names": ["random", "aggressive-v1", "balanced-v1", "passive-v1"],
    }
    assert len(cases) == 480
    assert len({case["engine_seed"] for case in cases}) == 480

    coverage = Counter((case["chart"], case["player_count"], case["focal_seat"]) for case in cases)
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
    held_out = _load_corpus_snapshot()["held_out"]
    cases = held_out["cases"]

    assert len(summaries) == 2 * len(cases) == 960
    for pair_index, case in enumerate(cases):
        candidate_game = summaries[2 * pair_index]
        incumbent_game = summaries[2 * pair_index + 1]

        assert candidate_game["game_index"] == 2 * pair_index
        assert incumbent_game["game_index"] == 2 * pair_index + 1

        focal_seat = case["focal_seat"]
        assert candidate_game["bot_ids"][focal_seat] == CANDIDATE_IDENTITY
        assert incumbent_game["bot_ids"][focal_seat] == INCUMBENT_IDENTITY
        assert candidate_game["bot_names"] == _bot_names_for_case(case, CANDIDATE_IDENTITY)
        assert incumbent_game["bot_names"] == _bot_names_for_case(case, INCUMBENT_IDENTITY)

        assert candidate_game["seed"] == incumbent_game["seed"] == case["engine_seed"]
        assert candidate_game["root_seed"] == incumbent_game["root_seed"] == 90001
        assert (
            candidate_game["ruleset_name"]
            == incumbent_game["ruleset_name"]
            == (f"live-{case['chart']}")
        )
        assert (
            candidate_game["player_count"] == incumbent_game["player_count"] == case["player_count"]
        )
        assert _without_focal_seat(
            candidate_game["bot_ids"],
            focal_seat,
        ) == _without_focal_seat(
            incumbent_game["bot_ids"],
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
