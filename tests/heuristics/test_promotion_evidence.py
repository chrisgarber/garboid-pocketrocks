from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

PROMOTION_SOURCE_COMMIT = "7a59af3e37d5124536f1f7ba7366a1953b929137"
SEARCH_SOURCE_COMMIT = "5fb33de9734234ce0902bf79b85a75c3a5585c23"
DEVELOPMENT_DIGEST = "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"
HELD_OUT_DIGEST = "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da"
CORPUS_SNAPSHOT_SHA256 = "6122d00ba4995580c3f2c4642be8e1a045f371c5b527f8fe967d3f284549cb0d"
NOTE_PATH = Path("docs/benchmarks/2026-07-30-heuristic-v3-candidate-promotions.md")
PROMOTIONS_ROOT = Path("docs/benchmarks/promotions")
EVOLUTION_ROOT = Path("docs/benchmarks/evolution")
ARTIFACT_NAMES = {
    "promotion-report.json",
    "paired-games.jsonl",
    "corpus-snapshot.json",
}


@dataclass(frozen=True, slots=True)
class PromotionExpectation:
    personality: str
    candidate: str
    incumbent: str
    search_name: str
    generation: int
    slot: int
    development_rating: float
    development_finish: float
    development_money: int
    rating: float
    interval_lower: float
    interval_upper: float
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str
    report_sha256: str
    games_sha256: str

    @property
    def promotion_directory(self) -> Path:
        return PROMOTIONS_ROOT / f"2026-07-30-{self.candidate}-vs-{self.incumbent}"

    @property
    def search_directory(self) -> Path:
        return EVOLUTION_ROOT / self.search_name

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return {
            "promotion-report.json": self.report_sha256,
            "paired-games.jsonl": self.games_sha256,
            "corpus-snapshot.json": CORPUS_SNAPSHOT_SHA256,
        }


EXPECTATIONS = (
    PromotionExpectation(
        personality="aggressive",
        candidate="aggressive-v3-candidate-g007-s008-c70e11540db9",
        incumbent="aggressive-v2",
        search_name="aggressive-v3-search-v1",
        generation=7,
        slot=8,
        development_rating=252.83828743380695,
        development_finish=36.58333333333334,
        development_money=3609,
        rating=231.9549699979907,
        interval_lower=191.07134861693405,
        interval_upper=280.1109053312352,
        freeze_digest="218e9682d8d174125d4b9e7550fec9afda01ddb4433084143968b6d525d335da",
        profile_digest="c70e11540db92d0c77ce5085670ff48105c91aede1ed52c4abb7874a64687b58",
        manifest_digest="627eb77836f8dceace745a8fb7f60573e2dad05aa47a423d902850c32a98f5e0",
        search_report_digest=("01ca66301d633be7228c3bc535fa2d84b0c5ee3898b92f9d06e98c0fdf13b902"),
        candidate_evaluations_digest=(
            "4140270b3fe1d744aef103b012ca85970aa561c3f36cb28d70e4e4aa39f9c7a5"
        ),
        report_sha256="f145de65af5467cb8e75cf36911742541c8b4f4a528a39c34e426150fc22385e",
        games_sha256="da526fdc314792f84adc3f86dc4a9e713fe799125f732b43e514a9e490e87bae",
    ),
    PromotionExpectation(
        personality="balanced",
        candidate="balanced-v3-candidate-g006-s010-e3971899626c",
        incumbent="balanced-v2",
        search_name="balanced-v3-search-v1",
        generation=6,
        slot=10,
        development_rating=176.30500419887812,
        development_finish=23.25,
        development_money=1991,
        rating=143.35885513014068,
        interval_lower=102.76507930109351,
        interval_upper=188.82823172726077,
        freeze_digest="05bcd898e7fc79062585cb989b67cb2e5641eed6cc59b1a60255de84c8ee2988",
        profile_digest="e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e",
        manifest_digest="da9e2162eec9dd934dc80e59d9950b49c74a3a4cd4d72e6273134b502e705152",
        search_report_digest=("95fd24f688ed2bb18cd08d00483fbef2a42b2b66809afa26860f71deba2d3f87"),
        candidate_evaluations_digest=(
            "fcf8985f40beddac274f5aa31523ec93b1cfecf0657047e30144ba97140b15e6"
        ),
        report_sha256="3e5f033b09a96913d565ee3e2bb4c5fd73b00504ce88a2194ea8f71742fbe18c",
        games_sha256="1c032dcca8c2cae59c69a5f35c235626b3f7897213f2a5f5b4ea52f2d842facb",
    ),
    PromotionExpectation(
        personality="passive",
        candidate="passive-v3-candidate-g006-s001-812832214cd5",
        incumbent="passive-v2",
        search_name="passive-v3-search-v1",
        generation=6,
        slot=1,
        development_rating=262.3829631208198,
        development_finish=32.0,
        development_money=3337,
        rating=303.80330913850275,
        interval_lower=256.50646874507333,
        interval_upper=360.10282152090446,
        freeze_digest="0617870c8641e9d25237354b5fe5a1df4f15637af0e46e8c11b1c13e7054adee",
        profile_digest="812832214cd5a16115104c50d33e94cba9929a3cc355b4779d6002b52b25e734",
        manifest_digest="bf533a434a4208e7b018606c53488fcc3a09499b6da2fcb4b1d020346001a9c1",
        search_report_digest=("46a1d7de4fed02520b384d119464ed7c0af239e1e8300fdb7485956ebc7203a2"),
        candidate_evaluations_digest=(
            "582836d5beb184da26f8b5c27c9d96cea728a867017a9081e1bd137ce57f2b25"
        ),
        report_sha256="eab086f3836336f206965c694449826ab10408c64c382a4fdb71fca03a24eec9",
        games_sha256="d9d691bbe52a621e8c66f5ef14d6c028a39aa17e9942d2204073e142e09f672d",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [_load_json_line(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_json_line(line: str) -> dict[str, Any]:
    payload: object = json.loads(line)
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_corpus_metadata(snapshot: dict[str, Any]) -> dict[str, object]:
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


def _bot_names_for_case(case: dict[str, Any], focal_identity: str) -> list[str]:
    return [
        focal_identity if opponent_name is None else opponent_name
        for opponent_name in case["opponent_names_by_seat"]
    ]


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda value: value.personality)
def test_reports_pin_every_successful_held_out_decision(
    expected: PromotionExpectation,
) -> None:
    report = _load_json(expected.promotion_directory / "promotion-report.json")
    snapshot = _load_json(expected.promotion_directory / "corpus-snapshot.json")

    assert report["repository_commit"] == PROMOTION_SOURCE_COMMIT
    assert report["candidate"] == {"bot_id": expected.candidate, "name": expected.candidate}
    assert report["incumbent"] == {"bot_id": expected.incumbent, "name": expected.incumbent}
    assert report["corpora"]["development"] == _report_corpus_metadata(snapshot["development"])
    assert report["corpora"]["held_out"] == _report_corpus_metadata(snapshot["held_out"])
    assert snapshot["development"]["digest"] == DEVELOPMENT_DIGEST
    assert snapshot["held_out"]["digest"] == HELD_OUT_DIGEST
    assert report["execution"] == {
        "batch_size": 64,
        "bot_ids": [
            expected.candidate,
            expected.incumbent,
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
        "workers": 4,
    }
    assert report["coverage"] == {
        "completed_games": 960,
        "completed_pairs": 480,
        "requested_games": 960,
        "requested_pairs": 480,
    }
    assert report["rating_difference"] == expected.rating
    assert report["confidence_interval_95"] == {
        "lower": expected.interval_lower,
        "upper": expected.interval_upper,
    }
    assert report["bootstrap"] == {"converged": 1000, "requested": 1000, "seed": 0}
    assert report["faults"] == {"by_identity": [], "total": 0, "unattributed": 0}
    assert report["warnings"] == []
    assert report["failures"] == []
    assert report["promoted"] is True
    assert report["candidate_provenance"] == {
        "candidate_bot_id": expected.candidate,
        "candidate_evaluations_digest": expected.candidate_evaluations_digest,
        "candidate_name": expected.candidate,
        "development_corpus_digest": DEVELOPMENT_DIGEST,
        "development_corpus_name": "development-v1",
        "freeze_digest": expected.freeze_digest,
        "kind": "frozen_heuristic_candidate",
        "manifest_digest": expected.manifest_digest,
        "predecessor_name": expected.incumbent,
        "profile_digest": expected.profile_digest,
        "repository_commit": SEARCH_SOURCE_COMMIT,
        "search_name": expected.search_name,
        "search_report_digest": expected.search_report_digest,
    }


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda value: value.personality)
def test_every_successful_artifact_generation_has_canonical_hashes(
    expected: PromotionExpectation,
) -> None:
    directory = expected.promotion_directory

    assert {path.name for path in directory.iterdir()} == ARTIFACT_NAMES
    assert {
        name: _sha256(directory / name) for name in sorted(ARTIFACT_NAMES)
    } == expected.artifact_hashes


def test_shared_corpus_snapshot_pins_the_full_evaluation_matrix() -> None:
    snapshots = [
        _load_json(expected.promotion_directory / "corpus-snapshot.json")
        for expected in EXPECTATIONS
    ]
    first = snapshots[0]

    assert all(snapshot == first for snapshot in snapshots[1:])
    assert first["development"]["digest"] == DEVELOPMENT_DIGEST
    assert first["held_out"]["digest"] == HELD_OUT_DIGEST
    assert len(first["development"]["cases"]) == 240

    held_out = first["held_out"]
    assert held_out["recipe"] == {
        "charts": ["A", "B", "C", "D", "E"],
        "name": "held-out-v1",
        "opponent_names": ["random", "aggressive-v1", "balanced-v1", "passive-v1"],
        "player_counts": [3, 4, 5],
        "purpose": "held_out",
        "repetitions_per_seat_cell": 8,
        "root_seed": 90001,
        "schema_version": 1,
    }
    cases = held_out["cases"]
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


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda value: value.personality)
def test_paired_games_are_480_ordered_fault_free_twins(
    expected: PromotionExpectation,
) -> None:
    summaries = _load_json_lines(expected.promotion_directory / "paired-games.jsonl")
    snapshot = _load_json(expected.promotion_directory / "corpus-snapshot.json")
    cases = snapshot["held_out"]["cases"]

    assert len(summaries) == 2 * len(cases) == 960
    for pair_index, case in enumerate(cases):
        candidate_game = summaries[2 * pair_index]
        incumbent_game = summaries[2 * pair_index + 1]
        focal_seat = case["focal_seat"]

        assert candidate_game["game_index"] == 2 * pair_index
        assert incumbent_game["game_index"] == 2 * pair_index + 1
        assert candidate_game["bot_ids"][focal_seat] == expected.candidate
        assert incumbent_game["bot_ids"][focal_seat] == expected.incumbent
        assert candidate_game["bot_names"] == _bot_names_for_case(case, expected.candidate)
        assert incumbent_game["bot_names"] == _bot_names_for_case(case, expected.incumbent)
        assert candidate_game["seed"] == incumbent_game["seed"] == case["engine_seed"]
        assert candidate_game["root_seed"] == incumbent_game["root_seed"] == 90001
        assert (
            candidate_game["ruleset_name"]
            == incumbent_game["ruleset_name"]
            == (f"live-{case['chart']}")
        )
        assert (
            candidate_game["player_count"]
            == incumbent_game["player_count"]
            == (case["player_count"])
        )
        assert (
            candidate_game["bot_ids"][:focal_seat] + candidate_game["bot_ids"][focal_seat + 1 :]
            == incumbent_game["bot_ids"][:focal_seat] + incumbent_game["bot_ids"][focal_seat + 1 :]
        )
        assert candidate_game["fault_counts"] == [0] * case["player_count"]
        assert incumbent_game["fault_counts"] == [0] * case["player_count"]


@pytest.mark.parametrize("expected", EXPECTATIONS, ids=lambda value: value.personality)
def test_development_search_selected_the_promoted_frozen_candidate(
    expected: PromotionExpectation,
) -> None:
    search_report = _load_json(expected.search_directory / "search-report.json")
    frozen = _load_json(expected.search_directory / "frozen-candidate.json")

    assert search_report["repository_commit"] == SEARCH_SOURCE_COMMIT
    assert search_report["search"] == {
        "elite_count": 4,
        "generation_count": 8,
        "manifest_digest": expected.manifest_digest,
        "name": expected.search_name,
        "personality": expected.personality,
        "population_size": 12,
        "predecessor_name": expected.incumbent,
    }
    assert search_report["best_result"] == {
        "candidate_identity": expected.candidate,
        "final_money_delta": expected.development_money,
        "generation": expected.generation,
        "normalized_finish_delta": expected.development_finish,
        "rating_delta": expected.development_rating,
        "slot": expected.slot,
    }
    assert search_report["coverage"] == {
        "completed_baseline_games": 240,
        "completed_candidate_games": 23040,
        "completed_generations": 8,
        "evaluated_candidates": 96,
        "proposed_candidates": 96,
        "requested_baseline_games": 240,
        "requested_candidate_games": 23040,
    }
    assert search_report["selected_candidate_identity"] == expected.candidate
    assert search_report["frozen_candidate_identity"] == expected.candidate
    assert search_report["status"] == "frozen_improvement"
    assert search_report["failures"] == []
    assert frozen["identity"] == expected.candidate
    assert frozen["predecessor_name"] == expected.incumbent
    assert frozen["development_scores"] == {
        "final_money_delta": expected.development_money,
        "normalized_finish_delta": expected.development_finish,
        "rating_delta": expected.development_rating,
    }


def test_preserved_sandbox_failure_contains_no_held_out_outcome() -> None:
    directory = PROMOTIONS_ROOT / (
        "2026-07-30-aggressive-v3-candidate-g007-s008-c70e11540db9"
        "-vs-aggressive-v2-operational-failure-sandbox"
    )
    report = _load_json(directory / "promotion-report.json")

    assert {name: _sha256(directory / name) for name in sorted(ARTIFACT_NAMES)} == {
        "corpus-snapshot.json": CORPUS_SNAPSHOT_SHA256,
        "paired-games.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "promotion-report.json": (
            "cdb53525a85d73ba1c2eae15f20e227e85579b06947870c9e2e9e1413c19278e"
        ),
    }
    assert report["repository_commit"] == PROMOTION_SOURCE_COMMIT
    assert report["coverage"] == {
        "completed_games": 0,
        "completed_pairs": 0,
        "requested_games": 960,
        "requested_pairs": 480,
    }
    assert report["rating_difference"] is None
    assert report["confidence_interval_95"] is None
    assert report["bootstrap"] == {"converged": 0, "requested": 1000, "seed": 0}
    assert report["promoted"] is False
    assert report["failures"] == [
        {
            "code": "simulation_failed",
            "message": (
                "The promotion games could not be completed: The simulator stopped unexpectedly "
                "with PermissionError: [Errno 1] Operation not permitted"
            ),
        }
    ]
    assert (directory / "paired-games.jsonl").read_bytes() == b""


def test_benchmark_note_records_reproduction_results_and_all_or_nothing_decision() -> None:
    note = NOTE_PATH.read_text(encoding="utf-8")
    normalized = " ".join(note.split()).lower()

    assert PROMOTION_SOURCE_COMMIT in note
    assert SEARCH_SOURCE_COMMIT in note
    assert DEVELOPMENT_DIGEST in note
    assert HELD_OUT_DIGEST in note
    assert "all-or-nothing" in normalized
    assert "zero completed games" in normalized
    assert "no held-out outcome" in normalized
    assert "permission-corrected rerun" in normalized
    assert "balanced and passive commands each ran once" in normalized
    assert "aggressive successful invocation ran once" in normalized
    assert "each command ran once" not in normalized
    for expected in EXPECTATIONS:
        assert expected.candidate in note
        assert expected.incumbent in note
        assert expected.search_name in note
        assert str(expected.development_rating) in note
        assert str(expected.rating) in note
        assert str(expected.interval_lower) in note
        assert str(expected.interval_upper) in note
        assert expected.report_sha256 in note
        assert expected.games_sha256 in note
        assert f"--candidate {expected.candidate}" in note
        assert f"--incumbent {expected.incumbent}" in note
        assert f"--output-dir {expected.promotion_directory}" in normalized


def test_every_local_link_in_the_benchmark_note_resolves() -> None:
    note = NOTE_PATH.read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+]\(([^)]+)\)", note)

    assert links
    for link in links:
        assert not link.startswith(("http://", "https://"))
        assert (NOTE_PATH.parent / link).resolve().exists(), link
