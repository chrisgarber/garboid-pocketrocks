from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

PROMOTIONS_ROOT = Path("docs/benchmarks/promotions")
V4_PROMOTION_SOURCE_COMMIT = "109d0602ab035df82b382b92f4a63a133617b5c1"
V4_SEARCH_SOURCE_COMMIT = "a66c49e559849b35a290827b51b2e5098524e2d1"
DEVELOPMENT_DIGEST = "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"
HELD_OUT_DIGEST = "de686b97e9318d840554514d71158e7d30e4b1603c6692d68b73bc77947b10da"
CORPUS_SNAPSHOT_SHA256 = "6122d00ba4995580c3f2c4642be8e1a045f371c5b527f8fe967d3f284549cb0d"


@dataclass(frozen=True, slots=True)
class V4PromotionExpectation:
    personality: str
    candidate: str
    incumbent: str
    rating_difference: float
    interval_lower: float
    interval_upper: float
    report_sha256: str
    games_sha256: str

    @property
    def directory(self) -> Path:
        return PROMOTIONS_ROOT / f"2026-07-30-{self.candidate}-vs-{self.incumbent}"


V4_EXPECTATIONS = (
    V4PromotionExpectation(
        "aggressive",
        "aggressive-v4-candidate-g011-s004-000d194163fa",
        "aggressive-v3",
        -3.3070227531331966,
        -38.19114288857132,
        30.14083215446108,
        "c3f6faa6f8d70b387d3962e66fc96bdaa6385edb9a3cf40ed40e72cf04689878",
        "c426336cff83e6c7d14e99bf668b4e0560ba17a1ffb42d258ca64e24042be07d",
    ),
    V4PromotionExpectation(
        "balanced",
        "balanced-v4-candidate-g009-s000-4d391ce068d7",
        "balanced-v3",
        -7.903601047803477,
        -17.151275621918607,
        0.11911328931605498,
        "e61dad7b0116d5a26286169159b0a0fc0d81bb6d055dcf8f3c4c93e2f71eb30b",
        "71332fde536a5ac6cac5d8d77ae634712cc1e71f3becc9a9937b8e6b69c65ea3",
    ),
    V4PromotionExpectation(
        "passive",
        "passive-v4-candidate-g005-s005-cf4f7b924ee3",
        "passive-v3",
        -0.09074482590222033,
        -27.15037852540195,
        29.972291195257924,
        "212d275550ea659bb9be7a70d3fd01f53c51f85ee40dfa16c947d3623bc2e1b8",
        "f243a0bf900d7dbf8186b12d563fbfe62b57c7dadceb2d9b426b3c6270452a3e",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload: object = json.loads(line)
        assert isinstance(payload, dict)
        rows.append(cast(dict[str, Any], payload))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("expected", V4_EXPECTATIONS, ids=lambda value: value.personality)
def test_v4_reports_pin_truthful_failed_held_out_decisions(
    expected: V4PromotionExpectation,
) -> None:
    report_path = expected.directory / "promotion-report.json"
    games_path = expected.directory / "paired-games.jsonl"
    snapshot_path = expected.directory / "corpus-snapshot.json"
    report = _load_json(report_path)

    assert report["schema_version"] == 2
    assert report["repository_commit"] == V4_PROMOTION_SOURCE_COMMIT
    assert report["candidate"] == {"bot_id": expected.candidate, "name": expected.candidate}
    assert report["incumbent"] == {"bot_id": expected.incumbent, "name": expected.incumbent}
    assert report["rating_difference"] == expected.rating_difference
    assert report["confidence_interval_95"] == {
        "lower": expected.interval_lower,
        "upper": expected.interval_upper,
    }
    assert report["coverage"] == {
        "completed_games": 960,
        "completed_pairs": 480,
        "requested_games": 960,
        "requested_pairs": 480,
    }
    assert report["faults"] == {"by_identity": [], "total": 0, "unattributed": 0}
    assert report["warnings"] == []
    assert report["promoted"] is False
    assert [failure["code"] for failure in report["failures"]] == ["interval_includes_zero"]

    provenance = report["candidate_provenance"]
    assert provenance["kind"] == "frozen_phase_aware_heuristic_candidate"
    assert provenance["freeze_schema_version"] == 2
    assert provenance["candidate_name"] == expected.candidate
    assert provenance["predecessor_name"] == expected.incumbent
    assert provenance["development_corpus_digest"] == DEVELOPMENT_DIGEST
    assert provenance["repository_commit"] == V4_SEARCH_SOURCE_COMMIT
    assert set(provenance["experts"]) == {"early", "middle", "late"}

    assert _sha256(report_path) == expected.report_sha256
    assert _sha256(games_path) == expected.games_sha256
    assert _sha256(snapshot_path) == CORPUS_SNAPSHOT_SHA256


def test_v4_promotions_reuse_the_complete_held_out_matrix() -> None:
    snapshots = [
        _load_json(expected.directory / "corpus-snapshot.json") for expected in V4_EXPECTATIONS
    ]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])

    snapshot = snapshots[0]
    assert snapshot["development"]["digest"] == DEVELOPMENT_DIGEST
    assert snapshot["held_out"]["digest"] == HELD_OUT_DIGEST
    cases = snapshot["held_out"]["cases"]
    assert len(cases) == 480
    coverage = Counter((case["chart"], case["player_count"], case["focal_seat"]) for case in cases)
    assert set(coverage.values()) == {8}


@pytest.mark.parametrize("expected", V4_EXPECTATIONS, ids=lambda value: value.personality)
def test_v4_paired_games_are_ordered_fault_free_twins(
    expected: V4PromotionExpectation,
) -> None:
    summaries = _load_json_lines(expected.directory / "paired-games.jsonl")
    cases = _load_json(expected.directory / "corpus-snapshot.json")["held_out"]["cases"]

    assert len(summaries) == 2 * len(cases) == 960
    for pair_index, case in enumerate(cases):
        candidate_game = summaries[2 * pair_index]
        incumbent_game = summaries[2 * pair_index + 1]
        focal_seat = case["focal_seat"]
        assert candidate_game["game_index"] == 2 * pair_index
        assert incumbent_game["game_index"] == 2 * pair_index + 1
        assert candidate_game["bot_ids"][focal_seat] == expected.candidate
        assert incumbent_game["bot_ids"][focal_seat] == expected.incumbent
        assert candidate_game["seed"] == incumbent_game["seed"] == case["engine_seed"]
        assert candidate_game["fault_counts"] == [0] * case["player_count"]
        assert incumbent_game["fault_counts"] == [0] * case["player_count"]
