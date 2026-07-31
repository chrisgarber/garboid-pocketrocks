from __future__ import annotations

import csv
import hashlib
import json
import pickle
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from garboid_pocketrocks.bots import (
    BOT_SPECS,
    BOT_SPECS_BY_NAME,
    DEFAULT_TOURNAMENT_BOT_SPECS,
    HeuristicBotBrain,
)
from garboid_pocketrocks.bots.heuristic import PhaseAwareHeuristicBotBrain
from garboid_pocketrocks.evolution.planning import plan_development_games
from garboid_pocketrocks.heuristics.frozen import (
    FROZEN_CANDIDATES,
    FROZEN_CANDIDATES_BY_NAME,
    FrozenCandidate,
    FrozenCandidateCatalogError,
    FrozenPhaseAwareCandidate,
    load_frozen_candidates,
)
from garboid_pocketrocks.heuristics.phases import PHASE_SELECTOR_NAME
from garboid_pocketrocks.promotion.corpus import load_promotion_corpus
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner

REPOSITORY_ROOT = Path(__file__).parents[2]
CATALOG_DIR = REPOSITORY_ROOT / "src" / "garboid_pocketrocks" / "heuristics" / "frozen_candidates"
BENCHMARK_DIR = REPOSITORY_ROOT / "docs" / "benchmarks" / "evolution"
EXPECTED_V3_IDENTITIES = (
    "aggressive-v3-candidate-g007-s008-c70e11540db9",
    "balanced-v3-candidate-g006-s010-e3971899626c",
    "passive-v3-candidate-g006-s001-812832214cd5",
)
EXPECTED_V4_IDENTITIES = (
    "aggressive-v4-candidate-g011-s004-000d194163fa",
    "balanced-v4-candidate-g009-s000-4d391ce068d7",
    "passive-v4-candidate-g005-s005-cf4f7b924ee3",
)
EXPECTED_IDENTITIES = tuple(sorted((*EXPECTED_V3_IDENTITIES, *EXPECTED_V4_IDENTITIES)))
EXPECTED_V3_PROVENANCE: dict[str, dict[str, Any]] = {
    EXPECTED_V3_IDENTITIES[0]: {
        "predecessor": "aggressive-v2",
        "search_name": "aggressive-v3-search-v1",
        "profile": "c70e11540db92d0c77ce5085670ff48105c91aede1ed52c4abb7874a64687b58",
        "manifest": "627eb77836f8dceace745a8fb7f60573e2dad05aa47a423d902850c32a98f5e0",
        "report": "01ca66301d633be7228c3bc535fa2d84b0c5ee3898b92f9d06e98c0fdf13b902",
        "evaluations": "4140270b3fe1d744aef103b012ca85970aa561c3f36cb28d70e4e4aa39f9c7a5",
        "freeze": "218e9682d8d174125d4b9e7550fec9afda01ddb4433084143968b6d525d335da",
    },
    EXPECTED_V3_IDENTITIES[1]: {
        "predecessor": "balanced-v2",
        "search_name": "balanced-v3-search-v1",
        "profile": "e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e",
        "manifest": "da9e2162eec9dd934dc80e59d9950b49c74a3a4cd4d72e6273134b502e705152",
        "report": "95fd24f688ed2bb18cd08d00483fbef2a42b2b66809afa26860f71deba2d3f87",
        "evaluations": "fcf8985f40beddac274f5aa31523ec93b1cfecf0657047e30144ba97140b15e6",
        "freeze": "05bcd898e7fc79062585cb989b67cb2e5641eed6cc59b1a60255de84c8ee2988",
    },
    EXPECTED_V3_IDENTITIES[2]: {
        "predecessor": "passive-v2",
        "search_name": "passive-v3-search-v1",
        "profile": "812832214cd5a16115104c50d33e94cba9929a3cc355b4779d6002b52b25e734",
        "manifest": "bf533a434a4208e7b018606c53488fcc3a09499b6da2fcb4b1d020346001a9c1",
        "report": "46a1d7de4fed02520b384d119464ed7c0af239e1e8300fdb7485956ebc7203a2",
        "evaluations": "582836d5beb184da26f8b5c27c9d96cea728a867017a9081e1bd137ce57f2b25",
        "freeze": "0617870c8641e9d25237354b5fe5a1df4f15637af0e46e8c11b1c13e7054adee",
    },
}
DEVELOPMENT_DIGEST = "17c016350dbe717641b8cd499b0908e3bc0faa811a3b4f5e574f8713a5bf2b3d"
EXPECTED_V4_PROVENANCE: dict[str, dict[str, Any]] = {
    EXPECTED_V4_IDENTITIES[0]: {
        "predecessor": "aggressive-v3",
        "search_name": "aggressive-v4-search-v2",
        "profile": "000d194163fac76a1e2928631379d7aab9b308025d6edb7583110cd962736b04",
        "manifest": "71c06a1a246e81c935156ff818f66dcac454719168fabdd4af63ba94249ca69b",
        "report": "05a77d4bdc177b0aa8b84c43e9ef364ab51533adebd9f2dc67ac3b1da85473bb",
        "evaluations": "5a1ad67b7db378aaa009e3b59bd891ae71519e682f1eab241e3bf3f41f0c35a2",
        "selection": "a45a3cbbf9bf57ffecadc6d84437c19f3e2d98379dba7e41dc04063c036e2ed8",
        "games": "0a633649d4cf4c8ac0b593249f62bae8d64f1a687636a1b5f0722c630192d858",
        "freeze": "2b63de8569efa4922cbb751ac052a20e46ca2accfdb3b0091d2f67d9858690dd",
        "rating_delta": 24.713329019553612,
        "normalized_finish_delta": 2.75,
        "final_money_delta": 259,
        "experts": {
            "early": {
                "bid_shading": "0.6",
                "future_cash_weight": "1.95",
                "liquidity_strength": "0.9",
                "objective_progress_weight": "0.15",
            },
            "middle": {
                "bid_shading": "0.4",
                "future_cash_weight": "1.9",
                "liquidity_strength": "1",
                "objective_progress_weight": "0.1",
            },
            "late": {
                "bid_shading": "0.5",
                "future_cash_weight": "2",
                "liquidity_strength": "1.4",
                "objective_progress_weight": "0.15",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "bdb47a31da14ee95cddc9579b2dc477b6d6a04f99420ede12ebc7ccc3e81b07d"
            ),
            "winner-diagnostics.json": (
                "83d2a3043dd16d0fc30f7f47ee775e78cbaf781d396c2b8497c1c54ad46e5e0a"
            ),
            "winner-diagnostics.md": (
                "e49e4d27635a0bb2b70d83e41458bb15fe021af519ac2d7f3bedfb41c707be39"
            ),
        },
    },
    EXPECTED_V4_IDENTITIES[1]: {
        "predecessor": "balanced-v3",
        "search_name": "balanced-v4-search-v2",
        "profile": "4d391ce068d794767aff27aaa2782a63f57255402d41fe3ee7b0196edaed036e",
        "manifest": "e1f1bed8f09aef9193ffeb0ed3e0be822be96df7fd69985c9e4111f5c725933c",
        "report": "3c84573a97def0068bc417714232d8c7870a331029037aede73235c8d7b6efab",
        "evaluations": "1756519cb83597435fb395a569950f9f4a022c7aa3af48e9c2cc366c2a16b8e5",
        "selection": "bce530095669125a9e1162e93cc5a3c7df3ca2cba6a2393fa4fff9d467357cb6",
        "games": "54c38f79dc3690d1d0ef7eafa35d0f9cb4e8610ee166e19b37e9d053c409e273",
        "freeze": "126fbbd3d7d20dc66a239c0e7608365352c5077fee81c6b0d88c4410c5b28df3",
        "rating_delta": 9.953994694126777,
        "normalized_finish_delta": 0.9166666666666572,
        "final_money_delta": 92,
        "experts": {
            "early": {
                "bid_shading": "0.35",
                "future_cash_weight": "1.35",
                "liquidity_strength": "0.25",
                "objective_progress_weight": "0.3",
            },
            "middle": {
                "bid_shading": "0.35",
                "future_cash_weight": "1.55",
                "liquidity_strength": "0.3",
                "objective_progress_weight": "0.35",
            },
            "late": {
                "bid_shading": "0.35",
                "future_cash_weight": "1.45",
                "liquidity_strength": "0.45",
                "objective_progress_weight": "0.25",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "c6a6372898b25f26b7f34b14bca83743769492a68510f2f2f1aaf77c3f4a6e99"
            ),
            "winner-diagnostics.json": (
                "4ff4b1694b7807e39b58556a050a03d5ed77f825505dff08f70db857712e1029"
            ),
            "winner-diagnostics.md": (
                "7458b3e55f4efb352d14b09c79cb0907195625b9cd6aba004caa607f22d5b24d"
            ),
        },
    },
    EXPECTED_V4_IDENTITIES[2]: {
        "predecessor": "passive-v3",
        "search_name": "passive-v4-search-v2",
        "profile": "cf4f7b924ee3759d05eff38f47340951fb51c55827e469d8ea96a14e3cd4ccc4",
        "manifest": "334579f896a0d4281c8926bb4cc5d9bffd9b3c63b8be3d0ae3375699792d4bc6",
        "report": "dc6f291668c934bf3f16028d1fb5a03d4b36f4ae34f5209142724202d5fbd78c",
        "evaluations": "9723aafbcaba9566975c8f59650824c1ff5523924e4556d1e760b3f615ea1e13",
        "selection": "b561137db2cdaaae3ec930b5b52b0d81fae74bf34bf058228d48d94f80ad827c",
        "games": "1c9acc0e2b948d4661350547c761242d03d85cd74de6dc8a6c47cdb74f0c7f21",
        "freeze": "36285933ff9a36b45004a5cfd14dd828a7ebda10c6bf34eb87d7511ae8d68f84",
        "rating_delta": 12.092948249983237,
        "normalized_finish_delta": 1.8333333333333428,
        "final_money_delta": 3,
        "experts": {
            "early": {
                "bid_shading": "0.45",
                "future_cash_weight": "1.8",
                "liquidity_strength": "1.5",
                "objective_progress_weight": "0.95",
            },
            "middle": {
                "bid_shading": "0.45",
                "future_cash_weight": "1.75",
                "liquidity_strength": "1.5",
                "objective_progress_weight": "0.95",
            },
            "late": {
                "bid_shading": "0.4",
                "future_cash_weight": "2",
                "liquidity_strength": "1.5",
                "objective_progress_weight": "0.95",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "04d94fb3d302a39517bc383cec38bcf751670ed91c220e31beb3cc9fdfbf4db1"
            ),
            "winner-diagnostics.json": (
                "82822bf049202cbaed6db262c5f307e97221cfc1cf7bd34afe1db76c194344d5"
            ),
            "winner-diagnostics.md": (
                "994cfb68e0830841c9473babb4359c42b64933137bc42b549ea379af56ecbea6"
            ),
        },
    },
}
EXPECTED_V4_PHASE_OUTCOME_DIGESTS = {
    EXPECTED_V4_IDENTITIES[0]: "b5b2f0a1264120070ab3cf700f8fe54c6c50f4f23a8a399cfc2741276cff168f",
    EXPECTED_V4_IDENTITIES[1]: "fedc33758460714c2adf135270d5445fa46a9b6c083116ce7d433fc4ad61c70a",
    EXPECTED_V4_IDENTITIES[2]: "34bd3041dc5e1d5d17afbe70a9de86b50feb7460279d4f81f00fc16931d03936",
}
SAFE_PHASE_OUTCOME_FIELDS = (
    "selected_expert_phase",
    "decision_count",
    "eventual_final_money_sum",
    "eventual_normalized_finish_sum",
    "outright_win_decision_count",
    "tied_first_decision_count",
    "decisions_from_faulted_game_seat",
)
SYNTHETIC_V4_EXPERTS = {
    "early": {
        "liquidity_strength": "0.3",
        "future_cash_weight": "1.4",
        "objective_progress_weight": "0.2",
        "bid_shading": "0.25",
    },
    "middle": {
        "liquidity_strength": "0.5",
        "future_cash_weight": "1.2",
        "objective_progress_weight": "0.4",
        "bid_shading": "0.35",
    },
    "late": {
        "liquidity_strength": "0.7",
        "future_cash_weight": "0.8",
        "objective_progress_weight": "0.6",
        "bid_shading": "0.45",
    },
}
V4_PHASE_SELECTOR = {
    "kind": PHASE_SELECTOR_NAME,
    "early": "3*future>=2*total",
    "middle": "3*future>=total",
    "late": "otherwise",
}
V4_BOUNDARY_EVIDENCE = {
    "report_path": "docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md",
    "report_digest": "9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01",
    "slices_path": (
        "docs/benchmarks/tournaments/"
        "2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv"
    ),
    "slices_digest": "4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae",
}
SYNTHETIC_V4_WINNER_DIAGNOSTICS = {
    "winner-decision-slices.csv": "1" * 64,
    "winner-diagnostics.json": "2" * 64,
    "winner-diagnostics.md": "3" * 64,
}


def _phase_profile_digest() -> str:
    payload = {
        "experts": SYNTHETIC_V4_EXPERTS,
        "phase_selector": PHASE_SELECTOR_NAME,
    }
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _expert_profile_digest(coefficients: dict[str, str]) -> str:
    encoded = (
        json.dumps(coefficients, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _schema_v2_frozen_payload() -> dict[str, Any]:
    profile_digest = _phase_profile_digest()
    return {
        "boundary_evidence": V4_BOUNDARY_EVIDENCE,
        "development_corpus": {
            "digest": DEVELOPMENT_DIGEST,
            "name": "development-v1",
        },
        "development_scores": {
            "final_money_delta": 17,
            "normalized_finish_delta": 1.25,
            "rating_delta": 9.5,
        },
        "experts": SYNTHETIC_V4_EXPERTS,
        "generation": 1,
        "identity": f"aggressive-v4-candidate-g001-s002-{profile_digest[:12]}",
        "parent_identity": "aggressive-v4-candidate-g000-s000-000000000000",
        "personality": "aggressive",
        "phase_selector": V4_PHASE_SELECTOR,
        "predecessor_name": "aggressive-v3",
        "profile_digest": profile_digest,
        "repository_commit": "a" * 40,
        "schema_version": 2,
        "search": {
            "manifest_digest": "b" * 64,
            "name": "aggressive-v4-search-v2",
        },
        "slot": 2,
        "source_evidence": {
            "candidate_evaluations_sha256": "c" * 64,
            "development_games_sha256": "d" * 64,
            "search_report_sha256": "e" * 64,
            "selection_log_sha256": "f" * 64,
            "winner_diagnostics": SYNTHETIC_V4_WINNER_DIAGNOSTICS,
        },
    }


def _schema_v2_catalog_entry(
    payload: dict[str, Any],
    *,
    candidate_bytes: bytes,
) -> dict[str, Any]:
    source_evidence = payload["source_evidence"]
    boundary_evidence = payload["boundary_evidence"]
    winner_diagnostics = source_evidence["winner_diagnostics"]
    return {
        "boundary_report_digest": boundary_evidence["report_digest"],
        "boundary_slices_digest": boundary_evidence["slices_digest"],
        "candidate_evaluations_sha256": source_evidence["candidate_evaluations_sha256"],
        "development_corpus_digest": payload["development_corpus"]["digest"],
        "development_corpus_name": payload["development_corpus"]["name"],
        "development_games_sha256": source_evidence["development_games_sha256"],
        "file": f"{payload['identity']}.json",
        "identity": payload["identity"],
        "manifest_digest": payload["search"]["manifest_digest"],
        "personality": payload["personality"],
        "predecessor_name": payload["predecessor_name"],
        "profile_digest": payload["profile_digest"],
        "repository_commit": payload["repository_commit"],
        "search_name": payload["search"]["name"],
        "search_report_sha256": source_evidence["search_report_sha256"],
        "selection_log_sha256": source_evidence["selection_log_sha256"],
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "winner_decision_slices_sha256": winner_diagnostics["winner-decision-slices.csv"],
        "winner_diagnostics_json_sha256": winner_diagnostics["winner-diagnostics.json"],
        "winner_diagnostics_markdown_sha256": winner_diagnostics["winner-diagnostics.md"],
    }


def _write_mixed_schema_catalog(
    tmp_path: Path,
    *,
    payload_mutation: Any | None = None,
    entry_mutation: Any | None = None,
    mirror_mutated_payload_in_entry: bool = False,
) -> Path:
    source_index = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))
    source_index["candidates"] = [
        entry for entry in source_index["candidates"] if "-v3-candidate-" in entry["identity"]
    ]
    for entry in source_index["candidates"]:
        source = CATALOG_DIR / entry["file"]
        (tmp_path / entry["file"]).write_bytes(source.read_bytes())

    payload = json.loads(json.dumps(_schema_v2_frozen_payload()))
    if payload_mutation is not None:
        payload_mutation(payload)
    candidate_bytes = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    entry_payload = payload if mirror_mutated_payload_in_entry else _schema_v2_frozen_payload()
    candidate_path = tmp_path / f"{entry_payload['identity']}.json"
    candidate_path.write_bytes(candidate_bytes)

    entry = _schema_v2_catalog_entry(
        entry_payload,
        candidate_bytes=candidate_bytes,
    )
    if entry_mutation is not None:
        entry_mutation(entry)
    source_index["candidates"].append(entry)
    source_index["candidates"].sort(key=lambda item: item["identity"])
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(source_index, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_path


def test_catalog_exposes_exact_frozen_records_and_provenance() -> None:
    assert tuple(item.identity for item in FROZEN_CANDIDATES) == EXPECTED_IDENTITIES
    assert tuple(FROZEN_CANDIDATES_BY_NAME) == EXPECTED_IDENTITIES

    v3_candidates = tuple(
        candidate for candidate in FROZEN_CANDIDATES if type(candidate) is FrozenCandidate
    )
    assert tuple(candidate.identity for candidate in v3_candidates) == EXPECTED_V3_IDENTITIES
    for v3_candidate in v3_candidates:
        assert type(v3_candidate) is FrozenCandidate
        expected = EXPECTED_V3_PROVENANCE[v3_candidate.identity]
        assert v3_candidate.bot_spec.name == v3_candidate.identity
        assert v3_candidate.bot_spec.bot_id == v3_candidate.identity
        assert v3_candidate.predecessor_name == expected["predecessor"]
        assert v3_candidate.search_name == expected["search_name"]
        assert v3_candidate.development_corpus_name == "development-v1"
        assert v3_candidate.development_corpus_digest == DEVELOPMENT_DIGEST
        assert v3_candidate.freeze_digest == expected["freeze"]
        assert v3_candidate.profile_digest == expected["profile"]
        assert v3_candidate.manifest_digest == expected["manifest"]
        assert v3_candidate.search_report_digest == expected["report"]
        assert v3_candidate.candidate_evaluations_digest == expected["evaluations"]

    v4_candidates = tuple(
        candidate
        for candidate in FROZEN_CANDIDATES
        if isinstance(candidate, FrozenPhaseAwareCandidate)
    )
    assert tuple(candidate.identity for candidate in v4_candidates) == EXPECTED_V4_IDENTITIES
    for v4_candidate in v4_candidates:
        expected = EXPECTED_V4_PROVENANCE[v4_candidate.identity]
        expected_experts = expected["experts"]
        assert v4_candidate.bot_spec.name == v4_candidate.identity
        assert v4_candidate.bot_spec.bot_id == v4_candidate.identity
        assert v4_candidate.predecessor_name == expected["predecessor"]
        assert v4_candidate.search_name == expected["search_name"]
        assert v4_candidate.development_corpus_name == "development-v1"
        assert v4_candidate.development_corpus_digest == DEVELOPMENT_DIGEST
        assert v4_candidate.repository_commit == "a66c49e559849b35a290827b51b2e5098524e2d1"
        assert v4_candidate.freeze_digest == expected["freeze"]
        assert v4_candidate.profile_digest == expected["profile"]
        assert v4_candidate.manifest_digest == expected["manifest"]
        assert v4_candidate.search_report_digest == expected["report"]
        assert v4_candidate.candidate_evaluations_digest == expected["evaluations"]
        assert v4_candidate.selection_log_digest == expected["selection"]
        assert v4_candidate.development_games_digest == expected["games"]
        assert v4_candidate.rating_delta == expected["rating_delta"]
        assert v4_candidate.normalized_finish_delta == expected["normalized_finish_delta"]
        assert v4_candidate.final_money_delta == expected["final_money_delta"]
        assert v4_candidate.profile.phase_selector == PHASE_SELECTOR_NAME
        assert dict(v4_candidate.phase_selector_rules) == V4_PHASE_SELECTOR
        assert v4_candidate.boundary_report_path == V4_BOUNDARY_EVIDENCE["report_path"]
        assert v4_candidate.boundary_report_digest == V4_BOUNDARY_EVIDENCE["report_digest"]
        assert v4_candidate.boundary_slices_path == V4_BOUNDARY_EVIDENCE["slices_path"]
        assert v4_candidate.boundary_slices_digest == V4_BOUNDARY_EVIDENCE["slices_digest"]
        assert dict(v4_candidate.winner_diagnostics_digests) == expected["diagnostics"]
        assert dict(v4_candidate.expert_digests) == {
            phase: _expert_profile_digest(expected_experts[phase])
            for phase in ("early", "middle", "late")
        }
        for phase in ("early", "middle", "late"):
            expert = getattr(v4_candidate.profile, phase)
            for coefficient_name, coefficient in expected_experts[phase].items():
                assert getattr(expert, coefficient_name) == float(coefficient)


def test_catalog_files_are_exact_copies_of_committed_development_freezes() -> None:
    index = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))

    assert set(index) == {"schema_version", "candidates"}
    assert index["schema_version"] == 1
    assert [entry["identity"] for entry in index["candidates"]] == list(EXPECTED_IDENTITIES)
    for entry in index["candidates"]:
        identity = entry["identity"]
        installed = CATALOG_DIR / entry["file"]
        search_dir = BENCHMARK_DIR / entry["search_name"]
        benchmark = search_dir / "frozen-candidate.json"
        assert installed.read_bytes() == benchmark.read_bytes()
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == entry["sha256"]
        report = json.loads((search_dir / "search-report.json").read_text(encoding="utf-8"))
        manifest = json.loads((search_dir / "search-manifest.json").read_text(encoding="utf-8"))
        corpus = json.loads(
            (search_dir / "development-corpus-snapshot.json").read_text(encoding="utf-8")
        )
        assert (
            hashlib.sha256((search_dir / "search-report.json").read_bytes()).hexdigest()
            == entry["search_report_sha256"]
        )
        assert (
            hashlib.sha256((search_dir / "candidate-evaluations.jsonl").read_bytes()).hexdigest()
            == entry["candidate_evaluations_sha256"]
        )
        assert report["status"] == "frozen_improvement"
        assert report["frozen_candidate_identity"] == identity
        assert report["search"]["name"] == entry["search_name"]
        assert manifest["digest"] == entry["manifest_digest"]
        assert corpus["recipe"]["purpose"] == "development"
        assert corpus["recipe"]["name"] == entry["development_corpus_name"]
        assert corpus["digest"] == entry["development_corpus_digest"]
        if identity in EXPECTED_V3_PROVENANCE:
            expected = EXPECTED_V3_PROVENANCE[identity]
            assert entry["sha256"] == expected["freeze"]
            continue

        expected = EXPECTED_V4_PROVENANCE[identity]
        payload = json.loads(installed.read_text(encoding="utf-8"))
        assert entry["sha256"] == expected["freeze"]
        assert payload["identity"] == identity
        assert payload["profile_digest"] == expected["profile"]
        assert payload["experts"] == expected["experts"]
        assert payload["phase_selector"] == V4_PHASE_SELECTOR
        assert payload["boundary_evidence"] == V4_BOUNDARY_EVIDENCE
        assert payload["source_evidence"] == {
            "candidate_evaluations_sha256": expected["evaluations"],
            "development_games_sha256": expected["games"],
            "search_report_sha256": expected["report"],
            "selection_log_sha256": expected["selection"],
            "winner_diagnostics": expected["diagnostics"],
        }
        assert (
            hashlib.sha256((search_dir / "selection-log.jsonl").read_bytes()).hexdigest()
            == expected["selection"]
            == entry["selection_log_sha256"]
        )
        assert (
            hashlib.sha256((search_dir / "development-games.jsonl").read_bytes()).hexdigest()
            == expected["games"]
            == entry["development_games_sha256"]
        )
        for artifact_name, catalog_field in (
            ("winner-diagnostics.json", "winner_diagnostics_json_sha256"),
            ("winner-diagnostics.md", "winner_diagnostics_markdown_sha256"),
        ):
            assert (
                hashlib.sha256((search_dir / artifact_name).read_bytes()).hexdigest()
                == expected["diagnostics"][artifact_name]
                == entry[catalog_field]
            )


@pytest.mark.parametrize("identity", EXPECTED_V4_IDENTITIES)
def test_v4_privacy_redaction_withholds_detailed_slices_and_publishes_phase_totals(
    identity: str,
) -> None:
    expected = EXPECTED_V4_PROVENANCE[identity]
    search_dir = BENCHMARK_DIR / expected["search_name"]
    detailed_slices = search_dir / "winner-decision-slices.csv"
    replacement = search_dir / "winner-phase-outcomes.csv"
    redaction_path = search_dir / "privacy-redaction.json"
    diagnostics_path = search_dir / "winner-diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    redaction = json.loads(redaction_path.read_text(encoding="utf-8"))
    frozen = json.loads((search_dir / "frozen-candidate.json").read_text(encoding="utf-8"))
    catalog = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))
    catalog_entry = next(entry for entry in catalog["candidates"] if entry["identity"] == identity)
    report = json.loads((search_dir / "search-report.json").read_text(encoding="utf-8"))
    report_digests = {
        artifact["name"]: artifact["sha256"]
        for artifact in report["winner_diagnostics"]["artifacts"]
    }
    withheld_digest = expected["diagnostics"]["winner-decision-slices.csv"]

    assert not detailed_slices.exists()
    assert frozen["source_evidence"]["winner_diagnostics"]["winner-decision-slices.csv"] == (
        withheld_digest
    )
    assert catalog_entry["winner_decision_slices_sha256"] == withheld_digest
    assert report_digests["winner-decision-slices.csv"] == withheld_digest

    assert set(redaction) == {
        "promotion_result_changed",
        "reason",
        "reason_code",
        "replacement_artifact",
        "replacement_basis",
        "schema_version",
        "search_name",
        "search_selection_changed",
        "surviving_diagnostic_artifacts",
        "withheld_artifact",
    }
    assert redaction["schema_version"] == 1
    assert redaction["search_name"] == expected["search_name"]
    assert redaction["reason_code"] == "withhold_high_dimensional_singleton_decision_slices"
    assert redaction["reason"] == (
        "The detailed slice table grouped decisions so narrowly that most rows represented "
        "one decision and could be linked to reproducible game seeds."
    )
    assert redaction["replacement_basis"] == (
        "winner-phase-outcomes.csv is a deterministic three-row projection of phase_outcomes "
        "in winner-diagnostics.json; no simulation was rerun."
    )
    assert redaction["search_selection_changed"] is False
    assert redaction["promotion_result_changed"] is False
    assert redaction["withheld_artifact"] == {
        "name": "winner-decision-slices.csv",
        "sha256": withheld_digest,
    }

    replacement_bytes = replacement.read_bytes()
    replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
    assert redaction["replacement_artifact"] == {
        "name": "winner-phase-outcomes.csv",
        "sha256": EXPECTED_V4_PHASE_OUTCOME_DIGESTS[identity],
    }
    assert replacement_digest == EXPECTED_V4_PHASE_OUTCOME_DIGESTS[identity]
    assert replacement_bytes.endswith(b"\n")
    assert b"\r" not in replacement_bytes

    reader = csv.DictReader(replacement_bytes.decode("utf-8").splitlines())
    rows = list(reader)
    assert tuple(reader.fieldnames or ()) == SAFE_PHASE_OUTCOME_FIELDS
    assert len(rows) == 3
    assert rows == [
        {field: str(outcome[field]) for field in SAFE_PHASE_OUTCOME_FIELDS}
        for outcome in diagnostics["phase_outcomes"]
    ]

    surviving = redaction["surviving_diagnostic_artifacts"]
    assert set(surviving) == {"winner-diagnostics.json", "winner-diagnostics.md"}
    for artifact_name, digest in surviving.items():
        assert hashlib.sha256((search_dir / artifact_name).read_bytes()).hexdigest() == digest
        assert expected["diagnostics"][artifact_name] == digest


def test_frozen_specs_are_picklable_local_only_and_not_released() -> None:
    released_names = {spec.name for spec in BOT_SPECS}
    default_names = {spec.name for spec in DEFAULT_TOURNAMENT_BOT_SPECS}

    for candidate in FROZEN_CANDIDATES:
        spec = candidate.bot_spec
        assert candidate.identity not in BOT_SPECS_BY_NAME
        assert candidate.identity not in released_names
        assert candidate.identity not in default_names
        assert isinstance(spec.brain_factory, partial)

        restored = pickle.loads(pickle.dumps(spec))
        brain = restored.make_brain(seed=8675309)
        if isinstance(candidate, FrozenPhaseAwareCandidate):
            assert isinstance(brain, PhaseAwareHeuristicBotBrain)
            assert brain.profile == candidate.profile
        else:
            assert isinstance(brain, HeuristicBotBrain)
            assert brain.valuator.profile == candidate.profile


def test_mixed_catalog_loads_a_strict_local_only_phase_aware_candidate(
    tmp_path: Path,
) -> None:
    candidates = load_frozen_candidates(_write_mixed_schema_catalog(tmp_path))
    phase_candidate = next(
        candidate for candidate in candidates if isinstance(candidate, FrozenPhaseAwareCandidate)
    )

    assert (
        tuple(
            candidate.identity
            for candidate in candidates
            if not isinstance(candidate, FrozenPhaseAwareCandidate)
        )
        == EXPECTED_V3_IDENTITIES
    )
    assert phase_candidate.identity == _schema_v2_frozen_payload()["identity"]
    assert phase_candidate.predecessor_name == "aggressive-v3"
    assert phase_candidate.search_name == "aggressive-v4-search-v2"
    assert phase_candidate.profile_digest == _phase_profile_digest()
    assert phase_candidate.profile.phase_selector == PHASE_SELECTOR_NAME
    assert dict(phase_candidate.phase_selector_rules) == V4_PHASE_SELECTOR
    expert_digests = dict(phase_candidate.expert_digests)
    assert expert_digests == {
        phase: _expert_profile_digest(SYNTHETIC_V4_EXPERTS[phase])
        for phase in ("early", "middle", "late")
    }
    assert expert_digests["early"] != expert_digests["middle"]
    assert expert_digests["middle"] != expert_digests["late"]
    assert phase_candidate.profile.early.liquidity_strength == 0.3
    assert phase_candidate.profile.middle.future_cash_weight == 1.2
    assert phase_candidate.profile.late.bid_shading == 0.45
    assert phase_candidate.boundary_report_digest == V4_BOUNDARY_EVIDENCE["report_digest"]
    assert phase_candidate.boundary_slices_digest == V4_BOUNDARY_EVIDENCE["slices_digest"]
    assert phase_candidate.selection_log_digest == "f" * 64
    assert phase_candidate.development_games_digest == "d" * 64
    assert dict(phase_candidate.winner_diagnostics_digests) == SYNTHETIC_V4_WINNER_DIAGNOSTICS

    spec = phase_candidate.bot_spec
    assert spec.name == phase_candidate.identity
    assert spec.bot_id == phase_candidate.identity
    assert isinstance(spec.brain_factory, partial)
    restored = pickle.loads(pickle.dumps(spec))
    brain = restored.make_brain(seed=8675309)
    assert isinstance(brain, PhaseAwareHeuristicBotBrain)
    assert brain.profile == phase_candidate.profile

    released_names = {item.name for item in BOT_SPECS}
    default_names = {item.name for item in DEFAULT_TOURNAMENT_BOT_SPECS}
    assert phase_candidate.identity not in BOT_SPECS_BY_NAME
    assert phase_candidate.identity not in released_names
    assert phase_candidate.identity not in default_names


def _swap_early_and_middle_experts(payload: dict[str, Any]) -> None:
    experts = payload["experts"]
    experts["early"], experts["middle"] = experts["middle"], experts["early"]


def _use_out_of_range_generation(payload: dict[str, Any]) -> None:
    payload["generation"] = 12
    payload["identity"] = payload["identity"].replace("-g001-", "-g012-")


def _use_out_of_range_slot(payload: dict[str, Any]) -> None:
    payload["slot"] = 16
    payload["identity"] = payload["identity"].replace("-s002-", "-s016-")


def _forge_identity_profile_prefix(payload: dict[str, Any]) -> None:
    payload["identity"] = f"{payload['identity'][:-12]}000000000000"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"schema_version": 1}), "keys|schema"),
        (lambda payload: payload.update({"schema_version": True}), "integer"),
        (lambda payload: payload.update({"generation": True}), "integer"),
        (lambda payload: payload.update({"personality": "balanced"}), "identity"),
        (lambda payload: payload.update({"slot": 3}), "identity"),
        (
            lambda payload: payload["experts"]["early"].update({"liquidity_strength": True}),
            "decimal",
        ),
        (
            lambda payload: payload["experts"]["early"].update({"liquidity_strength": "0.30"}),
            "canonical",
        ),
        (
            lambda payload: payload["experts"]["early"].update({"liquidity_strength": "0.31"}),
            "0.05",
        ),
        (
            lambda payload: payload["experts"]["early"].update({"future_cash_weight": "2.05"}),
            "0.05",
        ),
        (
            lambda payload: payload["experts"].pop("middle"),
            "expert|missing",
        ),
        (
            lambda payload: payload["experts"]["late"].update({"unknown": "0"}),
            "unknown",
        ),
        (_swap_early_and_middle_experts, "profile digest"),
        (
            lambda payload: payload["phase_selector"].update({"middle": "forged"}),
            "selector",
        ),
        (
            lambda payload: payload["phase_selector"].pop("early"),
            "selector|missing",
        ),
        (
            lambda payload: payload["phase_selector"].update({"unknown": "value"}),
            "selector|unknown",
        ),
        (
            lambda payload: payload["phase_selector"].update(
                {
                    "early": V4_PHASE_SELECTOR["late"],
                    "late": V4_PHASE_SELECTOR["early"],
                }
            ),
            "selector",
        ),
        (
            lambda payload: payload["boundary_evidence"].update({"report_digest": "0" * 64}),
            "boundary",
        ),
        (
            lambda payload: payload["boundary_evidence"].update(
                {"report_path": "docs/benchmarks/forged.md"}
            ),
            "boundary",
        ),
        (
            lambda payload: payload["boundary_evidence"].update(
                {"slices_path": "docs/benchmarks/forged.csv"}
            ),
            "boundary",
        ),
        (
            lambda payload: payload["boundary_evidence"].update({"slices_digest": "0" * 64}),
            "boundary",
        ),
        (
            lambda payload: payload.update({"predecessor_name": "balanced-v3"}),
            "predecessor",
        ),
        (
            lambda payload: payload["search"].update({"name": "aggressive-v3-search-v1"}),
            "search",
        ),
        (
            lambda payload: payload["search"].update({"name": "balanced-v4-search-v2"}),
            "search",
        ),
        (
            lambda payload: payload.update(
                {"parent_identity": ("aggressive-v3-candidate-g000-s000-000000000000")}
            ),
            "parent",
        ),
        (
            lambda payload: payload.update(
                {"parent_identity": ("balanced-v4-candidate-g000-s000-000000000000")}
            ),
            "parent",
        ),
        (
            lambda payload: payload.update(
                {"parent_identity": ("aggressive-v4-candidate-g001-s000-000000000000")}
            ),
            "parent",
        ),
        (
            lambda payload: payload.update(
                {"parent_identity": ("aggressive-v4-candidate-g000-s999-000000000000")}
            ),
            "parent",
        ),
        (
            lambda payload: payload["development_corpus"].update({"name": "held-out-v1"}),
            "development",
        ),
        (
            lambda payload: payload["development_corpus"].update({"digest": "0" * 64}),
            "development",
        ),
        (
            lambda payload: payload["source_evidence"].pop("candidate_evaluations_sha256"),
            "source evidence|missing",
        ),
        (
            lambda payload: payload["source_evidence"].update({"unknown": "value"}),
            "source evidence|unknown",
        ),
        (
            lambda payload: payload["source_evidence"].update({"search_report_sha256": True}),
            "search report|digest",
        ),
        (
            lambda payload: payload["source_evidence"].update(
                {"candidate_evaluations_sha256": True}
            ),
            "candidate evaluations|digest",
        ),
        (
            lambda payload: payload["source_evidence"].update({"selection_log_sha256": True}),
            "selection log|digest",
        ),
        (
            lambda payload: payload["source_evidence"].update({"development_games_sha256": True}),
            "development games|digest",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].pop(
                "winner-decision-slices.csv"
            ),
            "winner diagnostics|missing",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].pop(
                "winner-diagnostics.json"
            ),
            "winner diagnostics|missing",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].pop(
                "winner-diagnostics.md"
            ),
            "winner diagnostics|missing",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].update(
                {"winner-decision-slices.csv": True}
            ),
            "winner diagnostics|digest",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].update(
                {"winner-diagnostics.json": True}
            ),
            "winner diagnostics|digest",
        ),
        (
            lambda payload: payload["source_evidence"]["winner_diagnostics"].update(
                {"winner-diagnostics.md": True}
            ),
            "winner diagnostics|digest",
        ),
        (
            lambda payload: payload["development_scores"].update({"rating_delta": True}),
            "finite",
        ),
        (
            lambda payload: payload["development_scores"].update({"rating_delta": 0}),
            "positive",
        ),
        (
            lambda payload: payload["development_scores"].update({"final_money_delta": False}),
            "integer",
        ),
    ),
)
def test_schema_v2_catalog_rejects_tampered_frozen_payload(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    index_path = _write_mixed_schema_catalog(
        tmp_path,
        payload_mutation=mutation,
    )

    with pytest.raises(FrozenCandidateCatalogError, match=message):
        load_frozen_candidates(index_path)


@pytest.mark.parametrize(
    ("entry_mutation", "message"),
    (
        (
            lambda entry: entry.update({"selection_log_sha256": True}),
            "selection log|digest|string",
        ),
        (
            lambda entry: entry.update({"selection_log_sha256": "0" * 64}),
            "selection log|catalog",
        ),
        (
            lambda entry: entry.update({"search_report_sha256": "0" * 64}),
            "search report|catalog",
        ),
        (
            lambda entry: entry.update({"candidate_evaluations_sha256": "0" * 64}),
            "candidate evaluations|catalog",
        ),
        (
            lambda entry: entry.update({"development_games_sha256": "0" * 64}),
            "development games|catalog",
        ),
        (
            lambda entry: entry.update({"boundary_report_digest": "0" * 64}),
            "boundary report|catalog",
        ),
        (
            lambda entry: entry.update({"boundary_slices_digest": "0" * 64}),
            "boundary slices|catalog",
        ),
        (
            lambda entry: entry.update({"winner_decision_slices_sha256": "0" * 64}),
            "winner decision slices|catalog",
        ),
        (
            lambda entry: entry.update({"winner_diagnostics_json_sha256": "0" * 64}),
            "winner diagnostics|catalog",
        ),
        (
            lambda entry: entry.update({"winner_diagnostics_markdown_sha256": "0" * 64}),
            "winner diagnostics|catalog",
        ),
        (
            lambda entry: entry.pop("development_games_sha256"),
            "missing",
        ),
        (
            lambda entry: entry.update({"unknown": "value"}),
            "unknown",
        ),
    ),
)
def test_schema_v2_catalog_rejects_tampered_index_provenance(
    tmp_path: Path,
    entry_mutation: Any,
    message: str,
) -> None:
    index_path = _write_mixed_schema_catalog(
        tmp_path,
        entry_mutation=entry_mutation,
    )

    with pytest.raises(FrozenCandidateCatalogError, match=message):
        load_frozen_candidates(index_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_use_out_of_range_generation, "identity|generation"),
        (_use_out_of_range_slot, "identity|slot"),
        (_forge_identity_profile_prefix, "profile digest"),
    ),
)
def test_schema_v2_catalog_rejects_invalid_v4_identity_bounds_and_digest_prefix(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    index_path = _write_mixed_schema_catalog(
        tmp_path,
        payload_mutation=mutation,
        mirror_mutated_payload_in_entry=True,
    )

    with pytest.raises(FrozenCandidateCatalogError, match=message):
        load_frozen_candidates(index_path)


def test_schema_v2_catalog_rejects_duplicate_nested_json_keys(tmp_path: Path) -> None:
    index_path = _write_mixed_schema_catalog(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    phase_entry = next(entry for entry in index["candidates"] if "-v4-" in entry["identity"])
    candidate_path = tmp_path / phase_entry["file"]
    source = candidate_path.read_text(encoding="utf-8")
    source = source.replace(
        '"winner_diagnostics": {',
        '"winner_diagnostics": {}, "winner_diagnostics": {',
        1,
    )
    candidate_path.write_text(source, encoding="utf-8")
    phase_entry["sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    index_path.write_text(
        json.dumps(index, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FrozenCandidateCatalogError, match="duplicate JSON key"):
        load_frozen_candidates(index_path)


def test_v4_identity_rejects_schema_v1_payload_before_payload_key_dispatch(
    tmp_path: Path,
) -> None:
    index_path = _write_mixed_schema_catalog(
        tmp_path,
        payload_mutation=lambda payload: payload.update({"schema_version": 1}),
    )

    with pytest.raises(FrozenCandidateCatalogError) as captured:
        load_frozen_candidates(index_path)

    assert captured.value.code == "candidate_schema_identity_mismatch"


def _write_modified_catalog(
    tmp_path: Path,
    *,
    index_mutation: Any | None = None,
    payload_mutation: Any | None = None,
) -> Path:
    source_index = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))
    index = json.loads(json.dumps(source_index))
    source_entry = index["candidates"][0]
    source_path = CATALOG_DIR / source_entry["file"]
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload_mutation is not None:
        payload_mutation(payload)
    candidate_bytes = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    candidate_path = tmp_path / source_entry["file"]
    candidate_path.write_bytes(candidate_bytes)
    source_entry["sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    if index_mutation is not None:
        index_mutation(index)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(index, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda index: index.update({"unknown": True}), "unknown"),
        (
            lambda index: index["candidates"][0].update({"sha256": "0" * 64}),
            "digest",
        ),
        (
            lambda index: index["candidates"][0].update({"file": "../escape.json"}),
            "file",
        ),
        (
            lambda index: index["candidates"][0].pop("identity"),
            "missing.*identity",
        ),
        (
            lambda index: index["candidates"].append(index["candidates"][0].copy()),
            "duplicate",
        ),
    ),
)
def test_catalog_rejects_invalid_index(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    index_path = _write_modified_catalog(tmp_path, index_mutation=mutation)

    with pytest.raises(FrozenCandidateCatalogError, match=message):
        load_frozen_candidates(index_path)


def test_schema_v1_catalog_rejects_unknown_entry_keys_before_file_checks(
    tmp_path: Path,
) -> None:
    def mutate_entry_and_file(index: dict[str, Any]) -> None:
        entry = index["candidates"][0]
        entry["unknown"] = True
        entry["file"] = "../escape.json"

    index_path = _write_modified_catalog(
        tmp_path,
        index_mutation=mutate_entry_and_file,
    )

    with pytest.raises(FrozenCandidateCatalogError) as captured:
        load_frozen_candidates(index_path)

    assert captured.value.code == "invalid_object_keys"


def test_v3_identity_rejects_schema_v2_payload_before_payload_key_dispatch(
    tmp_path: Path,
) -> None:
    index_path = _write_modified_catalog(
        tmp_path,
        payload_mutation=lambda payload: payload.update({"schema_version": 2}),
    )

    with pytest.raises(FrozenCandidateCatalogError) as captured:
        load_frozen_candidates(index_path)

    assert captured.value.code == "candidate_schema_identity_mismatch"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"unknown": True}), "unknown"),
        (
            lambda payload: payload.update({"identity": "aggressive-v3-candidate-wrong"}),
            "identity",
        ),
        (
            lambda payload: payload["coefficients"].update({"bid_shading": "0.45"}),
            "profile digest",
        ),
        (
            lambda payload: payload.update({"predecessor_name": "balanced-v2"}),
            "predecessor",
        ),
        (
            lambda payload: payload["search"].update({"name": "balanced-v3-search-v1"}),
            "search",
        ),
        (
            lambda payload: payload["development_corpus"].update({"digest": "0" * 64}),
            "development corpus",
        ),
        (
            lambda payload: payload.pop("source_evidence"),
            "source evidence",
        ),
    ),
)
def test_catalog_rejects_tampered_frozen_payload(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    index_path = _write_modified_catalog(tmp_path, payload_mutation=mutation)

    with pytest.raises(FrozenCandidateCatalogError, match=message):
        load_frozen_candidates(index_path)


def test_frozen_candidates_complete_development_matrix_without_faults() -> None:
    corpus = load_promotion_corpus(
        REPOSITORY_ROOT / "configs" / "promotion" / "development-v1.json",
        registry=BOT_SPECS_BY_NAME,
    )
    expected_matrix = set(product(tuple("ABCDE"), (3, 4, 5)))

    for candidate in FROZEN_CANDIDATES:
        plan = plan_development_games(
            corpus,
            candidate=candidate.bot_spec,
            incumbent=BOT_SPECS_BY_NAME[candidate.predecessor_name],
            registry=BOT_SPECS_BY_NAME,
        )
        assert {
            (job.value_chart, job.player_count) for job in plan.candidate_jobs
        } == expected_matrix

        result = MonteCarloRunner.run_jobs(
            plan.candidate_config,
            plan.candidate_jobs,
            workers=1,
            batch_size=64,
        )

        assert len(result.game_summaries) == len(corpus.cases) == 240
        assert tuple(summary.game_index for summary in result.game_summaries) == tuple(range(240))
        assert sum(sum(summary.fault_counts) for summary in result.game_summaries) == 0
