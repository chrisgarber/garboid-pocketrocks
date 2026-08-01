from __future__ import annotations

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
EXPECTED_V3_IDENTITIES = (
    "aggressive-v3-candidate-g007-s009-9c43f610b2f0",
    "balanced-v3-candidate-g005-s005-90544d0f26d2",
    "passive-v3-candidate-g006-s000-739e30e8d844",
)
EXPECTED_V4_IDENTITIES = (
    "aggressive-v4-candidate-g011-s014-9a2908cce71c",
    "balanced-v4-candidate-g005-s010-ae48ac912b3a",
    "passive-v4-candidate-g011-s012-fcf5cb322e51",
)
EXPECTED_IDENTITIES = tuple(sorted((*EXPECTED_V3_IDENTITIES, *EXPECTED_V4_IDENTITIES)))
EXPECTED_V3_PROVENANCE: dict[str, dict[str, Any]] = {
    EXPECTED_V3_IDENTITIES[0]: {
        "predecessor": "aggressive-v2",
        "search_name": "aggressive-v3-search-v1",
        "profile": "9c43f610b2f03b2e5d3cb71c2a3ce7a62f456f3fdce73d3a52c501bab9a81042",
        "manifest": "b4e21379ebdcf9462330a1e55412993cf69273a257afc7bb9197cb2e7d89532f",
        "report": "db53ccbe258e2a08969d43710f7979d2e3b16f18415c4c11b81654a6301abea2",
        "evaluations": "47b82f973f36b3a21d7aa30d2872b542c23a55bfaf5666c715cc446fdb509139",
        "freeze": "89b61c17485da34b0c496cd353f0ccd1c533ba843175e967e3af939528bf5735",
        "development": "6531e85f69f4e085b8ac789348be6c21614455dd77ba7ac791f5390479e17638",
    },
    EXPECTED_V3_IDENTITIES[1]: {
        "predecessor": "balanced-v2",
        "search_name": "balanced-v3-search-v1",
        "profile": "90544d0f26d20a9fc3b51013fd244908be1071a13eeef12fb0d4b2c4aef251de",
        "manifest": "9333619e4d573f12736162065f02901d25fa5e267ea6fd1838371de4b9f04c07",
        "report": "98056eda744f19d9f67dd763c64e5d5e2e97a3e0538534e933058d3a34227dfb",
        "evaluations": "4b8c1ea06575971914d183cfcba99964b50b37d60c740b52fe417b8f80a85eb7",
        "freeze": "2da90c344b4f573f83718faad717a9accea2768b21f1982d8f543a5f2bb25fa8",
        "development": "d556cc940c92ebf3633fde83485d4ba776b6e582f34cf8d445a5c190824b3228",
    },
    EXPECTED_V3_IDENTITIES[2]: {
        "predecessor": "passive-v2",
        "search_name": "passive-v3-search-v1",
        "profile": "739e30e8d84482d3e165c230a8bb9f518eccb822adb55337e1f8abae82399dd7",
        "manifest": "53a2927888b43aa17cbbc9fe7cd15a48b071054044aef557778dfc860fd1d8cd",
        "report": "ba48cc835a6918f27cdba453fac9c2672c3b59b6ed53dc58a43519b01ddeb4d3",
        "evaluations": "12c80080fec844c13f11bf310ba4c0df6e24f3c5ba09d34e8adb733b2b359492",
        "freeze": "dc64f303def70e1ef17d3ebddc8837c645941078bbb7b14bbfa211d0e7571fad",
        "development": "64124822038895aa4048469244c99177c877271368cc246d2250b737a14bf658",
    },
}
AGGRESSIVE_DEVELOPMENT_DIGEST = "6531e85f69f4e085b8ac789348be6c21614455dd77ba7ac791f5390479e17638"
EXPECTED_V4_PROVENANCE: dict[str, dict[str, Any]] = {
    EXPECTED_V4_IDENTITIES[0]: {
        "predecessor": "aggressive-v3",
        "search_name": "aggressive-v4-search-v2",
        "development_name": "development-aggressive-v3-broad-v1",
        "development": AGGRESSIVE_DEVELOPMENT_DIGEST,
        "profile": "9a2908cce71cefadc7e8c1adfe56c1d0e5875e2ceafd8205183bed00c93a5d75",
        "manifest": "0b85403b121451b2a4ab495a1c73af93c311f2743c5e52ce6accd8ef606df324",
        "report": "733fb7ac94106887861ac372018eced9ca06a012a46a0741b98c746272ae49da",
        "evaluations": "f716f83591688782f9c08f96a072cf619ffd2fe159160c996a6e02cba2a7cd61",
        "selection": "00ff789758cc7f144c8c01b6edda462f4d00baedb1d26e4e59d969c1c7398339",
        "games": "60075dba729927c31a261fe925f876ff733cbcb8ead222346fed4791f9a3079a",
        "freeze": "5d057574b2fd2346df3f94ef8accd0de3070247f9e964b4df969c6534fdec66d",
        "rating_delta": 129.78608041843881,
        "normalized_finish_delta": 28.166666666666686,
        "final_money_delta": 1879,
        "experts": {
            "early": {
                "bid_shading": "0.2",
                "future_cash_weight": "0.5",
                "liquidity_strength": "1.45",
                "objective_progress_weight": "1",
            },
            "middle": {
                "bid_shading": "0.3",
                "future_cash_weight": "0.5",
                "liquidity_strength": "1.45",
                "objective_progress_weight": "0.85",
            },
            "late": {
                "bid_shading": "0.15",
                "future_cash_weight": "0.7",
                "liquidity_strength": "1.3",
                "objective_progress_weight": "0.8",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "b9562f9b8a3748431bc9a3d458de7f9acc346884c2ac3ccf3116fdc4f2a3aa27"
            ),
            "winner-diagnostics.json": (
                "f1f3a6e3af692453308e86fee4e08b8f31a370ea4b37a529a8f58a717260f972"
            ),
            "winner-diagnostics.md": (
                "93d53667130eaf31e2355a04473b862ffeb2e5f1649fcfb2a9c9cf5687178472"
            ),
        },
    },
    EXPECTED_V4_IDENTITIES[1]: {
        "predecessor": "balanced-v3",
        "search_name": "balanced-v4-search-v2",
        "development_name": "development-balanced-v3-broad-v1",
        "development": "d556cc940c92ebf3633fde83485d4ba776b6e582f34cf8d445a5c190824b3228",
        "profile": "ae48ac912b3a84246be09e8f88f5e5c6a8d6dcf9668b030fc3cebfbdb376d32f",
        "manifest": "4c738ca18f93bd59112115a7c992d168fef6c26f0c114aed9011a80f7a8f2763",
        "report": "fc153c9fb8905dae4fcd0028b83b5a19fb1974d20f35b1c5cdd8617f7601420a",
        "evaluations": "7c57f855a08a4465facf28326dafc8954d25e378a6190c3688d43f9418935a95",
        "selection": "1d2a3e08c28d5192648f2d7d198ef6a9846be91a36dd6303be369081ea3dc992",
        "games": "a16e95ce7ea8b7898f79316191becc82f1a7aa7c918228af0ff67126cfc2fcdb",
        "freeze": "373b94c6112f6838fc4ec83488427890d9e5403e2414344a3cc6f736a9d60886",
        "rating_delta": 12.552936995797609,
        "normalized_finish_delta": 2.166666666666657,
        "final_money_delta": 100,
        "experts": {
            "early": {
                "bid_shading": "0.3",
                "future_cash_weight": "1.05",
                "liquidity_strength": "1.25",
                "objective_progress_weight": "0.55",
            },
            "middle": {
                "bid_shading": "0.3",
                "future_cash_weight": "1.05",
                "liquidity_strength": "1.5",
                "objective_progress_weight": "0.9",
            },
            "late": {
                "bid_shading": "0.3",
                "future_cash_weight": "1.05",
                "liquidity_strength": "1.4",
                "objective_progress_weight": "0.9",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "c4ac4fa036f1a606db8b4e0d2bea8fa6bcb01c91cb95e4097c265cb54bc80c96"
            ),
            "winner-diagnostics.json": (
                "c957f6208435572dafdb7a8a86f606d83fbdc24ce1a0b948938aa5b698a9d055"
            ),
            "winner-diagnostics.md": (
                "fd1dab2cb46154295334c9d534241dc4d58d66022203747c8a452c49dc41cdba"
            ),
        },
    },
    EXPECTED_V4_IDENTITIES[2]: {
        "predecessor": "passive-v3",
        "search_name": "passive-v4-search-v2",
        "development_name": "development-passive-v3-broad-v1",
        "development": "64124822038895aa4048469244c99177c877271368cc246d2250b737a14bf658",
        "profile": "fcf5cb322e5170da2df14a6b7e77be502d9e5456ba4e881fb5c303c76c581bca",
        "manifest": "11f7a8e38b89e9eea9b4f670632f9acd950a74ed48414b1a4e263ba27e90b048",
        "report": "ba01bae0b3c574b1687f57c55cfd3facc8ddb8d62db060e1cf81156b82894a9e",
        "evaluations": "a35a8627d7417d9af61e03706fcbc6c747a9fc25a630cd188b9770953e7ebcf2",
        "selection": "97e7004eb1d212d68435363d036fe034a23af3bb943c3ea5c43e3165c2dcbc60",
        "games": "49834d9e0b4a46505e25d4e2d16b2ed4576727ccac8d1ca1a37e5db02f1a3ae2",
        "freeze": "d39cc5b1c85dbfad9365b86374a30c096c3475580877d14dcbc397c17e43af95",
        "rating_delta": 71.59108456759373,
        "normalized_finish_delta": 15.583333333333343,
        "final_money_delta": 721,
        "experts": {
            "early": {
                "bid_shading": "0.1",
                "future_cash_weight": "1.75",
                "liquidity_strength": "1.4",
                "objective_progress_weight": "0.6",
            },
            "middle": {
                "bid_shading": "0.45",
                "future_cash_weight": "1.9",
                "liquidity_strength": "0.45",
                "objective_progress_weight": "0.1",
            },
            "late": {
                "bid_shading": "0.5",
                "future_cash_weight": "2",
                "liquidity_strength": "0.25",
                "objective_progress_weight": "0.1",
            },
        },
        "diagnostics": {
            "winner-decision-slices.csv": (
                "3abc8b43c82b2ef1cfe82b2a206a459d877d655a36bcd04dcac7801b3ffc00b1"
            ),
            "winner-diagnostics.json": (
                "b5e23fba5090e4d5ea70ce90722929915da62e3a0b2ca1c560ffab9737c02e22"
            ),
            "winner-diagnostics.md": (
                "2a2e546a248f7e44107c34397a993fde9a9f93bdc16e0bed6e7885d6daa24005"
            ),
        },
    },
}
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
            "digest": AGGRESSIVE_DEVELOPMENT_DIGEST,
            "name": "development-aggressive-v3-broad-v1",
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
        assert v3_candidate.development_corpus_name == (
            f"development-{v3_candidate.personality}-v3-broad-v1"
        )
        assert v3_candidate.development_corpus_digest == expected["development"]
        assert v3_candidate.worst_challenger_finish_delta is not None
        assert v3_candidate.worst_challenger_finish_delta > 0.0
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
        assert v4_candidate.development_corpus_name == expected["development_name"]
        assert v4_candidate.development_corpus_digest == expected["development"]
        assert v4_candidate.repository_commit == "b306d77de634efba21542b18589946a3fd8fc703"
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


def test_catalog_index_hashes_and_describes_packaged_freezes() -> None:
    index = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))

    assert set(index) == {"schema_version", "candidates"}
    assert index["schema_version"] == 1
    assert [entry["identity"] for entry in index["candidates"]] == list(EXPECTED_IDENTITIES)
    for entry in index["candidates"]:
        identity = entry["identity"]
        installed = CATALOG_DIR / entry["file"]
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == entry["sha256"]
        payload = json.loads(installed.read_text(encoding="utf-8"))
        assert payload["identity"] == entry["identity"]
        assert payload["personality"] == entry["personality"]
        assert payload["predecessor_name"] == entry["predecessor_name"]
        assert payload["repository_commit"] == entry["repository_commit"]
        assert payload["profile_digest"] == entry["profile_digest"]
        assert payload["search"]["name"] == entry["search_name"]
        assert payload["search"]["manifest_digest"] == entry["manifest_digest"]
        assert payload["development_corpus"]["name"] == entry["development_corpus_name"]
        assert payload["development_corpus"]["digest"] == entry["development_corpus_digest"]
        assert payload["source_evidence"]["search_report_sha256"] == entry["search_report_sha256"]
        assert (
            payload["source_evidence"]["candidate_evaluations_sha256"]
            == entry["candidate_evaluations_sha256"]
        )
        if identity in EXPECTED_V3_PROVENANCE:
            expected = EXPECTED_V3_PROVENANCE[identity]
            assert entry["sha256"] == expected["freeze"]
            continue

        expected = EXPECTED_V4_PROVENANCE[identity]
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
        assert entry["selection_log_sha256"] == expected["selection"]
        assert entry["development_games_sha256"] == expected["games"]
        assert (
            entry["winner_decision_slices_sha256"]
            == expected["diagnostics"]["winner-decision-slices.csv"]
        )
        assert (
            entry["winner_diagnostics_json_sha256"]
            == expected["diagnostics"]["winner-diagnostics.json"]
        )
        assert (
            entry["winner_diagnostics_markdown_sha256"]
            == expected["diagnostics"]["winner-diagnostics.md"]
        )


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


def test_v3_identity_accepts_current_schema_v2_payload(
    tmp_path: Path,
) -> None:
    index_path = _write_modified_catalog(
        tmp_path,
        index_mutation=lambda index: index.update({"candidates": index["candidates"][:1]}),
    )

    candidates = load_frozen_candidates(index_path)

    assert len(candidates) == 1
    assert candidates[0].identity == EXPECTED_V3_IDENTITIES[0]


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
    expected_matrix = set(product(tuple("ABCDE"), (3, 4, 5)))

    for candidate in FROZEN_CANDIDATES:
        corpus_name = (
            "development-heuristic-v4-v1.json"
            if isinstance(candidate, FrozenPhaseAwareCandidate)
            else f"development-{candidate.personality}-v3-broad-v1.json"
        )
        corpus = load_promotion_corpus(
            REPOSITORY_ROOT / "configs" / "promotion" / corpus_name,
            registry=BOT_SPECS_BY_NAME,
        )
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
