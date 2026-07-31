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
from garboid_pocketrocks.evolution.planning import plan_development_games
from garboid_pocketrocks.heuristics.frozen import (
    FROZEN_CANDIDATES,
    FROZEN_CANDIDATES_BY_NAME,
    FrozenCandidateCatalogError,
    load_frozen_candidates,
)
from garboid_pocketrocks.promotion.corpus import load_promotion_corpus
from garboid_pocketrocks.simulator.monte_carlo import MonteCarloRunner

REPOSITORY_ROOT = Path(__file__).parents[2]
CATALOG_DIR = REPOSITORY_ROOT / "src" / "garboid_pocketrocks" / "heuristics" / "frozen_candidates"
EXPECTED_IDENTITIES = (
    "aggressive-v3-candidate-g007-s008-c70e11540db9",
    "balanced-v3-candidate-g006-s010-e3971899626c",
    "passive-v3-candidate-g006-s001-812832214cd5",
)
EXPECTED_PROVENANCE = {
    EXPECTED_IDENTITIES[0]: {
        "predecessor": "aggressive-v2",
        "search_name": "aggressive-v3-search-v1",
        "profile": "c70e11540db92d0c77ce5085670ff48105c91aede1ed52c4abb7874a64687b58",
        "manifest": "627eb77836f8dceace745a8fb7f60573e2dad05aa47a423d902850c32a98f5e0",
        "report": "01ca66301d633be7228c3bc535fa2d84b0c5ee3898b92f9d06e98c0fdf13b902",
        "evaluations": "4140270b3fe1d744aef103b012ca85970aa561c3f36cb28d70e4e4aa39f9c7a5",
        "freeze": "218e9682d8d174125d4b9e7550fec9afda01ddb4433084143968b6d525d335da",
    },
    EXPECTED_IDENTITIES[1]: {
        "predecessor": "balanced-v2",
        "search_name": "balanced-v3-search-v1",
        "profile": "e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e",
        "manifest": "da9e2162eec9dd934dc80e59d9950b49c74a3a4cd4d72e6273134b502e705152",
        "report": "95fd24f688ed2bb18cd08d00483fbef2a42b2b66809afa26860f71deba2d3f87",
        "evaluations": "fcf8985f40beddac274f5aa31523ec93b1cfecf0657047e30144ba97140b15e6",
        "freeze": "05bcd898e7fc79062585cb989b67cb2e5641eed6cc59b1a60255de84c8ee2988",
    },
    EXPECTED_IDENTITIES[2]: {
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


def test_catalog_exposes_exact_frozen_records_and_provenance() -> None:
    assert tuple(item.identity for item in FROZEN_CANDIDATES) == EXPECTED_IDENTITIES
    assert tuple(FROZEN_CANDIDATES_BY_NAME) == EXPECTED_IDENTITIES

    for candidate in FROZEN_CANDIDATES:
        expected = EXPECTED_PROVENANCE[candidate.identity]
        assert candidate.bot_spec.name == candidate.identity
        assert candidate.bot_spec.bot_id == candidate.identity
        assert candidate.predecessor_name == expected["predecessor"]
        assert candidate.search_name == expected["search_name"]
        assert candidate.development_corpus_name == "development-v1"
        assert candidate.development_corpus_digest == DEVELOPMENT_DIGEST
        assert candidate.freeze_digest == expected["freeze"]
        assert candidate.profile_digest == expected["profile"]
        assert candidate.manifest_digest == expected["manifest"]
        assert candidate.search_report_digest == expected["report"]
        assert candidate.candidate_evaluations_digest == expected["evaluations"]


def test_catalog_index_hashes_and_describes_packaged_freezes() -> None:
    index = json.loads((CATALOG_DIR / "index.json").read_text(encoding="utf-8"))

    assert set(index) == {"schema_version", "candidates"}
    assert index["schema_version"] == 1
    assert [entry["identity"] for entry in index["candidates"]] == list(EXPECTED_IDENTITIES)
    for entry in index["candidates"]:
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
        assert isinstance(brain, HeuristicBotBrain)
        assert brain.valuator.profile == candidate.profile


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
        REPOSITORY_ROOT / "configs" / "promotion" / "historical" / "development-v1-17c01635.json",
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
