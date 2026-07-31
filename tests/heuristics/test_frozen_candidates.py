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
    "aggressive-v3-candidate-g007-s009-9c43f610b2f0",
    "balanced-v3-candidate-g005-s005-90544d0f26d2",
    "passive-v3-candidate-g006-s000-739e30e8d844",
)
EXPECTED_PROVENANCE = {
    EXPECTED_IDENTITIES[0]: {
        "predecessor": "aggressive-v2",
        "search_name": "aggressive-v3-search-v1",
        "profile": "9c43f610b2f03b2e5d3cb71c2a3ce7a62f456f3fdce73d3a52c501bab9a81042",
        "manifest": "b4e21379ebdcf9462330a1e55412993cf69273a257afc7bb9197cb2e7d89532f",
        "report": "db53ccbe258e2a08969d43710f7979d2e3b16f18415c4c11b81654a6301abea2",
        "evaluations": "47b82f973f36b3a21d7aa30d2872b542c23a55bfaf5666c715cc446fdb509139",
        "freeze": "89b61c17485da34b0c496cd353f0ccd1c533ba843175e967e3af939528bf5735",
        "development": "6531e85f69f4e085b8ac789348be6c21614455dd77ba7ac791f5390479e17638",
    },
    EXPECTED_IDENTITIES[1]: {
        "predecessor": "balanced-v2",
        "search_name": "balanced-v3-search-v1",
        "profile": "90544d0f26d20a9fc3b51013fd244908be1071a13eeef12fb0d4b2c4aef251de",
        "manifest": "9333619e4d573f12736162065f02901d25fa5e267ea6fd1838371de4b9f04c07",
        "report": "98056eda744f19d9f67dd763c64e5d5e2e97a3e0538534e933058d3a34227dfb",
        "evaluations": "4b8c1ea06575971914d183cfcba99964b50b37d60c740b52fe417b8f80a85eb7",
        "freeze": "2da90c344b4f573f83718faad717a9accea2768b21f1982d8f543a5f2bb25fa8",
        "development": "d556cc940c92ebf3633fde83485d4ba776b6e582f34cf8d445a5c190824b3228",
    },
    EXPECTED_IDENTITIES[2]: {
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


def test_catalog_exposes_exact_frozen_records_and_provenance() -> None:
    assert tuple(item.identity for item in FROZEN_CANDIDATES) == EXPECTED_IDENTITIES
    assert tuple(FROZEN_CANDIDATES_BY_NAME) == EXPECTED_IDENTITIES

    for candidate in FROZEN_CANDIDATES:
        expected = EXPECTED_PROVENANCE[candidate.identity]
        assert candidate.bot_spec.name == candidate.identity
        assert candidate.bot_spec.bot_id == candidate.identity
        assert candidate.predecessor_name == expected["predecessor"]
        assert candidate.search_name == expected["search_name"]
        assert candidate.development_corpus_name == (
            f"development-{candidate.personality}-v3-broad-v1"
        )
        assert candidate.development_corpus_digest == expected["development"]
        assert candidate.worst_challenger_finish_delta is not None
        assert candidate.worst_challenger_finish_delta > 0.0
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
    expected_matrix = set(product(tuple("ABCDE"), (3, 4, 5)))

    for candidate in FROZEN_CANDIDATES:
        corpus = load_promotion_corpus(
            REPOSITORY_ROOT
            / "configs"
            / "promotion"
            / f"development-{candidate.personality}-v3-broad-v1.json",
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
