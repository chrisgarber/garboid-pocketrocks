from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.bots.heuristic import AggressiveHeuristicV2Brain
from garboid_pocketrocks.bots.registry import BOT_SPECS, BOT_SPECS_BY_NAME
from garboid_pocketrocks.hybrid import experts as expert_module
from garboid_pocketrocks.hybrid.experts import (
    PromotedExpertCatalogError,
    check_expert_availability,
    load_promoted_experts,
)

_CATALOG_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "garboid_pocketrocks"
    / "hybrid"
    / "promoted_experts-v1.json"
)


def _payload() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_CATALOG_PATH.read_text(encoding="utf-8")))


def _entry(payload: dict[str, object], index: int = 0) -> dict[str, object]:
    entries = cast(list[object], payload["experts"])
    return cast(dict[str, object], entries[index])


def _promotion(entry: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], entry["promotion"])


def _refresh_evidence_digest(entry: dict[str, object]) -> None:
    raw = json.dumps(
        {
            "candidate_identity": entry["promoted_candidate_identity"],
            "incumbent_name": entry["incumbent_name"],
            "promotion": entry["promotion"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    entry["promotion_evidence_digest"] = hashlib.sha256(raw).hexdigest()


def _validate(payload: dict[str, object]) -> None:
    expert_module._validate_catalog_payload(payload, bot_specs=BOT_SPECS_BY_NAME)


def test_initial_roster_binds_promotions_and_executables() -> None:
    experts = load_promoted_experts()

    assert tuple(expert.name for expert in experts) == (
        "aggressive-v3",
        "balanced-v3",
        "passive-v3",
        "vector_ppo_large_v1_g350k",
    )
    assert tuple(expert.bot_spec for expert in experts) == tuple(
        BOT_SPECS_BY_NAME[expert.name] for expert in experts
    )
    assert all(expert.confidence_interval_lower > 0.0 for expert in experts)
    assert all(expert.development_corpus_name == "development-v1" for expert in experts)
    assert all(expert.held_out_corpus_name == "held-out-v1" for expert in experts)
    assert tuple(expert.executable_kind for expert in experts) == (
        "heuristic_profile",
        "heuristic_profile",
        "heuristic_profile",
        "neural_checkpoint",
    )
    assert all("hybrid" not in spec.name for spec in BOT_SPECS)


def test_unreviewed_catalog_edit_is_rejected_before_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload()
    _promotion(_entry(payload))["promoted"] = False
    edited_path = tmp_path / "promoted_experts-v1.json"
    edited_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(expert_module, "_CATALOG_PATH", edited_path)

    with pytest.raises(PromotedExpertCatalogError) as raised:
        load_promoted_experts()

    assert raised.value.code == "catalog_digest_mismatch"


@pytest.mark.parametrize(
    ("field_path", "new_value", "expected_code"),
    [
        (("promoted",), False, "promotion_failed"),
        (("confidence_interval_95", "lower"), 0.0, "nonpositive_promotion_interval"),
        (("bootstrap", "converged"), 999, "incomplete_bootstrap"),
        (("coverage", "completed_games"), 959, "incomplete_promotion_coverage"),
        (("faults",), 1, "promotion_faults"),
        (("failures",), ["worker_failed"], "promotion_failures"),
        (("warnings",), ["partial result"], "promotion_warnings"),
        (("rule", "charts"), ["A", "B", "C", "D"], "promotion_rule_mismatch"),
        (
            ("corpora", "held_out", "digest"),
            "f" * 64,
            "promotion_corpus_mismatch",
        ),
    ],
)
def test_failed_or_incomplete_promotion_receipts_are_ineligible(
    field_path: tuple[str, ...], new_value: object, expected_code: str
) -> None:
    payload = _payload()
    entry = _entry(payload)
    target = _promotion(entry)
    for field in field_path[:-1]:
        target = cast(dict[str, object], target[field])
    target[field_path[-1]] = new_value
    _refresh_evidence_digest(entry)

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == expected_code


def test_receipt_edit_without_matching_digest_is_rejected() -> None:
    payload = _payload()
    _promotion(_entry(payload))["rating_difference"] = 999.0

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == "promotion_evidence_digest_mismatch"


def test_promotion_evidence_digest_binds_candidate_and_incumbent() -> None:
    payload = _payload()
    entry = _entry(payload)
    entry["incumbent_name"] = "random"

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == "promotion_incumbent_mismatch"


@pytest.mark.parametrize("replacement", ["aggressive", "latest", "aggressive-v4"])
def test_dynamic_latest_or_unapproved_roster_entries_are_rejected(replacement: str) -> None:
    payload = _payload()
    _entry(payload)["expert_name"] = replacement

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == "unexpected_expert_roster"


def test_duplicate_roster_entry_is_rejected() -> None:
    payload = _payload()
    _entry(payload, 1)["expert_name"] = "aggressive-v3"

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == "unexpected_expert_roster"


def test_missing_registered_botspec_is_rejected() -> None:
    payload = _payload()
    bot_specs = dict(BOT_SPECS_BY_NAME)
    del bot_specs["aggressive-v3"]

    with pytest.raises(PromotedExpertCatalogError) as raised:
        expert_module._validate_catalog_payload(payload, bot_specs=bot_specs)

    assert raised.value.code == "unregistered_expert"


def test_released_v3_alias_must_run_the_promoted_frozen_profile() -> None:
    payload = _payload()
    bot_specs = dict(BOT_SPECS_BY_NAME)
    bot_specs["aggressive-v3"] = BotSpec.for_simulation("aggressive-v3", AggressiveHeuristicV2Brain)

    with pytest.raises(PromotedExpertCatalogError) as raised:
        expert_module._validate_catalog_payload(payload, bot_specs=bot_specs)

    assert raised.value.code == "released_botspec_mismatch"


def test_edited_checkpoint_digest_is_rejected() -> None:
    payload = _payload()
    executable = cast(dict[str, object], _entry(payload, 3)["executable"])
    executable["digest"] = "f" * 64

    with pytest.raises(PromotedExpertCatalogError) as raised:
        _validate(payload)

    assert raised.value.code == "promoted_checkpoint_digest_mismatch"


def test_runtime_availability_is_separate_from_eligibility() -> None:
    expert = load_promoted_experts()[0]

    def missing_dependency(seed: int | None) -> Any:
        del seed
        raise ModuleNotFoundError("optional runtime is not installed")

    unavailable = replace(
        expert,
        bot_spec=BotSpec.for_simulation(expert.name, missing_dependency),
    )

    assert check_expert_availability(expert).available is True
    diagnostic = check_expert_availability(unavailable)
    assert diagnostic.available is False
    assert diagnostic.reason == "runtime_dependency_missing"
    assert "optional runtime" in cast(str, diagnostic.detail)
