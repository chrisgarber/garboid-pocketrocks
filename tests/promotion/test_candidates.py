from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.bots.random_bot import RandomBot
from garboid_pocketrocks.heuristics.frozen import (
    FROZEN_CANDIDATES_BY_NAME,
    FrozenPhaseAwareCandidate,
)
from garboid_pocketrocks.promotion.candidates import (
    FrozenCandidateProvenance,
    FrozenPhaseAwareCandidateProvenance,
    PromotionCandidateError,
    ResolvedPromotionCandidate,
    resolve_promotion_candidate,
    validate_promotion_candidate,
)
from garboid_pocketrocks.promotion.corpus import (
    PromotionCorpus,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)

from .helpers import EvilFactory, EvilString, evil_provenance
from .test_runner import _corpora

_DIGESTS = tuple(character * 64 for character in "abcde")
_BALANCED_V4_IDENTITY = "balanced-v4-candidate-g005-s010-ae48ac912b3a"
_V4_IDENTITIES = (
    "aggressive-v4-candidate-g011-s014-9a2908cce71c",
    _BALANCED_V4_IDENTITY,
    "passive-v4-candidate-g011-s012-fcf5cb322e51",
)
_PHASES = ("early", "middle", "late")
_COEFFICIENTS = (
    "liquidity_strength",
    "future_cash_weight",
    "objective_progress_weight",
    "bid_shading",
)
_DIAGNOSTIC_NAMES = (
    "winner-decision-slices.csv",
    "winner-diagnostics.json",
    "winner-diagnostics.md",
)


def _development_for_frozen(frozen: Any) -> PromotionCorpus:
    corpus_file = f"development-{frozen.personality}-v3-broad-v1.json"
    return load_promotion_corpus(
        Path("configs/promotion") / corpus_file,
        registry=BOT_SPECS_BY_NAME,
    )


@dataclass(frozen=True, slots=True)
class _FrozenCandidateFixture:
    identity: str
    bot_spec: BotSpec
    predecessor_name: str
    development_corpus_name: str
    development_corpus_digest: str
    search_name: str
    repository_commit: str
    freeze_digest: str
    profile_digest: str
    manifest_digest: str
    search_report_digest: str
    candidate_evaluations_digest: str


@dataclass(frozen=True, slots=True)
class _PhaseAwareProvenanceSubclass(FrozenPhaseAwareCandidateProvenance):
    pass


def _frozen_candidate() -> _FrozenCandidateFixture:
    development, _ = _corpora(pair_count=1)
    identity = "balanced-v3-candidate-test"
    return _FrozenCandidateFixture(
        identity=identity,
        bot_spec=BotSpec.for_simulation(
            identity,
            RandomBot.build_brain,
        ),
        predecessor_name="balanced-v2",
        development_corpus_name=development.recipe.name,
        development_corpus_digest=development.digest,
        search_name="balanced-v3-search-v1",
        repository_commit="1" * 40,
        freeze_digest=_DIGESTS[0],
        profile_digest=_DIGESTS[1],
        manifest_digest=_DIGESTS[2],
        search_report_digest=_DIGESTS[3],
        candidate_evaluations_digest=_DIGESTS[4],
    )


def test_registered_candidate_resolution_preserves_the_existing_path() -> None:
    registered = BotSpec.for_simulation("registered", RandomBot.build_brain)
    frozen = replace(_frozen_candidate(), bot_spec=registered)

    resolved = resolve_promotion_candidate(
        "registered",
        registry={"registered": registered},
        frozen_candidates={"registered": frozen},
    )

    assert resolved.bot_spec is registered
    assert resolved.frozen_provenance is None


def test_frozen_candidate_resolution_records_all_bound_provenance() -> None:
    frozen = _frozen_candidate()

    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )

    assert resolved.bot_spec is frozen.bot_spec
    assert resolved.frozen_provenance == FrozenCandidateProvenance(
        candidate_name=frozen.bot_spec.name,
        candidate_bot_id=frozen.bot_spec.bot_id,
        predecessor_name="balanced-v2",
        development_corpus_name=frozen.development_corpus_name,
        development_corpus_digest=frozen.development_corpus_digest,
        search_name=frozen.search_name,
        repository_commit=frozen.repository_commit,
        freeze_digest=_DIGESTS[0],
        profile_digest=_DIGESTS[1],
        manifest_digest=_DIGESTS[2],
        search_report_digest=_DIGESTS[3],
        candidate_evaluations_digest=_DIGESTS[4],
    )


def test_phase_aware_frozen_candidate_resolution_records_exact_schema_v2_provenance() -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[_BALANCED_V4_IDENTITY]
    assert type(frozen) is FrozenPhaseAwareCandidate

    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )

    assert resolved.bot_spec is frozen.bot_spec
    assert resolved.frozen_provenance == FrozenPhaseAwareCandidateProvenance(
        candidate_name=_BALANCED_V4_IDENTITY,
        candidate_bot_id=_BALANCED_V4_IDENTITY,
        predecessor_name="balanced-v3",
        development_corpus_name="development-balanced-v3-broad-v1",
        development_corpus_digest=(
            "d556cc940c92ebf3633fde83485d4ba776b6e582f34cf8d445a5c190824b3228"
        ),
        search_name="balanced-v4-search-v2",
        repository_commit="b306d77de634efba21542b18589946a3fd8fc703",
        freeze_digest="373b94c6112f6838fc4ec83488427890d9e5403e2414344a3cc6f736a9d60886",
        profile_digest="ae48ac912b3a84246be09e8f88f5e5c6a8d6dcf9668b030fc3cebfbdb376d32f",
        manifest_digest="4c738ca18f93bd59112115a7c992d168fef6c26f0c114aed9011a80f7a8f2763",
        search_report_digest=("fc153c9fb8905dae4fcd0028b83b5a19fb1974d20f35b1c5cdd8617f7601420a"),
        candidate_evaluations_digest=(
            "7c57f855a08a4465facf28326dafc8954d25e378a6190c3688d43f9418935a95"
        ),
        freeze_schema_version=2,
        personality="balanced",
        phase_selector_rules=(
            ("kind", "public-resource-horizon-v1"),
            ("early", "3*future>=2*total"),
            ("middle", "3*future>=total"),
            ("late", "otherwise"),
        ),
        expert_profiles=(
            (
                "early",
                (
                    ("liquidity_strength", 1.25),
                    ("future_cash_weight", 1.05),
                    ("objective_progress_weight", 0.55),
                    ("bid_shading", 0.3),
                ),
            ),
            (
                "middle",
                (
                    ("liquidity_strength", 1.5),
                    ("future_cash_weight", 1.05),
                    ("objective_progress_weight", 0.9),
                    ("bid_shading", 0.3),
                ),
            ),
            (
                "late",
                (
                    ("liquidity_strength", 1.4),
                    ("future_cash_weight", 1.05),
                    ("objective_progress_weight", 0.9),
                    ("bid_shading", 0.3),
                ),
            ),
        ),
        expert_digests=(
            (
                "early",
                "431e66c676a0cb6cf5d0ccfa576c4cab7b46e7e9aa57ec98d254fd5dc3849f18",
            ),
            (
                "middle",
                "0ba361b8f4e45479586f5c50c310239e709e60abbbf1f412b51ea57bf24494b4",
            ),
            (
                "late",
                "67fda27f11e8a1ea59f14c0e330a187a6c6ebe85d6b501d906e5a21030fe96c2",
            ),
        ),
        boundary_report_path="docs/benchmarks/2026-07-30-heuristic-v4-phase-boundaries.md",
        boundary_report_digest=("9961f26f32270dcebc98df443588e96cbde2f953858cd131c66a37aeecaa9b01"),
        boundary_slices_path=(
            "docs/benchmarks/tournaments/"
            "2026-07-30-heuristic-v3-phase-boundaries-development/"
            "phase-boundary-slices.csv"
        ),
        boundary_slices_digest=("4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae"),
        selection_log_digest=("1d2a3e08c28d5192648f2d7d198ef6a9846be91a36dd6303be369081ea3dc992"),
        development_games_digest=(
            "a16e95ce7ea8b7898f79316191becc82f1a7aa7c918228af0ff67126cfc2fcdb"
        ),
        winner_diagnostics_digests=(
            (
                "winner-decision-slices.csv",
                "c4ac4fa036f1a606db8b4e0d2bea8fa6bcb01c91cb95e4097c265cb54bc80c96",
            ),
            (
                "winner-diagnostics.json",
                "c957f6208435572dafdb7a8a86f606d83fbdc24ce1a0b948938aa5b698a9d055",
            ),
            (
                "winner-diagnostics.md",
                "fd1dab2cb46154295334c9d534241dc4d58d66022203747c8a452c49dc41cdba",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("identity", "personality", "predecessor_name"),
    (
        (_V4_IDENTITIES[0], "aggressive", "aggressive-v3"),
        (_V4_IDENTITIES[1], "balanced", "balanced-v3"),
        (_V4_IDENTITIES[2], "passive", "passive-v3"),
    ),
)
def test_each_real_v4_candidate_resolves_and_validates_against_its_exact_v3_predecessor(
    identity: str,
    personality: str,
    predecessor_name: str,
) -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[identity]
    assert type(frozen) is FrozenPhaseAwareCandidate
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.bot_spec is frozen.bot_spec
    assert type(resolved.frozen_provenance) is FrozenPhaseAwareCandidateProvenance
    assert resolved.frozen_provenance.freeze_schema_version == 2
    assert resolved.frozen_provenance.personality == personality
    assert resolved.frozen_provenance.predecessor_name == predecessor_name
    assert resolved.frozen_provenance.search_name == f"{personality}-v4-search-v2"
    development = _development_for_frozen(frozen)

    validate_promotion_candidate(
        resolved,
        incumbent=BOT_SPECS_BY_NAME[predecessor_name],
        development=development,
        registry=BOT_SPECS_BY_NAME,
    )


def _replace_first_expert_coefficient_with_integer(
    provenance: FrozenPhaseAwareCandidateProvenance,
) -> FrozenPhaseAwareCandidateProvenance:
    phase, coefficients = provenance.expert_profiles[0]
    coefficient_name, _ = coefficients[0]
    return replace(
        provenance,
        expert_profiles=(
            (phase, ((coefficient_name, 0), *coefficients[1:])),
            *provenance.expert_profiles[1:],
        ),
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (
            lambda provenance: replace(provenance, freeze_schema_version=True),
            "schema version",
        ),
        (
            lambda provenance: replace(provenance, personality="latest"),
            "personality",
        ),
        (
            lambda provenance: replace(
                provenance,
                phase_selector_rules=provenance.phase_selector_rules[:-1],
            ),
            "phase selector",
        ),
        (
            lambda provenance: replace(
                provenance,
                phase_selector_rules=tuple(reversed(provenance.phase_selector_rules)),
            ),
            "phase selector",
        ),
        (_replace_first_expert_coefficient_with_integer, "expert profiles"),
        (
            lambda provenance: replace(
                provenance,
                expert_digests=provenance.expert_digests[:-1],
            ),
            "expert digests",
        ),
        (
            lambda provenance: replace(provenance, boundary_report_path=""),
            "boundary evidence",
        ),
        (
            lambda provenance: replace(provenance, boundary_slices_digest=True),
            "evidence digests",
        ),
        (
            lambda provenance: replace(provenance, selection_log_digest="not-a-digest"),
            "evidence digests",
        ),
        (
            lambda provenance: replace(
                provenance,
                winner_diagnostics_digests=provenance.winner_diagnostics_digests[:-1],
            ),
            "winner diagnostics",
        ),
    ),
)
def test_nested_phase_aware_provenance_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
    message: str,
) -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[_BALANCED_V4_IDENTITY]
    assert type(frozen) is FrozenPhaseAwareCandidate
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert type(resolved.frozen_provenance) is FrozenPhaseAwareCandidateProvenance
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: FROZEN_CANDIDATES_BY_NAME,
    )
    development = _development_for_frozen(frozen)
    tampered = replace(
        resolved,
        frozen_provenance=tamper(resolved.frozen_provenance),
    )

    with pytest.raises(PromotionCandidateError, match=message):
        validate_promotion_candidate(
            tampered,
            incumbent=BOT_SPECS_BY_NAME["balanced-v3"],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


def _replace_named_pair(
    pairs: tuple[tuple[str, str], ...],
    *,
    name: str,
    value: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (pair_name, value if pair_name == name else pair_value) for pair_name, pair_value in pairs
    )


def _shape_valid_phase_provenance_tamper(
    provenance: FrozenPhaseAwareCandidateProvenance,
    binding: str,
) -> FrozenPhaseAwareCandidateProvenance:
    if binding == "personality":
        return replace(provenance, personality="aggressive")
    if binding == "phase_selector":
        return replace(
            provenance,
            phase_selector_rules=_replace_named_pair(
                provenance.phase_selector_rules,
                name="middle",
                value="changed-public-rule",
            ),
        )
    if binding.startswith("expert_profile:"):
        _, target_phase, target_coefficient = binding.split(":")
        changed_profiles = []
        for phase, coefficients in provenance.expert_profiles:
            changed_profiles.append(
                (
                    phase,
                    tuple(
                        (
                            coefficient_name,
                            coefficient_value + 0.05
                            if phase == target_phase and coefficient_name == target_coefficient
                            else coefficient_value,
                        )
                        for coefficient_name, coefficient_value in coefficients
                    ),
                )
            )
        return replace(provenance, expert_profiles=tuple(changed_profiles))
    if binding.startswith("expert_digest:"):
        _, phase = binding.split(":")
        return replace(
            provenance,
            expert_digests=_replace_named_pair(
                provenance.expert_digests,
                name=phase,
                value="0" * 64,
            ),
        )
    if binding == "boundary_report_path":
        return replace(provenance, boundary_report_path="docs/benchmarks/changed.md")
    if binding == "boundary_slices_path":
        return replace(provenance, boundary_slices_path="docs/benchmarks/changed.csv")
    if binding == "boundary_report_digest":
        return replace(provenance, boundary_report_digest="0" * 64)
    if binding == "boundary_slices_digest":
        return replace(provenance, boundary_slices_digest="0" * 64)
    if binding == "selection_log_digest":
        return replace(provenance, selection_log_digest="0" * 64)
    if binding == "development_games_digest":
        return replace(provenance, development_games_digest="0" * 64)
    if binding.startswith("winner_diagnostic:"):
        _, artifact_name = binding.split(":", maxsplit=1)
        return replace(
            provenance,
            winner_diagnostics_digests=_replace_named_pair(
                provenance.winner_diagnostics_digests,
                name=artifact_name,
                value="0" * 64,
            ),
        )
    raise AssertionError(f"unknown phase-aware provenance binding {binding}")


@pytest.mark.parametrize(
    "binding",
    (
        "personality",
        "phase_selector",
        *(
            f"expert_profile:{phase}:{coefficient}"
            for phase in _PHASES
            for coefficient in _COEFFICIENTS
        ),
        *(f"expert_digest:{phase}" for phase in _PHASES),
        "boundary_report_path",
        "boundary_report_digest",
        "boundary_slices_path",
        "boundary_slices_digest",
        "selection_log_digest",
        "development_games_digest",
        *(f"winner_diagnostic:{name}" for name in _DIAGNOSTIC_NAMES),
    ),
)
def test_every_shape_valid_phase_aware_binding_must_match_the_canonical_catalog(
    binding: str,
) -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[_BALANCED_V4_IDENTITY]
    assert type(frozen) is FrozenPhaseAwareCandidate
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    provenance = resolved.frozen_provenance
    assert type(provenance) is FrozenPhaseAwareCandidateProvenance
    tampered = replace(
        resolved,
        frozen_provenance=_shape_valid_phase_provenance_tamper(provenance, binding),
    )
    development = _development_for_frozen(frozen)

    with pytest.raises(PromotionCandidateError, match="trusted frozen candidate record"):
        validate_promotion_candidate(
            tampered,
            incumbent=BOT_SPECS_BY_NAME["balanced-v3"],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


@pytest.mark.parametrize("kind", ("subclass", "lookalike"))
def test_phase_aware_provenance_requires_the_exact_public_type(kind: str) -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[_BALANCED_V4_IDENTITY]
    assert type(frozen) is FrozenPhaseAwareCandidate
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    provenance = resolved.frozen_provenance
    assert type(provenance) is FrozenPhaseAwareCandidateProvenance
    values = {
        field.name: getattr(provenance, field.name)
        for field in fields(FrozenPhaseAwareCandidateProvenance)
    }
    forged = (
        _PhaseAwareProvenanceSubclass(**values)
        if kind == "subclass"
        else cast(FrozenCandidateProvenance, SimpleNamespace(**values))
    )
    development = _development_for_frozen(frozen)

    with pytest.raises(PromotionCandidateError, match="exact provenance type"):
        validate_promotion_candidate(
            replace(resolved, frozen_provenance=forged),
            incumbent=BOT_SPECS_BY_NAME["balanced-v3"],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


def test_alternate_phase_aware_catalog_record_cannot_impersonate_the_canonical_record() -> None:
    frozen = FROZEN_CANDIDATES_BY_NAME[_BALANCED_V4_IDENTITY]
    assert type(frozen) is FrozenPhaseAwareCandidate
    alternate = replace(
        frozen,
        generation=0,
        slot=15,
        parent_identity=None,
    )
    resolved = resolve_promotion_candidate(
        frozen.identity,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates={frozen.identity: alternate},
    )
    development = _development_for_frozen(frozen)

    with pytest.raises(PromotionCandidateError, match="trusted frozen candidate record"):
        validate_promotion_candidate(
            resolved,
            incumbent=BOT_SPECS_BY_NAME["balanced-v3"],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


@pytest.mark.parametrize(
    "name",
    (
        "not-a-candidate",
        "../../frozen-candidate.json",
        "balanced-latest",
    ),
)
def test_unknown_paths_and_aliases_are_not_candidate_sources(name: str) -> None:
    with pytest.raises(PromotionCandidateError, match="unknown bot name"):
        resolve_promotion_candidate(
            name,
            registry={},
            frozen_candidates={},
        )


@pytest.mark.parametrize(
    "changed_field",
    (
        "catalog_alias",
        "bot_id",
        "freeze_digest",
    ),
)
def test_frozen_catalog_identity_and_digest_tampering_is_rejected(
    changed_field: str,
) -> None:
    frozen = _frozen_candidate()
    requested_name = frozen.bot_spec.name
    catalog_name = requested_name
    if changed_field == "catalog_alias":
        catalog_name = "balanced-v3-candidate-alias"
        requested_name = catalog_name
    elif changed_field == "bot_id":
        frozen = replace(
            frozen,
            bot_spec=replace(frozen.bot_spec, bot_id="different-id"),
        )
    else:
        frozen = replace(frozen, freeze_digest="not-a-digest")

    with pytest.raises(PromotionCandidateError):
        resolve_promotion_candidate(
            requested_name,
            registry={},
            frozen_candidates={catalog_name: frozen},
        )


@pytest.mark.parametrize(
    ("changed_binding", "message"),
    (
        ("predecessor", "predecessor"),
        ("development", "development corpus"),
    ),
)
def test_frozen_candidate_must_match_invoked_predecessor_and_development_corpus(
    monkeypatch: pytest.MonkeyPatch,
    changed_binding: str,
    message: str,
) -> None:
    frozen = _frozen_candidate()
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: {frozen.bot_spec.name: frozen},
    )
    development, _ = _corpora(pair_count=1)
    incumbent = BOT_SPECS_BY_NAME["balanced-v2"]
    if changed_binding == "predecessor":
        incumbent = BOT_SPECS_BY_NAME["aggressive-v2"]
    else:
        development = replace(development, digest="f" * 64)

    with pytest.raises(PromotionCandidateError, match=message):
        validate_promotion_candidate(
            resolved,
            incumbent=incumbent,
            development=development,
            registry={},
        )


@pytest.mark.parametrize("tampering", ("candidate", "provenance"))
def test_frozen_candidate_must_match_the_exact_trusted_catalog_record(
    monkeypatch: pytest.MonkeyPatch,
    tampering: str,
) -> None:
    frozen = _frozen_candidate()
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: {frozen.bot_spec.name: frozen},
    )
    if tampering == "candidate":
        resolved = replace(
            resolved,
            bot_spec=BotSpec.for_simulation(
                frozen.bot_spec.name,
                lambda seed: RandomBot.build_brain(seed),
            ),
        )
    else:
        assert resolved.frozen_provenance is not None
        resolved = replace(
            resolved,
            frozen_provenance=replace(
                resolved.frozen_provenance,
                freeze_digest="f" * 64,
            ),
        )
    development, _ = _corpora(pair_count=1)

    with pytest.raises(
        PromotionCandidateError,
        match="trusted frozen candidate|identity",
    ):
        validate_promotion_candidate(
            resolved,
            incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
            development=development,
            registry={},
        )


def test_registered_candidate_rejects_forged_frozen_provenance() -> None:
    registered = BOT_SPECS_BY_NAME["balanced-v2"]
    frozen = _frozen_candidate()
    frozen_resolution = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )
    assert frozen_resolution.frozen_provenance is not None
    forged = replace(
        frozen_resolution.frozen_provenance,
        candidate_name=registered.name,
        candidate_bot_id=registered.bot_id,
    )
    development, _ = _corpora(pair_count=1)
    catalog_loads = 0

    def load_catalog() -> dict[str, _FrozenCandidateFixture]:
        nonlocal catalog_loads
        catalog_loads += 1
        return {frozen.bot_spec.name: frozen}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
            load_catalog,
        )

        with pytest.raises(PromotionCandidateError):
            validate_promotion_candidate(
                ResolvedPromotionCandidate(registered, forged),
                incumbent=BotSpec.for_simulation("incumbent", RandomBot.build_brain),
                development=development,
                registry={registered.name: registered},
            )

    assert catalog_loads == 1


def test_frozen_catalog_candidate_requires_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _frozen_candidate()
    development, _ = _corpora(pair_count=1)
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: {frozen.bot_spec.name: frozen},
    )

    with pytest.raises(PromotionCandidateError, match="provenance"):
        validate_promotion_candidate(
            ResolvedPromotionCandidate(frozen.bot_spec, None),
            incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
            development=development,
            registry={},
        )


def test_non_catalog_test_candidate_does_not_require_frozen_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = BotSpec.for_simulation("test-only-candidate", RandomBot.build_brain)
    development, _ = _corpora(pair_count=1)
    catalog_loads = 0

    def load_empty_catalog() -> dict[str, _FrozenCandidateFixture]:
        nonlocal catalog_loads
        catalog_loads += 1
        return {}

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        load_empty_catalog,
    )

    validate_promotion_candidate(
        ResolvedPromotionCandidate(candidate, None),
        incumbent=BotSpec.for_simulation("incumbent", RandomBot.build_brain),
        development=development,
        registry={candidate.name: candidate},
    )

    assert catalog_loads == 1


@pytest.mark.parametrize(
    "field_name",
    tuple(field.name for field in fields(FrozenCandidateProvenance)),
)
def test_every_frozen_provenance_field_must_match_the_trusted_catalog(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    frozen = _frozen_candidate()
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: {frozen.bot_spec.name: frozen},
    )
    assert resolved.frozen_provenance is not None
    original_value = getattr(resolved.frozen_provenance, field_name)
    changed_value = (
        "f" * len(original_value)
        if set(original_value) <= set("0123456789abcdef")
        else f"changed-{original_value}"
    )
    tampered = replace(
        resolved,
        frozen_provenance=replace(
            resolved.frozen_provenance,
            **{field_name: changed_value},
        ),
    )
    development, _ = _corpora(pair_count=1)

    with pytest.raises(
        PromotionCandidateError,
        match="trusted frozen candidate|identity",
    ):
        validate_promotion_candidate(
            tampered,
            incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
            development=development,
            registry={},
        )


@pytest.mark.parametrize(
    "corpus_tampering",
    ("purpose", "name", "stored_digest", "content"),
)
def test_frozen_candidate_requires_exact_recomputed_development_corpus(
    monkeypatch: pytest.MonkeyPatch,
    corpus_tampering: str,
) -> None:
    development, _ = _corpora(pair_count=1)
    development = replace(
        development,
        digest=recompute_promotion_corpus_digest(development),
    )
    frozen = replace(
        _frozen_candidate(),
        development_corpus_name=development.recipe.name,
        development_corpus_digest=development.digest,
    )
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry={},
        frozen_candidates={frozen.bot_spec.name: frozen},
    )
    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        lambda: {frozen.bot_spec.name: frozen},
    )
    if corpus_tampering == "purpose":
        development = replace(
            development,
            recipe=replace(development.recipe, purpose="held_out"),
        )
        development = replace(
            development,
            digest=recompute_promotion_corpus_digest(development),
        )
    elif corpus_tampering == "name":
        development = replace(
            development,
            recipe=replace(development.recipe, name="other-development"),
        )
        development = replace(
            development,
            digest=recompute_promotion_corpus_digest(development),
        )
    elif corpus_tampering == "stored_digest":
        development = replace(development, digest="f" * 64)
    else:
        development = replace(development, cases=())

    with pytest.raises(PromotionCandidateError, match="development corpus"):
        validate_promotion_candidate(
            resolved,
            incumbent=BOT_SPECS_BY_NAME["balanced-v2"],
            development=development,
            registry={},
        )


def test_canonical_released_candidate_is_catalog_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = BOT_SPECS_BY_NAME["balanced-v2"]
    resolved = resolve_promotion_candidate(
        candidate.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates={},
    )
    development, _ = _corpora(pair_count=1)

    def forbidden_catalog_load() -> object:
        raise AssertionError("canonical released candidate loaded the frozen catalog")

    monkeypatch.setattr(
        "garboid_pocketrocks.promotion.candidates.load_frozen_candidate_catalog",
        forbidden_catalog_load,
    )

    validate_promotion_candidate(
        resolved,
        incumbent=BotSpec.for_simulation("any-incumbent", RandomBot.build_brain),
        development=development,
        registry={},
    )


def test_caller_registry_cannot_release_a_forged_frozen_identity() -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    forged = BotSpec.for_simulation(frozen.bot_spec.name, RandomBot.build_brain)
    development, _ = _corpora(pair_count=1)

    with pytest.raises(PromotionCandidateError, match="provenance"):
        validate_promotion_candidate(
            ResolvedPromotionCandidate(forged, None),
            incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
            development=development,
            registry={forged.name: forged},
        )


def test_frozen_candidate_requires_the_exact_canonical_predecessor() -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged_predecessor = BotSpec.for_simulation(
        frozen.predecessor_name,
        RandomBot.build_brain,
    )
    development, _ = _corpora(pair_count=1)

    with pytest.raises(PromotionCandidateError, match="exact canonical predecessor"):
        validate_promotion_candidate(
            resolved,
            incumbent=forged_predecessor,
            development=development,
            registry={forged_predecessor.name: forged_predecessor},
        )


def test_frozen_candidate_rejects_a_different_equal_bot_spec() -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    forged = replace(frozen.bot_spec, brain_factory=EvilFactory())
    development = _development_for_frozen(frozen)
    assert forged == frozen.bot_spec

    with pytest.raises(PromotionCandidateError, match="trusted frozen candidate record"):
        validate_promotion_candidate(
            replace(resolved, bot_spec=forged),
            incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("search_name", "forged-search-name"),
        ("profile_digest", "f" * 64),
    ),
)
def test_frozen_candidate_rejects_lying_string_subclasses(
    field_name: str,
    forged_value: str,
) -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    trusted_value = getattr(resolved.frozen_provenance, field_name)
    lying_value = EvilString(forged_value)
    forged_provenance = replace(
        resolved.frozen_provenance,
        **{field_name: lying_value},
    )
    development = _development_for_frozen(frozen)
    assert str(lying_value) != trusted_value
    assert lying_value == trusted_value

    with pytest.raises(PromotionCandidateError, match="built-in strings"):
        validate_promotion_candidate(
            replace(resolved, frozen_provenance=forged_provenance),
            incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


def test_frozen_candidate_rejects_a_lying_provenance_subclass() -> None:
    frozen = next(iter(FROZEN_CANDIDATES_BY_NAME.values()))
    resolved = resolve_promotion_candidate(
        frozen.bot_spec.name,
        registry=BOT_SPECS_BY_NAME,
        frozen_candidates=FROZEN_CANDIDATES_BY_NAME,
    )
    assert resolved.frozen_provenance is not None
    forged_provenance = evil_provenance(resolved.frozen_provenance)
    development = _development_for_frozen(frozen)
    assert forged_provenance.search_name != resolved.frozen_provenance.search_name
    assert forged_provenance == resolved.frozen_provenance

    with pytest.raises(PromotionCandidateError, match="exact provenance type"):
        validate_promotion_candidate(
            replace(resolved, frozen_provenance=forged_provenance),
            incumbent=BOT_SPECS_BY_NAME[frozen.predecessor_name],
            development=development,
            registry=BOT_SPECS_BY_NAME,
        )


def test_caller_registry_cannot_replace_a_canonical_released_spec() -> None:
    canonical = BOT_SPECS_BY_NAME["balanced-v2"]
    forged = BotSpec.for_simulation(canonical.name, RandomBot.build_brain)
    development, _ = _corpora(pair_count=1)

    with pytest.raises(PromotionCandidateError, match="canonical released registry spec"):
        validate_promotion_candidate(
            ResolvedPromotionCandidate(forged, None),
            incumbent=BOT_SPECS_BY_NAME["aggressive-v2"],
            development=development,
            registry={forged.name: forged},
        )
