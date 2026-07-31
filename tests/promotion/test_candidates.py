from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.bots.random_bot import RandomBot
from garboid_pocketrocks.heuristics.frozen import FROZEN_CANDIDATES_BY_NAME
from garboid_pocketrocks.promotion.candidates import (
    FrozenCandidateProvenance,
    PromotionCandidateError,
    ResolvedPromotionCandidate,
    resolve_promotion_candidate,
    validate_promotion_candidate,
)
from garboid_pocketrocks.promotion.corpus import (
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)

from .helpers import EvilFactory, EvilString, evil_provenance
from .test_runner import _corpora

_DIGESTS = tuple(character * 64 for character in "abcde")


@dataclass(frozen=True, slots=True)
class _FrozenCandidateFixture:
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


def _frozen_candidate() -> _FrozenCandidateFixture:
    development, _ = _corpora(pair_count=1)
    return _FrozenCandidateFixture(
        bot_spec=BotSpec.for_simulation(
            "balanced-v3-candidate-test",
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
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
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
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
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
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
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
