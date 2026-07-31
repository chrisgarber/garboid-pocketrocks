from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch")

import garboid_pocketrocks.neural.bootstrap_promotion as promotion  # noqa: E402
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec  # noqa: E402
from garboid_pocketrocks.neural.heuristic_bootstrap import (  # noqa: E402
    REFERENCE_PARAMETER_DIGEST,
)
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    FrozenBootstrapBrainFactory,
)
from garboid_pocketrocks.promotion.corpus import (  # noqa: E402
    CorpusPurpose,
    PromotionCase,
    PromotionCorpus,
    PromotionCorpusRecipe,
    load_promotion_corpus,
    recompute_promotion_corpus_digest,
)
from garboid_pocketrocks.promotion.runner import (  # noqa: E402
    PromotionRun,
    PromotionRunConfig,
    PromotionRunner,
)

_STRATEGY = "fixed-compute-control-v1"
_IDENTITY = "vector_ppo_large_fixed_compute_control_v1_test"
_SUMMARY_DIGEST = "2" * 64
_PARAMETER_DIGEST = "3" * 64
_HELD_OUT_PATH = Path("configs/promotion/held-out-v1.json")


def test_public_gate_has_no_selection_or_result_override_knobs() -> None:
    parameters = inspect.signature(promotion.run_selected_bootstrap_promotion).parameters

    assert {
        "candidate",
        "candidate_identity",
        "strategy",
        "training_checkpoint",
        "inference_checkpoint",
        "bootstrap_samples",
        "bootstrap_seed",
        "batch_size",
        "output_dir",
        "overwrite",
        "repository_commit",
    }.isdisjoint(parameters)
    assert set(parameters) == {
        "evidence_output_dir",
        "frozen_arms_dir",
        "development_corpus",
        "held_out_corpus_path",
        "workers",
    }


def _corpus(*, purpose: str, name: str, root_seed: int) -> PromotionCorpus:
    corpus = PromotionCorpus(
        recipe=PromotionCorpusRecipe(
            schema_version=1,
            name=name,
            purpose=cast(CorpusPurpose, purpose),
            root_seed=root_seed,
            repetitions_per_seat_cell=1,
            charts=("A",),
            player_counts=(3,),
            opponent_names=("random", "aggressive-v1", "balanced-v1", "passive-v1"),
        ),
        cases=(
            PromotionCase(
                case_id=f"{name}-case-1",
                chart="A",
                player_count=3,
                focal_seat=0,
                engine_seed=root_seed + 1,
                opponent_names_by_seat=(None, "random", "aggressive-v1"),
            ),
        ),
        digest="",
    )
    return replace(corpus, digest=recompute_promotion_corpus_digest(corpus))


def _candidate(frozen_root: Path) -> BotSpec:
    candidate_path = (frozen_root / _STRATEGY).resolve()
    return BotSpec.for_simulation(
        _IDENTITY,
        FrozenBootstrapBrainFactory(
            candidate_path=candidate_path,
            expected_identity=_IDENTITY,
        ),
    )


def _frozen(development: PromotionCorpus, *, digest: str | None = None) -> Any:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            identity=_IDENTITY,
            strategy=_STRATEGY,
            development_corpus_name=development.recipe.name,
            development_corpus_digest=digest or development.digest,
            summary_digest=_SUMMARY_DIGEST,
            parameter_digest=_PARAMETER_DIGEST,
        )
    )


def _write_bound_files(tmp_path: Path) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "evidence-manifest.json").write_text(
        '{"schema_version":1}\n',
        encoding="utf-8",
    )
    frozen_root = tmp_path / "frozen"
    candidate_path = frozen_root / _STRATEGY
    candidate_path.mkdir(parents=True)
    (candidate_path / "manifest.json").write_text(
        '{"schema_version":1}\n',
        encoding="utf-8",
    )
    return evidence, frozen_root


def _mock_verified_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    development: PromotionCorpus,
) -> tuple[Path, Path, BotSpec]:
    evidence, frozen_root = _write_bound_files(tmp_path)
    candidate = _candidate(frozen_root)
    monkeypatch.setattr(
        promotion,
        "selected_bootstrap_bot_spec",
        lambda **kwargs: candidate,
    )
    monkeypatch.setattr(
        promotion,
        "load_frozen_bootstrap_candidate",
        lambda path: _frozen(development),
    )
    monkeypatch.setattr(
        promotion,
        "load_inference_checkpoint",
        lambda path, *, device: SimpleNamespace(
            manifest=SimpleNamespace(parameter_digest=REFERENCE_PARAMETER_DIGEST)
        ),
    )
    return evidence, frozen_root, candidate


def _forbid_held_out_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("held-out data or the promotion runner was touched too early")

    monkeypatch.setattr(promotion, "load_promotion_corpus", forbidden)
    monkeypatch.setattr(PromotionRunner, "run", forbidden)


def test_selection_failure_does_not_touch_held_out_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )

    def reject_selection(**kwargs: object) -> BotSpec:
        del kwargs
        raise ValueError("selection evidence changed")

    monkeypatch.setattr(promotion, "selected_bootstrap_bot_spec", reject_selection)
    _forbid_held_out_and_runner(monkeypatch)

    with pytest.raises(ValueError, match="selection evidence changed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=tmp_path / "evidence",
            frozen_arms_dir=tmp_path / "frozen",
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )


def test_freeze_failure_does_not_touch_held_out_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    _, frozen_root = _write_bound_files(tmp_path)
    candidate = _candidate(frozen_root)
    monkeypatch.setattr(
        promotion,
        "selected_bootstrap_bot_spec",
        lambda **kwargs: candidate,
    )

    def reject_freeze(path: Path) -> Any:
        del path
        raise ValueError("freeze digest changed")

    monkeypatch.setattr(promotion, "load_frozen_bootstrap_candidate", reject_freeze)
    _forbid_held_out_and_runner(monkeypatch)

    with pytest.raises(ValueError, match="freeze digest changed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=tmp_path / "evidence",
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )


def test_development_binding_failure_does_not_touch_held_out_or_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    _, frozen_root = _write_bound_files(tmp_path)
    candidate = _candidate(frozen_root)
    monkeypatch.setattr(
        promotion,
        "selected_bootstrap_bot_spec",
        lambda **kwargs: candidate,
    )
    monkeypatch.setattr(
        promotion,
        "load_frozen_bootstrap_candidate",
        lambda path: _frozen(development, digest="f" * 64),
    )
    _forbid_held_out_and_runner(monkeypatch)

    with pytest.raises(
        promotion.BootstrapPromotionError,
        match="does not bind the supplied development corpus",
    ):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=tmp_path / "evidence",
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )


def test_existing_receipt_blocks_before_held_out_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    evidence, frozen_root, _ = _mock_verified_inputs(tmp_path, monkeypatch, development)
    locations = promotion.bootstrap_promotion_locations(evidence)
    locations.attempt_receipt_json.write_text("already consumed\n", encoding="utf-8")
    _forbid_held_out_and_runner(monkeypatch)

    with pytest.raises(promotion.BootstrapPromotionError, match="already consumed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )


def test_incumbent_parameter_mismatch_blocks_before_held_out_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    evidence, frozen_root, _ = _mock_verified_inputs(tmp_path, monkeypatch, development)
    monkeypatch.setattr(
        promotion,
        "load_inference_checkpoint",
        lambda path, *, device: SimpleNamespace(
            manifest=SimpleNamespace(parameter_digest="0" * 64)
        ),
    )
    _forbid_held_out_and_runner(monkeypatch)

    with pytest.raises(promotion.BootstrapPromotionError, match="parameter digest changed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )

    assert not promotion.bootstrap_promotion_locations(evidence).attempt_receipt_json.exists()


def test_wrong_held_out_exam_consumes_attempt_without_running_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    evidence, frozen_root, _ = _mock_verified_inputs(tmp_path, monkeypatch, development)
    wrong_held_out = _corpus(
        purpose="held_out",
        name="different-held-out-v1",
        root_seed=2_000,
    )
    held_out_loads = 0

    def load_wrong_exam(path: Path, *, registry: object) -> PromotionCorpus:
        nonlocal held_out_loads
        del path, registry
        held_out_loads += 1
        return wrong_held_out

    def forbidden_runner(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("games ran for an unpinned held-out corpus")

    monkeypatch.setattr(promotion, "load_promotion_corpus", load_wrong_exam)
    monkeypatch.setattr(PromotionRunner, "run", forbidden_runner)

    with pytest.raises(promotion.BootstrapPromotionError, match="committed held-out-v1"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=tmp_path / "different-held-out-v1.json",
            workers=1,
        )
    with pytest.raises(promotion.BootstrapPromotionError, match="already consumed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )

    assert held_out_loads == 1


def test_crashed_attempt_consumes_held_out_and_cannot_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    evidence, frozen_root, _ = _mock_verified_inputs(tmp_path, monkeypatch, development)
    held_out = load_promotion_corpus(_HELD_OUT_PATH, registry=BOT_SPECS_BY_NAME)
    held_out_loads = 0

    def load_held_out(path: Path, *, registry: object) -> PromotionCorpus:
        nonlocal held_out_loads
        del path, registry
        held_out_loads += 1
        return held_out

    def crash(*args: object, **kwargs: object) -> PromotionRun:
        del args, kwargs
        raise RuntimeError("simulator crashed")

    monkeypatch.setattr(promotion, "load_promotion_corpus", load_held_out)
    monkeypatch.setattr(PromotionRunner, "run", crash)

    with pytest.raises(RuntimeError, match="simulator crashed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )
    with pytest.raises(promotion.BootstrapPromotionError, match="already consumed"):
        promotion.run_selected_bootstrap_promotion(
            evidence_output_dir=evidence,
            frozen_arms_dir=frozen_root,
            development_corpus=development,
            held_out_corpus_path=_HELD_OUT_PATH,
            workers=1,
        )

    assert held_out_loads == 1
    assert promotion.bootstrap_promotion_locations(evidence).attempt_receipt_json.is_file()


def test_verified_winner_uses_fixed_gate_and_writes_bound_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = _corpus(
        purpose="development",
        name="development-fixture-v1",
        root_seed=1_000,
    )
    evidence, frozen_root, candidate = _mock_verified_inputs(
        tmp_path,
        monkeypatch,
        development,
    )
    held_out = load_promotion_corpus(_HELD_OUT_PATH, registry=BOT_SPECS_BY_NAME)
    locations = promotion.bootstrap_promotion_locations(evidence)
    registry_before = dict(BOT_SPECS_BY_NAME)
    calls: dict[str, object] = {}

    def load_held_out(path: Path, *, registry: object) -> PromotionCorpus:
        calls["held_out"] = (path, registry)
        return held_out

    def run_gate(config: PromotionRunConfig, **kwargs: object) -> PromotionRun:
        calls["runner"] = (config, kwargs)
        output_dir = cast(Path, kwargs["output_dir"])
        output_dir.mkdir()
        report_path = output_dir / "promotion-report.json"
        report_path.write_text('{"promoted":false}\n', encoding="utf-8")
        return cast(
            PromotionRun,
            SimpleNamespace(artifacts=SimpleNamespace(report_json=report_path)),
        )

    monkeypatch.setattr(promotion, "load_promotion_corpus", load_held_out)
    monkeypatch.setattr(PromotionRunner, "run", run_gate)

    result = promotion.run_selected_bootstrap_promotion(
        evidence_output_dir=evidence,
        frozen_arms_dir=frozen_root,
        development_corpus=development,
        held_out_corpus_path=_HELD_OUT_PATH,
        workers=3,
    )

    assert result.artifacts.report_json == locations.promotion_output_dir / "promotion-report.json"
    assert calls["held_out"] == (_HELD_OUT_PATH, BOT_SPECS_BY_NAME)
    runner_call = cast(tuple[object, object], calls["runner"])
    config = cast(PromotionRunConfig, runner_call[0])
    runner_options = cast(dict[str, object], runner_call[1])
    assert config.candidate is candidate
    assert config.incumbent is VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC
    assert config.development is development
    assert config.held_out is held_out
    assert config.bootstrap_samples == 1_000
    assert config.bootstrap_seed == 0
    assert config.batch_size == 64
    assert runner_options == {
        "registry": BOT_SPECS_BY_NAME,
        "workers": 3,
        "output_dir": locations.promotion_output_dir,
    }

    receipt_bytes = locations.attempt_receipt_json.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert (
        receipt["evidence_manifest_digest"]
        == hashlib.sha256((evidence / "evidence-manifest.json").read_bytes()).hexdigest()
    )
    assert receipt["selected_candidate_identity"] == _IDENTITY
    assert (
        receipt["selected_freeze_manifest_digest"]
        == hashlib.sha256((frozen_root / _STRATEGY / "manifest.json").read_bytes()).hexdigest()
    )
    assert receipt["selected_summary_digest"] == _SUMMARY_DIGEST
    assert receipt["selected_parameter_digest"] == _PARAMETER_DIGEST
    assert receipt["development_corpus_digest"] == development.digest
    assert receipt["expected_held_out_corpus_name"] == "held-out-v1"
    assert receipt["expected_held_out_corpus_digest"] == held_out.digest
    assert receipt["incumbent_parameter_digest"] == REFERENCE_PARAMETER_DIGEST
    assert receipt["promotion_rule"] == {
        "bootstrap_samples": 1_000,
        "bootstrap_seed": 0,
        "confidence_level": 0.95,
        "require_fault_free_games": True,
        "require_positive_lower_bound": True,
    }

    provenance = json.loads(locations.promotion_provenance_json.read_bytes())
    public_provenance = {
        key: value for key, value in provenance.items() if key != "provenance_digest"
    }
    assert provenance["attempt_receipt"]["sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert (
        provenance["promotion_report"]["sha256"]
        == hashlib.sha256(result.artifacts.report_json.read_bytes()).hexdigest()
    )
    assert (
        provenance["provenance_digest"]
        == hashlib.sha256(
            json.dumps(public_provenance, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )
    assert dict(BOT_SPECS_BY_NAME) == registry_before
    assert all(BOT_SPECS_BY_NAME[name] is spec for name, spec in registry_before.items())
