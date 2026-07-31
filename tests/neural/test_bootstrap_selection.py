from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import garboid_pocketrocks.neural.bootstrap_selection as selection_module
from garboid_pocketrocks.bots import BOT_SPECS_BY_NAME, BotSpec
from garboid_pocketrocks.evolution.evaluation import CandidateEvaluation, evaluate_candidate
from garboid_pocketrocks.evolution.planning import DevelopmentPlan, plan_development_games
from garboid_pocketrocks.knowledge import ruleset_name
from garboid_pocketrocks.neural.bootstrap_freeze import FrozenBootstrapCandidate
from garboid_pocketrocks.neural.bootstrap_selection import (
    DEVELOPMENT_CORPUS_DIGEST,
    BootstrapCandidateResult,
    BootstrapSelectionError,
    FrozenBootstrapArm,
    _evaluate_frozen_arms,
    _game_payload,
    _is_complete_fault_free,
    _is_strict_improvement,
    _load_verified_evidence,
    _ranking_key,
    _read_json_object,
    _recompute_evidence_selection,
    _require_pinned_incumbent_checkpoint,
    _require_safe_paths,
    _write_evidence_artifacts,
    run_bootstrap_development_selection,
)
from garboid_pocketrocks.neural.heuristic_bootstrap import HEURISTIC_BOOTSTRAP_ARMS
from garboid_pocketrocks.neural.tournament_bot import VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC
from garboid_pocketrocks.promotion.corpus import PromotionCorpus, load_promotion_corpus
from garboid_pocketrocks.simulator.monte_carlo import (
    GameJob,
    GameSummary,
    MonteCarloResult,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.session import SessionScore


def _evaluation(
    identity: str,
    *,
    rating: float = 1.0,
    finish: float = 2.0,
    money: int = 3,
    candidate_faults: int = 0,
    completed: int = 10,
    valid: bool = True,
    eligible: bool = True,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_identity=identity,
        incumbent_identity="vector_ppo_large_v1_g350k",
        requested_cases=10,
        completed_baseline_games=completed,
        completed_candidate_games=completed,
        rating_delta=rating,
        normalized_finish_delta=finish,
        final_money_delta=money,
        candidate_faults=candidate_faults,
        incumbent_faults=0,
        opponent_faults=0,
        unattributed_faults=0,
        faults_by_identity=((identity, candidate_faults),) if candidate_faults else (),
        failures=(),
        valid=valid,
        eligible=eligible,
    )


def _completed_summary(job: GameJob, focal_identity: str, focal_rank: int) -> GameSummary:
    focal_seat = next(seat for seat, spec in enumerate(job.lineup) if spec.bot_id == focal_identity)
    remaining_ranks = iter(rank for rank in range(1, job.player_count + 1) if rank != focal_rank)
    ranks = tuple(
        focal_rank if seat == focal_seat else next(remaining_ranks)
        for seat in range(job.player_count)
    )
    return GameSummary(
        game_index=job.game_index,
        root_seed=job.root_seed,
        seed=job.seed,
        player_count=job.player_count,
        ruleset_name=ruleset_name(job.value_chart, job.objectives_enabled),
        bot_names=tuple(spec.name for spec in job.lineup),
        bot_ids=tuple(spec.bot_id for spec in job.lineup),
        scores=tuple(
            SessionScore(seat=seat, final_money=100 - rank, rank=rank)
            for seat, rank in enumerate(ranks)
        ),
        decision_counts=(1,) * job.player_count,
        fault_counts=(0,) * job.player_count,
    )


def _write_valid_evidence(
    tmp_path: Path,
) -> tuple[Path, tuple[FrozenBootstrapArm, ...], PromotionCorpus, str]:
    corpus = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    factory = BOT_SPECS_BY_NAME["vector_ppo_large_v1_g350k"].brain_factory
    arms: list[FrozenBootstrapArm] = []
    for index, contract in enumerate(HEURISTIC_BOOTSTRAP_ARMS):
        identity = f"bootstrap-test-{index}"
        arms.append(
            FrozenBootstrapArm(
                strategy=contract.strategy,
                path=tmp_path / contract.strategy,
                candidate=cast(
                    FrozenBootstrapCandidate,
                    SimpleNamespace(
                        manifest=SimpleNamespace(
                            identity=identity,
                            summary_digest="a" * 64,
                        )
                    ),
                ),
                bot_spec=BotSpec(identity, identity, factory),
                freeze_manifest_digest=hashlib.sha256(contract.strategy.encode()).hexdigest(),
            )
        )
    frozen_arms = tuple(arms)
    first_plan = plan_development_games(
        corpus,
        candidate=frozen_arms[0].bot_spec,
        incumbent=VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
        registry=BOT_SPECS_BY_NAME,
    )
    baseline = MonteCarloResult(
        tuple(
            _completed_summary(
                job,
                "vector_ppo_large_v1_g350k",
                job.player_count,
            )
            for job in first_plan.baseline_jobs
        ),
        (),
        (),
    )
    candidate_results: list[BootstrapCandidateResult] = []
    for index, arm in enumerate(frozen_arms):
        plan = plan_development_games(
            corpus,
            candidate=arm.bot_spec,
            incumbent=VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
            registry=BOT_SPECS_BY_NAME,
        )
        candidate_result = MonteCarloResult(
            tuple(
                _completed_summary(job, arm.bot_spec.name, 1 if index == 0 else 2)
                for job in plan.candidate_jobs
            ),
            (),
            (),
        )
        candidate_results.append(
            BootstrapCandidateResult(
                arm=arm,
                plan=plan,
                result=candidate_result,
                evaluation=evaluate_candidate(plan, baseline, candidate_result),
            )
        )
    ranked = tuple(
        item.evaluation.candidate_identity
        for item in sorted(candidate_results, key=lambda item: _ranking_key(item.evaluation))
    )
    evidence = tmp_path / "evidence"
    _write_evidence_artifacts(
        evidence,
        development_corpus=corpus,
        frozen_arms=frozen_arms,
        baseline_result=baseline,
        candidate_results=tuple(candidate_results),
        ranked_identities=ranked,
        selected_identity=ranked[0],
    )
    return evidence, frozen_arms, corpus, ranked[0]


def test_ranking_uses_documented_metrics_then_immutable_identity() -> None:
    rating_winner = _evaluation("z-rating", rating=2.0, finish=0.0, money=0)
    finish_winner = _evaluation("z-finish", rating=1.0, finish=3.0, money=0)
    money_winner = _evaluation("z-money", rating=1.0, finish=2.0, money=4)
    identity_winner = _evaluation("a-identity", rating=1.0, finish=2.0, money=3)
    identity_loser = _evaluation("b-identity", rating=1.0, finish=2.0, money=3)

    ranked = sorted(
        (identity_loser, money_winner, rating_winner, identity_winner, finish_winner),
        key=_ranking_key,
    )

    assert [item.candidate_identity for item in ranked] == [
        "z-rating",
        "z-finish",
        "z-money",
        "a-identity",
        "b-identity",
    ]


@pytest.mark.parametrize(
    ("evaluation", "expected"),
    [
        (_evaluation("positive", rating=0.01), True),
        (_evaluation("zero", rating=0.0), False),
        (_evaluation("negative", rating=-0.01), False),
        (_evaluation("fault", candidate_faults=1, eligible=False), False),
        (_evaluation("incomplete", completed=9), False),
    ],
)
def test_selection_requires_complete_fault_free_strict_improvement(
    evaluation: CandidateEvaluation,
    expected: bool,
) -> None:
    assert _is_strict_improvement(evaluation) is expected
    assert _is_complete_fault_free(evaluation) is (
        evaluation.candidate_identity
        not in {
            "fault",
            "incomplete",
        }
    )


def test_public_api_rejects_held_out_corpus_before_reading_checkpoints(tmp_path: Path) -> None:
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    held_out = replace(
        development,
        recipe=replace(development.recipe, purpose="held_out"),
    )

    with pytest.raises(BootstrapSelectionError, match="only a development corpus"):
        run_bootstrap_development_selection(
            [],
            held_out,
            bootstrap_summary_path=tmp_path / "not-read.json",
            frozen_arms_dir=tmp_path / "frozen",
            evidence_output_dir=tmp_path / "evidence",
        )

    parameters = inspect.signature(run_bootstrap_development_selection).parameters
    assert "held_out" not in parameters
    assert "held_out_corpus" not in parameters


def test_raw_game_payload_records_utilities_and_faults() -> None:
    summary = GameSummary(
        game_index=4,
        root_seed=11,
        seed=12,
        player_count=3,
        ruleset_name="live-A-3p-objectives",
        bot_names=("candidate", "opponent-a", "opponent-b"),
        bot_ids=("candidate", "opponent-a", "opponent-b"),
        scores=(
            SessionScore(seat=0, final_money=8, rank=1),
            SessionScore(seat=1, final_money=2, rank=2),
            SessionScore(seat=2, final_money=-1, rank=3),
        ),
        decision_counts=(4, 5, 6),
        fault_counts=(0, 1, 0),
    )

    payload = _game_payload(
        summary,
        evidence="candidate",
        candidate_identity="candidate",
    )

    assert payload["utilities_by_seat"] == [
        {"seat": 0, "rank": 1, "normalized_finish": 1.0, "final_money": 8},
        {"seat": 1, "rank": 2, "normalized_finish": 0.5, "final_money": 2},
        {"seat": 2, "rank": 3, "normalized_finish": 0.0, "final_money": -1},
    ]
    assert payload["fault_counts_by_seat"] == [0, 1, 0]
    assert payload["engine_seed"] == 12


def test_transaction_targets_cannot_overlap_each_other_or_inputs(tmp_path: Path) -> None:
    summary = tmp_path / "bootstrap-summary.json"
    summary.write_text("{}", encoding="utf-8")

    with pytest.raises(BootstrapSelectionError, match="must not overlap"):
        _require_safe_paths(
            [],
            bootstrap_summary_path=summary,
            frozen_arms_dir=tmp_path / "results",
            evidence_output_dir=tmp_path / "results" / "evidence",
        )

    with pytest.raises(BootstrapSelectionError, match="inputs must be outside"):
        _require_safe_paths(
            [],
            bootstrap_summary_path=summary,
            frozen_arms_dir=tmp_path,
            evidence_output_dir=tmp_path.parent / "separate-evidence",
        )


def test_all_candidates_reuse_one_exact_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    canonical_factory = BOT_SPECS_BY_NAME["vector_ppo_large_v1_g350k"].brain_factory
    arms = tuple(
        FrozenBootstrapArm(
            strategy="fixed-compute-control-v1",
            path=tmp_path / identity,
            candidate=cast(FrozenBootstrapCandidate, object()),
            bot_spec=BotSpec(identity, identity, canonical_factory),
            freeze_manifest_digest="f" * 64,
        )
        for identity in ("candidate-a", "candidate-b")
    )
    baseline_config = object()
    baseline_jobs = (object(),)

    def fake_plan(
        _corpus: object,
        *,
        candidate: object,
        incumbent: object,
        registry: object,
    ) -> DevelopmentPlan:
        del incumbent, registry
        return cast(
            DevelopmentPlan,
            SimpleNamespace(
                candidate=candidate,
                baseline_config=baseline_config,
                baseline_jobs=baseline_jobs,
                candidate_config=f"candidate:{candidate.name}",  # type: ignore[attr-defined]
                candidate_jobs=(candidate,),
            ),
        )

    calls: list[object] = []
    empty_result = MonteCarloResult((), (), ())

    def fake_run_jobs(
        config: object,
        jobs: object,
        *,
        workers: int,
        batch_size: int | None,
    ) -> MonteCarloResult:
        del jobs, workers, batch_size
        calls.append(config)
        return empty_result

    baseline_ids: list[int] = []

    def fake_evaluate(
        plan: DevelopmentPlan,
        baseline: MonteCarloResult,
        candidate: MonteCarloResult,
    ) -> CandidateEvaluation:
        del candidate
        baseline_ids.append(id(baseline))
        return _evaluation(plan.candidate.name)

    monkeypatch.setattr(selection_module, "plan_development_games", fake_plan)
    monkeypatch.setattr(MonteCarloRunner, "run_jobs", fake_run_jobs)
    monkeypatch.setattr(selection_module, "evaluate_candidate", fake_evaluate)

    baseline, results = _evaluate_frozen_arms(arms, corpus, workers=3, batch_size=8)

    assert baseline is empty_result
    assert calls == [baseline_config, "candidate:candidate-a", "candidate:candidate-b"]
    assert baseline_ids == [id(empty_result), id(empty_result)]
    assert len(results) == 2


def test_selection_rejects_any_development_corpus_other_than_the_committed_pin(
    tmp_path: Path,
) -> None:
    development = load_promotion_corpus(
        Path("configs/promotion/development-v1.json"),
        registry=BOT_SPECS_BY_NAME,
    )
    assert development.digest == DEVELOPMENT_CORPUS_DIGEST
    changed = replace(development, digest="0" * 64)

    with pytest.raises(BootstrapSelectionError, match="exact committed development-v1"):
        run_bootstrap_development_selection(
            [],
            changed,
            bootstrap_summary_path=tmp_path / "not-read.json",
            frozen_arms_dir=tmp_path / "frozen",
            evidence_output_dir=tmp_path / "evidence",
        )


def test_selection_verifies_pinned_incumbent_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import garboid_pocketrocks.neural.checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "load_inference_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            manifest=SimpleNamespace(parameter_digest="0" * 64)
        ),
    )

    with pytest.raises(BootstrapSelectionError, match="parameter digest has changed"):
        _require_pinned_incumbent_checkpoint()


def test_bound_games_and_evaluations_recompute_the_same_winner(tmp_path: Path) -> None:
    evidence, frozen_arms, corpus, winner = _write_valid_evidence(tmp_path)
    decision = _read_json_object(evidence / "selection-decision.json", "selection decision")

    assert (
        _recompute_evidence_selection(
            evidence,
            corpus=corpus,
            frozen_arms=frozen_arms,
            decision=decision,
        )
        == winner
    )


def test_rehashed_decision_evaluation_and_game_edits_cannot_substitute_winner(
    tmp_path: Path,
) -> None:
    evidence, frozen_arms, corpus, winner = _write_valid_evidence(tmp_path)
    other = frozen_arms[1].bot_spec.name

    decision_path = evidence / "selection-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["selected_candidate_identity"] = other
    decision["ranked_candidate_identities"] = [
        other,
        winner,
        *decision["ranked_candidate_identities"][2:],
    ]
    decision_path.write_text(
        json.dumps(decision, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evaluations_path = evidence / "candidate-evaluations.jsonl"
    evaluations = [json.loads(line) for line in evaluations_path.read_text().splitlines()]
    evaluations[0]["scores"], evaluations[1]["scores"] = (
        evaluations[1]["scores"],
        evaluations[0]["scores"],
    )
    evaluations_path.write_text(
        selection_module._json_lines(evaluations),
        encoding="utf-8",
    )

    games_path = evidence / "development-games.jsonl"
    games = [json.loads(line) for line in games_path.read_text().splitlines()]
    first_candidate_game = games[len(corpus.cases)]
    focal_identity = frozen_arms[0].bot_spec.name
    focal_seat = first_candidate_game["bot_ids_by_seat"].index(focal_identity)
    other_seat = 0 if focal_seat != 0 else 1
    for field in ("rank", "normalized_finish", "final_money"):
        (
            first_candidate_game["utilities_by_seat"][focal_seat][field],
            first_candidate_game["utilities_by_seat"][other_seat][field],
        ) = (
            first_candidate_game["utilities_by_seat"][other_seat][field],
            first_candidate_game["utilities_by_seat"][focal_seat][field],
        )
    games_path.write_text(selection_module._json_lines(games), encoding="utf-8")

    manifest_path = evidence / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = evidence / record["name"]
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = _load_verified_evidence(evidence)

    with pytest.raises(BootstrapSelectionError, match="does not recompute"):
        _recompute_evidence_selection(
            evidence,
            corpus=corpus,
            frozen_arms=frozen_arms,
            decision=loaded["decision"],
        )
