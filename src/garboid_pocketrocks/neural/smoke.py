"""Deterministic end-to-end Stage 1 PPO mechanics smoke."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from garboid_pocketrocks.bots.heuristic import (
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
)
from garboid_pocketrocks.neural.checkpoint import (
    InferenceManifest,
    load_inference_checkpoint,
    parameter_digest,
    save_inference_checkpoint,
)
from garboid_pocketrocks.neural.config import (
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.encoding import (
    NeuralBatch,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.neural.metrics import CalibrationBucket, ValueMetrics
from garboid_pocketrocks.neural.model import NeuralPolicy, PolicyValueOutput
from garboid_pocketrocks.neural.policy import (
    PolicySelection,
    evaluate_masked_policy,
)
from garboid_pocketrocks.neural.ppo import (
    PPOConfig,
    PPOTrainer,
    PPOUpdateMetrics,
)
from garboid_pocketrocks.neural.rollout import (
    RolloutEpisode,
    RolloutTransition,
    collect_rollout,
)
from garboid_pocketrocks.neural.run_config import TrainingRunConfig
from garboid_pocketrocks.neural.seeding import (
    EpisodePlan,
    configure_deterministic_torch,
    derive_seed,
    plan_stage1_episodes,
)
from garboid_pocketrocks.neural.trainer import train
from garboid_pocketrocks.neural.training_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.rewards import RewardBreakdown
from garboid_pocketrocks.training.single_agent_env import PocketRocksEnv


class SmokeError(ValueError):
    """Raised when the Stage 1 mechanics smoke cannot complete safely."""


@dataclass(frozen=True, slots=True)
class SelfPlaySmokeResult:
    """Acceptance metrics for the full A-E, three-to-five-player smoke."""

    completed_updates: int
    completed_episodes: int
    completed_decisions: int
    cell_games: tuple[tuple[str, int, int], ...]
    games_per_second: float
    decisions_per_second: float
    illegal_actions: int
    faults: int
    value: ValueMetrics
    checkpoint_replay_verified: bool
    resume_verified: bool


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Resolved Stage 1 smoke inputs."""

    root_seed: int = 42
    updates: int = 2
    games_per_update: int = 16
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, int) or isinstance(self.root_seed, bool):
            raise SmokeError("root seed must be an integer")
        for name in ("updates", "games_per_update"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SmokeError(f"{name} must be a positive integer")
        if self.device != "cpu":
            raise SmokeError("Stage 1 smoke is fixed to CPU")


def smoke_run_config() -> TrainingRunConfig:
    """Load the committed 1,500-game self-play smoke contract."""

    return TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))


def run_self_play_smoke(
    config: TrainingRunConfig,
    output_dir: Path,
) -> SelfPlaySmokeResult:
    """Train once, validate the checkpoint, and execute a resume probe."""

    result = train(config, output_dir)
    loaded = load_training_checkpoint(
        result.final_checkpoint,
        device=torch.device("cpu"),
    )
    metrics = loaded.metrics
    collection = cast(dict[str, object], metrics["collection"])
    ppo = cast(dict[str, object], metrics["ppo"])
    value = _read_value_metrics(cast(dict[str, object], ppo["value"]))
    probe_path = result.run_dir / "resume-probe"
    save_training_checkpoint(
        probe_path,
        model=loaded.model,
        optimizer=loaded.optimizer,
        manifest=replace(
            loaded.manifest,
            lineage=(
                *loaded.manifest.lineage,
                str(result.final_checkpoint.resolve()),
            ),
        ),
        generator_states=loaded.generator_states,
        metrics=loaded.metrics,
    )
    resumed = load_training_checkpoint(
        probe_path,
        device=torch.device("cpu"),
    )
    smoke_result = SelfPlaySmokeResult(
        completed_updates=result.completed_updates,
        completed_episodes=result.completed_episodes,
        completed_decisions=result.completed_decisions,
        cell_games=loaded.manifest.progress.cell_games,
        games_per_second=_as_float(
            collection["games_per_second"],
            "games_per_second",
        ),
        decisions_per_second=_as_float(
            collection["decisions_per_second"],
            "decisions_per_second",
        ),
        illegal_actions=0,
        faults=0,
        value=value,
        checkpoint_replay_verified=(
            loaded.manifest.parameter_digest == parameter_digest(loaded.model.state_dict())
        ),
        resume_verified=(
            resumed.manifest.progress == loaded.manifest.progress
            and resumed.manifest.parameter_digest == loaded.manifest.parameter_digest
            and _nested_equal(
                resumed.optimizer.state_dict(),
                loaded.optimizer.state_dict(),
            )
        ),
    )
    _write_json_payload(
        result.run_dir / "self-play-smoke-result.json",
        asdict(smoke_result),
    )
    return smoke_result


@dataclass(frozen=True, slots=True)
class SmokeEpisodeMetrics:
    """Auditable terminal metrics for one completed learner game."""

    plan: EpisodePlan
    terminated: bool
    truncated: bool
    transition_count: int
    illegal_action_count: int
    fault_count: int
    illegal_probability: float
    final_money: int
    rank: int
    outright_first: bool
    tied_first: bool
    reward_breakdown: RewardBreakdown


@dataclass(frozen=True, slots=True)
class SmokeUpdateMetrics:
    """Deterministic mechanics and outcomes for one PPO update."""

    update_index: int
    episodes: tuple[SmokeEpisodeMetrics, ...]
    ppo: PPOUpdateMetrics
    parameters_changed: bool
    raw_outputs_finite: bool
    selected_policy_finite: bool
    masked_illegal_logits_negative_infinity: bool
    max_illegal_probability: float
    mean_rank: float
    reward_breakdown: RewardBreakdown


@dataclass(frozen=True, slots=True)
class CheckpointReplay:
    """Canonical fixture outputs verified across checkpoint reload."""

    bid_logits: tuple[float, ...]
    reveal_logits: tuple[float, ...]
    value: float
    greedy_action: int
    verified: bool


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Immutable result with deterministic and machine-local sections."""

    config: SmokeConfig
    updates: tuple[SmokeUpdateMetrics, ...]
    final_parameter_digest: str
    checkpoint_replay: CheckpointReplay
    elapsed_seconds: float
    output_dir: Path
    environment_metadata: tuple[tuple[str, str], ...]

    def deterministic_payload(self) -> dict[str, Any]:
        """Return only fields that must compare exactly across machines/runs."""

        return _json_safe(
            {
                "config": asdict(self.config),
                "updates": [asdict(update) for update in self.updates],
                "final_parameter_digest": self.final_parameter_digest,
                "checkpoint_replay": asdict(self.checkpoint_replay),
            }
        )


def run_smoke(config: SmokeConfig, output_dir: Path) -> SmokeResult:
    """Run deterministic Stage 1 training, checkpointing, and replay."""

    started = time.perf_counter()
    resolved_output = output_dir.resolve()
    _prepare_empty_output(resolved_output)
    configure_deterministic_torch(config.root_seed)
    torch.manual_seed(derive_seed(config.root_seed, "model"))
    model = NeuralPolicy(stage1_encoder_config(), stage1_model_config())
    ppo_config = PPOConfig()
    trainer = PPOTrainer(model, ppo_config)
    plans = plan_stage1_episodes(
        root_seed=config.root_seed,
        updates=config.updates,
        games_per_update=config.games_per_update,
    )
    update_metrics: list[SmokeUpdateMetrics] = []

    for update_index in range(config.updates):
        update_plans = tuple(plan for plan in plans if plan.update_index == update_index)
        rollout = collect_rollout(model, update_plans)
        parameters_before = {
            name: parameter.detach().clone() for name, parameter in model.state_dict().items()
        }
        ppo = trainer.update(
            rollout,
            update_seed=derive_seed(config.root_seed, "minibatch", update_index),
        )
        parameters_changed = any(
            not torch.equal(parameter, parameters_before[name])
            for name, parameter in model.state_dict().items()
        )
        if not parameters_changed:
            raise SmokeError("PPO update did not change any model parameter")
        update_metrics.append(
            _summarize_update(
                update_index,
                rollout.episodes,
                ppo,
                parameters_changed=parameters_changed,
            )
        )

    fixture_batch = _canonical_fixture_batch()
    model.eval()
    with torch.no_grad():
        before_output = model(fixture_batch)
        before_selection = evaluate_masked_policy(
            before_output,
            fixture_batch,
            generator=None,
            deterministic=True,
        )
    checkpoint_dir = resolved_output / "checkpoint"
    manifest = InferenceManifest.create(
        model,
        repository_commit=_repository_commit(),
        root_seed=config.root_seed,
        completed_episodes=config.updates * config.games_per_update,
        completed_updates=config.updates,
    )
    save_inference_checkpoint(checkpoint_dir, model, manifest)
    loaded = load_inference_checkpoint(
        checkpoint_dir,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        after_output = loaded.model(fixture_batch)
        after_selection = evaluate_masked_policy(
            after_output,
            fixture_batch,
            generator=None,
            deterministic=True,
        )
    replay = _checkpoint_replay(
        before_output,
        before_selection,
        after_output,
        after_selection,
    )
    elapsed = time.perf_counter() - started
    result = SmokeResult(
        config=config,
        updates=tuple(update_metrics),
        final_parameter_digest=loaded.manifest.parameter_digest,
        checkpoint_replay=replay,
        elapsed_seconds=elapsed,
        output_dir=resolved_output,
        environment_metadata=(
            ("numpy", np.__version__),
            ("platform", platform.platform()),
            ("python", platform.python_version()),
            ("torch", str(torch.__version__)),
        ),
    )
    _write_result(result)
    return result


def _summarize_update(
    update_index: int,
    episodes: tuple[RolloutEpisode, ...],
    ppo: PPOUpdateMetrics,
    *,
    parameters_changed: bool,
) -> SmokeUpdateMetrics:
    if not episodes:
        raise SmokeError("smoke update collected no games")
    episode_metrics = tuple(_summarize_episode(episode) for episode in episodes)
    transitions = tuple(transition for episode in episodes for transition in episode.transitions)
    raw_outputs_finite = all(
        all(math.isfinite(value) for value in transition.bid_logits)
        and all(math.isfinite(value) for value in transition.reveal_logits)
        for transition in transitions
    )
    selected_policy_finite = all(
        math.isfinite(transition.old_log_probability)
        and math.isfinite(transition.old_value)
        and math.isfinite(transition.reward)
        for transition in transitions
    )
    masked_illegal = all(_masked_logits_are_exact(transition) for transition in transitions)
    maximum_illegal_probability = max(transition.illegal_probability for transition in transitions)
    if (
        not raw_outputs_finite
        or not selected_policy_finite
        or not masked_illegal
        or maximum_illegal_probability != 0.0
        or any(metric.illegal_action_count or metric.fault_count for metric in episode_metrics)
    ):
        raise SmokeError("rollout mechanics assertion failed")
    _validate_ppo_metrics(ppo)
    return SmokeUpdateMetrics(
        update_index=update_index,
        episodes=episode_metrics,
        ppo=ppo,
        parameters_changed=parameters_changed,
        raw_outputs_finite=raw_outputs_finite,
        selected_policy_finite=selected_policy_finite,
        masked_illegal_logits_negative_infinity=masked_illegal,
        max_illegal_probability=maximum_illegal_probability,
        mean_rank=sum(episode.rank for episode in episode_metrics) / len(episode_metrics),
        reward_breakdown=_sum_rewards(episode.reward_breakdown for episode in episode_metrics),
    )


def _summarize_episode(episode: RolloutEpisode) -> SmokeEpisodeMetrics:
    illegal_actions = sum(
        not transition.observation.action_mask[transition.action]
        for transition in episode.transitions
    )
    illegal_probability = max(transition.illegal_probability for transition in episode.transitions)
    if not episode.terminated or episode.truncated:
        raise SmokeError("smoke collected an incomplete game")
    return SmokeEpisodeMetrics(
        plan=episode.plan,
        terminated=episode.terminated,
        truncated=episode.truncated,
        transition_count=len(episode.transitions),
        illegal_action_count=illegal_actions,
        fault_count=0,
        illegal_probability=illegal_probability,
        final_money=episode.final_money,
        rank=episode.rank,
        outright_first=episode.outright_first,
        tied_first=episode.tied_first,
        reward_breakdown=episode.reward_breakdown,
    )


def _masked_logits_are_exact(transition: RolloutTransition) -> bool:
    return all(
        math.isfinite(logit) if legal else logit == -math.inf
        for legal, logit in zip(
            transition.observation.action_mask,
            transition.masked_logits,
            strict=True,
        )
    )


def _validate_ppo_metrics(metrics: PPOUpdateMetrics) -> None:
    values = (
        metrics.total_loss,
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        *metrics.advantages,
        *metrics.ratios,
        *metrics.values,
        *metrics.entropies,
        *metrics.pre_clip_gradient_norms,
        *metrics.post_clip_gradient_norms,
    )
    if (
        metrics.epochs != 1
        or not all(math.isfinite(value) for value in values)
        or not metrics.post_clip_gradient_norms
        or max(metrics.post_clip_gradient_norms) > PPOConfig().max_gradient_norm
    ):
        raise SmokeError("PPO mechanics assertion failed")


def _checkpoint_replay(
    before_output: PolicyValueOutput,
    before_selection: PolicySelection,
    after_output: PolicyValueOutput,
    after_selection: PolicySelection,
) -> CheckpointReplay:
    verified = (
        torch.equal(before_output.bid_logits, after_output.bid_logits)
        and torch.equal(before_output.reveal_logits, after_output.reveal_logits)
        and torch.equal(before_output.value, after_output.value)
        and torch.equal(before_selection.actions, after_selection.actions)
    )
    if not verified:
        raise SmokeError("reloaded checkpoint changed canonical fixture inference")
    return CheckpointReplay(
        bid_logits=_tensor_row(before_output.bid_logits),
        reveal_logits=_tensor_row(before_output.reveal_logits),
        value=float(before_output.value[0].item()),
        greedy_action=int(before_selection.actions[0].item()),
        verified=True,
    )


def _canonical_fixture_batch() -> NeuralBatch:
    config = stage1_encoder_config()
    bounds = EnvironmentBounds(config.max_bid, config.max_hand_size)
    env = PocketRocksEnv(
        opponent_specs=(
            BALANCED_HEURISTIC_BOT_SPEC,
            PASSIVE_HEURISTIC_BOT_SPEC,
        ),
        value_charts=("A",),
        player_count=3,
        bounds=bounds,
        learner_seat=0,
    )
    env.reset(
        seed=314_159,
        options={"opponent_seed": 271_828},
    )
    observation = NeuralObservationEncoder(config, bounds).encode(
        env.learner_context,
        env.ruleset_knowledge,
        env.public_history,
    )
    return batch_observations((observation,), torch.device("cpu"))


def _sum_rewards(rewards: Iterable[RewardBreakdown]) -> RewardBreakdown:
    collected = tuple(rewards)
    return RewardBreakdown(
        accounting=sum(reward.accounting for reward in collected),
        terminal_resource=sum(reward.terminal_resource for reward in collected),
        placement=sum(reward.placement for reward in collected),
        shaping=sum(reward.shaping for reward in collected),
        penalty=sum(reward.penalty for reward in collected),
    )


def _tensor_row(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor[0].detach().cpu().tolist())


def _prepare_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise SmokeError("smoke output directory must be empty")
        return
    try:
        path.mkdir(parents=True)
    except OSError as error:
        raise SmokeError("smoke output directory could not be created") from error


def _repository_commit() -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def _json_safe(payload: object) -> dict[str, Any]:
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    parsed: object = json.loads(encoded)
    if not isinstance(parsed, dict):
        raise SmokeError("smoke payload must be a JSON object")
    return cast(dict[str, Any], parsed)


def _read_value_metrics(payload: dict[str, object]) -> ValueMetrics:
    calibration_payload = cast(
        list[dict[str, object]],
        payload["calibration"],
    )
    return ValueMetrics(
        count=int(cast(int, payload["count"])),
        mean_prediction=float(cast(float, payload["mean_prediction"])),
        mean_target=float(cast(float, payload["mean_target"])),
        mae=float(cast(float, payload["mae"])),
        rmse=float(cast(float, payload["rmse"])),
        bias=float(cast(float, payload["bias"])),
        explained_variance=(
            None
            if payload["explained_variance"] is None
            else float(cast(float, payload["explained_variance"]))
        ),
        correlation=(
            None if payload["correlation"] is None else float(cast(float, payload["correlation"]))
        ),
        calibration=tuple(
            CalibrationBucket(
                count=int(cast(int, bucket["count"])),
                minimum_prediction=float(cast(float, bucket["minimum_prediction"])),
                maximum_prediction=float(cast(float, bucket["maximum_prediction"])),
                mean_prediction=float(cast(float, bucket["mean_prediction"])),
                mean_target=float(cast(float, bucket["mean_target"])),
            )
            for bucket in calibration_payload
        ),
    )


def _write_json_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SmokeError(f"{name} must be finite")
    return float(value)


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(first, second) for first, second in zip(left, right, strict=True)
        )
    return bool(left == right)


def _write_result(result: SmokeResult) -> None:
    payload = {
        "config": asdict(result.config),
        "deterministic": result.deterministic_payload(),
        "non_deterministic": {
            "elapsed_seconds": result.elapsed_seconds,
            "environment_metadata": dict(result.environment_metadata),
            "output_dir": str(result.output_dir),
        },
    }
    (result.output_dir / "smoke-result.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
