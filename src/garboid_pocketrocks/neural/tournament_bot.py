"""Frozen neural policies used by the standard local tournament."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.diagnostics.trace import (
    ExplainedBotDecision,
    NeuralPolicyExplanation,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge

if TYPE_CHECKING:
    import torch

    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.neural.model import NeuralPolicy
    from garboid_pocketrocks.neural.policy import PolicySelection
    from garboid_pocketrocks.training.actions import ActionCodec

CHECKPOINTS_PATH = Path(__file__).with_name("checkpoints")
SMOKE_BOT_NAME = "vector_ppo_small_v1_g1500"
LARGE_BOT_NAME = "vector_ppo_large_v1_g350k"
SMOKE_CHECKPOINT_PATH = CHECKPOINTS_PATH / SMOKE_BOT_NAME
LARGE_CHECKPOINT_PATH = CHECKPOINTS_PATH / LARGE_BOT_NAME


@dataclass(frozen=True, slots=True)
class _Runtime:
    model: NeuralPolicy
    encoder: NeuralObservationEncoder
    codec: ActionCodec
    device: torch.device


@dataclass(frozen=True, slots=True)
class _NeuralPolicyChoice:
    decision: BotDecision
    action_index: int
    policy_selection: PolicySelection
    legal_action_mask: torch.Tensor


@cache
def _runtime(checkpoint_path: Path) -> _Runtime:
    import torch

    from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint
    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.training.actions import ActionCodec
    from garboid_pocketrocks.training.bounds import EnvironmentBounds

    torch.set_num_threads(1)
    device = torch.device("cpu")
    loaded = load_inference_checkpoint(checkpoint_path, device=device)
    bounds = EnvironmentBounds(
        loaded.manifest.encoder_config.max_bid,
        loaded.manifest.encoder_config.max_hand_size,
    )
    codec = ActionCodec(bounds)
    return _Runtime(
        model=loaded.model,
        encoder=NeuralObservationEncoder(
            loaded.manifest.encoder_config,
            bounds,
            action_codec=codec,
        ),
        codec=codec,
        device=device,
    )


class _FrozenNeuralBrain:
    """Deterministic inference wrapper shared by frozen checkpoints."""

    checkpoint_path: ClassVar[Path]

    def __init__(self, seed: int | None = None) -> None:
        del seed
        self._runtime = _runtime(self.checkpoint_path)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("frozen neural policy requires public history")

    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        return self._choose_raw(context, ruleset, history).decision

    def choose_explained_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> ExplainedBotDecision:
        """Choose once and retain finite diagnostics from that masked selection."""

        choice = self._choose_raw(context, ruleset, history)
        selection = choice.policy_selection
        legal_action_probabilities = tuple(
            float(probability.item())
            for probability in selection.probabilities[0][choice.legal_action_mask]
        )
        return ExplainedBotDecision(
            decision=choice.decision,
            explanation=NeuralPolicyExplanation(
                predicted_value=float(selection.value[0].item()),
                selected_probability=float(selection.probabilities[0, choice.action_index].item()),
                entropy=float(selection.entropy[0].item()),
                legal_action_probabilities=legal_action_probabilities,
            ),
        )

    def _choose_raw(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> _NeuralPolicyChoice:
        """Run masked inference once without constructing diagnostic records."""

        import torch

        from garboid_pocketrocks.neural.encoding import batch_observations
        from garboid_pocketrocks.neural.policy import evaluate_masked_policy

        observation = self._runtime.encoder.encode(context, ruleset, history)
        batch = batch_observations((observation,), self._runtime.device)
        with torch.inference_mode():
            output = self._runtime.model(batch)
            selection = evaluate_masked_policy(
                output,
                batch,
                generator=None,
                deterministic=True,
            )
        action_index = int(selection.actions[0].item())
        return _NeuralPolicyChoice(
            decision=self._runtime.codec.decode(action_index),
            action_index=action_index,
            policy_selection=selection,
            legal_action_mask=batch.action_mask[0],
        )


class FrozenBootstrapCandidateBrain(_FrozenNeuralBrain):
    """Deterministic brain backed only by a verified bootstrap freeze."""

    def __init__(
        self,
        candidate_path: Path,
        expected_identity: str,
        seed: int | None = None,
    ) -> None:
        del seed
        from garboid_pocketrocks.neural.bootstrap_freeze import (
            BootstrapFreezeError,
            load_frozen_bootstrap_candidate,
        )

        frozen = load_frozen_bootstrap_candidate(candidate_path)
        if frozen.manifest.identity != expected_identity:
            raise BootstrapFreezeError(
                "frozen bootstrap candidate identity changed after BotSpec creation"
            )
        self._runtime = _runtime(candidate_path / "inference")


@dataclass(frozen=True, slots=True)
class FrozenBootstrapBrainFactory:
    """Picklable factory that re-verifies a frozen candidate in each worker."""

    candidate_path: Path
    expected_identity: str

    def __post_init__(self) -> None:
        if not self.candidate_path.is_absolute():
            raise ValueError("frozen bootstrap candidate path must be absolute")
        if not self.expected_identity:
            raise ValueError("frozen bootstrap candidate identity must be nonempty")

    def __call__(self, seed: int | None = None) -> FrozenBootstrapCandidateBrain:
        return FrozenBootstrapCandidateBrain(
            self.candidate_path,
            self.expected_identity,
            seed,
        )


def frozen_bootstrap_bot_spec(candidate_path: Path) -> BotSpec:
    """Build a local BotSpec from one fully verified bootstrap freeze directory.

    This deliberately accepts neither a raw training checkpoint nor a raw
    inference checkpoint.  Verification runs now and again whenever a worker
    constructs the brain, so later payload tampering fails closed.
    """

    from garboid_pocketrocks.neural.bootstrap_freeze import (
        load_frozen_bootstrap_candidate,
    )

    resolved_path = candidate_path.resolve()
    frozen = load_frozen_bootstrap_candidate(resolved_path)
    identity = frozen.manifest.identity
    return BotSpec.for_simulation(
        identity,
        FrozenBootstrapBrainFactory(
            candidate_path=resolved_path,
            expected_identity=identity,
        ),
    )


class VectorPpoSmallV1G1500Brain(_FrozenNeuralBrain):
    """Frozen 1,500-game smoke policy."""

    checkpoint_path = SMOKE_CHECKPOINT_PATH


class VectorPpoLargeV1G350kBrain(_FrozenNeuralBrain):
    """Frozen 349,860-game large policy."""

    checkpoint_path = LARGE_CHECKPOINT_PATH


VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC = BotSpec.for_simulation(
    SMOKE_BOT_NAME,
    VectorPpoSmallV1G1500Brain,
)
VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC = BotSpec.for_simulation(
    LARGE_BOT_NAME,
    VectorPpoLargeV1G350kBrain,
)
