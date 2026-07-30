"""Frozen neural policy used by the standard local tournament."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge

if TYPE_CHECKING:
    import torch

    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.neural.model import NeuralPolicy
    from garboid_pocketrocks.training.actions import ActionCodec

SMOKE_BOT_NAME = "vector_ppo_small_v1_g1500"
SMOKE_CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / SMOKE_BOT_NAME


@dataclass(frozen=True, slots=True)
class _Runtime:
    model: NeuralPolicy
    encoder: NeuralObservationEncoder
    codec: ActionCodec
    device: torch.device


@cache
def _runtime() -> _Runtime:
    import torch

    from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint
    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.training.actions import ActionCodec
    from garboid_pocketrocks.training.bounds import EnvironmentBounds

    torch.set_num_threads(1)
    device = torch.device("cpu")
    loaded = load_inference_checkpoint(SMOKE_CHECKPOINT_PATH, device=device)
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


class VectorPpoSmallV1G1500Brain:
    """Deterministic inference wrapper for the frozen smoke checkpoint."""

    def __init__(self, seed: int | None = None) -> None:
        del seed
        self._runtime = _runtime()

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("smoke neural policy requires public history")

    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
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
        return self._runtime.codec.decode(int(selection.actions[0].item()))


VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC = BotSpec.for_simulation(
    SMOKE_BOT_NAME,
    VectorPpoSmallV1G1500Brain,
)
