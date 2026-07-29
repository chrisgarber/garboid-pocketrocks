from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from garboid_pocketrocks.bots.heuristic import (  # noqa: E402
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
)
from garboid_pocketrocks.neural.config import stage1_encoder_config  # noqa: E402
from garboid_pocketrocks.neural.encoding import (  # noqa: E402
    NeuralBatch,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.neural.smoke import (  # noqa: E402
    SmokeConfig,
    _canonical_fixture_batch,
    run_smoke,
)
from garboid_pocketrocks.rules import LIVE_RULESET  # noqa: E402
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler  # noqa: E402
from garboid_pocketrocks.training.bounds import EnvironmentBounds  # noqa: E402
from garboid_pocketrocks.training.single_agent_env import PocketRocksEnv  # noqa: E402


def _batch_tensors(batch: NeuralBatch) -> tuple[Tensor, ...]:
    return (
        batch.global_ids,
        batch.global_numeric,
        batch.objective_bits,
        batch.seat_numeric,
        batch.seat_valid,
        batch.private_hand_ids,
        batch.hand_valid,
        batch.history_ids,
        batch.history_numeric,
        batch.history_valid,
        batch.action_mask,
    )


def test_canonical_checkpoint_fixture_is_a_reachable_environment_decision() -> None:
    bounds = EnvironmentBounds(max_bid=100, max_hand_size=5)
    env = PocketRocksEnv(
        opponent_specs=(
            BALANCED_HEURISTIC_BOT_SPEC,
            PASSIVE_HEURISTIC_BOT_SPEC,
        ),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        player_count=3,
        bounds=bounds,
        learner_seat=0,
    )
    env.reset(seed=314_159, options={"opponent_seed": 271_828})
    observation = NeuralObservationEncoder(
        stage1_encoder_config(),
        bounds,
    ).encode(
        env.learner_context,
        env.ruleset_knowledge,
        env.public_history,
    )
    reachable = batch_observations((observation,), torch.device("cpu"))

    for fixture_tensor, reachable_tensor in zip(
        _batch_tensors(_canonical_fixture_batch()),
        _batch_tensors(reachable),
        strict=True,
    ):
        assert torch.equal(fixture_tensor, reachable_tensor)


@pytest.mark.neural_smoke
def test_two_by_sixteen_smoke_is_deterministic(tmp_path: Path) -> None:
    first = run_smoke(SmokeConfig(), tmp_path / "first")
    second = run_smoke(SmokeConfig(), tmp_path / "second")

    for result in (first, second):
        assert result.config == SmokeConfig(
            root_seed=42,
            updates=2,
            games_per_update=16,
            device="cpu",
        )
        assert len(result.updates) == 2
        assert sum(len(update.episodes) for update in result.updates) == 32
        assert result.checkpoint_replay.verified
        assert result.final_parameter_digest
        assert result.elapsed_seconds > 0.0
        assert result.output_dir.is_dir()
        for update_index, update in enumerate(result.updates):
            assert update.update_index == update_index
            assert len(update.episodes) == 16
            assert update.ppo.epochs == 1
            assert update.parameters_changed
            assert update.raw_outputs_finite
            assert update.selected_policy_finite
            assert update.masked_illegal_logits_negative_infinity
            assert update.max_illegal_probability == 0.0
            assert math.isfinite(update.mean_rank)
            assert all(
                math.isfinite(value)
                for value in (
                    update.reward_breakdown.accounting,
                    update.reward_breakdown.terminal_resource,
                    update.reward_breakdown.placement,
                    update.reward_breakdown.shaping,
                    update.reward_breakdown.penalty,
                    update.ppo.total_loss,
                    update.ppo.policy_loss,
                    update.ppo.value_loss,
                    update.ppo.entropy,
                    *update.ppo.advantages,
                    *update.ppo.ratios,
                    *update.ppo.values,
                    *update.ppo.entropies,
                    *update.ppo.pre_clip_gradient_norms,
                    *update.ppo.post_clip_gradient_norms,
                )
            )
            for episode in update.episodes:
                assert episode.plan.update_index == update_index
                assert episode.terminated
                assert not episode.truncated
                assert episode.illegal_action_count == 0
                assert episode.fault_count == 0
                assert episode.illegal_probability == 0.0
                assert math.isfinite(float(episode.final_money))
                assert math.isfinite(float(episode.rank))
                assert isinstance(episode.outright_first, bool)
                assert isinstance(episode.tied_first, bool)

        stored = cast(
            dict[str, Any],
            json.loads((result.output_dir / "smoke-result.json").read_text()),
        )
        assert stored["deterministic"] == result.deterministic_payload()
        assert set(stored["non_deterministic"]) == {
            "elapsed_seconds",
            "environment_metadata",
            "output_dir",
        }
        assert (result.output_dir / "checkpoint/manifest.json").is_file()
        assert (result.output_dir / "checkpoint/model.pt").is_file()

    assert first.deterministic_payload() == second.deterministic_payload()
    assert first.final_parameter_digest == second.final_parameter_digest
    assert first.output_dir != second.output_dir
    assert all(
        key not in first.deterministic_payload()
        for key in ("elapsed_seconds", "environment_metadata", "output_dir")
    )
