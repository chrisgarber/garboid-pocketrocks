from __future__ import annotations

import math
import random
from dataclasses import fields

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.rollout import (  # noqa: E402
    RolloutMetadata,
    collect_rollout,
)
from garboid_pocketrocks.neural.seeding import (  # noqa: E402
    configure_deterministic_torch,
    derive_seed,
    plan_stage1_episodes,
)
from garboid_pocketrocks.training.actions import ActionCodec  # noqa: E402
from garboid_pocketrocks.training.bounds import EnvironmentBounds  # noqa: E402


def test_stage1_episode_plan_is_named_stable_and_order_independent() -> None:
    plans = plan_stage1_episodes(root_seed=42, updates=2, games_per_update=16)

    assert len(plans) == 32
    assert [plan.learner_seat for plan in plans] == [
        index % 3 for index in range(32)
    ]
    assert plans == plan_stage1_episodes(root_seed=42, updates=2, games_per_update=16)

    by_key = {(plan.update_index, plan.episode_index): plan for plan in plans}
    reverse_by_key = {
        (plan.update_index, plan.episode_index): plan for plan in reversed(plans)
    }
    assert by_key == reverse_by_key

    seeds = {
        seed
        for plan in plans
        for seed in (plan.environment_seed, plan.opponent_seed, plan.policy_seed)
    }
    assert len(seeds) == 96
    assert all(0 <= seed < 2**63 for seed in seeds)
    for plan in plans:
        assert plan.environment_seed == derive_seed(
            42,
            "environment",
            plan.update_index,
            plan.episode_index,
        )
        assert plan.opponent_seed == derive_seed(
            42,
            "opponent",
            plan.update_index,
            plan.episode_index,
        )
        assert plan.policy_seed == derive_seed(
            42,
            "policy",
            plan.update_index,
            plan.episode_index,
        )
    with pytest.raises(ValueError, match="namespace"):
        derive_seed(42, "environment:0", 1)


def test_deterministic_setup_reseeds_all_runtimes_in_one_process() -> None:
    configure_deterministic_torch(13)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    configure_deterministic_torch(13)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


def test_rollout_collects_only_immutable_legal_learner_transitions() -> None:
    configure_deterministic_torch(101)
    torch.manual_seed(derive_seed(101, "model"))
    model = NeuralPolicy(stage1_encoder_config(), stage1_model_config())
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    plans = plan_stage1_episodes(root_seed=101, updates=1, games_per_update=2)

    rollout = collect_rollout(model, plans)
    random.random()
    np.random.random()
    torch.rand(4)
    repeated = collect_rollout(model, plans)

    assert len(rollout.episodes) == 2
    assert rollout.transitions
    assert [
        (
            episode.result,
            tuple(
                (
                    transition.action,
                    transition.old_log_probability,
                    transition.old_value,
                    transition.reward,
                )
                for transition in episode.transitions
            ),
        )
        for episode in rollout.episodes
    ] == [
        (
            episode.result,
            tuple(
                (
                    transition.action,
                    transition.old_log_probability,
                    transition.old_value,
                    transition.reward,
                )
                for transition in episode.transitions
            ),
        )
        for episode in repeated.episodes
    ]
    codec = ActionCodec(EnvironmentBounds(max_bid=100, max_hand_size=5))
    for episode in rollout.episodes:
        assert episode.terminated
        assert not episode.truncated
        assert episode.result is not None
        assert episode.opponent_names == ("balanced", "passive")
        assert episode.transitions
        assert math.isfinite(float(episode.final_money))
        assert 1 <= episode.rank <= 3
        assert episode.outright_first is (
            episode.rank == 1
            and sum(score.rank == 1 for score in episode.result.scores) == 1
        )
        assert episode.tied_first is (
            episode.rank == 1
            and sum(score.rank == 1 for score in episode.result.scores) > 1
        )
        assert math.isclose(
            episode.reward_breakdown.total,
            sum(transition.reward for transition in episode.transitions),
        )
        assert all(
            math.isfinite(component)
            for component in (
                episode.reward_breakdown.accounting,
                episode.reward_breakdown.terminal_resource,
                episode.reward_breakdown.placement,
                episode.reward_breakdown.shaping,
                episode.reward_breakdown.penalty,
            )
        )

        expected_metadata = RolloutMetadata(
            ruleset_name="live-A",
            player_count=3,
            learner_seat=episode.plan.learner_seat,
            opponent_names=("balanced", "passive"),
            environment_seed=episode.plan.environment_seed,
            opponent_seed=episode.plan.opponent_seed,
            policy_seed=episode.plan.policy_seed,
        )
        for transition in episode.transitions:
            assert transition.metadata == expected_metadata
            assert transition.context.bot_seat == episode.plan.learner_seat
            assert transition.observation.action_mask[transition.action]
            assert codec.mask(transition.context)[transition.action] == 1
            assert codec.encode(codec.decode(transition.action)) == transition.action
            assert transition.illegal_probability == 0.0
            assert math.isfinite(transition.old_log_probability)
            assert math.isfinite(transition.old_value)
            assert math.isfinite(transition.reward)
            assert all(math.isfinite(logit) for logit in transition.bid_logits)
            assert all(math.isfinite(logit) for logit in transition.reveal_logits)
            assert all(
                math.isfinite(component)
                for component in (
                    transition.reward_breakdown.accounting,
                    transition.reward_breakdown.terminal_resource,
                    transition.reward_breakdown.placement,
                    transition.reward_breakdown.shaping,
                    transition.reward_breakdown.penalty,
                )
            )
            for array_field in fields(transition.observation):
                array = getattr(transition.observation, array_field.name)
                assert not array.flags.writeable
                if np.issubdtype(array.dtype, np.floating):
                    assert np.isfinite(array).all()
            for action, legal in enumerate(transition.observation.action_mask):
                masked_logit = transition.masked_logits[action]
                if legal:
                    assert math.isfinite(masked_logit)
                else:
                    assert masked_logit == -math.inf

    assert rollout.transitions == tuple(
        transition
        for episode in rollout.episodes
        for transition in episode.transitions
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
