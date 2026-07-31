from __future__ import annotations

import io
import json

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.behavior_cloning import (  # noqa: E402
    BALANCED_V3_PROFILE_DIGEST,
    BALANCED_V3_TEACHER_IDENTITY,
    BehaviorCloningConfig,
    BehaviorCloningDataset,
    BehaviorCloningProvenance,
    BehaviorCloningTrainer,
    balanced_v3_profile_digest,
    collect_behavior_cloning_dataset,
    plan_behavior_cloning_games,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402


def _config(**overrides: object) -> BehaviorCloningConfig:
    values: dict[str, object] = {
        "root_seed": 42,
        "rounds": 1,
        "games_per_cell": 1,
        "epochs": 2,
        "minibatch_size": 64,
    }
    values.update(overrides)
    return BehaviorCloningConfig(**values)  # type: ignore[arg-type]


def _one_game_dataset(root_seed: int = 42) -> BehaviorCloningDataset:
    config = _config(root_seed=root_seed)
    plans = plan_behavior_cloning_games(config)
    return collect_behavior_cloning_dataset(
        plans[:1],
        encoder_config=training_encoder_config(),
    )


def _model(seed: int) -> NeuralPolicy:
    torch.manual_seed(seed)
    return NeuralPolicy(training_encoder_config(), training_model_config("small"))


def _state_bytes(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def test_teacher_identity_and_profile_are_explicitly_pinned() -> None:
    assert BALANCED_V3_TEACHER_IDENTITY == "balanced-v3"
    assert balanced_v3_profile_digest() == BALANCED_V3_PROFILE_DIGEST

    with pytest.raises(ValueError, match="teacher must be balanced-v3"):
        _config(teacher_identity="balanced")
    with pytest.raises(ValueError, match="profile digest"):
        _config(teacher_profile_digest="0" * 64)


def test_config_json_is_complete_exact_key_and_round_trips() -> None:
    config = _config()
    payload = config.to_json_dict()

    assert BehaviorCloningConfig.from_json_dict(json.loads(json.dumps(payload))) == config
    assert set(payload) == {
        "root_seed",
        "rounds",
        "games_per_cell",
        "epochs",
        "minibatch_size",
        "learning_rate",
        "max_gradient_norm",
        "teacher_identity",
        "teacher_profile_digest",
    }
    missing = dict(payload)
    missing.pop("epochs")
    with pytest.raises(ValueError, match="keys do not match"):
        BehaviorCloningConfig.from_json_dict(missing)
    with pytest.raises(ValueError, match="keys do not match"):
        BehaviorCloningConfig.from_json_dict({**payload, "unknown": 1})
    with pytest.raises(ValueError, match="root_seed must be an integer"):
        BehaviorCloningConfig.from_json_dict({**payload, "root_seed": True})
    with pytest.raises(ValueError, match="must be a JSON object"):
        BehaviorCloningConfig.from_json_dict([])


def test_planning_is_repeatable_balanced_and_namespaced() -> None:
    config = _config(rounds=2, games_per_cell=2)

    first = plan_behavior_cloning_games(config)
    second = plan_behavior_cloning_games(config)
    changed = plan_behavior_cloning_games(_config(root_seed=43, rounds=2, games_per_cell=2))

    assert first == second
    assert len(first) == 2 * 2 * 5 * 3
    assert len({plan.engine_seed for plan in first}) == len(first)
    assert tuple(plan.engine_seed for plan in first) != tuple(plan.engine_seed for plan in changed)
    cells = {(plan.ruleset_name, plan.player_count) for plan in first}
    assert cells == {(f"live-{chart}", players) for chart in "ABCDE" for players in (3, 4, 5)}
    assert all(
        sum(plan.ruleset_name == ruleset and plan.player_count == players for plan in first) == 4
        for ruleset, players in cells
    )


def test_collection_is_public_immutable_legal_and_repeatable() -> None:
    first = _one_game_dataset()
    second = _one_game_dataset()

    assert first.dataset_digest == second.dataset_digest
    assert tuple(example.action for example in first.examples) == tuple(
        example.action for example in second.examples
    )
    assert first.teacher_identity == BALANCED_V3_TEACHER_IDENTITY
    assert first.teacher_profile_digest == BALANCED_V3_PROFILE_DIGEST
    assert first.game_count == 1
    assert first.cell_game_counts == (("live-A", 3, 1),)
    assert all(bool(example.observation.action_mask[example.action]) for example in first.examples)
    assert all(
        not getattr(example.observation, name).flags.writeable
        for example in first.examples
        for name in (
            "global_ids",
            "global_numeric",
            "objective_bits",
            "seat_numeric",
            "seat_valid",
            "private_hand_ids",
            "hand_valid",
            "history_ids",
            "history_numeric",
            "history_valid",
            "action_mask",
        )
    )


def test_policy_only_training_is_exactly_repeatable_and_reports_finite_metrics() -> None:
    dataset = _one_game_dataset()
    config = _config()
    initial = _model(91).state_dict()
    first_model = _model(1)
    second_model = _model(2)
    first_model.load_state_dict(initial)
    second_model.load_state_dict(initial)
    first = BehaviorCloningTrainer(first_model, config)
    second = BehaviorCloningTrainer(second_model, config)

    torch.manual_seed(100)
    first_metrics = first.train(dataset)
    torch.manual_seed(200)
    second_metrics = second.train(dataset)

    assert first_metrics == second_metrics
    assert first_metrics.example_count == len(dataset.examples)
    assert first_metrics.optimizer_steps > 0
    assert all(update.negative_log_likelihood >= 0.0 for update in first_metrics.updates)
    assert all(0.0 <= update.teacher_agreement <= 1.0 for update in first_metrics.updates)
    assert all(update.entropy >= 0.0 for update in first_metrics.updates)
    assert _state_bytes(first_model.state_dict()) == _state_bytes(second_model.state_dict())
    assert _state_bytes(first.optimizer.state_dict()) == _state_bytes(second.optimizer.state_dict())
    assert torch.equal(
        first_model.value_head.weight,
        initial["value_head.weight"],
    )
    assert torch.equal(
        first_model.value_head.bias,
        initial["value_head.bias"],
    )

    provenance = BehaviorCloningProvenance.create(config, dataset, first_metrics)
    payload = provenance.to_json_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload == {
        "schema_version": 1,
        "method": "behavior_cloning",
        "config_digest": config.config_digest,
        "teacher": {
            "identity": BALANCED_V3_TEACHER_IDENTITY,
            "profile_digest": BALANCED_V3_PROFILE_DIGEST,
        },
        "demonstrations": {
            "dataset_digest": dataset.dataset_digest,
            "games": 1,
            "examples": len(dataset.examples),
            "cell_game_counts": [["live-A", 3, 1]],
        },
        "training": {
            "epochs": config.epochs,
            "optimizer_steps": first_metrics.optimizer_steps,
        },
    }
