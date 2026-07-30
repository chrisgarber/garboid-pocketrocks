from __future__ import annotations

import math
import pickle

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.adapters.public_history import (  # noqa: E402
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots import BotSpec, RandomBot  # noqa: E402
from garboid_pocketrocks.diagnostics.trace import (  # noqa: E402
    NeuralPolicyExplanation,
    RecordedAction,
    legal_actions_for_context,
)
from garboid_pocketrocks.knowledge import canonical_knowledge  # noqa: E402
from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    LARGE_BOT_NAME,
    LARGE_CHECKPOINT_PATH,
    SMOKE_BOT_NAME,
    SMOKE_CHECKPOINT_PATH,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
    VectorPpoLargeV1G350kBrain,
    VectorPpoSmallV1G1500Brain,
)
from garboid_pocketrocks.simulator.runner import MatchRunner  # noqa: E402
from garboid_pocketrocks.simulator.session import SdkGameSession  # noqa: E402


def test_smoke_checkpoint_is_frozen_at_named_training_age() -> None:
    loaded = load_inference_checkpoint(
        SMOKE_CHECKPOINT_PATH,
        device=torch.device("cpu"),
    )

    assert SMOKE_BOT_NAME == "vector_ppo_small_v1_g1500"
    assert {item.name for item in SMOKE_CHECKPOINT_PATH.iterdir()} == {
        "manifest.json",
        "model.pt",
    }
    assert loaded.manifest.completed_episodes == 1_500
    assert loaded.manifest.completed_updates == 1
    assert loaded.manifest.supported_ruleset_names == tuple(f"live-{chart}" for chart in "ABCDE")
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
    assert len(loaded.manifest.parameter_digest) == 64


def test_large_checkpoint_is_frozen_at_rounded_training_age() -> None:
    loaded = load_inference_checkpoint(
        LARGE_CHECKPOINT_PATH,
        device=torch.device("cpu"),
    )

    assert LARGE_BOT_NAME == "vector_ppo_large_v1_g350k"
    assert {item.name for item in LARGE_CHECKPOINT_PATH.iterdir()} == {
        "manifest.json",
        "model.pt",
    }
    assert loaded.manifest.completed_episodes == 349_860
    assert loaded.manifest.completed_updates == 196
    assert loaded.manifest.supported_ruleset_names == tuple(f"live-{chart}" for chart in "ABCDE")
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
    assert len(loaded.manifest.parameter_digest) == 64


@pytest.mark.parametrize(
    "brain_type",
    (VectorPpoSmallV1G1500Brain, VectorPpoLargeV1G350kBrain),
)
def test_frozen_neural_brains_return_deterministic_legal_decisions(
    brain_type: type[VectorPpoSmallV1G1500Brain] | type[VectorPpoLargeV1G350kBrain],
) -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=19,
        value_chart="B",
        objectives_enabled=True,
        player_names=("neural", "random-1", "random-2"),
    )
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    knowledge = canonical_knowledge(3, value_chart="B")
    brain = brain_type(seed=7)

    first = brain.choose_decision_with_history(context, knowledge, history)
    second = brain.choose_decision_with_history(context, knowledge, history)

    assert first == second
    context.validate(first)


def test_frozen_neural_explanation_reuses_the_selected_masked_output() -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=19,
        value_chart="B",
        objectives_enabled=True,
        player_names=("neural", "random-1", "random-2"),
    )
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    knowledge = canonical_knowledge(3, value_chart="B")
    brain = VectorPpoSmallV1G1500Brain(seed=7)
    model_calls: list[None] = []
    hook = brain._runtime.model.register_forward_hook(lambda *_arguments: model_calls.append(None))

    try:
        explained = brain.choose_explained_decision(context, knowledge, history)
    finally:
        hook.remove()

    context.validate(explained.decision)
    assert len(model_calls) == 1
    assert isinstance(explained.explanation, NeuralPolicyExplanation)
    assert all(
        math.isfinite(value)
        for value in (
            explained.explanation.predicted_value,
            explained.explanation.selected_probability,
            explained.explanation.entropy,
        )
    )
    assert 0.0 <= explained.explanation.selected_probability <= 1.0
    assert explained.explanation.entropy >= 0.0
    assert len(explained.explanation.legal_action_probabilities) == len(
        legal_actions_for_context(context)
    )
    assert math.fsum(explained.explanation.legal_action_probabilities) == pytest.approx(1.0)
    selected_index = legal_actions_for_context(context).index(
        RecordedAction.from_decision(explained.decision)
    )
    assert explained.explanation.legal_action_probabilities[selected_index] == pytest.approx(
        explained.explanation.selected_probability
    )


def test_ordinary_neural_choice_does_not_construct_an_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=19,
        value_chart="B",
        objectives_enabled=True,
        player_names=("neural", "random-1", "random-2"),
    )
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    knowledge = canonical_knowledge(3, value_chart="B")
    brain = VectorPpoSmallV1G1500Brain(seed=7)
    expected = brain.choose_decision_with_history(context, knowledge, history)

    def reject_explanation(**_values: object) -> None:
        raise RuntimeError("explanation construction is disabled")

    monkeypatch.setattr(
        "garboid_pocketrocks.neural.tournament_bot.NeuralPolicyExplanation",
        reject_explanation,
    )

    assert brain.choose_decision_with_history(context, knowledge, history) == expected
    with pytest.raises(RuntimeError, match="explanation construction"):
        brain.choose_explained_decision(context, knowledge, history)


def test_smoke_spec_is_pickle_safe_and_completes_a_match() -> None:
    restored = pickle.loads(pickle.dumps(VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC))
    random_spec = BotSpec.from_bot_class(RandomBot)

    assert restored == VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC
    match = MatchRunner.run(
        (restored, random_spec, random_spec),
        player_count=3,
        seed=23,
        value_chart="E",
    )
    assert match.result.scores
    assert not match.faults


def test_large_spec_is_pickle_safe_and_completes_a_match() -> None:
    restored = pickle.loads(pickle.dumps(VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC))
    random_spec = BotSpec.from_bot_class(RandomBot)

    assert restored == VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC
    match = MatchRunner.run(
        (restored, random_spec, random_spec),
        player_count=3,
        seed=23,
        value_chart="E",
    )
    assert match.result.scores
    assert not match.faults
