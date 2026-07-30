from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.adapters.public_history import (  # noqa: E402
    public_history_from_sdk_events,
)
from garboid_pocketrocks.bots import BotSpec, RandomBot  # noqa: E402
from garboid_pocketrocks.knowledge import canonical_knowledge  # noqa: E402
from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    SMOKE_BOT_NAME,
    SMOKE_CHECKPOINT_PATH,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
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


def test_smoke_brain_returns_a_deterministic_legal_decision() -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=19,
        value_chart="B",
        objectives_enabled=True,
        player_names=("smoke", "random-1", "random-2"),
    )
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    knowledge = canonical_knowledge(3, value_chart="B")
    brain = VectorPpoSmallV1G1500Brain(seed=7)

    first = brain.choose_decision_with_history(context, knowledge, history)
    second = brain.choose_decision_with_history(context, knowledge, history)

    assert first == second
    context.validate(first)


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
