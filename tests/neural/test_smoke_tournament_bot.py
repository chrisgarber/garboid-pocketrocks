from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint  # noqa: E402
from garboid_pocketrocks.neural.tournament_bot import (  # noqa: E402
    SMOKE_BOT_NAME,
    SMOKE_CHECKPOINT_PATH,
)


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
