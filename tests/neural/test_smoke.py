from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from garboid_pocketrocks.neural.run_config import ParallelConfig  # noqa: E402
from garboid_pocketrocks.neural.smoke import (  # noqa: E402
    SmokeResult,
    run_smoke,
    smoke_run_config,
)


@pytest.mark.neural_smoke
def test_full_curriculum_smoke_contract_at_one_game_per_cell(
    tmp_path: Path,
) -> None:
    result: SmokeResult = run_smoke(
        replace(
            smoke_run_config(),
            games_per_cell=1,
            device="cpu",
            parallel=ParallelConfig(
                workers=2,
                active_games_per_worker=4,
                max_inference_batch=64,
            ),
        ),
        tmp_path / "self-play",
    )

    assert result.completed_episodes == 15
    assert result.completed_updates == 1
    assert result.checkpoint_digest_verified
    assert result.resume_verified
    assert {games for _, _, games in result.cell_games} == {1}
    assert result.games_per_second > 0.0
    assert result.decisions_per_second > 0.0
    assert result.value.count > 0
