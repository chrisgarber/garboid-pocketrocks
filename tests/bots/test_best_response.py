from __future__ import annotations

import inspect

import pytest

from garboid_pocketrocks.bots.best_response import (
    MONTE_CARLO_V1_BOT_SPEC,
    MonteCarloV1Brain,
)
from garboid_pocketrocks.bots.registry import BOT_SPECS_BY_NAME
from garboid_pocketrocks.heuristics.montecarlo import MONTE_CARLO_V1
from garboid_pocketrocks.simulator import MonteCarloConfig, MonteCarloRunner

FIELD = ("aggressive-v3", "balanced-v3", "passive-v3", "fixed-objective-overlay-v2")


def _run(
    *,
    players: int = 4,
    games: int = 48,
    seed: int = 99,
    workers: int = 1,
    batch_size: int | None = None,
    charts: tuple[str, ...] = ("A", "E"),
    objectives: tuple[bool, ...] = (True, False),
) -> tuple[tuple[str, int, int], ...]:
    config = MonteCarloConfig(
        bot_specs=(MONTE_CARLO_V1_BOT_SPEC,)
        + tuple(BOT_SPECS_BY_NAME[name] for name in FIELD[: players - 1]),
        games=games,
        player_counts=(players,),
        value_charts=charts,
        objectives_enabled=objectives,
        root_seed=seed,
    )
    result = MonteCarloRunner.run(config, workers=workers, batch_size=batch_size)
    return tuple(
        (stats.bot_name, stats.outright_wins, stats.faults) for stats in result.bot_statistics
    )


@pytest.mark.parametrize("players", [3, 4, 5])
def test_no_faults_across_players_charts_and_objectives(players: int) -> None:
    """FaultMode.RAISE is the default, so an illegal decision surfaces as a fault."""

    for _name, _wins, faults in _run(players=players):
        assert faults == 0


def test_brain_is_history_aware() -> None:
    """The opponent model needs resolved bids, which only public history carries."""

    brain = MONTE_CARLO_V1_BOT_SPEC.make_brain(seed=3)
    assert "history" in inspect.signature(brain.choose_decision).parameters


def test_seed_reproducible() -> None:
    """A sampling brain must derive its randomness from the supplied seed."""

    assert _run(games=40, seed=1234) == _run(games=40, seed=1234)


def test_scalar_and_batch_paths_agree() -> None:
    """Scalar execution is the behavioural oracle for the batch engine."""

    assert _run(games=40, seed=77, batch_size=None) == _run(games=40, seed=77, batch_size=8)


def test_spec_survives_spawn_workers() -> None:
    """The brain factory must be importable by process workers."""

    assert _run(games=24, workers=2, batch_size=4)


def test_released_settings_use_every_searched_factor() -> None:
    """Each factor was searched from zero; a zero here means one stopped earning."""

    settings = MONTE_CARLO_V1
    assert settings.scarcity_weight > 0.0
    assert settings.denial_weight > 0.0
    assert settings.pressure_weight > 0.0
    assert settings.standings_weight > 0.0
    assert settings.joint_sampling is True
    assert settings.profile.bid_shading == 0.0


def test_released_generation_is_the_registered_one() -> None:
    spec = BOT_SPECS_BY_NAME["monte-the-bookie-v1"]
    assert spec.bot_id == "monte-the-bookie-v1"
    assert isinstance(spec.make_brain(seed=0), MonteCarloV1Brain)
