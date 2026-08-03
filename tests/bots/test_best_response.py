from __future__ import annotations

import inspect

import pytest

from garboid_pocketrocks.bots.best_response import (
    MONTE_CARLO_V1_BOT_SPEC,
    MONTE_CARLO_V2_BOT_SPEC,
    MonteCarloV1Brain,
)
from garboid_pocketrocks.bots.registry import BOT_SPECS_BY_NAME
from garboid_pocketrocks.heuristics.bid_priors import BID_PRIOR_V2
from garboid_pocketrocks.heuristics.montecarlo import MONTE_CARLO_V1, MONTE_CARLO_V2
from garboid_pocketrocks.simulator import MonteCarloConfig, MonteCarloRunner

FIELD = ("surplus-v10", "fixed-objective-overlay-v3", "aggressive-v3", "surplus-v9")


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


def test_v1_generation_stays_frozen() -> None:
    """v1 is released. Its coefficients and prior must not drift when v2 lands."""

    assert MONTE_CARLO_V1.profile.liquidity_strength == pytest.approx(3.62309)
    assert MONTE_CARLO_V1.standings_weight == pytest.approx(0.39481)
    assert MONTE_CARLO_V1.denial_weight == pytest.approx(0.45427)
    assert MONTE_CARLO_V1.prior is None  # v1 uses the shipped BID_PRIOR_V1


def test_v2_retunes_every_factor_and_carries_the_v2_prior() -> None:
    assert MONTE_CARLO_V2.prior is BID_PRIOR_V2
    # Retuning for placement rather than wins pulled the speculative terms down.
    assert MONTE_CARLO_V2.denial_weight < MONTE_CARLO_V1.denial_weight
    assert MONTE_CARLO_V2.pressure_weight < MONTE_CARLO_V1.pressure_weight
    assert MONTE_CARLO_V2.standings_weight < MONTE_CARLO_V1.standings_weight
    assert MONTE_CARLO_V2.profile.objective_progress_weight == 0.0
    # And pushed the term that protects the floor up.
    assert MONTE_CARLO_V2.profile.liquidity_strength > MONTE_CARLO_V1.profile.liquidity_strength
    assert MONTE_CARLO_V2.profile.bid_shading == 0.0


def test_v2_plays_the_current_strong_field_without_faults() -> None:
    config = MonteCarloConfig(
        bot_specs=(MONTE_CARLO_V2_BOT_SPEC,) + tuple(BOT_SPECS_BY_NAME[name] for name in FIELD[:3]),
        games=48,
        player_counts=(4,),
        value_charts=("A", "E"),
        objectives_enabled=(True,),
        root_seed=4242,
    )
    result = MonteCarloRunner.run(config, workers=1)
    for stats in result.bot_statistics:
        assert stats.faults == 0
