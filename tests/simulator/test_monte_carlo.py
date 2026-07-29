from __future__ import annotations

import sys

import pytest
from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks import simulator
from garboid_pocketrocks.bots import BotBrain, BotSpec, RandomBot
from garboid_pocketrocks.rules import LIVE_RULESET, RulesetKnowledge
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.runner import FaultMode
from garboid_pocketrocks.simulator.sampling import FixedRulesetSampler


def _random_spec(name: str = "random", bot_id: str = "random") -> BotSpec:
    return BotSpec(name, bot_id, RandomBot.build_brain)


def _small_random_config(
    *,
    games: int = 6,
    seed: int = 101,
    capture_replays: bool = False,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        bot_specs=tuple(_random_spec(f"random-{index}", f"random-{index}") for index in range(3)),
        games=games,
        player_counts=(3,),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        root_seed=seed,
        capture_replays=capture_replays,
    )


def test_public_simulator_api_exports_monte_carlo_and_sampling_types() -> None:
    assert simulator.MonteCarloRunner is MonteCarloRunner
    assert simulator.FixedRulesetSampler is FixedRulesetSampler


def test_worker_count_does_not_change_monte_carlo_result() -> None:
    config = _small_random_config()

    assert MonteCarloRunner.run(config, workers=1) == MonteCarloRunner.run(
        config,
        workers=2,
    )


def test_game_planning_is_reproducible_and_root_seeded() -> None:
    first = MonteCarloRunner.plan(_small_random_config(seed=7))
    second = MonteCarloRunner.plan(_small_random_config(seed=7))
    different = MonteCarloRunner.plan(_small_random_config(seed=8))

    assert first == second
    assert first != different
    assert [job.game_index for job in first] == list(range(6))


def test_aggregation_reconciles_games_seats_rulesets_and_decisions() -> None:
    config = _small_random_config(games=9)

    result = MonteCarloRunner.run(config)

    assert [game.game_index for game in result.game_summaries] == list(range(9))
    assert sum(statistics.games for statistics in result.bot_statistics) == 27
    for statistics in result.bot_statistics:
        assert statistics.games == sum(bucket.games for bucket in statistics.per_seat)
        assert statistics.games == sum(bucket.games for bucket in statistics.per_ruleset)
        assert statistics.games == sum(statistics.rank_counts)
        assert (
            statistics.mean_rank()
            == sum(rank * count for rank, count in enumerate(statistics.rank_counts, start=1))
            / statistics.games
        )
        assert all(margin <= 0 for margin in statistics.score_margins)
        assert statistics.decision_count == sum(
            bucket.decision_count for bucket in statistics.per_seat
        )
        assert statistics.faults == 0
        assert statistics.mean_final_money() == statistics.mean()
        assert statistics.median_final_money() == statistics.median()
        assert statistics.final_money_population_spread() == statistics.population_spread()
        assert statistics.final_money_quantile(0.5) == statistics.quantile(0.5)


def test_three_bots_occupy_each_seat_equally() -> None:
    result = MonteCarloRunner.run(_small_random_config(games=12))

    for statistics in result.bot_statistics:
        assert tuple(bucket.games for bucket in statistics.per_seat) == (4, 4, 4)


def test_same_seed_produces_equal_frozen_results() -> None:
    config = _small_random_config(games=3)

    assert MonteCarloRunner.run(config) == MonteCarloRunner.run(config)


def test_replay_capture_is_ordered_and_includes_provenance() -> None:
    captured = MonteCarloRunner.run(_small_random_config(games=3, seed=77, capture_replays=True))
    omitted = MonteCarloRunner.run(_small_random_config(games=1))

    assert [replay.game_index for replay in captured.replays] == [0, 1, 2]
    assert all(replay.root_seed == 77 for replay in captured.replays)
    assert omitted.replays == ()


def test_repeated_bot_identity_is_aggregated_once() -> None:
    repeated = _random_spec()
    config = MonteCarloConfig(
        bot_specs=(repeated, repeated, repeated),
        games=3,
        player_counts=(3,),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        root_seed=4,
    )

    result = MonteCarloRunner.run(config)

    assert len(result.bot_statistics) == 1
    assert result.bot_statistics[0].games == 9
    assert tuple(bucket.games for bucket in result.bot_statistics[0].per_seat) == (3, 3, 3)


def test_unpicklable_closure_works_serially_and_fails_clearly_in_workers() -> None:
    marker = object()

    def closure_factory(seed: int | None) -> BotBrain:
        assert marker is not None
        return RandomBot.build_brain(seed)

    closure_spec = BotSpec("closure", "closure", closure_factory)
    config = MonteCarloConfig(
        bot_specs=(closure_spec, _random_spec("one", "one"), _random_spec("two", "two")),
        games=1,
        player_counts=(3,),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        root_seed=9,
    )

    assert len(MonteCarloRunner.run(config, workers=1).game_summaries) == 1
    with pytest.raises(SimulationError, match="closure.*pickl"):
        MonteCarloRunner.run(config, workers=2)


def test_main_module_factory_fails_before_spawn_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _raising_brain
    monkeypatch.setattr(factory, "__module__", "__main__")
    monkeypatch.setattr(sys.modules["__main__"], factory.__name__, factory, raising=False)
    config = MonteCarloConfig(
        bot_specs=(
            BotSpec("main-factory", "main-factory", factory),
            _random_spec("one", "one"),
            _random_spec("two", "two"),
        ),
        games=1,
        player_counts=(3,),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        root_seed=9,
    )

    with pytest.raises(SimulationError, match="main-factory.*importable"):
        MonteCarloRunner.run(config, workers=2)


class RaisingBrain:
    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("brain exploded")


def _raising_brain(seed: int | None) -> RaisingBrain:
    del seed
    return RaisingBrain()


def test_faults_are_aggregated_for_the_responsible_bot() -> None:
    config = MonteCarloConfig(
        bot_specs=(
            BotSpec("raising", "raising", _raising_brain),
            _random_spec("one", "one"),
            _random_spec("two", "two"),
        ),
        games=1,
        player_counts=(3,),
        ruleset_sampler=FixedRulesetSampler(LIVE_RULESET),
        root_seed=3,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    result = MonteCarloRunner.run(config)
    raising = next(
        statistics for statistics in result.bot_statistics if statistics.bot_id == "raising"
    )

    assert raising.faults > 0
