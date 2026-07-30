from __future__ import annotations

import sys
from dataclasses import replace

import pytest
from pocketrocks import BotDecision, DecisionContext
from pocketrocks.sim.constants import ACTION_WIRE_IDS

from garboid_pocketrocks import simulator
from garboid_pocketrocks.bots import BotBrain, BotSpec, RandomBot
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.simulator import monte_carlo
from garboid_pocketrocks.simulator.errors import SimulationError
from garboid_pocketrocks.simulator.monte_carlo import (
    MonteCarloConfig,
    MonteCarloRunner,
)
from garboid_pocketrocks.simulator.runner import FaultMode


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
        value_charts=("A",),
        root_seed=seed,
        capture_replays=capture_replays,
    )


def test_public_simulator_api_exports_monte_carlo() -> None:
    assert simulator.MonteCarloRunner is MonteCarloRunner


def test_sdk_variant_configuration_requires_value_charts() -> None:
    with pytest.raises(ValueError, match="value_charts"):
        replace(_small_random_config(games=0), value_charts=())


def test_sdk_variant_configuration_rejects_unknown_chart() -> None:
    with pytest.raises(ValueError, match="A-E"):
        replace(_small_random_config(games=0), value_charts=("Z",))


def test_sdk_variant_configuration_requires_objective_modes() -> None:
    with pytest.raises(ValueError, match="objectives_enabled"):
        replace(_small_random_config(games=0), objectives_enabled=())


def test_sdk_variant_configuration_requires_boolean_objective_modes() -> None:
    with pytest.raises(ValueError, match="booleans"):
        replace(
            _small_random_config(games=0),
            objectives_enabled=(True, 1),  # type: ignore[arg-type]
        )


def test_sdk_variant_planning_is_deterministic_and_normalizes_charts() -> None:
    config = replace(
        _small_random_config(games=40, seed=77),
        value_charts=("a", "E"),
        objectives_enabled=(True, False),
    )

    first = MonteCarloRunner.plan(config)
    second = MonteCarloRunner.plan(config)

    assert first == second
    assert config.value_charts == ("A", "E")
    assert {(job.value_chart, job.objectives_enabled) for job in first} == {
        ("A", True),
        ("A", False),
        ("E", True),
        ("E", False),
    }


def test_worker_count_does_not_change_monte_carlo_result() -> None:
    config = _small_random_config()

    assert MonteCarloRunner.run(config, workers=1) == MonteCarloRunner.run(
        config,
        workers=2,
    )


def test_run_jobs_executes_a_valid_explicit_plan() -> None:
    config = _small_random_config()
    jobs = MonteCarloRunner.plan(config)

    assert MonteCarloRunner.run_jobs(config, jobs) == MonteCarloRunner.run(config)


def test_run_jobs_rejects_wrong_job_count() -> None:
    config = _small_random_config()
    jobs = MonteCarloRunner.plan(config)

    with pytest.raises(ValueError, match="job count"):
        MonteCarloRunner.run_jobs(config, jobs[:-1])


def test_run_jobs_rejects_noncontiguous_game_indices() -> None:
    config = _small_random_config()
    jobs = MonteCarloRunner.plan(config)
    invalid = (replace(jobs[0], game_index=99), *jobs[1:])

    with pytest.raises(ValueError, match="game indices"):
        MonteCarloRunner.run_jobs(config, invalid)


def test_run_jobs_rejects_a_different_root_seed() -> None:
    config = _small_random_config()
    jobs = MonteCarloRunner.plan(config)
    invalid = (replace(jobs[0], root_seed=999), *jobs[1:])

    with pytest.raises(ValueError, match="root seed"):
        MonteCarloRunner.run_jobs(config, invalid)


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
        value_charts=("A",),
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
        value_charts=("A",),
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
        value_charts=("A",),
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
        value_charts=("A",),
        root_seed=3,
        fault_mode=FaultMode.RECORD_AND_PASS,
    )

    result = MonteCarloRunner.run(config)
    raising = next(
        statistics for statistics in result.bot_statistics if statistics.bot_id == "raising"
    )

    assert raising.faults > 0


class ScriptedMetricsBrain:
    def __init__(self, bidding_decision: str) -> None:
        self.bidding_decision = bidding_decision

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            return (
                BotDecision.pass_turn()
                if self.bidding_decision == "pass"
                else BotDecision.select_info_to_reveal(0)
            )
        if self.bidding_decision == "pass":
            return BotDecision.pass_turn()
        if self.bidding_decision == "zero":
            return BotDecision.submit_bid(0)
        legal_max_amount = context.legal_max_amount
        assert legal_max_amount is not None
        return BotDecision.submit_bid(min(1, legal_max_amount))


def _pass_metrics_brain(seed: int | None) -> ScriptedMetricsBrain:
    del seed
    return ScriptedMetricsBrain("pass")


def _zero_metrics_brain(seed: int | None) -> ScriptedMetricsBrain:
    del seed
    return ScriptedMetricsBrain("zero")


def _one_metrics_brain(seed: int | None) -> ScriptedMetricsBrain:
    del seed
    return ScriptedMetricsBrain("one")


def _scripted_metrics_config(*, games: int = 1) -> MonteCarloConfig:
    return MonteCarloConfig(
        bot_specs=(
            BotSpec("pass", "pass", _pass_metrics_brain),
            BotSpec("zero", "zero", _zero_metrics_brain),
            BotSpec("one", "one", _one_metrics_brain),
        ),
        games=games,
        player_counts=(3,),
        value_charts=("A",),
        root_seed=31337,
        capture_replays=True,
    )


def test_behavior_statistics_defines_empty_rates_and_six_action_buckets() -> None:
    behavior = monte_carlo.BehaviorStatistics(
        bidding_requests=0,
        passes=0,
        nonzero_bids=(),
        reveal_choices=(),
        wins_by_action=(0, 0, 0, 0, 0, 0),
        resource_cards_won=0,
        objectives_claimed=0,
    )

    assert behavior.pass_rate() == 0.0
    assert behavior.mean_nonzero_bid() == 0.0
    assert len(behavior.wins_by_action) == 6


def test_pass_and_zero_bid_are_bidding_passes_but_reveals_are_excluded() -> None:
    result = MonteCarloRunner.run(_scripted_metrics_config())
    by_id = {statistics.bot_id: statistics for statistics in result.bot_statistics}
    pass_behavior = by_id["pass"].behavior
    zero_behavior = by_id["zero"].behavior
    one_behavior = by_id["one"].behavior

    assert pass_behavior.bidding_requests > 0
    assert pass_behavior.passes == pass_behavior.bidding_requests
    assert zero_behavior.passes == zero_behavior.bidding_requests
    assert zero_behavior.nonzero_bids == ()
    assert one_behavior.bidding_requests == pass_behavior.bidding_requests
    assert one_behavior.passes < one_behavior.bidding_requests
    assert pass_behavior.reveal_choices == ()
    assert all(choice == 0 for choice in zero_behavior.reveal_choices)
    assert (
        sum(len(decisions) for _, decisions in result.replays[0].decisions if len(decisions) == 1)
        > 0
    )
    assert sum(statistics.behavior.bidding_requests for statistics in result.bot_statistics) == sum(
        len(decisions)
        for _, decisions in result.replays[0].decisions
        if len(decisions) == result.replays[0].player_count
    )


def test_behavior_wins_resources_and_objectives_match_exact_events() -> None:
    result = MonteCarloRunner.run(_scripted_metrics_config())
    replay = result.replays[0]
    bot_ids_by_seat = result.game_summaries[0].bot_ids

    expected_wins = {bot_id: [0] * 6 for bot_id in bot_ids_by_seat}
    expected_resources = {bot_id: 0 for bot_id in bot_ids_by_seat}
    expected_objectives = {bot_id: 0 for bot_id in bot_ids_by_seat}
    for turn in replay.turns:
        bot_id = bot_ids_by_seat[turn.winner_seat]
        expected_wins[bot_id][ACTION_WIRE_IDS[turn.action] - 1] += 1
        expected_resources[bot_id] += len(turn.bundle_suits)
        expected_objectives[bot_id] += len(turn.claimed_objective_wire_ids)

    for statistics in result.bot_statistics:
        behavior = statistics.behavior
        assert len(behavior.wins_by_action) == 6
        assert behavior.wins_by_action == tuple(expected_wins[statistics.bot_id])
        assert behavior.resource_cards_won == expected_resources[statistics.bot_id]
        assert behavior.objectives_claimed == expected_objectives[statistics.bot_id]
    assert sum(sum(values) for values in expected_wins.values()) > 0
    assert sum(expected_resources.values()) > 0
    assert sum(expected_objectives.values()) > 0


def test_duplicate_bot_ids_aggregate_behavior() -> None:
    repeated = BotSpec("pass", "pass", _pass_metrics_brain)
    config = MonteCarloConfig(
        bot_specs=(repeated, repeated, repeated),
        games=2,
        player_counts=(3,),
        value_charts=("A",),
        root_seed=41,
    )

    statistics = MonteCarloRunner.run(config).bot_statistics[0]

    assert statistics.games == 6
    assert statistics.behavior.bidding_requests > 0
    assert statistics.behavior.passes == statistics.behavior.bidding_requests
    assert sum(statistics.behavior.wins_by_action) > 0


def test_worker_count_preserves_behavior_statistics() -> None:
    config = _scripted_metrics_config(games=3)

    serial = MonteCarloRunner.run(config, workers=1)
    parallel = MonteCarloRunner.run(config, workers=2)

    assert serial == parallel
    assert all(statistics.behavior.bidding_requests > 0 for statistics in parallel.bot_statistics)
