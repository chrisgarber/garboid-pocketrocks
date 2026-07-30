from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from garboid_pocketrocks.simulator.runner import FaultMode
from garboid_pocketrocks.tournament.schedule import (
    TournamentConfig,
    TournamentPlanner,
)

from .helpers import random_specs


def test_default_config_describes_full_tournament() -> None:
    config = TournamentConfig(bot_specs=random_specs())

    assert config.games == 10_000
    assert config.player_counts == (3, 4, 5)
    assert config.charts == ("A", "B", "C", "D", "E")
    assert config.fault_mode is FaultMode.RECORD_AND_PASS
    assert config.bootstrap_samples == 200


def test_five_player_tournament_requires_five_distinct_bots() -> None:
    with pytest.raises(ValueError, match="5 distinct"):
        TournamentConfig(bot_specs=random_specs(4))


def test_config_rejects_duplicate_names_and_ids() -> None:
    first, second, third, fourth, fifth = random_specs()

    with pytest.raises(ValueError, match="names"):
        TournamentConfig(bot_specs=(first, replace(second, name=first.name), third, fourth, fifth))
    with pytest.raises(ValueError, match="IDs"):
        TournamentConfig(
            bot_specs=(first, replace(second, bot_id=first.bot_id), third, fourth, fifth)
        )


def test_default_plan_allocates_exactly_ten_thousand_games() -> None:
    plan = TournamentPlanner.plan(TournamentConfig(bot_specs=random_specs()))
    counts = tuple(quota.games for quota in plan.quotas)

    assert sum(counts) == 10_000
    assert max(counts) - min(counts) == 1
    assert len(plan.quotas) == 15
    assert len(plan.jobs) == 10_000


def test_plan_is_seeded_unique_and_condition_balanced() -> None:
    config = TournamentConfig(
        bot_specs=random_specs(8),
        games=150,
        root_seed=42,
        bootstrap_samples=0,
    )

    first = TournamentPlanner.plan(config)
    second = TournamentPlanner.plan(config)
    different = TournamentPlanner.plan(replace(config, root_seed=43))

    assert first == second
    assert first != different
    assert [job.game_index for job in first.jobs] == list(range(150))
    assert len({job.seed for job in first.jobs}) == 150
    assert all(len({spec.bot_id for spec in job.lineup}) == job.player_count for job in first.jobs)

    condition_appearances: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for job in first.jobs:
        condition = (job.ruleset.name.removeprefix("live-"), job.player_count)
        condition_appearances[condition].update(spec.bot_id for spec in job.lineup)
    for appearances in condition_appearances.values():
        counts = tuple(appearances[spec.bot_id] for spec in config.bot_specs)
        assert max(counts) - min(counts) <= 1


def test_seats_are_balanced_within_each_player_count() -> None:
    config = TournamentConfig(
        bot_specs=random_specs(7),
        games=210,
        root_seed=7,
        bootstrap_samples=0,
    )
    plan = TournamentPlanner.plan(config)
    seats: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    for job in plan.jobs:
        for seat, spec in enumerate(job.lineup):
            seats[job.player_count, spec.bot_id][seat] += 1

    for (player_count, _), counts in seats.items():
        values = tuple(counts[seat] for seat in range(player_count))
        assert max(values) - min(values) <= 1


def test_seat_repair_can_use_a_donor_with_one_extra_appearance() -> None:
    config = TournamentConfig(
        bot_specs=random_specs(5),
        games=31,
        root_seed=4,
        bootstrap_samples=0,
    )
    plan = TournamentPlanner.plan(config)
    seats: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    for job in plan.jobs:
        for seat, spec in enumerate(job.lineup):
            seats[job.player_count, spec.bot_id][seat] += 1

    for (player_count, _), counts in seats.items():
        values = tuple(counts[seat] for seat in range(player_count))
        assert max(values) - min(values) <= 1


def test_pair_exposures_reconcile_with_jobs() -> None:
    config = TournamentConfig(
        bot_specs=random_specs(6),
        games=30,
        bootstrap_samples=0,
    )
    plan = TournamentPlanner.plan(config)
    expected: Counter[tuple[str, str]] = Counter()
    for job in plan.jobs:
        ids = sorted(spec.bot_id for spec in job.lineup)
        expected.update(
            (left, right) for index, left in enumerate(ids) for right in ids[index + 1 :]
        )

    assert {
        (exposure.first_bot_id, exposure.second_bot_id): exposure.games
        for exposure in plan.pair_exposures
    } == expected


def test_pair_exposures_include_pairs_that_never_met() -> None:
    config = TournamentConfig(
        bot_specs=random_specs(20),
        games=15,
        bootstrap_samples=0,
    )

    plan = TournamentPlanner.plan(config)

    assert len(plan.pair_exposures) == 190
    assert min(exposure.games for exposure in plan.pair_exposures) == 0
