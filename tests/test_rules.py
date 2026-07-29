from collections.abc import Callable
from dataclasses import replace

import pytest
from pocketrocks import ActionId

from garboid_pocketrocks.rules import (
    LIVE_RULESET,
    VALUE_CHARTS,
    PlayerSetup,
    Ruleset,
    RulesetValidationError,
    live_ruleset,
)


def test_live_ruleset_matches_current_server_rules() -> None:
    assert LIVE_RULESET.resource_counts == (6, 6, 6, 6, 6)
    assert LIVE_RULESET.action_count(ActionId.AUCTION1) == 12
    assert LIVE_RULESET.action_count(ActionId.AUCTION2) == 8
    assert LIVE_RULESET.action_count(ActionId.LOAN10) == 3
    assert LIVE_RULESET.action_count(ActionId.LOAN20) == 2
    assert LIVE_RULESET.action_count(ActionId.INVEST5) == 3
    assert LIVE_RULESET.action_count(ActionId.INVEST10) == 2
    assert LIVE_RULESET.setup_for(3) == PlayerSetup(3, 30, 5)
    assert LIVE_RULESET.setup_for(4) == PlayerSetup(4, 25, 4)
    assert LIVE_RULESET.setup_for(5) == PlayerSetup(5, 20, 3)
    assert LIVE_RULESET.value_chart == VALUE_CHARTS["A"]
    assert len(LIVE_RULESET.objective_pool) == 30
    assert LIVE_RULESET.active_objective_count == 4


@pytest.mark.parametrize(
    ("invalid_ruleset", "message"),
    [
        (
            lambda ruleset: replace(ruleset, resource_counts=(6, 6)),
            "five resource counts",
        ),
        (
            lambda ruleset: replace(ruleset, action_counts=(12, 8)),
            "six action counts",
        ),
        (
            lambda ruleset: replace(ruleset, value_chart=(0, 4)),
            "six value-chart buckets",
        ),
        (
            lambda ruleset: replace(ruleset, active_objective_count=31),
            "active objective count",
        ),
    ],
)
def test_ruleset_rejects_invalid_shapes(
    invalid_ruleset: Callable[[Ruleset], Ruleset],
    message: str,
) -> None:
    with pytest.raises(RulesetValidationError, match=message):
        invalid_ruleset(LIVE_RULESET)


def test_ruleset_rejects_insufficient_auction_capacity() -> None:
    with pytest.raises(RulesetValidationError, match="auction capacity"):
        replace(
            LIVE_RULESET,
            action_counts=(1, 0, 0, 0, 0, 0),
        )


def test_ruleset_knowledge_resolves_player_setup() -> None:
    knowledge = LIVE_RULESET.knowledge(4)
    assert knowledge.player_count == 4
    assert knowledge.starting_cash == 25
    assert knowledge.private_cards_per_player == 4
    assert knowledge.resource_counts == LIVE_RULESET.resource_counts
    assert knowledge.action_counts == LIVE_RULESET.action_counts


def test_disabling_objectives_also_disables_active_selection() -> None:
    ruleset = live_ruleset(objectives_enabled=False)

    assert not ruleset.objectives_enabled
    assert ruleset.active_objective_count == 0
