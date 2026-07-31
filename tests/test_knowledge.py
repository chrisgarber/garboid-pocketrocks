from dataclasses import replace

import pytest
from pocketrocks import OBJECTIVES, DecisionContext
from pocketrocks.sim.constants import VALUE_CHARTS

from garboid_pocketrocks.knowledge import (
    canonical_knowledge,
    knowledge_for_context,
    ruleset_name,
    value_chart_from_ruleset_name,
)


def test_ruleset_names_are_canonical_sdk_boundaries() -> None:
    assert ruleset_name("a") == "live-A"
    assert ruleset_name("E", objectives_enabled=False) == "live-E-no-objectives"
    assert value_chart_from_ruleset_name("live-A") == "A"
    assert value_chart_from_ruleset_name("live-E-no-objectives") == "E"


@pytest.mark.parametrize("name", ("A", "live-Z", "live-", ""))
def test_unknown_ruleset_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="ruleset"):
        value_chart_from_ruleset_name(name)


@pytest.mark.parametrize("player_count", (3, 4, 5))
@pytest.mark.parametrize("chart", ("A", "B", "C", "D", "E"))
def test_canonical_knowledge_matches_sdk_variants(
    player_count: int,
    chart: str,
) -> None:
    knowledge = canonical_knowledge(player_count, value_chart=chart)

    assert knowledge.player_count == player_count
    assert knowledge.name == f"live-{chart}"
    assert knowledge.value_chart == VALUE_CHARTS[chart]
    assert knowledge.resource_counts == (6, 6, 6, 6, 6)
    assert knowledge.action_counts == (12, 8, 3, 2, 3, 2)
    assert knowledge.objective_pool == tuple(sorted(OBJECTIVES))
    assert knowledge.active_objective_count == 4
    assert knowledge.objectives_enabled


def test_canonical_knowledge_can_disable_objectives() -> None:
    knowledge = canonical_knowledge(3, value_chart="E", objectives_enabled=False)

    assert knowledge.name == "live-E-no-objectives"
    assert knowledge.active_objective_count == 0
    assert not knowledge.objectives_enabled


def test_knowledge_for_context_reconciles_sdk_visible_fields() -> None:
    context = DecisionContext(
        request_id="knowledge-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="submitBid",
        player_count=4,
        starting_cash=27,
        value_chart=VALUE_CHARTS["C"],
        objective_ids=(1, 2, 3, 4),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=(27, 27, 27, 27),
        tiebreak_seat=3,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 4,
        revealed_info_counts_by_seat=(
            (1, 0, 1, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
        owned_objective_ids_by_seat=((), (), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(2, 3),
        legal_max_amount=27,
        revealable_count=2,
    )

    knowledge = knowledge_for_context(context)

    assert knowledge.player_count == 4
    assert knowledge.starting_cash == 27
    assert knowledge.private_cards_per_player == 4
    assert knowledge.value_chart == VALUE_CHARTS["C"]
    assert knowledge.active_objective_count == 4
    assert knowledge.objectives_enabled


@pytest.mark.parametrize("player_count", (2, 6))
def test_canonical_knowledge_rejects_unsupported_player_counts(
    player_count: int,
) -> None:
    with pytest.raises(ValueError, match="3-5"):
        canonical_knowledge(player_count)


def test_knowledge_for_context_rejects_unknown_chart() -> None:
    context = DecisionContext(
        request_id="unknown-chart",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=(1, 2, 3, 4, 5, 6),
        objective_ids=(),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=(30, 30, 30),
        tiebreak_seat=0,
        won_resource_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        revealed_info_counts_by_seat=((0, 0, 0, 0, 0),) * 3,
        owned_objective_ids_by_seat=((), (), ()),
        bot_seat=0,
        current_hand_suit_ids=(),
        legal_max_amount=30,
        revealable_count=0,
    )

    with pytest.raises(ValueError, match="unknown value chart"):
        knowledge_for_context(context)


def test_knowledge_is_immutable() -> None:
    knowledge = canonical_knowledge(3)

    changed = replace(knowledge, name="test")

    assert changed.name == "test"
    assert knowledge.name == "live-A"
