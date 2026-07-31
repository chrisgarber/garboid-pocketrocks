from __future__ import annotations

from dataclasses import replace

import pytest
from pocketrocks import ActionId

from garboid_pocketrocks.heuristics.phases import (
    PublicResourceHorizon,
    public_resource_horizon,
    select_expert_phase,
)
from garboid_pocketrocks.knowledge import canonical_knowledge

from .helpers import make_context


def _won_resources(
    player_count: int,
    resource_count: int,
) -> tuple[tuple[int, ...], ...]:
    cells = [0] * (player_count * 5)
    for index in range(resource_count):
        cells[index % len(cells)] += 1
    return tuple(tuple(cells[seat * 5 : (seat + 1) * 5]) for seat in range(player_count))


@pytest.mark.parametrize(
    ("future", "total", "expected"),
    (
        (10, 15, "early"),
        (9, 15, "middle"),
        (5, 15, "middle"),
        (4, 15, "late"),
        (10, 14, "early"),
        (4, 14, "late"),
    ),
)
def test_expert_phase_uses_inclusive_resource_thirds(
    future: int,
    total: int,
    expected: str,
) -> None:
    horizon = PublicResourceHorizon(
        total_biddable_resources=total,
        future_biddable_resources=future,
    )

    assert select_expert_phase(horizon) == expected


@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_public_horizon_is_independent_of_private_hand(
    player_count: int,
) -> None:
    ruleset = canonical_knowledge(player_count)
    total = sum(ruleset.resource_counts) - (player_count * ruleset.private_cards_per_player)
    context = make_context(
        player_count=player_count,
        starting_cash=ruleset.starting_cash,
        value_chart=ruleset.value_chart,
        cash=(ruleset.starting_cash,) * player_count,
        won=_won_resources(player_count, total - 11),
        revealed=((0, 0, 0, 0, 0),) * player_count,
        owned_objectives=((),) * player_count,
        current_resources=(1, 2),
        hand=(1,),
    )
    changed_private_hand = replace(context, current_hand_suit_ids=(2, 3, 4, 5))

    expected = PublicResourceHorizon(
        total_biddable_resources=total,
        future_biddable_resources=10,
    )
    assert public_resource_horizon(context, ruleset) == expected
    assert public_resource_horizon(changed_private_hand, ruleset) == expected
    assert select_expert_phase(expected) == "early"


def test_current_action_controls_which_visible_resources_leave_the_horizon() -> None:
    ruleset = canonical_knowledge(3)
    won_before_auction = _won_resources(3, 3)
    auction_one = make_context(
        action_id=ActionId.AUCTION1,
        current_resources=(1, 2),
        won=won_before_auction,
        hand=(3, 4, 5),
    )
    auction_two = replace(auction_one, current_action_id=int(ActionId.AUCTION2))
    auction_two_at_deck_tail = replace(
        auction_two,
        current_resource_ids=(1, 0),
    )
    loan = replace(auction_one, current_action_id=int(ActionId.LOAN10))
    reveal_after_auction_one = replace(
        auction_one,
        decision_kind="selectInfoToReveal",
        won_resource_counts_by_seat=_won_resources(3, 4),
    )

    assert (
        public_resource_horizon(
            auction_one,
            ruleset,
        ).future_biddable_resources
        == 11
    )
    assert (
        public_resource_horizon(
            auction_two,
            ruleset,
        ).future_biddable_resources
        == 10
    )
    assert (
        public_resource_horizon(
            auction_two_at_deck_tail,
            ruleset,
        ).future_biddable_resources
        == 11
    )
    assert public_resource_horizon(loan, ruleset).future_biddable_resources == 12
    assert (
        public_resource_horizon(
            reveal_after_auction_one,
            ruleset,
        ).future_biddable_resources
        == 11
    )


@pytest.mark.parametrize(
    ("context_change", "ruleset_change", "message"),
    (
        ({"player_count": 4}, {}, "player count"),
        (
            {"won_resource_counts_by_seat": ((0, 0, 0, 0, 0),) * 2},
            {},
            "rows",
        ),
        (
            {
                "won_resource_counts_by_seat": (
                    (0, 0, 0, 0),
                    (0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0),
                )
            },
            {},
            "width",
        ),
        ({}, {"resource_counts": (0, 0, 0, 0, 0)}, "total"),
        (
            {"won_resource_counts_by_seat": ((6, 6, 6, 6, 6),) * 3},
            {},
            "future",
        ),
        ({"decision_kind": "unknown"}, {}, "decision kind"),
        ({"current_action_id": True}, {}, "current action"),
    ),
)
def test_public_horizon_rejects_inconsistent_public_inputs(
    context_change: dict[str, object],
    ruleset_change: dict[str, object],
    message: str,
) -> None:
    context = replace(make_context(), **context_change)
    ruleset = replace(canonical_knowledge(3), **ruleset_change)

    with pytest.raises(ValueError, match=message):
        public_resource_horizon(context, ruleset)


@pytest.mark.parametrize(
    ("future", "total"),
    ((0, 0), (-1, 15), (16, 15)),
)
def test_resource_horizon_rejects_impossible_bounds(
    future: int,
    total: int,
) -> None:
    with pytest.raises(ValueError):
        PublicResourceHorizon(
            total_biddable_resources=total,
            future_biddable_resources=future,
        )
