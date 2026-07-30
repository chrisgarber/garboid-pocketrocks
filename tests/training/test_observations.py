from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pocketrocks import DecisionContext

from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.training.bounds import EnvironmentBounds
from garboid_pocketrocks.training.observations import ObservationEncoder


def _context(*, decision_kind: str = "submitBid") -> DecisionContext:
    return DecisionContext(
        request_id="test",
        deadline_at=0,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(1, 10),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=(30, 27, 25),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((1, 0, 0, 0, 0), (0, 2, 0, 0, 0), (0, 0, 3, 0, 0)),
        revealed_info_counts_by_seat=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)),
        owned_objective_ids_by_seat=((1,), (), (10,)),
        bot_seat=1,
        current_hand_suit_ids=(2, 5),
        legal_max_amount=27 if decision_kind == "submitBid" else None,
        revealable_count=2,
    )


def test_observation_space_has_public_fixed_keys_and_contains_bid_and_reveal() -> None:
    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))
    knowledge = canonical_knowledge(3)

    assert set(encoder.observation_space.spaces) == {
        "phase",
        "player_count",
        "bot_seat",
        "starting_cash",
        "value_chart",
        "active_objectives",
        "current_action",
        "current_resources",
        "cash_by_seat",
        "priority_seat",
        "won_resources",
        "revealed_info",
        "owned_objectives",
        "private_hand",
        "rules_resource_counts",
        "rules_action_counts",
        "rules_private_cards",
        "rules_objective_pool",
        "rules_active_objective_count",
        "rules_objectives_enabled",
        "action_mask",
    }

    bid = encoder.encode(_context(), knowledge)
    reveal = encoder.encode(_context(decision_kind="selectInfoToReveal"), knowledge)
    assert encoder.observation_space.contains(bid)
    assert encoder.observation_space.contains(reveal)
    assert bid["cash_by_seat"].dtype == np.int16
    assert bid["action_mask"].dtype == np.int8
    assert bid["won_resources"].shape == (5, 5)
    assert bid["owned_objectives"].shape == (5, 30)


def test_observation_uses_no_hidden_engine_state() -> None:
    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))
    context = _context()
    knowledge = canonical_knowledge(3)

    baseline = encoder.encode(context, knowledge)
    # Private opponent cards and deck order are unavailable in DecisionContext,
    # so independently changing hypothetical values cannot affect its encoding.
    changed_hidden_state = {"opponent_hand": (5, 5, 5), "deck": (4, 3, 2, 1)}
    assert changed_hidden_state != {"opponent_hand": (1,), "deck": (1, 2, 3, 4)}
    repeated = encoder.encode(context, knowledge)
    assert all(np.array_equal(repeated[key], baseline[key]) for key in baseline)


def test_observation_conditions_on_public_ruleset_knowledge() -> None:
    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))
    context = _context()
    knowledge = canonical_knowledge(3)

    baseline = encoder.encode(context, knowledge)
    changed = encoder.encode(
        context,
        replace(knowledge, action_counts=(11, 8, 3, 2, 3, 2)),
    )

    assert not np.array_equal(baseline["rules_action_counts"], changed["rules_action_counts"])


def test_hidden_public_ruleset_field_is_zero_filled_only_for_that_field() -> None:
    encoder = ObservationEncoder(
        EnvironmentBounds(max_bid=100, max_hand_size=5),
        hidden_ruleset_fields=frozenset({"rules_action_counts"}),
    )
    encoded = encoder.encode(_context(), canonical_knowledge(3))

    assert np.array_equal(encoded["rules_action_counts"], np.zeros(6, dtype=np.int16))
    assert np.any(encoded["rules_resource_counts"])


def test_encoder_rejects_incompatible_bounds_and_unknown_hidden_fields() -> None:
    with pytest.raises(ValueError, match="hidden ruleset"):
        ObservationEncoder(
            EnvironmentBounds(max_bid=100, max_hand_size=5),
            hidden_ruleset_fields=frozenset({"not_a_field"}),
        )

    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))
    with pytest.raises(ValueError, match="legal maximum"):
        encoder.encode(
            replace(_context(), legal_max_amount=101),
            canonical_knowledge(3),
        )


def test_observation_accepts_negative_chart_values_within_int16() -> None:
    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))
    context = replace(_context(), value_chart=(-1, 4, 8, 12, 16, 20))

    encoded = encoder.encode(
        context,
        replace(canonical_knowledge(3), value_chart=context.value_chart),
    )

    assert encoder.observation_space.contains(encoded)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (replace(_context(), value_chart=(0, 4, 8, 12, 16, 40_000)), "value chart"),
        (replace(_context(), cash_by_seat=(30, 40_000, 25)), "cash"),
        (
            replace(
                _context(),
                won_resource_counts_by_seat=((40_000, 0, 0, 0, 0),) * 3,
            ),
            "won resource",
        ),
    ],
)
def test_observation_rejects_values_outside_numeric_dtype_bounds(
    context: DecisionContext,
    message: str,
) -> None:
    encoder = ObservationEncoder(EnvironmentBounds(max_bid=100, max_hand_size=5))

    with pytest.raises(ValueError, match=message):
        encoder.encode(context, canonical_knowledge(3))
