from __future__ import annotations

import math
from dataclasses import replace

import pytest
from pocketrocks import ActionId, DecisionContext

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V3  # noqa: E402
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator  # noqa: E402
from garboid_pocketrocks.knowledge import (  # noqa: E402
    RulesetKnowledge,
    canonical_knowledge,
)
from garboid_pocketrocks.neural.heuristic_auxiliary import (  # noqa: E402
    HEURISTIC_AUXILIARY_TARGET_NAME,
    HeuristicAuxiliaryError,
    HeuristicAuxiliaryValueConfig,
    heuristic_auxiliary_label,
    masked_heuristic_smooth_l1_loss,
)


def _context(*, decision_kind: str = "submitBid") -> DecisionContext:
    knowledge = canonical_knowledge(3)
    is_bid = decision_kind == "submitBid"
    return DecisionContext(
        request_id="heuristic-auxiliary-test",
        deadline_at=0,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=knowledge.starting_cash,
        value_chart=knowledge.value_chart,
        objective_ids=(1, 2, 3, 4),
        current_action_id=int(ActionId.AUCTION1) if is_bid else None,
        current_resource_ids=(1, 0) if is_bid else (0, 0),
        cash_by_seat=(30, 27, 25),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((1, 0, 0, 0, 0), (0, 2, 0, 0, 0), (0, 0, 3, 0, 0)),
        revealed_info_counts_by_seat=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)),
        owned_objective_ids_by_seat=((1,), (2,), (3,)),
        bot_seat=1,
        current_hand_suit_ids=(2, 5, 1, 4),
        legal_max_amount=7 if is_bid else None,
        revealable_count=4,
    )


def _ruleset(context: DecisionContext) -> RulesetKnowledge:
    return replace(
        canonical_knowledge(context.player_count),
        active_objective_count=len(context.objective_ids),
    )


def test_config_is_exact_json_and_pins_the_balanced_v3_target() -> None:
    config = HeuristicAuxiliaryValueConfig.balanced_v3()

    assert config.target == HEURISTIC_AUXILIARY_TARGET_NAME
    assert config.to_json_dict() == {
        "target": "balanced-v3-bid-win-delta-v1",
        "loss_coefficient": 0.1,
        "smooth_l1_delta": 0.25,
        "teacher_profile_digest": (
            "e3971899626ca3f651b2992d0cc429dc3ffd57fcdbb7cfac8249e6f0f9d9b03e"
        ),
    }
    assert HeuristicAuxiliaryValueConfig.from_json_dict(config.to_json_dict()) == config

    with pytest.raises(HeuristicAuxiliaryError, match="keys must be exact"):
        HeuristicAuxiliaryValueConfig.from_json_dict(
            {**config.to_json_dict(), "reward_coefficient": 1.0}
        )
    with pytest.raises(HeuristicAuxiliaryError, match="zero loss coefficient"):
        HeuristicAuxiliaryValueConfig(target="disabled", loss_coefficient=0.1)
    with pytest.raises(HeuristicAuxiliaryError, match="positive loss coefficient"):
        HeuristicAuxiliaryValueConfig(target=HEURISTIC_AUXILIARY_TARGET_NAME)
    with pytest.raises(HeuristicAuxiliaryError, match="must be finite"):
        HeuristicAuxiliaryValueConfig.balanced_v3(loss_coefficient=math.nan)
    with pytest.raises(HeuristicAuxiliaryError, match="profile digest"):
        HeuristicAuxiliaryValueConfig.balanced_v3(
            teacher_profile_digest="0" * 64,
        )


def test_bid_label_is_balanced_v3_bid_zero_win_delta_scaled_by_starting_cash() -> None:
    context = _context()
    ruleset = _ruleset(context)
    config = HeuristicAuxiliaryValueConfig.balanced_v3()

    first = heuristic_auxiliary_label(context, ruleset, config)
    second = heuristic_auxiliary_label(context, ruleset, config)
    evaluation = HeuristicValuator(HEURISTIC_V3.balanced).evaluate_bid(context, ruleset)

    assert first == second
    assert first.included is True
    assert first.target == pytest.approx(evaluation.points[0].win_delta / context.starting_cash)


def test_reveal_and_disabled_targets_are_masked() -> None:
    reveal = _context(decision_kind="selectInfoToReveal")
    enabled = heuristic_auxiliary_label(
        reveal,
        _ruleset(reveal),
        HeuristicAuxiliaryValueConfig.balanced_v3(),
    )
    disabled = heuristic_auxiliary_label(
        _context(),
        _ruleset(_context()),
        HeuristicAuxiliaryValueConfig(),
    )

    assert enabled.target == 0.0
    assert enabled.included is False
    assert disabled.target == 0.0
    assert disabled.included is False


def test_label_rejects_rules_that_do_not_match_the_original_context() -> None:
    context = _context()

    with pytest.raises(HeuristicAuxiliaryError, match="player count"):
        heuristic_auxiliary_label(
            context,
            replace(_ruleset(context), player_count=4),
            HeuristicAuxiliaryValueConfig.balanced_v3(),
        )


def test_masked_smooth_l1_uses_only_included_rows_and_reports_metrics() -> None:
    predictions = torch.tensor((0.0, 0.5, 2.0), requires_grad=True)
    targets = torch.tensor((1.0, 99.0, 1.5))
    included = torch.tensor((True, False, True))
    config = HeuristicAuxiliaryValueConfig.balanced_v3(
        loss_coefficient=0.2,
        smooth_l1_delta=0.5,
    )

    result = masked_heuristic_smooth_l1_loss(predictions, targets, included, config)
    expected = torch.nn.functional.smooth_l1_loss(
        predictions[included],
        targets[included],
        reduction="mean",
        beta=0.5,
    )

    torch.testing.assert_close(result.unweighted, expected)
    torch.testing.assert_close(result.weighted, expected * 0.2)
    assert result.metrics.included_count == 2
    assert result.metrics.total_count == 3
    assert result.metrics.included_fraction == pytest.approx(2 / 3)
    assert result.metrics.mean_prediction == pytest.approx(1.0)
    assert result.metrics.mean_target == pytest.approx(1.25)
    assert result.metrics.mean_absolute_error == pytest.approx(0.75)

    result.weighted.backward()  # type: ignore[no-untyped-call]
    assert predictions.grad is not None
    assert predictions.grad[0] != 0.0
    assert predictions.grad[1] == 0.0
    assert predictions.grad[2] != 0.0


def test_all_masked_loss_is_a_differentiable_zero() -> None:
    predictions = torch.tensor((0.25, -0.5), requires_grad=True)
    result = masked_heuristic_smooth_l1_loss(
        predictions,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        HeuristicAuxiliaryValueConfig(),
    )

    assert result.unweighted.item() == 0.0
    assert result.weighted.item() == 0.0
    assert result.metrics.included_count == 0
    assert result.metrics.mean_prediction is None
    result.weighted.backward()  # type: ignore[no-untyped-call]
    torch.testing.assert_close(predictions.grad, torch.zeros(2))


@pytest.mark.parametrize(
    ("predictions", "targets", "included", "message"),
    (
        (torch.zeros((1, 1)), torch.zeros(1), torch.zeros(1, dtype=torch.bool), "one-dimensional"),
        (torch.zeros(2), torch.zeros(1), torch.zeros(2, dtype=torch.bool), "matching shapes"),
        (torch.zeros(1), torch.zeros(1), torch.zeros(1), "mask must be boolean"),
        (torch.tensor((math.nan,)), torch.zeros(1), torch.tensor((True,)), "must be finite"),
    ),
)
def test_masked_loss_rejects_malformed_inputs(
    predictions: Tensor,
    targets: Tensor,
    included: Tensor,
    message: str,
) -> None:
    with pytest.raises(HeuristicAuxiliaryError, match=message):
        masked_heuristic_smooth_l1_loss(
            predictions,
            targets,
            included,
            HeuristicAuxiliaryValueConfig.balanced_v3(),
        )
