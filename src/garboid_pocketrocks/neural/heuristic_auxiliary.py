"""Training-only heuristic value labels and their masked regression loss.

The label deliberately stays separate from rewards and PPO returns.  It asks a
second model head to estimate how valuable the frozen balanced-v3 heuristic
considers winning the current lot before paying for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from pocketrocks import DecisionContext
from torch import Tensor
from torch.nn import functional as F

from garboid_pocketrocks.heuristics.profiles import HEURISTIC_V3
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.neural.heuristic_teachers import BALANCED_V3_PROFILE_DIGEST

HEURISTIC_AUXILIARY_TARGET_NAME: Literal["balanced-v3-bid-win-delta-v1"] = (
    "balanced-v3-bid-win-delta-v1"
)
HeuristicAuxiliaryTargetName = Literal[
    "disabled",
    "balanced-v3-bid-win-delta-v1",
]


class HeuristicAuxiliaryError(ValueError):
    """Raised when an auxiliary label or loss violates its contract."""


@dataclass(frozen=True, slots=True)
class HeuristicAuxiliaryValueConfig:
    """Immutable configuration for the isolated auxiliary-value objective."""

    target: HeuristicAuxiliaryTargetName = "disabled"
    loss_coefficient: float = 0.0
    smooth_l1_delta: float = 0.25
    teacher_profile_digest: str = BALANCED_V3_PROFILE_DIGEST

    def __post_init__(self) -> None:
        if self.target not in ("disabled", HEURISTIC_AUXILIARY_TARGET_NAME):
            raise HeuristicAuxiliaryError("unknown heuristic auxiliary target")
        if self.teacher_profile_digest != BALANCED_V3_PROFILE_DIGEST:
            raise HeuristicAuxiliaryError("balanced-v3 profile digest does not match the pin")
        for name, value in (
            ("loss coefficient", self.loss_coefficient),
            ("Smooth L1 delta", self.smooth_l1_delta),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise HeuristicAuxiliaryError(f"{name} must be finite")
        if self.loss_coefficient < 0.0:
            raise HeuristicAuxiliaryError("loss coefficient must be nonnegative")
        if self.smooth_l1_delta <= 0.0:
            raise HeuristicAuxiliaryError("Smooth L1 delta must be positive")
        if self.target == "disabled" and self.loss_coefficient != 0.0:
            raise HeuristicAuxiliaryError("disabled target requires a zero loss coefficient")
        if self.target != "disabled" and self.loss_coefficient <= 0.0:
            raise HeuristicAuxiliaryError("enabled target requires a positive loss coefficient")

    @classmethod
    def balanced_v3(
        cls,
        *,
        loss_coefficient: float = 0.10,
        smooth_l1_delta: float = 0.25,
        teacher_profile_digest: str = BALANCED_V3_PROFILE_DIGEST,
    ) -> HeuristicAuxiliaryValueConfig:
        """Return the one supported behavior-changing experiment config."""

        return cls(
            target=HEURISTIC_AUXILIARY_TARGET_NAME,
            loss_coefficient=loss_coefficient,
            smooth_l1_delta=smooth_l1_delta,
            teacher_profile_digest=teacher_profile_digest,
        )

    @classmethod
    def from_json_dict(cls, value: object) -> HeuristicAuxiliaryValueConfig:
        """Parse an exact-key JSON object without coercing values."""

        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise HeuristicAuxiliaryError("heuristic auxiliary config must be an object")
        expected = {
            "target",
            "loss_coefficient",
            "smooth_l1_delta",
            "teacher_profile_digest",
        }
        if set(value) != expected:
            raise HeuristicAuxiliaryError("heuristic auxiliary config keys must be exact")
        target = value["target"]
        if not isinstance(target, str):
            raise HeuristicAuxiliaryError("heuristic auxiliary target must be a string")
        return cls(
            target=cast(HeuristicAuxiliaryTargetName, target),
            loss_coefficient=_strict_number(value["loss_coefficient"], "loss coefficient"),
            smooth_l1_delta=_strict_number(value["smooth_l1_delta"], "Smooth L1 delta"),
            teacher_profile_digest=_strict_string(
                value["teacher_profile_digest"],
                "teacher profile digest",
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a complete JSON-safe representation."""

        return {
            "target": self.target,
            "loss_coefficient": float(self.loss_coefficient),
            "smooth_l1_delta": float(self.smooth_l1_delta),
            "teacher_profile_digest": self.teacher_profile_digest,
        }


@dataclass(frozen=True, slots=True)
class HeuristicAuxiliaryLabel:
    """One normalized target plus whether this decision should train on it."""

    target: float
    included: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.target):
            raise HeuristicAuxiliaryError("heuristic auxiliary target must be finite")
        if not isinstance(self.included, bool):
            raise HeuristicAuxiliaryError("heuristic auxiliary mask must be a boolean")


@dataclass(frozen=True, slots=True)
class HeuristicAuxiliaryMetrics:
    """Detached diagnostics for one masked auxiliary minibatch."""

    included_count: int
    total_count: int
    included_fraction: float
    mean_prediction: float | None
    mean_target: float | None
    mean_absolute_error: float | None
    smooth_l1_loss: float


@dataclass(frozen=True, slots=True)
class HeuristicAuxiliaryLoss:
    """Differentiable loss values and detached reporting metrics."""

    weighted: Tensor
    unweighted: Tensor
    metrics: HeuristicAuxiliaryMetrics


def heuristic_auxiliary_label(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
    config: HeuristicAuxiliaryValueConfig,
) -> HeuristicAuxiliaryLabel:
    """Build a deterministic label from the original public decision inputs.

    Bid labels are the balanced-v3 heuristic's dollar value for winning at bid
    zero, divided by starting cash.  This captures its value estimate without
    copying its shaded action.  Reveal decisions are intentionally masked.
    """

    if config.target == "disabled":
        return HeuristicAuxiliaryLabel(target=0.0, included=False)
    _validate_ruleset_matches_context(context, ruleset)
    if context.decision_kind == "selectInfoToReveal":
        return HeuristicAuxiliaryLabel(target=0.0, included=False)
    if context.decision_kind != "submitBid":
        raise HeuristicAuxiliaryError("unsupported decision kind for heuristic auxiliary label")
    if context.starting_cash <= 0:
        raise HeuristicAuxiliaryError("starting cash must be positive")

    evaluation = HeuristicValuator(HEURISTIC_V3.balanced).evaluate_bid(context, ruleset)
    if not evaluation.points or evaluation.points[0].bid != 0:
        raise HeuristicAuxiliaryError("heuristic valuation must begin with bid zero")
    target = evaluation.points[0].win_delta / context.starting_cash
    if not math.isfinite(target):
        raise HeuristicAuxiliaryError("heuristic auxiliary target must be finite")
    return HeuristicAuxiliaryLabel(target=target, included=True)


def masked_heuristic_smooth_l1_loss(
    predictions: Tensor,
    targets: Tensor,
    included: Tensor,
    config: HeuristicAuxiliaryValueConfig,
) -> HeuristicAuxiliaryLoss:
    """Compute mean Smooth L1 loss over included rows only.

    An all-masked batch returns a differentiable zero attached to predictions,
    which lets a PPO caller add it unconditionally without special cases.
    """

    _validate_loss_inputs(predictions, targets, included)
    included_count = int(included.sum().item())
    total_count = predictions.numel()
    if config.target == "disabled" and included_count:
        raise HeuristicAuxiliaryError("disabled auxiliary target cannot include rows")
    if included_count == 0:
        unweighted = predictions.sum() * 0.0
        metrics = HeuristicAuxiliaryMetrics(
            included_count=0,
            total_count=total_count,
            included_fraction=0.0,
            mean_prediction=None,
            mean_target=None,
            mean_absolute_error=None,
            smooth_l1_loss=0.0,
        )
    else:
        selected_predictions = predictions[included]
        selected_targets = targets[included]
        unweighted = F.smooth_l1_loss(
            selected_predictions,
            selected_targets,
            reduction="mean",
            beta=float(config.smooth_l1_delta),
        )
        absolute_errors = torch.abs(selected_predictions.detach() - selected_targets.detach())
        metrics = HeuristicAuxiliaryMetrics(
            included_count=included_count,
            total_count=total_count,
            included_fraction=included_count / total_count,
            mean_prediction=float(selected_predictions.detach().mean().item()),
            mean_target=float(selected_targets.detach().mean().item()),
            mean_absolute_error=float(absolute_errors.mean().item()),
            smooth_l1_loss=float(unweighted.detach().item()),
        )
    return HeuristicAuxiliaryLoss(
        weighted=unweighted * float(config.loss_coefficient),
        unweighted=unweighted,
        metrics=metrics,
    )


def _validate_ruleset_matches_context(
    context: DecisionContext,
    ruleset: RulesetKnowledge,
) -> None:
    if ruleset.player_count != context.player_count:
        raise HeuristicAuxiliaryError("ruleset player count does not match context")
    if ruleset.starting_cash != context.starting_cash:
        raise HeuristicAuxiliaryError("ruleset starting cash does not match context")
    if ruleset.value_chart != context.value_chart:
        raise HeuristicAuxiliaryError("ruleset value chart does not match context")
    if ruleset.active_objective_count != len(context.objective_ids):
        raise HeuristicAuxiliaryError("ruleset objective count does not match context")


def _validate_loss_inputs(predictions: Tensor, targets: Tensor, included: Tensor) -> None:
    if predictions.ndim != 1 or targets.ndim != 1 or included.ndim != 1:
        raise HeuristicAuxiliaryError("heuristic auxiliary loss inputs must be one-dimensional")
    if predictions.shape != targets.shape or predictions.shape != included.shape:
        raise HeuristicAuxiliaryError("heuristic auxiliary loss inputs must have matching shapes")
    if predictions.numel() == 0:
        raise HeuristicAuxiliaryError("heuristic auxiliary loss requires at least one row")
    if not torch.is_floating_point(predictions) or not torch.is_floating_point(targets):
        raise HeuristicAuxiliaryError("predictions and targets must be floating-point tensors")
    if included.dtype != torch.bool:
        raise HeuristicAuxiliaryError("heuristic auxiliary mask must be boolean")
    if predictions.device != targets.device or predictions.device != included.device:
        raise HeuristicAuxiliaryError("heuristic auxiliary loss inputs must share a device")
    if predictions.dtype != targets.dtype:
        raise HeuristicAuxiliaryError("predictions and targets must share a dtype")
    if not torch.isfinite(predictions).all().item() or not torch.isfinite(targets).all().item():
        raise HeuristicAuxiliaryError("heuristic auxiliary loss inputs must be finite")


def _strict_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise HeuristicAuxiliaryError(f"{name} must be a finite number")
    return float(value)


def _strict_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise HeuristicAuxiliaryError(f"{name} must be a string")
    return value
