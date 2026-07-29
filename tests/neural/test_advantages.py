from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from garboid_pocketrocks.neural.advantages import (  # noqa: E402
    AdvantageError,
    compute_gae,
)


def test_compute_gae_matches_hand_calculated_terminated_trajectory() -> None:
    batch = compute_gae(
        rewards=torch.tensor([1.0, 0.5]),
        values=torch.tensor([0.2, 0.4]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        bootstrap_value=torch.tensor(0.0),
        gamma=1.0,
        gae_lambda=0.95,
    )

    torch.testing.assert_close(batch.advantages, torch.tensor([1.295, 0.1]))
    torch.testing.assert_close(batch.returns, torch.tensor([1.495, 0.5]))


def test_truncation_bootstraps_but_termination_does_not() -> None:
    common = {
        "rewards": torch.tensor([0.5]),
        "values": torch.tensor([0.4]),
        "bootstrap_value": torch.tensor(0.7),
        "gamma": 1.0,
        "gae_lambda": 0.95,
    }

    truncated_batch = compute_gae(
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        **common,
    )
    terminated_batch = compute_gae(
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        **common,
    )

    torch.testing.assert_close(truncated_batch.advantages, torch.tensor([0.8]))
    torch.testing.assert_close(truncated_batch.returns, torch.tensor([1.2]))
    torch.testing.assert_close(terminated_batch.advantages, torch.tensor([0.1]))
    torch.testing.assert_close(terminated_batch.returns, torch.tensor([0.5]))


def test_zero_reward_reveal_step_preserves_undiscounted_prior_return() -> None:
    without_reveal = compute_gae(
        rewards=torch.tensor([1.0, 0.5]),
        values=torch.tensor([0.2, 0.4]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        bootstrap_value=torch.tensor(0.0),
        gamma=1.0,
        gae_lambda=1.0,
    )
    with_reveal = compute_gae(
        rewards=torch.tensor([1.0, 0.0, 0.5]),
        values=torch.tensor([0.2, 0.3, 0.4]),
        terminated=torch.tensor([False, False, True]),
        truncated=torch.tensor([False, False, False]),
        bootstrap_value=torch.tensor(0.0),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(with_reveal.returns[0], without_reveal.returns[0])


@dataclass(frozen=True, slots=True)
class _Arguments:
    rewards: Tensor
    values: Tensor
    terminated: Tensor
    truncated: Tensor
    bootstrap_value: Tensor
    gamma: float
    gae_lambda: float


_VALID_ARGUMENTS = _Arguments(
    rewards=torch.tensor([1.0, 0.5]),
    values=torch.tensor([0.2, 0.4]),
    terminated=torch.tensor([False, True]),
    truncated=torch.tensor([False, False]),
    bootstrap_value=torch.tensor(0.0),
    gamma=1.0,
    gae_lambda=0.95,
)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(
            replace(_VALID_ARGUMENTS, values=torch.tensor([0.2])),
            id="mismatched-values",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, terminated=torch.tensor([False])),
            id="mismatched-terminated",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, truncated=torch.tensor([False])),
            id="mismatched-truncated",
        ),
        pytest.param(
            replace(
                _VALID_ARGUMENTS,
                terminated=torch.tensor([False, True]),
                truncated=torch.tensor([False, True]),
            ),
            id="terminated-and-truncated",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, rewards=torch.tensor([math.nan, 0.5])),
            id="nonfinite-reward",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, values=torch.tensor([0.2, math.inf])),
            id="nonfinite-value",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, bootstrap_value=torch.tensor(math.nan)),
            id="nonfinite-bootstrap",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, gae_lambda=math.inf),
            id="nonfinite-lambda",
        ),
        pytest.param(
            replace(_VALID_ARGUMENTS, gamma=0.99),
            id="discounted-gamma",
        ),
    ],
)
def test_invalid_inputs_raise_advantage_error(arguments: _Arguments) -> None:

    with pytest.raises(AdvantageError):
        compute_gae(
            arguments.rewards,
            arguments.values,
            arguments.terminated,
            arguments.truncated,
            bootstrap_value=arguments.bootstrap_value,
            gamma=arguments.gamma,
            gae_lambda=arguments.gae_lambda,
        )
