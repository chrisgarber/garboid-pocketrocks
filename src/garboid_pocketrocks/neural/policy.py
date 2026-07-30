"""Universal action projection, legal masking, and policy selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from garboid_pocketrocks.neural.encoding import NeuralBatch
from garboid_pocketrocks.neural.model import PolicyValueOutput
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds

_BID_PHASE_ID = 1
_REVEAL_PHASE_ID = 2


class PolicyError(ValueError):
    """Raised when model output and a legal-action batch are incompatible."""


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """One selected universal action and its masked distribution."""

    actions: Tensor
    log_probability: Tensor
    probabilities: Tensor
    masked_logits: Tensor
    entropy: Tensor
    value: Tensor


def evaluate_masked_policy(
    output: PolicyValueOutput,
    batch: NeuralBatch,
    *,
    generator: torch.Generator | None,
    deterministic: bool,
) -> PolicySelection:
    """Project the active phase head, mask illegality, and select an action."""

    masked, log_probabilities, probabilities = _distribution(output, batch)
    if deterministic:
        actions = torch.argmax(masked, dim=-1)
    else:
        actions = torch.multinomial(
            probabilities,
            1,
            generator=generator,
        ).squeeze(1)
    return _selection(
        output,
        masked,
        log_probabilities,
        probabilities,
        actions,
    )


def evaluate_row_seeded_policy(
    output: PolicyValueOutput,
    batch: NeuralBatch,
    *,
    row_seeds: Sequence[int],
) -> PolicySelection:
    """Sample every row from its own CPU seed, independent of batch packing."""

    batch_size = output.value.shape[0] if output.value.ndim == 1 else 0
    if len(row_seeds) != batch_size:
        raise PolicyError("row-seeded sampling requires one seed per row")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63
        for seed in row_seeds
    ):
        raise PolicyError("row seeds must be unsigned 63-bit integers")

    masked, log_probabilities, probabilities = _distribution(output, batch)
    probabilities_cpu = probabilities.detach().cpu()
    actions_cpu = torch.stack(
        tuple(
            torch.multinomial(
                probabilities_cpu[row],
                1,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            for row, seed in enumerate(row_seeds)
        )
    ).squeeze(1)
    actions = actions_cpu.to(probabilities.device)
    return _selection(
        output,
        masked,
        log_probabilities,
        probabilities,
        actions,
    )


def _distribution(
    output: PolicyValueOutput,
    batch: NeuralBatch,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate(output, batch)
    batch_size = output.value.shape[0]
    bounds = EnvironmentBounds(
        max_bid=output.bid_logits.shape[1] - 1,
        max_hand_size=output.reveal_logits.shape[1] - 1,
    )
    codec = ActionCodec(bounds)
    phases = batch.global_ids[:, 0]
    bid_rows = phases == _BID_PHASE_ID
    reveal_rows = phases == _REVEAL_PHASE_ID

    universal = torch.full(
        (batch_size, codec.size),
        -torch.inf,
        dtype=output.bid_logits.dtype,
        device=output.bid_logits.device,
    )
    universal[bid_rows, : bounds.max_bid + 1] = output.bid_logits[bid_rows]
    universal[reveal_rows, 0] = output.reveal_logits[reveal_rows, 0]
    universal[reveal_rows, bounds.max_bid + 1 :] = output.reveal_logits[
        reveal_rows,
        1:,
    ]
    masked = universal.masked_fill(~batch.action_mask, -torch.inf)
    log_probabilities = torch.log_softmax(masked, dim=-1)
    probabilities = torch.softmax(masked, dim=-1)
    return masked, log_probabilities, probabilities


def _selection(
    output: PolicyValueOutput,
    masked: Tensor,
    log_probabilities: Tensor,
    probabilities: Tensor,
    actions: Tensor,
) -> PolicySelection:
    selected_log_probability = log_probabilities.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)
    safe_log_probabilities = log_probabilities.masked_fill(
        ~torch.isfinite(log_probabilities),
        0.0,
    )
    entropy = -(probabilities * safe_log_probabilities).sum(dim=-1)
    return PolicySelection(
        actions=actions,
        log_probability=selected_log_probability,
        probabilities=probabilities,
        masked_logits=masked,
        entropy=entropy,
        value=output.value,
    )


def _validate(
    output: PolicyValueOutput,
    batch: NeuralBatch,
) -> None:
    if output.bid_logits.ndim != 2 or output.reveal_logits.ndim != 2:
        raise PolicyError("policy heads must be matrices")
    if output.value.ndim != 1:
        raise PolicyError("policy values must be a vector")
    batch_size = output.value.shape[0]
    if (
        output.bid_logits.shape[0] != batch_size
        or output.reveal_logits.shape[0] != batch_size
        or batch.global_ids.shape[0] != batch_size
        or batch.action_mask.shape[0] != batch_size
    ):
        raise PolicyError("policy output and observation batch sizes differ")
    max_bid = output.bid_logits.shape[1] - 1
    max_hand_size = output.reveal_logits.shape[1] - 1
    expected_actions = 1 + max_bid + max_hand_size
    if max_bid < 0 or max_hand_size < 0:
        raise PolicyError("policy heads must include universal pass")
    if batch.action_mask.shape != (batch_size, expected_actions):
        raise PolicyError("action mask does not match policy heads")
    if batch.action_mask.dtype != torch.bool:
        raise PolicyError("action mask must be boolean")
    if not torch.all(batch.action_mask[:, 0]).item():
        raise PolicyError("every action mask must enable universal pass")
    phases = batch.global_ids[:, 0]
    if not torch.all((phases == _BID_PHASE_ID) | (phases == _REVEAL_PHASE_ID)).item():
        raise PolicyError("batch contains an unknown policy phase")
    if not (
        torch.isfinite(output.bid_logits).all().item()
        and torch.isfinite(output.reveal_logits).all().item()
        and torch.isfinite(output.value).all().item()
    ):
        raise PolicyError("raw policy output must be finite")
