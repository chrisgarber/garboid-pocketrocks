from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

torch = pytest.importorskip("torch")

from torch import Generator, Tensor  # noqa: E402

from garboid_pocketrocks.neural.encoding import NeuralBatch  # noqa: E402
from garboid_pocketrocks.neural.model import PolicyValueOutput  # noqa: E402
from garboid_pocketrocks.neural.policy import (  # noqa: E402
    PolicyError,
    evaluate_masked_policy,
    evaluate_row_seeded_policy,
)


def _batch(
    phases: tuple[int, ...],
    action_mask: Tensor,
) -> NeuralBatch:
    batch_size = len(phases)
    global_ids = torch.zeros((batch_size, 6), dtype=torch.int64)
    global_ids[:, 0] = torch.tensor(phases)
    return NeuralBatch(
        global_ids=global_ids,
        global_numeric=torch.zeros((batch_size, 21)),
        objective_bits=torch.zeros((batch_size, 60)),
        seat_numeric=torch.zeros((batch_size, 5, 41)),
        seat_valid=torch.zeros((batch_size, 5), dtype=torch.bool),
        private_hand_ids=torch.zeros((batch_size, 5), dtype=torch.int64),
        hand_valid=torch.zeros((batch_size, 5), dtype=torch.bool),
        history_ids=torch.zeros((batch_size, 76, 6), dtype=torch.int64),
        history_numeric=torch.zeros((batch_size, 76, 42)),
        history_valid=torch.zeros((batch_size, 76), dtype=torch.bool),
        action_mask=action_mask,
    )


def _outputs(
    batch_size: int,
    *,
    requires_grad: bool = False,
) -> PolicyValueOutput:
    bid_logits = torch.arange(101, dtype=torch.float32).div(100).repeat(batch_size, 1).clone()
    reveal_logits = torch.arange(6, dtype=torch.float32).div(10).repeat(batch_size, 1).clone()
    value = torch.linspace(-0.5, 0.5, batch_size)
    bid_logits.requires_grad_(requires_grad)
    reveal_logits.requires_grad_(requires_grad)
    value.requires_grad_(requires_grad)
    return PolicyValueOutput(
        bid_logits=bid_logits,
        reveal_logits=reveal_logits,
        value=value,
    )


def _legal_masks(phases: tuple[int, ...]) -> Tensor:
    masks = torch.zeros((len(phases), 106), dtype=torch.bool)
    for row, phase in enumerate(phases):
        masks[row, 0] = True
        if phase == 1:
            masks[row, 1:8] = True
        else:
            masks[row, 101:104] = True
    return cast(Tensor, masks)


def test_active_heads_project_to_exact_universal_indices() -> None:
    phases = (1, 2)
    batch = _batch(phases, torch.ones((2, 106), dtype=torch.bool))
    output = _outputs(2)

    selected = evaluate_masked_policy(
        output,
        batch,
        generator=None,
        deterministic=True,
    )

    assert selected.masked_logits.shape == (2, 106)
    torch.testing.assert_close(selected.masked_logits[0, :101], output.bid_logits[0])
    assert torch.isneginf(selected.masked_logits[0, 101:]).all()
    torch.testing.assert_close(selected.masked_logits[1, 0], output.reveal_logits[1, 0])
    assert torch.isneginf(selected.masked_logits[1, 1:101]).all()
    torch.testing.assert_close(selected.masked_logits[1, 101:], output.reveal_logits[1, 1:])


def test_context_illegal_actions_have_negative_infinity_and_zero_probability() -> None:
    phases = (1, 2)
    masks = _legal_masks(phases)

    selected = evaluate_masked_policy(
        _outputs(2),
        _batch(phases, masks),
        generator=None,
        deterministic=True,
    )

    assert torch.isneginf(selected.masked_logits[~masks]).all()
    assert selected.probabilities[~masks].sum().item() == 0.0


def test_stochastic_sampling_is_always_legal_over_one_thousand_draws() -> None:
    phases = tuple(1 if index % 2 == 0 else 2 for index in range(1_000))
    masks = _legal_masks(phases)
    generator = torch.Generator(device="cpu").manual_seed(12)

    selected = evaluate_masked_policy(
        _outputs(1_000),
        _batch(phases, masks),
        generator=generator,
        deterministic=False,
    )

    assert masks.gather(1, selected.actions.unsqueeze(1)).all()


def test_seeded_generators_produce_identical_sample_sequences() -> None:
    phases = tuple(1 if index % 2 == 0 else 2 for index in range(256))
    batch = _batch(phases, _legal_masks(phases))
    output = _outputs(256)
    first_generator = torch.Generator(device="cpu").manual_seed(93)
    second_generator = torch.Generator(device="cpu").manual_seed(93)

    first = evaluate_masked_policy(
        output,
        batch,
        generator=first_generator,
        deterministic=False,
    )
    second = evaluate_masked_policy(
        output,
        batch,
        generator=second_generator,
        deterministic=False,
    )

    assert torch.equal(first.actions, second.actions)


def test_row_seeded_sampling_is_independent_of_batch_order() -> None:
    phases = (1, 2, 1, 2)
    masks = _legal_masks(phases)
    batch = _batch(phases, masks)
    output = _outputs(4)
    seeds = (101, 202, 303, 404)

    original = evaluate_row_seeded_policy(
        output,
        batch,
        row_seeds=seeds,
    )
    permutation = torch.tensor((2, 0, 3, 1))
    permuted = evaluate_row_seeded_policy(
        PolicyValueOutput(
            bid_logits=output.bid_logits[permutation],
            reveal_logits=output.reveal_logits[permutation],
            value=output.value[permutation],
        ),
        NeuralBatch(
            global_ids=batch.global_ids[permutation],
            global_numeric=batch.global_numeric[permutation],
            objective_bits=batch.objective_bits[permutation],
            seat_numeric=batch.seat_numeric[permutation],
            seat_valid=batch.seat_valid[permutation],
            private_hand_ids=batch.private_hand_ids[permutation],
            hand_valid=batch.hand_valid[permutation],
            history_ids=batch.history_ids[permutation],
            history_numeric=batch.history_numeric[permutation],
            history_valid=batch.history_valid[permutation],
            action_mask=batch.action_mask[permutation],
        ),
        row_seeds=tuple(seeds[index] for index in permutation.tolist()),
    )

    inverse = torch.argsort(permutation)
    assert torch.equal(original.actions, permuted.actions[inverse])
    torch.testing.assert_close(
        original.log_probability,
        permuted.log_probability[inverse],
    )
    torch.testing.assert_close(original.value, permuted.value[inverse])


@pytest.mark.parametrize(
    "row_seeds, message",
    [
        ((1,), "one seed per row"),
        ((1, True), "unsigned 63-bit"),
        ((1, -1), "unsigned 63-bit"),
        ((1, 2**63), "unsigned 63-bit"),
    ],
)
def test_row_seeded_sampling_validates_each_seed(
    row_seeds: tuple[int, ...],
    message: str,
) -> None:
    phases = (1, 2)

    with pytest.raises(PolicyError, match=message):
        evaluate_row_seeded_policy(
            _outputs(2),
            _batch(phases, _legal_masks(phases)),
            row_seeds=row_seeds,
        )


def test_stochastic_selection_passes_generator_to_torch_multinomial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = torch.Generator(device="cpu").manual_seed(7)
    seen: list[Generator | None] = []
    original = torch.multinomial

    def spy(
        input: Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: Generator | None = None,
        out: Tensor | None = None,
    ) -> Tensor:
        seen.append(generator)
        return cast(
            Tensor,
            original(
                input,
                num_samples,
                replacement,
                generator=generator,
                out=out,
            ),
        )

    monkeypatch.setattr(torch, "multinomial", spy)
    phases = (1,)
    evaluate_masked_policy(
        _outputs(1),
        _batch(phases, _legal_masks(phases)),
        generator=supplied,
        deterministic=False,
    )

    assert seen == [supplied]


def test_deterministic_selection_breaks_equal_ties_by_lowest_universal_index() -> None:
    phases = (1, 2)
    masks = torch.zeros((2, 106), dtype=torch.bool)
    masks[0, (0, 3, 7)] = True
    masks[1, (0, 101, 103)] = True
    output = _outputs(2)
    bid_logits = output.bid_logits.clone()
    reveal_logits = output.reveal_logits.clone()
    bid_logits[0] = -1.0
    bid_logits[0, (3, 7)] = 5.0
    reveal_logits[1] = -1.0
    reveal_logits[1, (0, 1)] = 5.0

    selected = evaluate_masked_policy(
        replace(
            output,
            bid_logits=bid_logits,
            reveal_logits=reveal_logits,
        ),
        _batch(phases, masks),
        generator=None,
        deterministic=True,
    )

    assert selected.actions.tolist() == [3, 0]


def test_selected_log_probability_entropy_and_backward_gradients_are_finite() -> None:
    phases = (1, 2)
    masks = _legal_masks(phases)
    output = _outputs(2, requires_grad=True)

    selected = evaluate_masked_policy(
        output,
        _batch(phases, masks),
        generator=None,
        deterministic=True,
    )
    selected_probabilities = selected.probabilities.gather(
        1,
        selected.actions.unsqueeze(1),
    ).squeeze(1)
    safe_log_probabilities = torch.zeros_like(selected.probabilities)
    positive = selected.probabilities > 0
    safe_log_probabilities[positive] = torch.log(selected.probabilities[positive])
    expected_entropy = -(selected.probabilities * safe_log_probabilities).sum(dim=-1)

    torch.testing.assert_close(
        selected.log_probability,
        torch.log(selected_probabilities),
    )
    torch.testing.assert_close(selected.entropy, expected_entropy)
    assert torch.isfinite(output.bid_logits).all()
    assert torch.isfinite(output.reveal_logits).all()
    assert torch.isfinite(selected.log_probability).all()
    assert torch.isfinite(selected.entropy).all()

    loss = -selected.log_probability.mean() - selected.entropy.mean() + selected.value.mean()
    loss.backward()  # type: ignore[no-untyped-call]

    assert output.bid_logits.grad is not None
    assert output.reveal_logits.grad is not None
    assert output.value.grad is not None
    assert torch.isfinite(output.bid_logits.grad).all()
    assert torch.isfinite(output.reveal_logits.grad).all()
    assert torch.isfinite(output.value.grad).all()


def test_every_action_mask_must_enable_universal_pass() -> None:
    masks = torch.zeros((1, 106), dtype=torch.bool)
    masks[0, 1] = True

    with pytest.raises(PolicyError, match="pass"):
        evaluate_masked_policy(
            _outputs(1),
            _batch((1,), masks),
            generator=None,
            deterministic=True,
        )
