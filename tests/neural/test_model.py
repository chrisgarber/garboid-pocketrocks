from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.encoding import NeuralBatch  # noqa: E402
from garboid_pocketrocks.neural.model import (  # noqa: E402
    NeuralPolicy,
    PolicyValueOutput,
)


def _batch(batch_size: int) -> NeuralBatch:
    global_ids = torch.tensor([[1, 3, 1, 1, 0, 2]], dtype=torch.int64).repeat(
        batch_size,
        1,
    )
    global_numeric = torch.linspace(0.0, 1.0, 21).repeat(batch_size, 1)
    objective_bits = torch.zeros((batch_size, 60), dtype=torch.float32)
    objective_bits[:, (0, 9, 30, 39)] = 1.0
    seat_numeric = torch.zeros((batch_size, 5, 41), dtype=torch.float32)
    seat_numeric[:, :3] = torch.linspace(0.0, 1.0, 3 * 41).reshape(3, 41)
    seat_valid = torch.tensor([[True, True, True, False, False]]).repeat(
        batch_size,
        1,
    )
    private_hand_ids = torch.tensor([[2, 5, 0, 0, 0]], dtype=torch.int64).repeat(
        batch_size,
        1,
    )
    hand_valid = torch.tensor([[True, True, False, False, False]]).repeat(
        batch_size,
        1,
    )
    history_ids = torch.zeros((batch_size, 76, 6), dtype=torch.int64)
    history_ids[:, :4] = torch.tensor(
        (
            (1, 0, 0, 0, 1, 0),
            (2, 1, 1, 0, 0, 0),
            (3, 0, 0, 0, 0, 0),
            (4, 0, 0, 0, 2, 3),
        ),
        dtype=torch.int64,
    )
    history_numeric = torch.zeros((batch_size, 76, 42), dtype=torch.float32)
    history_numeric[:, :4] = torch.linspace(0.0, 1.0, 4 * 42).reshape(4, 42)
    history_valid = torch.zeros((batch_size, 76), dtype=torch.bool)
    history_valid[:, :4] = True
    action_mask = torch.zeros((batch_size, 106), dtype=torch.bool)
    action_mask[:, :8] = True
    return NeuralBatch(
        global_ids=global_ids,
        global_numeric=global_numeric,
        objective_bits=objective_bits,
        seat_numeric=seat_numeric,
        seat_valid=seat_valid,
        private_hand_ids=private_hand_ids,
        hand_valid=hand_valid,
        history_ids=history_ids,
        history_numeric=history_numeric,
        history_valid=history_valid,
        action_mask=action_mask,
    )


def _model() -> NeuralPolicy:
    torch.manual_seed(41)
    return NeuralPolicy(stage1_encoder_config(), stage1_model_config())


def _assert_output_close(
    left: PolicyValueOutput,
    right: PolicyValueOutput,
) -> None:
    torch.testing.assert_close(left.bid_logits, right.bid_logits)
    torch.testing.assert_close(left.reveal_logits, right.reveal_logits)
    torch.testing.assert_close(left.value, right.value)


@pytest.mark.parametrize("batch_size", (1, 4))
def test_model_emits_exact_finite_head_shapes(batch_size: int) -> None:
    output = _model()(_batch(batch_size))

    assert output.bid_logits.shape == (batch_size, 101)
    assert output.reveal_logits.shape == (batch_size, 6)
    assert output.value.shape == (batch_size,)
    assert torch.isfinite(output.bid_logits).all()
    assert torch.isfinite(output.reveal_logits).all()
    assert torch.isfinite(output.value).all()


def test_model_is_repeatable_and_single_batched_equivalent() -> None:
    model = _model()
    single = model(_batch(1))
    repeated = model(_batch(1))
    batched = model(_batch(4))

    _assert_output_close(single, repeated)
    _assert_output_close(
        single,
        PolicyValueOutput(
            bid_logits=batched.bid_logits[:1],
            reveal_logits=batched.reveal_logits[:1],
            value=batched.value[:1],
        ),
    )


def test_padded_history_tokens_cannot_change_output() -> None:
    model = _model()
    batch = _batch(1)
    changed_ids = batch.history_ids.clone()
    changed_ids[0, -1] = torch.tensor((4, 6, 5, 4, 3, 2))

    baseline = model(batch)
    changed = model(replace(batch, history_ids=changed_ids))

    _assert_output_close(baseline, changed)


def test_valid_history_tokens_change_output() -> None:
    model = _model()
    batch = _batch(1)
    changed_ids = batch.history_ids.clone()
    changed_ids[0, 1, 1] = 2

    baseline = model(batch)
    changed = model(replace(batch, history_ids=changed_ids))

    assert not torch.equal(baseline.bid_logits, changed.bid_logits)
    assert not torch.equal(baseline.reveal_logits, changed.reveal_logits)
    assert not torch.equal(baseline.value, changed.value)


def test_forward_has_no_hidden_state_interface() -> None:
    model = _model()

    assert tuple(inspect.signature(model.forward).parameters) == ("batch",)
    assert set(PolicyValueOutput.__dataclass_fields__) == {
        "bid_logits",
        "reveal_logits",
        "value",
    }
