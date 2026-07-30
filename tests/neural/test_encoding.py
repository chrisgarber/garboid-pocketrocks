from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pocketrocks import DecisionContext

torch = pytest.importorskip("torch")

from garboid_pocketrocks.adapters.public_history import (  # noqa: E402
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.knowledge import (  # noqa: E402
    RulesetKnowledge,
    canonical_knowledge,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    NeuralEncoderConfig,
    NeuralModelConfig,
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.encoding import (  # noqa: E402
    NeuralEncodingError,
    NeuralObservation,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.training.actions import ActionCodec  # noqa: E402
from garboid_pocketrocks.training.bounds import EnvironmentBounds  # noqa: E402

_BOUNDS = EnvironmentBounds(max_bid=100, max_hand_size=5)


def _context(
    *,
    player_count: int = 3,
    learner_seat: int = 1,
    decision_kind: str = "submitBid",
) -> DecisionContext:
    cash = tuple(30 - (2 * seat) for seat in range(player_count))
    won = tuple(
        tuple(1 if suit == seat % 5 else 0 for suit in range(5)) for seat in range(player_count)
    )
    revealed = tuple(
        tuple(1 if suit == (seat + 1) % 5 else 0 for suit in range(5))
        for seat in range(player_count)
    )
    owned = tuple((seat + 1,) for seat in range(player_count))
    return DecisionContext(
        request_id="neural-test",
        deadline_at=0,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=player_count,
        starting_cash=canonical_knowledge(player_count).starting_cash,
        value_chart=canonical_knowledge(player_count).value_chart,
        objective_ids=(1, 10),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=cash,
        tiebreak_seat=(learner_seat + 1) % player_count,
        won_resource_counts_by_seat=won,
        revealed_info_counts_by_seat=revealed,
        owned_objective_ids_by_seat=owned,
        bot_seat=learner_seat,
        current_hand_suit_ids=(2, 5),
        legal_max_amount=7 if decision_kind == "submitBid" else None,
        revealable_count=2,
    )


def _history(
    *,
    player_count: int = 3,
    initial_tiebreak_seat: int = 0,
    reveal_seat: int = 2,
    bids_by_seat: tuple[int, ...] = (2, 5, 1),
) -> PublicHistory:
    return (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=player_count,
            starting_cash=canonical_knowledge(player_count).starting_cash,
            value_chart=canonical_knowledge(player_count).value_chart,
            initial_tiebreak_seat=initial_tiebreak_seat,
            objective_ids=(1, 10),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(1, 0),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=bids_by_seat,
        ),
        PublicInformationRevealed(
            kind=PublicEventKind.INFORMATION_REVEALED,
            seat=reveal_seat,
            suit_id=3,
        ),
    )


def _encoder(
    *,
    config: NeuralEncoderConfig | None = None,
    action_codec: ActionCodec | None = None,
) -> NeuralObservationEncoder:
    return NeuralObservationEncoder(
        config or stage1_encoder_config(),
        _BOUNDS,
        action_codec=action_codec,
    )


def _assert_equal(left: NeuralObservation, right: NeuralObservation) -> None:
    assert np.array_equal(left.global_ids, right.global_ids)
    assert np.array_equal(left.global_numeric, right.global_numeric)
    assert np.array_equal(left.objective_bits, right.objective_bits)
    assert np.array_equal(left.seat_numeric, right.seat_numeric)
    assert np.array_equal(left.seat_valid, right.seat_valid)
    assert np.array_equal(left.private_hand_ids, right.private_hand_ids)
    assert np.array_equal(left.hand_valid, right.hand_valid)
    assert np.array_equal(left.history_ids, right.history_ids)
    assert np.array_equal(left.history_numeric, right.history_numeric)
    assert np.array_equal(left.history_valid, right.history_valid)
    assert np.array_equal(left.action_mask, right.action_mask)


def test_stage1_configs_are_exact_and_json_round_trip() -> None:
    encoder_config = stage1_encoder_config()
    expected_history = (
        1
        + (2 * sum(canonical_knowledge(3).action_counts))
        + (3 * canonical_knowledge(3).private_cards_per_player)
    )

    assert encoder_config == NeuralEncoderConfig(
        schema_version=1,
        supported_ruleset_names=("live-A",),
        supported_player_counts=(3,),
        max_bid=100,
        max_hand_size=5,
        max_history_events=expected_history,
        max_cash=100,
        max_abs_chart=20,
        max_resource_cards=30,
        max_action_cards=30,
    )
    assert expected_history == 76
    assert stage1_model_config() == NeuralModelConfig(
        categorical_embedding_size=8,
        suit_embedding_size=4,
        seat_hidden_size=32,
        event_embedding_size=64,
        gru_hidden_size=64,
        snapshot_hidden_size=128,
        trunk_hidden_size=128,
    )

    encoder_payload: Any = json.loads(json.dumps(asdict(encoder_config)))
    model_payload: Any = json.loads(json.dumps(asdict(stage1_model_config())))
    assert NeuralEncoderConfig(**encoder_payload) == encoder_config
    assert NeuralModelConfig(**model_payload) == stage1_model_config()


def test_live_a_encoding_has_exact_shapes_dtypes_masks_and_hand_order() -> None:
    context = _context()
    history = _history()
    encoded = _encoder().encode(context, canonical_knowledge(3), history)

    assert encoded.global_ids.shape == (6,)
    assert encoded.global_numeric.shape == (21,)
    assert encoded.objective_bits.shape == (60,)
    assert encoded.seat_numeric.shape == (5, 41)
    assert encoded.seat_valid.tolist() == [True, True, True, False, False]
    assert encoded.private_hand_ids.shape == (5,)
    assert encoded.private_hand_ids.tolist() == [2, 5, 0, 0, 0]
    assert encoded.hand_valid.tolist() == [True, True, False, False, False]
    assert encoded.history_ids.shape == (76, 6)
    assert encoded.history_numeric.shape == (76, 42)
    assert int(encoded.history_valid.sum()) == len(history)
    assert np.array_equal(
        encoded.action_mask,
        ActionCodec(_BOUNDS).mask(context).astype(bool),
    )

    for values in (
        encoded.global_numeric,
        encoded.objective_bits,
        encoded.seat_numeric,
        encoded.history_numeric,
    ):
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
    for ids in (
        encoded.global_ids,
        encoded.private_hand_ids,
        encoded.history_ids,
    ):
        assert ids.dtype == np.int64
    for mask in (
        encoded.seat_valid,
        encoded.hand_valid,
        encoded.history_valid,
        encoded.action_mask,
    ):
        assert mask.dtype == np.bool_


def _absolute_rows[T](
    relative_rows: tuple[T, ...],
    learner_seat: int,
) -> tuple[T, ...]:
    absolute: list[T | None] = [None] * len(relative_rows)
    for relative_seat, row in enumerate(relative_rows):
        absolute[(learner_seat + relative_seat) % len(relative_rows)] = row
    return cast(tuple[T, ...], tuple(absolute))


def _rotated_context(player_count: int, learner_seat: int) -> DecisionContext:
    relative_cash = tuple(20 + seat for seat in range(player_count))
    relative_won = tuple(
        tuple(relative_seat + suit for suit in range(5)) for relative_seat in range(player_count)
    )
    relative_revealed = tuple(
        tuple((2 * relative_seat) + suit for suit in range(5))
        for relative_seat in range(player_count)
    )
    relative_owned = tuple((relative_seat + 1,) for relative_seat in range(player_count))
    setup = canonical_knowledge(player_count)
    return DecisionContext(
        request_id=f"rotation-{player_count}-{learner_seat}",
        deadline_at=0,
        received_at=0,
        decision_kind="submitBid",
        player_count=player_count,
        starting_cash=setup.starting_cash,
        value_chart=canonical_knowledge(player_count).value_chart,
        objective_ids=(1, 2, 3, 4),
        current_action_id=2,
        current_resource_ids=(4, 5),
        cash_by_seat=_absolute_rows(relative_cash, learner_seat),
        tiebreak_seat=(learner_seat + 2) % player_count,
        won_resource_counts_by_seat=_absolute_rows(relative_won, learner_seat),
        revealed_info_counts_by_seat=_absolute_rows(relative_revealed, learner_seat),
        owned_objective_ids_by_seat=_absolute_rows(relative_owned, learner_seat),
        bot_seat=learner_seat,
        current_hand_suit_ids=(5, 2, 4),
        legal_max_amount=7,
        revealable_count=3,
    )


def _rotated_history(player_count: int, learner_seat: int) -> PublicHistory:
    relative_bids = tuple(3 + seat for seat in range(player_count))
    return _history(
        player_count=player_count,
        initial_tiebreak_seat=(learner_seat + 1) % player_count,
        reveal_seat=(learner_seat + 2) % player_count,
        bids_by_seat=_absolute_rows(relative_bids, learner_seat),
    )


@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_all_seat_indexed_fields_are_learner_rotation_invariant(
    player_count: int,
) -> None:
    config = replace(
        stage1_encoder_config(),
        supported_player_counts=(3, 4, 5),
        max_history_events=77,
    )
    encoder = _encoder(config=config)
    baseline: NeuralObservation | None = None

    for learner_seat in range(player_count):
        context = _rotated_context(player_count, learner_seat)
        encoded = encoder.encode(
            context,
            canonical_knowledge(player_count),
            _rotated_history(player_count, learner_seat),
        )

        assert encoded.global_ids[5] == (context.tiebreak_seat - context.bot_seat) % player_count
        assert encoded.seat_numeric[0, 0] == pytest.approx(
            context.cash_by_seat[context.bot_seat] / config.max_cash
        )
        assert encoded.private_hand_ids.tolist() == [5, 2, 4, 0, 0]
        if baseline is None:
            baseline = encoded
        else:
            _assert_equal(encoded, baseline)


def test_encoder_cannot_observe_hypothetical_private_state() -> None:
    context = _context()
    knowledge = canonical_knowledge(3)
    history = _history()
    encoder = _encoder()
    baseline = encoder.encode(context, knowledge, history)

    hidden_state = {
        "opponent_hands": ((1, 1), (5, 5)),
        "resource_deck": (5, 4, 3, 2, 1),
    }
    hidden_state["opponent_hands"] = ((3, 3), (4, 4))
    hidden_state["resource_deck"] = (1, 2, 3, 4, 5)

    _assert_equal(encoder.encode(context, knowledge, history), baseline)


def test_history_longer_than_checkpoint_bound_is_rejected() -> None:
    setup = _history()[0]
    turn = _history()[1]
    history = cast(PublicHistory, (setup,) + ((turn,) * 76))

    with pytest.raises(
        NeuralEncodingError,
        match="^history exceeds checkpoint bound$",
    ):
        _encoder().encode(_context(), canonical_knowledge(3), history)


@dataclass(frozen=True, slots=True)
class _InvalidCase:
    context: DecisionContext
    knowledge: RulesetKnowledge
    config: NeuralEncoderConfig
    message: str


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            _InvalidCase(
                replace(_context(), cash_by_seat=(30, 101, 25)),
                canonical_knowledge(3),
                stage1_encoder_config(),
                "cash",
            ),
            id="cash",
        ),
        pytest.param(
            _InvalidCase(
                replace(_context(), value_chart=(0, 4, 8, 12, 16, 21)),
                replace(
                    canonical_knowledge(3),
                    value_chart=(0, 4, 8, 12, 16, 21),
                ),
                stage1_encoder_config(),
                "chart",
            ),
            id="chart",
        ),
        pytest.param(
            _InvalidCase(
                _context(),
                replace(
                    canonical_knowledge(3),
                    resource_counts=(7, 6, 6, 6, 6),
                ),
                stage1_encoder_config(),
                "resource",
            ),
            id="resource-counts",
        ),
        pytest.param(
            _InvalidCase(
                _context(),
                replace(
                    canonical_knowledge(3),
                    action_counts=(13, 8, 3, 2, 3, 2),
                ),
                stage1_encoder_config(),
                "action",
            ),
            id="action-counts",
        ),
        pytest.param(
            _InvalidCase(
                replace(
                    _context(),
                    current_hand_suit_ids=(1, 2, 3, 4, 5, 1),
                    revealable_count=6,
                ),
                canonical_knowledge(3),
                stage1_encoder_config(),
                "hand",
            ),
            id="hand-size",
        ),
        pytest.param(
            _InvalidCase(
                replace(_context(), legal_max_amount=101),
                canonical_knowledge(3),
                stage1_encoder_config(),
                "bid",
            ),
            id="bid-maximum",
        ),
        pytest.param(
            _InvalidCase(
                _context(),
                canonical_knowledge(3),
                replace(stage1_encoder_config(), max_history_events=75),
                "history bound",
            ),
            id="ruleset-history-bound",
        ),
    ),
)
def test_values_outside_checkpoint_bounds_are_rejected(case: _InvalidCase) -> None:
    with pytest.raises(NeuralEncodingError, match=case.message):
        _encoder(config=case.config).encode(case.context, case.knowledge, _history())


def test_negative_private_cards_per_player_is_rejected() -> None:
    knowledge = replace(
        canonical_knowledge(3),
        private_cards_per_player=-1,
    )

    with pytest.raises(NeuralEncodingError, match="hand"):
        _encoder().encode(_context(), knowledge, _history())


def test_history_bids_use_bid_bound_while_remaining_cash_normalized() -> None:
    config = replace(
        stage1_encoder_config(),
        max_bid=10,
        max_cash=100,
    )
    bounds = EnvironmentBounds(max_bid=10, max_hand_size=5)
    encoder = NeuralObservationEncoder(config, bounds)
    history = _history(bids_by_seat=(2, 20, 1))

    with pytest.raises(NeuralEncodingError, match="history bid"):
        encoder.encode(_context(), canonical_knowledge(3), history)


class _FixedMaskCodec(ActionCodec):
    def __init__(
        self,
        bounds: EnvironmentBounds,
        mask: NDArray[np.int8],
    ) -> None:
        super().__init__(bounds)
        self._mask = mask

    def mask(
        self,
        context: DecisionContext,
    ) -> NDArray[np.int8]:
        del context
        return self._mask.copy()


@pytest.mark.parametrize(
    "mask",
    (
        np.zeros(106, dtype=np.int8),
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                np.ones(1, dtype=np.int8),
                np.zeros(104, dtype=np.int8),
            )
        ),
    ),
    ids=("all-zero", "missing-pass"),
)
def test_encoder_rejects_invalid_universal_action_masks(
    mask: NDArray[np.int8],
) -> None:
    codec = _FixedMaskCodec(_BOUNDS, mask)

    with pytest.raises(NeuralEncodingError, match="action mask"):
        _encoder(action_codec=codec).encode(
            _context(),
            canonical_knowledge(3),
            _history(),
        )


@pytest.mark.parametrize(
    ("context", "knowledge", "message"),
    (
        pytest.param(
            _context(player_count=4, learner_seat=0),
            canonical_knowledge(4),
            "player count",
            id="player-count",
        ),
        pytest.param(
            _context(),
            replace(canonical_knowledge(3), name="live-B"),
            "ruleset",
            id="ruleset",
        ),
    ),
)
def test_stage1_encoder_rejects_inputs_outside_live_a_three_player_support(
    context: DecisionContext,
    knowledge: RulesetKnowledge,
    message: str,
) -> None:
    history = _history(player_count=context.player_count)

    with pytest.raises(NeuralEncodingError, match=message):
        _encoder().encode(context, knowledge, history)


def test_batch_observations_stacks_every_field_on_requested_device() -> None:
    encoder = _encoder()
    observation = encoder.encode(
        _context(),
        canonical_knowledge(3),
        _history(),
    )

    batch = batch_observations((observation, observation), torch.device("cpu"))

    assert batch.global_ids.shape == (2, 6)
    assert batch.global_numeric.shape == (2, 21)
    assert batch.objective_bits.shape == (2, 60)
    assert batch.seat_numeric.shape == (2, 5, 41)
    assert batch.seat_valid.shape == (2, 5)
    assert batch.private_hand_ids.shape == (2, 5)
    assert batch.hand_valid.shape == (2, 5)
    assert batch.history_ids.shape == (2, 76, 6)
    assert batch.history_numeric.shape == (2, 76, 42)
    assert batch.history_valid.shape == (2, 76)
    assert batch.action_mask.shape == (2, 106)
    assert batch.global_ids.dtype == torch.int64
    assert batch.global_numeric.dtype == torch.float32
    assert batch.action_mask.dtype == torch.bool
    assert batch.global_ids.device == torch.device("cpu")
    assert torch.equal(batch.global_ids[0], torch.as_tensor(observation.global_ids))
