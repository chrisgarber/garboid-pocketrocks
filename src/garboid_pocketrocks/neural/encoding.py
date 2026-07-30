"""Learner-relative neural observations built only from deployable information."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from pocketrocks import ActionId, DecisionContext, Suit
from torch import Tensor

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.knowledge import RulesetKnowledge
from garboid_pocketrocks.neural.config import NeuralEncoderConfig
from garboid_pocketrocks.training.actions import ActionCodec
from garboid_pocketrocks.training.bounds import EnvironmentBounds

_MAX_SEATS = 5
_MAX_OBJECTIVES = 30
_GLOBAL_ID_SIZE = 6
_GLOBAL_NUMERIC_SIZE = 21
_OBJECTIVE_BIT_SIZE = 60
_SEAT_NUMERIC_SIZE = 41
_HISTORY_ID_SIZE = 6
_HISTORY_NUMERIC_SIZE = 42

_PHASE_IDS = {
    "submitBid": 1,
    "selectInfoToReveal": 2,
}
_EVENT_KIND_IDS = {
    PublicEventKind.GAME_SETUP: 1,
    PublicEventKind.TURN_OPENED: 2,
    PublicEventKind.AUCTION_RESOLVED: 3,
    PublicEventKind.INFORMATION_REVEALED: 4,
}


class NeuralEncodingError(ValueError):
    """Raised when public inputs exceed a checkpoint's declared support."""


@dataclass(frozen=True, slots=True)
class NeuralObservation:
    """One fixed-shape NumPy observation."""

    global_ids: NDArray[np.int64]
    global_numeric: NDArray[np.float32]
    objective_bits: NDArray[np.float32]
    seat_numeric: NDArray[np.float32]
    seat_valid: NDArray[np.bool_]
    private_hand_ids: NDArray[np.int64]
    hand_valid: NDArray[np.bool_]
    history_ids: NDArray[np.int64]
    history_numeric: NDArray[np.float32]
    history_valid: NDArray[np.bool_]
    action_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class NeuralBatch:
    """A device-resident batch of fixed-shape neural observations."""

    global_ids: Tensor
    global_numeric: Tensor
    objective_bits: Tensor
    seat_numeric: Tensor
    seat_valid: Tensor
    private_hand_ids: Tensor
    hand_valid: Tensor
    history_ids: Tensor
    history_numeric: Tensor
    history_valid: Tensor
    action_mask: Tensor


class NeuralObservationEncoder:
    """Encode current public state and full public history relative to the learner."""

    def __init__(
        self,
        config: NeuralEncoderConfig,
        bounds: EnvironmentBounds,
        *,
        action_codec: ActionCodec | None = None,
    ) -> None:
        expected_bounds = EnvironmentBounds(
            max_bid=config.max_bid,
            max_hand_size=config.max_hand_size,
        )
        if bounds != expected_bounds:
            raise NeuralEncodingError("environment bounds do not match checkpoint config")
        codec = action_codec or ActionCodec(bounds)
        if codec.bounds != bounds:
            raise NeuralEncodingError("action codec bounds do not match environment bounds")
        self.config = config
        self.bounds = bounds
        self.action_codec = codec

    def encode(
        self,
        context: DecisionContext,
        knowledge: RulesetKnowledge,
        history: PublicHistory,
    ) -> NeuralObservation:
        """Validate and encode one learner decision without hidden game state."""

        self._validate(context, knowledge, history)
        action_mask = self._action_mask(context)
        player_count = context.player_count
        learner_seat = context.bot_seat

        global_ids = np.asarray(
            (
                _PHASE_IDS[context.decision_kind],
                player_count,
                context.current_action_id or 0,
                context.current_resource_ids[0],
                context.current_resource_ids[1],
                _relative_seat(context.tiebreak_seat, learner_seat, player_count),
            ),
            dtype=np.int64,
        )
        global_numeric = np.asarray(
            (
                context.starting_cash / self.config.max_cash,
                *(value / self.config.max_abs_chart for value in context.value_chart),
                *(count / self.config.max_resource_cards for count in knowledge.resource_counts),
                *(count / self.config.max_action_cards for count in knowledge.action_counts),
                knowledge.private_cards_per_player / self.config.max_hand_size,
                knowledge.active_objective_count / _MAX_OBJECTIVES,
                float(knowledge.objectives_enabled),
            ),
            dtype=np.float32,
        )
        objective_bits = np.concatenate(
            (
                _objective_bits(context.objective_ids),
                _objective_bits(knowledge.objective_pool),
            )
        ).astype(np.float32, copy=False)

        seat_numeric = np.zeros((_MAX_SEATS, _SEAT_NUMERIC_SIZE), dtype=np.float32)
        seat_valid = np.zeros(_MAX_SEATS, dtype=np.bool_)
        for relative_seat in range(player_count):
            absolute_seat = (learner_seat + relative_seat) % player_count
            seat_valid[relative_seat] = True
            seat_numeric[relative_seat, 0] = (
                context.cash_by_seat[absolute_seat] / self.config.max_cash
            )
            seat_numeric[relative_seat, 1:6] = (
                np.asarray(
                    context.won_resource_counts_by_seat[absolute_seat],
                    dtype=np.float32,
                )
                / self.config.max_resource_cards
            )
            seat_numeric[relative_seat, 6:11] = (
                np.asarray(
                    context.revealed_info_counts_by_seat[absolute_seat],
                    dtype=np.float32,
                )
                / self.config.max_resource_cards
            )
            seat_numeric[relative_seat, 11:] = _objective_bits(
                context.owned_objective_ids_by_seat[absolute_seat]
            )

        private_hand_ids = np.zeros(self.config.max_hand_size, dtype=np.int64)
        private_hand_ids[: len(context.current_hand_suit_ids)] = context.current_hand_suit_ids
        hand_valid = np.zeros(self.config.max_hand_size, dtype=np.bool_)
        hand_valid[: len(context.current_hand_suit_ids)] = True

        history_ids, history_numeric, history_valid = self._encode_history(
            history,
            learner_seat=learner_seat,
            player_count=player_count,
        )
        return NeuralObservation(
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

    def _validate(
        self,
        context: DecisionContext,
        knowledge: RulesetKnowledge,
        history: PublicHistory,
    ) -> None:
        config = self.config
        if len(history) > config.max_history_events:
            raise NeuralEncodingError("history exceeds checkpoint bound")
        if knowledge.name not in config.supported_ruleset_names:
            raise NeuralEncodingError("ruleset is outside checkpoint support")
        if context.player_count not in config.supported_player_counts:
            raise NeuralEncodingError("player count is outside checkpoint support")
        if knowledge.player_count != context.player_count:
            raise NeuralEncodingError("ruleset and context player counts differ")
        if not 0 <= context.bot_seat < context.player_count:
            raise NeuralEncodingError("learner seat is outside player count")
        if not 0 <= context.tiebreak_seat < context.player_count:
            raise NeuralEncodingError("priority seat is outside player count")
        if context.decision_kind not in _PHASE_IDS:
            raise NeuralEncodingError("unsupported decision phase")

        self._validate_context_shapes(context)
        self._validate_knowledge_shapes(knowledge)

        if context.starting_cash != knowledge.starting_cash:
            raise NeuralEncodingError("context and ruleset starting cash differ")
        if context.value_chart != knowledge.value_chart:
            raise NeuralEncodingError("context and ruleset value chart differ")
        if context.starting_cash < 0 or context.starting_cash > config.max_cash:
            raise NeuralEncodingError("starting cash exceeds checkpoint cash bound")
        _require_values(
            "cash",
            context.cash_by_seat,
            minimum=0,
            maximum=config.max_cash,
        )
        _require_abs_values(
            "chart",
            context.value_chart,
            maximum=config.max_abs_chart,
        )
        if sum(knowledge.resource_counts) > config.max_resource_cards:
            raise NeuralEncodingError("resource counts exceed checkpoint bound")
        _require_values(
            "resource count",
            knowledge.resource_counts,
            minimum=0,
            maximum=config.max_resource_cards,
        )
        if sum(knowledge.action_counts) > config.max_action_cards:
            raise NeuralEncodingError("action counts exceed checkpoint bound")
        _require_values(
            "action count",
            knowledge.action_counts,
            minimum=0,
            maximum=config.max_action_cards,
        )
        if not 0 <= knowledge.private_cards_per_player <= config.max_hand_size:
            raise NeuralEncodingError("ruleset hand size exceeds checkpoint bound")
        if len(context.current_hand_suit_ids) > config.max_hand_size:
            raise NeuralEncodingError("private hand exceeds checkpoint hand bound")
        if not 0 <= context.revealable_count <= config.max_hand_size:
            raise NeuralEncodingError("revealable hand size exceeds checkpoint bound")
        if context.legal_max_amount is not None and not (
            0 <= context.legal_max_amount <= config.max_bid
        ):
            raise NeuralEncodingError("bid maximum exceeds checkpoint bound")

        _require_values(
            "won resource count",
            (count for row in context.won_resource_counts_by_seat for count in row),
            minimum=0,
            maximum=config.max_resource_cards,
        )
        _require_values(
            "revealed resource count",
            (count for row in context.revealed_info_counts_by_seat for count in row),
            minimum=0,
            maximum=config.max_resource_cards,
        )
        _require_suit_ids("private hand", context.current_hand_suit_ids, allow_missing=False)
        _require_suit_ids("current resources", context.current_resource_ids, allow_missing=True)
        _require_objective_ids("active objectives", context.objective_ids)
        _require_objective_ids("objective pool", knowledge.objective_pool)
        for objectives in context.owned_objective_ids_by_seat:
            _require_objective_ids("owned objectives", objectives)
        if context.current_action_id is not None and not (
            1 <= context.current_action_id <= len(ActionId)
        ):
            raise NeuralEncodingError("current action ID is unknown")

        required_history = (
            1
            + (2 * sum(knowledge.action_counts))
            + (context.player_count * knowledge.private_cards_per_player)
        )
        if required_history > config.max_history_events:
            raise NeuralEncodingError("checkpoint history bound cannot cover ruleset")
        self._validate_history(history, context)

    def _validate_context_shapes(self, context: DecisionContext) -> None:
        player_count = context.player_count
        if len(context.value_chart) != 6:
            raise NeuralEncodingError("value chart must contain six entries")
        if len(context.current_resource_ids) != 2:
            raise NeuralEncodingError("current resources must contain two entries")
        if len(context.cash_by_seat) != player_count:
            raise NeuralEncodingError("cash rows must match player count")
        if len(context.won_resource_counts_by_seat) != player_count or any(
            len(row) != len(Suit) for row in context.won_resource_counts_by_seat
        ):
            raise NeuralEncodingError("won resource rows must match player and suit counts")
        if len(context.revealed_info_counts_by_seat) != player_count or any(
            len(row) != len(Suit) for row in context.revealed_info_counts_by_seat
        ):
            raise NeuralEncodingError("revealed resource rows must match player and suit counts")
        if len(context.owned_objective_ids_by_seat) != player_count:
            raise NeuralEncodingError("objective ownership rows must match player count")

    def _validate_knowledge_shapes(self, knowledge: RulesetKnowledge) -> None:
        if len(knowledge.resource_counts) != len(Suit):
            raise NeuralEncodingError("ruleset requires five resource counts")
        if len(knowledge.action_counts) != len(ActionId):
            raise NeuralEncodingError("ruleset requires six action counts")
        if len(knowledge.value_chart) != 6:
            raise NeuralEncodingError("ruleset value chart must contain six entries")
        if not 0 <= knowledge.active_objective_count <= len(knowledge.objective_pool):
            raise NeuralEncodingError("active objective count exceeds objective pool")

    def _validate_history(
        self,
        history: PublicHistory,
        context: DecisionContext,
    ) -> None:
        if not history or not isinstance(history[0], PublicGameSetup):
            raise NeuralEncodingError("history must begin with game setup")
        setup = history[0]
        if (
            setup.player_count != context.player_count
            or setup.starting_cash != context.starting_cash
            or setup.value_chart != context.value_chart
        ):
            raise NeuralEncodingError("history setup does not match current context")
        if not 0 <= setup.initial_tiebreak_seat < context.player_count:
            raise NeuralEncodingError("history priority seat is outside player count")
        _require_abs_values(
            "history chart",
            setup.value_chart,
            maximum=self.config.max_abs_chart,
        )
        _require_objective_ids("history objectives", setup.objective_ids)

        for event in history[1:]:
            if isinstance(event, PublicTurnOpened):
                if not 1 <= event.action_id <= len(ActionId):
                    raise NeuralEncodingError("history action ID is unknown")
                _require_suit_ids(
                    "history resources",
                    event.resource_ids,
                    allow_missing=True,
                )
            elif isinstance(event, PublicAuctionResolved):
                if len(event.bids_by_seat) != context.player_count:
                    raise NeuralEncodingError("history bids must match player count")
                _require_values(
                    "history bid",
                    event.bids_by_seat,
                    minimum=0,
                    maximum=self.config.max_bid,
                )
            elif isinstance(event, PublicInformationRevealed):
                if not 0 <= event.seat < context.player_count:
                    raise NeuralEncodingError("history actor is outside player count")
                _require_suit_ids(
                    "history revealed suit",
                    (event.suit_id,),
                    allow_missing=False,
                )
            elif isinstance(event, PublicGameSetup):
                raise NeuralEncodingError("history contains multiple game setup events")
            else:
                raise NeuralEncodingError("history contains an unsupported event")

    def _action_mask(self, context: DecisionContext) -> NDArray[np.bool_]:
        try:
            raw_mask = self.action_codec.mask(context)
        except ValueError as error:
            raise NeuralEncodingError("action mask could not be encoded") from error
        if raw_mask.shape != (self.action_codec.size,):
            raise NeuralEncodingError("action mask has the wrong shape")
        if not np.logical_or(raw_mask == 0, raw_mask == 1).all():
            raise NeuralEncodingError("action mask must be binary")
        action_mask = raw_mask.astype(np.bool_, copy=False)
        if not action_mask.any():
            raise NeuralEncodingError("action mask must enable an action")
        if not action_mask[0]:
            raise NeuralEncodingError("action mask must enable universal pass")
        return action_mask

    def _encode_history(
        self,
        history: PublicHistory,
        *,
        learner_seat: int,
        player_count: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.bool_]]:
        ids = np.zeros(
            (self.config.max_history_events, _HISTORY_ID_SIZE),
            dtype=np.int64,
        )
        numeric = np.zeros(
            (self.config.max_history_events, _HISTORY_NUMERIC_SIZE),
            dtype=np.float32,
        )
        valid = np.zeros(self.config.max_history_events, dtype=np.bool_)

        for index, event in enumerate(history):
            valid[index] = True
            ids[index, 0] = _EVENT_KIND_IDS[event.kind]
            if isinstance(event, PublicGameSetup):
                ids[index, 4] = (
                    _relative_seat(
                        event.initial_tiebreak_seat,
                        learner_seat,
                        player_count,
                    )
                    + 1
                )
                numeric[index, 0] = event.starting_cash / self.config.max_cash
                numeric[index, 1:7] = (
                    np.asarray(
                        event.value_chart,
                        dtype=np.float32,
                    )
                    / self.config.max_abs_chart
                )
                numeric[index, 7:37] = _objective_bits(event.objective_ids)
            elif isinstance(event, PublicTurnOpened):
                ids[index, 1] = event.action_id
                ids[index, 2:4] = event.resource_ids
            elif isinstance(event, PublicAuctionResolved):
                for relative_seat in range(player_count):
                    absolute_seat = (learner_seat + relative_seat) % player_count
                    numeric[index, 37 + relative_seat] = (
                        event.bids_by_seat[absolute_seat] / self.config.max_cash
                    )
            elif isinstance(event, PublicInformationRevealed):
                ids[index, 4] = _relative_seat(event.seat, learner_seat, player_count) + 1
                ids[index, 5] = event.suit_id
        return ids, numeric, valid


def batch_observations(
    observations: Sequence[NeuralObservation],
    device: torch.device,
) -> NeuralBatch:
    """Stack NumPy observations and place all fields on one Torch device."""

    if not observations:
        raise NeuralEncodingError("cannot batch zero observations")
    return NeuralBatch(
        global_ids=torch.as_tensor(
            np.stack([observation.global_ids for observation in observations]),
            device=device,
        ),
        global_numeric=torch.as_tensor(
            np.stack([observation.global_numeric for observation in observations]),
            device=device,
        ),
        objective_bits=torch.as_tensor(
            np.stack([observation.objective_bits for observation in observations]),
            device=device,
        ),
        seat_numeric=torch.as_tensor(
            np.stack([observation.seat_numeric for observation in observations]),
            device=device,
        ),
        seat_valid=torch.as_tensor(
            np.stack([observation.seat_valid for observation in observations]),
            device=device,
        ),
        private_hand_ids=torch.as_tensor(
            np.stack([observation.private_hand_ids for observation in observations]),
            device=device,
        ),
        hand_valid=torch.as_tensor(
            np.stack([observation.hand_valid for observation in observations]),
            device=device,
        ),
        history_ids=torch.as_tensor(
            np.stack([observation.history_ids for observation in observations]),
            device=device,
        ),
        history_numeric=torch.as_tensor(
            np.stack([observation.history_numeric for observation in observations]),
            device=device,
        ),
        history_valid=torch.as_tensor(
            np.stack([observation.history_valid for observation in observations]),
            device=device,
        ),
        action_mask=torch.as_tensor(
            np.stack([observation.action_mask for observation in observations]),
            device=device,
        ),
    )


def _relative_seat(
    absolute_seat: int,
    learner_seat: int,
    player_count: int,
) -> int:
    return (absolute_seat - learner_seat) % player_count


def _objective_bits(ids: tuple[int, ...]) -> NDArray[np.float32]:
    output = np.zeros(_MAX_OBJECTIVES, dtype=np.float32)
    for identifier in ids:
        output[identifier - 1] = 1.0
    return output


def _require_values(
    name: str,
    values: Iterable[int],
    *,
    minimum: int,
    maximum: int,
) -> None:
    for value in values:
        if not minimum <= value <= maximum:
            raise NeuralEncodingError(f"{name} exceeds checkpoint bound")


def _require_abs_values(
    name: str,
    values: Sequence[int],
    *,
    maximum: int,
) -> None:
    for value in values:
        if abs(value) > maximum:
            raise NeuralEncodingError(f"{name} exceeds checkpoint bound")


def _require_suit_ids(
    name: str,
    ids: Sequence[int],
    *,
    allow_missing: bool,
) -> None:
    minimum = 0 if allow_missing else 1
    if any(not minimum <= identifier <= len(Suit) for identifier in ids):
        raise NeuralEncodingError(f"{name} contains an unknown suit ID")


def _require_objective_ids(name: str, ids: Sequence[int]) -> None:
    if len(set(ids)) != len(ids) or any(
        not 1 <= identifier <= _MAX_OBJECTIVES for identifier in ids
    ):
        raise NeuralEncodingError(f"{name} contains an unknown objective ID")
