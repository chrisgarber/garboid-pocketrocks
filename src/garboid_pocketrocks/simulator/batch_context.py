"""Build deterministic bot contexts directly from one SDK batch-engine row."""

from __future__ import annotations

import uuid
from typing import Literal

from pocketrocks import DecisionContext
from pocketrocks.sim import BatchSimEngine
from pocketrocks.sim.constants import STARTING_CASH

type DecisionKind = Literal["submitBid", "selectInfoToReveal"]

_SIM_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "pocketrocks-sim")
_DETERMINISTIC_DEADLINE = 2**63 - 1


def build_batch_context(
    engine: BatchSimEngine,
    *,
    row: int,
    seat: int,
    decision_kind: DecisionKind,
    action_id: int,
    resource_ids: tuple[int, int],
    turn_index: int,
    legal_max_amount: int | None,
) -> DecisionContext:
    """Materialize the same public state as the scalar SDK context builder."""

    hand = tuple(int(card) for card in engine.hand_cards[row, seat] if card > 0)
    request_id = str(
        uuid.uuid5(
            _SIM_NAMESPACE,
            f"{engine.seeds[row]}:{turn_index}:{seat}:{decision_kind}",
        )
    )
    return DecisionContext(
        request_id=request_id,
        deadline_at=_DETERMINISTIC_DEADLINE,
        received_at=0,
        decision_kind=decision_kind,
        player_count=engine.player_count,
        starting_cash=STARTING_CASH[engine.player_count],
        value_chart=tuple(int(value) for value in engine.value_charts[row]),
        objective_ids=tuple(
            int(objective_id) for objective_id in engine.objective_ids[row] if objective_id > 0
        ),
        current_action_id=action_id,
        current_resource_ids=resource_ids,
        cash_by_seat=tuple(int(value) for value in engine.cash[row]),
        tiebreak_seat=int(engine.tiebreak_seats[row]),
        won_resource_counts_by_seat=tuple(
            tuple(int(value) for value in counts) for counts in engine.won_counts[row]
        ),
        revealed_info_counts_by_seat=tuple(
            tuple(int(value) for value in counts) for counts in engine.revealed_counts[row]
        ),
        owned_objective_ids_by_seat=tuple(
            tuple(
                objective_index
                for objective_index, owned in enumerate(objectives, start=1)
                if owned
            )
            for objectives in engine.owned_objectives[row]
        ),
        bot_seat=seat,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max_amount,
        revealable_count=len(hand),
        metadata={},
    )
