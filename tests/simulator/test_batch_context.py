from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pocketrocks import DecisionContext
from pocketrocks.sim import BatchSimEngine, SimEngine
from pocketrocks.sim.constants import ACTION_WIRE_IDS
from pocketrocks.sim.context import build_sim_context

from garboid_pocketrocks.simulator.batch_context import build_batch_context

_DEADLINE = 2**63 - 1


def _deterministic(context: DecisionContext) -> DecisionContext:
    return replace(context, deadline_at=_DEADLINE, received_at=0)


@pytest.mark.parametrize(
    ("player_count", "value_chart", "objectives_enabled"),
    (
        (3, "A", True),
        (4, "E", False),
        (5, "C", True),
    ),
)
def test_batch_context_matches_scalar_context_through_full_game(
    player_count: int,
    value_chart: str,
    objectives_enabled: bool,
) -> None:
    seed = f"batch-context-{player_count}-{value_chart}-{objectives_enabled}"
    scalar = SimEngine(
        player_count,
        seed,
        value_chart=value_chart,
        objectives_enabled=objectives_enabled,
    )
    batch = BatchSimEngine.start(
        player_count=player_count,
        seeds=(seed,),
        value_charts=(value_chart,),
        objectives_enabled=(objectives_enabled,),
    )

    while True:
        scalar_action = scalar.flip_action()
        batch_actions = batch.flip_actions()
        if scalar_action is None:
            assert int(batch_actions[0]) == 0
            break

        action_id = int(batch_actions[0])
        assert action_id == ACTION_WIRE_IDS[scalar_action]
        resource_ids = (
            int(batch.upcoming[0, 0]),
            int(batch.upcoming[0, 1]),
        )
        legal = batch.legal_max_bids()
        bids: list[int] = []
        for seat in range(player_count):
            direct = build_batch_context(
                batch,
                row=0,
                seat=seat,
                decision_kind="submitBid",
                action_id=action_id,
                resource_ids=resource_ids,
                turn_index=scalar.turn_index,
                legal_max_amount=int(legal[0, seat]),
            )
            reconstructed = _deterministic(
                build_sim_context(
                    scalar,
                    seat,
                    "submitBid",
                    budget_ms=60_000,
                    turn_index=scalar.turn_index,
                )
            )
            assert direct == reconstructed
            bids.append((scalar.turn_index + (seat + 1) * 3) % (int(legal[0, seat]) + 1))

        scalar_outcome = scalar.resolve(bids)
        batch_outcome = batch.resolve_bids(np.asarray((bids,), dtype=np.int16))
        assert int(batch_outcome.winner_seats[0]) == scalar_outcome.winner_seat
        assert int(batch_outcome.paid[0]) == scalar_outcome.paid

        reveals = np.full(1, -1, dtype=np.int8)
        if scalar_outcome.reveal_needed == "auto":
            scalar.apply_reveal(scalar_outcome.winner_seat, 0, auto=True)
            reveals[0] = 0
        elif scalar_outcome.reveal_needed == "choice":
            winner = scalar_outcome.winner_seat
            direct = build_batch_context(
                batch,
                row=0,
                seat=winner,
                decision_kind="selectInfoToReveal",
                action_id=action_id,
                resource_ids=resource_ids,
                turn_index=scalar.turn_index - 1,
                legal_max_amount=None,
            )
            reconstructed = _deterministic(
                build_sim_context(
                    scalar,
                    winner,
                    "selectInfoToReveal",
                    budget_ms=60_000,
                    turn_index=scalar.turn_index - 1,
                )
            )
            assert direct == reconstructed
            scalar.apply_reveal(winner, 0, auto=False)
            reveals[0] = 0
        batch.apply_reveals(reveals)

    assert batch.scores().total[0].tolist() == [row.total for row in scalar.score()]
    assert batch.rankings()[0].tolist() == scalar.ranking()
