from __future__ import annotations

import pytest
from pocketrocks.sim.state import ScoreRow, TurnRecord

from garboid_pocketrocks.bots import RandomBotBrain
from garboid_pocketrocks.knowledge import canonical_knowledge
from garboid_pocketrocks.simulator.session import (
    PlayerSnapshot,
    SdkGameSession,
    SessionResult,
    SessionScore,
    SessionSnapshot,
    SessionTransition,
)
from garboid_pocketrocks.training.rewards import (
    RewardConfig,
    RewardEventKind,
    RewardTracker,
)


def _player(
    seat: int,
    *,
    cash: int = 30,
    loans: tuple[int, ...] = (),
    investments: tuple[tuple[int, int], ...] = (),
    objectives: tuple[int, ...] = (),
) -> PlayerSnapshot:
    return PlayerSnapshot(
        seat=seat,
        cash=cash,
        hand_suits=(),
        won_suits=(),
        revealed_suits=(),
        loans=loans,
        investments=investments,
        objective_ids=objectives,
    )


def _snapshot(*players: PlayerSnapshot) -> SessionSnapshot:
    return SessionSnapshot(
        turn_index=0,
        tiebreak_seat=2,
        current_action=None,
        game_over=False,
        players=players or (_player(0), _player(1), _player(2)),
    )


def _transition(
    before: SessionSnapshot,
    after: SessionSnapshot,
    *,
    result: SessionResult | None = None,
    turns: tuple[TurnRecord, ...] = (),
) -> SessionTransition:
    return SessionTransition(
        before=before,
        snapshot=after,
        pending=None,
        result=result,
        decisions=(),
        events=(),
        turn_records=turns,
    )


@pytest.mark.parametrize(
    ("after_player", "expected"),
    (
        (_player(0, cash=26), -4 / 30),
        (_player(0, cash=36, loans=(10,)), -4 / 30),
        (_player(0, cash=26, investments=((4, 5),)), 5 / 30),
    ),
)
def test_accounting_potential_matches_each_sdk_position_type(
    after_player: PlayerSnapshot,
    expected: float,
) -> None:
    before = _snapshot()
    after = _snapshot(after_player, _player(1), _player(2))
    tracker = RewardTracker()
    tracker.reset(before)

    rewards = tracker.update(_transition(before, after))

    assert rewards[0].accounting == pytest.approx(expected)


def test_terminal_resource_value_supplies_the_residual_final_money_delta() -> None:
    before = _snapshot()
    result = _result((35, 30, 30), (1, 2, 2))
    tracker = RewardTracker()
    tracker.reset(before)

    rewards = tracker.update(_transition(before, before, result=result))

    assert rewards[0].terminal_resource == pytest.approx(5 / 30)
    assert rewards[0].accounting == 0


def test_tied_winners_split_the_configured_win_bonus_once() -> None:
    before = _snapshot()
    result = _result((30, 30, 25), (1, 1, 3))
    transition = _transition(before, before, result=result)
    tracker = RewardTracker(RewardConfig(win_bonus=2.0, placement_bonuses=(1.0, 0.5)))
    tracker.reset(before)

    rewards = tracker.update(transition)
    second = tracker.update(transition)

    assert rewards[0].placement == pytest.approx(2.0)
    assert rewards[1].placement == pytest.approx(2.0)
    assert rewards[2].placement == 0
    assert second[0].placement == 0


def test_turn_shaping_is_auditable_and_validated() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RewardConfig(event_bonuses=((RewardEventKind.RESOURCES_AWARDED.value, 1.0),) * 2)
    with pytest.raises(ValueError, match="unknown"):
        RewardConfig(event_bonuses=(("not_an_event", 1.0),))

    before = _snapshot()
    turn = TurnRecord(
        turn_index=0,
        action="Auction1",
        upcoming_before=(1, 2),
        raw_bids=(0, 1, 0),
        effective_bids=(0, 1, 0),
        winner_seat=1,
        paid=1,
        bundle_suits=(1,),
        claimed_objective_wire_ids=(),
        reveal=None,
    )
    tracker = RewardTracker(
        RewardConfig(event_bonuses=((RewardEventKind.RESOURCES_AWARDED.value, 0.25),))
    )
    tracker.reset(before)

    rewards = tracker.update(_transition(before, before, turns=(turn,)))

    assert rewards[1].shaping == 0.25
    assert rewards[0].shaping == 0


def test_reward_config_rejects_negative_penalty_and_nonfinite_coefficients() -> None:
    with pytest.raises(ValueError, match="invalid action penalty"):
        RewardConfig(invalid_action_penalty=-1)
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(win_bonus=float("nan"))


@pytest.mark.parametrize("chart", ("A", "E"))
def test_complete_sdk_game_accounting_rewards_reconcile_to_final_money(
    chart: str,
) -> None:
    session = SdkGameSession.start(player_count=3, seed=71, value_chart=chart)
    tracker = RewardTracker(RewardConfig(win_bonus=0))
    tracker.reset(session.snapshot)
    brains = tuple(RandomBotBrain(seed=100 + seat) for seat in range(3))
    knowledge = canonical_knowledge(3, value_chart=chart)
    totals = [0.0, 0.0, 0.0]

    while not session.terminated:
        transition = session.step(
            {
                seat: brains[seat].choose_decision(context, knowledge)
                for seat, context in session.pending.contexts
            }
        )
        rewards = tracker.update(transition)
        for seat, reward in rewards.items():
            totals[seat] += reward.accounting + reward.terminal_resource

    assert session.result is not None
    for score in session.result.scores:
        assert totals[score.seat] == pytest.approx((score.final_money - 30) / 30)


def _result(
    totals: tuple[int, int, int],
    ranks: tuple[int, int, int],
) -> SessionResult:
    return SessionResult(
        scores=tuple(
            SessionScore(seat=seat, final_money=total, rank=ranks[seat])
            for seat, total in enumerate(totals)
        ),
        rows=tuple(
            ScoreRow(
                seat=seat,
                name=f"Bot {seat}",
                cash=total,
                items_value=0,
                objectives_value=0,
                investments_value=0,
                loans_value=0,
                total=total,
            )
            for seat, total in enumerate(totals)
        ),
        ranking=tuple(seat for seat, _rank in sorted(enumerate(ranks), key=lambda item: item[1])),
    )
