from __future__ import annotations

import pytest
from pocketrocks import BotDecision, DecisionContext, PocketRocksBot
from pocketrocks.sim import LocalGame, ScoreRow, SimEngine

from garboid_pocketrocks import simulator
from garboid_pocketrocks.simulator.errors import ActingSeatsError, IllegalDecisionError
from garboid_pocketrocks.simulator.seeding import derive_seed
from garboid_pocketrocks.simulator.session import SdkGameSession, _result


class _PassBot(PocketRocksBot):
    name: str

    async def choose_decision(self, context: DecisionContext) -> BotDecision:
        del context
        return BotDecision.pass_turn()


def _pass_pending(session: SdkGameSession) -> None:
    session.step({seat: BotDecision.pass_turn() for seat in session.pending.acting_seats})


def test_seed_derivation_has_fixed_cross_process_values() -> None:
    assert derive_seed(42, "game", 0) == 11218325959786588961
    assert derive_seed(42, "brain", 7) == 7511519613013083805


def test_session_is_the_public_simulator_engine_boundary() -> None:
    assert simulator.SdkGameSession is SdkGameSession


def test_start_exposes_one_sdk_bid_context_per_seat() -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=23,
        value_chart="A",
    )

    assert session.pending.acting_seats == (0, 1, 2)
    assert session.pending.decision_kind == "submitBid"
    assert tuple(context.bot_seat for _, context in session.pending.contexts) == (0, 1, 2)
    assert all(context.decision_kind == "submitBid" for _, context in session.pending.contexts)
    assert session.snapshot.turn_index == 0
    assert not session.terminated


def test_step_rejects_missing_seats_without_mutating_the_sdk_engine() -> None:
    session = SdkGameSession.start(player_count=3, seed=23)
    before = session.snapshot

    with pytest.raises(ActingSeatsError, match=r"expected seats=\[0, 1, 2\]"):
        session.step({0: BotDecision.pass_turn()})

    assert session.snapshot == before


def test_step_rejects_illegal_decision_without_mutating_the_sdk_engine() -> None:
    session = SdkGameSession.start(player_count=3, seed=23)
    before = session.snapshot

    with pytest.raises(IllegalDecisionError, match="seat=0"):
        session.step(
            {
                0: BotDecision.submit_bid(10_000),
                1: BotDecision.pass_turn(),
                2: BotDecision.pass_turn(),
            }
        )

    assert session.snapshot == before


def test_session_uses_choice_reveals_and_skips_automatic_reveal_decisions() -> None:
    session = SdkGameSession.start(player_count=3, seed=31)
    choice_reveal_steps = 0

    while not session.terminated:
        if session.pending.decision_kind == "selectInfoToReveal":
            choice_reveal_steps += 1
            assert len(session.pending.acting_seats) == 1
        _pass_pending(session)

    reveals = tuple(turn.reveal for turn in session.history if turn.reveal is not None)
    automatic_reveals = tuple(reveal for reveal in reveals if reveal.auto)
    choice_reveals = tuple(reveal for reveal in reveals if not reveal.auto)

    assert automatic_reveals
    assert choice_reveal_steps == len(choice_reveals)
    assert len(reveals) > choice_reveal_steps


def test_session_matches_sdk_local_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POCKETROCKS_SKIP_VERSION_CHECK", "1")
    session = SdkGameSession.start(
        player_count=3,
        seed=42,
        value_chart="E",
        player_names=("Pass A", "Pass B", "Pass C"),
    )
    while not session.terminated:
        _pass_pending(session)
    bots = [_PassBot(), _PassBot(), _PassBot()]
    for bot, name in zip(bots, ("Pass A", "Pass B", "Pass C"), strict=True):
        bot.name = name
    expected = LocalGame(
        bots,
        seed=42,
        value_chart="E",
    ).play()

    assert session.result is not None
    assert session.result.rows == expected.scores
    assert session.result.ranking == expected.ranking
    assert session.history == expected.history


def test_result_uses_competition_ranks_for_equal_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimEngine(3, "equal-total-result")
    rows = [
        ScoreRow(0, "A", 30, 0, 0, 0, 0, 30),
        ScoreRow(1, "B", 30, 0, 0, 0, 0, 30),
        ScoreRow(2, "C", 20, 0, 0, 0, 0, 20),
    ]
    monkeypatch.setattr(engine, "score", lambda: rows)
    monkeypatch.setattr(engine, "ranking", lambda: [0, 1, 2])

    result = _result(engine)

    assert result.ranking == (0, 1, 2)
    assert tuple(score.rank for score in result.scores) == (1, 1, 3)


@pytest.mark.parametrize("player_count", (3, 4, 5))
def test_same_seed_produces_the_same_complete_game(player_count: int) -> None:
    left = SdkGameSession.start(player_count=player_count, seed=104_729)
    right = SdkGameSession.start(player_count=player_count, seed=104_729)

    while not left.terminated:
        _pass_pending(left)
    while not right.terminated:
        _pass_pending(right)

    assert left.snapshot == right.snapshot
    assert left.result == right.result
    assert left.history == right.history
