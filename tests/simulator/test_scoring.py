import pytest
from pocketrocks import BotDecision, Suit

from garboid_pocketrocks.rules import LIVE_RULESET, VALUE_CHARTS, live_ruleset
from garboid_pocketrocks.simulator.engine import GameEngine
from garboid_pocketrocks.simulator.model import (
    GameState,
    InvestmentPosition,
    LoanPosition,
    Phase,
    PlayerState,
    ResourceCard,
)


def test_terminal_scoring_reveals_cards_and_uses_competition_ranks() -> None:
    hidden = ResourceCard(card_id=1, suit=Suit.BRICK)
    owned = ResourceCard(card_id=2, suit=Suit.BRICK)
    state = GameState(
        ruleset=LIVE_RULESET,
        player_count=3,
        seed=1,
        turn_index=9,
        phase=Phase.REVEAL,
        players=(
            PlayerState(
                seat=0,
                cash=10,
                private_hand=(hidden,),
                won_resources=(owned,),
                loans=(LoanPosition(principal=10, winning_bid=0),),
                investments=(InvestmentPosition(locked=3, payout=5),),
                owned_objective_ids=(1,),
            ),
            PlayerState(seat=1, cash=16),
            PlayerState(seat=2, cash=17),
        ),
        resource_deck=(),
        action_deck=(),
        visible_resources=(),
        current_action=None,
        active_objective_ids=(1,),
        priority_seat=0,
        reveal_seat=0,
    )

    transition = GameEngine.step(state, {0: BotDecision.pass_turn()})

    assert transition.terminated
    assert transition.pending is None
    assert transition.state.phase is Phase.TERMINAL
    assert transition.state.players[0].private_hand == ()
    assert transition.state.players[0].revealed_info == (hidden,)
    assert transition.result is not None
    assert tuple(
        (score.seat, score.final_money, score.rank) for score in transition.result.scores
    ) == ((0, 17, 1), (1, 16, 3), (2, 17, 1))


@pytest.mark.parametrize("chart", tuple(VALUE_CHARTS))
def test_terminal_resource_value_uses_selected_chart(chart: str) -> None:
    hidden = ResourceCard(card_id=10, suit=Suit.ORE)
    owned = ResourceCard(card_id=11, suit=Suit.ORE)
    ruleset = live_ruleset(chart)
    state = GameState(
        ruleset=ruleset,
        player_count=3,
        seed=2,
        turn_index=3,
        phase=Phase.REVEAL,
        players=(
            PlayerState(
                seat=0,
                cash=0,
                private_hand=(hidden,),
                won_resources=(owned,),
            ),
            PlayerState(seat=1, cash=0),
            PlayerState(seat=2, cash=0),
        ),
        resource_deck=(),
        action_deck=(),
        visible_resources=(),
        current_action=None,
        active_objective_ids=(),
        priority_seat=0,
        reveal_seat=0,
    )

    transition = GameEngine.step(state, {0: BotDecision.pass_turn()})

    assert transition.result is not None
    assert transition.result.scores[0].final_money == VALUE_CHARTS[chart][1]
