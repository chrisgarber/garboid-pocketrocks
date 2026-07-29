from collections import Counter

from pocketrocks import OBJECTIVES, ActionId, BotDecision, Suit

from garboid_pocketrocks.simulator.context import DecisionBatch
from garboid_pocketrocks.simulator.engine import EngineTransition
from garboid_pocketrocks.simulator.model import (
    GameResult,
    GameState,
    Phase,
    ResourceCard,
    Score,
)


def all_resource_cards(state: GameState) -> tuple[ResourceCard, ...]:
    return (
        *state.resource_deck,
        *state.visible_resources,
        *(
            card
            for player in state.players
            for card in (
                *player.private_hand,
                *player.revealed_info,
                *player.won_resources,
            )
        ),
    )


def assert_resource_conservation(state: GameState) -> None:
    all_cards = all_resource_cards(state)
    expected_counts = dict(
        zip(
            Suit,
            state.ruleset.resource_counts,
            strict=True,
        )
    )

    assert len(all_cards) == sum(state.ruleset.resource_counts)
    assert len({card.card_id for card in all_cards}) == len(all_cards)
    assert Counter(card.suit for card in all_cards) == expected_counts


def assert_objective_ownership(state: GameState) -> None:
    active_objectives = set(state.active_objective_ids)
    owned_objectives = [
        objective_id
        for player in state.players
        for objective_id in player.owned_objective_ids
    ]

    assert len(active_objectives) == len(state.active_objective_ids)
    assert set(owned_objectives) <= active_objectives
    assert all(count == 1 for count in Counter(owned_objectives).values())


def assert_cash_is_nonnegative(state: GameState) -> None:
    assert all(player.cash >= 0 for player in state.players)


def _calculated_legal_max(state: GameState, seat: int) -> int:
    action = state.current_action
    cash = state.players[seat].cash
    if action is None:
        return cash
    if action.action_id is ActionId.LOAN10:
        return cash + 10
    if action.action_id is ActionId.LOAN20:
        return cash + 20
    return cash


def assert_pending_accepts_calculated_legal_range(
    state: GameState,
    pending: DecisionBatch,
) -> None:
    for seat, context in pending.contexts:
        assert context.is_legal(BotDecision.pass_turn())
        if state.phase is Phase.BIDDING:
            legal_max = _calculated_legal_max(state, seat)
            assert context.legal_max_amount == legal_max
            for amount in range(legal_max + 1):
                assert context.is_legal(BotDecision.submit_bid(amount))
        else:
            revealable_count = len(state.players[seat].private_hand)
            assert context.revealable_count == revealable_count
            for index in range(revealable_count):
                assert context.is_legal(BotDecision.select_info_to_reveal(index))


def recompute_result(state: GameState) -> GameResult:
    revealed_by_suit = tuple(
        sum(card.suit is suit for player in state.players for card in player.revealed_info)
        for suit in Suit
    )
    money_by_seat: list[int] = []
    for player in state.players:
        resource_value = sum(
            sum(card.suit is suit for card in player.won_resources)
            * state.ruleset.value_chart[min(revealed_by_suit[int(suit) - 1], 5)]
            for suit in Suit
        )
        objective_value = sum(
            OBJECTIVES[objective_id].payout
            for objective_id in player.owned_objective_ids
        )
        money_by_seat.append(
            player.cash
            + resource_value
            + objective_value
            + sum(
                investment.locked + investment.payout
                for investment in player.investments
            )
            - sum(loan.principal for loan in player.loans)
        )

    return GameResult(
        scores=tuple(
            Score(
                seat=seat,
                final_money=money,
                rank=1 + sum(other > money for other in money_by_seat),
            )
            for seat, money in enumerate(money_by_seat)
        )
    )


def assert_transition_invariants(transition: EngineTransition) -> None:
    assert_resource_conservation(transition.state)
    assert_objective_ownership(transition.state)
    assert_cash_is_nonnegative(transition.state)

    if transition.result is None:
        assert transition.pending is not None
        assert_pending_accepts_calculated_legal_range(
            transition.state,
            transition.pending,
        )
        return

    assert transition.state.phase is Phase.TERMINAL
    assert transition.pending is None
    assert transition.state.resource_deck == ()
    assert transition.state.visible_resources == ()
    assert transition.result == recompute_result(transition.state)
