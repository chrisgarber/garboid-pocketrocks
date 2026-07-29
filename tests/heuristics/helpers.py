from __future__ import annotations

from pocketrocks import OBJECTIVES, ActionId, DecisionContext

from garboid_pocketrocks.rules import RulesetKnowledge


def make_knowledge(
    *,
    player_count: int = 3,
    starting_cash: int = 30,
    private_cards: int = 0,
    resource_counts: tuple[int, ...] = (2, 2, 2, 2, 2),
    value_chart: tuple[int, ...] = (0, 4, 8, 12, 16, 20),
) -> RulesetKnowledge:
    return RulesetKnowledge(
        name="heuristic-test",
        player_count=player_count,
        starting_cash=starting_cash,
        private_cards_per_player=private_cards,
        resource_counts=resource_counts,
        action_counts=(12, 8, 3, 2, 3, 2),
        value_chart=value_chart,
        objective_pool=tuple(sorted(OBJECTIVES)),
        active_objective_count=0,
        objectives_enabled=False,
    )


def make_context(
    *,
    decision_kind: str = "submitBid",
    action_id: ActionId | None = ActionId.AUCTION1,
    current_resources: tuple[int, int] = (1, 0),
    cash: tuple[int, ...] = (30, 30, 30),
    won: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    revealed: tuple[tuple[int, ...], ...] = ((0, 0, 0, 0, 0),) * 3,
    owned_objectives: tuple[tuple[int, ...], ...] = ((), (), ()),
    objectives: tuple[int, ...] = (),
    hand: tuple[int, ...] = (),
    legal_max: int | None = 30,
    bot_seat: int = 0,
    player_count: int = 3,
    starting_cash: int = 30,
    value_chart: tuple[int, ...] = (0, 4, 8, 12, 16, 20),
    revealable_count: int | None = None,
) -> DecisionContext:
    return DecisionContext(
        request_id="heuristic-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=player_count,
        starting_cash=starting_cash,
        value_chart=value_chart,
        objective_ids=objectives,
        current_action_id=int(action_id) if action_id is not None else None,
        current_resource_ids=current_resources,
        cash_by_seat=cash,
        tiebreak_seat=player_count - 1,
        won_resource_counts_by_seat=won,
        revealed_info_counts_by_seat=revealed,
        owned_objective_ids_by_seat=owned_objectives,
        bot_seat=bot_seat,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand) if revealable_count is None else revealable_count,
    )
