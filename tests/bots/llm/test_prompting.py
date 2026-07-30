from __future__ import annotations

from dataclasses import replace

from pocketrocks import OBJECTIVES, ActionId, DecisionContext, Suit

from garboid_pocketrocks.bots.llm.prompting import PocketRocksPromptSkill
from garboid_pocketrocks.rules import LIVE_RULESET, RulesetKnowledge


def _knowledge() -> RulesetKnowledge:
    return replace(
        LIVE_RULESET.knowledge(3),
        name="prompt-test",
        active_objective_count=2,
    )


def _context(
    *,
    decision_kind: str = "submitBid",
    legal_max: int | None = 12,
    hand: tuple[int, ...] = (int(Suit.BRICK), int(Suit.WOOD)),
) -> DecisionContext:
    return DecisionContext(
        request_id="llm-prompt-test",
        deadline_at=2**63 - 1,
        received_at=0,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        player_count=3,
        starting_cash=30,
        value_chart=(0, 4, 8, 12, 16, 20),
        objective_ids=(1, 10),
        current_action_id=int(ActionId.AUCTION1),
        current_resource_ids=(int(Suit.ORE), 0),
        cash_by_seat=(30, 22, 17),
        tiebreak_seat=2,
        won_resource_counts_by_seat=(
            (1, 0, 0, 0, 0),
            (0, 2, 0, 0, 0),
            (0, 0, 1, 1, 0),
        ),
        revealed_info_counts_by_seat=(
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 0, 1, 0, 0),
        ),
        owned_objective_ids_by_seat=((1,), (), (10,)),
        bot_seat=0,
        current_hand_suit_ids=hand,
        legal_max_amount=legal_max,
        revealable_count=len(hand),
    )


def test_bid_prompt_embeds_rules_and_complete_sdk_visible_snapshot() -> None:
    prompt = PocketRocksPromptSkill().render(_context(), _knowledge())

    assert "highest bid wins" in prompt
    assert "tie" in prompt
    assert "Loan 10" in prompt
    assert "final money" in prompt
    assert "Ruleset: prompt-test" in prompt
    assert "Players: 3; you are seat 0; priority seat: 2" in prompt
    assert "Auction for 1 resource card" in prompt
    assert "Offered resources: Ore" in prompt
    assert "Seat 0 (YOU): cash=$30" in prompt
    assert "Seat 1: cash=$22" in prompt
    assert "won={Brick: 1, Wood: 0, Ore: 0, Sheep: 0, Wheat: 0}" in prompt
    assert "revealed={Brick: 0, Wood: 1, Ore: 0, Sheep: 0, Wheat: 0}" in prompt
    assert "Your private hand: [0: Brick, 1: Wood]" in prompt
    assert "Prior bids and current loan/investment positions are not exposed" in prompt
    assert "Resource deck counts: Brick=6, Wood=6, Ore=6, Sheep=6, Wheat=6" in prompt
    assert "Action deck counts: Auction 1=12" in prompt
    assert "Value chart: revealed 0=$0, 1=$4, 2=$8, 3=$12, 4=$16, 5+=$20" in prompt
    assert f"Objective 1: {OBJECTIVES[1].description}; payout=${OBJECTIVES[1].payout}" in prompt
    assert prompt.rstrip().endswith(
        "Return exactly one base-10 integer from 0 through 12. 0 means bid zero/pass."
    )


def test_reveal_prompt_maps_every_zero_based_hand_index() -> None:
    context = _context(
        decision_kind="selectInfoToReveal",
        legal_max=None,
        hand=(int(Suit.WHEAT), int(Suit.BRICK), int(Suit.WHEAT)),
    )

    prompt = PocketRocksPromptSkill().render(context, _knowledge())

    assert "Reveal choices: 0: Wheat; 1: Brick; 2: Wheat" in prompt
    assert prompt.rstrip().endswith(
        "Return exactly one base-10 integer from 0 through 2: the card index to reveal."
    )


def test_retry_correction_is_immediately_before_repeated_contract() -> None:
    prompt = PocketRocksPromptSkill().render(
        _context(),
        _knowledge(),
        correction="The previous response contained prose.",
    )

    assert prompt.rstrip().endswith(
        "Correction: The previous response contained prose.\n"
        "Return exactly one base-10 integer from 0 through 12. 0 means bid zero/pass."
    )
