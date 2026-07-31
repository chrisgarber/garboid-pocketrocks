from __future__ import annotations

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge

_TARGET_BIDS = {
    ActionId.AUCTION1: 6,
    ActionId.AUCTION2: 12,
    ActionId.LOAN10: 1,
    ActionId.LOAN20: 1,
    ActionId.INVEST5: 7,
    ActionId.INVEST10: 7,
}


def _target_bid(action_id: int) -> int | None:
    try:
        action = ActionId(action_id)
    except ValueError:
        return None
    return _TARGET_BIDS.get(action)


class FixedBidBotBrain:
    """Deterministic baseline that bids a fixed amount for each action."""

    def __init__(self, seed: int | None = None) -> None:
        del seed

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del ruleset
        if context.decision_kind == "selectInfoToReveal":
            if context.revealable_count <= 0:
                return BotDecision.pass_turn()
            return BotDecision.select_info_to_reveal(0)

        legal_max = context.legal_max_amount
        target = _target_bid(context.current_action_id)
        if legal_max is None or legal_max <= 0 or target is None:
            return BotDecision.pass_turn()
        return BotDecision.submit_bid(min(target, legal_max))


FIXED_BID_BOT_SPEC = BotSpec.for_simulation("fixed-bid", FixedBidBotBrain)
