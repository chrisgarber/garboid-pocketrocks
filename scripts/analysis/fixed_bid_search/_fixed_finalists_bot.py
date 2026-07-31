from __future__ import annotations

from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.knowledge import RulesetKnowledge

_ACTIONS = (
    ActionId.AUCTION1,
    ActionId.AUCTION2,
    ActionId.LOAN10,
    ActionId.LOAN20,
    ActionId.INVEST5,
    ActionId.INVEST10,
)


class _FinalistBrain:
    values: tuple[int, ...]

    def __init__(self, seed: int | None = None) -> None:
        del seed
        self._targets = dict(zip(_ACTIONS, self.values, strict=True))

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
        try:
            action = ActionId(context.current_action_id)
        except TypeError, ValueError:
            return BotDecision.pass_turn()
        target = self._targets.get(action)
        if legal_max is None or legal_max <= 0 or target is None:
            return BotDecision.pass_turn()
        return BotDecision.submit_bid(min(target, legal_max))


class FixedBidTunedV1Brain(_FinalistBrain):
    values = (5, 10, 2, 5, 4, 7)


class FixedBidDiverseV1Brain(_FinalistBrain):
    values = (4, 9, 2, 5, 4, 7)
