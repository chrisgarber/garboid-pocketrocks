"""Monte Carlo best-response bidding.

Named ``best_response`` rather than ``monte_carlo`` because ``simulator.monte_carlo``
already owns that name for the experiment runner.

Every existing brain answers "what is this lot worth" and then bids a shaded fraction
of the answer. In a sealed-bid first-price auction the deciding quantity is instead the
distribution of the maximum opposing bid, so this brain estimates that and bids the
amount with the best expected surplus.

Reuses the existing engines rather than replacing them: ``belief.build_belief`` for the
hypergeometric posterior over hidden terminal prices, ``cash.evaluate_action_curve`` for
per-bid cash accounting, ``objectives.evaluate_objectives`` for completion and contested
progress, and ``reveals.choose_reveal`` for the reveal policy. What is new is the
decision rule and the public-history signals in ``heuristics.ledger``.
"""

from __future__ import annotations

import numpy as np
from pocketrocks import ActionId, BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.heuristics.belief import (
    BeliefState,
    build_belief,
    offered_resource_counts,
)
from garboid_pocketrocks.heuristics.bid_priors import BID_PRIOR_V1, BidPrior
from garboid_pocketrocks.heuristics.cash import evaluate_action_curve
from garboid_pocketrocks.heuristics.errors import HeuristicInputError
from garboid_pocketrocks.heuristics.ledger import (
    Ledger,
    auction_pressure,
    deficit_to_leader,
    denial_value,
    projected_scores,
    reconstruct_ledger,
)
from garboid_pocketrocks.heuristics.montecarlo import (
    MONTE_CARLO_V1,
    MonteCarloSettings,
)
from garboid_pocketrocks.heuristics.objectives import evaluate_objectives
from garboid_pocketrocks.heuristics.opponents import (
    BidSampler,
    observe_bids,
    scan_priority,
    turn_index_from_history,
)
from garboid_pocketrocks.heuristics.reveals import choose_reveal
from garboid_pocketrocks.heuristics.valuation import HeuristicValuator
from garboid_pocketrocks.knowledge import RulesetKnowledge

_LOAN_HEADROOM = {int(ActionId.LOAN10): 10, int(ActionId.LOAN20): 20}


class MonteCarloBotBrain:
    """Monte Carlo best response in a sealed-bid first-price auction.

    For every legal bid, expected surplus is estimated by sampling scenarios that
    jointly draw the hidden terminal prices of the offered suits and every opponent's
    bid, then resolving the auction exactly -- tie priority included.

    On top of expected surplus it also weighs, all from public information: the payout
    a rival would collect by taking this lot, how many auctions are genuinely left in
    the action deck, and whether it is behind on a reconstructed projection of
    everyone's final score.
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        settings: MonteCarloSettings = MONTE_CARLO_V1,
    ) -> None:
        self._settings = settings
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._prior: BidPrior = settings.prior if settings.prior is not None else BID_PRIOR_V1
        # Fallback for the no-history path; also supplies the reveal policy.
        self._valuator = HeuristicValuator(settings.profile)

    # -- BotBrain ---------------------------------------------------------

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory = (),
    ) -> BotDecision:
        """Use public history when available, with a deterministic live fallback."""
        if context.decision_kind == "selectInfoToReveal":
            return self._reveal(context, ruleset)
        if history:
            try:
                return self._monte_carlo_bid(context, ruleset, history)
            except HeuristicInputError:
                # Same degradation garboid's heuristic uses: a context that contradicts
                # public knowledge means pass rather than guess.
                return BotDecision.pass_turn()
        try:
            return BotDecision.submit_bid(self._valuator.evaluate_bid(context, ruleset).chosen_bid)
        except HeuristicInputError:
            return BotDecision.pass_turn()

    # -- internals ---------------------------------------------------------

    def _reveal(self, context: DecisionContext, ruleset: RulesetKnowledge) -> BotDecision:
        if context.revealable_count <= 0:
            return BotDecision.pass_turn()
        return BotDecision.select_info_to_reveal(choose_reveal(context, ruleset))

    def _monte_carlo_bid(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        settings = self._settings
        legal_max = context.legal_max_amount or 0
        action_id = context.current_action_id
        if legal_max <= 0 or action_id is None:
            return BotDecision.pass_turn()

        belief = build_belief(context, ruleset)
        action = ActionId(int(action_id))
        offered = offered_resource_counts(context, action)
        cash = context.cash_by_seat[context.bot_seat]

        ledger = reconstruct_ledger(history, ruleset)
        pressure = auction_pressure(ledger, ruleset)
        # Blend garboid's resource-count horizon with the exact remaining-auction
        # share. Both live in [0, 1]; weight 0 keeps garboid's behaviour verbatim.
        horizon = (
            1.0 - settings.pressure_weight
        ) * belief.normalized_horizon + settings.pressure_weight * pressure

        curve = evaluate_action_curve(
            action_id=action,
            cash=cash,
            legal_max=legal_max,
            horizon=horizon,
            starting_cash=ruleset.starting_cash,
            liquidity_strength=settings.profile.liquidity_strength,
            future_cash_weight=settings.profile.future_cash_weight,
            gross_value=0.0,
        )
        cost = np.array([point.win_delta for point in curve], dtype=np.float64)
        bids = np.arange(cost.size, dtype=np.int64)

        gain = self._objective_value(context, ruleset, belief, offered)
        gain += settings.denial_weight * denial_value(context, offered)

        values = self._sample_lot_values(context, belief, offered, settings.scenarios)
        values = values * self._scarcity_multiplier(belief, offered)

        sampler = BidSampler(
            context,
            observe_bids(history, context.player_count),
            rng=self._rng,
            prior_weight=settings.prior_weight,
            loan_headroom=_LOAN_HEADROOM.get(int(action_id), 0),
            prior=self._prior,
            turn_index=turn_index_from_history(history),
        )
        best_opposing, tie_priority = sampler.sample(settings.scenarios)
        my_priority = scan_priority(context)[context.bot_seat]

        # Exact win resolution: outbid, or match and hold better tie priority.
        outbid = bids[:, None] > best_opposing[None, :]
        matched = (bids[:, None] == best_opposing[None, :]) & (my_priority < tie_priority[None, :])
        wins = outbid | matched

        payoff = values[None, :] + gain + cost[:, None]
        surplus = np.where(wins, payoff, 0.0).mean(axis=1)
        objective = surplus + self._standings_bonus(context, ruleset, belief, ledger, wins, payoff)

        # Prefer the cheaper of two near-equal bids: in a first-price auction paying
        # less for the same expected surplus is strictly better.
        best = int(np.argmax(objective - settings.tie_break_epsilon * bids))
        if objective[best] <= 0.0:
            return BotDecision.pass_turn()
        bid = min(best, legal_max)
        return BotDecision.pass_turn() if bid == 0 else BotDecision.submit_bid(bid)

    # -- factors -----------------------------------------------------------

    def _standings_bonus(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        belief: BeliefState,
        ledger: Ledger,
        wins: np.ndarray,
        payoff: np.ndarray,
    ) -> np.ndarray:
        """Reward bids that carry this seat past the projected leader.

        PocketRocks pays for finishing first, not for accumulating surplus, and the
        two genuinely diverge -- garboid's own tournament has the policy with the
        field's best win rate ranked only fourth on rating. Every committed strategy
        maximizes value with no regard for the scoreboard. This term prices *crossing
        the leader*, a threshold event that expected surplus smooths away.
        """
        weight = self._settings.standings_weight
        if weight <= 0.0:
            return np.zeros(payoff.shape[0], dtype=np.float64)

        estimates = tuple(suit.expected_terminal_price for suit in belief.suits)
        scores = projected_scores(context, ledger, estimates)
        deficit = deficit_to_leader(scores, context.bot_seat)

        # Winning adds the lot's realized payoff to our projection; losing changes
        # nothing. So the leader is cleared when the payoff exceeds the deficit.
        crosses = wins & (payoff > deficit)
        return weight * ruleset.starting_cash * crosses.mean(axis=1)

    def _scarcity_multiplier(self, belief: BeliefState, offered: tuple[int, ...]) -> float:
        """Premium for a lot whose suits are nearly gone from the biddable pool.

        A card's value does not change with scarcity, but the opportunity cost of
        skipping it does: a missed final card cannot be bought back later.
        """
        weight = self._settings.scarcity_weight
        if weight <= 0.0:
            return 1.0
        remaining = belief.expected_future_biddable_counts
        offered_total = sum(offered)
        if offered_total <= 0:
            return 1.0
        pressure = 0.0
        for index, count in enumerate(offered):
            if count > 0:
                pressure += count / (1.0 + max(0.0, float(remaining[index])))
        return 1.0 + weight * (pressure / offered_total)

    def _sample_lot_values(
        self,
        context: DecisionContext,
        belief: BeliefState,
        offered: tuple[int, ...],
        scenarios: int,
    ) -> np.ndarray:
        """Draw the lot's terminal value from the hidden-count posterior.

        With ``joint_sampling`` the unseen cards are dealt into opponents' hidden hand
        slots as one **multivariate hypergeometric** draw, which is the true joint: all
        suits compete for the same finite pool of hidden slots, so one suit turning out
        to be hoarded makes the others rarer. Drawing each suit from its marginal
        ``terminal_price_pmf`` independently ignores that coupling.
        """
        chart = np.asarray(context.value_chart, dtype=np.float64)
        active = [index for index, count in enumerate(offered) if count > 0]
        if not active:
            return np.zeros(scenarios, dtype=np.float64)

        if self._settings.joint_sampling:
            joint = self._joint_counts(belief, scenarios)
            if joint is not None:
                total = np.zeros(scenarios, dtype=np.float64)
                for index in active:
                    known = int(belief.suits[index].known_terminal_reveals)
                    counts = np.minimum(known + joint[:, index], chart.size - 1)
                    total += offered[index] * chart[counts]
                return total

        total = np.zeros(scenarios, dtype=np.float64)
        for index in active:
            pmf = np.asarray(belief.suits[index].terminal_price_pmf, dtype=np.float64)
            mass = pmf.sum()
            if mass <= 0.0:
                total += offered[index] * float(belief.suits[index].expected_terminal_price)
                continue
            buckets = self._rng.choice(pmf.size, size=scenarios, p=pmf / mass)
            total += offered[index] * chart[buckets]
        return total

    def _joint_counts(self, belief: BeliefState, scenarios: int) -> np.ndarray | None:
        """Deal the unseen pool into hidden hand slots, all suits at once.

        Returns ``None`` when the belief's bookkeeping does not describe a well-formed
        finite population (nothing hidden, or counts that do not sum), in which case
        the caller falls back to the marginal pmfs.
        """
        first = belief.suits[0]
        slots = int(first.opponent_hidden_slots)
        population = int(first.unseen_population)
        colors = np.array([int(suit.unseen_suit_count) for suit in belief.suits], dtype=np.int64)
        if slots <= 0 or population <= 0 or int(colors.sum()) != population:
            return None
        if slots > population:
            return None
        return self._rng.multivariate_hypergeometric(colors, slots, size=scenarios)

    @staticmethod
    def _objective_value(
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        belief: BeliefState,
        offered: tuple[int, ...],
    ) -> float:
        if not ruleset.objectives_enabled or not context.objective_ids:
            return 0.0
        values = evaluate_objectives(
            active_objective_ids=context.objective_ids,
            owned_objective_ids_by_seat=context.owned_objective_ids_by_seat,
            won_resource_counts_by_seat=context.won_resource_counts_by_seat,
            bot_seat=context.bot_seat,
            offered_counts=offered,
            horizon=belief.normalized_horizon,
            progress_weight=0.2,
        )
        return float(sum(value.total for value in values))


class MonteCarloV1Brain(MonteCarloBotBrain):
    """Released Monte Carlo best-response generation."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed, settings=MONTE_CARLO_V1)


MONTE_CARLO_V1_BOT_SPEC = BotSpec.for_simulation(
    "monte-the-bookie-v1",
    MonteCarloV1Brain,
)
