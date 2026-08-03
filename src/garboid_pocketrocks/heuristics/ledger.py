"""Scoreboard and deck signals recovered from public history.

Everything here stays inside the
`public information boundary <../../../docs/architecture/public-information-boundary.md>`_:
each value is derived from the cumulative public event stream, the current decision
context, and the pinned SDK's rules constants. No opponent hand, deck order or engine
state is read.

1. **A reconstructed ledger.** ``DecisionContext`` exposes cash, won resources, public
   reveals and owned objectives, but not who took which loan or investment -- and those
   are worth ``-principal`` and ``+lock + payout`` at scoring. They are recoverable:
   every turn publishes its action and its resolved bids, and the winner is a
   deterministic function of the bids plus the tie-break seat. Replaying that yields
   each seat's full scoring ledger.
2. **Projected standings.** With the ledger every seat's final score can be projected.
   PocketRocks pays for finishing first rather than for accumulating surplus, so a bot
   that is behind should accept variance and a bot that is ahead should decline it.
3. **The remaining action deck.** The deck composition is fixed and public
   (``RulesetKnowledge.action_counts``) and every opened turn is published, so the
   number of auctions still to come is known exactly rather than approximated by a
   resource-count horizon.
4. **Objective denial.** ``evaluate_objectives`` prices the objectives the acting bot
   would complete. This module prices the ones a rival completes by taking the lot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pocketrocks import OBJECTIVES, ActionId, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicGameSetup,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.objectives import objective_is_met
from garboid_pocketrocks.knowledge import RulesetKnowledge

LOAN_PRINCIPAL = {int(ActionId.LOAN10): 10, int(ActionId.LOAN20): 20}
INVEST_PAYOUT = {int(ActionId.INVEST5): 5, int(ActionId.INVEST10): 10}
AUCTION_IDS = (int(ActionId.AUCTION1), int(ActionId.AUCTION2))


@dataclass(slots=True)
class SeatLedger:
    """Scoring components of one seat that are public but never surfaced."""

    loan_debt: int = 0
    investment_return: int = 0
    auctions_won: int = 0
    paid_total: int = 0


@dataclass(frozen=True, slots=True)
class Ledger:
    seats: tuple[SeatLedger, ...]
    remaining_actions: dict[int, int] = field(default_factory=dict)

    @property
    def remaining_auctions(self) -> int:
        return sum(self.remaining_actions.get(action, 0) for action in AUCTION_IDS)


def _winning_seat(bids: tuple[int, ...], tiebreak_seat: int) -> int:
    """Highest bid wins; ties scan forward from the seat *after* the tiebreak holder.

    Mirrors the resolution garboid's public-history adapter uses to attribute reveals,
    so a reconstructed winner always agrees with the reveal attribution.
    """
    highest = max(bids)
    count = len(bids)
    for offset in range(1, count + 1):
        seat = (tiebreak_seat + offset) % count
        if bids[seat] == highest:
            return seat
    return tiebreak_seat


def reconstruct_ledger(history: PublicHistory, ruleset: RulesetKnowledge) -> Ledger:
    """Replay public events into per-seat scoring components and deck remainder."""
    seats = tuple(SeatLedger() for _ in range(ruleset.player_count))
    remaining = {int(action): ruleset.action_counts[int(action) - 1] for action in ActionId}
    tiebreak = 0
    pending: int | None = None

    for event in history:
        if isinstance(event, PublicGameSetup):
            tiebreak = event.initial_tiebreak_seat
        elif isinstance(event, PublicTurnOpened):
            pending = int(event.action_id)
            if pending in remaining:
                remaining[pending] -= 1
        elif isinstance(event, PublicAuctionResolved):
            if pending is None:
                continue
            bids = tuple(int(bid) for bid in event.bids_by_seat)
            if not bids:
                continue
            winner = _winning_seat(bids, tiebreak)
            paid = bids[winner]
            if winner < len(seats):
                ledger = seats[winner]
                ledger.paid_total += paid
                if pending in LOAN_PRINCIPAL:
                    ledger.loan_debt += LOAN_PRINCIPAL[pending]
                elif pending in INVEST_PAYOUT:
                    ledger.investment_return += paid + INVEST_PAYOUT[pending]
                elif pending in AUCTION_IDS:
                    ledger.auctions_won += 1
            tiebreak = winner
            pending = None

    return Ledger(seats=seats, remaining_actions=remaining)


def projected_scores(
    context: DecisionContext,
    ledger: Ledger,
    suit_values: list[float] | tuple[float, ...],
) -> tuple[float, ...]:
    """Estimated final score for every seat, using only public information.

    ``cash + owned resources at estimated value + completed objectives
    + investment returns - loan debt``. Cash already reflects money spent, and a
    loan's principal is already inside cash, so the debt is subtracted once.
    """
    scores: list[float] = []
    for seat in range(context.player_count):
        total = float(context.cash_by_seat[seat])
        counts = context.won_resource_counts_by_seat[seat]
        total += sum(count * suit_values[index] for index, count in enumerate(counts))
        total += sum(
            OBJECTIVES[oid].payout
            for oid in context.owned_objective_ids_by_seat[seat]
            if oid in OBJECTIVES
        )
        if seat < len(ledger.seats):
            total += ledger.seats[seat].investment_return
            total -= ledger.seats[seat].loan_debt
        scores.append(total)
    return tuple(scores)


def deficit_to_leader(scores: tuple[float, ...], seat: int) -> float:
    """How far behind the best opponent this seat is. Negative means ahead."""
    others = [score for index, score in enumerate(scores) if index != seat]
    if not others:
        return 0.0
    return max(others) - scores[seat]


def denial_value(context: DecisionContext, offered: tuple[int, ...]) -> int:
    """Objective payout the strongest rival would collect by taking this lot.

    Losing an auction is not neutral when it hands somebody a completed objective.
    Only unclaimed objectives count, and only the best single outcome across rivals,
    since exactly one seat wins the lot.
    """
    if not context.objective_ids:
        return 0
    taken = {oid for seat in context.owned_objective_ids_by_seat for oid in seat}
    unclaimed = [oid for oid in context.objective_ids if oid not in taken]
    if not unclaimed:
        return 0

    best = 0
    for seat in range(context.player_count):
        if seat == context.bot_seat:
            continue
        before = list(context.won_resource_counts_by_seat[seat])
        after = list(before)
        for index, count in enumerate(offered):
            after[index] += count
        gained = sum(
            OBJECTIVES[oid].payout
            for oid in unclaimed
            if oid in OBJECTIVES
            and objective_is_met(oid, tuple(after))
            and not objective_is_met(oid, tuple(before))
        )
        best = max(best, gained)
    return best


def auction_pressure(ledger: Ledger, ruleset: RulesetKnowledge) -> float:
    """Share of the auctions still to come, in ``[0, 1]``.

    Known exactly rather than estimated: the action deck composition is public and
    every opened turn is published. Near zero means cash has almost no future use, so
    holding it back is pure waste.
    """
    total = sum(ruleset.action_counts[action - 1] for action in AUCTION_IDS)
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, ledger.remaining_auctions / total))
