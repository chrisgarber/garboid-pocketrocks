"""Opponent bid modelling from public history.

PocketRocks auctions are sealed-bid first-price, so the quantity that decides a turn is
not "what is this lot worth" but "what is the least I can pay and still win" -- that is,
the distribution of the maximum opposing bid. Every resolved auction publishes every
seat's bid through ``PublicAuctionResolved.bids_by_seat``, well inside the public
information boundary, so that distribution is estimable.

Each rival is resampled from what it has actually bid this game and blended toward the
fitted population prior in ``bid_priors``, trusting the seat more as its own sample
grows. Sampled bids are capped by what that seat can afford, which is public.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pocketrocks import ActionId, DecisionContext

from garboid_pocketrocks.adapters.public_history import (
    PublicAuctionResolved,
    PublicHistory,
    PublicTurnOpened,
)
from garboid_pocketrocks.heuristics.bid_priors import (
    UNINFORMED_BID_PRIOR,
    BidPrior,
)

#: Bidding actions grouped by how much a rival is likely to pay for them. Rivals
#: behave very differently on resource lots than on loans, so pooling them would
#: blur the only signal we have.
RESOURCE_ACTIONS = frozenset({int(ActionId.AUCTION1), int(ActionId.AUCTION2)})


@dataclass(frozen=True, slots=True)
class OpponentBids:
    """Per-seat bid observations for this game, split by action class."""

    resource: tuple[tuple[int, ...], ...]  # per seat, bids on Auction1/Auction2
    other: tuple[tuple[int, ...], ...]  # per seat, bids on loans/investments

    def for_action(self, action_id: int | None) -> tuple[tuple[int, ...], ...]:
        if action_id is not None and int(action_id) in RESOURCE_ACTIONS:
            return self.resource
        return self.other


def observe_bids(history: PublicHistory, player_count: int) -> OpponentBids:
    """Collect every seat's resolved bids, keyed by seat and action class.

    Walks the allowlisted public history. ``PublicTurnOpened`` tells us which action
    the following ``PublicAuctionResolved`` belongs to, so the two are paired by
    position rather than assumed.
    """
    resource: list[list[int]] = [[] for _ in range(player_count)]
    other: list[list[int]] = [[] for _ in range(player_count)]
    pending_action: int | None = None

    for event in history:
        if isinstance(event, PublicTurnOpened):
            pending_action = int(event.action_id)
        elif isinstance(event, PublicAuctionResolved):
            bucket = resource if (pending_action in RESOURCE_ACTIONS) else other
            for seat, bid in enumerate(event.bids_by_seat):
                if seat < player_count:
                    bucket[seat].append(int(bid))
            pending_action = None

    return OpponentBids(
        resource=tuple(tuple(seat_bids) for seat_bids in resource),
        other=tuple(tuple(seat_bids) for seat_bids in other),
    )


def turn_index_from_history(history: PublicHistory) -> int:
    """Zero-based index of the turn now being bid on.

    ``DecisionContext`` carries no turn number, but every resolved auction is a public
    event, so the count of them is exactly how many turns have already finished.
    """
    return sum(1 for event in history if isinstance(event, PublicAuctionResolved))


def scan_priority(context: DecisionContext) -> tuple[int, ...]:
    """Tie-break priority per seat: lower wins ties.

    Ties resolve by scanning forward from the seat *after* the tiebreak holder and
    wrapping, so the holder is checked last. No committed strategy uses this, yet it
    decides every equal-bid auction -- and equal bids are common because bids are
    small integers.
    """
    count = context.player_count
    priority = [0] * count
    for offset in range(1, count + 1):
        seat = (context.tiebreak_seat + offset) % count
        priority[seat] = offset - 1
    return tuple(priority)


class BidSampler:
    """Samples the maximum opposing bid, and who would win a tie at that value.

    Each opponent is resampled from its own observed bids (a within-game bootstrap),
    shrunk toward a cash-scaled prior while observations are scarce. Sampled bids are
    capped by what that seat can actually afford, which is public.
    """

    def __init__(
        self,
        context: DecisionContext,
        bids: OpponentBids,
        *,
        rng: np.random.Generator,
        prior_weight: float,
        loan_headroom: int = 0,
        prior: BidPrior = UNINFORMED_BID_PRIOR,
        turn_index: int = 0,
    ) -> None:
        self._context = context
        self._rng = rng
        self._prior_weight = prior_weight
        self._loan_headroom = loan_headroom
        self._prior = prior
        self._turn_index = turn_index
        self._observed = bids.for_action(context.current_action_id)
        self._priority = scan_priority(context)

    def sample(self, scenarios: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(max_opposing_bid, winning_tie_priority)`` per scenario.

        ``winning_tie_priority`` is the best (lowest) priority among the opponents
        that achieved the maximum, so a caller can resolve an equal bid exactly.
        """
        ctx = self._context
        me = ctx.bot_seat
        opponents = [seat for seat in range(ctx.player_count) if seat != me]
        if not opponents:
            return (
                np.zeros(scenarios, dtype=np.int64),
                np.full(scenarios, len(self._priority), dtype=np.int64),
            )

        draws = np.empty((scenarios, len(opponents)), dtype=np.int64)
        for column, seat in enumerate(opponents):
            draws[:, column] = self._sample_seat(seat, scenarios)

        best = draws.max(axis=1)
        # Priority of the strongest tied opponent: mask non-maximal seats out.
        priorities = np.array([self._priority[seat] for seat in opponents], dtype=np.int64)
        at_max = draws == best[:, None]
        masked = np.where(at_max, priorities[None, :], np.int64(len(self._priority) + 1))
        return best, masked.min(axis=1)

    def _sample_seat(self, seat: int, scenarios: int) -> np.ndarray:
        ctx = self._context
        cap = ctx.cash_by_seat[seat] + self._loan_headroom
        observed = self._observed[seat] if seat < len(self._observed) else ()
        action_id = ctx.current_action_id

        prior_draws = self._prior.draw(action_id, self._turn_index, cap, scenarios, self._rng)
        if not observed:
            return prior_draws

        # Blend this seat's own history with the fitted population prior, trusting
        # the seat more as its sample grows. Jitter keeps a rival that has bid one
        # exact value from being treated as deterministic.
        sampled = self._rng.choice(np.asarray(observed, dtype=np.int64), size=scenarios)
        weight = len(observed) / (len(observed) + self._prior_weight)
        prior_mean = float(np.mean(prior_draws)) if prior_draws.size else 0.0
        blended = weight * sampled + (1.0 - weight) * prior_draws
        jitter = self._rng.normal(0.0, 1.0 + 0.15 * prior_mean, size=scenarios)
        drawn = np.rint(blended + jitter).astype(np.int64)
        return np.clip(drawn, 0, max(0, cap))
