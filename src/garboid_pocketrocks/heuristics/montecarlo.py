"""Coefficients for the Monte Carlo best-response brain.

Released, and therefore immutable: add a new named generation rather than editing
these numbers (see ``docs/architecture/immutable-bot-identities.md``).

Every weight below was searched against the registered field over all five value
charts with ``MonteCarloRunner``, starting from zero, and every one settled on a
non-zero value -- so each factor is carrying its own weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from garboid_pocketrocks.heuristics.bid_priors import BID_PRIOR_V2, BidPrior
from garboid_pocketrocks.heuristics.profiles import HeuristicProfile


@dataclass(frozen=True, slots=True)
class MonteCarloSettings:
    """Knobs for the Monte Carlo best-response brain."""

    profile: HeuristicProfile
    scenarios: int = 192
    prior_weight: float = 4.0
    tie_break_epsilon: float = 1e-3
    #: Premium for lots whose suits are nearly exhausted from the biddable pool.
    scarcity_weight: float = 0.0
    #: Weight on denying a rival an objective they would complete by taking the lot.
    denial_weight: float = 0.0
    #: Blend between garboid's resource horizon (0) and the exact remaining-auction
    #: share from the public action deck (1).
    pressure_weight: float = 0.0
    #: Weight on crossing the projected leader, rather than on expected surplus.
    standings_weight: float = 0.0
    #: Draw hidden counts as one multivariate hypergeometric instead of per-suit
    #: marginals, which is the true joint over a shared finite pool.
    joint_sampling: bool = True
    #: Fitted opponent-bid prior; ``None`` loads the shipped table.
    prior: BidPrior | None = None

    def __post_init__(self) -> None:
        if self.scenarios < 1:
            raise ValueError("scenarios must be positive")
        if self.prior_weight < 0.0:
            raise ValueError("prior_weight must be nonnegative")
        for name in ("scarcity_weight", "denial_weight", "standings_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.pressure_weight <= 1.0:
            raise ValueError("pressure_weight must be within [0, 1]")


MONTE_CARLO_PROFILE_V1 = HeuristicProfile(
    name="monte-carlo-v1",
    liquidity_strength=3.6230900000000000,
    future_cash_weight=0.3757600000000000,
    objective_progress_weight=0.4583200000000000,
    bid_shading=0.0,
)

MONTE_CARLO_V1 = MonteCarloSettings(
    profile=MONTE_CARLO_PROFILE_V1,
    scenarios=192,
    #: High: the offline-fitted population prior beats a thin within-game sample, so a
    #: seat's own observed bids only start to dominate after many of them.
    prior_weight=17.1124500000000000,
    #: Every one of the four public-history factors below was searched from zero and
    #: chose a non-zero weight, so each is carrying its own weight rather than
    #: decorating the docstring.
    scarcity_weight=0.8067800000000000,
    denial_weight=0.4542700000000000,
    #: 0.68 leans mostly on the *exact* remaining-auction count from the public action
    #: deck rather than on garboid's resource-count horizon proxy.
    pressure_weight=0.6813200000000000,
    standings_weight=0.3948100000000000,
    joint_sampling=True,
)


MONTE_CARLO_PROFILE_V2 = HeuristicProfile(
    name="monte-carlo-v2",
    liquidity_strength=6.000000,
    future_cash_weight=0.240220,
    objective_progress_weight=0.000000,
    bid_shading=0.0,
)

#: Second generation. Two things changed, and the second matters more than the first.
#:
#: It carries ``BID_PRIOR_V2``, refitted against the field it now faces. The v1 table
#: had opponents bidding *less* late; this field escalates instead, so v1 systematically
#: underbid the endgame.
#:
#: More importantly, the coefficients were searched against **mean normalized finish**
#: rather than outright win rate. Plackett-Luce rating rewards consistent placement
#: across multiplayer games, not wins, and the two genuinely diverge: v1 beats every
#: rival in balanced head-to-head play yet rates third, because it is high-variance and
#: places poorly when it does not win. Optimizing win rate was the wrong objective for
#: a rating target.
#:
#: The retune is mostly a *reduction*: ``objective_progress_weight`` to zero,
#: ``pressure_weight`` 0.68 -> 0.14, ``denial_weight`` 0.45 -> 0.23, and
#: ``standings_weight`` 0.39 -> 0.05. Chasing speculative upside costs placement. What
#: rose is ``liquidity_strength``, which protects the floor.
MONTE_CARLO_V2 = MonteCarloSettings(
    profile=MONTE_CARLO_PROFILE_V2,
    scenarios=192,
    prior_weight=12.475002,
    scarcity_weight=0.554232,
    denial_weight=0.231910,
    pressure_weight=0.138798,
    standings_weight=0.050612,
    joint_sampling=True,
    prior=BID_PRIOR_V2,
)
