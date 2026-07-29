# Live Visible Resources Design

## Problem

PocketRocks sends the two visible resource cards in
`DecisionContext.current_resource_ids` on every action. The heuristic code
currently treats those cards as the active offer. This rejects financial
actions and overstates the lot for Auction 1. `HeuristicBotBrain` converts the
resulting `HeuristicInputError` into a pass.

## Design

Derive the offered lot from the active action while preserving the SDK context:

- Auction 1 offers only the first visible resource.
- Auction 2 offers both visible resources, or the single remaining resource.
- Loans and investments offer no resources.

One shared helper will provide offered-resource counts to belief construction
and valuation. Validation will continue to check the two-card board shape,
suit IDs, ordering, and that an auction has a first visible resource. It will
not reject visible board cards during financial actions.

The existing financial economics remain unchanged:

- loans receive temporary liquidity value that decreases with the remaining
  resource horizon and repay principal at scoring;
- investments receive their explicit fixed payout, return the locked bid at
  scoring, and pay only a temporary liquidity cost.

## Testing

Regression tests will reproduce the live contexts:

- Invest 10 with visible resources `(2, 4)` must evaluate to a positive bid;
- Auction 1 with visible resources `(3, 2)` must value and remove only suit 3;
- Auction 2 must continue to value both visible resources.

The focused heuristic suite and full project verification will run after the
change.

## Deferred Follow-up

Explore opportunity-aware loan valuation as a separate change. A possible
endgame heuristic could estimate the marginal value of liquidity from the
specific remaining action mix, known resource distribution, cash by seat, and
remaining objectives instead of using only the normalized resource horizon.
This work is intentionally outside the live-context bug fix.
