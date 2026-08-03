# Monte Carlo best response v1

Date: 2026-08-03

`monte-the-bookie-v1` is a new bot generation. It keeps the existing belief, cash,
objective and reveal engines unchanged and replaces only the bid rule.

The existing brains all answer "what is this lot worth" and then bid a shaded fraction
of the answer. Auctions here are sealed-bid first-price, so the quantity that decides a
turn is the distribution of the maximum opposing bid. This generation estimates that
distribution and bids the amount with the greatest expected surplus.

No released name, coefficient, checkpoint or bot ID changed. The addition is a new
simulation-only identity whose `bot_id` is its versioned name.

## What it adds

All inputs stay inside the
[public information boundary](../architecture/public-information-boundary.md).

| Signal | Where it comes from |
| --- | --- |
| Opponent bid distribution | `PublicAuctionResolved.bids_by_seat`, resampled per seat |
| Fitted population prior | 66,260 recorded bidding decisions, as bid/cash quantiles |
| Loan debt and investment returns per seat | replayed turns plus derived auction winners |
| Projected final standings | the reconstructed ledger plus estimated resource values |
| Auctions still to come | `RulesetKnowledge.action_counts` minus opened turns |
| Rival objective completion | offered lot applied to each rival's public counts |

The brain samples hidden terminal prices and opponent bids together, resolves the
auction exactly including tie-break priority, and adds a bonus for bids that carry it
past the projected leader. Suit counts are drawn as one multivariate hypergeometric
deal into the opponents' hidden hand slots, so the suits stay coupled through the
shared finite population.

Coefficients were searched with `MonteCarloRunner` against the registered field over
charts A-E, each weight starting from zero. Every one settled on a non-zero value.

| Coefficient | Value |
| --- | ---: |
| Liquidity strength | 3.62 |
| Future cash weight | 0.38 |
| Objective progress weight | 0.46 |
| Bid shading | 0.00 |
| Prior weight | 17.11 |
| Scarcity weight | 0.81 |
| Denial weight | 0.45 |
| Deck-pressure weight | 0.68 |
| Standings weight | 0.39 |

Bid shading is zero because shading substitutes for an opponent model. Liquidity
strength is far above every hand-tuned profile, which is consistent with the direction
`HEURISTIC_V3` already moved: once the probability of winning a lot is estimated
directly, cash can be priced much more aggressively.

## Full tournament

The default field with this generation added played 15,000 games over charts A-E and
three through five players, root seed 0, 200 bootstrap samples, 11 workers. All 200
bootstrap fits converged, the schedule produced 4,285 to 4,287 appearances per bot, and
no bot faulted.

| Rank | Bot | Rating | 95% interval | Win rate | Mean money |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | monte-the-bookie-v1 | 1666.57 | 1658.9 to 1677.3 | 52.5% | 55.94 |
| 2 | fixed-objective-overlay-v2 | 1610.95 | 1601.4 to 1619.8 | 42.9% | 55.25 |
| 3 | fixed-objective-overlay-v1 | 1582.72 | 1576.4 to 1590.5 | 33.2% | 52.57 |
| 4 | aggressive-v3 | 1547.09 | 1540.2 to 1554.3 | 28.0% | 49.74 |
| 5 | fixed-bid-tuned-v1 | 1527.67 | 1519.7 to 1534.1 | 31.1% | 50.93 |
| 6 | fixed-bid | 1519.30 | 1512.1 to 1526.4 | 25.9% | 49.77 |
| 7 | aggressive-v2 | 1513.79 | 1506.8 to 1519.8 | 20.4% | 48.35 |
| 8 | balanced-v2 | 1493.35 | 1488.0 to 1499.8 | 16.0% | 47.67 |
| 9 | vector_ppo_large_v1_g350k | 1484.87 | 1474.6 to 1492.8 | 28.2% | 47.37 |
| 10 | balanced-v3 | 1470.93 | 1464.5 to 1476.3 | 14.6% | 47.01 |
| 11 | passive-v2 | 1451.31 | 1445.0 to 1456.6 | 11.7% | 45.67 |
| 12 | fixed-bid-diverse-v1 | 1427.65 | 1419.7 to 1435.2 | 20.7% | 46.40 |
| 13 | passive-v3 | 1367.56 | 1360.6 to 1374.4 | 9.1% | 43.21 |
| 14 | passive-v1 | 1336.25 | 1328.6 to 1342.8 | 5.0% | 41.33 |

The interval does not overlap second place. Absolute ratings are not comparable with
earlier tournaments, which ran different rosters.

## Held-out promotion

Candidate `monte-the-bookie-v1` against incumbent `fixed-objective-overlay-v2`, the
strongest committed identity, on `configs/promotion/held-out-v1.json`. 480 matched
pairs, 960 games, 1,000 bootstrap replicates at seed 0.

| Field | Value |
| --- | --- |
| Rating difference | 66.65 |
| 95% interval | 21.09 to 111.62 |
| Pairs completed | 480 of 480 |
| Bootstrap converged | 1,000 of 1,000 |
| Faults | 0 |
| Failure reasons | none |
| Promoted | yes |

The margin here is much smaller than the tournament gap suggests, because the
incumbent is the field's strongest bot rather than its average. The lower bound is
positive, so the gate passes, but the two are closer than the leaderboard implies.

## Notes and limitations

- Coefficients were searched against a field snapshot taken before `HEURISTIC_V3`, the
  objective overlays and the tuned fixed-bid variants entered the default roster. The
  results above are measured against the current field, but the tuning is one field
  behind and a refit would probably gain a little more.
- Suit sampling is exact, but opponent bids are sampled independently per seat. Rivals
  in fact respond to a shared board, so a correlated bid model is the obvious next step.
- `prior_weight` settled at 17.11, meaning the offline population prior dominates a
  seat's own in-game sample until many bids have accumulated. Most of the
  opponent-modelling strength is in the fitted table rather than in-game adaptation.
- Roughly 1.4 ms of NumPy work per bidding decision, which makes a 15,000-game
  tournament about two minutes on eleven workers rather than about twenty seconds.
- Chart C remains the hardest configuration and chart D the richest, matching the
  ordering recorded in earlier reports.

## Reproduction

```bash
uv run --extra neural garboid-tournament \
  --games 15000 --seed 0 --bootstrap-samples 200 --workers 11 \
  --output-dir tournament-results

uv run --extra neural garboid-promote \
  --candidate monte-the-bookie-v1 \
  --incumbent fixed-objective-overlay-v2 \
  --output-dir promotion-results/monte-carlo-v1
```

Refit the opponent prior under a new name with:

```bash
PYTHONPATH=src uv run --extra neural python scripts/analysis/fit_bid_prior.py
```
