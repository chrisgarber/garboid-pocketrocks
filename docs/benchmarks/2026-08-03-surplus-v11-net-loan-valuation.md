# Surplus v11 net-loan valuation

Date: 2026-08-03

## Decision

Release `surplus-v11` and advance the local `surplus` alias from v10 to v11.
Historical v1-v10 identities and behavior remain selectable. This is a local
simulation release, not a remote promotion.

V11 makes loan affordability and value explicit. The game permits a loan bid
up to current cash plus the offered principal. For every candidate fee `b`, v11
computes:

```text
post-win cash = current cash + principal - b
shortfall reduction = max(0, target cash - current cash)
                    - max(0, target cash - post-win cash)
loan surplus = (2/3 * shortfall reduction) - b
```

The reservation bid is the highest legal fee with nonnegative loan surplus,
subject to a 40% principal ceiling. This allows bids funded by the loan itself
while reducing willingness to pay when the proceeds mostly overshoot the
remaining liquidity need.

The jointly tuned frozen policy:

- targets cash equal to `7/8` of projected remaining resource spend;
- opens and caps loan bidding at 40% of principal (`4`/`8`) before market
  evidence, then follows public same-action rival prices within that ceiling;
- values reduction of projected cash shortfall at `2/3` per dollar;
- reserves `1/8` of projected resource spend before resource bids;
- reserves `2/5` before investments;
- retains v10's resource, objective, progress, denial, reserve-release,
  player-count shading, and resource-market weights.

## Development search

All development receipts remain ignored under
`artifacts/tournaments/surplus-v11-*`. Search stages used players 3-5, charts
A-E, batch size 64, no bootstrap, and fresh fixed root seeds. The seven-stage
coordinate pass used 3,000 games per stage; a 4,800-game combination field,
6,000-game interaction grid, and 6,000-game matched ablation followed. Total
development coverage was 37,800 games.

| Stage | Root seed | Knob or comparison | Selected result |
|---|---:|---|---|
| Marginal liquidity value | `2026080701` | `1/2`, `2/3`, `3/4`, `1`, `3/2` | `2/3`; 1534.86 rating versus v10's 1525.19 |
| Loan trigger | `2026080702` | `1/2` through `1` | `7/8`; 1541.64 versus v10's 1529.77 |
| Loan fee ceiling | `2026080703` | 30%-50% | 40%; 1537.63 versus v10's 1532.50 |
| Opening loan fee | `2026080704` | 20%-40% | 40%; 1580.37 versus v10's 1530.50 |
| Resource reserve | `2026080705` | 0%-25% | v10's `1/8` retained |
| Investment reserve | `2026080706` | 10%-50% | `2/5`; 1537.67, first in the stage |
| Trigger confirmation | `2026080707` | repeat after other changes | `7/8` narrowly led `5/8` |
| Combination field | `2026080708` | finalists and local ablations | exposed a trigger interaction; `3/4` led this field |
| Trigger/value grid | `2026080709` | 3 triggers by 3 liquidity values | restored `7/8` with `2/3` as the best v11 combination |
| Matched feature ablation | `2026080710` | selected v11 versus identical no-net policy | v11 1552.88; no-net 1547.49; v10 1502.64 |

The matched ablation attributes 5.39 rating points and $0.27 mean money to the
net-proceeds calculation beyond coefficient changes. The entire selected v11
package gained 50.24 rating points and $2.57 mean money over v10 in that
development field. No development run faulted.

Important negative and interaction results:

- a `1/2` liquidity multiplier was too conservative and fell below v10;
- opening at 20%-25% of principal lost heavily;
- the best trigger changed in the crowded finalist field, so it was retested
  in a full interaction grid instead of accepting the coordinate winner;
- increasing the resource reserve did not improve the selected loan model;
- the feature's direct ablation gain was modest relative to the gain from the
  jointly retuned loan and investment parameters.

## Held-out confirmation

After freezing the policy, confirmation used untouched root seed `2026080799`,
6,000 games, players 3-5, charts A-E, batch size 64, record-and-pass faults,
decision diagnostics, and 200 complete-game bootstrap resamples. Every bot
appeared in 4,800 games and every bootstrap fit converged.

| Bot | PL rating | 95% interval | Mean money | Outright wins | Win rate | Faults |
|---|---:|---:|---:|---:|---:|---:|
| surplus-v11 | 1602.94 | 1596.4-1609.8 | 53.02 | 2,251 | 46.90% | 0 |
| surplus-v10 | 1520.96 | 1515.1-1527.6 | 49.36 | 1,467 | 30.56% | 0 |
| fixed-objective-overlay-v2 | 1519.42 | 1514.3-1524.9 | 47.48 | 1,013 | 21.10% | 0 |
| fixed-bid | 1451.24 | 1445.1-1456.6 | 43.28 | 498 | 10.38% | 0 |
| fixed-bid-tuned-v1 | 1405.44 | 1399.3-1411.8 | 41.52 | 521 | 10.85% | 0 |

V11 gained 81.98 rating points and $3.66 mean money over v10. Their intervals
are disjoint. In the 3,800 games containing both, v11's head-to-head score was
63.04% with a 95% Wilson interval of 61.49%-64.56%.

## Advanced diagnostics

| Measure per game | surplus-v11 | surplus-v10 |
|---|---:|---:|
| Final cash | 5.64 | 5.91 |
| Resource cards | 5.06 | 3.96 |
| Resource spend | 31.11 | 24.07 |
| Item value | 55.36 | 43.92 |
| Item value minus resource spend | 24.25 | 19.85 |
| Objective payout | 11.65 | 7.11 |
| Objectives claimed | 1.56 | 1.00 |
| Investment bonus | 2.79 | 3.15 |
| Loan principal | 25.11 | 13.71 |
| Loan fees | 9.84 | 4.91 |

The mechanism is aggressive rather than cost-minimizing in isolation: v11
borrows and pays substantially more, then converts the liquidity into 1.10
additional resources, $4.40 more item surplus after auction spend, and 0.56
additional objectives per game. It invests slightly less and finishes with
$0.27 less cash, but the resource and objective gains more than compensate.

Loan bids can exceed current cash because their legal maximum includes the
principal. V11 did so on 7.75% of loan requests, by at most $8; v10 did so on
9.50%. Winning v11 fees had medians of $4 for Loan10 and $8 for Loan20.

V11 led v10 in mean money and outright win rate in 14 of 15 chart/player-count
cells. The exception was 3-player chart C, where v11 averaged $0.99 less and
its outright win rate was 4.58 percentage points lower. The largest mean-money
gain was $6.52 in 5-player chart E. The global improvement is strong but not
uniform across every condition.

## Reproduction and provenance

- Development source base commit: `a22c50d92f690bb0bd3a059cbd252dddbd9b7cb7`
- Release base commit: `4f224f17fc054b0b8694aff98191608ff5263727`
- Surplus policy SHA-256: `95f87a1467feeb46b34fd8ab237374eec2ba27390c8661457507f02e966252b6`
- Held-out summary SHA-256: `d3afc730f207af305614bba52d328d331aa0fc9384ad7be2d282d4068d027a5a`
- Development search summary SHA-256: `08c454d303e9fa7a9280449745fc0708c784ecf25dc335b1c53060996ed6bcc4`

```bash
uv run --offline garboid-tournament \
  --games 6000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080799 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,surplus-v10,surplus-v11 \
  --output-dir artifacts/tournaments/surplus-v11-held-out

uv run --offline garboid-visualize \
  artifacts/tournaments/surplus-v11-held-out
```
