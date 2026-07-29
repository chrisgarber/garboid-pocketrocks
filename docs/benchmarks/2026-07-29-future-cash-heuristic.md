# Future-cash heuristic benchmark

This report records the calibration and frozen validation of the heuristic
bots after adding an explicit opportunity cost for spending cash needed by
future resource auctions.

## Change

The existing logarithmic cash-option utility remains in place. A separate
piecewise-linear value now protects cash up to the public resource horizon:

```text
future_cash_value = weight * min(cash, starting_cash * resource_horizon)
```

Each action curve includes the change in this value between its current and
post-action cash. The term applies uniformly to resource auctions, loans, and
investments and is exposed separately in valuation breakdowns.

The calibrated profile constants are:

| Profile | Liquidity | Future cash | Objective progress | Bid shading |
|---|---:|---:|---:|---:|
| aggressive | 0.75 | 1.50 | 0.25 | 0.05 |
| balanced | 0.40 | 0.75 | 0.20 | 0.25 |
| passive | 0.15 | 0.60 | 0.15 | 0.30 |

Aggressive still shades bids least and values objective progress most.
Passive still shades bids most and retains the most terminal cash. The new
future-cash weights chiefly prevent all profiles from exhausting their
budgets before late resource auctions.

## Method

- Lineup: aggressive, balanced, passive
- Player count: 3
- Rulesets: live charts A-E, sampled uniformly
- Calibration: 20,000 games, root seed `20260730`
- Frozen validation: 100,000 games, root seed `20260729`
- Detailed late-auction diagnostic: 100,000 games, root seed `20260729`
- Multiprocessing results are deterministic and contain zero bot faults

Only future-cash weights and bid shading were calibrated. The final constants
were frozen before validation; the validation result did not drive further
tuning.

## Calibration result

| Profile | Outright win | Early resource spend/game | Terminal cash | Mean nonzero bid |
|---|---:|---:|---:|---:|
| aggressive | 32.285% | $10.767 | $2.649 | $6.173 |
| balanced | 35.505% | $8.791 | $4.061 | $5.941 |
| passive | 29.560% | $7.567 | $4.809 | $5.775 |

The calibration spread was 5.945 percentage points. It met the intended
personality order for both early resource spending
(`aggressive > balanced > passive`) and terminal cash
(`passive > balanced > aggressive`).

## Frozen 100,000-game validation

| Profile | Outright win | Tied first | Mean score | Mean rank | Pass rate | Mean nonzero bid | Objectives/100 | Resources/game | Terminal cash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggressive | 32.987% | 1.877% | 65.007 | 1.968 | 14.241% | $6.154 | 156.216 | 6.008 | $2.661 |
| balanced | 36.124% | 1.898% | 66.425 | 1.910 | 12.483% | $5.947 | 111.971 | 5.018 | $4.074 |
| passive | 28.180% | 1.694% | 64.049 | 2.057 | 12.519% | $5.778 | 78.725 | 3.974 | $4.871 |

The mean winning score was 77.529 and the median was 76. All bot-fault counts
were zero.

Balanced is now the strongest profile, while aggressive and passive remain
competitive and behaviorally distinct. Relative to the previous 100,000-game
result, outright wins changed from 16.856% / 15.497% / 65.725% to
32.987% / 36.124% / 28.180% for aggressive / balanced / passive.

The frozen result narrowly missed the aspirational balance band. Passive was
0.820 percentage points below its 29% lower bound, and the 7.944-point
top-to-bottom spread was 0.944 points above the 7-point target. Per the
precommitted method, no coefficients were retuned after seeing validation.

### Validation by value chart

| Chart | Aggressive | Balanced | Passive |
|---|---:|---:|---:|
| A | 34.983% | 34.623% | 27.565% |
| B | 32.405% | 36.579% | 28.075% |
| C | 36.495% | 34.009% | 26.805% |
| D | 30.583% | 38.293% | 28.528% |
| E | 30.472% | 37.111% | 29.927% |

Balanced leads charts B, D, and E. Aggressive leads A and C. Passive remains
closest on E and weakest on C, so chart effects are material but no chart
recreates the previous passive dominance.

## Late two-resource auction behavior

| Turn band | Mean winning price | Wins/game | Payments/game |
|---|---:|---:|---:|
| 1-5 | $11.634 | 1.327 | $15.441 |
| 6-12 | $11.658 | 1.864 | $21.729 |
| 13+ | $7.837 | 1.245 | $9.755 |

The late-game mean rose from the earlier $5.172 baseline to $7.837, a
$2.665 or 51.5% increase. Prices still decline after turn 12, but two-resource
auctions no longer lose most of their value.

Late two-resource decision contexts show why prices still decline:

| Profile | Mean cash | Mean submitted bid | Mean reservation | Pass rate | Cash-cap binding |
|---|---:|---:|---:|---:|---:|
| aggressive | $7.805 | $5.674 | $6.684 | 20.736% | 78.088% |
| balanced | $8.418 | $5.158 | $7.386 | 5.379% | 82.316% |
| passive | $8.501 | $4.794 | $7.497 | 2.836% | 83.270% |

The average modeled resource value remains about $18.10-$18.12 in those
contexts. The remaining price decline is therefore driven primarily by
limited cash and auction competition, not by the bots deciding that two
resources have little intrinsic value.

## Interpretation

The heuristic was spending too much too early. Explicitly valuing cash needed
for the remaining public resource horizon fixes the largest symptom without
flattening the profiles:

- aggressive spends most early, bids highest on average, completes the most
  objectives, and finishes with the least cash;
- balanced trades some aggression for flexibility and produces the best
  score and win rate;
- passive spends least early, bids lowest on average, and finishes with the
  most cash.

The result is substantially more balanced than the previous passive
two-thirds-win equilibrium. The small frozen-validation miss is recorded
rather than hidden through seed-specific tuning.
