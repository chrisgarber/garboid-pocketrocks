# Fixed-bid search and objective overlay

## Result

The bounded fixed-bid search retained two immutable alternatives to the
original `(5, 10, 2, 4, 4, 9)` policy. Values are ordered as one-resource
auction, two-resource auction, loan 10, loan 20, invest 5, and invest 10.

| Identity | Frozen values | Intended role |
| --- | --- | --- |
| `fixed-bid-tuned-v1` | `(5, 10, 2, 5, 4, 7)` | strongest direct fixed-bid matchup |
| `fixed-bid-diverse-v1` | `(4, 9, 2, 5, 4, 7)` | different auction values with comparable field strength |

The search screened 64 bounded vectors for 600 games at root seed `24701`,
then refined the control and ten finalists for 1,800 games at each of roots
`9929` and `424242`. Screening and refinement used ten-bot fields containing
the original and one candidate, so fixed-value bots were 20% of the roster.

The two frozen finalists were then run together with the prior 11-bot curated
field for 7,500 games at each of roots `8675309` and `13579`. Fixed-value bots
were 3 of 13 identities (23.1%), every identity appeared 2,308 or 2,309 times
per run, and all policies recorded zero faults.

| Bot | Rating at `8675309` | Rating at `13579` | Mean delta from original |
| --- | ---: | ---: | ---: |
| original `fixed-bid` | 1656.4 | 1654.6 | — |
| `fixed-bid-tuned-v1` | 1669.7 | 1687.9 | +23.3 |
| `fixed-bid-diverse-v1` | 1685.0 | 1672.8 | +23.4 |

Across the two runs, the original scored 0.462 against tuned in 1,218 shared
games (normal 95% interval 0.435–0.490). The original scored 0.506 against
diverse (0.478–0.534), while diverse performed better against the field.
Tuned scored 0.584 against diverse (0.556–0.611). The policies therefore fill
different roles: tuned is the stronger direct duelist and diverse earns its
global strength through other matchups.

## Objective-aware overlay

`fixed-objective-overlay-v1` keeps the original six targets and changes only
auction bids, by at most two. Its signal adds:

- the value of completing or progressing an unclaimed objective;
- the offered bundle's expected-price edge over the average resource suit;
- 1.75 times the information edge between the bot's private posterior and a
  public-only posterior that does not know its hidden suit identities.

Signals at least 1 or 5 add one or two; signals at most -0.75 or -4 subtract
one or two. Objective progress has weight 0.20. Loans, investments, reveal
choice, legal caps, and invalid-input fallback retain original fixed-bid
behavior.

A 6,000-game development comparison at root `2026073101` selected this frozen
private-heavy configuration from three bounded overlays. A separate 15,000-
game validation at root `2026073102` used all charts and player counts and a
ten-bot field with only the original and overlay in the fixed family. Both had
6,000 appearances and zero faults.

| Bot | PL rating | 95% interval | Win rate | Mean money | Objectives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed-objective-overlay-v1` | 1732.02 | 1724.44–1738.98 | 50.73% | 61.47 | 7,896 |
| original `fixed-bid` | 1671.30 | 1663.41–1678.60 | 43.55% | 58.88 | 6,301 |

The overlay gained 60.72 rating points, 7.18 percentage points of outright
wins, 2.59 mean final money, and 25.3% more objective claims in that frozen
validation field.

## Integrated permanent-field tournament

After registration, the complete 14-bot default field was run for 15,000
games at fresh root `2026073103`, with all charts and player counts, batch size
64, eight workers, and 200 bootstrap samples. The three pure fixed-value bots
are 21.4% of the permanent roster. All 200 bootstrap fits converged and every
bot recorded zero faults.

| Rank | Bot | PL rating | 95% interval | Games | Win rate | Mean money |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed-objective-overlay-v1` | 1680.13 | 1672.66–1689.55 | 4,286 | 39.45% | 57.97 |
| 2 | `fixed-bid-tuned-v1` | 1664.36 | 1656.16–1672.41 | 4,286 | 41.81% | 58.48 |
| 3 | `fixed-bid-diverse-v1` | 1656.93 | 1647.55–1665.15 | 4,286 | 42.79% | 59.15 |
| 4 | `aggressive-v2` | 1656.08 | 1649.65–1664.17 | 4,286 | 32.29% | 56.23 |
| 5 | original `fixed-bid` | 1650.32 | 1642.26–1659.20 | 4,285 | 36.13% | 56.85 |

The overlay was +29.82 rating points above the original in the exact permanent
field. Tuned was +14.04 and diverse +6.61. These marginal bootstrap intervals
describe each rating; they are not paired confidence intervals on the deltas.
The earlier two-root comparison supplies the stronger cross-seed evidence for
the fixed candidates.

## Reproduction

The exact exploratory manifests and runners are committed under
[`scripts/analysis/fixed_bid_search`](../../scripts/analysis/fixed_bid_search/README.md)
and
[`scripts/analysis/fixed_objective_overlay_search`](../../scripts/analysis/fixed_objective_overlay_search/README.md).
Those runbooks contain the commands for every screen, refinement, and
validation root reported above.

The integrated tournament used base commit
`e916ee93e3a79b48b3638a683d04765135720a58` plus the policies documented here:

```bash
UV_CACHE_DIR=/tmp/garboid-uv-cache \
  uv run --extra neural garboid-tournament \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-bid-diverse-v1,fixed-objective-overlay-v1,aggressive-v1,balanced-v1,passive-v1,aggressive-v2,balanced-v2,passive-v2,sdk-greedy-value-v1,vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k \
  --games 15000 \
  --seed 2026073103 \
  --workers 8 \
  --bootstrap-samples 200 \
  --output-dir /tmp/reproduced-fixed-family-default
```

The command writes machine-readable and interactive results to the specified
local output directory. These generated tournament artifacts are intentionally
excluded from version control.

This is an exploratory tournament benchmark, not a held-out promotion gate.
