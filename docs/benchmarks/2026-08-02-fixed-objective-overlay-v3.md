# `fixed-objective-overlay-v3` tournament

## Result

The final guarantee-only `fixed-objective-overlay-v3` ranked first in a 13-bot
fixed-seed field. It led `fixed-objective-overlay-v2` by 116.19 Plackett-Luce
rating points, 22.56 percentage points of outright win rate, 6.93 mean final
money, and 0.1275 mean normalized finish. Both generations completed without
faults, and their marginal 95% bootstrap intervals do not overlap.

V3 scored 73.75% across the 1,217 tournament games containing both v3 and v2,
counting tied finishes as one half.

| Rank | Bot | PL rating | 95% interval | Games | Win rate | Mean money | Mean finish | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed-objective-overlay-v3` | 1725.60 | 1716.18–1735.22 | 4,615 | 64.40% | 62.13 | 0.7994 | 0 |
| 2 | `fixed-objective-overlay-v2` | 1609.40 | 1601.04–1617.10 | 4,616 | 41.83% | 55.20 | 0.6719 | 0 |

This is strong fixed-seed exploratory evidence, not a held-out promotion gate.

## Final policy

V2 remains immutable and selectable as `fixed-objective-overlay-v2`. V3 uses
the fixed action profile `(5, 10, 3, 6, 4, 7)` in wire-action order:

| Action | V2 | V3 |
| --- | ---: | ---: |
| One-resource auction | 5 | 5 |
| Two-resource auction | 10 | 10 |
| Loan 10 | 2 | 3 |
| Loan 20 | 5 | 6 |
| Invest 5 | 4 | 4 |
| Invest 10 | 7 | 7 |

The objective and private-information overlay remains unchanged. Resource and
investment targets were constrained to stay flat or decrease during tuning;
loan targets could move in either direction. The selected profile raises only
the two loan bids.

V3 has exactly one dynamic override. On a resource auction, when every
opponent has less cash than v3's planned bid, it submits the cheapest bid that
guarantees the win:

- bid the maximum opponent cash when v3 wins a tie at that amount;
- otherwise bid one more than the maximum opponent cash; and
- never raise the original planned bid.

This calculation uses current cash and exact tiebreak order. Public bid history
does not affect v3 decisions, and there is no probabilistic `history_price`
rule.

## Tuning decision

The constrained coordinate and combination screen ran 1,200 games per profile
at seeds `2026080401` and `2026080402`. The starting v2 profile was
`(5, 10, 2, 5, 4, 7)`. The strongest screen profile was
`(5, 10, 3, 6, 4, 7)`, averaging +134.57 rating points over v2 and an 80.15%
direct score.

Sixteen finalists and targeted fixed-bid reductions then ran 4,000 games per
profile at seeds `2026080411` and `2026080412`. The selected profile averaged
+112.49 rating points over v2 and a 73.46% direct score. Every tested downward
resource or investment adjustment weakened that profile; the closest was
Invest 5 from four to three, at +109.09.

| Refinement profile | Mean rating delta vs v2 | Direct score |
| --- | ---: | ---: |
| `(5, 10, 3, 6, 4, 7)` | +112.49 | 73.46% |
| `(5, 10, 3, 6, 3, 7)` | +109.09 | 72.54% |
| `(5, 10, 3, 6, 2, 7)` | +107.49 | 72.38% |
| `(5, 10, 3, 6, 4, 6)` | +105.50 | 73.38% |
| untuned `(5, 10, 2, 5, 4, 7)` | +25.03 | 53.31% |

On a separate 15,000-game selection check at seed `2026080421`, using the same
search identity and schedule, the selected profile scored 1716.35 (+105.25
over v2) and 71.90% directly. The untuned guarantee-only profile scored 1638.78
(+1.44 over v2) and 48.60% directly.

Search output remains under the gitignored
`artifacts/tuning/fixed-objective-overlay-v3-fixed/` tree. The reproducible
search drivers are in
`scripts/analysis/fixed_objective_overlay_v3_search/`.

## Decision telemetry

The final tournament enabled decision reports for the exact registered v3
identity.

| Metric | Result |
| --- | ---: |
| V3 bid decisions | 72,440 |
| Resource-auction decisions | 47,211 |
| Exact guarantee selected | 2,390 (5.06% of resource auctions) |
| Submitted bid changed | 1,986 (4.21% of resource auctions) |
| Submitted-bid reduction | 6,281 units |
| Guaranteed auctions won | 2,390 / 2,390 (100%) |

The guarantee classification can leave the bid unchanged when the fixed
profile already selected the exact safe amount. The submitted-bid reduction is
the sum of planned minus chosen bids, not a claim that every reduced unit would
otherwise have been paid.

## Discarded prototype

An earlier development prototype also used a probabilistic public-history
pricing rule. That behavior exceeded the requested scope and was removed before
the constrained search and final evidence above. Its tournament outputs remain
gitignored run exhaust and do not define v3.

## Final configuration

- Games: 15,000
- Root seed: `2026080431`
- Charts: A through E
- Player counts: three, four, and five
- Field size: 13 non-neural released bots
- Workers: 8
- Batch size: 64
- Bootstrap samples: 200; all 200 converged
- Decision reports: enabled
- Source revision: `ca35324d07474e6623a815fc8d85a1fd5ec044ec` plus the v3 change
- `summary.json` SHA-256: `877cba558a737cea38b0a1f339df479ac3180e2968cb3308f1cf63f02fcbb4c1`

Reproduce the final tournament to a fresh ignored artifact directory:

```bash
UV_CACHE_DIR=/tmp/garboid-uv-cache \
  uv run garboid-tournament \
  --bots fixed-objective-overlay-v3,fixed-objective-overlay-v2,fixed-objective-overlay-v1,fixed-bid-tuned-v1,aggressive-v2,fixed-bid-diverse-v1,balanced-v2,fixed-bid,passive-v2,passive-v1,aggressive-v3,balanced-v3,passive-v3 \
  --games 15000 \
  --seed 2026080431 \
  --workers 8 \
  --bootstrap-samples 200 \
  --decision-reports \
  --output-dir artifacts/tournaments/fixed-objective-overlay-v3-fixed-retuned-20260802
```

The command writes its full reports under the gitignored `artifacts/` tree.
