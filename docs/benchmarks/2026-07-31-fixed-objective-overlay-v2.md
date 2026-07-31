# `fixed-objective-overlay-v2` tournament

## Result

`fixed-objective-overlay-v2` ranked first in its 15-bot default-field
tournament. It scored 1700.24, 41.69 rating points above the preserved v1
generation. Their marginal 95% bootstrap intervals did not overlap.

| Rank | Bot | PL rating | 95% interval | Games | Win rate | Mean money | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed-objective-overlay-v2` | 1700.24 | 1692.00–1709.06 | 4,000 | 44.65% | 59.16 | 0 |
| 2 | `fixed-objective-overlay-v1` | 1658.56 | 1649.47–1666.26 | 4,000 | 35.38% | 56.42 | 0 |
| 3 | `fixed-bid-tuned-v1` | 1645.00 | 1635.49–1654.58 | 3,999 | 38.56% | 57.11 | 0 |
| 4 | `aggressive-v2` | 1643.26 | 1636.74–1650.77 | 3,999 | 30.73% | 54.95 | 0 |
| 5 | `fixed-bid-diverse-v1` | 1639.60 | 1631.25–1649.22 | 4,000 | 42.43% | 58.25 | 0 |

V2 gained 9.28 percentage points of outright wins and 2.73 mean final money
over v1 in the same field. Every bot completed without faults, and all 200
bootstrap fits converged.

## Policy and identity

V1 remains immutable and selectable as `fixed-objective-overlay-v1`. Both
generations use the same shared overlay engine, posterior, objective signal,
private-information weight, thresholds, and maximum adjustment. Their only
policy difference is the immutable fixed profile underneath that engine:

| Generation | Base profile |
| --- | --- |
| v1 | `(5, 10, 2, 4, 4, 9)` |
| v2 | `(5, 10, 2, 5, 4, 7)` |

The vectors are ordered as one-resource auction, two-resource auction, loan
10, loan 20, invest 5, and invest 10. Because both profiles have auction bases
of 5 and 10, their objective-aware auction logic is exactly the same. V2 bids
5 rather than 4 for loan 20 and 7 rather than 9 for invest 10.

## Configuration

- Games: 15,000
- Root seed: `2026073104`
- Charts: A through E
- Player counts: three, four, and five
- Workers: 8
- Batch size: 64
- Bootstrap samples: 200/200 converged
- Pair exposure: 902 minimum, 905 median, 910 maximum
- Model fit: converged in 32 iterations

Reproduce to a fresh directory with the exact field:

```bash
UV_CACHE_DIR=/tmp/garboid-uv-cache \
  uv run --extra neural garboid-tournament \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-bid-diverse-v1,fixed-objective-overlay-v1,fixed-objective-overlay-v2,aggressive-v1,balanced-v1,passive-v1,aggressive-v2,balanced-v2,passive-v2,sdk-greedy-value-v1,vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k \
  --games 15000 \
  --seed 2026073104 \
  --workers 8 \
  --bootstrap-samples 200 \
  --output-dir /tmp/reproduced-fixed-objective-overlay-v2-default
```

## Artifacts

The machine-readable and interactive reports are committed under
[`docs/benchmarks/tournaments/2026-07-31-fixed-objective-overlay-v2-default/`](tournaments/2026-07-31-fixed-objective-overlay-v2-default/):

- [`ratings.csv`](tournaments/2026-07-31-fixed-objective-overlay-v2-default/ratings.csv)
- [`summary.json`](tournaments/2026-07-31-fixed-objective-overlay-v2-default/summary.json)
- [`report.html`](tournaments/2026-07-31-fixed-objective-overlay-v2-default/report.html)

This is an exploratory tournament benchmark, not a held-out promotion gate.
