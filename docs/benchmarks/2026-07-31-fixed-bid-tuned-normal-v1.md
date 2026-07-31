# `fixed-bid-tuned-normal-v1` tournament

## Result

`fixed-bid-tuned-normal-v1` ranked seventh in a ten-bot comparison field. It
scored 1488.10, which is 43.97 rating points below its deterministic parent,
`fixed-bid-tuned-v1`. Their marginal 95% bootstrap intervals did not overlap.

| Rank | Bot | PL rating | 95% interval | Games | Win rate | Mean money | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `fixed-objective-overlay-v2` | 1639.75 | 1633.39–1647.83 | 6,000 | 47.50% | 56.12 | 0 |
| 2 | `fixed-objective-overlay-v1` | 1614.17 | 1609.56–1619.96 | 6,000 | 37.40% | 53.83 | 0 |
| 3 | `fixed-bid-tuned-v1` | 1532.07 | 1523.72–1538.76 | 6,000 | 33.52% | 51.03 | 0 |
| 4 | `aggressive-v2` | 1523.08 | 1517.04–1528.16 | 6,000 | 20.68% | 48.27 | 0 |
| 5 | `vector_ppo_large_v1_g350k` | 1521.55 | 1513.45–1528.29 | 6,000 | 34.52% | 49.02 | 0 |
| 6 | `balanced-v2` | 1501.85 | 1496.52–1506.63 | 6,000 | 16.40% | 47.50 | 0 |
| 7 | `fixed-bid-tuned-normal-v1` | 1488.10 | 1482.74–1495.50 | 6,000 | 27.30% | 48.40 | 0 |

The random offset reduced outright wins by 6.22 percentage points and mean
final money by 2.63 relative to the static tuned policy. The candidate remains
registered for explicit experiments but is not part of the top-ten default.

## Policy

The bot starts with the tuned fixed profile `(5, 10, 2, 5, 4, 7)`. On every
bid decision it independently samples a standard normal offset `N(0,1)`,
rounds it to the nearest integer, adds it to that action's target, and clips
the result to the legal range. A nonpositive result passes. Reveal behavior is
unchanged. Simulation-provided brain seeds make the random sequence stable
across scalar, batch, serial, and parallel execution.

## Configuration

- Games: 15,000
- Root seed: `2026073105`
- Charts: A through E
- Player counts: three, four, and five
- Workers: 8
- Batch size: 64
- Bootstrap samples: 200/200 converged
- Fixed-value identities: 2 of 10 (20%)
- Pair exposure: 2,110 minimum, 2,111 median, 2,113 maximum
- Model fit: converged in 16 iterations

Reproduce to a fresh directory with the exact field:

```bash
UV_CACHE_DIR=/tmp/garboid-uv-cache \
  uv run --extra neural garboid-tournament \
  --bots fixed-bid-tuned-normal-v1,fixed-bid-tuned-v1,fixed-objective-overlay-v2,fixed-objective-overlay-v1,aggressive-v2,balanced-v2,vector_ppo_large_v1_g350k,passive-v2,passive-v1,balanced-v1 \
  --games 15000 \
  --seed 2026073105 \
  --workers 8 \
  --bootstrap-samples 200 \
  --output-dir /tmp/reproduced-fixed-bid-tuned-normal-v1
```

## Artifacts

The reports are committed under
[`docs/benchmarks/tournaments/2026-07-31-fixed-bid-tuned-normal-v1/`](tournaments/2026-07-31-fixed-bid-tuned-normal-v1/):

- [`ratings.csv`](tournaments/2026-07-31-fixed-bid-tuned-normal-v1/ratings.csv)
- [`summary.json`](tournaments/2026-07-31-fixed-bid-tuned-normal-v1/summary.json)
- [`report.html`](tournaments/2026-07-31-fixed-bid-tuned-normal-v1/report.html)

This is an exploratory tournament benchmark, not a held-out promotion gate.
