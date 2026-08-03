# `monte-the-bookie-v1` current-best tournament

Date: 2026-08-03

## Result

`monte-the-bookie-v1` ranked third in a 16-bot fixed-seed field containing the
current curated tournament roster plus `surplus-v10`. It finished behind
`surplus-v10` and `fixed-objective-overlay-v3`, and ahead of its former strongest
comparison, `fixed-objective-overlay-v2`. No bot faulted.

The Monte Carlo bot's interval overlaps `fixed-objective-overlay-v3`'s interval,
so this run does not establish a difference between those two bots. Its upper
bound is below `surplus-v10`'s lower bound, while its lower bound is above
`fixed-objective-overlay-v2`'s upper bound. These are marginal bootstrap
intervals from one fixed-seed tournament, not a held-out promotion test.

| Rank | Bot | PL rating (95% interval) | Appearances | Outright win rate | Mean money | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `surplus-v10` | 1652.27 (1642.01–1662.77) | 3,750 | 51.95% | 56.15 | 0 |
| 2 | `fixed-objective-overlay-v3` | 1641.33 (1631.32–1651.42) | 3,750 | 48.64% | 56.76 | 0 |
| 3 | `monte-the-bookie-v1` | 1625.71 (1615.93–1634.67) | 3,750 | 45.41% | 53.64 | 0 |
| 4 | `fixed-objective-overlay-v2` | 1571.78 (1564.47–1579.33) | 3,750 | 34.11% | 52.05 | 0 |

## Configuration and provenance

- Source revision: `657dfa1369bf48b86d945a1a4f280b8bfb18006e`
- Games: 15,000
- Root seed: `2026080301`
- Player counts: three, four, and five
- Value charts: A through E
- Field size: 16 released bot identities
- Batch size: 64
- Workers: 11
- Bootstrap samples: 200; all 200 converged
- Appearances: 3,749 to 3,751 per bot
- Faults: zero across the field
- Decision reports: disabled
- `summary.json` SHA-256:
  `b7772987a0b0d23120a7df7646664f71a4287bc77acd30895ace7bcc074c8b3e`

The committed machine-readable result is
[`tournaments/2026-08-03-monte-the-bookie-v1-current-best/summary.json`](tournaments/2026-08-03-monte-the-bookie-v1-current-best/summary.json).
It is committed because the PR tournament-reporting contract requires durable,
reviewable machine-readable evidence alongside the concise benchmark note.
Raw run outputs remain under the gitignored `artifacts/` tree.

## Reproduction

```bash
uv run --extra neural garboid-tournament \
  --bots monte-the-bookie-v1,fixed-objective-overlay-v3,fixed-objective-overlay-v2,fixed-objective-overlay-v1,surplus-v10,fixed-bid-tuned-v1,aggressive-v2,fixed-bid-diverse-v1,balanced-v2,fixed-bid,vector_ppo_large_v1_g350k,passive-v2,passive-v1,aggressive-v3,balanced-v3,passive-v3 \
  --games 15000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080301 \
  --workers 11 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --output-dir artifacts/tournaments/pr49-current-best-20260803
```
