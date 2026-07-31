# Hand-tuned heuristic v3 personalities

Date: 2026-07-31

The user chose to update v3 in place. The strongest broad-field evolved policy
became `balanced-v3`; aggressive and passive were hand-tuned around it to make
their names behaviorally meaningful again. The evaluated worktree was based on
commit `64ce081caa875ddab34371937b92e82172d6543c` with the profile-only changes
below.

| Personality | Liquidity | Future cash | Objective progress | Bid shading |
| --- | ---: | ---: | ---: | ---: |
| Aggressive | 1.50 | 0.50 | 1.00 | 0.05 |
| Balanced | 1.40 | 1.05 | 0.70 | 0.30 |
| Passive | 0.25 | 2.00 | 0.10 | 0.50 |

Aggressive couples strong objective and loan-cash valuation with almost no
shading. Passive does the opposite: it protects early future cash, applies
little speculative objective value, and offers half its reservation price.

## Full tournament

The default 14-bot field played 15,000 games over charts A-E and three through
five players, using root seed 0, 200 bootstrap samples, and eight workers. All
200 bootstrap fits converged, the schedule produced 4,285 or 4,286 appearances
per bot, and no bot faulted.

| Rank | Bot | Rating | 95% interval | Mean money |
| ---: | --- | ---: | ---: | ---: |
| 1 | balanced-v3 | 1719.66 | 1712.02 to 1727.90 | 62.07 |
| 2 | aggressive-v2 | 1673.25 | 1665.89 to 1681.91 | 57.09 |
| 3 | fixed-bid | 1668.75 | 1658.67 to 1679.03 | 59.13 |
| 4 | passive-v3 | 1651.15 | 1643.15 to 1659.04 | 61.42 |
| 5 | balanced-v2 | 1650.78 | 1641.58 to 1658.34 | 57.01 |
| 6 | vector_ppo_large_v1_g350k | 1620.85 | 1609.36 to 1630.48 | 55.88 |
| 7 | passive-v2 | 1620.68 | 1612.68 to 1627.04 | 55.61 |
| 8 | aggressive-v3 | 1620.29 | 1613.47 to 1628.33 | 53.38 |

The tournament summary SHA-256 is
`8417d489f6634a47a2fff390b6f500482dc9fbca465578071127aea0a85c4cb9`.

## Behavior check

The decision report reconciled 1,099,392 decisions. The profiles separated
strongly on every requested behavior:

| Measure | Aggressive v3 | Balanced v3 | Passive v3 |
| --- | ---: | ---: | ---: |
| Mean submitted auction bid | 6.33 | 5.04 | 4.19 |
| Mean objective value per auction | 3.71 | 2.01 | 0.97 |
| Mean maximum observed objectives owned | 1.28 | 0.81 | 0.51 |
| Loan opportunity bid rate | 87.42% | 72.12% | 1.08% |
| Mean cash before an auction | 10.85 | 16.80 | 19.10 |
| Mean cash before a late auction | 3.23 | 7.38 | 11.36 |
| Mean auction cash advantage over opponents | -2.06 | +4.83 | +7.57 |

Aggressive v3 now plays exactly as named, but the extra risk costs about 99
rating points relative to balanced v3 and leaves it eighth of 14. Passive v3
also plays as named and remains competitive at fourth, statistically level
with balanced v2. The evidence supports retaining passive v3 and retiring
aggressive v3 from curated defaults if competitive strength is the release
criterion.

The behavior summary SHA-256 is
`e6abcce101ca89e4f2e0cc6bcbb457d7aa66426da55af9177cf2af806960a0fc`.
The run is reproducible with:

```bash
uv run --extra neural garboid-tournament \
  --games 15000 --seed 0 --bootstrap-samples 200 --workers 8 \
  --decision-reports \
  --output-dir artifacts/tournaments/2026-07-31-hand-tuned-v3-full
```

Compact ratings, summary, behavior aggregates, and the local aggregation
script remain under the gitignored tournament directory. The reproducible raw
decision traces, game summaries, and additive decision slices were removed
after aggregation.
