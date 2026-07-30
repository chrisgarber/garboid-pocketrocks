# `vector_ppo_large_v1_g350k` default tournament

## Result

The large PPO policy trained for exactly 349,860 games ranked fourth of nine
bots. It finished ahead of every v1 heuristic and the smoke neural policy; only
the three v2 heuristics ranked higher.

| Rank | Bot | PL rating | 95% interval | Games | Outright win rate | Mean money | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | aggressive-v2 | 1690.35 | 1685.0–1696.9 | 6,667 | 40.0% | 59.18 | 0 |
| 2 | balanced-v2 | 1685.15 | 1678.8–1690.6 | 6,666 | 39.0% | 59.82 | 0 |
| 3 | passive-v2 | 1661.35 | 1655.3–1665.7 | 6,667 | 35.5% | 58.55 | 0 |
| 4 | vector_ppo_large_v1_g350k | 1653.88 | 1646.8–1661.7 | 6,667 | 43.6% | 58.73 | 0 |
| 5 | passive-v1 | 1623.67 | 1618.1–1629.8 | 6,667 | 32.1% | 57.76 | 0 |
| 6 | balanced-v1 | 1479.75 | 1473.8–1485.1 | 6,666 | 12.3% | 47.25 | 0 |
| 7 | aggressive-v1 | 1390.04 | 1383.7–1395.3 | 6,666 | 6.7% | 41.56 | 0 |
| 8 | vector_ppo_small_v1_g1500 | 1269.04 | 1260.6–1275.8 | 6,667 | 8.3% | 27.73 | 0 |
| 9 | random | 1046.78 | 1035.4–1057.5 | 6,667 | 1.4% | 12.56 | 0 |

The fixed-seed result supports a substantial improvement over the smoke policy.
The large model's rating is 384.85 points higher, its interval is wholly above
the smoke interval, its outright win rate is 43.6% rather than 8.3%, and its
mean final money is 58.73 rather than 27.73. Its interval is also wholly above
every v1 heuristic. The interval overlaps passive-v2, so this tournament does
not distinguish those two policies conclusively.

## Configuration

Command:

```bash
/usr/bin/time -lp uv run --extra neural garboid-tournament \
  --output-dir docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default
```

- Tournament source commit: `701745504a057671e136ea865807f46594ada5b3`
- Training source commit: `154e17be349670c14342fcaa8b5dc7c7d413f760`
- Exact training age: 349,860 games and 196 updates
- Games: 15,000
- Players: 3, 4, and 5
- Charts: A, B, C, D, and E
- Root seed: 0
- Workers: 17
- Engine batch size: 64
- Bootstrap samples: 200/200 converged
- Pair exposure: 2,637 minimum, 2,639 median, 2,641 maximum
- Model fit: converged in 21 iterations

The checkpoint parameter digest is
`088160ad4006b2bac3691980d7f3e9dc56635fd57e6ad2b94068497e199f0e5c`;
the model file SHA-256 is
`2ff577e25cf20f4217290a18ecf3d23188d0a5cddb392aa571ba54f2e1cd8974`.

## Runtime

The complete simulation, fit, 200 bootstrap fits, and artifact generation took
51.88 seconds, or 289.13 games/second end to end. `/usr/bin/time` reported
639.30 user seconds and 19.00 system seconds, averaging 12.69 CPU-core
equivalents. Maximum resident set size was 518,275,072 bytes.

The previous eight-bot smoke tournament took 35.45 seconds. Adding the
1.65-million-parameter large policy and a ninth field entrant added 16.43
seconds while leaving the SDK batch engine active. Both neural wrappers still
evaluate one observation per forward pass inside each engine batch.

## Artifacts

The machine-readable and interactive reports are committed under
[`docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/`](tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/):

- [`ratings.csv`](tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/ratings.csv)
- [`summary.json`](tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/summary.json)
- [`report.html`](tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/report.html)

