# `vector_ppo_small_v1_g1500` default tournament

## Result

The frozen 1,500-game small PPO smoke policy ranked seventh of eight bots. It
comfortably exceeded random but remained below every v1 and v2 heuristic.

| Rank | Bot | PL rating | 95% interval | Games | Outright win rate | Mean money | Faults |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | aggressive-v2 | 1711.42 | 1706.2–1716.4 | 7,500 | 43.0% | 61.43 | 0 |
| 2 | balanced-v2 | 1710.86 | 1704.1–1717.5 | 7,500 | 43.5% | 62.38 | 0 |
| 3 | passive-v2 | 1697.74 | 1692.5–1704.4 | 7,500 | 41.1% | 61.53 | 0 |
| 4 | passive-v1 | 1650.78 | 1644.2–1656.1 | 7,500 | 35.1% | 59.94 | 0 |
| 5 | balanced-v1 | 1496.66 | 1491.5–1502.1 | 7,500 | 13.8% | 48.61 | 0 |
| 6 | aggressive-v1 | 1392.77 | 1388.0–1397.6 | 7,500 | 7.4% | 42.50 | 0 |
| 7 | vector_ppo_small_v1_g1500 | 1270.72 | 1263.6–1277.7 | 7,500 | 8.5% | 27.28 | 0 |
| 8 | random | 1069.05 | 1060.3–1079.3 | 7,500 | 2.0% | 13.79 | 0 |

The neural policy recorded 639 outright wins in 7,500 appearances. Its rating
interval is entirely above random's interval, while remaining entirely below
aggressive-v1's interval.

## Configuration

Command:

```bash
uv run --extra neural garboid-tournament \
  --output-dir docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default
```

- Source commit: `cf1be37d7d15093f256821c58bf387b77e43f173`
- Games: 15,000
- Players: 3, 4, and 5
- Charts: A, B, C, D, and E
- Root seed: 0
- Workers: 17
- Engine batch size: 64
- Bootstrap samples: 200/200 converged
- Pair exposure: 3,392 minimum, 3,393 median, 3,394 maximum
- Model fit: converged in 18 iterations

The committed checkpoint parameter digest is
`4c75fa7aa08432a7f503d83d23332b0ee5d4f63f8d1e4abb3d26e02d5c0ee16a`.

## Runtime

The complete simulation, fit, 200 bootstrap fits, and artifact generation took
35.45 seconds, or 423.13 games/second end to end. `/usr/bin/time` reported
384.42 user seconds and 18.60 system seconds, averaging about 11.4 CPU-core
equivalents over the run.

Two 1,500-game, zero-bootstrap probes isolated the neural policy's steady-state
cost:

| Field | Wall time | Games/second |
| --- | ---: | ---: |
| Seven non-neural defaults | 2.82s | 531.91 |
| Eight defaults including neural | 6.24s | 240.38 |

The short neural probe pays model/process startup over much smaller worker
chunks. At the full 15,000-game size, that startup is amortized and total
runtime remains close to the historical roughly 30-second tournament.

`BatchSimEngine` remains active. The current neural adapter evaluates one
observation at a time inside each engine batch; batching neural observations
into shared forward passes is the next clear throughput optimization.

## Artifacts

The machine-readable and interactive reports are committed under
`docs/benchmarks/tournaments/2026-07-30-vector-ppo-small-v1-g1500-default/`.
