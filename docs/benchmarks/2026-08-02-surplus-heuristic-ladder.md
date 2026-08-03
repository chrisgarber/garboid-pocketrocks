# Surplus heuristic generation ladder

Date: 2026-08-02

## Hypothesis and generations

The surplus family treats a bid as a purchase in final-score dollars. Its base
policy estimates the terminal value of the awarded resource bundle and shades
that value by `(player_count - 1) / player_count` for a first-price auction.
Each behavior-changing experiment has a separate local simulation identity:

- `surplus-v1`: values only private-hand and publicly revealed information;
- `surplus-v2`: replaces that lower bound with an exact finite-deck posterior;
- `surplus-v3`: adds the payout of newly completed, unclaimed objectives;
- `surplus-v4`: branches from v2 and bids the guaranteed bonus on investments;
- `surplus-v5`: caps v4 bids at the observed action-specific rival upper quartile plus one;
- `surplus-v6`: retries the v3 objective premium behind v5's learned market cap.

At the time of this tournament, the unversioned `surplus` alias selected v5,
the strongest generation in this development field. v3 and v6 remain
registered negative results. The alias later advanced to objective-aware v7;
see the [v7 tuning report](2026-08-02-surplus-v7-objective-tuning.md).

## Development tournament

This was an exploratory development comparison, not a held-out promotion gate.

- Source base commit: `ca35324d07474e6623a815fc8d85a1fd5ec044ec`
- Surplus policy SHA-256: `5e632212714c128be158033ab58a3b672aeeb7173758e6ee4c58dfec58a6c974`
- Root seed: `20260802`
- Coverage: 5,500 games over player counts 3, 4, and 5 and value charts A-E
- Field: all six surplus generations plus `random`, `sdk-greedy-value-v1`,
  `fixed-bid`, `fixed-bid-tuned-v1`, and `fixed-objective-overlay-v2`
- Exposure: 1,999-2,001 games per bot
- Bootstrap: 200 complete-game resamples; all 200 fits converged
- Fault mode: record-and-pass; every bot recorded zero faults
- Ignored summary SHA-256:
  `c5eab5244896b0ea47bb49c1bede818db311568a4f5fc72d501d04ae66424941`

| Generation | PL rating | 95% interval | Mean money | Outright wins | Games |
|---|---:|---:|---:|---:|---:|
| surplus-v1 | 1451.81 | 1440.17-1463.27 | 41.83 | 342 | 2,000 |
| surplus-v2 | 1493.24 | 1483.34-1504.76 | 44.45 | 348 | 2,000 |
| surplus-v3 | 1458.67 | 1448.57-1468.11 | 42.49 | 251 | 1,999 |
| surplus-v4 | 1515.66 | 1505.82-1524.53 | 45.49 | 343 | 2,001 |
| surplus-v5 | 1542.95 | 1532.45-1552.86 | 46.35 | 381 | 1,999 |
| surplus-v6 | 1512.95 | 1503.43-1523.69 | 45.01 | 311 | 2,000 |

The posterior, investment, and public-market features each moved the strongest
line forward. Raw objective premiums moved it backward in both attempts. v5's
interval is above every other surplus generation's interval in this field, so
v5 is the development winner. It remains below the established fixed policies;
no promotion claim is made.

## Reproduction

```bash
uv run --offline garboid-tournament \
  --games 5500 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 20260802 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,sdk-greedy-value-v1,surplus-v1,surplus-v2,surplus-v3,surplus-v4,surplus-v5,surplus-v6 \
  --output-dir artifacts/tournaments/surplus-ladder-final
```

The generated tournament directory stays under ignored `artifacts/`; this note
is the compact committed evidence.
