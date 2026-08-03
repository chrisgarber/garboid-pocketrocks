# `monte-the-bookie-v2`

Date: 2026-08-03

A second Monte Carlo best-response generation. `monte-the-bookie-v1` is unchanged and
stays registered and selectable.

## Why v1 was behind

The [current-best note](2026-08-03-monte-the-bookie-v1-current-best.md) has v1 third,
26.56 rating points behind `surplus-v10`. Two causes, and the second was the real one.

**The bid prior was fitted on a field that no longer exists.** It was built before the
surplus family and the objective overlays through v3. Refitting against the current
field over 73,573 bidding decisions shows the shape has inverted:

| Bucket | v1 table | Current field |
| --- | ---: | ---: |
| `auction1/late` | 0.380 | 0.537 |
| `auction2/late` | 0.419 | 0.687 |

The old field ran out of cash and bid *less* late. This field escalates. A bot carrying
the v1 table therefore underbids the endgame systematically.

**The coefficients were searched against the wrong objective.** Every earlier search
maximized outright win rate. Plackett-Luce rating rewards consistent placement across
multiplayer games, and the two diverge sharply here. In balanced two-versus-two duels
over 1,200 games each, v1 already beats every rival:

| Rival | v1 share |
| --- | ---: |
| `surplus-v10` | 56.2% |
| `surplus-v9` | 56.5% |
| `fixed-objective-overlay-v3` | 51.2% |
| `fixed-objective-overlay-v2` | 61.1% |
| `fixed-bid-tuned-v1` | 73.5% |
| `aggressive-v3` | 58.8% |
| `vector_ppo_large_v1_g350k` | 70.2% |

v1 wins duels and still rates third, because it is high-variance: it wins or finishes
poorly, and rating punishes the second half of that.

## What changed

v2 carries `BID_PRIOR_V2` and coefficients searched against **mean normalized finish**
instead of win rate, over three, four and five players.

The retune is mostly a reduction. Speculative upside costs placement:

| Coefficient | v1 | v2 |
| --- | ---: | ---: |
| Liquidity strength | 3.62 | 6.00 |
| Future cash weight | 0.38 | 0.24 |
| Objective progress weight | 0.46 | 0.00 |
| Prior weight | 17.11 | 12.48 |
| Scarcity weight | 0.81 | 0.55 |
| Denial weight | 0.45 | 0.23 |
| Deck-pressure weight | 0.68 | 0.14 |
| Standings weight | 0.39 | 0.05 |

Only `liquidity_strength` rose, and that is the term protecting the downside. The
scoreboard term that made v1 take variance when behind is now effectively off: it helped
win rate and hurt rating.

## Tournament

Curated roster plus `surplus-v9`/`surplus-v10` and both Monte generations. 15,000 games,
root seed 2026080301, three through five players, charts A-E, batch size 64, 11 workers,
200 bootstrap samples all converged, no faults.

| Rank | Bot | PL rating (95% interval) | Outright win rate |
| ---: | --- | ---: | ---: |
| 1 | `monte-the-bookie-v2` | 1664.77 (1652.8-1676.3) | 51.2% |
| 2 | `surplus-v10` | 1617.36 (1607.7-1627.3) | 44.6% |
| 3 | `monte-the-bookie-v1` | 1601.44 (1592.9-1611.7) | 40.7% |
| 4 | `fixed-objective-overlay-v3` | 1599.03 (1589.4-1608.2) | 41.5% |
| 5 | `surplus-v9` | 1589.97 (1582.5-1600.5) | 40.0% |
| 6 | `fixed-objective-overlay-v2` | 1542.99 (1535.2-1551.3) | 27.7% |
| 7 | `fixed-objective-overlay-v1` | 1541.85 (1533.5-1551.3) | 25.2% |
| 8 | `aggressive-v3` | 1521.46 (1512.5-1531.8) | 22.1% |

v2's interval does not overlap `surplus-v10`. The 26.56-point deficit became a
47.41-point lead.

## Held-out promotion

480 matched pairs, 960 games, 1,000 bootstrap replicates at seed 0, zero faults.

| Incumbent | Difference | 95% interval | Promoted |
| --- | ---: | --- | --- |
| `surplus-v10` | +73.55 | 26.6 to 121.1 | yes |
| `fixed-objective-overlay-v3` | +46.74 | -3.2 to 96.6 | no |
| `monte-the-bookie-v1` | +12.04 | -31.7 to 57.8 | no |

The gate establishes v2 over `surplus-v10`. It does **not** establish v2 over
`fixed-objective-overlay-v3` or over v1: both point estimates are positive, both
intervals include zero.

## Notes and limitations

- The tournament separates v2 from v1 with non-overlapping intervals, but the promotion
  corpus does not. 960 matched games has less resolving power than 15,000 tournament
  games, so treat "better than v1" as suggested rather than demonstrated.
- Head-to-head v2 gains sharply against the surplus family (63.0% and 63.3%, from 56.2%
  and 56.5%) and gives up a little against the overlays and v1, landing at 48-49% where
  v1 held 51-61%. v2 trades duel strength for placement, which is the intended direction
  for a rating target but is a real trade rather than a free win.
- The finish-metric search was run twice with different seeds; the second found a worse
  optimum (0.733 against 0.747) and was discarded. These coefficients are one good local
  optimum, not a converged one.
- v1 remains the better choice if outright wins matter more than placement.

## Reproduction

```bash
uv run --extra neural garboid-tournament \
  --games 15000 --seed 2026080301 --bootstrap-samples 200 --workers 11 \
  --output-dir tournament-results

uv run --extra neural garboid-promote \
  --candidate monte-the-bookie-v2 --incumbent surplus-v10 \
  --output-dir promotion-results/monte-the-bookie-v2
```
