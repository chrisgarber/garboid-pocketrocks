# Surplus v10 action-aware liquidity

Date: 2026-08-02

## Decision

Release `surplus-v10` and advance the local `surplus` alias from v9 to v10.
Historical v1-v9 identities and behavior remain selectable. This is a local
simulation release, not a remote promotion.

V10 replaces v9's resource-count-only liquidity thresholds with a public
estimate of the market cost of the remaining resource deck:

```text
future resources
* ((remaining Auction1 * Auction1 price)
   + (remaining Auction2 * Auction2 price))
  / (remaining Auction1 + 2 * remaining Auction2)
```

Remaining action counts come from the known action deck minus public turn
openings. Auction prices are the upper-quartile rival bid plus one, falling
back to 5 for `Auction1` and 10 for `Auction2` before observations exist.

The frozen policy:

- reserves `1/8` of projected resource spend before resource bids;
- reserves `3/10` before locking cash in investments;
- seeks a loan while cash is below `3/4` of projected resource spend;
- opens loan bidding at 35% of principal (`3`/`7`) and follows observed rival
  prices up to a 40% ceiling (`4`/`8`);
- retains v9's resource, immediate-objective, progress, denial, reserve-release,
  player-count shading, and market-cap weights.

This preserves v9's economic model while making cash demand depend on the
remaining action mix and actual market prices.

## Development search

Every ordinary candidate was compared with `fixed-bid`,
`fixed-bid-tuned-v1`, `fixed-objective-overlay-v2`, and `surplus-v9` across
player counts 3-5 and charts A-E. Receipts remain ignored under
`artifacts/tournaments/`.

| Stage | Seed | Games | Search | Result |
|---|---:|---:|---|---|
| Initial trace | 2026080601 | 2,400 | Neutral action-aware translation of v9 thresholds | Untuned v10 led v9 by 6.66 rating points |
| Coarse liquidity | 2026080602 | 1,200/candidate | Resource and investment reserves, trigger, fallbacks | Lighter resource reserve and lower trigger led |
| Combined liquidity | 2026080603 | 1,600/candidate | Winning reserves, triggers, fallbacks, loan fee | Loan clearing price dominated smaller interactions |
| Adaptive loan pricing | 2026080604 | 1,600/candidate | Opening price and adaptive ceilings | Conservative opening plus 40% ceiling led fixed ceilings |
| Loan-pressure robustness | 2026080605 | 2,400/field | Standard, $3/$6-$4/$8, and mixed fields | Candidate beat v9 in all three fields |
| Surrounding values | 2026080606 | 1,600/candidate | Resources, objectives, progress, investments, release | V9's original value weights remained locally strongest |
| Final combinations | 2026080607 | 2,400/candidate | Objective and reserve combinations | Center policy led on rating and mean money |

The robustness fields used synthetic fixed bidders at loan schedules `3/6`,
`3/7`, and `4/8`. The finalist beat v9 by 83.11 points in the standard field,
25.56 in the loan-pressure field, and 18.68 in the mixed-pressure field. It
therefore does not rely solely on clearing one exact v9 loan threshold.

Important negative results:

- lowering the loan opening bid to 25% lost more than 100 rating points;
- fixed loan ceilings remained sensitive to the comparison field;
- changing resource value away from 3/4 hurt after adding dynamic liquidity;
- larger objective-progress weights remained harmful.

## Held-out confirmation

Confirmation used untouched root seed `2026080699`, all player counts 3-5,
all value charts A-E, batch size 64, and record-and-pass fault handling. No
coefficient was changed after reading these results.

The 6,000-game focused field gave each bot 4,800 games and ran 200
complete-game bootstrap resamples; all fits converged:

| Bot | PL rating | 95% interval | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|---:|
| surplus-v10 | 1625.95 | 1619.3-1632.7 | 55.24 | 2,409 | 0 |
| surplus-v9 | 1535.42 | 1529.0-1541.9 | 49.90 | 1,467 | 0 |
| fixed-objective-overlay-v2 | 1513.29 | 1507.2-1519.2 | 47.45 | 1,005 | 0 |
| fixed-bid | 1432.05 | 1426.9-1437.9 | 42.44 | 424 | 0 |
| fixed-bid-tuned-v1 | 1393.29 | 1387.6-1399.4 | 41.26 | 495 | 0 |

V10 gained 90.53 rating points and 5.34 mean-money points over v9. Their
intervals are disjoint.

Decision traces show that the gain matches the intended mechanism:

| Measure per game | surplus-v10 | surplus-v9 |
|---|---:|---:|
| Final cash | 6.84 | 5.57 |
| Resource cards | 5.00 | 3.79 |
| Item value | 55.65 | 41.68 |
| Item value minus resource spend | 24.63 | 18.48 |
| Objective payout | 11.28 | 7.05 |
| Objectives claimed | 1.51 | 1.00 |
| Investment bonus | 5.03 | 3.73 |
| Loan principal | 28.43 | 11.71 |
| Loan fees | 9.85 | 3.51 |

In games containing both generations, v10 averaged 51.28 versus v9's 44.75
and finished ahead 62.5% of the time.

The 5,200-game all-generation field gave each bot 1,385-1,388 games:

| Bot | PL rating | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|
| surplus-v10 | 1801.55 | 60.57 | 870 | 0 |
| surplus-v9 | 1782.93 | 60.46 | 807 | 0 |
| fixed-objective-overlay-v2 | 1695.57 | 56.73 | 629 | 0 |
| fixed-bid | 1627.66 | 53.64 | 495 | 0 |
| surplus-v8 | 1621.72 | 50.33 | 366 | 0 |
| fixed-bid-tuned-v1 | 1621.16 | 54.19 | 536 | 0 |

The broad-field gain over v9 is smaller at 18.62 rating points and 0.11 mean
money, but v10 ranks first on the untouched seed in both fields. Ratings
remain field-dependent; this is an improvement claim, not evidence against
every possible unseen strategy.

## Reproduction and provenance

- Source base commit: `ca35324d07474e6623a815fc8d85a1fd5ec044ec`
- Surplus policy SHA-256: `22d1fdbf88159ea5b592e44eb67bd81ec63fcfda3ec31047079c0e2a1e1295f9`
- Focused summary SHA-256: `05e49f4c47b630732fe8084f379c8a3e7e1d6f9c7a514f6e95e91f9f12fe1897`
- All-generation summary SHA-256: `84fa63a2aeb70d2198a1dd6829724ff3847e7133d0816f84ddc7087bc7d3d7c6`

```bash
uv run --offline garboid-tournament \
  --games 6000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080699 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,surplus-v9,surplus-v10 \
  --output-dir artifacts/tournaments/surplus-v10-confirmation

uv run --offline garboid-tournament \
  --games 5200 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080699 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 0 \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,sdk-greedy-value-v1,surplus-v1,surplus-v2,surplus-v3,surplus-v4,surplus-v5,surplus-v6,surplus-v7,surplus-v8,surplus-v9,surplus-v10 \
  --output-dir artifacts/tournaments/surplus-v10-all-generations
```
