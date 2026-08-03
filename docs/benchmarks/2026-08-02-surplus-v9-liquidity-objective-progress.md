# Surplus v9 liquidity and objective progress

Date: 2026-08-02

## Decision

Release `surplus-v9` and advance the local `surplus` alias from v8 to v9.
Historical v1-v8 identities and behavior remain selectable. This is a local
simulation release, not a remote promotion.

V9 addresses v8's measured liquidity collapse while retaining its finite-deck
posterior. It adds four public-state behaviors:

- reserve cash in proportion to the number of resource cards still available;
- reserve more cash before locking money in an investment;
- buy loan liquidity only while future resource demand exceeds current cash;
- value partial progress toward unclaimed objectives without duplicating an
  immediately completed objective's payout.

The frozen value entering a resource bid is:

```text
(3/4 * expected resource value)
+ (3/8 * own newly completed objective payout)
+ (1/8 * partial objective progress value)
+ (1/32 * largest rival newly completed objective payout)
```

The combined value retains player-count shading and the upper-quartile public
market cap. Resource bids preserve `3/4` cash per future resource card;
investment bids preserve `3/2`. A completed objective may release half its
payout from the reserve. Loan bids activate below five times the remaining
resource count and are capped at 30% of principal, producing bids of 3 for
`Loan10` and 6 for `Loan20` before the market cap.

## Development search

Every candidate was compared with `fixed-bid`, `fixed-bid-tuned-v1`,
`fixed-objective-overlay-v2`, and `surplus-v8` across player counts 3-5 and
charts A-E. Receipts remain ignored under `artifacts/tournaments/`.

| Stage | Seed | Games per candidate | Search | Result |
|---|---:|---:|---|---|
| Initial trace | 2026080501 | 1,800 | First liquidity/progress policy | Resources and objectives rose, but investment reserve was too strict |
| Coarse liquidity | 2026080502 | 900 | Resource reserve, investment reserve, loan trigger | Stronger resource reserves and earlier loans reduced the v8 gap |
| Liquidity neighborhood | 2026080503 | 1,200 | Reserve and loan fee combinations | Loan clearing price dominated the interaction |
| Loan pricing | 2026080504 | 1,200 | Fees 30%-50%, triggers, reserves | 30% fee and earlier trigger led; higher fees hurt |
| Value retune | 2026080505 | 1,200 | Resource, immediate objective, progress, investment | Smaller progress and immediate-objective weights led |
| Combined weights | 2026080506 | 1,800 | Independent winners composed together | Restoring resource value to 3/4 was necessary |
| Finalists | 2026080507 | 2,400 | Progress, investment, objective, trigger, reserve | Stronger resource reserve led |
| Reserve neighborhood | 2026080508 | 2,400 | Reserve 1/2-1 and loan triggers 4-5 | Reserve 3/4 and trigger 5 were frozen |

The important negative result was that the first apparently sensible reserve
policy rated below v8. It successfully increased resources and objectives but
gave up too much guaranteed investment bonus. Strength appeared only after
loan bids were high enough to clear the field and every surrounding valuation
weight was retuned.

## Held-out confirmation

Confirmation used untouched root seed `2026080599`, all player counts 3-5,
all value charts A-E, batch size 64, and record-and-pass fault handling. No
coefficient was changed after reading these results.

The 6,000-game focused field gave each bot 4,800 games and ran 200
complete-game bootstrap resamples; all fits converged:

| Bot | PL rating | 95% interval | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|---:|
| surplus-v9 | 1693.45 | 1686.1-1700.6 | 59.85 | 3,002 | 0 |
| fixed-objective-overlay-v2 | 1524.50 | 1517.8-1531.5 | 49.08 | 1,210 | 0 |
| surplus-v8 | 1487.32 | 1482.9-1492.1 | 46.12 | 557 | 0 |
| fixed-bid | 1410.62 | 1405.0-1417.2 | 42.56 | 510 | 0 |
| fixed-bid-tuned-v1 | 1384.11 | 1377.2-1391.3 | 41.65 | 546 | 0 |

V9 gained 206.13 rating points and 13.73 mean-money points over v8. Its
interval is disjoint from every comparison bot.

Decision traces show that the gain matches the intended mechanism:

| Measure per game | surplus-v9 | surplus-v8 |
|---|---:|---:|
| Final cash | 7.05 | 2.40 |
| Resource cards | 5.40 | 2.23 |
| Item value | 59.91 | 25.37 |
| Item value minus resource spend | 26.19 | 11.63 |
| Objective payout | 13.42 | 2.20 |
| Objectives claimed | 1.78 | 0.38 |
| Investment bonus | 5.48 | 8.12 |
| Loan principal | 31.40 | 0.00 |
| Loan fees | 9.41 | 0.00 |

The 5,200-game all-generation field gave each bot 1,485-1,486 games:

| Bot | PL rating | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|
| surplus-v9 | 1817.29 | 61.98 | 952 | 0 |
| fixed-objective-overlay-v2 | 1738.19 | 59.03 | 785 | 0 |
| fixed-bid | 1658.44 | 55.60 | 620 | 0 |
| fixed-bid-tuned-v1 | 1649.65 | 55.92 | 656 | 0 |
| surplus-v8 | 1633.98 | 51.34 | 387 | 0 |
| surplus-v7 | 1609.97 | 50.12 | 376 | 0 |

V9 led the former top bot by 79.10 points and v8 by 183.31 points in this
broader field. Ratings remain field-dependent; the stable release claim is
that v9 beat v8 on the untouched seed in both fields.

## Reproduction and provenance

- Source base commit: `ca35324d07474e6623a815fc8d85a1fd5ec044ec`
- Surplus policy SHA-256: `85d6162c4e42e544641c0c8e8a3a8c58222c96eb62548b88349759f0767c81a8`
- Focused summary SHA-256: `3b194831b1fbf3bfdbbfdb75eaf6e3f1c809bb70577cd00e9744b1ee8b5e0c57`
- All-generation summary SHA-256: `c6417780eeab276d74513b7a51f09ea48da25dee7b39edea642045351ed160a5`

```bash
uv run --offline garboid-tournament \
  --games 6000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080599 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,surplus-v8,surplus-v9 \
  --output-dir artifacts/tournaments/surplus-v9-confirmation

uv run --offline garboid-tournament \
  --games 5200 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080599 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 0 \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,sdk-greedy-value-v1,surplus-v1,surplus-v2,surplus-v3,surplus-v4,surplus-v5,surplus-v6,surplus-v7,surplus-v8,surplus-v9 \
  --output-dir artifacts/tournaments/surplus-v9-all-generations
```
