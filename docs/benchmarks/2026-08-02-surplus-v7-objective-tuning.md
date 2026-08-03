# Surplus v7 objective tuning

Date: 2026-08-02

## Decision

At this stage, release `surplus-v7` and advance the local `surplus` alias from
v5 to v7. Historical v1-v6 identities and behavior remain selectable. The
alias later advanced to v8; see the
[opponent-threat report](2026-08-02-surplus-v8-opponent-objective-threat.md).

V7 retains v5's finite-deck posterior, investment bidding, and public market
cap, but changes the value entering its first-price bid:

```text
(13/16 * expected resource value) + (3/8 * newly completed objective payout)
```

The combined value is still shaded by `(player_count - 1) / player_count` and
capped at the upper quartile of prior rival bids for that action plus one.

## Development search

All search receipts are ignored under `artifacts/tournaments/`. Candidate
identities were ephemeral; only the selected v7 policy is released.

| Stage | Seed | Games | Search | Result |
|---|---:|---:|---|---|
| Broad values | 2026080201 | 4,000 | Resource 7/8, 1, 9/8; objective 1/8-3/4 | 7/8 resource and 1/2 objective led |
| Refined values | 2026080202 | 6,000 | Resource 13/16, 7/8, 15/16; objective 3/8, 1/2, 5/8 | 13/16 resource led all objective weights |
| Market cap | 2026080203 | 6,000 | Objective 3/8, 1/2, 5/8; quantile 1/2, 3/4, 1 | Upper quartile retained |
| Objective ablation | 2026080204 | 8,000 | Objective 0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4 | Discounted objectives beat zero |

The ablation held the 13/16 resource multiplier and upper-quartile market cap
fixed. Objective weight zero rated 1484.35; 1/4 and 3/8 rated 1503.71 and
1503.55 respectively. Thus objective awareness itself contributed about 19
rating points in that field. The 3/8 weight was selected because it remained
near the lead across development seeds rather than winning only the final
ablation.

Development summary SHA-256 digests:

- broad values: `2f4e1aefe65c5cb21aa76fe0f3f09fbcf7e92711848cdc5f67464734edf18064`
- refined values: `1d40cd16453047364fb9c851a8d01400519d7dffa33c018f6d57d9e99973b8ad`
- market cap: `37f9f1604b870671dbb3484dc86be4462d85a3bcaaa8fbdc162dc7c21ea25f81`
- objective ablation: `4e77087c9a52631424ce7da2500915ac36d03e1294af809149f77db5901b8585`

## Confirmation

The confirmation used new root seed `2026080299`, all player counts 3-5, all
value charts A-E, batch size 64, and record-and-pass fault handling.

The 5,000-game all-generation field gave each bot 1,666-1,667 games:

| Bot | PL rating | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|
| fixed-objective-overlay-v2 | 1791.33 | 61.96 | 1,039 | 0 |
| fixed-bid-tuned-v1 | 1712.30 | 59.92 | 897 | 0 |
| fixed-bid | 1690.52 | 58.23 | 768 | 0 |
| surplus-v7 | 1639.90 | 51.02 | 493 | 0 |
| surplus-v5 | 1506.54 | 45.39 | 291 | 0 |
| surplus-v6 | 1497.47 | 44.56 | 253 | 0 |

V7 gained 133.36 rating points over v5 in this untouched-seed field and cut
the gap to `fixed-bid` to 50.62 points. It remains 151.43 points below the top
bot, so this is a strong improvement rather than a promotion claim.

A focused 6,000-game field gave each bot 3,999-4,001 games and ran 200
complete-game bootstrap resamples; all 200 fits converged:

| Bot | PL rating | 95% interval | Mean money | Faults |
|---|---:|---:|---:|---:|
| fixed-objective-overlay-v2 | 1703.45 | 1695.15-1714.27 | 59.31 | 0 |
| fixed-bid | 1539.33 | 1532.91-1547.65 | 49.71 | 0 |
| surplus-v7 | 1508.81 | 1502.62-1513.91 | 45.42 | 0 |
| fixed-bid-tuned-v1 | 1508.63 | 1501.43-1517.77 | 49.43 | 0 |
| surplus-v5 | 1375.28 | 1369.06-1381.65 | 39.49 | 0 |
| surplus-v6 | 1364.50 | 1357.55-1371.04 | 38.79 | 0 |

V7's and v5's intervals are disjoint. Ratings are field-dependent; the stable
cross-field conclusion is the approximately 133-point v7-over-v5 gain.

## Reproduction and provenance

- Source base commit: `ca35324d07474e6623a815fc8d85a1fd5ec044ec`
- Surplus policy SHA-256: `b526359dac192bb6f8ab3e44153f032898a6c5c6a000137e6229fd51b02839b7`
- All-generation summary SHA-256: `5ccb95d758b4d4656f80cd8a738e5055c837e9b68376c7732cda010c24e4b94a`
- Focused summary SHA-256: `625124bc2bbd992c0270ef6b0b480c2ad62cdecfb59a6593f227d5b7a0ff1df3`

```bash
uv run --offline garboid-tournament \
  --games 5000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080299 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 0 \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,sdk-greedy-value-v1,surplus-v1,surplus-v2,surplus-v3,surplus-v4,surplus-v5,surplus-v6,surplus-v7 \
  --output-dir artifacts/tournaments/surplus-v7-all-generations

uv run --offline garboid-tournament \
  --games 6000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080299 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --bots fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,surplus-v5,surplus-v6,surplus-v7 \
  --output-dir artifacts/tournaments/surplus-v7-confirmation
```
