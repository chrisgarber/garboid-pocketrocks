# Surplus v8 opponent objective threat

Date: 2026-08-02

## Decision

Release `surplus-v8` and advance the local `surplus` alias from v7 to v8.
Historical v1-v7 identities and behavior remain selectable.

V8 adds one deployable public input: before bidding on a resource bundle, it
checks each rival's publicly won resources and calculates the largest
unclaimed objective payout that rival would complete by winning the bundle.
The final value entering the first-price bid is:

```text
(3/4 * expected resource value)
+ (3/8 * own newly completed objective payout)
+ (1/32 * largest rival newly completed objective payout)
```

The combined value retains v7's player-count shading, investment behavior,
finite-deck posterior, and upper-quartile public market cap.

## Development search

The search deliberately retuned surrounding coefficients after adding the new
information. All candidate receipts remain ignored under
`artifacts/tournaments/`; only v8's frozen policy is released.

| Stage | Seed | Games | Search | Result |
|---|---:|---:|---|---|
| Coarse threat | 2026080301 | 6,500 | Denial 0-1 | Moderate and full denial values hurt monotonically |
| Small threat | 2026080302 | 6,000 | Denial 0, 1/32, 1/16, 3/32, 1/8 | 1/16 was approximately tied with zero |
| Value retune | 2026080303 | 6,000 | Resource 3/4, 13/16, 7/8; own objective 1/4, 3/8, 1/2 | Resource 3/4 led every objective weight |
| Threat ablation | 2026080304 | 6,000 | Denial 0, 1/32, 1/16, 3/32, 1/8 at resource 3/4 | Denial 1/32 led zero by 4.37 rating points |
| Market retune | 2026080305 | 6,000 | Own objective 1/4, 3/8, 1/2; quantile 1/2, 3/4, 1 | Existing 3/8 and 3/4 setting tied the lead |

The new information had a small useful coefficient. At the final surrounding
weights, denial 1/32 rated 1514.69 versus 1510.32 for denial zero. Larger
denial values were inconsistent or harmful. The more important interaction was
the resource multiplier: after adding denial information, retuning it from
13/16 to 3/4 created most of v8's improvement. This is why v8 freezes both the
new feature and the surrounding coefficient change.

Development summary SHA-256 digests:

- coarse threat: `4d47b028141e7b93a4428df7732264e88273cfa677c50378157936a12ee34068`
- small threat: `36a9a445133b2b4a281179c6843ac4b4a7a35fe37a347948db30a8a94dc46435`
- value retune: `cf43f689dac57b266f6e0495e306a505d99ad9e18a67aa32f7f62f3f5daa4f9c`
- threat ablation: `1f0f6a6ff1bbf7d463258383b3e47a5b1ea72cf1169d03099e82885fd129a012`
- market retune: `a5e82e0ab2bf95dd62ab4dfa81d119f4e5097efb939b921fc610d0454a3e026b`

## Confirmation

Confirmation used untouched root seed `2026080399`, all player counts 3-5,
all value charts A-E, batch size 64, and record-and-pass fault handling.

The 5,200-game all-generation field gave each bot 1,599-1,601 games:

| Bot | PL rating | Mean money | Outright wins | Faults |
|---|---:|---:|---:|---:|
| fixed-objective-overlay-v2 | 1785.05 | 61.87 | 1,010 | 0 |
| fixed-bid-tuned-v1 | 1698.11 | 59.10 | 828 | 0 |
| fixed-bid | 1685.40 | 57.12 | 767 | 0 |
| surplus-v8 | 1653.13 | 52.66 | 492 | 0 |
| surplus-v7 | 1616.61 | 50.18 | 417 | 0 |
| surplus-v5 | 1510.08 | 45.37 | 287 | 0 |

V8 gained 36.52 rating points and 2.48 mean-money points over v7. It cut the
gap to `fixed-bid` to 32.26 points and remained 131.91 points below the top
bot. This is an improvement claim, not a promotion claim.

A focused 6,000-game field gave every bot 4,000 games and ran 200
complete-game bootstrap resamples; all 200 fits converged:

| Bot | PL rating | 95% interval | Mean money | Faults |
|---|---:|---:|---:|---:|
| fixed-objective-overlay-v2 | 1668.70 | 1661.47-1676.89 | 58.21 | 0 |
| surplus-v8 | 1513.27 | 1507.65-1518.72 | 46.72 | 0 |
| fixed-bid | 1500.74 | 1493.71-1508.43 | 48.24 | 0 |
| fixed-bid-tuned-v1 | 1485.45 | 1477.12-1493.60 | 48.25 | 0 |
| surplus-v7 | 1471.95 | 1465.21-1477.02 | 45.05 | 0 |
| surplus-v5 | 1359.89 | 1353.12-1367.70 | 39.58 | 0 |

V8's and v7's intervals are disjoint. Ratings remain field-dependent; the
stable conclusion is that v8 beat v7 on the untouched seed in both fields.

## Reproduction and provenance

- Source base commit: `ca35324d07474e6623a815fc8d85a1fd5ec044ec`
- Surplus policy SHA-256: `8a53fc38f0be51fab8b2a69283043bed50ae5637e856d5a228f19d5dc40fdcbb`
- All-generation summary SHA-256: `9b2f955e03c6ca298037775111d96ae892f528083db27e08fed1a7b6dd26403e`
- Focused summary SHA-256: `27a37a0d08482f12d2ebb017139dd0c140ac12c57782fa1c328ee2f8cac2747b`

```bash
uv run --offline garboid-tournament \
  --games 5200 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080399 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 0 \
  --bots random,fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,sdk-greedy-value-v1,surplus-v1,surplus-v2,surplus-v3,surplus-v4,surplus-v5,surplus-v6,surplus-v7,surplus-v8 \
  --output-dir artifacts/tournaments/surplus-v8-all-generations

uv run --offline garboid-tournament \
  --games 6000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026080399 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --bots fixed-bid,fixed-bid-tuned-v1,fixed-objective-overlay-v2,surplus-v5,surplus-v7,surplus-v8 \
  --output-dir artifacts/tournaments/surplus-v8-confirmation
```
