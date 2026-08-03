# Surplus v12 objective reachability

Date: 2026-08-03

## Decision

Release `surplus-v12` and advance the local `surplus` alias from v11 to v12.
Historical v1-v11 identities and behavior remain selectable. This is a local
simulation release, not a remote promotion.

V12 stops treating all partial objective progress as equally achievable. It
removes the bot's private hand, public reveals, won resources, and the current
offer from the known six-per-suit deck. The remaining unknown cards are split
between rival hidden hands and future auctions. For each minimal route to an
objective, a multivariate hypergeometric dynamic program calculates the chance
that the future auction deck contains every still-required suit. Flexible
objectives use their most reachable route.

The frozen progress multiplier is:

```text
reachability factor = 1/2 + (1/2 * best-route completion probability)
```

This keeps half of v11's progress value as a robustness floor while using the
new probability for the other half. V12 retains all v11 loan, investment,
resource, market-price, objective-completion, and denial settings except that
it releases `5/8`, rather than `1/2`, of the normal cash reserve for a bundle
that completes an objective. That interaction lets it be more selective about
speculative progress without becoming too conservative on guaranteed claims.

## Rejected first hypothesis

The first v12 candidate replaced pooled investment prices with cash-capped,
per-opponent historical quantiles. It passed several development runs but
failed an untouched 6,000-game field at root seed `2026080899`: v11 scored
1545.94 rating and $49.86 mean money, while the candidate scored 1541.05 and
$49.14. The candidate increased investment value from $4.55 to $12.91 per game
but reduced item value from $49.48 to $42.72 and objective payout from $9.18 to
$7.07. That mechanism was removed rather than retuned on the held-out seed.

The failure does not disprove opponent modeling in general. It shows that an
investment-only price predictor can increase auction wins while destroying
portfolio value. The backlog retains a richer action-, phase-, cash-, and
win-probability-conditioned version as a future hypothesis.

## Development search

All reachability development used fresh fixed root seeds after the rejected
candidate. Runs covered players 3-5 and charts A-E with batch size 64, no
bootstrap, and zero faults. Raw receipts remain ignored under `artifacts/`;
this table is the durable selection record. Total reachability development
coverage was 51,000 games.

| Stage | Games | Root seed | Search | Selected evidence |
|---|---:|---:|---|---|
| Reachability and race | 6,000 | `2026081101` | Floors `0`, `1/4`, `1/2`, `3/4`; race discounts `0`, `1/4`, `1/2`, `1` | Floor `1/2`, race `0`: 1513.44 rating versus v11's 1493.88 |
| Objective weights | 6,000 | `2026081102` | Progress `1/16`-`1/4`, immediate value `1/4`-`1/2`, denial `0`-`1/16` | Retained v11 progress, immediate, and denial weights |
| Economic interactions | 6,000 | `2026081103` | Resource value, resource reserve, objective reserve release | `3/4` completion reserve release was the most promising interaction; larger resource reserve was inconsistent head to head |
| Finalists | 9,000 | `2026081104` | Reachability finalists plus liquidity interactions | `3/4` completion release led candidates at 1507.25 versus v11's 1505.51 |
| Confirmation | 12,000 | `2026081105` | Four reachability floors and `5/8` versus `3/4` completion release | All leading settings exceeded v11 on rating; floor `1/2`, release `5/8` scored 1508.14 versus 1501.48 |
| Final selection | 12,000 | `2026081106` | Three balanced finalists | Selected floor `1/2`, release `5/8`: 1515.84 versus 1509.96, $47.99 versus $47.58, and 51.44% direct score over 3,619 shared games |

Race discounts were consistently less robust than supply reachability, so the
released race coefficient is zero. Increasing objective-progress weight also
hurt: `3/16` and `1/4` both trailed the retained `1/8`. The selected policy is
therefore a small information improvement plus one interacting reserve change,
not a broad retuning.

## Held-out confirmation

After freezing the policy, confirmation used untouched root seed
`2026081199`, 12,000 games, players 3-5, charts A-E, batch size 64,
record-and-pass faults, decision diagnostics, and 200 complete-game bootstrap
resamples. Every bot appeared in 9,600 games, all decision counts reconciled,
and no bot faulted.

| Bot | PL rating | 95% interval | Mean money | Outright wins | Win rate | Faults |
|---|---:|---:|---:|---:|---:|---:|
| surplus-v12 | 1539.11 | 1534.0-1544.1 | 49.61 | 3,330 | 34.69% | 0 |
| surplus-v11 | 1533.04 | 1528.3-1538.0 | 49.23 | 3,210 | 33.44% | 0 |
| fixed-objective-overlay-v3 | 1502.48 | 1498.4-1506.6 | 46.93 | 2,201 | 22.93% | 0 |
| fixed-objective-overlay-v2 | 1498.61 | 1495.0-1502.9 | 46.06 | 1,883 | 19.61% | 0 |
| fixed-bid | 1426.76 | 1422.8-1430.8 | 41.98 | 895 | 9.32% | 0 |

V12 ranked first, gaining 6.07 rating points, $0.38 mean money, and 1.25
percentage points of outright win rate over v11. The bootstrap intervals
overlap. Across the 7,600 games containing both bots, v12's direct score was
50.55%; an approximate normal 95% interval is 49.44%-51.66%. This is a modest,
broadly favorable result, not a statistically decisive head-to-head win.

## Advanced diagnostics

| Measure per game | surplus-v11 | surplus-v12 | V12 delta |
|---|---:|---:|---:|
| Final score | 49.23 | 49.61 | +0.38 |
| Final cash | 5.51 | 5.63 | +0.13 |
| Investment value | 4.82 | 4.95 | +0.13 |
| Item value | 48.76 | 48.59 | -0.17 |
| Loan principal | 18.60 | 18.64 | +0.04 liability |
| Objective payout | 8.74 | 9.07 | +0.33 |
| Resources won | 4.499 | 4.474 | -0.026 |
| Resource spend | 27.56 | 27.40 | -0.16 |
| Objectives claimed | 1.180 | 1.240 | +0.060 |

The new model is more selective rather than simply more aggressive. Mean
resource bids fell from $6.12 to $6.04 and resource pass rate rose from 5.26%
to 6.46%. V12 won slightly fewer cards and gave up $0.17 of item value, but it
claimed about six more objectives per 100 games and gained $0.33 of objective
payout. Higher retained cash and investment value supplied the rest of the
score improvement.

V12 improved mean money in all five chart aggregates and all three
player-count aggregates. It improved 11 of the 15 individual chart/player
cells. The largest gains were 5-player chart D (+$1.40), 3-player chart E
(+$1.22), and 5-player chart E (+$1.14). The largest regression was 5-player
chart C (-$0.32); the other three negative cells were between -$0.01 and
-$0.15. The result is broad but not uniform.

The interactive advanced report is generated locally at
`artifacts/tournaments/surplus-v12-objective-reachability-held-out/insights.html`.

## Reproduction and provenance

- Development source base commit: `47520aaecd6f910e81a3675ee6ce9cf0fc34e780`
- Release base commit: `47520aaecd6f910e81a3675ee6ce9cf0fc34e780`
- Surplus source SHA-256: `d6ec858e91d01fab1a62fa4ecfb247fc780fc69bee7196c91415411792f83760`
- Held-out summary SHA-256: `86f8f43ed6d1b98db3d42abd8d6b1d97f532952261bc17102f3a94624a9d3866`

```bash
uv run garboid-tournament \
  --games 12000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026081199 \
  --workers 1 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots fixed-bid,fixed-objective-overlay-v2,fixed-objective-overlay-v3,surplus-v11,surplus-v12 \
  --output-dir artifacts/tournaments/surplus-v12-objective-reachability-held-out

uv run garboid-visualize \
  artifacts/tournaments/surplus-v12-objective-reachability-held-out
```
