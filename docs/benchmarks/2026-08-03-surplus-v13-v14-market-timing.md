# Surplus v13-v14 market timing

Date: 2026-08-03

## Decision

Preserve `surplus-v13` and `surplus-v14` as immutable local simulation
identities and advance the unversioned `surplus` alias from v12 to v14.
V13 adds an expected-profit bid curve for investments. V14 retains that curve
and shifts resource-auction aggression from the beginning of the game toward
the end. This is a local simulation release, not a remote promotion.

V13 estimates the probability that each legal investment bid beats the
same-action rival maximum. A four-observation uniform prior covers clearing
prices from zero through twice the investment payout; public resolved auctions
then update that distribution. It chooses the bid maximizing:

```text
P(win at bid) * (investment payout - bid)
```

The old `2/5` investment liquidity reserve was retuned to zero because the
selected bids are already below the guaranteed payout and explicitly price
their win probability. The model applies only to investments: applying the
same curve to resource auctions was decisively harmful.

V14 addresses the remaining timing error. Let `r` be the fraction of public
resources still in the auction deck after the current offer. The frozen
resource-bid multiplier is:

```text
1 + (5/16 * (1 - 2r))
```

It shades early bids down by as much as `5/16`, leaves the middle of the deck
near v13, and raises late bids by as much as `5/16`. The multiplier is applied
after valuation and the learned market cap but before the existing liquidity
cap. V12 and v13 behavior remains unchanged and selectable.

## Development search

All development used fixed roots disjoint from the final confirmations,
players 3-5, charts A-E, batch size 64, and zero faults. Raw candidate receipts
remain ignored under `artifacts/`.

### V13

V13 development covered 72,500 games across roots `2026081301` through
`2026081307`.

| Stage | Games | Finding |
|---|---:|---|
| Hard early cash reserve | 6,000 | Rejected: every tested reserve from `1/8` through `1/2` lost to v12. |
| Expected-surplus prior | 6,000 | Full resource-and-investment application lost; prior weight `4` was the best investment candidate. |
| Strategic value inflation | 6,000 | Rejected: multipliers `3/2` through `3` overpaid. |
| Action-scope and reserve ablation | 7,500 | Investment-only, reserve `0` gained 37.70 rating and scored 56.86% directly against v12 in this development field. |
| Prior refinement | 7,500 | Prior `4` and no value inflation remained the balanced choice. |
| Value refinement | 7,500 | Higher values were inconsistent across rating and direct finish. |
| Stable-field finalists | 32,000 | Four separate 8,000-game fields selected the uninflated prior-4 policy: +13.88 rating and +$0.96 versus v12, but 51.22% direct score with an interval spanning 50%. |

The first 15,000-game confirmation at root `2026081399` showed that V13 was
an economic improvement but not a strength promotion: 1596.48 rating versus
v12's 1596.76, $52.29 versus $52.22, and 50.04% direct score. Investment net
value rose from $2.00 to $4.27 per game, but item value fell $2.76 and objective
payout fell $0.90. That result was not used to retune v13.

### V14

V14 development covered 87,500 games across roots `2026081401` through
`2026081407`.

| Stage | Games | Finding |
|---|---:|---|
| Investment reserve timing | 15,000 | Rejected after a stable-field check: the best crowded-field reserve lost 11.60 rating one-at-a-time. |
| Reserve finalists | 20,000 | Both `1/8` and `7/64` lost to v13. |
| Resource-reserve retuning | 7,500 | Rejected: lowering v13's resource reserve did not recover the missed item value. |
| Investment cash option cost | 7,500 | Rejected: investment-only phase costs did not improve v13. |
| Resource phase slope | 7,500 | A `1/4` slope gained 17.56 rating and $0.93 over v13 in the grid. |
| Stable slope finalists | 30,000 | Slopes `3/16`, `1/4`, and `5/16` all beat v13; `5/16` led at +24.11 rating, +$0.74, and 53.07% direct score (51.04%-55.10%) over 2,264 shared games. |

The `5/16` policy was frozen before either held-out v14 run. No coefficient
was changed after inspecting the confirmations.

## Final elite confirmation

The authoritative confirmation used untouched root `2026081599`, 15,000
games, players 3-5, charts A-E, batch size 64, record-and-pass faults, decision
diagnostics, and 200 complete-game bootstrap resamples. It integrated current
main (including Monte the Bookie v2), the surplus v11-v14 stack, and the
released PPO v2 checkpoint. All 1,103,558 decisions reconciled and every bot
finished with zero faults.

| Bot | Rank | PL rating | 95% interval | Appearances | Win rate | Mean money | Faults |
|---|---:|---:|---:|---:|---:|---:|---:|
| monte-the-bookie-v2 | 1/9 | 1569.91 | 1564.1-1575.1 | 6,665 | 34.5% | 50.08 | 0 |
| fixed-objective-overlay-v3 | 2/9 | 1517.60 | 1512.2-1521.4 | 6,666 | 21.7% | 46.72 | 0 |
| surplus-v14 | 3/9 | 1510.27 | 1505.0-1515.3 | 6,667 | 24.8% | 47.09 | 0 |
| surplus-v13 | 4/9 | 1499.61 | 1494.8-1505.2 | 6,667 | 26.2% | 46.98 | 0 |
| surplus-v12 | 5/9 | 1498.24 | 1492.1-1503.0 | 6,667 | 26.2% | 46.86 | 0 |

V14 gained 10.66 rating and $0.11 mean money over v13, and 12.03 rating and
$0.23 over v12. V14 and v13's bootstrap intervals narrowly overlap. Across
2,640 shared games, v14's direct score against v13 was 51.29% with an
approximate normal 95% interval of 49.41%-53.17%, so the predecessor comparison
is favorable but not statistically decisive. Against v12, v14 scored 53.41%
over 2,639 shared games (51.53%-55.29%); their bootstrap rating intervals do
not overlap.

## Wide-field robustness check

A post-release robustness run used fresh root `2026081699`, 30,000 games,
players 3-5, charts A-E, 200 bootstrap resamples, and decision diagnostics.
The 20-bot field added all three overlay generations, three fixed-bid variants,
three neural generations, three v3 personality bots, and the SDK greedy bot to
the elite field. Every bot appeared in 6,000 games, all 2,201,551 decisions
reconciled, and no bot faulted.

| Bot | Rank | PL rating | 95% interval | Mean money | Win rate | Faults |
|---|---:|---:|---:|---:|---:|---:|
| monte-the-bookie-v2 | 1/20 | 1681.46 | 1674.7-1688.6 | 57.50 | 47.9% | 0 |
| monte-the-bookie-v1 | 2/20 | 1615.87 | 1608.5-1623.3 | 53.70 | 38.1% | 0 |
| surplus-v11 | 3/20 | 1606.40 | 1599.6-1613.3 | 53.46 | 38.4% | 0 |
| surplus-v12 | 4/20 | 1602.53 | 1594.9-1610.2 | 53.33 | 37.3% | 0 |
| surplus-v14 | 5/20 | 1600.71 | 1594.8-1607.8 | 52.88 | 34.9% | 0 |
| surplus-v13 | 6/20 | 1598.83 | 1593.2-1605.8 | 52.95 | 37.7% | 0 |
| fixed-objective-overlay-v3 | 7/20 | 1583.39 | 1577.5-1588.6 | 51.78 | 30.8% | 0 |

This confirms the user's earlier observation: v12 clearly outranked overlay v3
in a diverse field. Their bootstrap rating intervals do not overlap. Across
999 games containing both bots, v12's direct score was 56.86% with an
approximate normal 95% interval of 53.83%-59.89%; its mean-money advantage in
those shared games was $3.35 ($2.30-$4.40). V12 scored above 50% in 13 of 15
chart/player-count cells.

The reversal from the nine-bot elite field is a multiplayer interaction, not a
contradiction in arithmetic. In that elite field v12's direct score against
overlay v3 was 48.94% (47.06%-50.82%) with a statistically unresolved $0.24
money edge. Adding diverse co-opponents changed auction prices, objective
competition, and which strategies benefited from the other seats.

In the wide field, v12 earned $8.56 more item value and $3.78 more objective
payout per game than overlay v3, while overlay earned $3.97 more investment
value and carried $6.41 less loan liability. V12 won 4.739 resource cards and
claimed 1.350 objectives per game, compared with overlay's 4.022 cards and
0.884 objectives. The resource/objective advantage outweighed overlay's
investment and debt advantage by $1.55 in marginal mean score and $3.35 when
the bots shared a game.

This run does not establish v12 as stronger than v14: their point estimates
differ by only 1.82 rating and their bootstrap intervals substantially overlap.
It does establish that field composition matters and that overlay v3 should
not be described as generally ahead of v12.

- Wide-field summary SHA-256:
  `81f508c1ad5b9d2f5c4aa092d0544c9f03c48a26242a7c4af53ede3a6cfd23db`
- Machine-readable evidence:
  [summary.json](tournaments/2026-08-03-surplus-v12-overlay-v3-wide-field/summary.json)

```bash
uv run --extra neural garboid-tournament \
  --games 30000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026081699 \
  --workers 8 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots monte-the-bookie-v2,monte-the-bookie-v1,surplus-v14,surplus-v13,surplus-v12,surplus-v11,fixed-objective-overlay-v3,fixed-objective-overlay-v2,fixed-objective-overlay-v1,fixed-bid-tuned-v1,fixed-bid-diverse-v1,fixed-bid-tuned-normal-v1,vector_ppo_large_v2_g1750k,vector_ppo_large_v1_g350k,vector_ppo_small_v1_g1500,aggressive-v3,balanced-v3,passive-v3,sdk-greedy-value-v1,fixed-bid \
  --output-dir artifacts/tournaments/surplus-v12-overlay-v3-wide-field
```

## Advanced diagnostics

| Measure per game | surplus-v12 | surplus-v13 | surplus-v14 | Monte v2 |
|---|---:|---:|---:|---:|
| Final score | 46.86 | 46.98 | 47.09 | 50.08 |
| Final cash | 5.94 | 6.49 | 5.03 | 4.97 |
| Gross investment value | 3.43 | 4.01 | 4.45 | 6.99 |
| Investment net value | 1.82 | 2.53 | 2.82 | 4.10 |
| Item value | 41.40 | 41.00 | 42.92 | 44.13 |
| Loan liability | 11.15 | 11.54 | 10.79 | 11.21 |
| Objective payout | 7.23 | 7.02 | 5.48 | 5.21 |

V14 made the intended timing change. Relative to v13, early resource spend
fell from $10.23 to $3.83 per game, while midgame spend rose from $6.60 to
$10.03 and late spend rose from $6.36 to $10.19. Resource wins moved from
1.207/0.812/0.818 per game in the early/mid/late thirds to
0.451/1.122/1.136. Monte v2 is still more extreme at
$1.40/$8.46/$12.78, leaving further timing upside.

The shift recovered $1.91 of item value over v13, added $0.44 of gross
investment value, reduced loan liability by $0.75, and consumed $1.45 of
otherwise stranded cash. It also gave back $1.54 of objective payout. V14's
overall gain is therefore real but narrow: better auction timing outweighed
fewer objective claims. The next iteration should improve late bundle choice
or objective-conditioned timing rather than increasing the global slope.

The interactive report is generated locally at
`artifacts/tournaments/surplus-v14-latest-elite-held-out/insights.html`.

## Reproduction and provenance

- Release base commit: `f5f1de372892d9e96dbe51f338ee7c6c90860b35`
- Integration refs: main `7272327`, PPO stack `31ae825`, surplus v13
  `cff2337`, surplus v14 `58912ca`
- Surplus source SHA-256:
  `9fcbd94c18f6d00babbf832b09fdd085c91c5ae4591d44e040d9b9cb43f9eb8e`
- Held-out summary SHA-256:
  `e2bb2814e432bf76e06e6a86e8ef22fdf5927dc9cbe3babd1218e2e96c915815`
- Machine-readable evidence:
  [summary.json](tournaments/2026-08-03-surplus-v14-latest-elite-held-out/summary.json)
  is committed as the minimal aggregate PR evidence; per-game diagnostics and
  decision traces remain ignored under `artifacts/`.

```bash
uv run --extra neural garboid-tournament \
  --games 15000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 2026081599 \
  --workers 8 \
  --batch-size 64 \
  --bootstrap-samples 200 \
  --decision-reports \
  --bots surplus-v14,surplus-v13,surplus-v12,monte-the-bookie-v2,fixed-objective-overlay-v3,surplus-v11,monte-the-bookie-v1,vector_ppo_large_v2_g1750k,fixed-bid-tuned-v1 \
  --output-dir artifacts/tournaments/surplus-v14-latest-elite-held-out

uv run garboid-visualize \
  artifacts/tournaments/surplus-v14-latest-elite-held-out
```
