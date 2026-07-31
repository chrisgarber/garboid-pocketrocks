# Heuristic v4 public phase-boundary evidence

Date: 2026-07-30

This development-only diagnostic describes how the released heuristic v3 bots
behave across equal thirds of the publicly known resource horizon. It was run
before constructing or searching any heuristic v4 candidate. No alternative
phase threshold was tested, and no held-out promotion corpus or promotion
result was consulted.

This is descriptive evidence, not a causal estimate of what a phase-specific
expert will improve. It establishes that the fixed thirds have useful coverage,
that they are meaningfully different from the older turn-number diagnostic,
and that the development search can report each expert separately.

## Fixed phase rule

For each public decision, let:

- `R` be the number of resources still available to bid on after the current
  decision opportunity; and
- `T` be the public total number of biddable resources.

The already-committed selector uses integer arithmetic:

```text
early   when 3 * R >= 2 * T
middle  when 3 * R >= T
late    otherwise
```

The canonical rulesets in this run produced `T = 15` for three- and
five-player games and `T = 14` for four-player games. Both totals therefore
select early for `R >= 10`, middle for `5 <= R <= 9`, and late for `R <= 4`.
The rule uses public resource counts only. It does not use a private hand,
deck order, opponent model, game seed, or hidden history.

## Reproduction

Repository revision:
`aed57ea8037f795edfa9b597bfd16a06b25eb0df`

Runtime: Python 3.14.6

```bash
uv run garboid-tournament \
  --games 3000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 12012 \
  --workers 8 \
  --batch-size 64 \
  --bootstrap-samples 0 \
  --bots aggressive-v3,balanced-v3,passive-v3,random,aggressive-v1,balanced-v1,passive-v1 \
  --decision-reports \
  --output-dir /tmp/garboid-v4-phase-boundaries-12012
```

Each of the 15 chart/player-count conditions received 200 games. Bootstrap
was disabled because this run measures descriptive decision coverage rather
than a promotion comparison.

Only the distilled, privacy-safe additive
[phase-boundary-slices.csv](tournaments/2026-07-30-heuristic-v3-phase-boundaries-development/phase-boundary-slices.csv)
is committed. Its SHA-256 is:

```text
4f8aa60edf31b28c746cb8004a4dd5468ee8ab1b26462550c914b2e3fa50d7ae
```

The retained 3,412 rows group only by bot identity, chart, player count,
legacy turn phase, derived resource expert, and exact public horizon. They sum
the source's additive decision, pass, selected-value, final-money,
normalized-finish, win, tie, and fault measures. Opponent lineups, seats, objective
states, action details, and every per-game dimension have been removed.

The high-dimensional source `decision-slices.csv`, raw decision traces,
per-game summaries, `summary.json`, `report.html`, ratings, and every other
per-game or presentation artifact remain in temporary storage and are not
published here. The fixed development seed is recorded for exact reruns, so
publishing those raw rows beside it would unnecessarily expose reconstructable
private game state.

## Reconciliation

The temporary summary was read only for its reconciliation counters:

| Counter | Total |
| --- | ---: |
| Games | 3,000 |
| Game seats | 12,000 |
| Decisions in game summaries | 220,603 |
| Decisions in distilled phase-boundary slices | 220,603 |
| Decisions in raw traces | 220,603 |

The three v3 bots contributed 94,937 decisions to the phase analysis:
31,646 from aggressive v3, 31,693 from balanced v3, and 31,598 from passive
v3. The committed slices attribute zero decisions to faulted game seats.

## Exact public horizons

Every public horizon observed for the v3 bots is listed below. Counts are
sums of `decision_count` across both supported decision kinds.

| Total (`T`) | Future (`R`) | Selected expert | v3 decisions |
| ---: | ---: | --- | ---: |
| 14 | 0 | late | 1,902 |
| 14 | 1 | late | 2,003 |
| 14 | 2 | late | 2,197 |
| 14 | 3 | late | 2,315 |
| 14 | 4 | late | 2,323 |
| 14 | 5 | middle | 2,377 |
| 14 | 6 | middle | 2,382 |
| 14 | 7 | middle | 2,297 |
| 14 | 8 | middle | 2,351 |
| 14 | 9 | middle | 2,076 |
| 14 | 10 | early | 2,209 |
| 14 | 11 | early | 1,937 |
| 14 | 12 | early | 1,939 |
| 14 | 13 | early | 1,526 |
| 14 | 14 | early | 859 |
| 15 | 0 | late | 3,593 |
| 15 | 1 | late | 3,926 |
| 15 | 2 | late | 4,037 |
| 15 | 3 | late | 4,139 |
| 15 | 4 | late | 4,307 |
| 15 | 5 | middle | 4,431 |
| 15 | 6 | middle | 4,595 |
| 15 | 7 | middle | 4,703 |
| 15 | 8 | middle | 4,543 |
| 15 | 9 | middle | 4,472 |
| 15 | 10 | early | 4,426 |
| 15 | 11 | early | 4,160 |
| 15 | 12 | early | 3,912 |
| 15 | 13 | early | 3,988 |
| 15 | 14 | early | 3,133 |
| 15 | 15 | early | 1,879 |

All 31 possible `(T, R)` pairs occurred. The 30,693 decisions with `T = 14`
and 64,244 decisions with `T = 15` sum to the 94,937 v3 decisions.

## Equal-third coverage

| Resource expert | v3 decisions | Share |
| --- | ---: | ---: |
| early | 29,968 | 31.57% |
| middle | 34,227 | 36.05% |
| late | 30,742 | 32.38% |
| **Total** | **94,937** | **100.00%** |

The thirds are close in decision volume without having been fitted to these
data. All 135 combinations of three v3 bots, five charts, three player counts,
and three resource thirds are nonempty.

| Bot | Resource expert | Covered chart/player cells | Minimum decisions in a cell | Maximum | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| aggressive-v3 | early | 15/15 | 520 | 868 | 9,869 |
| aggressive-v3 | middle | 15/15 | 573 | 961 | 11,441 |
| aggressive-v3 | late | 15/15 | 522 | 841 | 10,336 |
| balanced-v3 | early | 15/15 | 526 | 911 | 10,056 |
| balanced-v3 | middle | 15/15 | 600 | 950 | 11,516 |
| balanced-v3 | late | 15/15 | 505 | 804 | 10,121 |
| passive-v3 | early | 15/15 | 542 | 866 | 10,043 |
| passive-v3 | middle | 15/15 | 584 | 933 | 11,270 |
| passive-v3 | late | 15/15 | 485 | 863 | 10,285 |

The least-populated cell was passive v3, chart D, three players, late phase,
with 485 decisions. This is enough descriptive coverage to require every
development search and report to retain all three experts across every
supported chart and player count.

## Turn phase compared with resource phase

The existing `game_phase` diagnostic labels turns 1–5 early, 6–12 middle, and
13 onward late. It is intentionally different from the resource expert
selector. The cross-tabulation below is decision-weighted.

| Existing turn phase | Resource early | Resource middle | Resource late | Total |
| --- | ---: | ---: | ---: | ---: |
| early | 24,089 | 3,422 | 0 | 27,511 |
| middle | 5,877 | 28,389 | 11,117 | 45,383 |
| late | 2 | 2,416 | 19,625 | 22,043 |
| **Total** | **29,968** | **34,227** | **30,742** | **94,937** |

The diagonal contains most decisions, so both views track broad game
progress. The substantial off-diagonal counts show that turn number is not a
safe substitute for the exact public resource horizon. A turn can contain
different actions and a different number of biddable resources, while the v4
selector answers the narrower question: how many biddable resource cards
remain after this public decision?

## Descriptive v3 behavior by resource phase

`Pass rate` is `pass_count / decision_count` for `submitBid` rows.
`Mean selected bid` is `selected_value_sum / selected_value_count` for
non-pass `submitBid` selections. `Decision-weighted normalized finish` is
`eventual_normalized_finish_sum / decision_count` across both supported
decision kinds.

| Bot | Resource expert | Decisions | Bid decisions | Pass rate | Mean selected bid | Decision-weighted normalized finish |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| aggressive-v3 | early | 9,869 | 8,929 | 5.57% | 3.376 | 0.747 |
| aggressive-v3 | middle | 11,441 | 8,991 | 10.51% | 4.954 | 0.739 |
| aggressive-v3 | late | 10,336 | 9,027 | 13.84% | 4.523 | 0.718 |
| balanced-v3 | early | 10,056 | 9,031 | 18.82% | 4.240 | 0.737 |
| balanced-v3 | middle | 11,516 | 8,969 | 13.93% | 5.151 | 0.732 |
| balanced-v3 | late | 10,121 | 8,966 | 13.65% | 3.968 | 0.727 |
| passive-v3 | early | 10,043 | 9,019 | 0.37% | 3.086 | 0.769 |
| passive-v3 | middle | 11,270 | 8,966 | 5.91% | 4.748 | 0.752 |
| passive-v3 | late | 10,285 | 8,934 | 12.17% | 4.671 | 0.742 |

These differences justify exposing phase-specific coefficients to a
development search: the released v3 personalities already encounter distinct
decision distributions in each fixed third. They do **not** show that a
particular coefficient change causes better outcomes. Final money and finish
are repeated once per decision, so the last column weights game seats with
more decisions more heavily and can be affected by charts, player counts,
lineups, and game trajectories.

No threshold was selected for a favorable pass rate, bid, or finish statistic.
The equal-thirds rule was fixed before this run. V4 search may tune coefficients
inside those immutable thirds using development data, while any claim that a
frozen v4 candidate beats v3 remains reserved for the separate common held-out
promotion exam.
