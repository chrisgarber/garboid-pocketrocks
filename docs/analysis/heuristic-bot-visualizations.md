# Heuristic Bot Simulation Visualizations

This runbook records the Monte Carlo datasets and chart recipes used to analyze
the aggressive, balanced, and passive heuristic bots on July 29, 2026. It is
the canonical reference for recreating those visualizations after bot or engine
changes.

The original analysis produced three visualization suites:

1. `heuristic-bot-monte-carlo.html`
2. `loan-auction-winning-bids.html`
3. `turn-asymmetry-value-charts.html`

Each suite was regenerated after a bot fix. The three `*-rerun.html` files were
byte-for-byte identical to their originals. Generated HTML and result JSON are
not committed because they become stale as the bots evolve. The analysis
programs and the transformations below are the reproducible source.

## Historical simulation contract

All three suites used:

| Setting | Value |
| --- | --- |
| Games | 100,000 |
| Players | 3 |
| Lineup | aggressive, balanced, passive |
| Rulesets | live value charts A, B, C, D, and E |
| Ruleset sampling | equal-weight deterministic sampling |
| Root seed | `20260729` |
| Workers | 16 |
| Trend source bins | 1,000 games |
| Displayed trend blocks | 5,000 games |

The historical charts used the unversioned heuristic aliases at the checked-out
commit. Always record the commit before a new run:

```bash
git rev-parse HEAD
git status --short
```

A clean tree and a recorded commit are part of the result provenance. If
explicit bot generations are available, record whether the run uses versioned
brains or the moving unversioned aliases. Results from different bot
generations must not be combined into one trend.

## Generate the datasets

Install the locked environment, create a local output directory, and run:

```bash
uv sync --locked
mkdir -p analysis-output

PYTHONPATH=src uv run python \
  scripts/analysis/heuristic_monte_carlo.py \
  > analysis-output/heuristic-monte-carlo.json

PYTHONPATH=src uv run python \
  scripts/analysis/auction_turn_asymmetry.py \
  > analysis-output/auction-turn-asymmetry.json

PYTHONPATH=src uv run python \
  scripts/analysis/endgame_resources_cash.py \
  > analysis-output/endgame-resources-cash.json

PYTHONPATH=src uv run python \
  scripts/analysis/late_resource_auctions.py \
  > analysis-output/late-resource-auctions.json
```

The four programs intentionally carry the historical constants at the top of
each file. Change `GAMES` or `ROOT_SEED` only for a new analysis, and label the
result accordingly. Worker count may change without changing deterministic
results.

Before charting, validate the JSON and retain a checksum with the recorded
commit:

```bash
jq empty analysis-output/*.json
shasum -a 256 analysis-output/*.json
```

## Shared definitions

These definitions are important because changing any of them changes the
meaning of the charts:

- An **outright win** has exactly one rank-one player. First-place ties are
  reported separately and are excluded from outright win rate.
- A **winning score** is the greatest final money in a game, including tied
  first-place scores.
- A **winning bid** is the resolved auction payment. A priority win for `$0`
  is included in means, quantiles, and counts.
- A **submitted bid** is each bot's decision on a bidding turn. Passing is
  included as a submitted bid of `$0`.
- A **turn** is the one-based action-deck turn
  (`turn_index + 1`), not the ordinal occurrence of a particular action.
- **Hidden matches** count unrevealed cards in the acting bot's hand whose suit
  is present in the offered resource lot. Counts of three or more use one `3+`
  bucket.
- **Private-information premium per resource** is:

  ```text
  (private expected lot value - public-observer expected lot value)
  / number of offered resources
  ```

  The public observer uses public reveals, publicly won cards, and the current
  offer, but not the acting bot's hand.
- **Terminal liquid cash** is cash before loan principal is repaid.
- **Terminal cash after debt** is liquid cash minus outstanding loan
  principal.
- A **cash-zero request** is a bid request where the acting bot has no liquid
  cash.
- A **hard-constrained request** has `legal_max_amount == 0`. This is the
  closest measure of "could not bid"; cash-zero requests are also reported
  because the two conditions answer slightly different questions.

## Suite 1: Tournament outcomes

Source: `analysis-output/heuristic-monte-carlo.json`

### Summary cards and detail table

For each bot, display:

- outright win rate: `bots.<bot>.win_rate`;
- mean final score: `bots.<bot>.score.mean`;
- mean rank: `bots.<bot>.mean_rank`;
- pass rate: `bots.<bot>.pass_rate`;
- mean nonzero bid: `bots.<bot>.mean_nonzero_bid`;
- objectives per 100 games: `bots.<bot>.objectives_per_100_games`;
- resource cards per game: `bots.<bot>.resource_cards_per_game`;
- faults: `bots.<bot>.faults`.

The full table also includes games, outright wins, first-place ties, rank
counts, and the final-score distribution.

### Cumulative outright win rate

- X axis: games completed, from 5,000 through 100,000.
- Y axis: cumulative outright win percentage.
- Start with `trends`, which contains 1,000-game bins.
- Select every fifth bin's `through_games` and
  `cumulative_win_rates.<bot>`.
- Draw one line per bot.

This chart answers whether the relative strength estimate has stabilized.

### Winning score by 5,000-game block

- Group each five consecutive `trends` entries.
- Average their `mean_winning_score` values. Each source bin has the same
  number of games, so the arithmetic mean is the weighted mean.
- X axis: the ending game number for each 5,000-game block.
- Y axis: mean winning score.

Do not plot this as a cumulative mean; the purpose is to expose drift between
blocks.

### Outright win rate by value chart

- X axis: charts A through E.
- Y axis: outright win percentage.
- Series: `per_chart.live-<chart>.bots.<bot>.win_rate`.

Use a shared scale for all bots. This shows whether a strategy's strength is
specific to one value chart.

### Final-score distribution by bot

Use the shared score axis and plot:

- whiskers: `p05` to `p95`;
- box: `p25` to `p75`;
- center mark: `median`;
- optional mean mark: `mean`.

Source: `bots.<bot>.score`.

### Action wins per 100 games

- Categories: Auction 1, Auction 2, Loan $10, Loan $20, Invest $5, Invest $10.
- Series: `bots.<bot>.action_wins_per_100_games.<action>`.
- Use grouped marks on one shared scale.

This is an action acquisition frequency, not an average price.

## Suite 2: Loan and action winning bids

Source: `analysis-output/auction-turn-asymmetry.json`

### Average winning bid by action and bot

- Categories: the six action types.
- Series: aggressive, balanced, and passive.
- Value:
  `all_action_winning_bids.<bot>.<action>.mean`.
- Include `$0` priority wins.

Use a shared-scale dot plot so Auction 1, Auction 2, loans, and investments can
be compared without six independent axes.

### Share of loan auctions won

For Loan $10 and Loan $20:

1. read `all_action_winning_bids.<bot>.<loan>.count`;
2. sum counts across bots for the denominator;
3. divide each bot count by that denominator;
4. render one 100% stacked bar per loan type.

This chart measures who wins the loans, not how often the loan action appears.

### Loan winning-bid distributions

For each bot and loan type, draw:

- whiskers: `p10` to `p90`;
- box: `p25` to `p75`;
- center mark: `median`;
- optional mark: `mean`;
- tooltip or table value: `free_rate`.

Source: `all_action_winning_bids.<bot>.<loan>`.

### Loan winning bid by action-deck turn

- X axis: one-based action-deck turn.
- Y axis: mean winning bid.
- Facets: Loan $10 and Loan $20.
- Series:
  `loan_winning_by_turn.<loan>.<bot>[].mean`.
- Omit points whose `count` is zero; do not convert missing observations to
  `$0`.

Keep this distinct from submitted-bid charts: each point includes only auctions
won by that bot.

### Loan detail table

For both loan types and all bots, include `count`, `mean`, `mean_paid`,
`free_rate`, `p10`, `p25`, `median`, `p75`, `p90`, and `max`.

## Suite 3: Turn, information, and value-chart effects

Source: `analysis-output/auction-turn-asymmetry.json`

### Resource-auction bids by chronological turn

Create two shared-layout charts, one for Auction 1 and one for Auction 2:

- X axis: one-based action-deck turn.
- Y axis: dollars.
- Bot series:
  `by_turn.<resource action>.submitted_by_bot.<bot>[].mean`.
- Market series:
  `by_turn.<resource action>.market_winning_bid[].mean`.

Submitted-bid means include passes as `$0`; market winning means include `$0`
priority wins. Include observation counts in the detail table because late
turns have fewer samples.

### Private-information premium

Create five shared-scale facets, one for each value chart:

- X axis: hidden matches `0`, `1`, `2`, and `3+`.
- Y axis: expected value premium per offered resource.
- Value:
  `information_asymmetry.<chart>.<bot>.<hidden>.mean_private_premium_per_card`.

The premium is a valuation delta. It is not the causal effect of private
information on the submitted bid.

### Actual bid per offered resource by hidden matches

Create five shared-scale value-chart facets:

- X axis: hidden matches `0`, `1`, `2`, and `3+`.
- Y axis: submitted bid divided by offered-resource count.
- Series:
  `information_asymmetry.<chart>.<bot>.<hidden>.mean_bid_per_card`.

Plot this beside the premium facets to compare valuation information with
actual bidding behavior. Keep the `count` available in tooltips or a table;
the `3+` bucket is naturally sparse.

### Scores by value chart

- X axis: charts A through E.
- Bot series:
  `by_chart.<chart>.bot_scores.<bot>.mean`.
- Market winning-score series:
  `by_chart.<chart>.winning_score.mean`.

Use one shared score scale. The market line is a game-level first-place score;
the bot lines are player-level final scores.

### Resource-auction prices by value chart

- X axis: charts A through E.
- Auction 1 series:
  `by_chart.<chart>.auction_prices.Auction 1 resource.market.mean`.
- Auction 2 series:
  `by_chart.<chart>.auction_prices.Auction 2 resources.market.mean`.

These are market clearing prices across all winners. The nested `by_winner`
data can be used for follow-up charts without changing the market series.

### Turn detail table

For every turn and resource-auction type, include:

- market count, mean, median, and free rate;
- each bot's submitted-bid count, mean, and standard deviation.

This table is the audit trail for the two turn charts.

## Endgame and cash-constraint tables

Source: `analysis-output/endgame-resources-cash.json`

These results were discussed after the visual suites and should accompany a
future rerun even when they are presented as tables rather than charts.

### Game length

Use `turns_per_game` to report mean, median, p10, p25, p75, p90, minimum, and
maximum completed action-deck turns.

### Resources at game end

For each bot, use `bots.<bot>.resources_won`. Report the same distribution
summary and retain the histogram for a distribution chart.

### Excess cash

For each bot, use `bots.<bot>.terminal_liquid_cash`:

- `positive_rate`: finishes with any cash;
- `at_least_5_rate`: finishes with at least `$5`;
- `at_least_10_rate`: finishes with at least `$10`;
- `zero_rate`: finishes with no liquid cash.

Also report `bots.<bot>.terminal_cash_after_loan_debt` so loan principal is not
mistaken for retained wealth.

### Unable to bid

For each bot, use `bots.<bot>.bid_constraints`:

- `cash_zero_request_rate`: share of bid requests made with zero cash;
- `cash_zero_share_of_passes`: share of passes that occurred with zero cash;
- `hard_constrained_request_rate`: share of requests with legal maximum zero;
- `hard_constrained_share_of_passes`: share of passes that were legally forced;
- `voluntary_share_of_passes`: share of passes made with positive cash;
- `bot_games_cash_zero_rate`: games in which the bot reached at least one
  zero-cash bid request;
- `bot_games_hard_constrained_rate`: games in which the bot reached at least
  one legally constrained bid request.

These metrics separate a strategic pass from a pass caused by exhausted
buying power.

## Late two-resource auction diagnostic

Source: `analysis-output/late-resource-auctions.json`

Use this dataset when Auction 2 prices appear to collapse:

- periods: early turns 1–5, middle turns 6–12, late turns 13+;
- per-bot valuation inputs:
  `auction2_by_band.<bot>.<period>`;
- market clearing prices:
  `auction2_market_by_band.<period>`;
- acquisition cost and frequency:
  `resource_payments_per_game.<bot>`;
- terminal cash:
  `mean_terminal_cash.<bot>`;
- detailed Auction 2 requests:
  `auction2_by_turn.<bot>.<turn>`.

Compare mean cash, legal maximum, submitted bid, reservation bid, gross value,
resource value, objective value, liquidity cost, zero-cash rate, pass rate,
and cap-binding rate. This distinguishes:

1. the lot genuinely becoming less valuable;
2. the heuristic reserving more cash;
3. the bot wanting to bid but being cash constrained;
4. fewer or compositionally different late-game observations.

## Visual encoding

Keep bot identity consistent across every suite:

| Series | Color token | Mark | Line |
| --- | --- | --- | --- |
| Aggressive | series 1 | circle | solid |
| Balanced | series 2 | square | dashed |
| Passive | series 3 | diamond | dotted |
| Market / winning score | foreground | line | long dash |

Use shared scales for direct comparisons and small multiples for chart A–E.
Pair color with marks or line styles so the plots remain readable without
color. Put exact values in tooltips or adjacent tables rather than adding
separate KPI panels.

## Validation checklist

Before publishing a rerun:

- [ ] Record the clean commit and bot generation.
- [ ] Confirm every dataset reports 100,000 games and root seed `20260729`.
- [ ] Confirm all five rulesets appear and their game counts sum to 100,000.
- [ ] Confirm each bot has 100,000 observations in the tournament summary.
- [ ] Confirm all bot fault counts are zero.
- [ ] Confirm total resolved action auctions equal `games * 30`.
- [ ] Confirm mean total terminal resources across bots matches the resource
      deck size.
- [ ] Confirm `$0` wins remain in winning-bid distributions.
- [ ] Confirm passes remain in submitted-bid means as `$0`.
- [ ] Confirm missing loan-turn observations remain missing rather than `$0`.
- [ ] Compare JSON checksums when testing determinism across worker counts.
- [ ] Label any changed games, seed, rulesets, or bot generations on every
      regenerated visual.

## Suggested Codex request

After generating the JSON, this request recreates the same scope without
depending on old conversation context:

```text
Use the visualize skill to recreate all chart suites documented in
docs/analysis/heuristic-bot-visualizations.md from the JSON files in
analysis-output/. Preserve the documented definitions, series encodings,
shared scales, facets, and detail tables. Do not omit $0 priority wins or
submitted passes.
```
