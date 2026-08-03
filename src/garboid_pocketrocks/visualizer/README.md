# Tournament visualizer

The visualizer turns a tournament artifact directory into one self-contained
HTML report with two analysis engines:

- **Tournament insights** compare the whole field through rating intervals,
  directed matchup scores, value-chart sensitivity, model calibration, and a
  detailed leaderboard.
- **Bot deep dive** explains one selected bot through opponent results,
  objective claims, auction economics, loan pricing, chart sensitivity,
  cash pressure, acquisition mix, and terminal score composition.

Generate the richest input and then build the report:

```bash
uv run --extra neural garboid-tournament \
  --decision-reports \
  --output-dir tournament-results

uv run garboid-visualize tournament-results
```

The second command writes `tournament-results/insights.html`. Use `--output`
for another path and `--overwrite` to replace an existing report.

## Data tiers

`summary.json` is sufficient for ratings, conditions, calibration, and the
leaderboard. A tournament created without decision diagnostics therefore still
produces a useful field report.

`--decision-reports` unlocks the behavior views through three public datasets:

- `game-summaries.jsonl` supplies lineups and final ranks for opponent results;
- `game-details.jsonl` supplies the seed-free resolved turn ledger, objective
  claims, winning payments, resource bundles, and terminal score components;
- `decision-traces.jsonl` supplies cash, legal bid caps, selected actions, and
  strategy explanations at each public decision.

The visualizer aggregates traces in one streaming pass. It embeds only the
small aggregate payload in HTML rather than copying millions of raw decisions.

## Metric definitions

- **Head-to-head score** is 1 when the focal bot finishes above the opponent,
  1/2 on a tied rank, and 0 below. Error bars are Wilson 95% intervals.
- **Games with an objective claim** is the share of games where the focal bot
  claimed at least one objective, conditioned on the named opponent appearing.
  The report also retains objectives per 100 games.
- **Resource-auction profit** is terminal chart value of the won bundle minus
  its winning payment.
- **Investment purchase price** is the liquidity locked by the winning bid.
  That principal is returned at scoring, so net profit is always the card's
  fixed $5 or $10 payout regardless of purchase price.
- **Loans** are excluded from profit because principal is repaid at scoring.
- **Loan valuation** shows the distributions of prices a bot was willing to
  win at. `principal - payment` is labeled up-front liquidity, not profit.
- **Cash-starved** means a bid request with zero liquid cash. A separate
  hard-constrained rate records requests whose legal maximum was zero.

Intervals describe sampling uncertainty in this tournament. Opponent and
condition slices are observational comparisons, not causal effects.

## Visualization roadmap

The current report implements the highest-signal, artifact-supported views.
Good next additions are:

- a **budget trajectory** aligning average cash, bids, and acquisition value by
  action-deck turn;
- an **objective funnel** from opportunity, to progress, to claim, to eventual
  win, separating frequent low-value claims from decisive claims;
- **bid response curves** showing bid as a function of cash, offered value, bid
  cap, phase, and private-information advantage;
- a **valuation gap plot** comparing heuristic reservation value or neural
  action probability with the chosen bid and market-clearing price;
- a **strategy fingerprint** combining pass rate, bid-to-cash ratio, purchase
  mix, timing, liquidity floor, and score composition;
- **seat and player-count stability** with uncertainty, to expose policies that
  only work from favorable priority positions or field sizes;
- **comeback paths** comparing early deficit and terminal finish to distinguish
  deliberate late strategies from unrecovered cash starvation;
- **policy confidence calibration** for neural bots, relating action entropy and
  selected probability to auction success and final finish;
- tournament-wide **market price curves**, **rating convergence**, and
  **condition coverage** views for detecting drift or schedule imbalance.
