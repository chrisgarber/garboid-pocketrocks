# Plackett–Luce Bot Tournament Design

## Purpose

The repository needs one repeatable command that estimates the relative strength
of every registered simulator bot. The tournament must remain useful as the bot
population grows, exercise all five live value charts and all supported game
sizes, prevent a bot from playing against another instance of itself, and
produce both machine-readable results and an interpretable report.

The primary estimate will be a generalized Plackett–Luce model fitted to whole
multiplayer finishes. The model will not decompose a multiplayer game into
independent Elo duels. Its fitted worths will also be displayed on a familiar
1500-centered, 400-point logistic rating scale.

## Goals

- Run 10,000 games by default through the existing deterministic Monte Carlo
  engine.
- Cover value charts A–E and 3-, 4-, and 5-player games as evenly as integer
  quotas permit.
- Use a deterministic, balanced schedule containing distinct bot identities in
  every game.
- Estimate one global bot worth from the complete multiplayer rankings,
  including tied ranks.
- Generate a concise leaderboard, stable JSON and CSV artifacts, and a
  self-contained HTML report with useful diagnostic charts.
- Preserve exact results across worker counts for a fixed configuration and
  root seed.
- Make newly registered bots available to tournaments without adding them to a
  second tournament-specific list.

## Non-goals

- The first version will not optimize bot parameters or select a production bot
  automatically.
- It will not claim that a global worth captures every matchup interaction.
- It will not fit chart-specific or player-count-specific strength models.
  Per-condition descriptive statistics will identify those possible
  interactions for future work.
- It will not provide a browser dashboard or hosted report.
- It will not admit duplicate bot identities merely to fill a five-player
  lineup.

## Command-line experience

Add the command:

```bash
uv run garboid-tournament \
  --games 10000 \
  --players 3,4,5 \
  --charts A,B,C,D,E \
  --seed 42 \
  --workers 4 \
  --output-dir tournament-results
```

Those values, except for worker count and output directory, are the defaults.
The command selects all registered simulator bots unless `--bots` supplies a
comma-separated subset. `--exclude-bots` removes named bots after inclusion.
`--bootstrap-samples` defaults to 200 and accepts zero to skip intervals during
quick experiments.

The command prints a compact leaderboard after successful artifact generation.
It creates:

- `ratings.csv`
- `summary.json`
- `report.html`

The output directory must either not exist or be empty unless `--overwrite` is
explicitly supplied. Overwrite replaces only the three known report artifacts;
it does not recursively delete the directory.

The current registry has four distinct bot identities. Therefore,
`--players 3,4` works immediately, while the full default produces a clear
preflight error until a fifth real bot is registered. This is intentional: a
duplicate strategy instance would violate the unique-lineup requirement and
distort uncertainty estimates. The in-progress v2 aggressive, balanced, and
passive bots will become tournament participants simply by adding their
`BotSpec` values to the shared registry; no scheduler or scoring change will be
required.

## Components

### Shared bot registry

Create `garboid_pocketrocks.bots.registry` as the source of simulator-ready
`BotSpec` values. Both `garboid-simulate` and `garboid-tournament` derive their
name lookup from this registry. The live launcher may retain its factory
registry because it constructs networked bot wrappers rather than `BotSpec`
values.

Registry construction validates that bot names and bot IDs are both unique.
Tournament configuration repeats this validation for programmatic callers that
provide their own bot set.

### Tournament configuration and plan

`TournamentConfig` contains:

- ordered bot specs;
- total game count, default 10,000;
- player counts, default `(3, 4, 5)`;
- live chart names, default `("A", "B", "C", "D", "E")`;
- root seed;
- Monte Carlo fault mode, default `record_and_pass`;
- bootstrap sample count, default 200.

Validation requires:

- at least one chart and one player count;
- only known live charts and supported player counts;
- at least `max(player_counts)` distinct bot IDs;
- at least one game in every chart/player-count cell;
- unique bot names and IDs;
- a nonnegative bootstrap count.

`TournamentPlan` holds the configuration, explicit `GameJob` values, and the
integer game quota for every condition cell.

The total games are divided by round-robin remainder allocation over the
lexicographically ordered `(chart, player_count)` cells. Cell totals therefore
differ by at most one and always sum to the requested total.

### Balanced deterministic scheduler

The scheduler generates each cell independently while tracking global
exposure. It never enumerates every combination, which would become
prohibitively large as the registry grows.

For each game it selects a lineup one bot at a time. Candidate ordering uses:

1. appearances in the current chart/player-count cell;
2. accumulated pair appearances with bots already selected for the lineup;
3. appearances across the complete tournament;
4. a stable hash derived from the root seed, cell, game index, and bot ID.

After selecting a lineup, the scheduler chooses its seat permutation by
enumerating at most `5!` possibilities and minimizing, in order:

1. the highest resulting seat exposure for any selected bot;
2. the sum of squared resulting seat exposures;
3. a stable seeded tie-break key.

Every planned job receives a unique global `game_index`. Game seeds are derived
from the tournament root seed and global index using the existing
`derive_seed` helper. Planning is deterministic and independent of worker
count.

The scheduler guarantees unique bot IDs within a lineup. Tests will require
near-balanced appearances and seats; exact equality is required when the
arithmetic permits it. Pair exposure is a greedy balance objective rather than
an exact combinatorial guarantee and will be tested against a bounded spread.

### Monte Carlo execution seam

Expose a public `MonteCarloRunner.run_jobs(config, jobs, workers=...)` method.
The existing `run` method continues to call `plan` and delegates execution to
this method, preserving existing behavior. `run_jobs` validates job count,
indices, supported rulesets, player counts, bot identities, and root seed before
using the existing serial or process-pool execution and aggregation paths.

The tournament scheduler constructs a compatible `MonteCarloConfig` whose
ruleset support contains every selected live chart, then passes its explicit
jobs to this seam. Game rules, bot invocation, event collection, fault
handling, replay behavior, and score aggregation remain owned by the existing
simulator.

Tournament runs do not capture full replays by default because 10,000 replays
would create unnecessary memory and disk pressure. `GameSummary` already
contains every field required for model fitting and reporting.

## Generalized Plackett–Luce model

### Observations

Each `GameSummary` becomes an ordered sequence of nonempty rank groups. Bots
with the same `Score.rank` form a tied group. For example:

```text
{passive} > {balanced, aggressive} > {random}
```

The observation retains the game as one multiplayer ranking. It does not emit
independent pairwise training records.

### Tie-aware likelihood

Let `A` be the bots remaining at a ranking stage, `C` the selected winning
group at that stage, `alpha_i > 0` bot worth, and `delta_k >= 0` the prevalence
parameter for a tie of order `k`, with `delta_1 = 1`.

The stage choice has unnormalized weight:

```text
delta_|C| * (product(alpha_i for i in C)) ** (1 / |C|)
```

The normalizer sums that expression over every nonempty subset of `A` up to the
largest tie order supported by the observed data. With at most five players,
enumerating these subsets is small and exact. Removing `C` and repeating for
the remaining rank groups gives the ranking likelihood.

This is the Davidson–Luce tie extension used by generalized Plackett–Luce
models. Actual equal-money finishes are modeled as ties; they are never ordered
by seat, seed, or bot name.

### Finite estimates and fitting

Every real bot receives two fractional pseudo-rankings of weight 0.5: one win
over a fixed-worth ghost item and one loss to it. The ghost never appears in
reported results. These pseudo-rankings make the comparison network strongly
connected and mildly shrink sparse, undefeated, or winless bots toward equal
worth.

Fit real-bot log-worths and observed tie-order log parameters by penalized
maximum likelihood with an analytic gradient and SciPy L-BFGS-B. `scipy` will
become a direct runtime dependency. Tie parameters use finite numerical bounds;
unobserved higher tie orders are omitted instead of inferred. Convergence
requires the optimizer's success status and a finite objective and gradient.
Failure raises a domain-specific error that includes the optimizer message and
suggests inspecting faults and comparison coverage.

The result records:

- normalized worths summing to one across real bots;
- log-worths centered to mean zero across real bots;
- fitted tie prevalence parameters;
- optimizer iterations and terminal gradient norm;
- the pseudo-ranking weight.

### Display rating

For centered real-bot worth `alpha_i`, display:

```text
rating_i = 1500 + 400 * log10(alpha_i / geometric_mean(alpha))
```

This is a presentation transform, not a sequential Elo update. A 400-point
difference corresponds to 10:1 worth odds, and the mean centered log-rating is
1500. Reports call the column `PL rating`, while JSON also exposes raw worth
and log-worth to prevent ambiguity.

### Bootstrap intervals

Confidence intervals use a deterministic nonparametric bootstrap over complete
games. Each replicate samples `len(games)` game summaries with replacement,
refits the full model, and transforms worths to ratings. Pseudo-rankings are
added fresh to each replicate and are not themselves resampled.

The 2.5th and 97.5th empirical percentiles form the displayed 95% interval.
Bootstrap seeds derive from the tournament root seed. Replicates run after the
Monte Carlo tournament so the main strength estimate and artifacts remain
available even when bootstrap sampling is disabled. A failed replicate is
reported and excluded; the report fails if fewer than 90% of requested
replicates converge.

## Metrics and artifacts

### Leaderboard fields

Each bot row contains:

- rank;
- name and bot ID;
- normalized PL worth;
- PL rating;
- optional 95% bootstrap interval;
- games played;
- outright wins and first-place ties;
- mean normalized finish, where first is 1 and last is 0 for any player count;
- mean final money;
- mean winning final money over all rank-1 finishes, including tied firsts;
- faults.

Sorting is descending PL rating, then bot name.

### `summary.json`

The versioned JSON document contains:

- schema version;
- exact tournament configuration;
- root seed and schedule cell quotas;
- optimizer and bootstrap metadata;
- global leaderboard rows;
- fitted tie parameters;
- descriptive bot statistics by chart and player count;
- calibration bins;
- artifact filenames.

JSON uses stable key ordering. It retains integer counts and unrounded floating
point values.

### `ratings.csv`

CSV contains one row per bot with the leaderboard scalar fields. It is intended
for diffs, spreadsheets, and downstream experiments. Floats use enough digits
to round-trip.

### `report.html`

The report is a self-contained, responsive HTML document with no remote assets
or JavaScript requirement. It contains an accessible leaderboard table,
configuration and convergence metadata, fault warnings, and inline SVG charts:

1. **PL rating leaderboard.** Horizontal points and interval whiskers make
   ordering and uncertainty immediately visible.
2. **Rating versus mean winning money.** A labeled scatter plot shows whether
   winning score magnitude tracks modeled strength or identifies bots that win
   narrowly or only in high-value games.
3. **PL calibration.** Every within-game bot pair contributes an observed
   outcome of 1, 0, or 0.5 for a tie. Bins compare the pairwise probability
   implied by final PL worths with the observed mean outcome, against a perfect
   calibration diagonal. This diagnostic does not change the listwise fit.

Charts include text alternatives and use a consistent bot ordering. Empty
winning-money samples or disabled bootstrap intervals render explicit `n/a`
states rather than fabricated values.

## Faults and diagnostics

Tournament execution defaults to `record_and_pass` so one faulty bot does not
discard a long experiment. Faults remain attributed to the responsible bot and
are prominent in all artifacts. A bot fault is not converted into an explicit
rank penalty; its fallback pass influences the simulated result naturally.

Preflight errors occur before games start for:

- insufficient distinct bots;
- duplicate names or IDs;
- invalid chart/player selections;
- too few games to cover all requested cells;
- unsafe output-directory replacement;
- non-picklable bot factories when workers exceed one.

Post-simulation errors occur before artifacts are finalized for non-finite
data, optimizer failure, or insufficient converged bootstrap replicates.
Artifacts are first written to temporary siblings and atomically replaced so a
failed report does not leave mixed-version output.

## Testing strategy

Implementation follows test-driven development.

### Scheduler unit tests

- identical seed/configuration produces identical jobs;
- different seeds change deterministic tie breaks;
- all 15 default cells receive quotas differing by at most one and totaling
  10,000;
- every lineup contains distinct bot IDs;
- every job uses the requested chart and player count;
- appearances, pairs, and seats stay within explicit balance bounds;
- global indices and derived seeds are unique;
- four bots with requested five-player games fail before execution.

### Model unit tests

- equal synthetic worths recover approximately equal ratings;
- a bot consistently ranked above peers receives the highest worth;
- permuting bot names or seats permutes estimates without changing values;
- tied rank groups are invariant to order within the group;
- two-way and higher-order ties produce fitted tie parameters;
- winless and undefeated bots retain finite worth through pseudo-rankings;
- normalized worths sum to one and centered ratings average 1500 on the log
  scale;
- a known worth ratio maps to the expected rating difference;
- non-convergence and non-finite inputs fail clearly;
- bootstrap output is deterministic and ordered.

### Report unit tests

- JSON round-trips with the declared schema and unrounded values;
- CSV rows follow rating order and round-trip scalar values;
- HTML contains all leaderboard rows, SVG charts, accessible alternatives,
  configuration metadata, and fault warnings;
- bot names are safely escaped in every format;
- output replacement affects only the known artifact files.

### Integration tests

- a small five-bot tournament covers every requested chart/player cell;
- serial and two-worker runs produce identical plans, summaries, estimates,
  and non-bootstrap artifacts;
- the CLI defaults advertise 10,000 games and all 15 conditions;
- include/exclude filters resolve through the shared registry;
- existing `garboid-simulate` behavior remains compatible after registry
  extraction.

The full 10,000-game tournament is a benchmark command, not a normal unit test.
The test suite verifies its exact default plan shape without paying the runtime
cost on every run.

## Documentation

Update the README with:

- the default tournament command;
- the distinction between PL worth and the Elo-style display scale;
- artifact descriptions;
- current distinct-bot preflight behavior;
- reproducibility guarantees;
- a short interpretation warning that global worth averages over charts,
  player counts, and opponent mixtures.

The first real 10,000-game result should be checked in as a dated benchmark
report only after at least five distinct production-relevant bots are
registered.

## Success criteria

The feature is complete when:

1. a programmatic caller can plan and execute a deterministic tournament using
   explicit unique lineups through the Monte Carlo engine;
2. the default plan contains exactly 10,000 games distributed across all 15
   chart/player-count cells;
3. a five-or-more-bot population yields finite tie-aware Plackett–Luce worths
   and 1500-centered display ratings;
4. serial and process-worker execution produce identical tournament results;
5. CSV, JSON, and self-contained HTML artifacts are generated atomically;
6. the HTML report includes the leaderboard, winning-money relationship, and
   calibration charts;
7. the relevant focused tests, full test suite, lint, and type checking pass.

## Statistical references

- Plackett, R. L. (1975), “The Analysis of Permutations,” *Applied
  Statistics* 24(2).
- Luce, R. D. (1959), *Individual Choice Behavior*.
- Firth, D., Kosmidis, I., and Turner, H. (2019), “Davidson–Luce model for
  multi-item choice with ties.”
- Turner, H. L., van Etten, J., Firth, D., and Kosmidis, I. (2020),
  “Modelling rankings in R: the PlackettLuce package,” *Computational
  Statistics* 35, 1027–1057.
