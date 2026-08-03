# Tournament

The tournament builds a deterministic multiplayer schedule, executes it
through the simulator, fits a tie-aware Plackett–Luce model, and writes stable
human- and machine-readable artifacts.

## Run the standard field

```bash
uv run --extra neural garboid-tournament \
  --output-dir artifacts/tournaments/default
```

Current defaults are 15,000 games, player counts three through five, value
charts A through E, batch size 64, 200 bootstrap samples, and all available
CPUs except one. Ordinary tournaments retain root seed zero when `--seed` is
omitted. Use `--bots` or `--exclude-bots` for an explicit field,
`--bootstrap-samples 0` for a quick run, and `--overwrite` only when replacing
an existing known artifact generation.

The curated default field retains the released `fixed-objective-overlay-v1`
and v2 generations and adds the cash- and tiebreak-aware
`fixed-objective-overlay-v3`. It also includes the strongest fixed-bid,
heuristic-personality, and frozen large-neural comparisons. Bots outside the
curated field remain registered and explicitly selectable; pruning the default
does not delete their immutable identities or historical evidence. Moving
unversioned heuristic aliases are omitted because they duplicate versioned
behavior.

Decision diagnostics are opt-in:

```bash
uv run --extra neural garboid-tournament \
  --decision-reports \
  --output-dir artifacts/tournaments/diagnostics
```

The flag records each already-selected public decision while retaining batch
execution. It does not change the schedule, actions, summaries, ratings,
analysis, or bootstrap results. Explanation-aware policies are invoked only
when the flag is present.

Bots can emit public numeric result metrics from the same opt-in explained
decision call. Each metric declares a namespace, nested result path, and a
generic `sum` or `mean` aggregation; tournament reporting does not need
bot-specific logic. `fixed-objective-overlay-v3` uses that interface to count
resource auctions, exact guarantee-cap applications, decisions whose bid
changed, and total submitted-bid reduction. Inspect its machine-readable totals
with:

```bash
jq '.decision_diagnostics.bot_metrics.fixed_objective_overlay_v3_rules' \
  artifacts/tournaments/diagnostics/summary.json
```

When decision diagnostics are enabled and `--seed` is omitted, the CLI uses a
fresh, private 63-bit root seed. Pass an explicit `--seed` only when a fixed,
byte-reproducible diagnostic run is required. That explicit seed is sensitive:
keep it private when sharing artifacts because it can reconstruct hidden game
state. Diagnostic artifacts and CLI output never publish the resolved seed.

## Schedule

Every requested chart/player-count cell receives a quota. Lineups contain
distinct bot IDs and balance condition, pair, and global exposure. A
deterministic seat pass minimizes per-bot seat imbalance. Game seeds, lineup
tie-breaks, seats, and bootstrap samples derive from the root seed.

The same plan produces the same game summaries with serial or parallel,
scalar or batch execution. Faults use record-and-pass so one broken bot does
not discard the field; counts remain attributable by bot and seat.

## Rating semantics

The fit consumes complete multiplayer finishes and handles ties as grouped
finishing choices. `worth` is positive strength normalized across the field.
Display rating is:

```text
1500 + 400 * log10(worth / geometric mean worth)
```

This is not a sequence of pairwise Elo updates. A 400-point difference is
10:1 worth odds. Weak ghost comparisons keep undefeated and winless estimates
finite. Confidence intervals bootstrap complete games rather than pairwise
fragments.

A global rating averages across the configured charts, player counts,
opponents, and seats. Use condition statistics and calibration diagnostics to
interpret interactions hidden by one number.

An exploratory tournament does not promote a bot. Before changing a released
identity, run the matched held-out final exam described in the
[promotion runbook](../promotion/README.md).

## Artifacts

Each successful run writes:

- `ratings.csv`: rating-ordered spreadsheet data;
- `summary.json`: exact configuration, schedule, diagnostics, leaderboard,
  condition statistics, and calibration;
- `report.html`: leaderboard, intervals, comparison plots, and calibration.

With `--decision-reports`, the same generation also writes:

- `game-summaries.jsonl`: public per-game outcomes and decision counts;
- `game-details.jsonl`: public resolved turns, objective claims, auction prices,
  resource bundles, and terminal score components;
- `decision-traces.jsonl`: one public row per policy decision;
- `decision-slices.csv`: filterable additive decision aggregates.

The CLI prints each diagnostic path after a successful run. Diagnostic
generations withhold the root seed from all published artifacts because it
could be used to reconstruct hidden state; games are identified only by their
opaque `game_index`. Without the flag, output and the original three artifacts
remain unchanged.

Build the two-engine interactive report from any completed output directory:

```bash
uv run garboid-visualize tournament-results
```

See the [visualizer runbook](../visualizer/README.md) for data availability,
metric definitions, and chart ideas.

Writes are atomic. A bootstrap failure still preserves primary ratings and
reports the missing uncertainty explicitly.

For recorded evidence, see the
[benchmark reports](../../../docs/benchmarks/). Generated tournament artifacts
are local analysis outputs and are excluded from version control. Do not
rewrite a dated report; create a new dated result with its seed, configuration,
repository revision, and runtime.

## Extension points

- Add a bot through the shared registry without changing scheduler logic.
- Add diagnostics to the report from stable game summaries.
- Preserve the [identity](../../../docs/architecture/immutable-bot-identities.md)
  and [determinism](../../../docs/architecture/deterministic-evaluation.md)
  contracts when changing the field or estimator.
