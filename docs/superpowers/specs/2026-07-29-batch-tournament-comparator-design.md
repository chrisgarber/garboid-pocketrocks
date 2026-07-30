# Batch Tournament Comparator Design

## Purpose

The all-bot tournament currently spends time in two independent hot paths:

1. every scheduled game is run through a scalar SDK session; and
2. every Plackett–Luce bootstrap replicate rebuilds and repeatedly evaluates
   the same ranking-choice structure.

The comparator should use the SDK's vectorized rules kernel for bulk games and
compile the rating likelihood into sufficient statistics. It must preserve the
existing deterministic schedule, bot decisions, statistics, rating model,
bootstrap resamples, reports, and command-line defaults.

## Evidence

A 300-game tournament over all registered bots, player counts 3–5, and charts
A–E took 3.25 seconds with bootstrap disabled. The same tournament with only
20 bootstrap replicates took 6.23 seconds. During a 10,000-game default run,
simulation completed before interruption; the stack trace was inside the
200-replicate bootstrap executor.

The SDK fork commit
`51cad378ee1e70a78e39ebbb25957ea003444873` provides `BatchSimEngine`.
It resolves homogeneous-player-count batches through compact NumPy arrays, but
intentionally omits bot callbacks, `DecisionContext` objects, traces, and event
records. The comparator therefore needs a thin Garboid adapter rather than
calling the engine directly from the CLI.

## Goals

- Run non-replay tournament games through `BatchSimEngine`.
- Support the existing 3-, 4-, and 5-player schedule, charts A–E, objective
  flags, bot faults, and deterministic bot-brain seeding.
- Produce the same `MonteCarloResult`, bot statistics, behavior statistics,
  Plackett–Luce fit, bootstrap intervals, and report data as the scalar path.
- Preserve exact whole-game bootstrap resampling for a fixed seed.
- Reduce likelihood work by evaluating each distinct remaining-bot choice set
  once per optimizer evaluation.
- Keep replay capture on the scalar SDK path.
- Expose batch size as a CLI option with a measured default.
- Record separate simulation, fit, bootstrap, and total durations.

## Non-goals

- Change the tournament schedule or bot registry.
- Change the generalized Plackett–Luce or Davidson–Luce tie model.
- Replace nonparametric bootstrap intervals with Hessian or parametric
  approximations.
- Add timing thresholds to CI.
- Vectorize bot strategies themselves. Existing synchronous `BotBrain`
  implementations remain valid.

## Architecture

### SDK dependency

Pin the reviewed SDK fork commit that exports `BatchSimEngine`. NumPy is already
a direct Garboid dependency. The scalar SDK engine remains available and is
still the authoritative rich-trace implementation.

### Direct batch contexts

Add a private context adapter that materializes one deterministic
`DecisionContext` from a `BatchSimEngine` row. It reads only public game state:

- player count, starting cash, value chart, and active objectives;
- current action and offered resources;
- cash, tiebreak seat, won/revealed resource counts, and owned objectives;
- the acting bot's private hand and legal maximum.

Request IDs use the SDK simulation namespace and the same
seed/turn/seat/decision-kind key. Deadline fields retain Garboid's deterministic
sentinels. Full-game differential tests compare every generated batch context
with the scalar SDK context before any tournament execution uses the adapter.

### Batched match execution

`MonteCarloRunner.run_jobs` groups explicit scheduled jobs into chunks with one
player count. A chunk may mix value charts and objective flags because the SDK
stores those per row. Each row creates the same per-seat brain seeds as
`MatchRunner`.

For every phase:

1. flip actions for all active rows;
2. materialize bid contexts and call the existing brains;
3. validate decisions or apply the configured fallback while recording faults;
4. resolve all bids in one vector call;
5. materialize reveal contexts only for rows requiring a choice;
6. apply automatic and chosen reveals in one vector call; and
7. record compact per-seat decisions and outcome deltas.

The batch path produces compact completed-game records containing scores,
decision/fault counts, and behavior counters. Scalar matches are converted to
the same record type before aggregation. This removes replay/event object
construction from the bulk path without changing public statistics.

When `capture_replays=True`, `run_jobs` retains the existing scalar execution
because exact replay materialization is explicitly requested.

Process workers receive whole chunks instead of individual games. Chunk output
is sorted by `game_index`, so worker count and batch size cannot affect result
ordering.

### Compiled Plackett–Luce likelihood

Each ranking observation is a sequence of stage choices. At a stage, the
likelihood depends only on:

- the set of bots still remaining;
- the selected rank group; and
- the observation weight.

Compile observations by remaining-bot set. For each set, precompute candidate
subsets and their sparse feature coefficients. Aggregate selected-subset
weights. During an optimizer evaluation:

1. compute candidate log weights once for the remaining set;
2. multiply its log-normalizer and expected feature vector by total stage
   weight; and
3. subtract the weighted selected-subset features.

This is algebraically identical to the current per-observation loop, including
tie-prevalence parameters and ghost pseudo-rankings, but bounds likelihood work
by distinct choice sets rather than games times optimizer iterations.

### Weighted bootstrap observations

Convert tournament games to ranking observations once. For replicate `r`, keep
the existing `random.Random(derive_seed(...))` index draws, count the selected
game indices, and combine their counts into observation weights. This preserves
the exact old whole-game resample while avoiding repeated `GameSummary` to
observation conversion and duplicate object creation.

Each replicate uses the compiled sufficient-statistic likelihood. Serial and
parallel bootstrap results remain equal for a fixed seed.

### CLI and reporting

Add `--batch-size`, defaulting to 256 initially and adjusted only from benchmark
evidence. `--workers` continues to control both batch-execution processes and
bootstrap processes. Existing output artifacts remain stable except for
additive timing and batch configuration fields.

The terminal leaderboard continues to include every registered simulator bot;
the shared registry already includes `random`.

## Error Handling

- Invalid batch sizes fail during CLI/config validation.
- Batch context parity failures are test failures; there is no silent scalar
  retry that could hide divergence.
- Brain construction and decision failures follow the existing `FaultMode`.
- Process-pool import/pickle errors retain the existing clear bot-name error.
- Bootstrap convergence handling and warnings remain unchanged.
- Replay requests explicitly select scalar execution rather than producing
  incomplete batch replays.

## Testing Strategy

Development follows red-green-refactor.

1. Add a failing test proving compiled likelihood value and gradient match the
   reference implementation over untied and tied weighted observations.
2. Add a failing bootstrap test proving optimized serial and parallel results
   equal the reference resampling result for fixed fixtures.
3. Add failing full-game context parity tests across player counts, charts, and
   objective settings.
4. Add failing scalar-versus-batch `MonteCarloResult` tests for mixed tournament
   jobs, multiple batch sizes, faults, ties, and behavior statistics.
5. Verify replay capture still uses the scalar path.
6. Run tournament, simulator, bot, heuristic, full pytest, Ruff, strict mypy,
   and `git diff --check`.

Performance is verified outside CI with reproducible commands that time:

- scalar versus batch simulation with bootstrap disabled;
- reference versus compiled fitting/bootstrap on identical observations; and
- the complete default 10,000-game, 200-bootstrap all-bot tournament.

## Acceptance Criteria

- The registry used by the run includes `random`.
- Batch and scalar results compare equal for deterministic parity fixtures.
- Compiled and reference likelihoods agree within floating-point tolerance.
- Fixed-seed bootstrap intervals agree within floating-point tolerance and are
  independent of worker count.
- The optimized comparator is materially faster in both simulation and
  bootstrap benchmarks.
- The default 10,000-game tournament completes with zero bot faults.
- `ratings.csv`, `summary.json`, and `report.html` are generated and the final
  leaderboard is reported to the user.
