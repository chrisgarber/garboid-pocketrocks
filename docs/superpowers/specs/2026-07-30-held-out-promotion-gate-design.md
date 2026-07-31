# Held-Out Promotion Gate Design

## Purpose

A bot should not be called better because it won a convenient tournament.
Promotion uses a fixed final exam: the candidate and incumbent play matched
games that were not used while tuning the candidate, and the candidate
advances only when the uncertainty interval is entirely favorable.

This feature is strategy-neutral. Heuristic, random, and neural `BotSpec`
implementations use the same gate. It does not change a bot, train a bot, or
advance an identity by itself.

## Plain-language contract

The implementation and its documentation use concrete domain names:

- a **promotion corpus** is a versioned recipe for a fixed set of games;
- a **promotion case** is one chart, player count, focal seat, opponent lineup,
  and engine seed;
- a **paired game plan** contains two copies of a case: the candidate occupies
  the focal seat in one and the incumbent occupies it in the other;
- a **promotion report** records what ran, the measured rating difference,
  its uncertainty interval, every reason the gate failed, and the final
  decision.

User-facing text introduces “held-out” as “not used for tuning” and
“bootstrap interval” as “an uncertainty range produced by resampling complete
matched cases.” Public APIs avoid unexplained abbreviations.

## Approaches considered

### Dedicated paired promotion package — selected

Build a small `promotion` package on the simulator's existing `GameJob`,
`MonteCarloRunner`, and Plackett-Luce fitter. It owns corpus loading, paired
planning, paired bootstrap analysis, reporting, and a `garboid-promote` CLI.

This gives every bot type the same gate and makes the pairing rules explicit.
It also keeps ordinary exploratory tournaments separate from promotion.

### Compare two ordinary tournaments

Running `TournamentRunner` once per bot is superficially simpler, but changing
one identity changes lineup selection and seat balancing. The results would
not be matched case by case, weakening the central fairness guarantee.

### Extend neural evaluation

The existing neural evaluation helpers already rotate seats, but making them
the promotion authority would exclude non-neural bots and couple a general
release decision to training-specific data structures.

## Package boundaries

Create `src/garboid_pocketrocks/promotion/` with focused modules:

- `corpus.py` defines immutable corpus recipes, expands them into cases,
  computes canonical SHA-256 digests, and validates development/held-out
  separation.
- `planning.py` converts held-out cases into candidate/incumbent twin
  `GameJob` values and retains the expected identity and pair metadata needed
  to validate results.
- `analysis.py` validates completed games, fits the rating model, resamples
  complete pairs, and builds a fail-closed decision.
- `reporting.py` writes deterministic, atomic machine-readable artifacts.
- `runner.py` orchestrates validation, execution, analysis, and reporting.
- `cli.py` exposes the gate as `garboid-promote`.

The package README explains the workflow in plain English before introducing
the statistical terms. The root and tournament runbooks link to it without
duplicating the full contract.

## Immutable corpora

Commit two versioned JSON recipes:

- `configs/promotion/development-v1.json`
- `configs/promotion/held-out-v1.json`

Each recipe contains:

- schema version, stable corpus name, and purpose;
- charts A-E and player counts 3, 4, and 5;
- repetitions per chart/player-count/focal-seat cell;
- a root seed;
- the ordered, versioned opponent names used to fill non-focal seats.

The development recipe uses four repetitions and the held-out recipe uses
eight. Expanding all focal seats produces 240 development cases and 480
held-out cases. A promotion run therefore executes 960 held-out games: two
games per case.

The initial opponent ring is `random`, `aggressive-v1`, `balanced-v1`, and
`passive-v1`. The expansion rotates that ring deterministically so each
player-count and focal-seat cell sees stable mixtures. Candidate and incumbent
identities must be different from each other and from every opponent identity.

The loader normalizes and validates a recipe, expands every case, serializes
the expanded form with sorted keys and no nonfinite numbers, and hashes those
bytes. Reports store both corpus names and digests. A new corpus requires a
new name and committed file rather than silently changing the meaning of an
old result.

Development and held-out cases are loaded together. Their expanded engine-seed
sets must be disjoint. Duplicate seeds within either corpus, overlap between
corpora, duplicate cells, unsupported charts/player counts, unknown or
duplicate opponents, and malformed JSON fail before simulation.

The held-out corpus is operationally available, so software cannot prove that
a person never inspected it. The runbook establishes the process rule:
development results may guide tuning; held-out results are used only for the
final promotion decision.

## Paired game planning

Every held-out `PromotionCase` becomes a `PairedGamePlan`:

1. `candidate_game` places the candidate at the case's focal seat.
2. `incumbent_game` places the incumbent at that same seat.
3. Both games use the same engine seed, chart, player count, objectives flag,
   and opponents in the same other seats.

Because bot-brain seeds are derived by seat from the engine seed, unchanged
opponents also receive the same random streams in both twins. Game indices are
unique, while the pair identifier ties the two summaries back to the same
case.

Planning uses `FaultMode.RECORD_AND_PASS` so a bad decision becomes evidence
in the report instead of erasing the entire run. An invalid `BotDecision` is
already recorded as a bot fault by the simulator, so it necessarily fails the
gate.

## Rating difference and uncertainty

Analysis combines all candidate, incumbent, and opponent game summaries and
fits the existing tie-aware Plackett-Luce model. The reported point estimate is:

```text
candidate rating - incumbent rating
```

Bootstrap replicates resample whole `PairedGamePlan` units with replacement.
Both twins always remain together. Each replicate refits the same model and
records the candidate-minus-incumbent difference. The 2.5th and 97.5th
percentiles form the deterministic 95% interval.

The bootstrap uses the existing stable seed derivation and produces identical
results for one or multiple workers. At least 90% of requested replicates must
converge; otherwise no interval is published and promotion fails.

## Fail-closed decision

`PromotionReport.promoted` is true only when all of these hold:

- the candidate and incumbent identities match the requested `BotSpec`
  values and are distinct;
- development and held-out corpus validation succeeds;
- every expected pair has exactly one candidate game and one incumbent game;
- each result matches its expected seed, chart, player count, seats, and bot
  identities;
- every game completed;
- all candidate, incumbent, and opponent fault counts are zero;
- the fit, point estimate, and interval are finite;
- enough bootstrap replicates converged;
- the lower endpoint of the 95% interval is greater than zero.

Expected domain failures produce `promoted: false` and stable reason codes with
plain-English messages. Examples include `corpus_seed_overlap`,
`identity_mismatch`, `missing_paired_game`, `bot_fault`,
`nonfinite_analysis`, `bootstrap_incomplete`, and
`interval_includes_zero`.

Malformed invocation and filesystem failures remain ordinary exceptions.
Simulation infrastructure failures are caught at the orchestration boundary
and produce a failed report when the output directory is writable.

## Artifacts

Every attempted run writes an output directory containing:

- `promotion-report.json` — the authoritative decision;
- `paired-games.jsonl` — one canonical summary per executed twin;
- `corpus-snapshot.json` — normalized development and held-out recipes,
  expanded seeds, and digests.

The report schema includes:

- schema version and repository commit;
- candidate, incumbent, and opponent names and IDs;
- complete execution configuration;
- development and held-out corpus names, digests, and seeds;
- requested/completed pairs and games;
- rating difference and 95% interval, or `null` when unavailable;
- requested/converged bootstrap counts;
- total and per-identity faults;
- artifact filenames;
- ordered failure reasons;
- final `promoted` boolean.

JSON uses sorted keys, a terminal newline, and `allow_nan=False`. Files are
written atomically. Repeating a deterministic run with the same inputs
produces byte-identical artifacts except that the repository commit changes
when the source revision changes.

## Command line

Add:

```bash
uv run garboid-promote \
  --candidate vector_ppo_large_v1_g350k \
  --incumbent vector_ppo_small_v1_g1500 \
  --development-corpus configs/promotion/development-v1.json \
  --held-out-corpus configs/promotion/held-out-v1.json \
  --bootstrap-samples 1000 \
  --workers 8 \
  --output-dir promotion-results/neural-comparison
```

The CLI resolves registered bot names, prints a short plain-English decision
and interval, and points to `promotion-report.json`. It exits zero only when
the candidate is promoted, one for a completed but failed gate, and two for
invalid invocation or an operational error.

The Python runner continues to accept arbitrary `BotSpec` values so a newly
implemented candidate can be tested before it is added to the default
tournament lineup.

## Testing strategy

Use strict test-driven development.

Unit tests cover:

- JSON parsing, immutability, canonical digest stability, exact corpus
  expansion, and every seed-overlap rejection;
- twin-game equality for seed, chart, player count, focal seat, opponents, and
  opponent order;
- candidate/incumbent identity and opponent-collision rejection;
- paired bootstrap determinism, whole-pair resampling, convergence thresholds,
  finite output, and a positive/negative gate;
- every fail-closed reason, including missing twins, identity mismatch,
  nonfinite data, and bot faults;
- deterministic atomic report rendering and nonfinite JSON rejection;
- CLI defaults, plain-English output, exit codes, and unknown identities.

Integration tests use small fixture corpora to keep CI fast. They execute real
simulator games, prove serial/parallel equality, compare report bytes, and run
an intentionally illegal bot to demonstrate a failed decision. A synthetic
strength fixture pins one passing and one non-passing interval without making
a strength claim about a released bot.

The final gate runs the complete test suite, Ruff formatting/lint, core and
neural strict mypy, local documentation links, and `git diff --check`.

## Documentation and versioning

The promotion README leads with the fair-final-exam explanation, includes the
exact command, defines every report field, and explains why a non-promotion is
not evidence that the candidate is worse.

No existing bot identity, coefficients, alias, checkpoint, or tournament
default changes. No existing bot is declared stronger by this infrastructure
work. Later strategy issues consume the report and may advance an alias only
after their own held-out result passes.
