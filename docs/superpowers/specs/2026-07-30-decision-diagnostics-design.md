# Decision Diagnostics Design

## Purpose

Tournament ratings say which bot performed better, but not which choices
produced the difference. Decision diagnostics add an opt-in record of what a
bot was allowed to know, what it could legally do, what it selected, any
policy explanation available at that moment, and the public result of the
game.

This feature is behavior-preserving. It does not change a policy, call a
policy twice, promote an identity, or expose held-out promotion cases for
tuning.

## Plain-language contract

The implementation uses concrete names:

- a **decision trace** is one bot choice and the live-compatible information
  available when it made that choice;
- a **policy explanation** is a small, typed set of values the policy already
  computed while choosing;
- a **decision slice** groups comparable choices by public conditions;
- **reconciliation** proves that trace and slice totals agree with the
  ordinary game and tournament results.

Diagnostics are enabled with `garboid-tournament --decision-reports`. They
are opt-in because a full tournament can produce millions of decision rows.

## Approaches considered

### Capture at the shared bot-execution boundary — selected

Both scalar and batch simulation call the same bot-execution helper with the
decision context, public rules, and public history. The helper will return the
chosen action together with an optional typed explanation and whether the
action came from the policy or the existing fault fallback.

This is the only point where the policy must be invoked. Recording the result
there prevents diagnostics from asking the bot to choose a second time and
accidentally changing random state or behavior.

### Reconstruct diagnostics from replays

Replays contain actions and final results, but they do not retain the public
decision context or values computed by a policy. Reconstructing those values
would require rerunning a bot and would not prove that the explanation belongs
to the recorded choice.

### Put diagnostics into replay schema v2

Replays are inputs to deterministic reproduction. Explanations are optional
analysis output, not replay inputs. Keeping the schemas separate avoids
changing the replay contract and prevents diagnostic data from influencing
reproduction.

## Privacy boundary

Trace construction uses an explicit field allowlist. It never serializes a
complete SDK `DecisionContext`, arbitrary metadata, a simulator snapshot, a
game engine, a neural observation, or a heuristic belief state.

The recorded public decision context contains:

- decision kind, player count, starting cash, and value chart;
- active objective IDs;
- the current auction action and public resource cards;
- public cash, tiebreak seat, won-resource counts, revealed-information
  counts, and claimed objective IDs by seat;
- the acting seat, maximum legal bid, and number of reveal choices.

It excludes request IDs, deadlines, metadata, raw hand contents, opponent
hands, unclaimed objective ownership, deck order, unresolved bids, random
state, and future actions. A bot may legitimately use its own hand while
choosing, but the trace does not copy that private input. Typed explanation
values derived during the live-compatible policy call may be recorded.

Policy explanations are a closed union rather than a free-form dictionary:

- a heuristic bid explanation records the chosen bid and the finite,
  documented components of the selected bid evaluation;
- a neural policy explanation records the finite value estimate, entropy,
  selected-action probability, and probabilities for legal actions only.

Random and reveal policies may have no explanation. Unknown fields and
nonfinite values are rejected before output.

## Trace schema

`DecisionTrace` is one self-contained JSONL row. It contains:

- schema version;
- opaque game index, chart, and player count;
- ordered lineup entries with seat, bot name, and bot ID;
- step index, turn index, acting seat, bot name, and bot ID;
- the allowlisted public decision context;
- the exact public-history prefix seen by the policy;
- legal actions in canonical order;
- selected action;
- optional typed policy explanation;
- selection source, either `policy` or `fault_fallback`;
- eventual final money, rank, and whether first place was tied.

Seeds are intentionally absent. A simulator seed plus the public game
configuration can reconstruct private hands, deck order, and future actions,
so including it would turn an apparently public trace into a compact hidden
state leak.

Bid actions are ordered as pass followed by positive bids. Reveal actions are
ordered by reveal index. A zero bid is represented as pass, matching the
existing action codec. Automatic reveals do not create decision traces
because no policy decision occurred.

Rows are ordered by `(game_index, step_index, seat)`. JSON uses sorted keys,
rejects nonfinite numbers, uses fixed compact separators, and ends each row
with one newline. Fixed games and identities therefore produce
byte-equivalent trace files across input ordering, batch sizes, and worker
counts.

## Explanations without behavior changes

An optional explanation-aware brain protocol returns one
`ExplainedBotDecision`. The existing protocol remains valid and produces no
explanation.

The execution helper validates the returned `BotDecision` exactly once. A
policy exception or invalid action follows the existing configured fault
behavior; a recorded fallback has no policy explanation and is marked
`fault_fallback`.

Heuristic brains reuse the `BidEvaluation` already needed to select a bid.
Frozen neural brains reuse the masked deterministic inference output already
needed to select an action. Neither policy is evaluated a second time.

## Report slices

`decision-slices.csv` is sparse: it contains one row for each observed
combination of dimensions. Counts and sums remain additive so users can
filter or combine rows without averaging averages.

Dimensions are:

- game phase: early for turns 1–5, middle for 6–12, late for 13 onward;
- chart;
- player count;
- decision kind: bid or reveal request;
- current auction action;
- selected action kind: pass, bid, or reveal choice;
- cash horizon as exact future-biddable and total-biddable resource counts;
- objective state as exact actor-owned, opponent-owned, and unclaimed counts;
- zero-based seat;
- opponent composition as sorted bot IDs, preserving duplicates.

Measures include decision count, pass count, selected-value count and sum,
eventual final-money sum, eventual normalized-finish sum, decisions ending in
an outright win or tied first, and decisions from a faulted game-seat.
Outcome measures are explicitly decision-weighted evidence, not causal
estimates.

## Reconciliation

Reporting fails before writing if any of these checks disagree:

1. Every `(game_index, step_index, seat)` trace key is unique.
2. Every trace matches one game summary and the same chart, player count,
   seat, bot identity, and public outcome.
3. Trace counts per game-seat equal the existing game-summary decision
   counts.
4. Deduplicated game-seat outcomes reproduce bot game, win, tie, money, and
   fault totals.
5. Chart and player-count totals reproduce tournament condition statistics.
6. The sum of slice decision counts equals both the raw trace count and the
   sum of game-summary decision counts.

The results appear in `summary.json` and `report.html`, alongside links to
the detailed artifacts.

## Artifacts and command line

Without `--decision-reports`, tournament behavior and its three existing
artifacts are unchanged.

With the flag, the output generation additionally contains:

- `game-summaries.jsonl` — the public per-game bridge to tournament totals;
- `game-details.jsonl` — the public seed-free turn ledger and terminal score
  breakdown used for objective and auction economics;
- `decision-traces.jsonl` — one canonical row per policy decision;
- `decision-slices.csv` — filterable additive aggregates.

All artifacts are rendered before replacement. The writer stages the complete
known generation, preserves unrelated files, and restores the previous
generation if replacement fails.

The promotion CLI deliberately has no diagnostic flag. Development
tournaments may guide strategy work; held-out promotion cases remain a final
exam rather than a source of tuning slices.

## Testing strategy

Tests are written before implementation and cover:

- strict public-context and explanation schemas, including rejection of
  hidden fields and arbitrary metadata;
- canonical legal actions and selected-action membership;
- exact public-history prefixes and eventual public outcomes;
- explanation/action agreement for heuristic and neural policies;
- unchanged actions, faults, results, ratings, and bootstrap output with
  tracing on or off;
- scalar/batch and single/multiple-worker trace equivalence;
- every slice dimension, phase boundary, additive measure, and reconciliation
  failure;
- byte-identical artifacts for fixed inputs and reversed source order;
- finite canonical JSON, deterministic CSV, HTML escaping, and transaction
  rollback;
- CLI behavior with and without `--decision-reports`.
