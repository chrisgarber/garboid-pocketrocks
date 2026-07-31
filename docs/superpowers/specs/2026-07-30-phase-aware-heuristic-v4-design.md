# Phase-Aware Heuristic v4 Design

## Purpose

Heuristic v3 uses one set of four coefficients for an entire game. The same
tradeoff is therefore applied when nearly every resource is still available
and when only a few auctions remain.

Heuristic v4 gives each aggressive, balanced, and passive personality three
experts: one for the early game, one for the middle game, and one for the late
game. A small deterministic selector chooses among those experts using only
the public number of resources that can still be auctioned. Evolution tunes
the three experts together so it evaluates the behavior that will actually be
released.

The work does not add hidden information, opponent modeling, endgame search,
or neural inference. Heuristic v1, v2, and v3 remain immutable and selectable.

## Plain-language contract

- An **expert** is one ordinary four-coefficient heuristic profile assigned to
  one part of the game.
- A **phase-aware profile** contains the early, middle, and late experts for
  one personality.
- **Future biddable resources** are public resource cards that can appear in a
  later auction. Cards already won and cards in the current auction are not
  future resources.
- **Total biddable resources** are all resource cards that can be auctioned in
  the game after subtracting the private starting cards.
- The **expert phase** is the phase selected from those two public counts. It
  is distinct from the existing turn-based diagnostic `game_phase`.
- A **composite candidate** is one proposed phase-aware profile. It has twelve
  coefficient values: the existing four values for each of three experts.
- **Freezing** records a development winner and its complete provenance.
  Freezing is not promotion.
- **Promotion** is the unchanged held-out final exam against the matching v3
  predecessor. No v4 alias moves unless all three personalities pass.

The implementation uses names such as `future_biddable_resources`,
`total_biddable_resources`, `selected_expert_phase`, and
`PhaseAwareHeuristicProfile`. It avoids unexplained abbreviations in public
APIs and reports.

## Phase selection

Let:

- `R` be the number of future biddable resources, excluding resources in the
  current auction; and
- `T` be the public total number of biddable resources.

The selector is a pure function:

```text
early   when 3 * R >= 2 * T
middle  when 3 * R >= T
late    otherwise
```

Integer multiplication avoids floating-point boundary behavior. For canonical
rulesets, `T` is 15 in three- and five-player games and 14 in four-player
games. Both totals produce the same integer ranges:

| Future biddable resources | Selected expert |
| ---: | --- |
| 10 or more | early |
| 5 through 9 | middle |
| 0 through 4 | late |

The thresholds divide the remaining biddable resource cards into thirds. They
are independent of player count, seat, bot history, hidden hands, deck order,
and the number of policy calls made during a turn. Excluding the currently
offered cards is deliberate: those cards are the present decision, while the
phase answers how much game remains after that decision opportunity.

The selector requires `T > 0` and `0 <= R <= T`. One extracted helper computes
these counts without reading the acting bot's hand or history. It is
action-aware: Auction 1 removes only its first offered resource from the
future horizon, while Auction 2 removes both. Belief accounting and decision
diagnostics migrate to this helper so they cannot disagree at a phase
boundary. This intentionally corrects the old diagnostic-only calculation,
which subtracted both visible resource slots for every auction. Deterministic
public states therefore always select the same expert.

The existing diagnostic `game_phase` remains exactly as it is: early for
turns 1–5, middle for turns 6–12, and late for turn 13 onward. It is useful for
comparing decisions at the same turn position and changing it would silently
alter existing reports. Phase-aware explanations and slices add a separate
nullable `selected_expert_phase`. It is populated only when a phase-aware
heuristic actually selected an expert.

Existing trace schema v1 bytes remain unchanged. A phase-aware bid uses a new
`phase_aware_heuristic_bid` explanation and trace schema v2 containing the
selected phase plus exact future and total resource counts. The decoder keeps
the strict v1 path and dispatches to a separately strict v2 path. A diagnostic
generation containing any v2 trace uses report schema v2 and adds the
`selected_expert_phase` CSV column; a v1-only generation retains its existing
schema and columns byte-for-byte.

Ordinary bots, shared reveal decisions, invalid-input policy passes, and fault
fallbacks do not select a coefficient expert. Their phase is absent rather
than guessed. In a schema-v2 report those rows have an empty expert-phase
column and do not contribute to expert selection counts. Reconciliation still
includes them in ordinary decision totals.

## Boundary justification

The public resource horizon is already a decision-slice dimension. It is a
better policy input than turn number because it states directly how many
resource cards remain available to future auctions and is available in the
ordinary live-compatible decision context.

Before writing v4 search manifests, run one v3-only development tournament
with decision reports across charts A-E and player counts 3/4/5. Its committed
report records the distribution of decisions and outcomes by exact `(R, T)`
state and demonstrates that the thirds give nonempty, meaningful early,
middle, and late ranges across all supported conditions. The report uses a
new development seed and no held-out cases. Its path and SHA-256 digest are
bound into every v4 manifest.

The initial thirds are fixed after that boundary report and before v4 search.
The report justifies a simple equal-resource-horizon split; it does not search
alternative thresholds. Later winner diagnostics show how often each expert
was selected and how outcomes differed across the three fixed ranges, but may
not move the boundaries or trigger retuning. This prevents the same
development evidence from searching both boundary placement and expert
coefficients without an explicit larger budget.

After each personality's search selects one winner, run decision diagnostics
once for that winner on the development corpus. The report records:

- the fixed boundary formula and canonical 10+/5–9/0–4 interpretation;
- decision counts for each selected expert;
- coefficient values for each expert;
- phase-sliced final-money and normalized-finish sums;
- win and tied-first decision counts; and
- faults and reconciliation totals.

This is descriptive evidence, not a causal estimate and not a second
selection pass. No losing search candidate receives a decision-diagnostics
run, and diagnostic results cannot trigger retuning.

## Versioning decision

The behavior-change path is:

1. Preserve `HEURISTIC_V1`, `HEURISTIC_V2`, `HEURISTIC_V3`, their registered
   names and bot IDs, their factories, and representative decisions.
2. Add the composite profile types and initialize all three experts of each
   development candidate from its matching released v3 profile.
3. Search produces three local composite identities, one per personality,
   such as `balanced-v4-candidate-g007-s004-4f91c7d2a6b8`.
4. Frozen v4 candidates remain in the development candidate catalog and do
   not enter the released bot registry or default tournament lineup.
5. Compare every frozen composite only with its matching v3 predecessor on
   the common held-out corpus.
6. Create the nine released expert profiles inside `HEURISTIC_V4`, register
   explicit v4 bot names, and advance unversioned/latest policy and bot
   aliases only after aggressive, balanced, and passive all pass.

If any personality fails, all three frozen candidates and their evidence
remain available for analysis, but there is no partial v4 release and v3
remains latest. Remote `BOT_ID` constants never change.

## Approaches considered

### Three joint twelve-locus searches — selected

Run one search for each personality. Each genome contains the four existing
coefficients for all three experts, for twelve coefficient loci total.
Fitness comes from games played by the complete phase-aware candidate.

This captures interactions between experts and the selector: improving an
early expert is useful only in the context of the middle and late experts it
hands the game to. Keeping personalities in separate searches preserves a
clear v3 predecessor, candidate identity, and promotion decision.

### Nine independent four-locus searches

This would reuse the v3 search shape directly, but independently strong
experts need not form a strong composite policy. It would also evaluate each
expert outside the selection frequencies and game states it sees when
released.

### One joint thirty-six-locus search

Searching all personalities together could model interactions when they play
one another, but it couples three separately registered bots and makes a
personality-specific held-out comparison difficult to interpret. The
development corpus already supplies stable opponents.

### Search the phase boundaries

Learned boundaries add only two values, but they greatly expand the
opportunity to overfit development diagnostics and make selection less
explainable. Fixed thirds provide a clear first phase-aware generation. A
future version can propose boundary search with a separately justified
budget and new identities.

### Select by turn number

The existing diagnostic phase is easy to reuse, but turns and policy calls are
an indirect measure of remaining game. Public resource accounting expresses
the intended quantity directly and works without adding history as a policy
input.

## Runtime design

Add an immutable `PhaseAwareHeuristicProfile` containing:

- the canonical personality name;
- an early `HeuristicProfile`;
- a middle `HeuristicProfile`;
- a late `HeuristicProfile`; and
- the fixed phase-selector version.

All three experts must have the same canonical personality name. The profile
validates every coefficient with the existing `HeuristicProfile` rules and
does not permit custom thresholds.

The phase-aware brain performs one public resource-accounting pass, calls the
pure selector, and evaluates legal bids with the selected ordinary expert.
Reveal behavior continues to use the existing public reveal policy. The
explanation-aware path returns the selected expert phase together with the
ordinary heuristic bid explanation; it does not call the policy twice.

Candidate and released factories remain top-level or `functools.partial`
callables so multi-worker simulation can pickle them. A composite candidate's
name and simulation bot ID are identical and include a digest of all twelve
coefficients plus the selector version.

`HeuristicProfileSet` and the existing `LATEST_HEURISTICS`,
`AGGRESSIVE_PROFILE`, `BALANCED_PROFILE`, and `PASSIVE_PROFILE` aliases remain
scalar-profile APIs pinned to v3 for compatibility. V4 adds
`PhaseAwareHeuristicProfileSet` and the accurately named
`LATEST_HEURISTIC_POLICY_SET`, plus `AGGRESSIVE_POLICY`, `BALANCED_POLICY`,
and `PASSIVE_POLICY`. Those policy aliases, unversioned brains, unversioned
simulation specs, live wrappers, and default tournament entries advance to v4
only after all three promotions pass. This avoids pretending a
three-expert policy has the scalar coefficient attributes of one
`HeuristicProfile`.

## Search manifest schema v2

Existing schema-v1 v3 manifests, normalized payloads, digests, candidate
identities, frozen files, and catalog entries are immutable. Tests pin their
current bytes and hashes.

Schema v2 adds the minimum structure needed for a composite candidate:

- an immediate v3 predecessor name and identity;
- the selector name and fixed boundary formula;
- the pre-search v3 boundary-report path and SHA-256 digest;
- three named initial expert coefficient groups;
- the same four coefficient grids for every expert;
- a canonical locus order of early coefficients, then middle coefficients,
  then late coefficients; and
- explicit schema-v2 candidate and frozen-provenance types.

The manifest loader dispatches by schema version. It does not reinterpret a
v1 payload through v2 normalization. Both schemas reject unknown keys,
duplicate JSON keys, nonfinite values, held-out fields, and stale development
corpus bindings.

Each v4 search starts with all three experts equal to the matching released v3
profile. Generation zero contains that exact composite incumbent, one seeded
nonzero perturbation for each of the twelve loci, and one broader seeded
proposal for each phase. In later generations, each child changes exactly one
of the twelve loci by a nonzero number of allowed grid steps. Locus selection
is deterministic and stratified so every coefficient in every phase receives
comparable search coverage rather than relying on chance. The existing
deterministic `(mu + lambda)` selection, elite ordering, coefficient ranges,
decimal-grid arithmetic, and stable seed derivation remain in force.

## Search budget and fitness

Commit one schema-v2 manifest for aggressive, balanced, and passive. Each
manifest uses:

- twelve generations;
- sixteen evaluated proposals per generation;
- four elites;
- a four-step mutation radius; and
- the committed 240-case development corpus.

That is 192 composite proposals per personality and 576 total. The matching
v3 incumbent baseline is evaluated once per personality and reused for all
candidate comparisons. Each personality therefore executes 46,320
development games: 192 candidate evaluations times 240 cases, plus 240
incumbent games. The complete search budget is 138,960 games. This doubles the
v3 proposal budget; deterministic stratification supplies coverage across the
threefold increase from four to twelve loci while keeping the run bounded and
reviewable.

Fitness and eligibility remain the v3 search rules:

1. descending candidate-minus-incumbent Plackett-Luce rating;
2. descending candidate-minus-incumbent normalized-finish sum;
3. descending candidate-minus-incumbent final-money sum;
4. ascending canonical twelve-coefficient tuple; and
5. ascending candidate identity.

Candidate faults make that candidate ineligible. Missing games, identity or
case mismatches, and opponent or incumbent faults invalidate the generation.
A winner freezes only when it is complete, fault-free, and has a strictly
positive development rating difference.

After selection, the one winner-only diagnostics run per personality adds 240
candidate games, or 720 diagnostic games total. It uses the same development
corpus and cannot change selection.

## Search artifacts

Each complete search retains the existing authoritative artifact set:

- `search-manifest.json` — normalized schema-v2 recipe and digest;
- `search-report.json` — status, coverage, winner, failures, source commit,
  selector, and artifact names;
- `candidate-evaluations.jsonl` — all twelve coefficients, parent, coverage,
  faults, scores, eligibility, and ranking key for every proposal;
- `selection-log.jsonl` — every generation's ranked pool and elites;
- `development-games.jsonl` — reusable v3 baseline followed by composite
  candidate games in stable order;
- `development-corpus-snapshot.json` — the exact normalized development
  corpus; and
- `frozen-candidate.json` — present only for an eligible development winner.

The winner-only diagnostic generation adds:

- `game-summaries.jsonl`;
- `decision-traces.jsonl`;
- `decision-slices.csv`;
- `summary.json`; and
- `report.html`.

The search report points to the diagnostic generation and stores its artifact
digests. Reporting stays canonical, finite, deterministic, and transactional.
All files are rendered and validated before the prior known generation is
replaced; a replacement failure restores the previous generation and
preserves unrelated files.

## Frozen v4 candidates

A schema-v2 frozen candidate records:

- explicit composite identity, personality, and v3 predecessor;
- selector version and exact phase rule;
- early, middle, and late expert coefficients;
- full composite profile digest;
- search name and manifest digest;
- development corpus name and digest;
- selected generation, slot, and development scores;
- search report, evaluation-record, selection-log, game-evidence, and
  winner-diagnostics digests; and
- the repository commit used for search and diagnostics.

The catalog supports schema-v1 v3 candidates and schema-v2 v4 candidates
without changing any existing v1 entry or hash. Loading rejects unknown keys,
schema/content disagreement, an invalid selector, missing experts, wrong
coefficient order, a digest mismatch, identity/content disagreement, the
wrong predecessor, or incomplete provenance.

Promotion accepts a phase-aware candidate only through this frozen catalog.
It cannot accept an arbitrary profile, manifest, diagnostic directory, or
candidate file supplied by the caller.

## Common held-out promotion gate

The three frozen candidates use the same committed held-out corpus, paired
planning, fault mode, Plackett-Luce fit, whole-pair bootstrap, and positive
95% lower-bound rule already used for v3.

Each candidate is compared only with the matching released v3 bot:

- aggressive v4 candidate versus aggressive v3;
- balanced v4 candidate versus balanced v3; and
- passive v4 candidate versus passive v3.

Each comparison runs 480 matched cases, 960 games, and 1,000 requested
bootstrap replicates. The three-comparison held-out budget is 2,880 games.
The promotion report additionally binds the selector, all three expert
profiles, schema-v2 freeze, search evidence, winner diagnostics, and unchanged
development corpus.

Held-out output is never fed back into search or boundary selection. A failed
comparison is preserved and is not retuned against the same held-out corpus.
Only after all three reports independently say `promoted: true` may one
release commit define `HEURISTIC_V4`, register the three versioned v4 bots,
and advance every latest/unversioned heuristic alias together.

## Testing strategy

Use strict test-driven development.

Unit tests cover:

- exact selector boundaries for the canonical 10+/5–9/0–4 ranges and
  non-canonical totals;
- rejection of invalid totals and remaining-resource counts;
- proof that currently offered resources are excluded;
- deterministic selection from equivalent public states regardless of seat,
  history, private cards, or metadata;
- schema-v2 exact keys, canonical payloads, decimal grids, locus order, and
  development-only corpus binding;
- byte and digest snapshots proving every schema-v1 manifest and frozen v3
  candidate remains unchanged;
- generation zero, seeded samples, one-locus mutation, elite selection, and
  identity digests across all twelve loci;
- phase-aware brain agreement with a direct ordinary heuristic brain using
  the selected expert;
- picklable composite factories and identical scalar, batch, serial, and
  multi-worker decisions;
- explanation/action agreement and exactly one policy evaluation;
- separate legacy `game_phase` and nullable `selected_expert_phase`
  dimensions;
- winner-only diagnostic selection counts, phase-sliced outcomes, and full
  reconciliation;
- complete, canonical, transactional search and diagnostic artifacts;
- schema-v2 frozen catalog tamper detection and strict promotion provenance;
  and
- all-or-nothing v4/latest alias movement.

Integration tests cover:

- deterministic search reruns across batch sizes and worker counts;
- complete development coverage and exact evidence accounting;
- frozen candidates on charts A–E and three-, four-, and five-player games;
- zero illegal actions and zero bot faults;
- v1, v2, and v3 identities, coefficients, factories, registry entries, and
  representative decisions remaining unchanged;
- explicit v4 candidates remaining outside the released registry before
  promotion; and
- the held-out gate remaining the only authority that can justify a v4
  release.

Final verification runs the complete test suite, Ruff formatting and lint,
core and neural strict mypy, lock-file validation, documentation-link checks,
command-line help smoke tests, deterministic search reruns, all three
winner-only diagnostic runs, all three held-out comparisons, and
`git diff --check`.
