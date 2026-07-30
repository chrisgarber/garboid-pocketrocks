# Heuristic Evolution Design

## Purpose

The current heuristic bots use four understandable coefficients chosen by
hand. This project searches those same four knobs automatically on development
games, records every trial and selection, freezes the best candidates, and
then uses the existing held-out promotion gate as the only route to heuristic
v3.

The search changes no formulas, information inputs, reveal policy, or released
identity. Heuristic v1 and v2 remain immutable and selectable.

## Plain-language contract

- A **search manifest** is the fixed recipe for one personality's search.
- A **candidate** is one proposed setting of the existing four coefficients.
- A **generation** is one batch of candidates evaluated on the same
  development games.
- An **elite** is a top-ranked candidate allowed to parent the next
  generation.
- A **frozen candidate** is the selected result with a content-addressed
  identity and complete search provenance.
- **Promotion** is the separate held-out final exam. Development performance
  alone never creates v3 or moves a latest alias.

## Versioning decision

The issue already specifies the behavior-change path:

1. Keep `HEURISTIC_V1`, `HEURISTIC_V2`, their brains, identities, and
   decisions unchanged.
2. Search produces local identities such as
   `balanced-v3-candidate-4f91c7d2a6b8`; candidate name and simulation bot ID
   are identical.
3. Search candidates live in a separate frozen catalog, not the released bot
   registry, default tournament lineup, or live wrappers.
4. Each frozen personality is compared only with its matching v2 predecessor
   on the existing held-out corpus.
5. Create `HEURISTIC_V3` and advance latest aliases only if the intended v3
   profile set passes the common promotion rule. Unchanged personalities may
   be carried forward byte-for-byte from v2.

Remote `BOT_ID` constants never change.

## Approaches considered

### One personality per manifest with mutation-only evolution — selected

Each personality is one four-coefficient `HeuristicProfile`, and promotion
compares one candidate with one predecessor. Searching personalities
independently makes every result attributable and directly promotable.

The algorithm is deterministic `(mu + lambda)` evolution: retain the best
parents and generate a fixed number of mutated children. It is simple enough
to reproduce from the record without a machine-learning framework.

### Joint twelve-coefficient profile-set search

This can capture interactions among aggressive, balanced, and passive bots,
but produces a coupled result that the current one-candidate promotion gate
cannot evaluate cleanly. It also makes it harder to explain which behavior
improved.

### Random search, exhaustive grid, or CMA-ES

Random search does not meet the evolutionary goal, the full grid is too
large, and CMA-ES adds an opaque continuous optimizer and dependency. A
quantized mutation algorithm is smaller and easier to verify.

## Package boundaries

Create `src/garboid_pocketrocks/evolution/`:

- `manifest.py` strictly loads, normalizes, and hashes fixed search recipes.
- `candidates.py` creates deterministic genomes, identities, and picklable
  `BotSpec` factories.
- `planning.py` builds development-only matched jobs.
- `evaluation.py` validates game evidence and computes fitness.
- `search.py` proposes generations and selects elites.
- `reporting.py` owns the authoritative report and transactional artifacts.
- `runner.py` executes complete generations.
- `cli.py` exposes `garboid-evolve-heuristic`.

Create `src/garboid_pocketrocks/heuristics/frozen_candidates/` for the small
runtime catalog and immutable selected-candidate JSON files. Search trials
remain outside that catalog.

## Fixed manifests and coefficient ranges

Commit one versioned manifest for aggressive, balanced, and passive. A
manifest binds:

- schema version and stable search name;
- personality and exact v2 incumbent identity;
- development corpus name and SHA-256 digest;
- search seed;
- algorithm name, generation count, population size, elite count, and
  mutation radius;
- exact initial v2 coefficients;
- exact grid bounds and step for all four existing coefficient fields.

No other coefficient name is legal. Held-out paths, names, digests, or seeds
are forbidden.

The initial ranges are:

| Coefficient | Minimum | Maximum | Step |
| --- | ---: | ---: | ---: |
| liquidity strength | 0.00 | 1.50 | 0.05 |
| future-cash weight | 0.00 | 2.00 | 0.05 |
| objective-progress weight | 0.00 | 1.00 | 0.05 |
| bid shading | 0.00 | 1.00 | 0.05 |

Manifests store decimal strings. The loader uses `Decimal` to validate grids
and candidate generation, converting to `float` only when constructing the
existing `HeuristicProfile`.

Each initial manifest uses eight generations, twelve new candidates per
generation, four elites, and a four-step mutation radius. These settings
evaluate 96 candidates per personality while keeping the committed search
repeatable.

## Candidate generation

Generation zero contains:

1. a candidate with coefficients exactly equal to the v2 incumbent;
2. eleven seeded grid samples.

For every later generation:

1. parents cycle through the prior ranked elites;
2. a stable child seed is derived from the search seed, manifest digest,
   generation, and slot;
3. exactly one of the four coefficient fields is selected;
4. that coefficient moves by a nonzero integer number of grid steps within
   the configured mutation radius and range.

The selection pool is the previous elites plus the new children. Repeated
genomes remain separate recorded proposals. There is no crossover, adaptive
state, or result-dependent randomness beyond which recorded elites become
parents.

Candidate identities include personality, generation, slot, and the first
twelve hexadecimal characters of the canonical profile digest. The full
digest is recorded.

## Development evaluation

The search command accepts a development corpus only. It has no held-out
argument or automatic promotion step.

Each candidate plays the 240 committed development cases in the corpus focal
seat. Its matching v2 incumbent plays the exact twin cases with the same
engine seeds, charts, player counts, focal seats, and opponents. The incumbent
baseline is evaluated once and reused for every candidate.

Every simulation uses `FaultMode.RECORD_AND_PASS`. Candidate faults make that
candidate ineligible. Missing games, mismatched identities or cases, and any
opponent/incumbent fault invalidate the generation and fail the run.

Fitness is:

1. descending candidate-minus-incumbent Plackett-Luce rating;
2. descending candidate-minus-incumbent normalized-finish sum;
3. descending candidate-minus-incumbent final-money sum;
4. ascending canonical coefficient tuple;
5. ascending candidate identity.

The exact ranking key is recorded. The final candidate freezes only when it is
eligible, fault-free, and has a strictly positive development rating
difference. This is selection evidence, not promotion evidence.

## Frozen candidates

`frozen-candidate.json` contains:

- schema version, explicit candidate identity, personality, and predecessor;
- the four coefficients and full profile digest;
- search name and manifest digest;
- development corpus name and digest;
- selected generation, slot, and development scores;
- search report and evaluation-record SHA-256 digests;
- repository commit used for the search.

The catalog maps candidate identity to its package-relative file and file
SHA-256. Loading rejects unknown keys, a digest mismatch, identity/content
disagreement, wrong predecessor, invalid coefficients, or missing provenance.

Candidate brains are created by a top-level helper wrapped in
`functools.partial`, which is picklable for multi-worker simulation and still
constructs the ordinary `HeuristicBotBrain` with no new inputs.

## Search artifacts

One complete run writes:

- `search-manifest.json` — normalized manifest and digest;
- `search-report.json` — status, coverage, best result, failures, source
  commit, and artifact names;
- `candidate-evaluations.jsonl` — every candidate, coefficients, parent,
  coverage, faults, scores, eligibility, and ranking key;
- `selection-log.jsonl` — every generation's complete ranked pool and selected
  elites;
- `development-games.jsonl` — incumbent baseline followed by candidate game
  evidence in stable generation/slot/case order;
- `development-corpus-snapshot.json` — normalized expanded development corpus;
- `frozen-candidate.json` — present only for a complete, fault-free,
  development-improving winner.

The authoritative `SearchReport` owns all source evidence. Renderers accept
only that report, use canonical finite JSON with terminal newlines, and stage
the complete known generation before replacement. Replacement failures
restore the prior generation and preserve unrelated files.

The first implementation intentionally does not support resume. A fixed
manifest reruns deterministically, and avoiding a partially trusted checkpoint
format keeps the evidence boundary small.

## Promotion integration

The promotion CLI continues to resolve incumbents and corpus opponents only
from the released registry. Candidate resolution additionally checks the
committed frozen catalog. Arbitrary candidate file paths are not accepted.

For a frozen candidate, promotion requires:

- `--incumbent` equals the freeze's declared immediate predecessor;
- the development corpus name/digest equals the freeze;
- the candidate file and catalog digests match;
- the promotion report records the freeze, profile, search manifest, search
  record, and development-corpus digests.

The existing paired held-out planner and positive 95% bootstrap lower-bound
rule remain unchanged.

After search evidence is committed, each selected candidate receives one
authoritative held-out run against its matching v2 predecessor. A failure is
preserved and is not retuned against the same held-out corpus.

## Testing strategy

Use strict test-driven development:

- strict manifest schema, decimal-grid validation, corpus binding, and
  rejection of any held-out key;
- golden deterministic generation/parent/mutation sequences;
- picklable candidate factories and exact agreement with direct
  `HeuristicBotBrain` decisions;
- development-only paired planning and full 240-case coverage;
- deterministic evaluation across repeat runs, batches, and worker counts;
- fault/missing/mismatch rejection and exact fitness ordering;
- every proposal and selection accounted for;
- canonical byte-identical reporting and transactional rollback;
- frozen catalog tamper detection and promotion provenance;
- v1/v2 coefficient, identity, registry, and representative-behavior
  snapshots remain unchanged;
- frozen candidates complete charts A-E and three-, four-, and five-player
  games with no illegal actions or faults;
- held-out promotion remains the only code path that can justify v3/latest
  alias movement.

