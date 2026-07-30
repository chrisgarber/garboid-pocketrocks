# Issue 8 Repository Cleanup Design

## Context

Issue #8 asks for a simpler, better-named, better-documented repository before
the bot-improvement portfolio expands it. The current codebase is healthy:
tests, lint, and strict type checks pass, and deterministic simulator,
tournament, heuristic, and neural behavior is extensively covered. The cleanup
must preserve that behavior while making the next set of changes easier to
understand and review.

The main problems are structural rather than functional:

- the root README duplicates stale neural instructions and has no concise
  documentation index;
- historical Superpowers plans dominate the documentation tree even though
  many describe removed APIs and completed work;
- runnable neural configurations contain settings that the trainer parses but
  silently ignores;
- `garboid-train smoke` selects between two unrelated implementations based on
  optional flags, and `garboid-train evaluate` only inspects metadata;
- several simulator and neural routines combine an entire state machine in one
  function;
- internal names drift between `ruleset_name`, `value_chart`, evaluation,
  inspection, games, and summaries;
- compatibility aliases and legacy paths remain without active callers.

## Goals

1. Make the current architecture and supported commands discoverable from a
   short root README and a domain-oriented documentation index.
2. Keep only documentation that is current, reproducible, or records an
   enduring architectural decision.
3. Reject unsupported runtime settings before work begins instead of accepting
   and ignoring them.
4. Remove the obsolete Stage 1 smoke execution path and leave one deterministic
   smoke contract backed by the production trainer.
5. Use names that describe actual behavior and put compatibility translation at
   serialization boundaries.
6. Extract focused helpers from the largest state machines without changing
   seeded decisions, legality, rewards, faults, replay data, checkpoints, or
   tournament output.
7. Leave explicit extension points for promotion reports, diagnostic traces,
   heuristic search, neural leagues, architecture comparisons, and learned
   search.

## Non-goals

- Change bot strategy, heuristic coefficients, neural weights, action
  selection, rewards, or tournament ranking.
- Rename or mutate an existing bot identity, checkpoint directory, manifest,
  benchmark artifact, or released alias.
- Implement issues #9 through #20 as part of cleanup.
- Introduce speculative abstractions for work that has not yet been designed.
- Preserve undocumented legacy CLI behavior that has no current supported
  consumer.

## Repository documentation

The root `README.md` becomes a concise product and architecture entry point. It
documents supported setup and top-level commands, then links to a new
`docs/README.md` and focused domain runbooks.

Add documentation only where a domain has a distinct contract or operational
workflow:

- `docs/README.md`: documentation map and authority rules;
- `src/garboid_pocketrocks/simulator/README.md`: deterministic execution,
  replay, batching, and fault behavior;
- `src/garboid_pocketrocks/tournament/README.md`: schedules, rating semantics,
  artifacts, and reproduction;
- `src/garboid_pocketrocks/heuristics/README.md`: public-information boundary,
  immutable generations, and valuation components;
- retain and update `src/garboid_pocketrocks/neural/README.md` as the neural
  training and checkpoint runbook;
- `docs/architecture/` for short enduring decisions about SDK authority,
  public information, identity immutability, and deterministic evaluation.

Completed implementation plans and superseded specifications under
`docs/superpowers/` are removed after their enduring decisions and live links
are migrated. Git history remains the archive. Current issue #8 design and plan
stay in the branch while the work is under review.

Benchmark reports, machine-readable tournament artifacts, reproduction
runbooks, and the bot-versioning skill remain.

## Runtime configuration

Configuration parsing and historical checkpoint reading remain backward
compatible. Runtime execution gains a separate support validation step that
rejects any currently ignored non-default feature:

- interval checkpoint retention;
- start, periodic, or final evaluation;
- nonzero checkpoint-league mixing.

Committed runnable profiles use only supported settings. Future issues may
activate these fields by replacing the support validation with real behavior.
Old serialized run configurations remain readable so historical artifacts do
not become corrupt.

## CLI and legacy removal

`garboid-train smoke` has one meaning: run the current deterministic A-E,
three-to-five-player training smoke through the production trainer. The legacy
flag-triggered Stage 1 route, duplicate collector/result/checkpoint plumbing,
and tests that exist only for that route are removed.

`garboid-train inspect` describes checkpoint metadata inspection. The
misleading `evaluate` spelling is removed from help and command dispatch.
Actual strength evaluation will be introduced by issue #9 rather than hidden
behind an inspection command.

Shared production modules lose obsolete "Stage 1" descriptions. Public command
documentation and CLI tests pin the remaining semantics.

## State-machine extraction

Refactors follow existing behavior and data structures. They introduce helpers
only where a block has one stable responsibility.

### Simulator batch execution

Split batch setup, pending-decision construction, bid resolution, reveal
resolution, history updates, fault handling, and result construction into
named helpers. Keep `run_batch_matches` as the orchestration loop. Scalar/batch
parity, replay, faults, and deterministic results remain the acceptance
boundary.

### Neural collection and parallel coordination

Split vector collection into helpers for engine initialization, policy
inference requests, action application, transition/reward recording, episode
finalization, and metrics. Split parallel collectors into process lifecycle,
request/result validation, and ordered aggregation helpers. Preserve policy
snapshot timing, seed derivation, ordering, and metrics.

### PPO and belief construction

Extract the PPO update's validation, minibatch iteration, loss accounting, and
summary construction without changing formulas or random-number consumption.
Extract belief input validation, public-card accounting, and posterior
construction while preserving exact distributions and exceptions.

## Naming and compatibility

Canonical names describe the domain:

- `value_chart` is the SDK chart code (`"A"` through `"E"`);
- `ruleset_name` is retained only where an existing serialized artifact or SDK
  boundary requires `"live-A"` form;
- checkpoint metadata inspection is called `inspect`;
- canonical result fields are `game_summaries` and `bot_statistics`.

Unused `MonteCarloResult.games` and `.statistics` aliases are removed after
repository and packaging checks confirm no supported callers. Immutable JSON
field names and checkpoint schemas are not rewritten; conversion happens when
loading or writing those formats.

## Error handling

The cleanup favors explicit, early failures:

- unsupported configuration fails before training creates an output directory;
- malformed compatibility data reports the exact field and artifact;
- batch and parallel helpers preserve the existing fault mode rather than
  swallowing errors;
- serialization stays strict, deterministic, and atomic.

No fallback is added merely to keep a refactor running.

## Testing strategy

Every extraction starts from characterization tests where current coverage is
not already exact. The integrated gate requires:

- full core and neural pytest suites;
- Ruff and both strict mypy configurations;
- deterministic scalar/batch and serial/parallel equality;
- byte-equivalent replay and tournament artifacts where currently guaranteed;
- identical frozen neural actions for fixed inputs and checkpoints;
- existing checkpoint load and resume compatibility;
- CLI routing and help tests for the single smoke and inspect commands;
- runtime support-validation tests for every ignored setting;
- a documentation link check with no broken local links;
- searches confirming removed legacy APIs and historical documents have no
  live references.

## Delivery

All cleanup lands on `codex/issue-8-cleanup`, based on refreshed
`origin/main`, as a draft pull request linked to issue #8. Commits are grouped
by independently reviewable workstream: documentation, configuration/CLI,
simulator extraction, neural extraction, and final naming/dead-code cleanup.
Issue #8 closes only after the full integrated gate passes and the draft PR
contains evidence for the issue's naming, simplification, and documentation
requirements.
