# Documentation

The root [README](../README.md) is the quick start. Use this page to find the
current operational contract or the evidence behind it.

## Operational runbooks

Package READMEs are the current operational truth:

- [Simulator](../src/garboid_pocketrocks/simulator/README.md): deterministic
  matches, replay, batching, and faults.
- [Tournament](../src/garboid_pocketrocks/tournament/README.md): schedules,
  ratings, artifacts, and reproduction.
- [Promotion](../src/garboid_pocketrocks/promotion/README.md): matched
  held-out games, promotion decisions, and evidence.
- [Heuristic evolution](../src/garboid_pocketrocks/evolution/README.md):
  development-only coefficient search, transactional evidence, and frozen
  candidate handoff.
- [Heuristics](../src/garboid_pocketrocks/heuristics/README.md): public belief,
  valuation, profiles, and released generations.
- [Neural](../src/garboid_pocketrocks/neural/README.md): self-play training,
  checkpoints, resume, and inspection.
- [Public-belief search foundation](../src/garboid_pocketrocks/search/README.md):
  live-input reconstruction, deterministic belief sampling, and the SDK
  prerequisite for canonical tree transitions.

When a command or supported workflow changes, update its package runbook in
the same change.

## Architecture decisions

These documents contain enduring repository-wide rules:

- [SDK authority](architecture/sdk-authority.md)
- [Public-information boundary](architecture/public-information-boundary.md)
- [Immutable bot identities](architecture/immutable-bot-identities.md)
- [Deterministic evaluation](architecture/deterministic-evaluation.md)

Implementation details belong in code and tests, not in additional
architecture transcripts.

## Reproduction and evidence

- [`docs/analysis`](analysis/) contains reproduction runbooks. The
  [heuristic visualization runbook](analysis/heuristic-bot-visualizations.md)
  records its datasets, transformations, and provenance requirements.
- [`docs/benchmarks`](benchmarks/) contains dated benchmark reports.
- Tournament CSV, JSON, and HTML outputs are local analysis artifacts and are
  excluded from version control. Dated reports under
  [`docs/benchmarks`](benchmarks/) retain the summarized results and
  reproduction commands.
- Promotion and evolution conclusions live in concise dated benchmark notes.
  Their complete deterministic receipts stay in the gitignored `artifacts/`
  tree; versioned inputs and released frozen candidates or checkpoints remain
  in source control.

Benchmark notes and released artifacts are historical evidence. Do not
silently rewrite them when current behavior changes; add a newly dated result
instead. Raw per-game logs, bootstrap samples, repeated corpus snapshots, and
failed-run receipts are reproducible working data rather than documentation.

## Review artifacts

The current [issue #8 design](superpowers/specs/2026-07-30-issue-8-repository-cleanup-design.md)
and [implementation plan](superpowers/plans/2026-07-30-issue-8-repository-cleanup.md)
remain while the cleanup is under review. They are review artifacts, not
operational documentation. Completed implementation transcripts are archived
by Git history rather than kept in the live documentation tree.
