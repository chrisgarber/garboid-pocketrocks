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
- [Heuristics](../src/garboid_pocketrocks/heuristics/README.md): public belief,
  valuation, profiles, and released generations.
- [Neural](../src/garboid_pocketrocks/neural/README.md): self-play training,
  checkpoints, resume, and inspection.

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
- [`docs/benchmarks/tournaments`](benchmarks/tournaments/) contains the
  machine-readable CSV, JSON, and HTML artifacts referenced by tournament
  reports.
- [`docs/benchmarks/promotions`](benchmarks/promotions/) contains immutable
  promotion reports and paired game summaries referenced by dated promotion
  notes.

Reports and artifacts are historical evidence. Do not silently rewrite them
when current behavior changes; add a newly dated result instead.

## Review artifacts

The current [issue #8 design](superpowers/specs/2026-07-30-issue-8-repository-cleanup-design.md)
and [implementation plan](superpowers/plans/2026-07-30-issue-8-repository-cleanup.md)
remain while the cleanup is under review. They are review artifacts, not
operational documentation. Completed implementation transcripts are archived
by Git history rather than kept in the live documentation tree.
