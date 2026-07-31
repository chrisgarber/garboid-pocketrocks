---
name: versioning-bots
description: Use when changing bot strategy, decisions, coefficients, training, checkpoints, information inputs, action selection, fixing a bug that materially changes bot behavior or strength, or deciding which training and evaluation artifacts belong in Git
---

# Versioning Bots

## Overview

Preserve bot generations deliberately. Before a substantial behavior change,
let the user choose between creating a new version and updating the current
version in place.

## Classify the change

A change is substantial when the same game state could produce a different
decision, or when a learned bot's policy or expected strength changes. This
includes coefficients, thresholds, strategy, information use, training,
checkpoints, and behavior-changing bug fixes.

Do not interrupt behavior-preserving refactors, adapters, tests,
documentation, or observability work. Verify the behavior-preserving claim
with existing or new tests.

## Retain identities, not run exhaust

Write complete training, evolution, tournament, and promotion outputs under
the gitignored `artifacts/` tree. Treat per-game logs, replay streams,
candidate-evaluation streams, bootstrap samples, repeated corpus snapshots,
and failed-run receipts as ephemeral working data.

Commit only what is needed to reproduce or run a released generation:

- versioned code, manifests, and corpus recipes;
- frozen coefficients or a released checkpoint and its compact manifest;
- a concise dated benchmark note with the source commit, seeds, digests,
  aggregate coverage, uncertainty, faults, and decision.

Before staging a training change, inspect `git diff --stat`, `git diff
--numstat`, and the output sizes. If generated receipts dominate the change,
remove them from Git and keep their reproduction command in the benchmark
note. Do not preserve a zero-game operational failure merely to prove it
happened.

Commit a raw receipt only when the user explicitly requests it or when a
minimal machine-readable artifact is required at runtime. Record that reason
in the benchmark note.

## Required decision

Before design or implementation, check whether the user already chose a
versioning policy in the current task. Count it as a choice only when the user
explicitly says to create a named/new version, or to update a named/current
version in place without preserving its old behavior. Phrases such as "patch
the current bot" or "fix it now" do not resolve the preservation choice.

If no explicit choice exists, ask:

> This can materially change `<bot>` behavior. Should I preserve the current
> behavior as `<version>` and create the next version, or update `<version>`
> in place?

Stop implementation until the user answers. A request to move quickly, skip
design, or make a "small" change is not a versioning choice.

| Choice | Required result |
|---|---|
| New version | Freeze old artifacts; add explicit versioned simulation names; reserve `BOT_ID` for wrappers that connect remotely; advance unversioned latest aliases; add reproducibility tests; benchmark against the preceding version |
| In place | Record the choice in task context; update pinned behavior tests deliberately; keep the version label |
| Behavior-preserving | Proceed without asking; verify decisions remain unchanged |

## New-version checks

- Keep shared engines shared; version configuration or policy artifacts
  instead of copying implementation unless the algorithm cannot reproduce the
  old behavior.
- Keep released version names, coefficients, checkpoints, and any real remote
  bot IDs immutable.
- Use the versioned name as the identity for local-only simulation specs.
  Define `BOT_ID` only on wrappers that can connect to the remote service.
- Make historical versions selectable in simulation and Python APIs.
- Keep live launchers on latest aliases unless the user requests otherwise.
- Report cross-generation strength using fixed seeds and committed benchmark
  settings.

## Red flags

Pause if any of these thoughts appear:

- "It is only one coefficient."
- "It is a bug fix, not a new strategy."
- "The user said implement immediately."
- "I can decide the version policy for them."
- "I will ask after the code works."

Each indicates a material decision may be hidden by time pressure. Ask before
editing.

## Common mistakes

- Treating a tiny diff as behavior-preserving: classify effects, not line
  count.
- Versioning a rename: do not ask when decisions demonstrably stay identical.
- Moving latest aliases without preserving the old public entry points.
- Calling a generation stronger without a fixed-seed comparison.
