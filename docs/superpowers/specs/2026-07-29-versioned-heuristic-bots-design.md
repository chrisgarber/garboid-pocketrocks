# Versioned Heuristic Bots Design

## Goal

Preserve every substantial heuristic generation so simulations can compare
bot strength over time, while keeping existing commands and live deployments
on the latest generation.

The initial history contains:

- v1: the original liquidity-based heuristic;
- v2: the calibrated future-cash heuristic.

## Versioning contract

Each generation is an immutable set of aggressive, balanced, and passive
profiles. A released generation's coefficients, names, and behavior must not
change except when the user explicitly chooses to update that version in
place.

The public naming contract is:

| Name | Meaning |
|---|---|
| `aggressive-v1`, `balanced-v1`, `passive-v1` | frozen original generation |
| `aggressive-v2`, `balanced-v2`, `passive-v2` | frozen future-cash generation |
| `aggressive`, `balanced`, `passive` | aliases for the latest generation |

Unversioned aliases are compatibility entry points, not additional
generations. They must make the same decisions as the corresponding latest
version in identical contexts.

## Profile model

Add a frozen `HeuristicProfileSet` containing:

- a canonical version label;
- aggressive, balanced, and passive `HeuristicProfile` values.

Export `HEURISTIC_V1`, `HEURISTIC_V2`, and `LATEST_HEURISTICS`.

V1 uses the original coefficients and disables the later reserve term:

| Profile | Liquidity | Future cash | Objective progress | Bid shading |
|---|---:|---:|---:|---:|
| aggressive-v1 | 0.75 | 0.00 | 0.25 | 0.05 |
| balanced-v1 | 0.40 | 0.00 | 0.20 | 0.25 |
| passive-v1 | 0.15 | 0.00 | 0.15 | 0.50 |

Because a zero future-cash weight removes the new term exactly, the shared
valuator reproduces the original algorithm without copying implementation
code.

V2 freezes the calibrated coefficients:

| Profile | Liquidity | Future cash | Objective progress | Bid shading |
|---|---:|---:|---:|---:|
| aggressive-v2 | 0.75 | 1.50 | 0.25 | 0.05 |
| balanced-v2 | 0.40 | 0.75 | 0.20 | 0.25 |
| passive-v2 | 0.15 | 0.60 | 0.15 | 0.30 |

The existing `AGGRESSIVE_PROFILE`, `BALANCED_PROFILE`, and `PASSIVE_PROFILE`
exports remain latest aliases.

## Brains and bot specifications

Provide explicit brain and bot classes for every version and personality.
Versioned bot classes receive stable versioned simulation names and IDs.
Their factories remain top-level and pickleable so multiprocessing simulation
continues to work.

The existing unversioned brain, bot, and `BotSpec` exports remain available.
They use the latest profiles and retain their current live bot IDs and names.

Python callers can select either explicit versioned specs or latest aliases.
The simulation CLI registers all six explicit version names plus the three
latest aliases. Its help text lists the supported names.

The live launcher continues to start only:

- `random`;
- `aggressive`;
- `balanced`;
- `passive`.

Historical bot wrappers are simulation and Python API surfaces; they are not
started as additional live connections.

## Repository skill

Create `.agents/skills/versioning-bots/` as a repository-local skill. It
triggers before substantial changes to any bot, including:

- strategy or valuation logic;
- coefficients, thresholds, or action-selection behavior;
- information inputs that can change decisions;
- learned policy, training, or default checkpoint changes;
- bug fixes that materially change decisions or strength.

Pure behavior-preserving refactors, adapters, tests, documentation, and
observability changes do not trigger the version question.

Before implementing a triggering change, the skill asks the user to choose:

1. create a new version and preserve the current behavior; or
2. update the current version in place.

When a new version is chosen, the workflow freezes the old artifacts, creates
explicit versioned names, advances latest aliases, and adds reproducibility
tests and a benchmark record. When an in-place change is chosen, it records
that choice in the task context and updates the pinned tests deliberately.

The skill is concise and self-contained. It includes `agents/openai.yaml` for
discovery metadata but requires no scripts or reference assets.

## Tests

Use test-driven development.

Profile tests pin:

- both version labels;
- every coefficient in v1 and v2;
- latest aliases to v2;
- rejection of invalid or empty version sets.

Bot tests pin:

- stable names and IDs for every explicit version;
- each bot factory to the correct profile;
- unversioned decisions equal v2 decisions;
- versioned specs remain pickleable and deterministic.

CLI tests require all explicit and unversioned names to parse and verify that
the live launcher registry remains latest-only.

A deterministic mixed-generation simulation exercises multiprocessing and
confirms zero bot faults.

The repository skill is developed with pressure scenarios:

- a coefficient adjustment presented as "small";
- a strategy bug fix that changes decisions;
- a behavior-preserving refactor.

The first two must cause a versioning question before implementation. The last
must proceed without an unnecessary question. Validate the final skill folder
with the standard skill validator.

## Compatibility

- Existing imports and CLI commands continue to work.
- Existing live bot IDs do not change.
- Neural training continues to consume the latest unversioned bot specs.
- Replays and statistics distinguish explicit historical versions by name.
- The simulator, game engine, and SDK adapters remain unchanged.

## Non-goals

- Automatically deciding whether to create a new version.
- Launching historical generations against the live server.
- Copying the valuation engine for each generation.
- Migrating old benchmark documents or replay payloads.
