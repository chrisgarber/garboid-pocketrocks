# Garboid PocketRocks Repository Scaffold Design

**Date:** 2026-07-28
**Status:** Approved for planning

## Purpose

Create a public, MIT-licensed Python repository at
`chrisgarber/garboid-pocketrocks` for developing PocketRocks bots that can run
against the live service and compete locally in a future Monte Carlo simulator.

The first implementation milestone is repository scaffolding only. It establishes
the package boundaries, development tooling, continuous integration, dependency
lock, and documentation needed by later milestones without implementing bot or
game behavior prematurely.

## Goals

- Create a modern Python package using the latest stable CPython minor release,
  Python 3.14.
- Use mise to pin developer tools and uv to manage the Python project,
  environment, dependencies, and lockfile.
- Pin the PocketRocks Python SDK to an exact source commit because the SDK has no
  stable release tag.
- Establish clear namespaces for bot policies, live-service adapters, simulation,
  and training.
- Document the intended architecture and incremental project roadmap.
- Enforce formatting, linting, strict type checking, and tests in GitHub Actions.
- Keep secrets and generated local artifacts out of version control.

## Non-Goals

This milestone does not:

- implement a random bot;
- connect to the PocketRocks service;
- define the normalized bot policy protocol;
- implement game rules or a simulator;
- implement value heuristics;
- add machine-learning libraries or a neural policy;
- copy or depend on the competition repository's simulator.

Each of those behaviors will receive its own focused design and implementation
cycle.

## Architectural Direction

The project will use a shared policy core with adapters.

```text
Live server -> SDK DecisionContext -> SDK adapter
                                         |
                                  normalized game view
                                         |
                                    bot policy
                                         |
                                    policy action
                                         |
Live server <- SDK BotDecision <- SDK adapter
```

The future simulator will produce the same normalized game view and consume the
same policy action. Training will produce policies satisfying that contract. A
strategy can therefore run locally or against the live service without duplicating
its decision logic.

The normalized policy types will be designed with the random-bot milestone, when
they can be tested against concrete SDK contexts. The scaffold creates importable
namespace packages but does not speculate about those interfaces.

The competition repository's current engine and simulator are rules and
conformance references, not runtime dependencies. Our simulator will be
independently structured around the shared policy contract.

## Repository Structure

```text
garboid-pocketrocks/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-07-28-repository-scaffold-design.md
├── src/
│   └── garboid_pocketrocks/
│       ├── adapters/
│       │   └── __init__.py
│       ├── bots/
│       │   └── __init__.py
│       ├── simulator/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       └── __init__.py
├── tests/
│   └── test_package.py
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── mise.toml
├── pyproject.toml
└── uv.lock
```

### Namespace Responsibilities

- `adapters`: translate between external systems and the shared policy contract,
  beginning with the PocketRocks Python SDK.
- `bots`: reusable policy implementations, beginning with a random baseline and
  followed by value-heuristic and learned policies.
- `simulator`: deterministic game rules, match execution, Monte Carlo evaluation,
  and conformance fixtures.
- `training`: data generation, model definitions, training loops, checkpoints,
  and evaluation utilities.

## Tooling and Dependencies

### Python and mise

`mise.toml` will select the Python 3.14 release line and uv. Selecting the minor
line allows new stable Python 3.14 patch releases without opting into Python 3.15
prereleases.

Mise is responsible for developer runtime and tool selection. It does not replace
uv's project-level dependency resolution and locking.

### uv and Project Metadata

`pyproject.toml` will:

- define the `garboid-pocketrocks` distribution;
- use a `src` package layout;
- require Python `>=3.14,<3.15`;
- declare the SDK from its Git repository at commit
  `597857446d47ac0890609a4767cad561578a2519`;
- define development dependencies for pytest, Ruff, and mypy;
- configure pytest, Ruff, and strict mypy checks.

`uv.lock` will be committed and treated as generated dependency state. Local and
CI setup will use `uv sync --locked`.

### SDK Pin

The SDK dependency will use:

```text
git+https://github.com/jaiparera/pocketrocks-python-sdk.git@597857446d47ac0890609a4767cad561578a2519
```

The source repository's default branch is currently `develop` and exposes no
stable tag. Pinning the commit makes the scaffold reproducible while allowing an
explicit upgrade later.

No dependency on `jaiparera/pocketrockscompetition` will be added.

## Documentation

The README will include:

- the project's purpose;
- the shared-policy architectural direction;
- prerequisites and mise/uv setup commands;
- quality-check commands;
- a phased roadmap:
  1. repository scaffold;
  2. random bot with live SDK adapter;
  3. local simulator and Monte Carlo runner;
  4. strategy brainstorming and value-heuristic bots;
  5. round-robin evaluation;
  6. neural policy and local training;
- links to the competition, SDK, and live game.

The root `LICENSE` will contain the MIT license with copyright year 2026 and
copyright holder Christopher Garber.

## Secrets and Generated Artifacts

`.env.example` will list the supported live-service variables without values:

- `POCKETROCKS_API_KEY`
- `POCKETROCKS_BOT_ID`
- `POCKETROCKS_SERVER_URL`

`.gitignore` will exclude `.env`, virtual environments, Python caches, tool
caches, build output, coverage output, local training data, and model
checkpoints. It will not exclude source fixtures or intentionally committed small
test assets.

## Testing and Continuous Integration

The scaffold test will verify that:

- `garboid_pocketrocks` imports;
- each planned namespace imports;
- the package exposes its initial version.

GitHub Actions will run on pushes and pull requests. It will install Python 3.14
and uv, synchronize from the committed lockfile, then run:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

CI uses the same locked environment and commands documented for local
development.

## Future Error-Handling Principles

There is no runtime error handling in the scaffold because it has no game
behavior. Later milestones will follow these principles:

- validate strategy actions at adapter boundaries;
- fail safely in the live adapter within server deadlines;
- reject invalid simulator actions with explicit domain errors instead of
  silently coercing them;
- inject and record random seeds for reproducible simulations;
- surface dependency or protocol mismatches clearly during tests.

## Implementation and Publication

The work will be committed in two reviewable steps:

1. this approved design specification;
2. the verified repository scaffold.

After local verification:

- rename the default local branch to `main`;
- create the public GitHub repository `chrisgarber/garboid-pocketrocks`;
- set its description and MIT license metadata;
- push `main`.

GitHub CLI authentication for `chrisgarber` must be refreshed before remote
creation. No pull request is needed for the initial repository publication.

## Acceptance Criteria

The scaffold is complete when:

- the documented repository structure exists;
- Python 3.14 and uv are selectable through mise;
- `uv sync --locked` succeeds from a clean environment;
- format, lint, strict type checking, and tests pass locally;
- the README documents setup, architecture, and roadmap;
- secret and generated-file exclusions are present;
- the public MIT-licensed GitHub repository exists under `chrisgarber`;
- the verified `main` branch is pushed to GitHub.

## References

- [PocketRocks competition](https://github.com/jaiparera/pocketrockscompetition)
- [PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk)
- [PocketRocks](https://pocketrocks.xyz/)
- [Python downloads](https://www.python.org/downloads/)
- [uv project management](https://docs.astral.sh/uv/guides/projects/)
- [mise Python and uv integration](https://mise.jdx.dev/lang/python.html#mise-uv)
