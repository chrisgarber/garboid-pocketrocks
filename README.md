# Garboid PocketRocks

Bots, simulation, evaluation, and local training for
[PocketRocks](https://pocketrocks.xyz/).

This project builds bots for
[jaiparera/pocketrockscompetition](https://github.com/jaiparera/pocketrockscompetition)
and connects them to the live service through the
[PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk).

## Architecture

Bot strategies will implement one shared policy contract:

```text
Live server -> SDK adapter -> normalized game view -> policy -> action
Simulator -----------------> normalized game view -> policy -> action
```

The live adapter and simulator will share policy implementations, allowing the
same random, heuristic, and learned bots to run in both environments. The
competition repository's simulator is a rules reference rather than a runtime
dependency.

## Requirements

- [mise](https://mise.jdx.dev/)
- Git

Mise installs the Python 3.14 release line and uv version declared by this
repository. uv manages the virtual environment and locked dependencies.

## Setup

```bash
mise install
uv sync --locked
```

Live service credentials will eventually be read from `.env`. Start from the
committed variable names:

```bash
cp .env.example .env
```

Do not commit `.env`.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Roadmap

1. Establish the repository scaffold and quality gates.
2. Build a random baseline bot and connect it through the Python SDK.
3. Implement a deterministic game engine and Monte Carlo match runner.
4. Design and implement value-heuristic bot strategies.
5. Run seeded round-robin evaluations and compare strategies.
6. Build and locally train a neural policy.

Each milestone will be designed and tested independently.

## License

Garboid PocketRocks is available under the [MIT License](LICENSE).
