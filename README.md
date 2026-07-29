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

The API key in `.env` is secret. Bot IDs are public identifiers, so each bot's
ID can live in the committed template under a strategy-specific name.

## Random bot

`RandomBot` follows the
[official SDK starter](https://github.com/jaiparera/pocketrocks-python-sdk/tree/develop/starter)
and directly extends `PocketRocksBot`.

Its baseline strategy is intentionally uninformed:

- sample uniformly from every legal integer bid, treating zero as pass;
- sample uniformly from every revealable card index;
- pass when no bid or reveal is available.

Run it after filling in `.env`:

```bash
uv run garboid-random-bot
```

The command loads `RANDOM_BOT_ID` and passes it to the SDK. If that variable is
unset, the SDK's generic `POCKETROCKS_BOT_ID` setting remains available as a
fallback.

For a reproducible decision sequence:

```bash
uv run garboid-random-bot --seed 42
```

The automated tests use the SDK's in-memory `FakeTransport`, so they exercise
the complete SDK request/response path without connecting to the live service.
Starting the command above creates a real connection and may play games if the
configured bot is active.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Roadmap

1. ✅ Establish the repository scaffold and quality gates.
2. ✅ Build a random baseline bot and connect it through the Python SDK.
3. Implement a deterministic game engine and Monte Carlo match runner.
4. Design and implement value-heuristic bot strategies.
5. Run seeded round-robin evaluations and compare strategies.
6. Build and locally train a neural policy.

Each milestone will be designed and tested independently.

## License

Garboid PocketRocks is available under the [MIT License](LICENSE).
