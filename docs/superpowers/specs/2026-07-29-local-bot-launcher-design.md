# Local Bot Launcher Design

**Date:** 2026-07-29
**Status:** Approved for planning

## Purpose

Publish the existing PocketRocks bot wrappers for local use by assigning the
activated heuristic bot identities and providing one command that keeps every
defined bot connected to the live service concurrently.

This milestone is a local runtime convenience. It does not add persistent
hosting or change any strategy behavior.

## Goals

- Replace the three development-only heuristic bot IDs with their activated
  public identities.
- Define one ordered registry containing every live bot wrapper in the project:
  random, aggressive, balanced, and passive.
- Add a `garboid-bots` console command that launches all registered bots by
  default.
- Allow the command to launch a comma-separated subset for focused local runs.
- Run all selected SDK bot runtimes concurrently in one process.
- Shut down the complete group on user interruption or when one runtime stops.
- Test launcher behavior without contacting the live service.
- Document the real identities and local launcher command.

## Non-Goals

This milestone does not:

- deploy bots to persistent hosting;
- add process supervision outside the local Python process;
- change random or heuristic decision behavior;
- add per-bot API keys or server configuration;
- activate or deactivate bots through the PocketRocks dashboard;
- retry failures outside the SDK's existing reconnect behavior.

## Public Bot Identities

The bot wrapper class constants are the source of truth:

| Name | Wrapper | Bot ID |
| --- | --- | --- |
| `random` | `RandomBot` | `bot_e0e2c541-1615-4f47-983c-224e7d888d89` |
| `aggressive` | `AggressiveHeuristicBot` | `bot_386b81bb-14df-477a-8d4c-0231cf1b3b1a` |
| `balanced` | `BalancedHeuristicBot` | `bot_265c84aa-f28e-4a35-b4de-a4f4ee406415` |
| `passive` | `PassiveHeuristicBot` | `bot_9d33c9de-4d90-4608-9a58-d2c77d93e0bd` |

Every bot uses the same SDK configuration loaded from `.env`, including
`POCKETROCKS_API_KEY`, `POCKETROCKS_SERVER_URL`, capacity, logging, and
reconnection settings. `PocketRocksFastBot` supplies the wrapper's `BOT_ID`
explicitly so a generic `POCKETROCKS_BOT_ID` environment value cannot select
the wrong identity.

## Architecture

```text
garboid-bots
    |
    +-> bot registry
    |     random -> RandomBot
    |     aggressive -> AggressiveHeuristicBot
    |     balanced -> BalancedHeuristicBot
    |     passive -> PassiveHeuristicBot
    |
    +-> one asyncio task per selected wrapper
            |
            +-> PocketRocksBot.run_async()
                    |
                    +-> independent SDK transport and reconnect loop
```

The launcher belongs in a focused module under
`src/garboid_pocketrocks/bots/`. It owns only bot discovery, command-line
selection, bot construction, and group lifecycle. Strategy modules remain
independent, and the simulator continues using `BotSpec` values rather than
the live launcher.

The registry maps each public CLI name to its wrapper class and preserves the
order `random, aggressive, balanced, passive`. The registry is the launcher's
definition of "all defined bots"; tests prevent a wrapper from being
accidentally omitted.

## Command-Line Interface

The installed command is:

```bash
uv run garboid-bots
```

With no arguments, it launches all four registered bots. A focused run uses:

```bash
uv run garboid-bots --bots aggressive,passive
```

`--bots` accepts a comma-separated list of registry names. Whitespace around
names is ignored. The launcher rejects an empty selection, duplicate names,
and unknown names through `argparse` with a nonzero exit before constructing
any bot.

The launcher does not expose strategy seeds. The random bot retains its normal
nondeterministic default, which is appropriate for a long-running live
connection. Existing direct random-bot execution remains available for a
seeded single-bot run.

## Runtime Lifecycle

The synchronous entry point installs the SDK's default logging once and calls
`asyncio.run(...)`. The async launcher constructs one wrapper per selected
name, then starts each wrapper's `run_async()` in a structured task group.

Each wrapper creates its own SDK runtime, transport, worker queue, and
reconnection policy. Sharing the event loop does not share bot state or
network transports.

An SDK runtime normally remains active indefinitely. If one `run_async()`
returns, the launcher treats that as a stopped fleet member, cancels its
siblings, waits for their cancellation cleanup, identifies the stopped bot in
the error, and exits nonzero. This fail-fast rule avoids presenting a partial
fleet as healthy. Retryable disconnects do not trigger it because they remain
inside the SDK runtime's reconnect loop.

On Ctrl+C, `asyncio.run(...)` cancels the launcher task. Cancellation reaches
every bot runtime, whose `finally` cleanup disconnects its transport and
workers. The command then exits as an ordinary user interruption without
printing a misleading runtime failure.

If bot construction fails because credentials or SDK configuration are
missing or invalid, the command fails before launching any task. It does not
print secret values.

## Testing Strategy

Implementation follows test-driven development.

### Identity Tests

Update the existing heuristic bot identity assertions to require the three
activated IDs. Continue asserting that the four wrapper names are unique and
stable.

### Registry and Parsing Tests

Focused launcher tests prove:

- the registry contains all four wrapper classes in the documented order;
- no `--bots` argument selects all four;
- a comma-separated subset preserves the requested order;
- surrounding whitespace is accepted;
- an empty selection, duplicate name, or unknown name is rejected;
- invalid input does not construct any wrappers.

### Lifecycle Tests

Use small fake bot classes with controllable async runtimes rather than
`FakeTransport` or the live service. Tests prove:

- each selected wrapper is constructed exactly once;
- all selected runtimes start concurrently;
- cancelling the launcher cancels every active runtime;
- a runtime that returns causes its siblings to be cancelled;
- the resulting failure identifies the bot that stopped.

Existing SDK `FakeTransport` tests continue covering the real wrapper-to-SDK
integration boundary. The launcher tests cover only orchestration.

### Quality Gate

The repository gate remains:

```text
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

No automated test connects to PocketRocks.

## Documentation

The README will:

- replace the heuristic placeholder IDs and development-only warning with the
  activated identities;
- add a local all-bot command example;
- document `--bots` for subset execution;
- state that one API key and the existing SDK environment configuration are
  shared across the four independent connections;
- retain the direct seeded random-bot command.

## Live Verification

After local tests pass, a brief manual verification may run
`uv run garboid-bots` using the existing ignored `.env`. Because the four bot
identities are activated, this can receive and answer live game requests.
Verification must therefore be deliberate and interrupted after confirming
that all four connections succeed.

The `.env` file is checked only for existence. Its contents are never printed,
committed, or copied into command output.

## Acceptance Criteria

The milestone is complete when:

- all three heuristic wrappers contain their activated public bot IDs;
- `garboid-bots` is installed and launches all four registered wrappers by
  default;
- `--bots` launches a validated subset;
- one stopped runtime tears down the local group with a clear nonzero failure;
- Ctrl+C shuts down all connections together;
- launcher and identity tests pass without network access;
- format, lint, strict typing, and the complete test suite pass;
- the README documents the identities and launcher behavior;
- a brief live connection succeeds, or any credential/server failure is
  reported separately from the verified implementation.
