# Random Bot SDK Integration Design

**Date:** 2026-07-28
**Status:** Approved for planning

> **Historical note (2026-07-28):** This milestone was implemented and
> live-tested as designed. The later simulator/RL milestone extracted
> `RandomBotBrain`, introduced `PocketRocksFastBot`, and moved the public
> identity from `RANDOM_BOT_ID` into the committed `RandomBot.BOT_ID` class
> constant. The API key remains in ignored `.env`.

## Purpose

Implement the first playable Garboid PocketRocks bot as a direct subclass of the
official SDK's `PocketRocksBot`, following the SDK starter template closely.

The milestone proves two things in order:

1. the bot makes legal random decisions through the SDK's in-memory testing
   transport;
2. after credentials are supplied locally, the bot can establish a brief live
   connection to PocketRocks.

## Goals

- Add a runnable `RandomBot(PocketRocksBot)`.
- Return official SDK `BotDecision` values directly.
- Make random behavior optionally reproducible with a seed.
- Verify the complete SDK runtime path without a server using `FakeTransport`.
- Document credential setup and the live run command.
- Perform a brief live connection test after the user supplies `.env`.

## Non-Goals

This milestone does not:

- create a generic policy protocol;
- add an SDK-to-domain translation layer;
- implement game-state evaluation or bid heuristics;
- implement the local game simulator;
- benchmark coroutine overhead;
- add model or training dependencies;
- keep the live bot running after the connection check.

The simulator will initially call the same async bot interface. A synchronous
policy layer will only be extracted later if simulator design or profiling shows
that it provides concrete value.

## Architecture

The bot follows the SDK starter's intended shape:

```text
PocketRocks server
    -> SDK DecisionContext
    -> RandomBot.choose_decision(...)
    -> SDK BotDecision
    -> PocketRocks server
```

`RandomBot` extends `PocketRocksBot`. The SDK remains responsible for loading
configuration, authenticating, maintaining heartbeats, reconnecting, handling
concurrent games, enforcing deadlines, validating decisions, and writing wire
responses.

There is no separate adapter class. Inheritance from `PocketRocksBot` is the
integration boundary.

## Files

### `src/garboid_pocketrocks/bots/random_bot.py`

Defines:

- `RandomBot(PocketRocksBot)`;
- `RandomBot.__init__(seed: int | None = None, **sdk_options)`;
- async `RandomBot.choose_decision(context) -> BotDecision`;
- `main()`, which parses `--seed` and calls `RandomBot(seed=...).run()`.

The constructor forwards SDK configuration unchanged to `PocketRocksBot` and
owns a private `random.Random` instance.

### `tests/bots/test_random_bot.py`

Contains focused unit tests for decision legality and reproducibility plus one
full SDK runtime test using `pocketrocks.testing.FakeTransport`.

### `pyproject.toml`

Registers:

```toml
[project.scripts]
garboid-random-bot = "garboid_pocketrocks.bots.random_bot:main"
```

`python-dotenv` is declared directly because the random-bot CLI loads its
strategy-specific bot ID before passing it into the SDK.

### `.env.example`

Expands the starter configuration guidance with:

- required API key placeholder;
- committed public random-bot ID under `RANDOM_BOT_ID`;
- hosted server URL;
- optional capacity and log-level examples.

The real `.env` remains ignored and is never read into source control or printed
during verification.

### `README.md`

Documents:

- the random bot behavior;
- copying `.env.example` to `.env`;
- running `uv run garboid-random-bot`;
- optionally setting `--seed`;
- the distinction between fake-transport tests and a live connection.

## Decision Behavior

### Bid Requests

For `context.decision_kind == "submitBid"`:

1. If `legal_max_amount` is absent or nonpositive, return
   `BotDecision.pass_turn()`.
2. Otherwise sample an integer uniformly from the inclusive interval
   `0..legal_max_amount`.
3. Convert a sampled zero to `BotDecision.pass_turn()`.
4. Convert a positive result to `BotDecision.submit_bid(amount)`.

This explores every legal integer bid while preserving the SDK's explicit pass
action.

### Reveal Requests

For `context.decision_kind == "selectInfoToReveal"`:

1. If `revealable_count` is nonpositive, return `BotDecision.pass_turn()`.
2. Otherwise sample uniformly from valid indices
   `0..revealable_count - 1`.
3. Return `BotDecision.select_info_to_reveal(index)`.

Passing when no card can be revealed is legal according to the SDK and avoids
calling `randrange(0)`.

### Seeding

`seed=None` uses nondeterministic system seeding through `random.Random`.
Supplying the same integer seed produces the same sequence of choices for the
same sequence of contexts. Each bot instance owns its RNG; tests and simulated
matches do not share global random state.

## Testing Strategy

Implementation follows test-driven development.

### Unit Tests

Tests will prove:

- a missing, zero, or negative legal bid maximum passes;
- positive bid maxima produce only pass or in-range positive bids;
- zero revealable cards passes;
- positive reveal counts produce only valid reveal indices;
- every returned decision passes `DecisionContext.is_legal(...)`;
- two bot instances with the same seed produce equal decision sequences.

SDK test scenarios will create realistic `DecisionContext` objects instead of
hand-assembling wire data.

### Runtime Integration Test

An SDK `scenario(...)` will be encoded into a decision-request frame and supplied
to `FakeTransport`. A seeded `RandomBot` with dummy credentials, reconnect
disabled, and the fake transport will run through `run_async()`.

The test will decode the transport's sent frames and assert that:

- one decision response is emitted;
- its request ID matches the request;
- its action is legal for the originating context.

This exercises SDK decoding, context reconstruction, bot delegation, validation,
and response encoding without network access.

### Quality Gate

The existing gate remains:

```text
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

GitHub Actions must pass on the pushed commit before attempting the live
connection.

## Error Handling and Safety

- SDK configuration errors remain SDK errors; the bot does not duplicate them.
- Impossible or empty action ranges fail safely with a pass.
- SDK runtime validation remains authoritative for live decisions.
- The real `.env` is ignored and will not be displayed or committed.
- The live connection will be brief and manually interrupted after a successful
  SDK connection is observed.
- If the configured bot is active and assigned a game during that interval, the
  random bot may submit decisions. This is an expected consequence of a live
  connection and will be called out immediately before the run.

## Live Verification

After fake-transport verification, local checks, push, and successful CI:

1. Ask the user to create `.env` from `.env.example` and supply their real API
   key. The committed `RANDOM_BOT_ID` is passed explicitly to the SDK; its
   generic `POCKETROCKS_BOT_ID` remains a fallback when the bot-specific
   variable is absent.
2. Confirm only that `.env` exists; do not print or inspect its values.
3. Run `uv run garboid-random-bot --seed 0` in an interactive terminal.
4. Observe SDK output confirming a connection.
5. Interrupt the process promptly with Ctrl+C.
6. Report the connection result without exposing configuration.

If credentials are rejected or the server is unreachable, preserve the passing
fake-transport implementation and report the live failure separately.

## Acceptance Criteria

The milestone is complete when:

- `RandomBot` directly subclasses `PocketRocksBot`;
- it implements the approved seeded random bid and reveal behavior;
- the installed console command runs the bot through the SDK;
- unit tests and the fake-transport runtime test pass;
- format, lint, strict typing, and the complete test suite pass locally;
- GitHub Actions passes on the pushed implementation;
- a brief live connection succeeds after credentials are provided, or a
  credential/server failure is clearly separated from the verified code result.

## References

- [PocketRocks SDK starter](https://github.com/jaiparera/pocketrocks-python-sdk/tree/develop/starter)
- [PocketRocks Python SDK](https://github.com/jaiparera/pocketrocks-python-sdk)
- [Garboid PocketRocks scaffold design](2026-07-28-repository-scaffold-design.md)
