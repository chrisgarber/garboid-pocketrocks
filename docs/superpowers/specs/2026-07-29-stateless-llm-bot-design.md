# Stateless LLM Bot Design

## Summary

Add a PocketRocks brain that sends each SDK-visible decision state to a
replaceable LLM backend and accepts exactly one legal integer as the answer.
The first backend invokes the local Codex CLI. The same brain must work in the
deterministic simulator and behind a runnable `PocketRocksFastBot` live
wrapper.

Version 1 is deliberately stateless. `DecisionContext` has no game identifier,
so a live process cannot safely associate concurrent requests with persistent
LLM sessions. The interfaces will permit a future session-aware implementation
after the SDK exposes a reliable game ID, but this feature will not infer
session identity or retain conversational state.

## Goals

- Separate PocketRocks prompt construction from LLM execution.
- Provide a generic synchronous LLM brain usable by the simulator.
- Provide a stateless local Codex CLI backend.
- Give the LLM concise rules, the complete SDK-visible snapshot, named suits,
  actions, and objectives, and one exact integer output range.
- Strictly validate output, retry once, then return a deterministic legal
  fallback.
- Expose the bot through both the simulator registry and a runnable live-bot
  command.
- Make model, executable, and timeout configuration easy to vary without
  changing game logic.

## Non-Goals

- Persistent or stateful LLM conversations.
- Reconstructing history, loan positions, or investment positions omitted by
  `DecisionContext`.
- Modifying the PocketRocks SDK or wire protocol.
- Guaranteeing deterministic decisions from a remote model.
- Adding an HTTP/API-backed provider in this version.

## Architecture

The implementation lives under `garboid_pocketrocks.bots.llm` and uses
composition:

1. `LLMBackend` is a protocol with a synchronous
   `complete(prompt, timeout_seconds=...) -> str` operation.
2. `PromptSkill` is a protocol that renders a complete prompt from
   `DecisionContext` and `RulesetKnowledge`.
3. `PocketRocksPromptSkill` loads a packaged `SKILL.md` and appends a
   deterministic, human-readable state snapshot and legal-output contract.
4. `StatelessLLMBrain` coordinates prompt rendering, backend invocation,
   parsing, validation, retry, fallback, and diagnostics.
5. `CodexCLIBackend` invokes `codex exec` without a shell and returns only the
   final response text.
6. `CodexBot` composes the default prompt skill, brain, and backend for the
   live SDK and simulator registry.

The protocols do not mention Codex or subprocesses. A future provider can
implement `LLMBackend`; a future strategy can implement `PromptSkill`. A
future stateful brain may reuse both once requests can be keyed by game.

## Prompt Skill

The static rules live in
`src/garboid_pocketrocks/bots/llm/skills/pocketrocks/SKILL.md`. It is a
portable prompt asset, not Codex-specific configuration. Stateless calls embed
its full contents so any backend sees the same rules.

The skill explains:

- the objective of maximizing final money;
- simultaneous sealed bidding, all-zero wins, priority-order tie breaking, and
  the winner becoming the next priority seat;
- Auction 1, Auction 2, Loan 10, Loan 20, Invest 5, and Invest 10;
- the winner's required private-card reveal;
- per-suit value-chart pricing from total revealed counts;
- objective claims and payouts;
- final scoring, including returned investment locks and repaid loan
  principals;
- that the final response must be only the requested integer.

The dynamic section uses SDK reference helpers rather than duplicating numeric
decoder tables. It includes:

- ruleset name, player count, starting cash, resource/action deck counts, and
  private cards per player;
- value chart and active objectives with descriptions and payouts;
- the bot seat and priority seat;
- current action and offered resources;
- every seat's cash, won resources, revealed information, and owned
  objectives;
- the bot's ordered private hand;
- a zero-based reveal index map when revealing.

The snapshot is the complete information exposed by `DecisionContext` plus
public `RulesetKnowledge`. It explicitly says that prior bids and current
loan/investment positions are unavailable rather than fabricating them.

## Integer Contract

For a bid with legal maximum `N`, the last instruction is:

> Return exactly one base-10 integer from 0 through N. 0 means bid zero/pass.

For a reveal with `N` cards, the last instruction is:

> Return exactly one base-10 integer from 0 through N-1: the card index to
> reveal.

The parser strips surrounding whitespace and otherwise accepts only ASCII
digits. Prose, Markdown fences, signs, decimals, and JSON are invalid. The
parsed integer must lie in the prompt's legal inclusive range before it is
converted to an SDK decision and checked with `DecisionContext.validate`.

## Decision Flow and Failure Handling

When no positive bid is available, return pass without calling the backend.
When no card is revealable, return pass without calling the backend.

Otherwise:

1. Render the complete prompt.
2. Call the backend with a timeout bounded by the configured per-attempt
   maximum and the SDK request's remaining deadline.
3. Parse and validate the integer.
4. On a backend exception, timeout, malformed response, or out-of-range
   integer, log a warning and retry once.
5. The retry is another stateless call. It contains the original prompt plus a
   short correction naming the failure and restating the exact legal range.
6. On a second failure, log a final warning and return the deterministic
   fallback: pass for bidding, index `0` for reveals.

Warnings include the request ID, decision kind, attempt number, and concise
error. They do not log API keys or the full prompt. Backend subprocess stderr
is included only in raised diagnostic text and is bounded.

The configured timeout defaults to 30 seconds per attempt. Before each
attempt, the brain reserves a small deadline safety margin and divides the
remaining live deadline across remaining attempts. An already-expired request
falls back immediately. Simulator contexts have effectively unbounded
deadlines and use the configured maximum.

## Codex CLI Backend

`CodexCLIBackend` runs an argument vector through `subprocess.run`; it never
constructs a shell command. Each call:

- creates an isolated temporary working directory;
- sends the prompt over stdin;
- uses `codex exec --ephemeral`;
- skips the Git-repository requirement;
- ignores project rules and user configuration while retaining Codex auth;
- uses the read-only sandbox;
- disables color;
- writes the final message to an isolated output file;
- optionally selects a configured model;
- applies the timeout provided by the brain.

Nonzero exit status, timeout, missing output, and unreadable output become
backend errors. Output and stderr diagnostics are size-bounded. The backend
does not parse game decisions.

## Live and Simulator Integration

`CodexBot` is a `PocketRocksFastBot` with a clearly development-only bot ID and
name `codex`. Its default brain factory constructs a fresh stateless brain.
The live wrapper overrides its async decision bridge to run the synchronous
brain in a worker thread so the Codex subprocess cannot block the SDK event
loop or prevent runtime timeouts.

The `garboid-codex-bot` command accepts optional `--model`, `--timeout-seconds`,
and `--codex-executable` flags before constructing the live wrapper. SDK
connection settings continue to come from its existing environment variables.

The simulator registry accepts `codex` in `--bots`. Direct users may build a
custom `BotSpec` with any backend or prompt skill. Documentation warns that
LLM simulation is slow, nondeterministic, and may incur model usage.

## Testing Strategy

Use red-green TDD with no live model calls in the automated suite.

- Prompt tests assert named rules, full public/private snapshot fields,
  objective descriptions, exact bid ranges, and exact reveal index maps.
- Brain tests use a small scripted backend to prove valid bids/reveals,
  whitespace parsing, rejection of non-integer and out-of-range responses,
  correction retries, backend-exception retries, deterministic fallbacks,
  no-call fast paths, and deadline bounding.
- Backend tests replace the subprocess runner to assert the exact safe argument
  vector, stdin usage, model option, timeout, isolated output handling, and
  bounded errors.
- Integration tests prove the bot wrapper, factory, simulator registry, live
  async bridge, and CLI argument plumbing without invoking Codex.
- Existing formatting, typing, and full pytest suites remain green.

One opt-in/manual smoke command may invoke the installed Codex CLI with a tiny
integer prompt. It is not part of the default test suite because it requires
authentication, network access, time, and model usage.
