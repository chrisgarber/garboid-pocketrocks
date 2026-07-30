# Smoke Neural Tournament Bot Design

## Goal

Make the completed smoke-training policy a reproducible, standard tournament
competitor and measure it against the default historical bot field.

## Frozen identity and artifact

The local-only simulation identity is `vector_ppo_small_v1_g1500`:

- `vector_ppo` describes the training algorithm and vectorized self-play.
- `small_v1` freezes the network architecture generation.
- `g1500` freezes training age at 1,500 completed games.

Export the existing training checkpoint to an inference-only bundle containing
only `manifest.json` and `model.pt`. Commit the approximately 0.5 MB bundle
under the Python package so every checkout evaluates identical parameters.
The manifest and loader continue to validate the model checksum, parameter
digest, encoder contract, action-space contract, and supported charts and
player counts.

This bot is a local `BotSpec`; it does not receive a remote `BOT_ID`.

## Inference architecture

Add a history-aware brain protocol beside the existing `BotBrain` protocol.
`MatchRunner` detects that optional interface and supplies an immutable
`PublicHistory` built from the current `SdkGameSession.events` before each
decision. Existing random and heuristic brains continue through the unchanged
two-argument interface.

The neural brain:

1. Lazily loads the committed checkpoint on CPU, once per worker process.
2. Encodes `DecisionContext`, `RulesetKnowledge`, and exact public history with
   the checkpoint's `NeuralObservationEncoder`.
3. Builds a one-row neural batch and evaluates the policy under inference mode.
4. Chooses the highest-probability legal action deterministically.
5. Decodes the universal action into an SDK `BotDecision`.

The ordinary two-argument method raises a clear error because omitting public
history would evaluate a different policy than the one trained. The runner
always selects the history-aware method when available.

Torch remains an optional dependency. Commands that use the default tournament
must run with `uv run --extra neural garboid-tournament`; importing the general
bot registry remains safe without eagerly loading a model.

## Registration

Add `vector_ppo_small_v1_g1500` to the complete registry and to
`DEFAULT_TOURNAMENT_BOT_SPECS`. The default field becomes:

- random
- aggressive-v1, balanced-v1, passive-v1
- aggressive-v2, balanced-v2, passive-v2
- vector_ppo_small_v1_g1500

Unversioned aliases remain outside the default field because they duplicate v2
behavior.

## Failure behavior

Checkpoint, encoding, or inference errors follow the tournament's existing
fault mode. Normal runs raise immediately; fault-recording runs record the
exception and submit the legal pass fallback. Unsupported charts or player
counts fail checkpoint validation rather than silently changing inputs.

## Test strategy

Use test-driven development:

- First prove the default registry lacks the new identity.
- Prove a history-aware test brain receives the exact public event sequence.
- Prove the frozen checkpoint loads with its expected identity, game age,
  architecture profile, and parameter digest.
- Prove the neural brain returns legal deterministic decisions for bid and
  reveal contexts using exact session history.
- Prove the new `BotSpec` is pickle-safe for spawned tournament workers.
- Run focused bot, simulator, neural, and tournament tests, then the full
  formatting, lint, strict typing, and 645+ test suite.

## Tournament measurement

Run the current default tournament unchanged:

```text
15,000 games
3, 4, and 5 players
charts A, B, C, D, and E
batch size 64
root seed 0
200 bootstrap samples
```

Capture the generated CSV/JSON/HTML report and report the smoke bot's rank,
Plackett-Luce rating and interval, games, outright win rate, mean final money,
fault count, total runtime, and games per second. Commit compact tournament
artifacts that support the reported result, excluding large replay payloads if
the existing artifact policy already ignores them.

## Main-branch integration

Commit implementation and benchmark results on the focused branch, merge them
into local `main`, and restore the pre-existing unstaged
`tests/neural/test_rollout.py` edit without including it in these commits.
