# Simulator

The simulator runs synchronous bot brains over the SDK's scalar and batch game
engines. The SDK remains the sole [rules authority](../../../docs/architecture/sdk-authority.md).

## Run matches

From the repository root:

```bash
uv run garboid-simulate \
  --bots random,aggressive,balanced \
  --games 1000 \
  --players 3 \
  --ruleset live-A \
  --seed 42 \
  --workers 4
```

Registered bot names include the live aliases and explicit released
generations. `--format json` emits structured summaries. `--replay-dir PATH`
writes one deterministic replay per game and uses scalar execution so every
decision is captured.

## Execution contract

`SdkGameSession` presents the SDK scalar engine as one pending decision phase
at a time. A bid phase requests one decision from every seat. A choice reveal
requests only the winning seat; automatic reveals never invoke a bot.

`MatchRunner` constructs fresh brains in seat order from a per-game RNG,
passes exact public history to every brain, validates every decision,
and records replay steps in encounter order.

Monte Carlo planning derives each game seed and lineup from the root seed.
Plans remain stable across reruns and process-worker counts. Results are
returned in game-index order.

## Batch execution

Batch chunks contain one player count while allowing different seeds, value
charts, and objective modes per row. Each row retains its independent brain
seed stream, public history, replay decisions, faults, and turn records.

The batch path snapshots resources, objective claimants, and turn indices
before resolution. It preserves scalar contexts, effective bids, newly
claimed objectives, reveals, scores, competition ranks, and replay bytes.
Active rows and returned results remain in deterministic row order.

## Fault policy

`FaultMode.RAISE` propagates the original construction or decision exception.
`FaultMode.RECORD_AND_PASS` records:

- one turn-zero fault when brain construction fails;
- each runtime failure in encounter order, with turn, seat, name, type, and
  message.

The deterministic fallback is bid zero or reveal index zero. A brain that
failed construction stays absent and silently uses fallback decisions after
its single construction fault.

## Replay

Schema-version-two replay JSON contains configuration, bot names, ordered
decision steps, SDK turn records, and the terminal result. Monte Carlo capture
adds root-seed and game-index provenance. Replaying validates acting seats,
legal decisions, complete termination, turn history, and result equality.

## Extension points

- Add simulation identities through the shared bot registry.
- Add public strategy inputs only through the
  [information boundary](../../../docs/architecture/public-information-boundary.md).
- Preserve [deterministic evaluation](../../../docs/architecture/deterministic-evaluation.md)
  when adding batching, workers, or output fields.
- Treat replay schema changes as explicit compatibility work.
