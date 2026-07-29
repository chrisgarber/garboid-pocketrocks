# Neural Bot

This package contains the Stage 1 neural policy and PPO training proof for
PocketRocks. It trains one learner against frozen heuristic opponents using
only information available to a live SDK bot.

Stage 1 proves that public-history encoding, legal action selection, reward
calculation, PPO updates, deterministic execution, and checkpoint replay work
end to end. It is not yet a competition-strength bot or a resumable,
large-scale training system.

## Architecture

The model is a small recurrent actor-critic with approximately 125,620
parameters:

```text
Current public snapshot ──> Snapshot encoder ──┐
Relative player state ─────> Shared seat MLP ──┼─> Shared trunk ─┬─> Bid policy
Complete public history ───> Event MLP + GRU ──┘                 ├─> Reveal policy
                                                                └─> State value
```

### Snapshot

The current snapshot contains the decision phase, player count, current
action and resources, relative priority seat, value chart, public ruleset
configuration, active objectives, and the learner's private hand.

Categorical fields use small embeddings. The resulting 141 features pass
through:

```text
Linear(141, 128) -> Tanh
```

### Relative player state

Each of five possible player slots contains 41 public features: cash, won
resources, revealed information, and objective ownership. The learner is
always rotated to relative seat zero.

Every valid seat uses the same encoder:

```text
Linear(41, 32) -> Tanh
```

The five outputs are masked and flattened into 160 features.

### Public history

Each public event contains an event kind, action, resources, relative actor,
revealed suit, and relevant numeric data such as setup values or resolved
bids. An event becomes 78 features and is encoded by:

```text
Linear(78, 64) -> Tanh
GRU(input_size=64, hidden_size=64, num_layers=1)
```

The live-A Stage 1 envelope supports up to 76 public events. Every inference
replays the complete cumulative history from a zero GRU state. The model does
not retain mutable per-game state, making inference safe when one SDK process
handles multiple games.

### Shared trunk and heads

The network concatenates 128 snapshot features, 160 player features, and the
64-feature history summary:

```text
Linear(352, 128) -> Tanh
Linear(128, 128) -> Tanh
```

It then produces:

- 101 bid logits: pass plus bids from 1 through 100;
- 6 reveal logits: pass plus five hand positions;
- 1 scalar expected return.

The active policy head is projected into the universal 106-action encoding.
Context-illegal actions receive negative-infinity logits and therefore exactly
zero probability. Training samples with a dedicated seeded
`torch.Generator`; evaluation uses deterministic lowest-index argmax.

## Information boundary

Both actor and value heads use the same deployable information:

- the current SDK `DecisionContext`;
- public `RulesetKnowledge`;
- cumulative public setup, turn, resolved-bid, and reveal events;
- the learner's current private hand;
- the current legal-action mask.

They never receive opponent hands, shuffled deck order, unresolved sealed
bids, simulator RNG state, or an omniscient critic state.

All seat-indexed features and resolved bids are learner-relative. Suit,
objective, and private-hand indices retain their SDK meanings.

## Training update

One Stage 1 PPO update:

1. freezes the current policy for collection;
2. plays 16 complete live-A, three-player games;
3. rotates the learner seat against balanced and passive heuristic bots;
4. stores learner decisions, complete public histories, masks, rewards, old
   log probabilities, and old values;
5. computes gamma-one GAE independently for every game;
6. normalizes advantages over the rollout;
7. replays every stored history under the current model;
8. performs one clipped PPO epoch with a persistent Adam optimizer.

The default reward combines:

- normalized changes in public financial potential;
- claimed objective payouts;
- terminal resource value;
- a terminal first-place bonus, shared by tied winners.

Event-shaping and placement bonuses default to zero. Gamma is `1.0` because
learner decisions do not represent uniform game time: an auction winner
receives an extra reveal decision that should not discount later rewards.

At the current 16-game batch size, rollout generation is roughly 90% of the
update time and PPO optimization is roughly 10%.

## Running the deterministic smoke

Install the optional CPU neural dependencies:

```bash
uv sync --locked --extra neural
```

Run two updates of 16 games:

```bash
uv run --extra neural garboid-train smoke \
  --output-dir artifacts/neural-smoke
```

The output directory must be new or empty. The smoke verifies:

- every game terminates;
- every sampled action is legal;
- illegal-action probability is exactly zero;
- outputs, losses, advantages, and gradients are finite;
- model parameters change after every update;
- identical seeded runs produce identical deterministic metrics and parameter
  digests;
- a saved checkpoint reproduces fixture logits, value, and greedy action.

The smoke validates mechanics and determinism, not playing strength.

## Checkpoints

Stage 1 writes an inference-only bundle:

```text
checkpoint/
  manifest.json
  model.pt
```

The manifest records model and encoder configuration, supported rulesets and
player counts, dependency versions, action-space identity, completion counts,
and file/parameter digests. Loading is fail-closed for incompatible schemas,
tensor names, shapes, dtypes, bounds, checksums, or non-finite weights.

Stage 1 checkpoints omit optimizer and RNG state and therefore cannot resume
training.

## Package map

- `config.py`: checkpointed encoder, model, and Stage 1 dimensions.
- `encoding.py`: learner-relative observations and tensor batches.
- `model.py`: snapshot, seat, history, trunk, policy, and value network.
- `policy.py`: phase projection, legal masking, sampling, and greedy actions.
- `advantages.py`: gamma-one generalized advantage estimation.
- `seeding.py`: stable namespaced seeds and deterministic Torch setup.
- `rollout.py`: fixed-opponent learner trajectory collection.
- `ppo.py`: clipped PPO loss and persistent optimizer.
- `checkpoint.py`: inference checkpoint validation and serialization.
- `smoke.py`: deterministic end-to-end training proof.
- `cli.py`: the `garboid-train smoke` command.

## Current limitations

Stage 1 is restricted to live-A, three players, a maximum bid of 100, and a
five-card hand envelope. It does not yet include:

- resumable training artifacts;
- held-out evaluation or confidence intervals;
- charts B through E or varied rulesets;
- a checkpoint opponent league;
- behavior-cloning initialization;
- a registered live `NeuralBot` wrapper.

The end-state design and staged roadmap are documented in
[`docs/superpowers/specs/2026-07-28-neural-self-play-design.md`](../../../docs/superpowers/specs/2026-07-28-neural-self-play-design.md).
