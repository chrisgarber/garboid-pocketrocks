# Neural self-play design

**Date:** 2026-07-28

**Status:** Approved for implementation

## Purpose

Train one deployable PocketRocks policy at a time against frozen opponents,
then improve it through iterative self-play. The first learned policy should
work across supported player counts and rulesets, use only information
available to a live bot, and return the same SDK `BotDecision` type as the
random and heuristic brains.

The recommended system is recurrent, legal-action-masked PPO trained against a
checkpoint league. Its outer loop is AlphaZero-shaped—generate play with the
current policy, optimize, freeze a checkpoint, evaluate, and add promoted
checkpoints to the opponent pool—but it does not use AlphaZero's perfect-
information MCTS.

## Goals

- Reuse the deterministic engine, `ActionCodec`, observation fields,
  `RewardTracker`, ruleset samplers, and single-agent environment.
- Train a single learner seat against frozen `BotBrain` opponents.
- Preserve sealed-bid information boundaries during rollout and inference.
- Make histories and seat-indexed features learner-relative.
- Support charts A-E, 3-5 players, and bounded ruleset variations.
- Make every sampled action legal by construction.
- Use final money and winning as the primary optimization signal while
  allowing configurable public intermediate shaping.
- Save reproducible, inspectable, live-loadable checkpoints.
- Start with a deterministic CPU smoke run before spending meaningful compute.

## Non-goals

The first neural milestone does not:

- solve PocketRocks as an equilibrium;
- claim that PPO converges in this multiplayer, imperfect-information game;
- use hidden simulator state in the actor or critic;
- train all seats' changing policies simultaneously;
- use MCTS, determinization, CFR, ReBeL, or a learned world model;
- require distributed training or a GPU;
- tune a final competition-strength network;
- replace the existing Gymnasium or PettingZoo contracts;
- expose an unregistered development bot ID to the live service.

## Existing boundaries

The repository already provides:

- `BotBrain.choose_decision(context, ruleset) -> BotDecision`;
- `PocketRocksFastBot` as the synchronous-to-async SDK bridge;
- deterministic `GameEngine` transitions and complete replays;
- `PocketRocksEnv`, which controls one learner seat and advances frozen
  opponent brains internally;
- `PocketRocksAECEnv` for contract testing and future multi-agent work;
- `ObservationEncoder`, which contains only SDK context plus public
  `RulesetKnowledge`;
- `ActionCodec`, with one fixed discrete space and a per-request legality mask;
- `RewardTracker`, whose public accounting deltas reconcile to normalized final
  money before configured bonuses;
- fixed, weighted, and bounded-variation ruleset samplers;
- random and three frozen heuristic opponent brains.

The neural layer adds history, canonicalization, optimization, checkpoints,
and a live brain. Game rules remain solely in `GameEngine`.

## Considered approaches

### 1. Heuristic imitation only

Collect simulator contexts labeled by aggressive, balanced, and passive
brains, then train a masked classifier.

Advantages:

- simple and stable;
- quickly validates the encoder, network, and checkpoint path;
- provides a useful optional initialization.

Limitations:

- cannot exceed the demonstrations without another learning objective;
- learns the heuristic evaluator's biases, including the passive-heavy v1
  matchup behavior;
- does not learn from terminal outcomes.

Decision: retain behavior cloning as an optional warm start and diagnostic, not
as the primary training algorithm.

### 2. Naive AlphaZero-style MCTS

Use the simulator for tree search and train policy/value targets from search.

Advantages:

- search can exploit an exact forward model;
- policy improvement and value learning have a familiar iterative structure.

Limitations:

- PocketRocks has hidden hands and deck order;
- bidding is simultaneous rather than alternating;
- games have three to five players and are not zero-sum;
- searching one sampled hidden state leaks information, while determinizing
  independently can create strategy fusion;
- correct belief-state, simultaneous-action search is a substantially larger
  research project.

Decision: reject for the first neural bot. A future search milestone should
start from explicit public beliefs and an imperfect-information method, not
retrofit perfect-information MCTS.

### 3. Recurrent masked PPO with a checkpoint league

Train one on-policy learner against frozen heuristic and neural checkpoints.
Encode cumulative public history with a GRU, mask illegal actions at both
sampling and loss evaluation, and periodically freeze/evaluate candidates.

Advantages:

- fits the existing single-agent environment;
- avoids changing opponents during a PPO batch;
- handles partial observability without privileged state;
- supports every legal integer bid and reveal slot;
- checkpoint opponents make self-play reproducible and reduce latest-policy
  forgetting.

Limitations:

- PPO is not an equilibrium solver;
- recurrent rollout batching is more complex than feed-forward PPO;
- results depend on curriculum, reward scale, and opponent-pool coverage.

Decision: use this approach. It is the smallest credible path from the current
repository to a trainable and live-compatible policy.

## Information boundary and public history

### Permitted model inputs

The policy and value function may consume only:

- the current SDK `DecisionContext`;
- public `RulesetKnowledge`;
- cumulative events already public at that decision;
- the learner's own current hand;
- the action mask derived from the current context.

They must not consume:

- opponent private hands;
- remaining resource or action deck order;
- simulator RNG state or seed;
- unresolved opponents' sealed bids;
- engine-only loan, investment, or card objects;
- terminal scores before the game ends;
- an omniscient critic state.

Training labels may use a terminal result after it becomes available. An
auxiliary target may teach the network to estimate hidden quantities from
public inputs, but the hidden quantity itself may never be an inference input.
The first implementation has no auxiliary hidden-state heads.

### Canonical public event stream

The official SDK raw decision callback receives a request containing cumulative
`common_events`: game setup, turn opened, resolved bids, and information
reveals. Although the pinned SDK types this callback's frame as `object`, the
events are public server data.

Introduce a project-owned immutable public-history representation with four
event variants:

- game setup;
- turn opened with action and offered resources;
- auction resolved with all now-public bids;
- information revealed with suit.

Two isolated adapters produce this representation:

1. the live adapter reads the pinned SDK raw frame structurally;
2. the simulator adapter converts completed engine decisions/events.

The simulator adapter must not publish `DECISION_SUBMITTED` events until the
entire auction resolves. Parity fixtures must prove that equivalent SDK and
simulator histories produce identical tokens.

The public-history adapter is the only code allowed to inspect the raw SDK
frame. It must fail with a specific compatibility error when the expected
structure is absent; a recurrent checkpoint must not silently fall back to an
empty history.

### Stateless recurrence

Every raw SDK request contains cumulative history. Live inference therefore
replays the bounded history through the GRU on each request instead of keeping
mutable per-game hidden state. This is safe when one SDK bot process serves
multiple games and no stable game identifier is available.

The first trainer uses exactly the same rule: every decision replays its
cumulative history from a zero hidden state. Rollouts store that encoded
history, and every PPO epoch recomputes its representation under the current
weights. It never combines a carried hidden state with cumulative input and
never reuses a hidden state produced by old policy weights.

Incremental suffix processing is a later performance optimization. If added,
it requires an explicit history cursor, episode reset rules, and equivalence
tests against full replay. PPO updates must still reconstruct hidden states
from episode starts under the weights being optimized.

## Learner-relative canonicalization

Absolute seat numbers are an avoidable symmetry. Before tensorization, rotate
every active seat-indexed feature so the learner is relative seat zero:

```text
relative_seat = (absolute_seat - learner_seat) mod player_count
absolute_seat = (learner_seat + relative_seat) mod player_count
```

Apply this mapping to:

- cash;
- won resources;
- revealed information;
- owned objectives;
- priority seat;
- resolved-bid history.

Place active relative seats in slots `0..player_count-1` and zero-pad the
remaining slots through seat four. Include a seat-valid mask. Replace the
encoded bot seat with zero rather than teaching absolute positions.

Keep suit IDs and objective IDs stable because objectives attach semantics to
specific suits. Keep private-hand slot order stable because reveal actions
select an SDK hand index. Include hand-valid and history-valid masks.

A rotation metamorphic test must construct equivalent games with different
absolute learner seats and obtain identical canonical tensors, logits, and
values after mapping actions back to the SDK representation.

## Neural observation encoder

Add a neural encoder beside the existing `ObservationEncoder`; do not change
the public Gymnasium observation contract merely to suit one model.

The neural encoder consumes the current encoded snapshot plus canonical public
history and produces:

- categorical IDs for phase, action, resources, private-hand suits, objective
  membership, and event fields;
- learner-relative per-seat features and masks;
- deterministic normalized numeric features for cash, counts, and chart
  values;
- the unchanged universal action mask.

Normalization limits come from the declared support of the curriculum and
`EnvironmentBounds`, not statistics discovered from hidden states. The
checkpoint stores these limits. A ruleset outside them is rejected before
play.

Whole public histories are padded to a maximum derived from the supported
action-deck sizes. The first implementation does not truncate history. A
checkpoint whose history bound cannot cover a requested ruleset is
incompatible.

## Policy/value network

Use one shared `NeuralPolicy` PyTorch module:

1. embed categorical snapshot fields;
2. encode each relative seat with a shared seat MLP;
3. embed each canonical public event;
4. pass event embeddings through a single-layer GRU;
5. concatenate the GRU summary, current snapshot encoding, and seat encoding;
6. pass the result through a shared MLP trunk;
7. emit phase-specific policy logits and one scalar value.

The smoke configuration uses a GRU hidden size of 64. Embedding and trunk sizes
remain explicit checkpointed configuration, not module globals.

### Policy heads

Use two heads:

- bid head: pass plus every amount `1..max_bid`;
- reveal head: pass plus every hand slot `0..max_hand_size-1`.

Map the active head into `ActionCodec`'s universal action indices. Set all
inactive-phase and context-illegal logits to negative infinity before creating
the categorical distribution. Assert that every mask contains pass, so the
distribution is never empty.

Entropy, log probability, greedy selection, and PPO ratios all use the same
masked distribution. Invalid-action penalties are a diagnostic fallback, not
the mechanism enforcing legality.

Training samples from masked probabilities with
`torch.multinomial(..., generator=policy_generator)`, because
`torch.distributions.Categorical.sample()` does not accept a generator.
Evaluation and live play use deterministic argmax, breaking exact ties by the
lowest universal action index.

### Value head

The value head predicts the learner's expected discounted training return from
the same deployable information as the policy. The first implementation does
not use a centralized or omniscient critic.

## Reward contract

The learning objective preserves both signals the game provides:

- normalized final money gives useful credit even in a loss;
- a configurable terminal first-place bonus makes winning most valuable and is
  divided among tied winners.

The existing accounting reward remains the recommended dense money signal. Its
public accounting changes plus terminal resource remainder telescope to:

```text
(final_money - starting_cash) / starting_cash
```

Thus intermediate accounting does not invent a new target; it redistributes
the terminal money signal to public financial transitions. The terminal win
bonus remains separate. Optional placement and public-event bonuses are
additional shaping and default to zero.

The trainer stores every `RewardBreakdown` component and reports undiscounted
final money, outright/tied-first results, and shaping separately. Model
selection may not use shaped return alone.

Recommended initial reward configuration:

- accounting weight: `1.0`;
- win bonus: `1.0`;
- placement bonuses: empty;
- public event bonuses: empty;
- invalid-action penalty: `0.0`, because masked sampling must make it
  unreachable.

Experiments may add intermediate event shaping through configuration. Each run
records the exact coefficients, and evaluation always uses raw score metrics.
If discounting makes earlier accounting deltas undesirable, a terminal-only
ablation sets accounting delivery to terminal without changing the final
money target; that ablation is an implementation option, not the default.

## Rollout and PPO

### On-policy unit

`PocketRocksEnv` controls one randomly rotated learner seat. Every other seat
uses a frozen `BotBrain` selected before the episode. Only learner transitions
enter the PPO batch. Rewards accumulated while opponents act are already
returned at the learner's next decision.

Opponent assignments, ruleset, player count, learner seat, engine seed, and
policy-sampling seed derive from one root seed with disjoint namespaces.
Changing vector scheduling must not change planned episodes.

Freeze the learner weights used for collection until the rollout batch is
complete. Freeze every opponent for the entire episode. This preserves PPO's
on-policy assumption as far as the controlled learner is concerned.

Each stored transition contains:

- neural observation/history or reproducible encoded tensors;
- legal-action mask;
- sampled action;
- old masked log probability;
- old value;
- reward and `RewardBreakdown`;
- termination/truncation flags;
- ruleset, learner seat, and opponent checkpoint identities.

### GAE

Compute generalized advantage estimates over each learner trajectory:

```text
delta_t = reward_t + (1 - terminal_t) * value_(t+1) - value_t
adv_t   = delta_t + lambda * (1 - terminal_t) * adv_(t+1)
```

The MVP uses `gamma=1.0` and `lambda=0.95`. `PocketRocksEnv` steps at learner
decisions rather than uniform game time: winning an auction introduces an
extra reveal decision that losing does not. Discounting once per learner step
would therefore reduce post-win returns merely because the winner must reveal
a card. An undiscounted episodic return preserves the final-money objective.
True terminal states do not bootstrap. Time-limit truncations, if later
introduced, do. Normalize advantages over valid transitions in the update
batch.

A later discounted experiment must use per-transition game-time discounts,
with reveal-only transitions discounted by `1.0`, and prove that otherwise
equivalent public trajectories do not differ solely because one contains the
required reveal decision.

### PPO update

Use the standard clipped surrogate objective with initial:

- policy clip ratio `0.2`;
- one value-loss coefficient;
- one entropy coefficient;
- global gradient-norm clipping;
- Adam optimizer;
- explicit learning rate;
- no mixed precision in the deterministic smoke run.

All values are configuration recorded in the checkpoint. Recompute new log
probabilities through the stored legal mask. A mismatch between stored action
space/encoder schema and the current trainer is a hard error.

Model minibatches contain independently padded cumulative histories. History
padding is excluded from GRU pooling, while all collected learner transitions
participate in policy, value, entropy, and advantage statistics. GAE remains
trajectory-based, but PPO minibatches need not preserve contiguous episode
subsequences because each model input reconstructs its complete public
history. Incremental recurrent batching, truncated sequences, and burn-in are
later optimizations.

## Ruleset curriculum

The model has one fixed action/observation envelope large enough for the union
of all curriculum stages. Progress stages by completed episodes or explicit
evaluation gates, never elapsed wall time.

Recommended curriculum:

1. live-A, three players, fixed heuristic opponents;
2. charts A-E, three players;
3. charts A-E, three to five players, objectives enabled and disabled;
4. bounded `RulesetVariationSampler` support covering configured resource
   counts, action decks, starting cash, private-hand sizes, value charts, and
   objective pools.

Because `PocketRocksEnv` currently has a fixed player count, the trainer keeps
an environment factory per supported count and deterministically chooses a
factory per episode. It does not mutate player count inside an active
environment.

Reserve disjoint root-seed ranges and at least one ruleset combination from
each later stage for evaluation. Training success must be reported separately
for seen and held-out seeds/rulesets.

## Opponents and checkpoint league

### Bootstrap phase

Start against frozen importable brains:

- aggressive heuristic;
- balanced heuristic;
- passive heuristic;
- a smaller random-bot share for coverage.

The initial three-player smoke lineup is learner, balanced, and passive. Rotate
the learner seat. Heuristic imitation may initialize the policy before PPO,
but PPO must also work from seeded random initialization.

### League phase

After the bootstrap evaluation gate:

1. collect learner episodes against a fixed opponent mixture;
2. update with PPO;
3. save a candidate checkpoint on a configured update interval;
4. evaluate candidate and incumbent on paired rulesets, seeds, and seat
   rotations;
5. promote qualifying candidates;
6. add immutable promoted snapshots to the league.

The default league mixture should retain heuristic opponents while adding the
current champion and older promoted policies. Exact weights are run
configuration. Do not train against only the latest checkpoint.

Promotion uses unshaped paired evaluation utility as the primary scalar:

```text
normalized final-money delta + configured first-place share
```

Also report outright wins, tied firsts, mean rank, final money, faults, pass
rate, bid size, and action wins. A candidate that improves shaped return but
regresses raw evaluation utility is not promoted.

The outer loop resembles AlphaZero's iterative self-play/checkpoint cycle, but
PPO—not search visits—supplies policy improvement targets. The league provides
the frozen, diverse opponent population inspired by population-based
self-play.

## Neural brain and live adapter

### `NeuralBotBrain`

`NeuralBotBrain` is synchronous and owns:

- an inference-only checkpoint bundle;
- neural observation and public-history encoders;
- the policy in evaluation mode;
- a configured device, CPU by default.

It returns an SDK `BotDecision`, validates it against the context, and raises a
specific neural inference error rather than substituting a different policy.
Simulator fault mode decides whether an error becomes a pass.

For a league opponent, a top-level serializable
`CheckpointBrainFactory(checkpoint_path, device)` constructs the brain.
Avoid closures so `BotSpec` remains usable by spawned Monte Carlo workers.

### History-aware brain protocol

Do not break stateless random and heuristic brains. Add an optional
history-aware protocol:

```python
class HistoryAwareBotBrain(Protocol):
    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision: ...
```

`MatchRunner` and `PocketRocksEnv` call this method when supported and otherwise
use the existing `BotBrain` method. `PocketRocksFastBot.choose_raw_decision`
performs the same dispatch after adapting SDK public events.

`NeuralBotBrain.choose_decision` without history is not a silent fallback; it
raises a missing-history error. This prevents simulator/live behavior drift.

### Live wrapper

`NeuralBot` subclasses `PocketRocksFastBot`, overrides the raw decision path,
and binds one registered bot ID to one inference checkpoint. The initial class
uses an explicitly documented development-only ID and is not exposed as a
live console command until the user supplies a registered ID.

Live inference:

- loads the checkpoint once at startup;
- runs under inference mode on CPU unless configured otherwise;
- reconstructs the complete public event sequence per request;
- canonicalizes relative to `context.bot_seat`;
- selects deterministic masked argmax;
- validates and returns the SDK decision before its deadline.

## Checkpoints

Use a versioned checkpoint directory, not an opaque pickled application
object:

```text
checkpoint/
  manifest.json
  model.pt
  optimizer.pt       # resumable training bundle only
  metrics.json
```

The manifest contains:

- checkpoint and encoder schema versions;
- repository commit;
- Python, PyTorch, NumPy, and SDK versions;
- model architecture configuration;
- `EnvironmentBounds`, normalization limits, and maximum history length;
- action-codec and observation-schema hashes;
- reward, PPO, curriculum, and opponent-sampling configuration;
- global environment steps, episodes, PPO updates, and lineage;
- ruleset-support description;
- root seed and Python/NumPy/Torch RNG states for resumable bundles;
- league membership and champion identity;
- a canonical parameter digest computed from ordered tensor names, dtypes,
  shapes, and bytes;
- checksums for checkpoint files.

Store model and optimizer `state_dict`s only. Load only trusted local training
bundles and use PyTorch's weights-only loading where supported. An inference
bundle omits optimizer and RNG state. Write to a temporary sibling directory,
fsync, then atomically rename so interruption cannot leave a promoted partial
checkpoint.

Loading rejects incompatible schema hashes, insufficient bounds, unsupported
architecture versions, corrupt checksums, or non-finite weights.

## CLI and artifacts

The end-state interface is one training command with subcommands:

```text
garboid-train smoke
garboid-train train --config CONFIG.json --output-dir ARTIFACT_DIR
garboid-train resume --checkpoint CHECKPOINT --output-dir ARTIFACT_DIR
garboid-train evaluate --checkpoint CHECKPOINT --config EVAL.json
garboid-train inspect --checkpoint CHECKPOINT
```

JSON keeps configuration in the standard library and avoids another runtime
dependency. CLI flags may override seed, device, and output directory; the
resolved configuration is written into every run directory.

Default artifact directories are gitignored and contain:

- resolved configuration;
- structured update/evaluation metrics;
- promoted and candidate checkpoints;
- league manifest;
- deterministic evaluation JSON;
- optional replay samples selected by seed.

The command never overwrites a nonempty run directory unless `resume` names
the exact compatible checkpoint. Interrupt handling eventually saves a
clearly marked non-promoted recovery checkpoint after the current atomic
update. The staged acceptance section below defines which commands and
artifact guarantees belong to each implementation stage.

## PyTorch dependency strategy

Keep the simulator lightweight. Add PyTorch as an optional `neural` project
extra, not a core dependency:

```toml
[project.optional-dependencies]
neural = [
  "torch>=2.13,<2.14",
]
```

Use `uv sync --extra neural` for training and neural inference. Import PyTorch
lazily so random bots, heuristic bots, the simulator, and core tests still run
without the extra. Do not add torchvision, torchaudio, a training framework, or
a distributed runtime.

Keep torch-dependent modules in the neural package boundary. The core mypy job
excludes that package and verifies only typed, torch-free protocols exposed to
the rest of the application; the neural CI job installs the extra and
type-checks the complete source tree and neural tests.

PyTorch 2.13 provides stable CPython 3.14 wheels. The first lock and CI neural
job must prove installation on the repository's `>=3.14,<3.15` constraint.
The deterministic smoke uses standard CPU wheels. GPU/CUDA-specific indexes
are a later documented installation choice and must not change the lock used
by the CPU reference run.

CI has:

- the existing core job without the neural extra;
- one Python 3.14 CPU neural job with the extra and focused smoke-contract
  tests.

## Deterministic low-iteration smoke

The first executable milestone is deliberately small:

- device: CPU;
- deterministic PyTorch algorithms enabled;
- one Torch thread;
- all Python, NumPy, Torch, environment, and opponent seeds derived from root
  seed 42;
- live-A, three players;
- learner seat rotated;
- balanced and passive frozen opponents;
- GRU hidden size 64;
- `gamma=1.0`, `lambda=0.95`;
- two PPO updates;
- 16 complete games per update;
- one PPO epoch per update;
- no event or placement shaping.

The smoke is a mechanics test, not evidence of strategy quality. It must assert:

- every game terminates without engine or bot faults;
- every sampled action is legal;
- illegal-action probability is exactly zero after masking;
- logits, values, advantages, losses, and gradients are finite;
- at least one trainable parameter changes after each update;
- the checkpoint reloads and reproduces fixture logits, value, and greedy
  decision;
- two runs with the same seed on the same host architecture and pinned lock
  produce identical planned episodes, deterministic metric fields, and
  canonical parameter digest.

Exact comparison excludes elapsed time, filesystem paths, and other
environmental metadata. If a platform kernel prevents bit-for-bit tensor
determinism, the smoke remains CPU-only and fails with a documented
unsupported-operation error rather than silently weakening determinism.

## Test strategy

Implementation follows test-driven development.

### Public-information tests

- changing hidden decks or opponent hands without changing the public view
  leaves neural inputs and outputs unchanged;
- unresolved sealed bids never enter another player's history;
- simulator and pinned-SDK raw fixtures encode identical public histories;
- malformed or missing raw history fails closed;
- incremental and full-history GRU evaluation agree.

### Encoding and symmetry tests

- learner-relative seat rotation is metamorphic across 3-5 players;
- seat, hand, and history masks cover exactly valid entries;
- every supported ruleset fits checkpoint bounds;
- out-of-envelope rulesets fail before rollout;
- all charts and objective configurations produce finite tensors;
- action-head mapping round-trips through `ActionCodec`.

### Model and masking tests

- output shapes match configured bounds;
- bid and reveal heads activate only in their phase;
- masked logits yield zero illegal probability;
- greedy tie-breaking is stable;
- batched and single inference agree;
- loaded fixture checkpoints reproduce outputs.

### Reward and PPO tests

- unshaped accounting plus terminal remainder reconciles to normalized final
  money;
- tied winners divide the win bonus;
- shaping components remain separately reported;
- GAE matches a hand-calculated terminated and truncated trajectory;
- an intervening reveal-only learner decision does not introduce terminal
  discount;
- PPO clipping, entropy, value loss, and padding masks match hand calculations;
- old and new log probabilities use identical action masks.

### League and checkpoint tests

- episode plans are independent of worker scheduling;
- opponent checkpoints remain frozen during collection;
- paired evaluation rotates seats and uses identical seeds;
- promotion ignores shaped-return-only improvement;
- factories are picklable under spawned workers;
- atomic save/load, checksums, schema rejection, and resume RNG restoration work.
- update-boundary resume matches the uninterrupted next update on the pinned
  same-platform environment.

### Integration tests

- the deterministic smoke contract above;
- a neural `BotSpec` completes simulator games with zero illegal decisions;
- live raw adapter fixtures return legal SDK decisions;
- fixed and varied rulesets run for 3-5 players;
- core tests pass when PyTorch is absent;
- focused neural tests pass on Python 3.14 CPU.

## Staged implementation and acceptance

The design describes the complete direction, but implementation is gated in
four stages so infrastructure does not block the first learning proof.

### Stage 1: deterministic PPO mechanics

- project-owned public-history schema plus SDK and simulator adapters;
- learner-relative neural encoder, GRU policy/value model, and legal masking;
- fixed live-A, three-player rollout against balanced and passive heuristics;
- `gamma=1.0` GAE and one-epoch PPO updates;
- `garboid-train smoke`;
- minimal inference checkpoint containing the manifest and model state;
- the two-update, 16-games-per-update deterministic smoke contract.

Stage 1 is complete when its focused public-information, encoding, masking,
GAE/PPO, checkpoint-reload, and smoke tests pass. It does not require a league,
resume, variable player counts, a registered live bot, or evidence of playing
strength.

### Stage 2: durable runs and evaluation

- atomic resumable bundles with optimizer and RNG state;
- update-boundary resume equivalence;
- structured artifacts;
- `train`, `resume`, `evaluate`, and `inspect` subcommands;
- paired held-out evaluation and uncertainty reporting.

### Stage 3: curriculum and checkpoint league

- charts A-E and three-to-five-player environment factories;
- bounded variable-ruleset curriculum and held-out rulesets;
- frozen checkpoint opponent factories;
- promotion gates and a mixed historical league.

### Stage 4: live deployment and scale acceptance

- registered `NeuralBot` wrapper through the SDK raw callback;
- live/simulator history parity fixtures at the deployment boundary;
- one inference bundle exercised through both adapters;
- a fixed 1,000-game evaluation with zero illegal decisions and bot faults.

## Evaluation and success criteria

Separate engineering acceptance from playing strength.

End-state engineering acceptance requires:

- all public-information, masking, symmetry, PPO, checkpoint, and smoke tests
  pass;
- smoke determinism and update-boundary resume equivalence pass;
- zero illegal decisions and bot faults in a fixed 1,000-game evaluation;
- one checkpoint runs through both simulator and SDK raw adapters;
- core installation remains PyTorch-free.

Learning acceptance for the first pilot requires:

- a candidate changes behavior from initialization without collapsing to an
  always-pass or always-max-bid policy;
- raw normalized final money and first-place share are reported on held-out
  paired seeds for every chart and player count reached by the curriculum;
- the candidate meets its configured promotion gate against the incumbent;
- results include confidence intervals or paired bootstrap intervals, not only
  one win-rate point estimate.

No smoke test asserts that two PPO updates beat a heuristic. A longer run may
be called stronger only from the held-out raw score metrics, never training
return alone.

## Implementation ambiguities and recommendations

### SDK raw-frame stability

Ambiguity: cumulative public events are available to
`choose_raw_decision(frame, context)`, but the public callback types `frame` as
`object` and the concrete event classes live under the pinned SDK's internal
package.

Recommendation: use a small structural adapter owned by this project, never
import SDK internal classes into model/trainer code, pin conformance fixtures,
and fail closed on schema drift. This is preferable to dropping history or
maintaining unsafe mutable live GRU state.

### Reward timing

Ambiguity: giving the same final-money objective through intermediate
accounting deltas changes discounted timing compared with a terminal-only
reward.

Recommendation: keep the existing telescoping accounting reward as the first
default because it supplies the requested intermediate signal, retain terminal
money/win metrics as promotion authority, and run a terminal-only ablation
before interpreting strength. Keep arbitrary event shaping off initially.

### League promotion threshold

Ambiguity: an exact confidence threshold and evaluation game count are compute-
budget choices, not architectural facts.

Recommendation: configure them explicitly; use paired seeds/seat rotations and
unshaped evaluation utility as the gate. Never promote from training return.

### Variable-ruleset envelope

Ambiguity: future deck sizes and cash limits can exceed a trained model's fixed
spaces.

Recommendation: derive and checkpoint bounds from a declared finite curriculum
support. Reject larger live rulesets and retrain or migrate the model rather
than clipping observations or bids.

### Initial model scale

Ambiguity: final embedding and trunk widths require measurement.

Recommendation: fix only the smoke GRU size at 64, expose all widths in
configuration, and increase capacity only after profiling inference latency and
underfitting. Do not begin with a transformer or distributed learner.

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [ReBeL: A General Game-Theoretic Reinforcement Learning Framework](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
- [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z)
- [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
- [PyTorch release compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Pinned PocketRocks SDK bot raw-decision API](https://github.com/jaiparera/pocketrocks-python-sdk/blob/597857446d47ac0890609a4767cad561578a2519/src/pocketrocks/bot.py)
- [Existing simulator/RL design](2026-07-28-simulator-rl-design.md)
- [Heuristic bot design](2026-07-28-heuristic-bots-design.md)
