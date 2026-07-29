# Neural PPO Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic Stage 1 proof for a stateless full-history GRU policy/value network trained by legal-action-masked PPO for two updates of 16 live-A three-player games.

**Architecture:** Keep public-history types and adapters Torch-free, and keep every Torch import inside `garboid_pocketrocks.neural`. `PocketRocksEnv` remains the game authority and exposes its current SDK context, public ruleset knowledge, and simulator-produced public history to a collector; the collector encodes complete learner-relative histories, samples only legal universal actions, and hands complete trajectories to gamma-1 GAE and one-epoch PPO. A minimal inference checkpoint stores only a versioned manifest and model state, while durable/resumable artifacts, leagues, variable-ruleset training, and live deployment remain outside Stage 1.

**Tech Stack:** Python 3.14, uv, optional PyTorch `>=2.13,<2.14` CPU extra, NumPy, Gymnasium, the pinned PocketRocks SDK, pytest, mypy, Ruff, GitHub Actions.

## Global Constraints

- Implement only Stage 1 from `docs/superpowers/specs/2026-07-28-neural-self-play-design.md`.
- The Stage 1 runtime is fixed to live-A, three players, learner seat rotation, balanced and passive frozen opponents, CPU, one Torch thread, GRU hidden size 64, `gamma=1.0`, `lambda=0.95`, two PPO updates, 16 complete games per update, and one PPO epoch per update.
- Use root seed `42`; derive Python, NumPy, Torch, environment, opponent, policy-sampling, and minibatch seeds through named, disjoint namespaces.
- The actor and value function may consume only current `DecisionContext`, public `RulesetKnowledge`, cumulative public history, the learner's own hand, and the current legal-action mask.
- Never expose opponent hands, deck order, RNG state, unresolved sealed bids, engine-only financial/card objects, or terminal scores before termination.
- Every inference replays the complete padded history through a zero-initialized GRU. Do not add mutable hidden-state caching, truncation, burn-in, or incremental recurrence.
- Preserve the universal `ActionCodec` mapping. Mask inactive heads and illegal universal actions to negative infinity for sampling, greedy selection, entropy, log probabilities, and PPO ratios.
- Training sampling must call `torch.multinomial(probabilities, 1, generator=policy_generator)`. Evaluation uses `torch.argmax`, whose first-maximum behavior supplies lowest-universal-index tie-breaking.
- Keep PyTorch optional. Core installation, imports, type-checking, simulation, random bots, and heuristic bots must work without the `neural` extra.
- Stage 1 checkpoints contain `manifest.json` and `model.pt` only. Do not add optimizer/RNG resume state, atomic promotion, league metadata, paired evaluation, or live bot IDs.
- Follow strict red-green-refactor TDD for every behavior change. Run the named RED command before editing production code and confirm the expected failure.
- Use `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache` for every uv command.
- Do not weaken deterministic-algorithm errors. An unsupported deterministic CPU kernel is a hard smoke failure.

## Locked file map

Core, Torch-free files:

- `src/garboid_pocketrocks/adapters/public_history.py` — immutable canonical public-event schema and structural SDK raw-frame adapter.
- `src/garboid_pocketrocks/adapters/simulator_history.py` — stateful conversion of completed engine transition batches; publishes bids only from batches that contain auction resolution.
- `src/garboid_pocketrocks/training/single_agent_env.py` — exposes the current learner context, knowledge, and immutable public-history snapshot; accepts an explicit opponent seed.
- `tests/adapters/test_public_history.py` — SDK failure-closed and simulator/SDK parity tests.
- `tests/training/test_single_agent_env.py` — environment history and explicit opponent-seed regression tests.

Torch boundary:

- `src/garboid_pocketrocks/neural/__init__.py` — empty/lazy namespace; importing it does not import Torch.
- `src/garboid_pocketrocks/neural/config.py` — JSON-safe encoder/model configurations and the exact Stage 1 encoder/model defaults.
- `src/garboid_pocketrocks/neural/encoding.py` — learner-relative NumPy encoding and batch tensorization.
- `src/garboid_pocketrocks/neural/model.py` — shared snapshot/seat/history encoder, one-layer GRU, trunk, two policy heads, scalar value.
- `src/garboid_pocketrocks/neural/policy.py` — phase-head projection, legal masking, generated sampling, deterministic greedy action.
- `src/garboid_pocketrocks/neural/advantages.py` — terminated/truncated-aware gamma-1 GAE.
- `src/garboid_pocketrocks/neural/seeding.py` — stable named seed derivation and deterministic Torch setup.
- `src/garboid_pocketrocks/neural/rollout.py` — fixed live-A episode plans and learner-only rollout collection.
- `src/garboid_pocketrocks/neural/ppo.py` — clipped loss and one-epoch update.
- `src/garboid_pocketrocks/neural/checkpoint.py` — Stage 1 manifest/model-only inference save/load and parameter digest.
- `src/garboid_pocketrocks/neural/smoke.py` — two-by-sixteen deterministic orchestration and comparable result payload.
- `src/garboid_pocketrocks/neural/cli.py` — `garboid-train smoke`.
- `tests/neural/` — Torch-extra tests; each module calls `pytest.importorskip("torch")` before importing Torch-dependent project modules.

Configuration and documentation:

- `pyproject.toml` / `uv.lock` — optional neural extra, CLI entry point, pytest marker, and core mypy exclusion.
- `mypy.neural.ini` — full strict neural type-check configuration.
- `.github/workflows/ci.yml` — unchanged core-without-Torch job plus a separate Python 3.14 CPU neural job.
- `README.md` — Stage 1 install, smoke command, checkpoint limits, and non-strength disclaimer.

## Stable interfaces

The tasks below must use these names and signatures exactly:

```python
# adapters/public_history.py
PublicHistory = tuple[PublicEvent, ...]


def public_history_from_sdk_frame(frame: object) -> PublicHistory: ...


# adapters/simulator_history.py
class SimulatorPublicHistoryAdapter:
    @classmethod
    def from_initial_transition(
        cls, transition: EngineTransition
    ) -> SimulatorPublicHistoryAdapter: ...
    def append(self, events: Sequence[GameEvent]) -> None: ...
    @property
    def history(self) -> PublicHistory: ...


# training/single_agent_env.py
@property
def learner_context(self) -> DecisionContext: ...
@property
def ruleset_knowledge(self) -> RulesetKnowledge: ...
@property
def public_history(self) -> PublicHistory: ...


# neural/encoding.py
class NeuralObservationEncoder:
    def encode(
        self,
        context: DecisionContext,
        knowledge: RulesetKnowledge,
        history: PublicHistory,
    ) -> NeuralObservation: ...


def batch_observations(
    observations: Sequence[NeuralObservation], device: torch.device
) -> NeuralBatch: ...


# neural/model.py and neural/policy.py
class NeuralPolicy(nn.Module):
    def __init__(
        self,
        encoder_config: NeuralEncoderConfig,
        model_config: NeuralModelConfig,
    ) -> None: ...
    def forward(self, batch: NeuralBatch) -> PolicyValueOutput: ...


def evaluate_masked_policy(
    output: PolicyValueOutput,
    batch: NeuralBatch,
    *,
    generator: torch.Generator | None,
    deterministic: bool,
) -> PolicySelection: ...


# neural/advantages.py and neural/ppo.py
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> AdvantageBatch: ...


class PPOTrainer:
    def __init__(self, model: NeuralPolicy, config: PPOConfig) -> None: ...
    def update(self, rollout: RolloutBatch, *, update_seed: int) -> PPOUpdateMetrics: ...


# neural/checkpoint.py and neural/smoke.py
def save_inference_checkpoint(
    path: Path, model: NeuralPolicy, manifest: InferenceManifest
) -> None: ...
def load_inference_checkpoint(path: Path, *, device: torch.device) -> LoadedInferenceCheckpoint: ...
def run_smoke(config: SmokeConfig, output_dir: Path) -> SmokeResult: ...
```

## Parallel boundaries

```text
Task 1 dependency/CI boundary
  -> Task 2 public schema + SDK adapter
      -> Task 3 simulator adapter + environment exposure
          -> Task 4 canonical encoder
              -> Task 5 model/masking
                  -> Task 7 rollout collector
                      -> Task 8 PPO update
                          -> Task 10 smoke/CLI/docs

Task 6 GAE may run in parallel with Tasks 3-5 after Task 1.
Task 9 checkpoint may run in parallel with Tasks 7-8 after Task 5.
```

Tasks that share a file must not run concurrently. In particular, Task 1 and Task 10 both touch `pyproject.toml` and CI; Task 3 alone owns `single_agent_env.py`; Tasks 5, 8, and 9 consume but do not concurrently edit one another's files.

---

### Task 1: Optional PyTorch extra and separate quality boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`
- Create: `mypy.neural.ini`
- Create: `src/garboid_pocketrocks/neural/__init__.py`
- Create: `tests/test_neural_packaging.py`
- Create: `tests/neural/test_torch_runtime.py`

**Interfaces:**
- Consumes: the existing Python `>=3.14,<3.15` requirement and core `uv sync --locked` job.
- Produces: installable extra `neural`, a Torch-free import boundary, and separate core/full mypy commands.

- [ ] **Step 1: Write the failing packaging test**

```python
# tests/test_neural_packaging.py
from pathlib import Path
import subprocess
import sys
import tomllib


def test_neural_extra_is_optional_and_version_bounded() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["neural"] == ["torch>=2.13,<2.14"]
    assert "torch" not in "\n".join(project["project"]["dependencies"]).lower()


def test_core_training_import_does_not_import_torch() -> None:
    code = "import sys; import garboid_pocketrocks.training; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/test_neural_packaging.py -q`

Expected: FAIL because `project.optional-dependencies.neural` does not exist.

- [ ] **Step 3: Add the optional dependency and type-check split**

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
neural = [
  "torch>=2.13,<2.14",
]

[tool.mypy]
python_version = "3.14"
strict = true
files = ["src", "tests"]
exclude = [
  "^src/garboid_pocketrocks/neural/",
  "^tests/neural/",
]
warn_unused_configs = true
```

Create `mypy.neural.ini`:

```ini
[mypy]
python_version = 3.14
strict = True
files = src,tests
warn_unused_configs = True
```

Keep `src/garboid_pocketrocks/neural/__init__.py` to a docstring and `__all__: list[str] = []`; do not import Torch or other neural modules there.

- [ ] **Step 4: Add the neural-runtime test**

```python
# tests/neural/test_torch_runtime.py
import pytest

torch = pytest.importorskip("torch")


def test_neural_extra_supplies_supported_cpu_torch() -> None:
    major, minor, *_ = (int(part) for part in torch.__version__.split("+")[0].split("."))
    assert (major, minor) == (2, 13)
    assert torch.tensor([1.0], device="cpu").device.type == "cpu"
```

- [ ] **Step 5: Lock and prove both installations**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv sync --locked
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/test_neural_packaging.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv sync --locked --extra neural
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_torch_runtime.py -q
```

Expected: both tests PASS; the lock resolves a Python 3.14-compatible Torch 2.13 CPU artifact.

- [ ] **Step 6: Split CI without changing the core install**

Keep the existing job on `uv sync --locked`, and change its type-check step to continue using `uv run mypy src tests` under the new exclusion. Add:

```yaml
  neural:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v6
      - name: Install uv and Python
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b
        with:
          version: "0.11.26"
          python-version: "3.14"
          enable-cache: true
      - name: Synchronize neural dependencies
        run: uv sync --locked --extra neural
      - name: Type-check complete neural tree
        run: uv run --extra neural mypy --config-file mypy.neural.ini src tests
      - name: Run neural tests
        run: uv run --extra neural pytest tests/neural -q
```

- [ ] **Step 7: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/test_neural_packaging.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural mypy --config-file mypy.neural.ini src tests
```

Expected: PASS.

Commit:

```bash
git add pyproject.toml uv.lock mypy.neural.ini .github/workflows/ci.yml src/garboid_pocketrocks/neural/__init__.py tests/test_neural_packaging.py tests/neural/test_torch_runtime.py
git commit -m "chore: add optional neural runtime"
```

### Task 2: Canonical public-history schema and failure-closed SDK adapter

**Files:**
- Create: `src/garboid_pocketrocks/adapters/public_history.py`
- Modify: `src/garboid_pocketrocks/adapters/__init__.py`
- Create: `tests/adapters/test_public_history.py`

**Interfaces:**
- Consumes: only structurally accessed public fields from the pinned SDK raw decision frame.
- Produces: `PublicEvent`, `PublicHistory`, `PublicHistoryCompatibilityError`, and `public_history_from_sdk_frame(frame)`.

- [ ] **Step 1: Write SDK adapter RED tests**

Cover these exact assertions in `tests/adapters/test_public_history.py`:

```python
def test_sdk_frame_becomes_immutable_public_history() -> None:
    frame = decode_frame(
        scenario(players=3, starting_cash=30, initial_tiebreak_seat=0)
        .turn(ActionId.AUCTION1, resources=(Suit.BRICK, 0))
        .auction((2, 5, 1))
        .reveal(Suit.ORE)
        .turn(ActionId.LOAN10)
        .deciding(seat=2, hand=(Suit.WOOD,))
        .to_bytes(deadline_at=1000)
    )
    history = public_history_from_sdk_frame(frame)
    assert [event.kind for event in history] == [
        PublicEventKind.GAME_SETUP,
        PublicEventKind.TURN_OPENED,
        PublicEventKind.AUCTION_RESOLVED,
        PublicEventKind.INFORMATION_REVEALED,
        PublicEventKind.TURN_OPENED,
    ]
    assert cast(PublicAuctionResolved, history[2]).bids_by_seat == (2, 5, 1)
    assert cast(PublicInformationRevealed, history[3]).seat == 1


@pytest.mark.parametrize(
    "frame",
    [
        object(),
        SimpleNamespace(common_events=()),
        SimpleNamespace(common_events=(SimpleNamespace(kind="gameSetup", player_count=3),)),
    ],
)
def test_sdk_adapter_fails_closed_on_missing_or_malformed_history(frame: object) -> None:
    with pytest.raises(PublicHistoryCompatibilityError):
        public_history_from_sdk_frame(frame)
```

Use `decode_frame` only in the test fixture. Production code must not import `pocketrocks.internal`.

- [ ] **Step 2: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/adapters/test_public_history.py -q`

Expected: collection FAIL because `adapters.public_history` does not exist.

- [ ] **Step 3: Implement the immutable schema**

Define four frozen, slotted dataclasses:

```python
class PublicEventKind(StrEnum):
    GAME_SETUP = "game_setup"
    TURN_OPENED = "turn_opened"
    AUCTION_RESOLVED = "auction_resolved"
    INFORMATION_REVEALED = "information_revealed"


@dataclass(frozen=True, slots=True)
class PublicGameSetup:
    kind: Literal[PublicEventKind.GAME_SETUP]
    player_count: int
    starting_cash: int
    value_chart: tuple[int, ...]
    initial_tiebreak_seat: int
    objective_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublicTurnOpened:
    kind: Literal[PublicEventKind.TURN_OPENED]
    action_id: int
    resource_ids: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PublicAuctionResolved:
    kind: Literal[PublicEventKind.AUCTION_RESOLVED]
    bids_by_seat: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublicInformationRevealed:
    kind: Literal[PublicEventKind.INFORMATION_REVEALED]
    seat: int
    suit_id: int
```

Set `PublicEvent` to their union and `PublicHistory = tuple[PublicEvent, ...]`.

- [ ] **Step 4: Implement structural parsing and validation**

`public_history_from_sdk_frame` must:

1. require a tuple/list `common_events`;
2. require exactly one first `gameSetup`;
3. validate player count 3–5, chart length 6, seats, suit/action IDs, bid tuple length, and nonnegative bids;
4. track current tiebreak seat;
5. derive an auction winner clockwise from the current tiebreak, update tiebreak, and attach that seat to the following `infoRevealed`;
6. reject `infoRevealed` before an auction winner exists;
7. reject unknown event kinds rather than dropping them.

Use small `_require_attr`, `_require_int`, and `_winning_seat` helpers. Wrap `AttributeError`, `TypeError`, and `ValueError` as `PublicHistoryCompatibilityError` with the event index.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/adapters/test_public_history.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy src/garboid_pocketrocks/adapters tests/adapters
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/adapters tests/adapters
git commit -m "feat: add canonical SDK public history"
```

### Task 3: Simulator history adapter and environment observation boundary

**Files:**
- Create: `src/garboid_pocketrocks/adapters/simulator_history.py`
- Modify: `src/garboid_pocketrocks/adapters/__init__.py`
- Modify: `src/garboid_pocketrocks/training/single_agent_env.py`
- Modify: `tests/adapters/test_public_history.py`
- Modify: `tests/training/test_single_agent_env.py`

**Interfaces:**
- Consumes: Task 2 public events, `EngineTransition.events`, and `PocketRocksEnv`.
- Produces: a no-leak simulator adapter, parity with SDK fixtures, and read-only environment properties used by Task 7.

- [ ] **Step 1: Write sealed-bid and parity RED tests**

Build a simulator event fixture matching the Task 2 SDK narration. Assert:

```python
adapter = SimulatorPublicHistoryAdapter.from_initial_transition(started)
adapter.append(
    (
        GameEvent(EventKind.DECISION_SUBMITTED, seat=0, amount=2),
        GameEvent(EventKind.DECISION_SUBMITTED, seat=1, amount=5),
        GameEvent(EventKind.DECISION_SUBMITTED, seat=2, amount=1),
        GameEvent(EventKind.AUCTION_RESOLVED, seat=1, amount=5),
    )
)
assert cast(PublicAuctionResolved, adapter.history[-1]).bids_by_seat == (2, 5, 1)
```

Also assert that the SDK frame and equivalent simulator sequence produce
exactly equal `PublicHistory`, and that missing/duplicate seats at resolution
raise `SimulatorHistoryError`. Add one real auction-win-to-reveal engine
integration test. It must show that the reveal-choice `DECISION_SUBMITTED` is
not buffered as a bid and that the following `INFORMATION_REVEALED` is
appended normally. Include a passed bid and assert its `amount=None` becomes
public bid `0`.

- [ ] **Step 2: Write environment RED tests**

After `env.reset(seed=3, options={"opponent_seed": 91})`, assert:

```python
assert env.learner_context.bot_seat == env.learner_seat
assert env.ruleset_knowledge == LIVE_RULESET.knowledge(3)
assert env.public_history[0].kind is PublicEventKind.GAME_SETUP
before = env.public_history
action = int(np.flatnonzero(observation["action_mask"])[0])
env.step(action)
assert env.public_history[: len(before)] == before
```

Reset two environments with the same environment seed and opponent seed and assert equal transition/reward sequences. Use a test-only `RecordingBrainFactory` callable that records the seed received by `BotSpec.make_brain`; change only `opponent_seed` and assert the recorded opponent seeds change while the initial engine state remains equal.

- [ ] **Step 3: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/adapters/test_public_history.py tests/training/test_single_agent_env.py -q`

Expected: FAIL because the simulator adapter and environment properties do not exist.

- [ ] **Step 4: Implement `SimulatorPublicHistoryAdapter`**

At `from_initial_transition`, read `transition.state.ruleset`, player count, initial priority, setup objectives, and the initial `TURN_OPENED`. Process each later transition batch as a unit:

- when the batch contains `AUCTION_RESOLVED`, collect its
  `DECISION_SUBMITTED` events as bids, normalize `amount=None` to bid `0`,
  require every seat exactly once, append one `PublicAuctionResolved`, and
  update priority from the resolved event seat;
- in a reveal transition batch, ignore the reveal-choice
  `DECISION_SUBMITTED`; it is a private hand index, not a public bid;
- convert `INFORMATION_REVEALED` to `(seat, suit_id)`;
- convert `TURN_OPENED`;
- ignore all other completed public-accounting engine events;
- reject missing or duplicate bid seats in a resolution batch.

Return `tuple(self._events)` from `history` so callers cannot mutate adapter state.

- [ ] **Step 5: Integrate the adapter into `PocketRocksEnv`**

In `reset`, stop discarding `options`. Validate that `options` contains only optional integer `opponent_seed`; default it to the environment root seed for backward compatibility. Initialize history from the starting transition. In `_resolve_batch`, append every new transition's events before exposing the next learner context. Add guarded properties:

```python
@property
def learner_context(self) -> DecisionContext:
    if self._learner_context is None:
        raise RuntimeError("environment must be reset before observing")
    return self._learner_context
```

Repeat that pattern for `ruleset_knowledge` and `public_history`. Seed `_make_opponent_brains` from the explicit opponent seed, not the seat-selection RNG.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/adapters/test_public_history.py tests/training/test_single_agent_env.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/simulator tests/training -q
```

Expected: PASS; unresolved submitted bids never appear in history.

Commit:

```bash
git add src/garboid_pocketrocks/adapters src/garboid_pocketrocks/training/single_agent_env.py tests/adapters tests/training/test_single_agent_env.py
git commit -m "feat: expose simulator public history"
```

### Task 4: Learner-relative neural observation encoder

**Files:**
- Create: `src/garboid_pocketrocks/neural/config.py`
- Create: `src/garboid_pocketrocks/neural/encoding.py`
- Create: `tests/neural/test_encoding.py`

**Interfaces:**
- Consumes: `DecisionContext`, `RulesetKnowledge`, `PublicHistory`, `ActionCodec`, and `EnvironmentBounds`.
- Produces: `NeuralEncoderConfig`, `NeuralModelConfig`,
  `NeuralObservation`, `NeuralBatch`, `NeuralObservationEncoder`,
  `stage1_encoder_config()`, `stage1_model_config()`, and
  `batch_observations`.

- [ ] **Step 1: Write RED tests for exact shapes and masks**

For live-A use `EnvironmentBounds(max_bid=100, max_hand_size=5)` and history bound:

```python
1 + 2 * sum(LIVE_RULESET.action_counts) + 3 * LIVE_RULESET.setup_for(3).private_cards_per_player
```

which equals `76`. Assert:

- `global_ids.shape == (6,)` for phase, player count, action, resource 0, resource 1, relative priority;
- `global_numeric.shape == (21,)`;
- `objective_bits.shape == (60,)`;
- `seat_numeric.shape == (5, 41)` and `seat_valid == (True, True, True, False, False)`;
- `private_hand_ids.shape == (5,)` and hand order is unchanged;
- `history_ids.shape == (76, 6)`;
- `history_numeric.shape == (76, 42)`;
- `history_valid.sum() == len(history)`;
- `action_mask` equals `ActionCodec.mask(context).astype(bool)`.

Assert every floating tensor is finite and `float32`; IDs are `int64`; masks are `bool`.

- [ ] **Step 2: Write RED learner-rotation tests**

For synthetic equivalent contexts with learner seats 0 through `player_count - 1`, for each `player_count` in `(3, 4, 5)`, construct a test-only `NeuralEncoderConfig` whose declared `supported_player_counts` is `(3, 4, 5)`, rotate all seat-indexed context/history fields, and assert equal encodings. The production `stage1_encoder_config()` remains live-A/3-only. Explicitly assert:

```python
assert encoded.global_ids[5] == (context.tiebreak_seat - context.bot_seat) % player_count
assert encoded.seat_numeric[0, 0] == context.cash_by_seat[context.bot_seat] / config.max_cash
assert encoded.private_hand_ids.tolist() == list(context.current_hand_suit_ids) + padding
```

Keep suit IDs, objective IDs, and hand slot order unchanged.

- [ ] **Step 3: Write RED bounds and information-boundary tests**

Assert:

- changing a hypothetical opponent hand/deck object unavailable to the inputs changes nothing;
- history length 77 raises `NeuralEncodingError("history exceeds checkpoint bound")`;
- cash, chart, counts, hand size, or bid maximum outside config raises before tensorization;
- an all-zero action mask and a mask without universal pass raise;
- an out-of-support checkpoint config (anything except live-A/3 for Stage 1) raises.

- [ ] **Step 4: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_encoding.py -q`

Expected: collection FAIL because `neural.encoding` does not exist.

- [ ] **Step 5: Implement config and NumPy encoding**

Use this fixed field order:

```text
global IDs:
  phase, player_count, current_action, resource_0, resource_1, relative_priority
global numeric:
  starting_cash,
  value_chart[6],
  rules_resource_counts[5],
  rules_action_counts[6],
  rules_private_cards,
  rules_active_objective_count,
  rules_objectives_enabled
objective bits:
  active_objectives[30], rules_objective_pool[30]
seat numeric per relative seat:
  cash, won_resources[5], revealed_info[5], owned_objectives[30]
history IDs per event:
  kind, action, resource_0, resource_1, relative_actor_plus_one, revealed_suit
history numeric per event:
  setup_starting_cash, setup_chart[6], setup_objectives[30], resolved_bids[5]
```

Normalize cash/bids by `max_cash=100`, chart by `max_abs_chart=20`, resource counts by `max_resource_cards=30`, and action counts by `max_action_cards=30`. Encode missing categorical values as zero; reserve relative actor ID zero for “no actor,” hence `relative + 1`.

`stage1_encoder_config()` must return live-A/3 support, the 76-event bound, and the normalization constants above. Validate rather than clip. `batch_observations` stacks arrays and calls `torch.as_tensor` on the requested device.

Define the JSON-safe frozen `NeuralModelConfig` in `config.py` with the exact
Task 5 width fields and a `stage1_model_config()` factory. Test round-tripping
through `dataclasses.asdict` and reconstruction so checkpoint loading does not
depend on module globals.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_encoding.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural mypy --config-file mypy.neural.ini src/garboid_pocketrocks/neural tests/neural
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/config.py src/garboid_pocketrocks/neural/encoding.py tests/neural/test_encoding.py
git commit -m "feat: add canonical neural observations"
```

### Task 5: Stateless GRU policy/value model and generated legal masking

**Files:**
- Create: `src/garboid_pocketrocks/neural/model.py`
- Create: `src/garboid_pocketrocks/neural/policy.py`
- Create: `tests/neural/test_model.py`
- Create: `tests/neural/test_policy.py`

**Interfaces:**
- Consumes: Task 4 `NeuralBatch`, `ActionCodec`, and `NeuralModelConfig`.
- Produces: `NeuralPolicy`, `PolicyValueOutput`, `PolicySelection`, and `evaluate_masked_policy`.

- [ ] **Step 1: Write model RED tests**

With batch sizes 1 and 4, assert:

```python
output.bid_logits.shape == (batch_size, 101)
output.reveal_logits.shape == (batch_size, 6)
output.value.shape == (batch_size,)
```

Assert finite outputs, single/batched equivalence, and repeat-call equality. Change only a padded history token and assert unchanged output. Change a valid history token and assert changed output. The `forward` signature must not accept or return a hidden state.

- [ ] **Step 2: Write masking/sampling RED tests**

For both phases:

- map the active head to universal size `106`;
- set inactive and context-illegal logits to `-inf`;
- assert `probabilities[~action_mask].sum().item() == 0.0`;
- assert sampled actions are legal over 1,000 draws;
- pass two equally maximal legal logits and assert deterministic selection chooses the lowest universal index;
- seed two CPU `torch.Generator`s equally and assert identical sampled sequences;
- monkeypatch `torch.multinomial` with a spy and assert the supplied generator is passed;
- assert raw bid/reveal logits, selected log probability, entropy, and backward
  gradients are finite;
- assert invalid universal logits are exactly `-inf`;
- assert returned selected `log_probability` and entropy come from the same
  masked probabilities.

- [ ] **Step 3: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_model.py tests/neural/test_policy.py -q`

Expected: collection FAIL because model/policy modules do not exist.

- [ ] **Step 4: Implement the model**

Use explicit `NeuralModelConfig` values:

```python
NeuralModelConfig(
    categorical_embedding_size=8,
    suit_embedding_size=4,
    seat_hidden_size=32,
    event_embedding_size=64,
    gru_hidden_size=64,
    snapshot_hidden_size=128,
    trunk_hidden_size=128,
)
```

Use field-specific embeddings, one shared `Linear(41, 32) -> Tanh` seat MLP, one event projection to 64, one single-layer batch-first GRU with hidden size 64, and a zero hidden state allocated inside every `forward`. Gather the last valid GRU output from `history_valid`; never return the GRU state. Concatenate snapshot, all five shared-seat outputs with seat masks, and the GRU summary into a shared two-layer Tanh trunk. Emit bid/reveal/value linear heads. Do not add dropout, normalization with running state, attention, or a centralized critic.

- [ ] **Step 5: Implement phase projection and masking**

Build universal logits as follows:

```python
universal = torch.full((batch_size, codec.size), -torch.inf, device=...)
universal[bid_rows, : bounds.max_bid + 1] = output.bid_logits[bid_rows]
universal[reveal_rows, 0] = output.reveal_logits[reveal_rows, 0]
universal[reveal_rows, bounds.max_bid + 1 :] = output.reveal_logits[reveal_rows, 1:]
masked = universal.masked_fill(~batch.action_mask, -torch.inf)
log_probabilities = torch.log_softmax(masked, dim=-1)
probabilities = torch.softmax(masked, dim=-1)
```

Assert every row enables pass. For stochastic selection call `torch.multinomial(probabilities, 1, generator=generator)`; for deterministic selection call `torch.argmax(masked, dim=-1)`. Return selected actions, gathered log probabilities, probabilities, masked logits, and model values. For entropy, first replace nonfinite entries in `log_probabilities` with zero and then compute `-(probabilities * safe_log_probabilities).sum(-1)`; do not use `torch.where` around a precomputed `0 * -inf` expression.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_model.py tests/neural/test_policy.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural mypy --config-file mypy.neural.ini src/garboid_pocketrocks/neural tests/neural
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/model.py src/garboid_pocketrocks/neural/policy.py tests/neural/test_model.py tests/neural/test_policy.py
git commit -m "feat: add masked recurrent policy"
```

### Task 6: Gamma-1 GAE

**Files:**
- Create: `src/garboid_pocketrocks/neural/advantages.py`
- Create: `tests/neural/test_advantages.py`

**Interfaces:**
- Consumes: learner-step rewards, values, terminal/truncation flags.
- Produces: `AdvantageBatch(advantages, returns)` and `compute_gae`.

- [ ] **Step 1: Write hand-calculated RED tests**

For rewards `[1.0, 0.5]`, values `[0.2, 0.4]`, terminated `[False, True]`, bootstrap `0`, gamma `1`, lambda `.95`, assert advantages `[1.295, 0.1]` and returns `[1.495, 0.5]`.

For a truncated last step, assert its delta includes `bootstrap_value`; for a terminated last step assert it does not. With `gamma=1`, `lambda=1`, insert a zero-reward reveal-only step and assert the undiscounted return before the insertion is unchanged.

Assert mismatched lengths, both terminated and truncated on one step, nonfinite inputs, or gamma other than `1.0` in Stage 1 raise `AdvantageError`.

- [ ] **Step 2: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_advantages.py -q`

Expected: collection FAIL because `neural.advantages` does not exist.

- [ ] **Step 3: Implement reverse recurrence**

Use:

```python
next_value = bootstrap_value
next_advantage = torch.zeros_like(bootstrap_value)
for index in range(length - 1, -1, -1):
    nonterminal = (~terminated[index]).to(values.dtype)
    delta = rewards[index] + gamma * nonterminal * next_value - values[index]
    advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
    advantages[index] = advantage
    next_value = values[index]
    next_advantage = advantage
returns = advantages + values
```

Truncation is metadata, not a bootstrap blocker; require a finite bootstrap value when the last step is truncated.

- [ ] **Step 4: Run GREEN and commit**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_advantages.py -q`

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/advantages.py tests/neural/test_advantages.py
git commit -m "feat: add gamma one GAE"
```

### Task 7: Deterministic live-A rollout collector

**Files:**
- Create: `src/garboid_pocketrocks/neural/seeding.py`
- Create: `src/garboid_pocketrocks/neural/rollout.py`
- Create: `tests/neural/test_rollout.py`

**Interfaces:**
- Consumes: Tasks 3–5 environment/public history/encoder/policy, balanced/passive specs, `RewardBreakdown`.
- Produces: deterministic `EpisodePlan`, `RolloutTransition`, `RolloutEpisode`, `RolloutBatch`, `plan_stage1_episodes`, and `collect_rollout`.

- [ ] **Step 1: Write seed-plan RED tests**

Assert 32 plans for `root_seed=42`, two updates, and 16 games per update. Each plan contains:

```python
EpisodePlan(
    update_index: int,
    episode_index: int,
    learner_seat: int,
    environment_seed: int,
    opponent_seed: int,
    policy_seed: int,
)
```

Assert learner seats rotate `0,1,2,0,...`; every `(namespace, update, episode)` produces a unique unsigned 63-bit seed; repeated planning is identical; changing iteration order does not alter a plan keyed by update/episode.

- [ ] **Step 2: Write rollout RED tests**

Collect two episodes with an initialized model and assert:

- each game terminates and has a `GameResult`;
- opponents are exactly balanced and passive and remain fixed for the episode;
- every chosen action has `action_mask[action] == True`;
- every decoded decision is legal in its stored context;
- `illegal_probability == 0.0`;
- all observations, raw head logits, values, selected log probabilities,
  rewards, and reward-breakdown components are finite, while masked illegal
  universal logits are exactly `-inf`;
- transition metadata includes live-A, player count 3, learner seat, opponent names, and all three seeds;
- only learner decisions are stored;
- final money, rank, outright/tied-first, and each shaping component are stored separately.

Snapshot model parameters before/after collection and assert no change.

- [ ] **Step 3: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_rollout.py -q`

Expected: collection FAIL because seeding/rollout modules do not exist.

- [ ] **Step 4: Implement deterministic setup and planning**

`derive_seed(root_seed, namespace, *indices)` must hash a UTF-8 canonical string with BLAKE2b digest size 8 and clear the sign bit.
`configure_deterministic_torch(root_seed)` must internally derive separate
`"python"`, `"numpy"`, and `"torch"` seeds, seed those three runtimes, call
`torch.use_deterministic_algorithms(True)`, and set one intra-op and one
inter-op thread before work starts. Guard the thread-setting calls with one
module-local initialized flag so the second same-process smoke run reseeds
without illegally resetting Torch's inter-op thread pool.

Use named namespaces `"python"`, `"numpy"`, `"torch"`, `"model"`,
`"environment"`, `"opponent"`, `"policy"`, and `"minibatch"`. Derive model
initialization from `"model"` and each PPO update's shuffle seed from
`"minibatch"` plus its update index. Do not use Python's randomized `hash()`.

- [ ] **Step 5: Implement collection**

For each plan:

1. construct `PocketRocksEnv` with `FixedRulesetSampler(LIVE_RULESET)`, player count 3, bounds `(100, 5)`, the plan learner seat, opponents `(BALANCED_HEURISTIC_BOT_SPEC, PASSIVE_HEURISTIC_BOT_SPEC)`, and the default reward config;
2. call `reset(seed=environment_seed, options={"opponent_seed": opponent_seed})`;
3. encode `env.learner_context`, `env.ruleset_knowledge`, and `env.public_history`;
4. call the frozen model under `torch.no_grad()` and sample with a per-episode CPU generator seeded by `policy_seed`;
5. call `env.step(action)` and store immutable copies of observation arrays plus old masked log probability/value;
6. continue through termination, then store final score metrics from `env.transition.result`.

Do not store the engine state, decks, opponent hands, or unresolved decisions. Convert the list of episodes to `RolloutBatch`, retaining episode boundaries for GAE.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_rollout.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest tests/training/test_single_agent_env.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/seeding.py src/garboid_pocketrocks/neural/rollout.py tests/neural/test_rollout.py
git commit -m "feat: collect deterministic PPO rollouts"
```

### Task 8: One-epoch clipped PPO update

**Files:**
- Create: `src/garboid_pocketrocks/neural/ppo.py`
- Create: `tests/neural/test_ppo.py`

**Interfaces:**
- Consumes: rollout episode boundaries, Task 6 GAE, Task 5 masked policy.
- Produces: `PPOConfig`, `PPOLoss`, `PPOUpdateMetrics`, `ppo_loss`, and `PPOTrainer.update`.

- [ ] **Step 1: Write clipped-loss RED tests**

Use small tensors to assert:

```python
ratio = torch.exp(new_log_probability - old_log_probability)
policy_loss = -torch.minimum(
    ratio * advantage,
    torch.clamp(ratio, 0.8, 1.2) * advantage,
).mean()
value_loss = 0.5 * torch.square(new_value - return_target).mean()
total = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
```

Assert padding is absent from the flattened batch, advantage normalization uses all and only valid learner transitions, and recomputation uses the stored action mask. Change an illegal logit and assert unchanged new log probability/loss.

- [ ] **Step 2: Write update RED tests**

Collect a two-episode rollout, update once, and assert:

- exactly one epoch;
- every loss, ratio, entropy, value, advantage, and gradient norm is finite;
- the pre-clip gradient norm returned by `clip_grad_norm_` is reported
  separately, and a freshly computed post-clip norm is `<=0.5`;
- at least one trainable parameter changes;
- a second update using an independently identical initialized
  model/trainer, rollout, and minibatch seed produces identical metrics and
  state tensors;
- the rollout's old log probabilities/values remain unchanged.

- [ ] **Step 3: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_ppo.py -q`

Expected: collection FAIL because `neural.ppo` does not exist.

- [ ] **Step 4: Implement PPO**

Use exact Stage 1 defaults:

```python
PPOConfig(
    gamma=1.0,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coefficient=0.5,
    entropy_coefficient=0.01,
    max_gradient_norm=0.5,
    learning_rate=3e-4,
    epochs=1,
    minibatch_size=512,
)
```

Compute GAE separately per episode with zero bootstrap for true terminal episodes. Concatenate and normalize advantages using population standard deviation with an epsilon of `1e-8`. Re-encode each stored complete history under current weights on every epoch; do not reuse an old GRU hidden state. Shuffle flattened indices with a CPU generator seeded from the `"minibatch"` namespace.

`PPOTrainer` owns the model and constructs exactly one Adam optimizer in its
constructor; that optimizer persists across both smoke updates. Use
`clip_grad_norm_`, record its returned pre-clip norm, compute the actual
post-clip norm from gradients, and apply hard finite checks before and after
`optimizer.step()`.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_advantages.py tests/neural/test_ppo.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural mypy --config-file mypy.neural.ini src/garboid_pocketrocks/neural tests/neural
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/ppo.py tests/neural/test_ppo.py
git commit -m "feat: add one epoch PPO updates"
```

### Task 9: Minimal inference checkpoint

**Files:**
- Create: `src/garboid_pocketrocks/neural/checkpoint.py`
- Create: `tests/neural/test_checkpoint.py`

**Interfaces:**
- Consumes: Task 4/5 JSON-safe config and `NeuralPolicy.state_dict`.
- Produces: `InferenceManifest`, `LoadedInferenceCheckpoint`, `parameter_digest`, `save_inference_checkpoint`, and `load_inference_checkpoint`.

- [ ] **Step 1: Write checkpoint RED tests**

Save to an empty temporary directory and assert exactly:

```text
checkpoint/
  manifest.json
  model.pt
```

The manifest must include schema version 1, encoder schema version 1, repository commit, Python/Torch/NumPy/SDK versions, model and encoder configs, action-space size/hash, live-A/3 support, root seed, completed episodes/updates, model SHA-256, and canonical parameter digest.

Load with `weights_only=True`, encode one fixed public fixture, and assert bit-equal bid logits, reveal logits, value, and deterministic greedy universal action before/after reload. Assert checksum mismatch, nonfinite weights, unsupported schema, changed action hash, or nonempty destination raises `CheckpointError`.

- [ ] **Step 2: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_checkpoint.py -q`

Expected: collection FAIL because `neural.checkpoint` does not exist.

- [ ] **Step 3: Implement the Stage 1 bundle**

Compute the parameter digest by iterating sorted state-dict names and hashing name, dtype, shape, and contiguous CPU bytes. Save `model.state_dict()` only. Serialize manifest JSON with sorted keys and a trailing newline. On load, verify manifest schema/support/hash/checksum, construct configs/model, call `torch.load(..., map_location=device, weights_only=True)`, reject missing/unexpected keys and nonfinite tensors, then set `eval()`.

Do not save optimizer state, RNG state, recovery state, metrics, league state, or resume metadata in Stage 1.

- [ ] **Step 4: Run GREEN and commit**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_checkpoint.py -q`

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/checkpoint.py tests/neural/test_checkpoint.py
git commit -m "feat: save minimal neural checkpoints"
```

### Task 10: Deterministic two-by-sixteen smoke and `garboid-train smoke`

**Files:**
- Create: `src/garboid_pocketrocks/neural/smoke.py`
- Create: `src/garboid_pocketrocks/neural/cli.py`
- Create: `tests/neural/test_smoke.py`
- Create: `tests/neural/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: all prior Stage 1 interfaces.
- Produces: `SmokeConfig`, `SmokeResult`, `run_smoke`, and the `garboid-train smoke` entry point.

- [ ] **Step 1: Write smoke-contract RED test**

Mark the test `@pytest.mark.neural_smoke`. Run `run_smoke` twice into distinct empty temporary directories with default config and assert:

- each run has two updates and 32 complete games;
- every update has exactly 16 games and one PPO epoch;
- all games terminate with zero illegal actions and zero faults;
- illegal probability is exactly zero;
- raw head logits, selected log probabilities, values, advantages, losses, and
  gradients are finite, while illegal universal logits are exactly `-inf`;
- each update reports `parameters_changed=True`;
- reward components, final money, outright/tied-first, and mean rank are
  separately present;
- reloaded final checkpoint reproduces stored fixture logits/value/greedy action;
- episode plans, deterministic metric fields, and final parameter digest are exactly equal between runs;
- elapsed time, output paths, and environment metadata are excluded from `SmokeResult.deterministic_payload()`.

- [ ] **Step 2: Write CLI RED test**

Invoke:

```python
exit_code = main(
    [
        "smoke",
        "--output-dir",
        str(tmp_path / "run"),
        "--seed",
        "42",
        "--updates",
        "1",
        "--games-per-update",
        "1",
    ]
)
assert exit_code == 0
assert (tmp_path / "run/checkpoint/manifest.json").is_file()
```

Assert `garboid-train smoke --help` documents the exact default `2` updates, `16` games/update, CPU, and seed 42. Reject non-CPU device, nonpositive counts, or nonempty output directories with exit code 2.

- [ ] **Step 3: Run RED**

Run: `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/neural/test_smoke.py tests/neural/test_cli.py -q`

Expected: collection FAIL because smoke/CLI modules do not exist.

- [ ] **Step 4: Implement smoke orchestration**

`run_smoke` must:

1. validate Stage 1 config and call deterministic setup before model construction;
2. derive the model-initialization seed from the `"model"` namespace,
   initialize the encoder/model, and construct one persistent `PPOTrainer`;
3. for update 0 and 1, collect the corresponding 16 frozen-policy episodes;
4. derive that update's shuffle seed from the `"minibatch"` namespace,
   compute learner trajectories, and run exactly one PPO epoch through the
   same trainer/Adam instance;
5. assert finite mechanics and parameter change after each update;
6. save a minimal inference checkpoint after update 2;
7. reload it and compare one canonical fixture;
8. write `smoke-result.json` containing resolved config and deterministic/non-deterministic sections;
9. return an immutable `SmokeResult`.

This is a mechanics assertion only; do not assert that the policy beats either heuristic or crosses a win-rate threshold.

- [ ] **Step 5: Add the CLI**

Add:

```toml
[project.scripts]
garboid-train = "garboid_pocketrocks.neural.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "neural_smoke: deterministic CPU PPO smoke contract",
]
```

`main(argv: Sequence[str] | None = None) -> int` must support only `smoke` in Stage 1. The parser may expose `--seed`, `--updates`, `--games-per-update`, and `--output-dir`; device is fixed to CPU. Do not add `train`, `resume`, `evaluate`, or `inspect` parsers yet.

- [ ] **Step 6: Finish neural CI**

Keep the unit step and add a separately named exact smoke step:

```yaml
      - name: Run neural unit tests
        run: uv run --extra neural pytest tests/neural -m "not neural_smoke" -q
      - name: Run deterministic PPO smoke
        run: uv run --extra neural pytest tests/neural/test_smoke.py::test_two_by_sixteen_smoke_is_deterministic -q
```

- [ ] **Step 7: Document Stage 1**

In `README.md`, add:

```bash
uv sync --locked --extra neural
uv run --extra neural garboid-train smoke --output-dir artifacts/neural-smoke
```

Explain that the default executes exactly two updates × 16 live-A/3 games against balanced/passive on CPU; it verifies mechanics and determinism, not playing strength. Document `manifest.json`/`model.pt`, Torch remaining optional, and Stage 1's lack of resume, league, varied rulesets, or a registered live wrapper.

- [ ] **Step 8: Run focused GREEN**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest tests/adapters tests/training tests/neural -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural garboid-train smoke --output-dir /private/tmp/garboid-stage1-smoke
```

Expected: PASS; CLI reports two updates, 32 games, and a reloadable checkpoint.

- [ ] **Step 9: Run core-without-extra and full neural gates**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv lock --check
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv sync --locked
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv sync --locked --extra neural
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural mypy --config-file mypy.neural.ini src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural pytest -q
git diff --check
```

Expected: all commands PASS. The core sync does not install Torch; neural tests are skipped in the core run and pass in the neural run.

- [ ] **Step 10: Commit**

```bash
git add src/garboid_pocketrocks/neural/smoke.py src/garboid_pocketrocks/neural/cli.py tests/neural/test_smoke.py tests/neural/test_cli.py pyproject.toml .github/workflows/ci.yml README.md
git commit -m "feat: add deterministic neural smoke"
```

## Stage 1 completion checklist

- [ ] The project-owned public history fails closed on malformed SDK frames.
- [ ] Equivalent SDK and simulator histories produce identical immutable events.
- [ ] Submitted bids remain absent until the whole auction resolves.
- [ ] Neural inputs are learner-relative and invariant under synthetic 3–5-seat rotations.
- [ ] The full history is replayed from zero on every model call and never truncated.
- [ ] Every sampled action is legal by construction and illegal probability is exactly zero.
- [ ] Gamma-1 GAE handles terminal and truncation bootstrapping as specified.
- [ ] PPO recomputes full-history outputs with stored masks and changes parameters once per update.
- [ ] Minimal inference checkpoint reload reproduces fixture logits, value, and greedy action.
- [ ] Default smoke completes two deterministic updates of 16 live-A/3 games each.
- [ ] Core install/type-check/tests remain Torch-free; the separate neural job proves Torch 2.13 on Python 3.14 CPU.
- [ ] No league, resume bundle, variable-ruleset curriculum, registered neural live bot, or strength claim was added.

## Resolved ambiguities and extension seams

1. **SDK reveal events lack a seat.** Stage 1 derives the revealed seat from the latest public auction winner/tiebreak exactly as the pinned SDK reconstruction does and records the derived seat in the canonical event.
2. **SDK events lack turn indexes.** Canonical history intentionally omits turn indexes; ordered cumulative events are sufficient and preserve SDK/simulator parity.
3. **Engine submitted decisions include bids and reveals.** The simulator
   adapter processes each completed transition batch as a unit. It publishes
   bid submissions only when that same batch contains `AUCTION_RESOLVED`,
   emits one canonical `PublicAuctionResolved`, and ignores the private reveal
   index in reveal batches.
4. **Stage 1 checkpoint scope.** The manifest and model checksum/digest are included for inspectability and reproducible inference. Optimizer/RNG resume state, atomic promotion, recovery, league lineage, and update-boundary resume belong to the later durable-run stage.
5. **Stage 1 player/ruleset support.** The encoder's rotation primitive is tested synthetically for 3–5 seats, but the Stage 1 checkpoint support descriptor and rollout reject anything except live-A/3. Curriculum support is a later extension through new checked configurations, not silent clipping.
6. **Smoke output semantics.** Stage 1 requires an empty output directory and writes one result plus one inference checkpoint. General artifact management and overwrite/resume behavior are not introduced.
7. **Determinism scope.** Exact comparison is same host architecture and pinned lock. Unsupported deterministic kernels fail instead of switching algorithms or tolerances.
8. **PyTorch wheel availability.** Task 1's lock, install, and neural CI job are the acceptance gate for the spec's stated Torch 2.13/CPython 3.14 support; implementation stops at that task if uv cannot resolve a compatible CPU wheel.
