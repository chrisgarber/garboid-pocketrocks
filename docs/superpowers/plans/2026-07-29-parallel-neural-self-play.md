# Parallel Neural Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel, accelerator-aware, resumable PPO self-play trainer for charts A-E and three-to-five-player PocketRocks games, including a 1,500-game smoke and measured ten-minute/eight-hour run profiles.

**Architecture:** Freeze one actor-critic snapshot per PPO rollout and use it in every seat of mirror games so all seat trajectories remain on-policy. A central policy/optimizer process batches inference from active games or spawned engine workers, records value/throughput metrics, and writes atomic portable checkpoints; deterministic episode/decision seeds make results independent of worker completion order.

**Tech Stack:** Python 3.14, PyTorch 2.13 CPU/CUDA/MPS, NumPy, PocketRocks deterministic engine, multiprocessing `spawn`, uv, pytest, mypy, Ruff.

## Global Constraints

- Implement `docs/superpowers/specs/2026-07-29-parallel-neural-self-play-design.md`.
- Support exactly charts `live-A` through `live-E` and player counts 3, 4, and 5.
- Preserve the existing universal 106-action encoding, maximum bid 100, maximum hand size five, and five relative seat slots.
- Policy and value inputs are limited to the acting seat's SDK context, public ruleset knowledge/history, private hand, and legal-action mask.
- Use `gamma=1.0`; reveal decisions must not introduce terminal discount.
- Freeze all collection-policy weights for a complete PPO rollout.
- Derive episode and decision seeds from stable indices; worker scheduling must not change planned games or sampled actions.
- Keep one central optimizer. Historical-policy transitions never enter current-policy PPO batches.
- Support `auto`, `cpu`, `cuda`, and `mps`; explicit unavailable devices fail rather than falling back.
- Same-device update-boundary resume must be exact. Cross-device checkpoints are portable but cross-device training need not be bit-identical.
- The user-facing smoke default is 100 games per chart/player-count cell: 1,500 games total.
- Use packed arrays/tensors for full smoke rollouts; do not retain diagnostic logits in production storage.
- Write checkpoints atomically and fail closed on schemas, hashes, bounds, checksums, non-finite tensors, and optimizer incompatibility.
- Use root seed 42 for committed profiles.
- Use `UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache` for every uv command.
- Follow strict red-green-refactor TDD for every behavior change.

## Locked file map

Configuration and planning:

- Modify `src/garboid_pocketrocks/neural/config.py` — A-E/3-5 encoder contract.
- Create `src/garboid_pocketrocks/neural/run_config.py` — JSON-safe trainer/profile configuration.
- Create `src/garboid_pocketrocks/neural/planning.py` — deterministic cell, seat-policy, and decision seeds.
- Create `tests/neural/test_planning.py`.

Collection:

- Create `src/garboid_pocketrocks/neural/self_play.py` — one all-seat game and trajectory boundaries.
- Create `src/garboid_pocketrocks/neural/collector.py` — serial active-game batching and policy grouping.
- Modify `src/garboid_pocketrocks/neural/rollout.py` — shared packed rollout records while retaining Stage 1 compatibility.
- Create `tests/neural/test_self_play.py`.
- Create `tests/neural/test_collector.py`.

Optimization and metrics:

- Modify `src/garboid_pocketrocks/neural/policy.py` — row-seeded order-independent sampling.
- Modify `src/garboid_pocketrocks/neural/ppo.py` — arbitrary device/epochs plus KL and clip metrics.
- Create `src/garboid_pocketrocks/neural/metrics.py` — value, gameplay, timing, and cell summaries.
- Create `tests/neural/test_metrics.py`.

Durability:

- Create `src/garboid_pocketrocks/neural/training_checkpoint.py`.
- Create `tests/neural/test_training_checkpoint.py`.

Parallelism and devices:

- Create `src/garboid_pocketrocks/neural/worker.py` — spawned engine worker protocol.
- Create `src/garboid_pocketrocks/neural/parallel.py` — central inference event loop.
- Create `src/garboid_pocketrocks/neural/devices.py`.
- Create `src/garboid_pocketrocks/neural/benchmark.py`.
- Create `tests/neural/test_parallel.py`.
- Create `tests/neural/test_devices.py`.

Evaluation and orchestration:

- Create `src/garboid_pocketrocks/neural/evaluation.py`.
- Create `src/garboid_pocketrocks/neural/league.py`.
- Create `src/garboid_pocketrocks/neural/trainer.py`.
- Modify `src/garboid_pocketrocks/neural/smoke.py`.
- Modify `src/garboid_pocketrocks/neural/cli.py`.
- Create `tests/neural/test_evaluation.py`.
- Create `tests/neural/test_trainer.py`.
- Modify `tests/neural/test_cli.py`.
- Modify `tests/neural/test_smoke.py`.

Committed profiles and docs:

- Create `configs/neural/smoke.json`.
- Create `configs/neural/initial-10m.json`.
- Create `configs/neural/long-8h.json`.
- Modify `src/garboid_pocketrocks/neural/README.md`.
- Modify `README.md`.

---

### Task 1: Expand the support envelope and deterministic episode plans

**Files:**
- Modify: `src/garboid_pocketrocks/neural/config.py`
- Create: `src/garboid_pocketrocks/neural/run_config.py`
- Create: `src/garboid_pocketrocks/neural/planning.py`
- Create: `tests/neural/test_planning.py`
- Modify: `tests/neural/test_encoding.py`

**Interfaces:**
- Consumes: `NeuralEncoderConfig`, `live_ruleset(chart)`, `derive_seed`.
- Produces:

```python
def training_encoder_config() -> NeuralEncoderConfig: ...


@dataclass(frozen=True, slots=True)
class SeatPolicy:
    identity: str
    trainable: bool


@dataclass(frozen=True, slots=True)
class SelfPlayEpisodePlan:
    update_index: int
    episode_index: int
    ruleset_name: str
    player_count: int
    engine_seed: int
    seat_sampling_seeds: tuple[int, ...]
    seat_policies: tuple[SeatPolicy, ...]


def plan_mirror_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    policy_identity: str,
) -> tuple[SelfPlayEpisodePlan, ...]: ...


def decision_seed(plan: SelfPlayEpisodePlan, seat: int, decision_index: int) -> int: ...
```

- [ ] **Step 1: Write failing support and planning tests**

```python
def test_training_encoder_covers_every_live_chart_and_player_count() -> None:
    config = training_encoder_config()
    assert config.supported_ruleset_names == ("live-A", "live-B", "live-C", "live-D", "live-E")
    assert config.supported_player_counts == (3, 4, 5)
    for chart in "ABCDE":
        ruleset = live_ruleset(chart)
        for player_count in (3, 4, 5):
            required = (
                1
                + (2 * sum(ruleset.action_counts))
                + (player_count * ruleset.setup_for(player_count).private_cards_per_player)
            )
            assert required <= config.max_history_events


def test_mirror_plan_has_one_hundred_games_in_every_cell() -> None:
    plans = plan_mirror_episodes(
        root_seed=42,
        update_index=0,
        games_per_cell=100,
        policy_identity="current",
    )
    assert len(plans) == 1_500
    counts = Counter((plan.ruleset_name, plan.player_count) for plan in plans)
    assert set(counts.values()) == {100}
    assert set(counts) == {(f"live-{chart}", players) for chart in "ABCDE" for players in (3, 4, 5)}
    assert all(
        len(plan.seat_sampling_seeds) == plan.player_count
        and plan.seat_policies == (SeatPolicy("current", True),) * plan.player_count
        for plan in plans
    )


def test_plans_and_decision_seeds_are_named_stable() -> None:
    first = plan_mirror_episodes(
        root_seed=42, update_index=7, games_per_cell=2, policy_identity="candidate"
    )
    second = plan_mirror_episodes(
        root_seed=42, update_index=7, games_per_cell=2, policy_identity="candidate"
    )
    assert first == second
    assert len(
        {decision_seed(plan, seat, 0) for plan in first for seat in range(plan.player_count)}
    ) == sum(plan.player_count for plan in first)
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_planning.py tests/neural/test_encoding.py -q
```

Expected: collection fails because `planning`, `run_config`, and
`training_encoder_config` do not exist.

- [ ] **Step 3: Implement the finite support envelope**

Add to `config.py`:

```python
def training_encoder_config() -> NeuralEncoderConfig:
    rulesets = tuple(live_ruleset(chart) for chart in "ABCDE")
    player_counts = (3, 4, 5)
    max_history = max(
        1
        + (2 * sum(ruleset.action_counts))
        + (player_count * ruleset.setup_for(player_count).private_cards_per_player)
        for ruleset in rulesets
        for player_count in player_counts
    )
    return NeuralEncoderConfig(
        schema_version=1,
        supported_ruleset_names=tuple(ruleset.name for ruleset in rulesets),
        supported_player_counts=player_counts,
        max_bid=100,
        max_hand_size=5,
        max_history_events=max_history,
        max_cash=100,
        max_abs_chart=20,
        max_resource_cards=max(sum(ruleset.resource_counts) for ruleset in rulesets),
        max_action_cards=max(sum(ruleset.action_counts) for ruleset in rulesets),
    )
```

Import `live_ruleset` alongside `LIVE_RULESET`. Keep `stage1_encoder_config`
unchanged so Stage 1 checkpoints/tests remain loadable.

- [ ] **Step 4: Implement immutable JSON-safe run configuration**

Define in `run_config.py`:

```python
@dataclass(frozen=True, slots=True)
class ParallelConfig:
    workers: int | Literal["auto"] = "auto"
    active_games_per_worker: int = 8
    max_inference_batch: int = 256
    max_queue_delay_ms: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    root_seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    games_per_cell: int | None = 100
    max_updates: int | None = None
    max_wall_seconds: float | None = None
    target_decisions_per_update: int | None = None
    checkpoint_interval_seconds: float | None = None
    evaluation_interval_seconds: float | None = None
    evaluation_games_per_seat_cell: int = 2
    evaluate_at_start: bool = False
    evaluate_at_end: bool = False
    league_fraction: float = 0.0
    keep_periodic_checkpoints: int = 4
    parallel: ParallelConfig = ParallelConfig()
    ppo: PPOConfig = PPOConfig()
    reward: RewardConfig = RewardConfig()

    @classmethod
    def from_json(cls, path: Path) -> TrainingRunConfig: ...
    def to_json_dict(self) -> dict[str, object]: ...
```

Validate integers against booleans, require positive sizes/times, require
`0.0 <= league_fraction < 1.0`, require a positive checkpoint-retention count,
and require exactly one of `games_per_cell` or
`target_decisions_per_update`. Reject unknown JSON keys recursively. Serialize
tuples as JSON arrays and dataclasses as sorted objects.

- [ ] **Step 5: Implement balanced plans and per-decision seeds**

In `planning.py`, define the dataclasses exactly as in Interfaces. Build plans
in this stable order:

```python
for repetition in range(games_per_cell):
    for chart in "ABCDE":
        for player_count in (3, 4, 5):
            ...
```

Derive engine and seat seeds from `hashlib.blake2b` using canonical names:

```text
root:engine:update:episode
root:policy:update:episode:seat
root:decision:update:episode:seat:decision
```

Return unsigned 63-bit values. Validate that seat policy/seeds exactly match
`player_count`.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_planning.py tests/neural/test_encoding.py -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  mypy --config-file mypy.neural.ini src tests
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/config.py \
  src/garboid_pocketrocks/neural/run_config.py \
  src/garboid_pocketrocks/neural/planning.py \
  tests/neural/test_planning.py tests/neural/test_encoding.py
git commit -m "feat: plan full neural self-play curriculum"
```

### Task 2: Model one all-seat self-play game without information leaks

**Files:**
- Create: `src/garboid_pocketrocks/neural/self_play.py`
- Modify: `src/garboid_pocketrocks/neural/rollout.py`
- Create: `tests/neural/test_self_play.py`

**Interfaces:**
- Consumes: `SelfPlayEpisodePlan`, `GameEngine`, `SimulatorPublicHistoryAdapter`,
  `RewardTracker`, `NeuralObservationEncoder`, `ActionCodec`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class PendingPolicyRequest:
    episode_index: int
    seat: int
    decision_index: int
    policy_identity: str
    trainable: bool
    sampling_seed: int
    observation: NeuralObservation


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    episode_index: int
    seat: int
    decision_index: int
    action: int
    old_log_probability: float
    old_value: float


class SelfPlayGame:
    @classmethod
    def start(
        cls,
        plan: SelfPlayEpisodePlan,
        *,
        encoder_config: NeuralEncoderConfig,
        reward_config: RewardConfig,
    ) -> SelfPlayGame: ...
    def pending_requests(self) -> tuple[PendingPolicyRequest, ...]: ...
    def apply(self, responses: Sequence[PolicyResponse]) -> None: ...
    @property
    def terminated(self) -> bool: ...
    def episode(self) -> MultiSeatEpisode: ...
```

- [ ] **Step 1: Write failing simultaneous-action and trajectory tests**

```python
def test_all_pending_bids_are_requested_before_resolution() -> None:
    plan = plan_mirror_episodes(
        root_seed=42, update_index=0, games_per_cell=1, policy_identity="current"
    )[0]
    game = SelfPlayGame.start(
        plan,
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
    )
    requests = game.pending_requests()
    assert tuple(request.seat for request in requests) == (0, 1, 2)
    assert len({request.observation.global_ids.tobytes() for request in requests}) >= 1
    with pytest.raises(SelfPlayError, match="every pending seat"):
        game.apply((_pass_response(requests[0]),))


def test_every_seat_yields_a_terminated_trainable_trajectory() -> None:
    plan = _plan_for("live-E", 5)
    game = SelfPlayGame.start(
        plan,
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
    )
    while not game.terminated:
        game.apply(tuple(_pass_response(request) for request in game.pending_requests()))
    episode = game.episode()
    assert len(episode.trajectories) == 5
    assert tuple(trajectory.seat for trajectory in episode.trajectories) == (0, 1, 2, 3, 4)
    assert all(
        trajectory.trainable and trajectory.transitions for trajectory in episode.trajectories
    )
    assert all(trajectory.transitions[-1].terminated for trajectory in episode.trajectories)
    assert all(
        not transition.truncated
        for trajectory in episode.trajectories
        for transition in trajectory.transitions
    )


def test_unresolved_bids_never_enter_another_seats_observation() -> None:
    game = _started_game("live-A", 3)
    before = game.pending_requests()
    assert all(int(request.observation.history_valid.sum()) == 2 for request in before)
```

The initial two valid history rows are game setup and turn opened. Production
requests retain only `NeuralObservation`.

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_self_play.py -q
```

Expected: collection fails because `self_play` does not exist.

- [ ] **Step 3: Implement per-seat transition boundaries**

Use one `RewardTracker`, which already returns every seat's reward after an
engine step. Maintain:

```python
_open_transition_by_seat: dict[int, OpenTransition]
_pending_reward_by_seat: dict[int, RewardBreakdown]
_decision_count_by_seat: dict[int, int]
_completed_by_seat: dict[int, list[RolloutTransition]]
```

Before creating a seat's next request, finalize its previous open transition
with all rewards accumulated since that action. After `GameEngine.step`, add
the returned reward breakdown to every seat's pending total. At termination,
finalize every open transition with `terminated=True`.

Decode each response through `ActionCodec`; require exact
`(episode_index, seat, decision_index)` identity and the complete pending-seat
set. Call `context.validate(decision)` before `GameEngine.step`.

- [ ] **Step 4: Generalize rollout provenance without breaking Stage 1**

Add:

```python
@dataclass(frozen=True, slots=True)
class SeatTrajectory:
    seat: int
    policy_identity: str
    trainable: bool
    transitions: tuple[RolloutTransition, ...]


@dataclass(frozen=True, slots=True)
class MultiSeatEpisode:
    plan: SelfPlayEpisodePlan
    trajectories: tuple[SeatTrajectory, ...]
    result: GameResult
```

Add `RolloutBatch.from_multi_seat(episodes)` that flattens only
`trajectory.trainable` trajectories while preserving episode/seat boundaries
for GAE. Keep existing `RolloutEpisode` and `collect_rollout` behavior for
Stage 1 tests.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_self_play.py tests/neural/test_rollout.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/self_play.py \
  src/garboid_pocketrocks/neural/rollout.py \
  tests/neural/test_self_play.py
git commit -m "feat: collect all-seat self-play trajectories"
```

### Task 3: Batch active games and make sampling order-independent

**Files:**
- Modify: `src/garboid_pocketrocks/neural/policy.py`
- Create: `src/garboid_pocketrocks/neural/collector.py`
- Create: `tests/neural/test_collector.py`
- Modify: `tests/neural/test_policy.py`

**Interfaces:**
- Produces:

```python
def evaluate_row_seeded_policy(
    output: PolicyValueOutput,
    batch: NeuralBatch,
    *,
    row_seeds: Sequence[int],
) -> PolicySelection: ...


@dataclass(frozen=True, slots=True)
class CollectorMetrics:
    games: int
    decisions: int
    elapsed_seconds: float
    inference_seconds: float
    inference_batches: int
    inference_batch_sizes: tuple[int, ...]
    cell_games: tuple[tuple[str, int, int], ...]


def collect_self_play(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    active_games: int,
    max_inference_batch: int,
) -> tuple[RolloutBatch, CollectorMetrics]: ...
```

- [ ] **Step 1: Write RED tests for batching and order independence**

```python
def test_row_seeded_sampling_is_independent_of_batch_order() -> None:
    model, observations = _policy_and_observations(count=8)
    forward = _select(model, observations, seeds=range(100, 108))
    order = (7, 2, 5, 0, 6, 1, 4, 3)
    shuffled = _select(
        model,
        tuple(observations[index] for index in order),
        seeds=tuple(tuple(range(100, 108))[index] for index in order),
    )
    restored = {order[index]: int(action) for index, action in enumerate(shuffled.actions)}
    assert tuple(restored[index] for index in range(8)) == tuple(
        int(action) for action in forward.actions
    )


def test_active_game_collector_batches_and_covers_all_cells() -> None:
    plans = plan_mirror_episodes(
        root_seed=42, update_index=0, games_per_cell=1, policy_identity="current"
    )
    rollout, metrics = collect_self_play(
        {"current": _model()},
        plans,
        encoder_config=training_encoder_config(),
        reward_config=RewardConfig(),
        device=torch.device("cpu"),
        active_games=8,
        max_inference_batch=64,
    )
    assert metrics.games == 15
    assert metrics.decisions == len(rollout.transitions)
    assert max(metrics.inference_batch_sizes) > 1
    assert {(name, count) for name, count, games in metrics.cell_games if games == 1} == {
        (f"live-{chart}", players) for chart in "ABCDE" for players in (3, 4, 5)
    }
    assert all(
        transition.observation.action_mask[transition.action] for transition in rollout.transitions
    )
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_policy.py tests/neural/test_collector.py -q
```

Expected: FAIL because the row-seeded policy and collector are absent.

- [ ] **Step 3: Implement row-seeded sampling**

Reuse legal masking from `evaluate_masked_policy`, transfer the complete
probability batch to CPU once, and sample each row using an independent CPU
generator:

```python
probabilities_cpu = probabilities.detach().cpu()
actions_cpu = torch.stack(
    [
        torch.multinomial(
            probabilities_cpu[index],
            1,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        for index, seed in enumerate(row_seeds)
    ]
).squeeze(1)
actions = actions_cpu.to(probabilities.device)
```

Compute selected log probability and entropy on the original device. Validate
one unsigned 63-bit seed per row. This is intentionally per-row so batch
composition and worker completion order cannot change samples.

- [ ] **Step 4: Implement the serial active-game collector**

Maintain an ordered mapping from episode index to `SelfPlayGame`. Fill it up to
`active_games`. Gather requests in `(episode_index, seat, decision_index)`
order, group by `policy_identity`, split groups at `max_inference_batch`, and
call `batch_observations` once per split.

After selection, restore responses to each game and apply complete pending
batches. Refill slots until all plans terminate. Freeze/eval each policy for
the whole call and restore its prior `training` flag in `finally`.

Use `time.perf_counter()` around collection and policy forward calls. Reject
empty plans, unknown policy identities, duplicate episode indices, unsupported
devices, illegal actions, or non-finite outputs.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_policy.py tests/neural/test_collector.py \
  tests/neural/test_self_play.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/policy.py \
  src/garboid_pocketrocks/neural/collector.py \
  tests/neural/test_policy.py tests/neural/test_collector.py
git commit -m "feat: batch deterministic self-play inference"
```

### Task 4: Add packed rollout storage, accelerator PPO, and value diagnostics

**Files:**
- Modify: `src/garboid_pocketrocks/neural/rollout.py`
- Modify: `src/garboid_pocketrocks/neural/ppo.py`
- Create: `src/garboid_pocketrocks/neural/metrics.py`
- Modify: `tests/neural/test_ppo.py`
- Create: `tests/neural/test_metrics.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class ValueMetrics:
    count: int
    mean_prediction: float
    mean_target: float
    mae: float
    rmse: float
    bias: float
    explained_variance: float | None
    correlation: float | None
    calibration: tuple[CalibrationBucket, ...]


def value_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    buckets: int = 10,
) -> ValueMetrics: ...


@dataclass(frozen=True, slots=True)
class ValueMetricSlice:
    dimension: str
    key: str
    metrics: ValueMetrics


def stratified_value_metrics(
    predictions: Tensor,
    targets: Tensor,
    *,
    ruleset_names: Sequence[str],
    player_counts: Sequence[int],
    phases: Sequence[str],
) -> tuple[ValueMetricSlice, ...]: ...


@dataclass(frozen=True, slots=True)
class GameplayMetrics:
    games: int
    decisions: int
    first_place_share: float
    mean_rank: float
    mean_final_money: float
    pass_rate: float
    mean_positive_bid: float | None
    illegal_actions: int
    faults: int


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    ...
    approximate_kl: float
    clip_fraction: float
    value: ValueMetrics
```

- [ ] **Step 1: Write hand-calculated RED metrics tests**

```python
def test_value_metrics_match_hand_calculation() -> None:
    predictions = torch.tensor((0.0, 1.0, 2.0, 3.0))
    targets = torch.tensor((1.0, 1.0, 3.0, 3.0))
    result = value_metrics(predictions, targets, buckets=2)
    assert result.count == 4
    assert result.mae == pytest.approx(0.5)
    assert result.rmse == pytest.approx(math.sqrt(0.5))
    assert result.bias == pytest.approx(-0.5)
    assert result.explained_variance == pytest.approx(0.75)
    assert sum(bucket.count for bucket in result.calibration) == 4


def test_constant_targets_report_undefined_statistics_without_nan() -> None:
    result = value_metrics(torch.tensor((0.0, 1.0)), torch.ones(2), buckets=2)
    assert result.explained_variance is None
    assert result.correlation is None
    assert math.isfinite(result.rmse)


def test_value_metrics_are_sliced_by_chart_player_count_and_phase() -> None:
    slices = stratified_value_metrics(
        torch.tensor((0.0, 0.5, 1.0, 1.5)),
        torch.tensor((0.2, 0.7, 0.8, 1.2)),
        ruleset_names=("live-A", "live-A", "live-E", "live-E"),
        player_counts=(3, 5, 3, 5),
        phases=("early", "middle", "late", "late"),
    )
    assert {item.dimension for item in slices} == {"all", "ruleset", "player_count", "phase"}
    assert any(item.dimension == "ruleset" and item.key == "live-E" for item in slices)


def test_ppo_reports_kl_clip_fraction_and_value_quality() -> None:
    metrics = PPOTrainer(_model(), PPOConfig(epochs=2)).update(
        _multi_seat_rollout(), update_seed=17
    )
    assert metrics.epochs == 2
    assert math.isfinite(metrics.approximate_kl)
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert metrics.value.count == metrics.transition_count
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_metrics.py tests/neural/test_ppo.py -q
```

Expected: FAIL because `metrics` and the new fields do not exist.

- [ ] **Step 3: Implement packed observation storage**

Add `PackedRollout` with one NumPy array per `NeuralObservation` field and
one-dimensional arrays for action, old log probability/value, reward,
terminated, truncated, episode index, seat, chart index, player count, and
phase bucket. `PackedRollout.from_batch` stacks once; `observation(index)`
returns zero-copy views where NumPy permits.

Keep `RolloutBatch` as the collection boundary. Convert to `PackedRollout`
immediately before PPO and allow releasing the Python transition graph.

- [ ] **Step 4: Generalize PPO**

Remove the Stage 1 restrictions `device.type == "cpu"` and `epochs == 1`.
Require `epochs > 0`. Create the minibatch generator on CPU, transfer gathered
observation batches and target/index tensors to the model device, and iterate
one deterministic `randperm` per epoch using:

```python
epoch_seed = derive_local_seed(update_seed, "epoch", epoch_index)
```

Accumulate:

```python
approximate_kl_sum += float((old_log_probability - new_log_probability).mean())
clip_count += int((torch.abs(loss.ratio - 1.0) > config.clip_ratio).sum())
```

Compute global and stratified `ValueMetrics` from the pre-update old values and
return targets so the reported estimator quality describes the collected
policy consistently. Derive phase buckets from public turn index thirds:
`early`, `middle`, and `late`.

- [ ] **Step 5: Implement value metrics**

Use population variance. Explained variance is:

```python
1.0 - variance(targets - predictions) / variance(targets)
```

Return `None` when target variance is zero. Return correlation only when both
variances are positive. Sort by prediction with a stable index tie-break and
split into at most `buckets` nonempty, size-balanced calibration buckets.
Reject empty, non-1D, shape-mismatched, or non-finite tensors.

Aggregate `GameplayMetrics` from terminal results and stored actions. Treat
action zero as pass, actions `1..100` as positive bids, and reveal actions as
neither. Report all metrics globally and per chart/player-count cell.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_metrics.py tests/neural/test_ppo.py \
  tests/neural/test_advantages.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/rollout.py \
  src/garboid_pocketrocks/neural/ppo.py \
  src/garboid_pocketrocks/neural/metrics.py \
  tests/neural/test_ppo.py tests/neural/test_metrics.py
git commit -m "feat: report PPO value quality metrics"
```

### Task 5: Save atomic resumable training checkpoints

**Files:**
- Create: `src/garboid_pocketrocks/neural/training_checkpoint.py`
- Modify: `src/garboid_pocketrocks/neural/checkpoint.py`
- Create: `tests/neural/test_training_checkpoint.py`
- Modify: `tests/neural/test_checkpoint.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TrainingProgress:
    next_update_index: int
    completed_episodes: int
    completed_decisions: int
    cell_games: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class TrainingCheckpointManifest:
    schema_version: int
    repository_commit: str
    encoder_config: NeuralEncoderConfig
    model_config: NeuralModelConfig
    run_config: TrainingRunConfig
    progress: TrainingProgress
    lineage: tuple[str, ...]
    champion_identity: str | None
    league_identities: tuple[str, ...]
    parameter_digest: str
    file_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class LoadedTrainingCheckpoint:
    model: NeuralPolicy
    optimizer: torch.optim.Adam
    manifest: TrainingCheckpointManifest
    generator_states: dict[str, Tensor]
    metrics: dict[str, object]


def save_training_checkpoint(
    path: Path,
    *,
    model: NeuralPolicy,
    optimizer: torch.optim.Optimizer,
    manifest: TrainingCheckpointManifest,
    generator_states: Mapping[str, Tensor],
    metrics: Mapping[str, object],
) -> Path: ...


def load_training_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> LoadedTrainingCheckpoint: ...


def export_inference_checkpoint(
    training_checkpoint: Path,
    output_path: Path,
    *,
    device: torch.device,
) -> Path: ...
```

- [ ] **Step 1: Write RED atomic, corruption, and resume tests**

```python
def test_training_checkpoint_contains_exact_validated_files(tmp_path: Path) -> None:
    saved = _save_checkpoint(tmp_path / "checkpoint")
    assert {item.name for item in saved.iterdir()} == {
        "manifest.json",
        "model.pt",
        "optimizer.pt",
        "rng.pt",
        "metrics.json",
    }
    loaded = load_training_checkpoint(saved, device=torch.device("cpu"))
    assert loaded.manifest.progress.next_update_index == 3
    assert parameter_digest(loaded.model.state_dict()) == loaded.manifest.parameter_digest
    assert loaded.optimizer.state_dict()


def test_failed_save_does_not_replace_last_valid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _save_checkpoint(tmp_path / "checkpoint")
    digest = _manifest(checkpoint)["parameter_digest"]
    monkeypatch.setattr(torch, "save", Mock(side_effect=OSError("disk full")))
    with pytest.raises(TrainingCheckpointError, match="disk full"):
        _save_checkpoint(tmp_path / "checkpoint", replace=True)
    assert _manifest(checkpoint)["parameter_digest"] == digest


def test_update_boundary_resume_matches_uninterrupted_next_update(tmp_path: Path) -> None:
    uninterrupted = _run_two_updates()
    first = _run_one_update_and_save(tmp_path / "checkpoint")
    resumed = _load_and_run_next(first)
    assert resumed.plans == uninterrupted.second.plans
    assert resumed.metrics == uninterrupted.second.metrics
    assert resumed.parameter_digest == uninterrupted.second.parameter_digest
    assert resumed.optimizer_digest == uninterrupted.second.optimizer_digest


def test_training_checkpoint_exports_portable_inference_bundle(tmp_path: Path) -> None:
    training = _save_checkpoint(tmp_path / "training")
    inference = export_inference_checkpoint(
        training, tmp_path / "inference", device=torch.device("cpu")
    )
    loaded = load_inference_checkpoint(inference, device=torch.device("cpu"))
    assert loaded.manifest.supported_ruleset_names == (
        "live-A",
        "live-B",
        "live-C",
        "live-D",
        "live-E",
    )
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_training_checkpoint.py -q
```

Expected: collection fails because `training_checkpoint` does not exist.

- [ ] **Step 3: Implement versioned atomic save**

Write to `path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")`. Save model
and optimizer state dictionaries, plus a `dict[str, Tensor]` of generator
states. Serialize manifest/metrics with sorted JSON and `allow_nan=False`.
Compute SHA-256 for the four payload files, then write the final manifest.

Open each file and call `os.fsync`, fsync the temporary directory, validate by
loading it, rename the prior destination to a temporary backup, atomically
rename the new directory into place, fsync the parent, and remove the backup.
On failure, restore the backup and remove only the uniquely named temporary
directory.

- [ ] **Step 4: Implement fail-closed load**

Require the exact five filenames and schema version 1. Reconstruct
encoder/model/run/progress dataclasses with exact-key validation. Verify every
checksum before `torch.load(..., weights_only=True)`. Validate model names,
shapes, dtypes, finiteness, parameter digest, optimizer parameter-group count,
and tensor finiteness before returning.

Load model/optimizer through `map_location=device`. Generator tensors remain
CPU byte tensors until restored to their named generator.

Generalize `checkpoint.py` manifest validation to accept support sets from its
stored encoder configuration rather than the Stage 1 hard-coded tuple. Preserve
the existing Stage 1 fixture. `export_inference_checkpoint` loads the trusted
training bundle and calls `save_inference_checkpoint` with completion counters
and the full A-E/3-5 encoder support.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_training_checkpoint.py tests/neural/test_checkpoint.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/training_checkpoint.py \
  src/garboid_pocketrocks/neural/checkpoint.py \
  tests/neural/test_training_checkpoint.py tests/neural/test_checkpoint.py
git commit -m "feat: save resumable neural training checkpoints"
```

### Task 6: Add spawned game workers and central batched inference

**Files:**
- Create: `src/garboid_pocketrocks/neural/worker.py`
- Create: `src/garboid_pocketrocks/neural/parallel.py`
- Create: `tests/neural/test_parallel.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class WorkerInferenceBatch:
    worker_id: int
    sequence: int
    requests: tuple[PendingPolicyRequest, ...]


@dataclass(frozen=True, slots=True)
class WorkerResponseBatch:
    worker_id: int
    sequence: int
    responses: tuple[PolicyResponse, ...]


@dataclass(frozen=True, slots=True)
class WorkerEpisodes:
    worker_id: int
    episodes: tuple[MultiSeatEpisode, ...]


def collect_self_play_parallel(
    policies: Mapping[str, NeuralPolicy],
    plans: Sequence[SelfPlayEpisodePlan],
    *,
    encoder_config: NeuralEncoderConfig,
    reward_config: RewardConfig,
    device: torch.device,
    workers: int,
    active_games_per_worker: int,
    max_inference_batch: int,
    max_queue_delay_ms: float,
) -> tuple[RolloutBatch, CollectorMetrics]: ...
```

Extend `CollectorMetrics` with `queue_wait_seconds`, `ipc_seconds`,
`worker_busy_seconds`, `inference_batch_p50`, and `inference_batch_p95`.

- [ ] **Step 1: Write RED serial/parallel equivalence tests**

```python
@pytest.mark.parametrize("workers", (1, 2, 4))
def test_worker_count_does_not_change_seeded_rollout(workers: int) -> None:
    plans = plan_mirror_episodes(
        root_seed=42, update_index=0, games_per_cell=1, policy_identity="current"
    )
    serial = _collect(plans, workers=1)
    parallel = _collect(plans, workers=workers)
    assert _action_records(parallel) == _action_records(serial)
    assert _reward_records(parallel) == _reward_records(serial)
    assert _ordered_results(parallel) == _ordered_results(serial)


def test_worker_failure_terminates_pool_without_partial_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "_run_plan_shard", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(ParallelCollectionError, match="worker"):
        _collect(_plans(), workers=2)
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_parallel.py -q
```

Expected: collection fails because the worker protocol is absent.

- [ ] **Step 3: Implement the spawned worker**

Use `multiprocessing.get_context("spawn")` and one duplex `Pipe` per worker.
Shard plans by `episode_index % workers`, preserving order inside each shard.
Each worker runs the same active-game loop as the serial collector up to the
inference boundary. It sends `WorkerInferenceBatch`, waits for an exact
matching `WorkerResponseBatch`, applies responses, and finally sends
`WorkerEpisodes`.

Messages include monotonically increasing `sequence`. Reject mismatched worker
IDs, sequences, request identities, duplicate final episodes, or any message
after completion. Wrap worker exceptions into a small serializable
`WorkerFailure(worker_id, error_type, message)`.

- [ ] **Step 4: Implement the central inference event loop**

Use `multiprocessing.connection.wait` with the configured queue delay. Drain
all ready connections, sort requests by
`(episode_index, seat, decision_index)`, group by policy, and run the same
row-seeded selection as the serial collector. Return response subsets to the
originating worker/sequence.

On any failure, close pipes, terminate live children, join every process, and
raise `ParallelCollectionError` without returning a rollout. On success,
restore episodes to global episode order before constructing `RolloutBatch`.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_parallel.py tests/neural/test_collector.py -q
```

Expected: PASS on worker counts 1, 2, and 4.

Commit:

```bash
git add src/garboid_pocketrocks/neural/worker.py \
  src/garboid_pocketrocks/neural/parallel.py \
  tests/neural/test_parallel.py
git commit -m "feat: parallelize self-play game collection"
```

### Task 7: Resolve CPU/CUDA/MPS devices and calibrate throughput

**Files:**
- Create: `src/garboid_pocketrocks/neural/devices.py`
- Create: `src/garboid_pocketrocks/neural/benchmark.py`
- Create: `tests/neural/test_devices.py`

**Interfaces:**
- Produces:

```python
def available_devices() -> tuple[str, ...]: ...
def resolve_device(requested: str) -> torch.device: ...


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    device: str
    workers: int
    active_games_per_worker: int
    max_inference_batch: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    candidate: BenchmarkCandidate
    games: int
    decisions: int
    elapsed_seconds: float
    ppo_seconds: float
    total_seconds: float
    games_per_second: float
    decisions_per_second: float
    inference_batch_p50: float
    inference_batch_p95: float
    peak_rss_bytes: int | None


def calibrate(
    config: TrainingRunConfig,
    *,
    plans: Sequence[SelfPlayEpisodePlan],
) -> tuple[BenchmarkCandidate, tuple[BenchmarkResult, ...]]: ...
```

- [ ] **Step 1: Write RED device and selection tests**

```python
def test_explicit_unavailable_device_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceError, match="cuda"):
        resolve_device("cuda")


def test_auto_selects_fastest_complete_candidate() -> None:
    results = (
        _result("cpu", workers=1, decisions_per_second=100.0),
        _result("cpu", workers=4, decisions_per_second=250.0),
        _result("mps", workers=2, decisions_per_second=220.0),
    )
    assert choose_candidate(results).workers == 4


def test_calibration_covers_every_cell_with_small_balanced_subset() -> None:
    plans = calibration_plans(root_seed=42, games_per_cell=2)
    assert len(plans) == 30
    assert len(Counter((p.ruleset_name, p.player_count) for p in plans)) == 15
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_devices.py -q
```

Expected: collection fails because `devices` and `benchmark` do not exist.

- [ ] **Step 3: Implement device resolution**

`available_devices()` always returns `"cpu"`, adds `"cuda"` when
`torch.cuda.is_available()`, and adds `"mps"` when
`torch.backends.mps.is_available()`. Explicit requests must appear in that
tuple. `"auto"` remains unresolved until calibration; do not silently map it
to CPU.

Synchronize before stopping device timers:

```python
if device.type == "cuda":
    torch.cuda.synchronize(device)
elif device.type == "mps":
    torch.mps.synchronize()
```

- [ ] **Step 4: Implement bounded calibration**

Use two games per cell per candidate. Candidate workers are the unique valid
values from `(1, 2, 4, min(os.cpu_count() or 1, 8))`; active games per worker
are `(2, 4)` and max inference batches `(64, 256)`. Compare CPU and available
accelerators only when device is `auto`; explicit device compares parallel
settings on that device.

For each candidate, collect the balanced subset and execute one PPO epoch over
the packed transitions using a fresh identically initialized model. Choose
highest decisions per `total_seconds`, then lower total time, then fewer
workers, then lexicographic device as deterministic tie-breaks. Record failures
separately and require at least one successful candidate. RSS may be `None` on
unsupported platforms.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_devices.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/devices.py \
  src/garboid_pocketrocks/neural/benchmark.py \
  tests/neural/test_devices.py
git commit -m "feat: calibrate neural training devices"
```

### Task 8: Add paired evaluation and an immutable checkpoint league

**Files:**
- Create: `src/garboid_pocketrocks/neural/evaluation.py`
- Create: `src/garboid_pocketrocks/neural/league.py`
- Create: `tests/neural/test_evaluation.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    root_seed: int
    games_per_seat_cell: int
    bootstrap_samples: int = 2_000
    confidence: float = 0.95

@dataclass(frozen=True, slots=True)
class EvaluationReport:
    candidate_identity: str
    incumbent_identity: str
    games: int
    utility_delta: float
    confidence_low: float
    confidence_high: float
    cell_metrics: tuple[EvaluationCellMetrics, ...]
    promoted: bool

@dataclass(frozen=True, slots=True)
class HeuristicEvaluationReport:
    games: int
    cell_metrics: tuple[EvaluationCellMetrics, ...]
    illegal_actions: int
    faults: int

def plan_paired_evaluation(...) -> tuple[SelfPlayEpisodePlan, ...]: ...
def evaluate_candidate(...) -> EvaluationReport: ...
def evaluate_against_heuristics(
    candidate: NeuralPolicy,
    *,
    config: EvaluationConfig,
    device: torch.device,
) -> HeuristicEvaluationReport: ...
def plan_league_episodes(
    *,
    root_seed: int,
    update_index: int,
    games_per_cell: int,
    current_identity: str,
    historical_identities: Sequence[str],
    league_fraction: float,
) -> tuple[SelfPlayEpisodePlan, ...]: ...

@dataclass(frozen=True, slots=True)
class League:
    champion: str
    promoted: tuple[str, ...]
    def promote(self, report: EvaluationReport) -> League: ...
```

- [ ] **Step 1: Write RED seat-rotation and promotion tests**

```python
def test_paired_evaluation_rotates_candidate_through_every_seat_and_cell() -> None:
    plans = plan_paired_evaluation(
        root_seed=99,
        candidate_identity="candidate",
        incumbent_identity="incumbent",
        games_per_seat_cell=2,
    )
    counts = Counter(
        (
            plan.ruleset_name,
            plan.player_count,
            plan.seat_policies.index(SeatPolicy("candidate", False)),
        )
        for plan in plans
    )
    assert set(counts.values()) == {2}
    assert len(counts) == sum(players for players in (3, 4, 5)) * 5


def test_promotion_requires_positive_lower_confidence_bound() -> None:
    assert not promotion_decision(_report(low=-0.01, high=0.20))
    assert promotion_decision(_report(low=0.01, high=0.20))


def test_shaped_training_return_cannot_change_promotion() -> None:
    first = paired_utility(candidate_money=1.2, incumbent_money=1.0, first_share=0.1)
    second = paired_utility(candidate_money=1.2, incumbent_money=1.0, first_share=0.1)
    assert first == second


def test_league_games_train_only_current_policy_seats() -> None:
    plans = plan_league_episodes(
        root_seed=42,
        update_index=10,
        games_per_cell=10,
        current_identity="current",
        historical_identities=("champion", "older"),
        league_fraction=0.2,
    )
    assert any(any(not seat.trainable for seat in plan.seat_policies) for plan in plans)
    assert all(
        seat.trainable == (seat.identity == "current")
        for plan in plans
        for seat in plan.seat_policies
    )


def test_heuristic_evaluation_covers_all_cells_and_candidate_seats() -> None:
    report = evaluate_against_heuristics(
        _model(),
        config=EvaluationConfig(root_seed=99, games_per_seat_cell=1),
        device=torch.device("cpu"),
    )
    assert report.games == 5 * (3 + 4 + 5)
    assert len(report.cell_metrics) == 15
    assert report.illegal_actions == 0
    assert report.faults == 0
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_evaluation.py -q
```

Expected: collection fails because evaluation/league modules are absent.

- [ ] **Step 3: Implement paired plans and raw utility**

For each chart, player count, repetition, and candidate seat, create one plan
with the candidate assigned to that seat and incumbent to all others. Mark
every seat `trainable=False`. Use a held-out `"evaluation"` seed namespace that
does not overlap training.

For each matched plan, compute candidate normalized final money minus the mean
incumbent normalized final money, plus candidate first-place share minus mean
incumbent first-place share.

- [ ] **Step 4: Implement deterministic paired bootstrap and league promotion**

Seed NumPy's local generator from the evaluation root seed. Resample paired
game records with replacement `bootstrap_samples` times. Use linear quantiles
at `(1-confidence)/2` and `1-(1-confidence)/2`. Promote only when the lower
bound is greater than zero and faults/illegal actions are both zero.

`League.promote` requires `report.candidate_identity` not already present and
`report.incumbent_identity == champion`; return a new immutable league with the
candidate appended and champion replaced.

`plan_league_episodes` deterministically selects the configured fraction of
each cell as league games. Each league game rotates one current-policy seat and
fills remaining seats from champion/older identities in stable round-robin
order. Mirror games remain all-current and all-trainable.

For heuristic evaluation, generalize the existing Stage 1 single-learner
collector across charts A-E and player counts 3-5. Rotate the neural candidate
through every seat and fill other seats in stable cyclic order from aggressive,
balanced, passive, and random `BotSpec` values. Record only raw outcome and
behavior metrics; do not update the model or optimizer.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_evaluation.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/evaluation.py \
  src/garboid_pocketrocks/neural/league.py \
  tests/neural/test_evaluation.py
git commit -m "feat: evaluate and promote self-play checkpoints"
```

### Task 9: Orchestrate train/resume/evaluate/inspect and committed profiles

**Files:**
- Create: `src/garboid_pocketrocks/neural/trainer.py`
- Modify: `src/garboid_pocketrocks/neural/cli.py`
- Modify: `src/garboid_pocketrocks/neural/smoke.py`
- Create: `configs/neural/smoke.json`
- Create: `configs/neural/initial-10m.json`
- Create: `configs/neural/long-8h.json`
- Create: `tests/neural/test_trainer.py`
- Modify: `tests/neural/test_cli.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_dir: Path
    final_checkpoint: Path
    completed_updates: int
    completed_episodes: int
    completed_decisions: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SelfPlaySmokeResult:
    completed_updates: int
    completed_episodes: int
    completed_decisions: int
    cell_games: tuple[tuple[str, int, int], ...]
    games_per_second: float
    decisions_per_second: float
    illegal_actions: int
    faults: int
    value: ValueMetrics
    checkpoint_replay_verified: bool
    resume_verified: bool


def smoke_run_config() -> TrainingRunConfig: ...
def run_self_play_smoke(
    config: TrainingRunConfig,
    output_dir: Path,
) -> SelfPlaySmokeResult: ...


def train(config: TrainingRunConfig, output_dir: Path) -> TrainingRunResult: ...
def resume(
    checkpoint: Path,
    output_dir: Path,
    *,
    max_additional_updates: int | None = None,
) -> TrainingRunResult: ...
def inspect_checkpoint(checkpoint: Path) -> dict[str, object]: ...
```

- [ ] **Step 1: Write RED orchestration and CLI tests**

```python
def test_low_volume_train_resume_and_inspect(tmp_path: Path) -> None:
    config = replace(
        TrainingRunConfig(),
        device="cpu",
        games_per_cell=1,
        max_updates=1,
        max_wall_seconds=None,
        parallel=ParallelConfig(workers=1, active_games_per_worker=4, max_inference_batch=32),
    )
    first = train(config, tmp_path / "run")
    inspected = inspect_checkpoint(first.final_checkpoint)
    resumed = resume(
        first.final_checkpoint,
        tmp_path / "resumed",
        max_additional_updates=1,
    )
    assert inspected["completed_episodes"] == 15
    assert resumed.completed_updates == first.completed_updates + 1


def test_cli_exposes_all_training_commands() -> None:
    help_text = _run_cli("--help")
    for command in ("smoke", "train", "resume", "evaluate", "inspect"):
        assert command in help_text


def test_committed_profiles_have_exact_wall_envelopes() -> None:
    smoke = TrainingRunConfig.from_json(Path("configs/neural/smoke.json"))
    initial = TrainingRunConfig.from_json(Path("configs/neural/initial-10m.json"))
    long = TrainingRunConfig.from_json(Path("configs/neural/long-8h.json"))
    assert smoke.games_per_cell == 100
    assert initial.max_wall_seconds == 600.0
    assert initial.checkpoint_interval_seconds == 120.0
    assert long.max_wall_seconds == 28_800.0
    assert long.checkpoint_interval_seconds == 900.0
    assert long.evaluation_interval_seconds == 1_800.0
```

- [ ] **Step 2: Run RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_trainer.py tests/neural/test_cli.py -q
```

Expected: FAIL because orchestration commands/profiles are absent.

- [ ] **Step 3: Implement the update-boundary trainer**

`train`:

1. reject a nonempty output directory;
2. resolve/calibrate device and parallel settings;
3. write `resolved-config.json` and `benchmark.json`;
4. initialize model/trainer/progress;
5. plan the next balanced rollout;
6. clone the model into an eval-mode collection snapshot;
7. collect, pack, and update PPO;
8. append one sorted JSON line to `metrics.jsonl`;
9. save on the resolved checkpoint boundary;
10. evaluate on the resolved evaluation boundary;
11. stop before an update when remaining wall budget is less than rolling p95
    update duration;
12. save the final update-boundary checkpoint.

If `evaluate_at_start`, evaluate the seeded initial checkpoint before the first
update. If `evaluate_at_end`, evaluate the final candidate against the starting
or current champion checkpoint after the final update. Periodic evaluation uses
`evaluation_interval_seconds`; every evaluation uses
`evaluation_games_per_seat_cell`.

When `target_decisions_per_update` is configured, estimate decisions per game
from calibration and then the rolling mean of the last three updates. Resolve:

```python
games_per_cell = max(
    1,
    math.ceil(target_decisions_per_update / (15.0 * estimated_decisions_per_game)),
)
```

Record the resolved games-per-cell value in each update metric. Never change it
inside an update. The fixed smoke uses `games_per_cell=100`; duration profiles
set `games_per_cell=null` and use their decision target.

Before a promotion, training plans are mirror-only. After promotion, use
`plan_league_episodes` with the resolved `league_fraction`, load immutable
league models once per collection interval, and include only current-policy
transitions in PPO.

Checkpoint retention keeps `latest`, `best`, and the newest
`keep_periodic_checkpoints` interval snapshots. Validate a newly written
checkpoint and update aliases atomically before removing an older periodic
snapshot.

`resume` loads the manifest/config/progress/model/optimizer/generators, creates
a new run directory whose lineage points to the source checkpoint, and starts
at `next_update_index`. `max_additional_updates`, when supplied by tests or
CLI, limits only the resumed invocation and does not alter the checkpointed
training configuration.

Catch SIGINT/SIGTERM by setting a stop event. Do not mutate model/optimizer
mid-update in the signal handler.

- [ ] **Step 4: Implement CLI and committed JSON profiles**

Commands:

```text
smoke --output-dir PATH [--games-per-cell N] [--device ...] [--workers ...]
train --config PATH --output-dir PATH
resume --checkpoint PATH --output-dir PATH [--max-additional-updates N]
evaluate --checkpoint PATH --config PATH --output PATH
inspect --checkpoint PATH [--format text|json]
```

`configs/neural/smoke.json` uses root 42, `games_per_cell=100`,
`max_updates=1`, `device="auto"`, `workers="auto"`, one PPO epoch, and no wall
limit. It disables strength evaluation because smoke acceptance is mechanical.

`initial-10m.json` uses root 42, `games_per_cell=null`, a 600-second limit, two PPO epochs,
checkpoint interval 120 seconds, baseline/final evaluation, and
`target_decisions_per_update=8192`, `league_fraction=0.0`, and four retained
periodic checkpoints as the pre-calibration starting settings. It sets
`evaluate_at_start=true`, `evaluate_at_end=true`, and
`evaluation_games_per_seat_cell=2`.

`long-8h.json` uses root 42, `games_per_cell=null`, a 28,800-second limit, three PPO epochs,
checkpoint interval 900 seconds, evaluation interval 1,800 seconds, and
`target_decisions_per_update=32768`, `league_fraction=0.2`, and four retained
periodic checkpoints. The initial run's recommendation may override the
decision target in the resolved config. It sets `evaluate_at_end=true` and
`evaluation_games_per_seat_cell=4`.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_trainer.py tests/neural/test_cli.py -q
```

Expected: PASS.

Commit:

```bash
git add src/garboid_pocketrocks/neural/trainer.py \
  src/garboid_pocketrocks/neural/cli.py \
  src/garboid_pocketrocks/neural/smoke.py \
  configs/neural/smoke.json configs/neural/initial-10m.json \
  configs/neural/long-8h.json \
  tests/neural/test_trainer.py tests/neural/test_cli.py
git commit -m "feat: orchestrate durable neural training runs"
```

### Task 10: Verify the 1,500-game smoke and document measured run plans

**Files:**
- Modify: `tests/neural/test_smoke.py`
- Modify: `src/garboid_pocketrocks/neural/README.md`
- Modify: `README.md`
- Create: `docs/benchmarks/2026-07-29-neural-self-play-smoke.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: the user-facing smoke artifact and measured initial/long cadence
  recommendation.

- [ ] **Step 1: Replace the Stage 1 smoke integration contract with the full profile**

Keep the existing Stage 1 unit fixture test. Add a low-volume CI test:

```python
@pytest.mark.neural_smoke
def test_full_curriculum_smoke_contract_at_one_game_per_cell(tmp_path: Path) -> None:
    result = run_self_play_smoke(
        replace(
            smoke_run_config(),
            games_per_cell=1,
            device="cpu",
            parallel=ParallelConfig(workers=2, active_games_per_worker=4, max_inference_batch=64),
        ),
        tmp_path / "smoke",
    )
    assert result.completed_episodes == 15
    assert result.completed_updates == 1
    assert result.illegal_actions == 0
    assert result.faults == 0
    assert result.checkpoint_replay_verified
    assert result.resume_verified
    assert {games for _, _, games in result.cell_games} == {1}
    assert result.games_per_second > 0.0
    assert result.decisions_per_second > 0.0
    assert result.value.count > 0
```

- [ ] **Step 2: Run focused RED/GREEN smoke test**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural/test_smoke.py -q -m neural_smoke
```

Expected after implementation: PASS for the 15-game override.

- [ ] **Step 3: Run complete quality gates**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run mypy src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  mypy --config-file mypy.neural.ini src tests
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  pytest tests/neural -q
```

Expected: all commands PASS with no warnings or skipped required neural tests.

- [ ] **Step 4: Run the user-facing 1,500-game smoke**

Run:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  garboid-train smoke \
  --output-dir artifacts/neural-self-play-smoke
```

Expected:

- exactly 1,500 completed games;
- exactly 100 games in each chart/player-count cell;
- zero illegal actions and faults;
- finite policy/value/gradient metrics;
- changed parameters;
- checkpoint replay and update-boundary resume both verified;
- nonzero games/s and decisions/s;
- benchmark results for every attempted device/worker candidate.

- [ ] **Step 5: Write benchmark and resolved cadence documentation**

Create `docs/benchmarks/2026-07-29-neural-self-play-smoke.md` from the smoke
JSON, including:

- host, Python, Torch, chosen device, workers, active games, inference batch;
- total/collection/PPO/checkpoint/resume seconds;
- games/s and decisions/s overall and per player count;
- inference batch p50/p95 and IPC percentage;
- policy/value metrics;
- the exact initial run command;
- predicted updates/games/decisions for 600 seconds;
- the exact long resume command;
- predicted update/checkpoint/evaluation counts for 28,800 seconds;
- a warning that throughput projections are not strength claims.

- [ ] **Step 6: Document commands and commit**

Update both READMEs with install, smoke, train, resume, evaluate, inspect, GPU
selection, worker behavior, artifact layout, and checkpoint portability.

Commit:

```bash
git add tests/neural/test_smoke.py \
  src/garboid_pocketrocks/neural/README.md README.md \
  docs/benchmarks/2026-07-29-neural-self-play-smoke.md
git commit -m "docs: report parallel neural smoke benchmark"
```

## Final verification

- [ ] Confirm the final worktree contains only intended changes:

```bash
git status --short
git log --oneline --decorate -12
```

- [ ] Inspect the smoke result and checkpoint:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  garboid-train inspect \
  --checkpoint artifacts/neural-self-play-smoke/checkpoints/latest \
  --format json
```

- [ ] Execute the initial profile after smoke acceptance:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  garboid-train train \
  --config configs/neural/initial-10m.json \
  --output-dir artifacts/neural-initial-10m
```

- [ ] Use the initial run's final checkpoint and resolved recommendation to
  prepare, but do not start without an explicit user request, the eight-hour
  command:

```bash
UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache uv run --extra neural \
  garboid-train resume \
  --checkpoint artifacts/neural-initial-10m/checkpoints/latest \
  --output-dir artifacts/neural-long-8h
```
