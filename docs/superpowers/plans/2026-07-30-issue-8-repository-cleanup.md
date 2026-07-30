# Issue 8 Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify and document the repository without changing bot behavior,
frozen identities, checkpoint compatibility, seeded simulation, replay, or
tournament results.

**Architecture:** Remove unsupported and duplicate surfaces before extracting
state machines. Keep compatibility at artifact boundaries, establish small
shared helpers only where current code is duplicated, and use the scalar
simulator plus existing frozen policies as behavioral oracles.

**Tech Stack:** Python 3.14, PocketRocks SDK batch/scalar engines, PyTorch,
pytest, Ruff, mypy, uv.

## Global Constraints

- Base all work on `codex/issue-8-cleanup` from refreshed `origin/main`.
- Do not change bot decisions, coefficients, rewards, model weights, identities,
  checkpoint bytes, seed derivation, replay schemas, or tournament rating
  semantics.
- Preserve historical configuration and checkpoint deserialization; reject
  unsupported settings only when executing `train` or `resume`.
- Keep benchmark reports, machine-readable benchmark artifacts,
  `docs/analysis/heuristic-bot-visualizations.md`, and
  `.agents/skills/versioning-bots/`.
- Remove undocumented legacy CLI behavior rather than retaining aliases.
- Every behavior-preserving refactor must first have a characterization test
  at the same boundary.
- Use `-n 0` for order-sensitive targeted tests. The final suite may use the
  repository's configured parallel test execution.

---

### Task 1: Reject unsupported neural runtime controls

**Files:**

- Modify: `src/garboid_pocketrocks/neural/run_config.py`
- Modify: `src/garboid_pocketrocks/neural/trainer.py`
- Modify: `configs/neural/initial-10m.json`
- Modify: `configs/neural/long-8h.json`
- Modify: `tests/neural/test_trainer.py`
- Modify: `tests/neural/test_training_checkpoint.py`

**Interfaces:**

- Produces: `validate_runtime_support(config: TrainingRunConfig) -> None`
- Preserves: `TrainingRunConfig.from_json()` and checkpoint manifest loading
  for configurations containing unsupported historical fields.

- [ ] **Step 1: Write failing support-validation tests**

Add to `tests/neural/test_trainer.py`:

```python
@pytest.mark.parametrize(
    ("changes", "field"),
    (
        ({"checkpoint_interval_seconds": 60.0}, "checkpoint_interval_seconds"),
        ({"keep_periodic_checkpoints": 2}, "keep_periodic_checkpoints"),
        ({"evaluation_interval_seconds": 60.0}, "evaluation_interval_seconds"),
        ({"evaluation_games_per_seat_cell": 4}, "evaluation_games_per_seat_cell"),
        ({"evaluate_at_start": True}, "evaluate_at_start"),
        ({"evaluate_at_end": True}, "evaluate_at_end"),
        ({"league_fraction": 0.2}, "league_fraction"),
    ),
)
def test_runtime_support_rejects_ignored_controls(
    changes: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_runtime_support(replace(TrainingRunConfig(), **changes))


def test_train_rejects_unsupported_controls_before_creating_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="league_fraction"):
        train(
            replace(
                TrainingRunConfig(),
                league_fraction=0.2,
            ),
            output,
        )

    assert not output.exists()
```

In `tests/neural/test_training_checkpoint.py`, serialize and reload a
`TrainingRunConfig` containing every historical field above and assert the
loaded manifest still equals the original. This distinguishes readability from
runtime support.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_trainer.py \
  tests/neural/test_training_checkpoint.py -q
```

Expected: collection fails because `validate_runtime_support` does not exist.

- [ ] **Step 3: Implement execution-time support validation**

Add to `run_config.py`:

```python
def validate_runtime_support(config: TrainingRunConfig) -> None:
    """Reject parsed settings that the current trainer does not execute."""

    unsupported: list[str] = []
    if config.checkpoint_interval_seconds is not None:
        unsupported.append("checkpoint_interval_seconds")
    if config.keep_periodic_checkpoints != TrainingRunConfig().keep_periodic_checkpoints:
        unsupported.append("keep_periodic_checkpoints")
    if config.evaluation_interval_seconds is not None:
        unsupported.append("evaluation_interval_seconds")
    if config.evaluation_games_per_seat_cell != TrainingRunConfig().evaluation_games_per_seat_cell:
        unsupported.append("evaluation_games_per_seat_cell")
    if config.evaluate_at_start:
        unsupported.append("evaluate_at_start")
    if config.evaluate_at_end:
        unsupported.append("evaluate_at_end")
    if config.league_fraction != 0.0:
        unsupported.append("league_fraction")
    if unsupported:
        raise ValueError(
            "training configuration requests unsupported runtime controls: "
            + ", ".join(unsupported)
        )
```

Call it at the start of `train`, before `_prepare_run_dir`. In `resume`, call
it after resolving the effective configuration and before creating the resumed
run directory.

Remove the unsupported keys from both committed long-running JSON profiles.
Update `test_committed_profiles_have_exact_wall_envelopes` to assert the
supported wall/model/decision/thread settings and default values for the
removed controls.

- [ ] **Step 4: Verify GREEN and configuration compatibility**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_trainer.py \
  tests/neural/test_training_checkpoint.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/garboid_pocketrocks/neural/run_config.py \
  src/garboid_pocketrocks/neural/trainer.py \
  configs/neural/initial-10m.json \
  configs/neural/long-8h.json \
  tests/neural/test_trainer.py \
  tests/neural/test_training_checkpoint.py
git commit -m "fix: reject unsupported neural run controls"
```

### Task 2: Retire the legacy neural smoke path

**Files:**

- Modify: `src/garboid_pocketrocks/neural/cli.py`
- Modify: `src/garboid_pocketrocks/neural/smoke.py`
- Modify: `src/garboid_pocketrocks/neural/rollout.py`
- Modify: `src/garboid_pocketrocks/neural/seeding.py`
- Modify: `src/garboid_pocketrocks/neural/config.py`
- Modify: `src/garboid_pocketrocks/neural/metrics.py`
- Modify: `src/garboid_pocketrocks/neural/vector_parallel.py`
- Modify: `src/garboid_pocketrocks/neural/vector_pool.py`
- Modify: `src/garboid_pocketrocks/neural/advantages.py`
- Modify: `src/garboid_pocketrocks/neural/checkpoint.py`
- Modify: `src/garboid_pocketrocks/neural/ppo.py`
- Modify: `tests/neural/test_cli.py`
- Modify: `tests/neural/test_smoke.py`
- Modify: `tests/neural/test_collector.py`
- Modify: `tests/neural/test_metrics.py`
- Modify: `tests/neural/test_ppo.py`
- Modify: `tests/neural/test_torch_runtime.py`
- Modify: neural tests that call `stage1_encoder_config()` or
  `stage1_model_config()`
- Delete: `tests/neural/test_rollout.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**

- Produces: one `garboid-train smoke` implementation backed by the durable
  trainer.
- Produces: `SmokeResult`, `run_smoke()`, and `smoke_run_config()`.
- Produces: one canonical `RolloutBatch.episodes` tuple of
  `MultiSeatEpisode`.
- Removes: the metadata-only `evaluate` command, Stage 1 smoke flags/types,
  legacy single-seat rollout collection, and Stage 1 config names.

- [ ] **Step 1: Write failing CLI and rollout contract tests**

Update `tests/neural/test_cli.py`:

```python
def test_cli_exposes_only_supported_training_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in ("smoke", "train", "resume", "inspect"):
        assert command in output
    assert "evaluate" not in output


def test_smoke_help_has_one_current_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["smoke", "--help"]) == 0
    output = capsys.readouterr().out
    assert "--games-per-cell" in output
    assert "--workers" in output
    assert "--updates" not in output
    assert "--games-per-update" not in output
```

Change the low-volume CLI test to invoke:

```python
exit_code = main(
    [
        "smoke",
        "--output-dir",
        str(output_dir),
        "--seed",
        "42",
        "--device",
        "cpu",
        "--workers",
        "1",
        "--games-per-cell",
        "1",
    ]
)
assert exit_code == 0
assert (output_dir / "self-play-smoke-result.json").is_file()
for name in ("manifest.json", "model.pt", "optimizer.pt", "rng.pt", "metrics.json"):
    assert (output_dir / "checkpoints/latest" / name).is_file()
assert not (output_dir / "smoke-result.json").exists()
assert not (output_dir / "checkpoint").exists()
```

In `tests/neural/test_collector.py`, assert:

```python
assert rollout.episodes
assert all(isinstance(episode, MultiSeatEpisode) for episode in rollout.episodes)
assert not hasattr(rollout, "multi_seat_episodes")
assert rollout.transitions == tuple(
    transition
    for episode in rollout.episodes
    for trajectory in episode.trajectories
    if trajectory.trainable
    for transition in trajectory.transitions
)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_cli.py \
  tests/neural/test_collector.py \
  tests/neural/test_smoke.py -q
```

Expected: failures because legacy commands, flags, and rollout properties still
exist.

- [ ] **Step 3: Make the current smoke the only smoke**

In `cli.py`, import only `run_smoke` and `smoke_run_config` from `smoke.py`.
Delete the `evaluate` dispatch/parser and the legacy smoke branch/flags. Keep
the existing production configuration override:

```python
def _smoke(arguments: argparse.Namespace) -> int:
    config = smoke_run_config()
    workers = config.parallel.workers if arguments.workers is None else arguments.workers
    resolved = replace(
        config,
        root_seed=arguments.seed,
        device=arguments.device or config.device,
        games_per_cell=arguments.games_per_cell or config.games_per_cell,
        parallel=replace(config.parallel, workers=workers),
    )
    result = run_smoke(resolved, arguments.output_dir)
    print(
        f"completed {result.completed_episodes} games "
        f"({result.games_per_second:.2f} games/s, "
        f"{result.decisions_per_second:.2f} decisions/s); "
        f"checkpoint: {arguments.output_dir / 'checkpoints/latest'}"
    )
    return 0
```

In `smoke.py`, retain the current durable smoke implementation and its value
metrics/checkpoint-resume helpers. Rename `SelfPlaySmokeResult` to
`SmokeResult` and `run_self_play_smoke` to `run_smoke`. Delete all legacy
types/functions beginning with `SmokeConfig`, `SmokeEpisodeMetrics`,
`SmokeUpdateMetrics`, `CheckpointReplay`, the old `SmokeResult`, and the old
`run_smoke`, plus their private helpers.

- [ ] **Step 4: Normalize rollout and model configuration**

In `rollout.py`, make this the only representation:

```python
@dataclass(frozen=True, slots=True)
class RolloutBatch:
    episodes: tuple[MultiSeatEpisode, ...]

    @classmethod
    def from_multi_seat(
        cls,
        episodes: Iterable[MultiSeatEpisode],
    ) -> RolloutBatch:
        collected = tuple(episodes)
        if not collected:
            raise ValueError("rollout batch must contain at least one episode")
        if not any(
            trajectory.trainable for episode in collected for trajectory in episode.trajectories
        ):
            raise ValueError("rollout batch must contain a trainable trajectory")
        return cls(episodes=collected)

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        return tuple(
            transition
            for episode in self.episodes
            for trajectory in episode.trajectories
            if trajectory.trainable
            for transition in trajectory.transitions
        )
```

Update `PackedRollout.from_batch`, metrics, vector parallel collection, and the
persistent pool to use `rollout.episodes`.

Delete legacy `EpisodePlan`, `plan_stage1_episodes`, `collect_rollout`,
`RolloutEpisode`, and their single-agent helpers. Replace all
`stage1_model_config()` calls with `training_model_config("small")`, and
`stage1_encoder_config()` with `training_encoder_config()`. Preserve exact
small-model dataclass values so frozen checkpoint tests remain unchanged.

Move deterministic runtime coverage from the deleted
`tests/neural/test_rollout.py` into `test_torch_runtime.py`. Migrate PPO tests
to a one-game current collector fixture.

- [ ] **Step 5: Remove obsolete descriptions and CI references**

Remove "Stage 1" descriptions from shared production modules. Change CI's
neural smoke target to:

```text
tests/neural/test_smoke.py::test_full_curriculum_smoke_contract_at_one_game_per_cell
```

Run:

```bash
rg -n \
  'Stage 1|stage1_|SmokeConfig|SelfPlaySmokeResult|run_self_play_smoke|collect_rollout|RolloutEpisode|EpisodePlan|plan_stage1_episodes|configure_deterministic_torch|multi_seat_episodes|games_per_update|--games-per-update|--updates' \
  src tests README.md .github
```

Expected: no live-code or live-documentation matches.

- [ ] **Step 6: Verify the complete neural contract**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_smoke.py \
  tests/neural/test_cli.py \
  tests/neural/test_collector.py \
  tests/neural/test_metrics.py \
  tests/neural/test_ppo.py \
  tests/neural/test_torch_runtime.py \
  tests/neural/test_checkpoint.py \
  tests/neural/test_smoke_tournament_bot.py \
  tests/neural/test_vector_collector.py \
  tests/neural/test_vector_parallel.py \
  tests/neural/test_vector_pool.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: retire legacy neural smoke path"
```

### Task 3: Centralize domain naming and remove dead aliases

**Files:**

- Modify: `src/garboid_pocketrocks/knowledge.py`
- Modify: callers with duplicated ruleset/chart conversion
- Modify: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Modify: `tests/test_knowledge.py`
- Modify: `tests/simulator/test_monte_carlo.py`

**Interfaces:**

- Produces:
  `ruleset_name(value_chart: str, objectives_enabled: bool = True) -> str`
- Produces:
  `value_chart_from_ruleset_name(name: str) -> str`
- Removes: `MonteCarloResult.games` and `.statistics`.
- Preserves serialized `ruleset_name` fields and `live-A` through `live-E`
  artifact values.

- [ ] **Step 1: Write failing naming and alias tests**

Add:

```python
def test_ruleset_names_are_canonical_sdk_boundaries() -> None:
    assert ruleset_name("a") == "live-A"
    assert ruleset_name("E", objectives_enabled=False) == "live-E-no-objectives"
    assert value_chart_from_ruleset_name("live-A") == "A"
    assert value_chart_from_ruleset_name("live-E-no-objectives") == "E"


@pytest.mark.parametrize("name", ("A", "live-Z", "live-", ""))
def test_unknown_ruleset_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="ruleset"):
        value_chart_from_ruleset_name(name)
```

Add to the Monte Carlo tests:

```python
assert result.game_summaries
assert result.bot_statistics
assert not hasattr(result, "games")
assert not hasattr(result, "statistics")
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run pytest -n 0 tests/test_knowledge.py tests/simulator/test_monte_carlo.py -q
```

Expected: missing helper and alias assertions fail.

- [ ] **Step 3: Implement canonical boundary helpers**

Validate chart names against SDK-supported charts, normalize lowercase input,
and explicitly parse `live-X` plus `live-X-no-objectives`. Replace local
`_variant_name` helpers and direct `removeprefix("live-")` conversions in
simulator, neural, and tournament code. Do not rename stored dataclass or JSON
fields.

Delete only the two `MonteCarloResult` properties. Confirm that similarly named
`BotStatistics.games` fields remain.

- [ ] **Step 4: Verify broad deterministic consumers**

Run:

```bash
uv run pytest -n 0 \
  tests/test_knowledge.py \
  tests/simulator/test_monte_carlo.py \
  tests/neural/test_planning.py \
  tests/neural/test_metrics.py \
  tests/tournament/test_analysis.py \
  tests/tournament/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "refactor: centralize ruleset naming"
```

### Task 4: Extract simulator execution phases

**Files:**

- Create: `src/garboid_pocketrocks/simulator/bot_execution.py`
- Modify: `src/garboid_pocketrocks/simulator/runner.py`
- Modify: `src/garboid_pocketrocks/simulator/batch_match.py`
- Create: `tests/simulator/test_batch_match.py`
- Modify: `tests/simulator/test_runner.py`
- Modify: `tests/simulator/test_monte_carlo.py`

**Interfaces:**

- Produces shared construction/decision/fault handling while `runner.py`
  re-exports `FaultMode` and `BotFault`.
- Keeps `run_batch_matches()` as a short phase orchestrator.
- Preserves batch `MatchResult.events == ()`.

- [ ] **Step 1: Add scalar/batch characterization tests**

Create
`test_batch_match_matches_scalar_replay_bytes`, parameterized over player
counts `(3, 4, 5)`, charts `("A", "C", "E")`, and both objective modes.

The concrete assertions are:

- terminal `result`, `turns`, `faults`, and `replay` are equal;
- scalar and batch replay files written by `save_replay` have equal bytes;
- singleton versus multi-row batching returns identical row results in input
  order;
- history-aware brains receive the same setup/turn/resolution/reveal history;
- construction and runtime faults match under `RECORD_AND_PASS`;
- original exception type/message propagates under `RAISE`;
- decision steps are contiguous, bids use seat order, choice reveals contain
  only the winner, and automatic reveals create no decision step;
- empty jobs return `()`, mixed player counts fail, and lineup mismatch fails.

Use the existing test helpers from `test_runner.py` and
`test_batch_context.py`; do not use hidden engine state in assertions.

- [ ] **Step 2: Run characterization tests**

Run:

```bash
uv run pytest -n 0 \
  tests/simulator/test_batch_match.py \
  tests/simulator/test_batch_context.py \
  tests/simulator/test_runner.py \
  tests/simulator/test_replay.py \
  tests/simulator/test_session.py -q
```

Expected before extraction: PASS. These are characterization tests.

- [ ] **Step 3: Share bot execution and fault handling**

Move `FaultMode`, `BotFault`, brain construction, decision invocation,
fallback, and fault creation to `bot_execution.py`. The public internal
functions are `initialize_brains(lineup, *, seed, fault_mode)`, returning the
brain tuple and construction-fault tuple, and `choose_brain_decision(...)`,
accepting the brain, context, knowledge, public history, fault mode, mutable
fault list, turn index, seat, and bot name.

The implementation must retain the current seed order, original exception
propagation, one construction fault at turn zero, repeated runtime faults in
encounter order, bid-zero fallback, and reveal-index-zero fallback.

Import and re-export `FaultMode`/`BotFault` from `runner.py` so all existing
callers remain valid.

- [ ] **Step 4: Extract batch state and phases**

Add `_BatchRunState` and `_PendingBatchTurn` dataclasses. Make
`run_batch_matches` read:

```python
state = _initialize_batch(jobs)
while pending := _prepare_next_turn(state):
    _record_turn_opened(state, pending)
    _collect_bid_decisions(state, pending)
    outcome = _resolve_bids(state, pending)
    _record_bid_outcomes(state, pending, outcome)
    reveals = _resolve_reveals(state, pending, outcome)
    _record_reveals(state, reveals)
return _build_match_results(state)
```

Implement the exact private helpers named above plus
`_validate_batch_jobs`. Preserve snapshot copies, NumPy dtypes, history order,
choice-reveal visibility, effective bids, reward/fault turn indices,
competition ranks, and replay fields.

- [ ] **Step 5: Verify simulator and tournament parity**

Run:

```bash
uv run pytest -n 0 \
  tests/simulator/test_batch_match.py \
  tests/simulator/test_batch_context.py \
  tests/simulator/test_runner.py \
  tests/simulator/test_replay.py \
  tests/simulator/test_session.py \
  tests/simulator/test_monte_carlo.py \
  tests/tournament/test_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/garboid_pocketrocks/simulator tests/simulator
git commit -m "refactor: extract simulator batch phases"
```

### Task 5: Consolidate inference and extract vector collection phases

**Files:**

- Modify: `src/garboid_pocketrocks/neural/collector.py`
- Modify: `src/garboid_pocketrocks/neural/vector_collector.py`
- Modify: `src/garboid_pocketrocks/neural/parallel.py`
- Modify: `tests/neural/test_collector.py`
- Modify: `tests/neural/test_vector_collector.py`
- Modify: `tests/neural/test_parallel.py`

**Interfaces:**

- Produces one identity-grouped, row-seeded inference helper in `collector.py`.
- Keeps vector engine state owned by `_collect_engine_batch`; no general state
  machine class is introduced.

- [ ] **Step 1: Add ordering and failure characterization tests**

Add assertions that:

- model training mode is restored in `finally`, including inference failure;
- reversed request arrival preserves decisions keyed by
  `(episode_index, seat, decision_index)`;
- scalar/vector equality covers plan, trajectory identity/trainability,
  actions, complete `RewardBreakdown`, termination, encoded history/mask, and
  final scores.

- [ ] **Step 2: Run characterization tests**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_collector.py \
  tests/neural/test_vector_collector.py \
  tests/neural/test_parallel.py -q
```

Expected: PASS for existing behavior tests and RED only for newly exposed
failure-mode/order assertions.

- [ ] **Step 3: Consolidate row-seeded inference**

Add `_infer_policy_requests` to `collector.py`. It accepts the policy mapping,
pending-request sequence, device, maximum inference batch, and mutable batch
size list, and returns the ordered response tuple plus inference seconds.

It returns `(), 0.0` for no requests; groups by identity; iterates identities
sorted; sorts requests by `(episode_index, seat, decision_index)`; chunks
exactly; uses each existing `sampling_seed`; appends one batch size per model
call; and restores every model's prior mode in `finally`.

Delete `_infer_requests` from `vector_collector.py` and `_infer` from
`parallel.py`.

- [ ] **Step 4: Extract vector engine phases**

Keep existing state containers and extract:

```python
_initialize_public_histories
_append_turn_opened_events
_prepare_bid_requests
_record_bid_responses
_prepare_reveal_requests
_record_reveal_responses
_apply_and_record_reveals
_finalize_batch_episodes
```

Do not re-extract the already focused reward/result helpers. Preserve the exact
ordering of open-transition finalization, auction reward, public resolution,
choice reveal encoding, suit capture, engine mutation, history append, terminal
reward, and terminal transition finalization.

- [ ] **Step 5: Verify vector/scalar equivalence**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_collector.py \
  tests/neural/test_vector_collector.py \
  tests/neural/test_parallel.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/garboid_pocketrocks/neural/collector.py \
  src/garboid_pocketrocks/neural/vector_collector.py \
  src/garboid_pocketrocks/neural/parallel.py \
  tests/neural/test_collector.py \
  tests/neural/test_vector_collector.py \
  tests/neural/test_parallel.py
git commit -m "refactor: extract neural collection phases"
```

### Task 6: Extract parallel policy snapshots and coordination

**Files:**

- Create: `src/garboid_pocketrocks/neural/policy_snapshot.py`
- Modify: `src/garboid_pocketrocks/neural/vector_parallel.py`
- Modify: `src/garboid_pocketrocks/neural/vector_pool.py`
- Modify: `src/garboid_pocketrocks/neural/parallel.py`
- Modify: `tests/neural/test_vector_parallel.py`
- Modify: `tests/neural/test_vector_pool.py`
- Modify: `tests/neural/test_parallel.py`

**Interfaces:**

- Produces shared immutable policy snapshot/load functions.
- Preserves separate central-inference and local-inference execution models.
- Canonicalizes worker results by worker ID before aggregating diagnostic batch
  sizes.

- [ ] **Step 1: Add worker-order and pool-failure tests**

Add tests that reversed worker result arrival still produces episodes sorted
by `episode_index`, identical metrics, and `inference_batch_sizes` ordered by
worker ID. Assert post-dispatch failure closes a pool while pre-dispatch
validation failure leaves it usable.

- [ ] **Step 2: Run and verify RED where ordering is nondeterministic**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_parallel.py \
  tests/neural/test_vector_parallel.py \
  tests/neural/test_vector_pool.py -q
```

- [ ] **Step 3: Move policy snapshot operations**

Create `policy_snapshot.py` with the immutable snapshot record:

```python
@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    identity: str
    encoder_config: NeuralEncoderConfig
    model_config: NeuralModelConfig
    state_bytes: bytes
```

Add `snapshot_policies(policies)` returning snapshots in sorted identity order,
and `load_policy_snapshots(snapshots)` rebuilding a dictionary of CPU policies
after strict state loading. Use eager `torch.save` bytes. Remove the private
cross-module imports from `vector_pool.py`.

- [ ] **Step 4: Extract process coordination helpers**

For `vector_parallel.py`, extract:

```text
_shard_vector_plans
_spawn_vector_workers
_receive_vector_results
_shutdown_vector_workers
_aggregate_vector_results
```

For `VectorActorPool.collect`, extract:

```text
_prepare_collect_command
_dispatch_collect_commands
_receive_collect_results
_validate_worker_result
```

For `parallel.py`, extract:

```text
_spawn_plan_workers
_receive_ready_messages
_send_worker_responses
_shutdown_plan_workers
_build_parallel_metrics
```

Preserve request-ID increments, stale-result rejection, complete homogeneous
shards, canonical episode/cell ordering, parent policy freeze lifetime, and
post-dispatch shutdown semantics.

- [ ] **Step 5: Verify coordination**

Run:

```bash
uv run --extra neural pytest -n 0 \
  tests/neural/test_parallel.py \
  tests/neural/test_vector_parallel.py \
  tests/neural/test_vector_pool.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  src/garboid_pocketrocks/neural/policy_snapshot.py \
  src/garboid_pocketrocks/neural/parallel.py \
  src/garboid_pocketrocks/neural/vector_parallel.py \
  src/garboid_pocketrocks/neural/vector_pool.py \
  tests/neural/test_parallel.py \
  tests/neural/test_vector_parallel.py \
  tests/neural/test_vector_pool.py
git commit -m "refactor: extract neural actor coordination"
```

### Task 7: Extract PPO orchestration and belief accounting

**Files:**

- Modify: `src/garboid_pocketrocks/neural/ppo.py`
- Modify: `tests/neural/test_ppo.py`
- Modify: `src/garboid_pocketrocks/heuristics/belief.py`
- Modify: `tests/heuristics/test_belief.py`

**Interfaces:**

- Produces deterministic minibatch iteration and focused PPO metrics
  accumulation without changing the numerical training step.
- Produces `_PublicCardAccounting` plus focused expectation helpers without
  changing public belief results or error precedence.

- [ ] **Step 1: Add PPO RNG and exact-order characterization**

Assert:

```python
before = torch.get_rng_state().clone()
first = trainer.update(rollout, update_seed=123)
after = torch.get_rng_state()
assert torch.equal(before, after)
assert first == repeated
assert first_parameter_state == repeated_parameter_state
assert first_optimizer_state == repeated_optimizer_state
```

Also pin minibatch indices to one CPU `randperm` per epoch seeded by
`_derive_local_seed(update_seed, "epoch", epoch_index)`.

- [ ] **Step 2: Extract PPO validation, iteration, and accumulation**

Add:

```python
def _validate_update_seed(update_seed: int) -> None:
    if not isinstance(update_seed, int) or isinstance(update_seed, bool) or update_seed < 0:
        raise PPOError("update seed must be a nonnegative integer")


def _iter_minibatch_indices(
    *,
    transition_count: int,
    epochs: int,
    minibatch_size: int,
    update_seed: int,
) -> Iterator[NDArray[np.int64]]:
    for epoch_index in range(epochs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_derive_local_seed(update_seed, "epoch", epoch_index))
        permutation = torch.randperm(
            transition_count,
            generator=generator,
        ).numpy()
        for start in range(0, transition_count, minibatch_size):
            yield permutation[start : start + minibatch_size]


@dataclass(slots=True)
class _PPOUpdateAccumulator:
    total_loss_sum: float = 0.0
    policy_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    entropy_sum: float = 0.0
    approximate_kl_sum: float = 0.0
    clip_count: int = 0
    optimizer_steps: int = 0
    ratios: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    pre_clip_norms: list[float] = field(default_factory=list)
    post_clip_norms: list[float] = field(default_factory=list)
```

Add `_record_minibatch_metrics` and `_build_update_metrics`. Leave forward,
mask validation, loss, backward, finite checks, clipping, and optimizer step
inline and in their current order. Preserve sample-weighted sums and the
`transition_count * epochs` denominator.

- [ ] **Step 3: Verify PPO numerical identity**

Run:

```bash
uv run --extra neural pytest -n 0 tests/neural/test_ppo.py -q
```

Expected: PASS.

- [ ] **Step 4: Extract public card accounting**

Add:

```python
@dataclass(frozen=True, slots=True)
class _PublicCardAccounting:
    known_terminal_reveals: tuple[int, ...]
    unseen_by_suit: tuple[int, ...]
    known_future_by_suit: tuple[int, ...]
    opponent_hidden_slots: int
    unseen_population: int
    unknown_future_biddable: int
    future_biddable: int
    total_biddable: int
```

Extract `_account_public_cards`, `_expected_future_biddable_counts`,
`_expected_terminal_price`, and `_build_suit_beliefs`. Keep
`offered_resource_counts` in place because valuation imports it. Preserve error
order: knowledge, context, conservation, posterior, finite expectation.

- [ ] **Step 5: Verify exact belief behavior**

Run:

```bash
uv run pytest -n 0 tests/heuristics/test_belief.py -q
```

Expected: PASS, including property-generated engine contexts.

- [ ] **Step 6: Commit**

```bash
git add \
  src/garboid_pocketrocks/neural/ppo.py \
  tests/neural/test_ppo.py \
  src/garboid_pocketrocks/heuristics/belief.py \
  tests/heuristics/test_belief.py
git commit -m "refactor: extract training and belief accounting"
```

### Task 8: Build the documentation hierarchy and prune history

**Files:**

- Create: `docs/README.md`
- Create: `docs/architecture/sdk-authority.md`
- Create: `docs/architecture/public-information-boundary.md`
- Create: `docs/architecture/immutable-bot-identities.md`
- Create: `docs/architecture/deterministic-evaluation.md`
- Create: `src/garboid_pocketrocks/simulator/README.md`
- Create: `src/garboid_pocketrocks/tournament/README.md`
- Create: `src/garboid_pocketrocks/heuristics/README.md`
- Create: `tests/test_documentation.py`
- Modify: `README.md`
- Modify: `src/garboid_pocketrocks/neural/README.md`
- Delete: every pre-issue-8 file under `docs/superpowers/plans/`
- Delete: every superseded pre-issue-8 file under
  `docs/superpowers/specs/`
- Keep: the current issue #8 design and plan during review

**Interfaces:**

- Produces a single navigation path from root to current domain runbooks,
  architecture decisions, analyses, and benchmarks.
- Produces a persistent local Markdown link integrity test.

- [ ] **Step 1: Write a failing local-link integrity test**

Create `tests/test_documentation.py`. Scan:

```python
MARKDOWN_ROOTS = (
    Path("README.md"),
    Path("docs"),
    Path("src"),
    Path(".agents"),
)
```

For every `*.md`, extract Markdown link targets. Ignore `http:`, `https:`,
`mailto:`, and target-only `#anchors`. Strip an anchor from local targets,
URL-decode it, resolve it relative to the containing document, and collect
missing paths. Assert the sorted missing list is empty and include
`source -> target` in the failure.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
uv run pytest -n 0 tests/test_documentation.py -q
```

Expected: FAIL on existing stale/broken historical links.

- [ ] **Step 3: Write current documentation**

The root README contains only:

- project purpose and authoritative SDK/rules source;
- architecture data flow;
- setup;
- live bot, simulation, tournament, and neural command summaries;
- frozen identity warning;
- links to `docs/README.md` and domain runbooks.

`docs/README.md` identifies:

- root README as quick start;
- package READMEs as current operational truth;
- `docs/architecture` as enduring decisions;
- `docs/analysis` as reproduction runbooks;
- `docs/benchmarks` and tournament subdirectories as evidence;
- Superpowers issue #8 files as active review artifacts, not operational docs.

The four ADRs capture only current rules: SDK engine authority, allowlisted
public information, immutable released identities, and deterministic
development/held-out evaluation separation.

Package READMEs document commands, invariants, and extension points without
duplicating function-by-function listings.

- [ ] **Step 4: Remove superseded transcripts**

Delete all 15 pre-issue-8 implementation plans and all superseded
pre-issue-8 specs. Before deletion, ensure every enduring fact appears in a
current README/ADR. Update the root link to the old heuristic design to the new
heuristics runbook.

Retain benchmark reports/artifacts, the visualization runbook, and the
versioning-bots skill.

- [ ] **Step 5: Verify documentation and stale-reference removal**

Run:

```bash
uv run pytest -n 0 tests/test_documentation.py -q
rg -n \
  'docs/superpowers/(plans|specs)/(2026-07-28|2026-07-29|2026-07-30-(large|smoke|tournament))' \
  README.md docs src tests .github
```

Expected: link test PASS and no references to removed files.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: replace implementation transcripts with runbooks"
```

### Task 9: Integrated verification and cleanup audit

**Files:**

- Modify only files required by failures found during this gate.

**Interfaces:**

- Proves issue #8 cleanup is behavior-preserving and complete.

- [ ] **Step 1: Run formatting, lint, types, and lock checks**

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural mypy --config-file mypy.neural.ini src tests
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Run targeted deterministic contracts**

```bash
uv run pytest -n 0 \
  tests/simulator/test_batch_match.py \
  tests/simulator/test_batch_context.py \
  tests/simulator/test_monte_carlo.py \
  tests/simulator/test_replay.py \
  tests/tournament/test_runner.py \
  tests/neural/test_checkpoint.py \
  tests/neural/test_smoke_tournament_bot.py \
  tests/neural/test_collector.py \
  tests/neural/test_vector_collector.py \
  tests/neural/test_parallel.py \
  tests/neural/test_vector_parallel.py \
  tests/neural/test_vector_pool.py \
  tests/neural/test_ppo.py \
  tests/neural/test_training_checkpoint.py \
  tests/heuristics/test_belief.py \
  tests/test_documentation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full suite outside restricted multiprocessing**

```bash
uv run --extra neural pytest
```

Expected: every test passes. Any macOS semaphore permission failure must be
rerun outside the filesystem/process sandbox; it is not a code failure.

- [ ] **Step 4: Re-audit simplification**

Run:

```bash
rg -n \
  'Stage 1|stage1_|evaluate_at_|evaluation_interval_seconds|league_fraction|multi_seat_episodes|--updates|--games-per-update' \
  src tests README.md docs .github configs
wc -l \
  src/garboid_pocketrocks/simulator/batch_match.py \
  src/garboid_pocketrocks/neural/smoke.py \
  src/garboid_pocketrocks/neural/vector_collector.py \
  src/garboid_pocketrocks/neural/parallel.py \
  src/garboid_pocketrocks/neural/vector_parallel.py \
  src/garboid_pocketrocks/neural/ppo.py \
  src/garboid_pocketrocks/heuristics/belief.py
```

Expected:

- no Stage 1 or legacy CLI/rollout references;
- unsupported config names exist only in backward-compatible configuration
  parsing, validation, and tests;
- orchestration functions are visibly composed from named helpers;
- documentation points to current runbooks rather than removed transcripts.

- [ ] **Step 5: Commit any verification-only fixes**

If verification required changes:

```bash
git add -A
git commit -m "test: complete issue 8 cleanup verification"
```

If no files changed, do not create an empty commit.

- [ ] **Step 6: Prepare the draft PR**

Push `codex/issue-8-cleanup` and create a draft PR against `main` with:

- `Addresses #8` near the top;
- a summary of documentation, dead-code, naming, configuration, and state
  machine changes;
- explicit behavior-preservation constraints;
- exact verification commands/results;
- a note that issue #8 should close only after review and every acceptance
  claim remains supported.
