# Large 350k Neural Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze `vector_ppo_large_v1_g350k` as a portable inference bot, add it to the standard tournament, and run the fixed-seed 15,000-game benchmark against the existing default field.

**Architecture:** Export the validated 349,860-game training state as a model-only checkpoint beside the immutable smoke model. Refactor the tournament adapter into one shared deterministic implementation selected by explicit checkpoint subclasses, register both neural generations, and leave the SDK batch engine and tournament analysis unchanged.

**Tech Stack:** Python 3.14, PocketRocks SDK `BatchSimEngine`, PyTorch 2.13, pytest, Ruff, strict mypy, Plackett-Luce tournament analysis.

## Global Constraints

- Preserve `vector_ppo_small_v1_g1500` and its checkpoint byte-for-byte.
- Name the new local simulation bot `vector_ppo_large_v1_g350k`.
- Preserve the exact 349,860 completed games and 196 updates in the inference manifest.
- Commit only `manifest.json` and `model.pt`; exclude optimizer, RNG, and training metrics.
- Keep Torch optional at ordinary registry import time; tournament execution uses `uv run --extra neural`.
- Keep the tournament seed at 0, games at 15,000, charts at A-E, player counts at 3-5, batch size at 64, and bootstrap samples at 200.
- Do not claim strength improvement unless the fixed-seed tournament supports it.

---

### Task 1: Export the checkpoint and share the frozen neural adapter

**Files:**
- Create: `src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k/manifest.json`
- Create: `src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k/model.pt`
- Modify: `src/garboid_pocketrocks/neural/tournament_bot.py`
- Modify: `tests/neural/test_smoke_tournament_bot.py`

**Interfaces:**
- Consumes: `export_inference_checkpoint(training_checkpoint: Path, output_path: Path, *, device: torch.device) -> Path`.
- Produces: `LARGE_BOT_NAME`, `LARGE_CHECKPOINT_PATH`, `VectorPpoLargeV1G350kBrain`, and `VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC`.

- [ ] **Step 1: Add failing checkpoint and adapter tests**

Extend `tests/neural/test_smoke_tournament_bot.py` with imports and assertions:

```python
from garboid_pocketrocks.neural.tournament_bot import (
    LARGE_BOT_NAME,
    LARGE_CHECKPOINT_PATH,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    VectorPpoLargeV1G350kBrain,
)


def test_large_checkpoint_is_frozen_at_rounded_training_age() -> None:
    loaded = load_inference_checkpoint(
        LARGE_CHECKPOINT_PATH,
        device=torch.device("cpu"),
    )

    assert LARGE_BOT_NAME == "vector_ppo_large_v1_g350k"
    assert {item.name for item in LARGE_CHECKPOINT_PATH.iterdir()} == {
        "manifest.json",
        "model.pt",
    }
    assert loaded.manifest.completed_episodes == 349_860
    assert loaded.manifest.completed_updates == 196
    assert loaded.manifest.supported_ruleset_names == tuple(
        f"live-{chart}" for chart in "ABCDE"
    )
    assert loaded.manifest.supported_player_counts == (3, 4, 5)
    assert len(loaded.manifest.parameter_digest) == 64


@pytest.mark.parametrize(
    "brain_type",
    (VectorPpoSmallV1G1500Brain, VectorPpoLargeV1G350kBrain),
)
def test_frozen_neural_brains_return_deterministic_legal_decisions(
    brain_type: type[VectorPpoSmallV1G1500Brain] | type[VectorPpoLargeV1G350kBrain],
) -> None:
    session = SdkGameSession.start(
        player_count=3,
        seed=19,
        value_chart="B",
        objectives_enabled=True,
        player_names=("neural", "random-1", "random-2"),
    )
    context = session.pending.contexts[0][1]
    history = public_history_from_sdk_events(session.events)
    knowledge = canonical_knowledge(3, value_chart="B")
    brain = brain_type(seed=7)

    first = brain.choose_decision_with_history(context, knowledge, history)
    second = brain.choose_decision_with_history(context, knowledge, history)

    assert first == second
    context.validate(first)


def test_large_spec_is_pickle_safe_and_completes_a_match() -> None:
    restored = pickle.loads(pickle.dumps(VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC))
    random_spec = BotSpec.from_bot_class(RandomBot)

    assert restored == VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC
    match = MatchRunner.run(
        (restored, random_spec, random_spec),
        player_count=3,
        seed=23,
        value_chart="E",
    )
    assert match.result.scores
    assert not match.faults
```

Replace the existing smoke-only deterministic test with the parametrized test
so the same behavior contract covers both versions.

- [ ] **Step 2: Run the tests and verify the missing large symbols fail**

Run:

```bash
uv run --extra neural pytest tests/neural/test_smoke_tournament_bot.py -q
```

Expected: collection fails because the large checkpoint constants and brain do
not exist.

- [ ] **Step 3: Export the model-only checkpoint**

Run from the feature worktree:

```bash
uv run --extra neural python -c 'from pathlib import Path; import torch; from garboid_pocketrocks.neural.training_checkpoint import export_inference_checkpoint; export_inference_checkpoint(Path("/Users/Christopher.Garber/Documents/garboid-pocketrocks/artifacts/vector_ppo_large_v1_overnight_20260730/checkpoints/vector_ppo_large_v1_g350k"), Path("src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k"), device=torch.device("cpu"))'
```

Verify:

```bash
find src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k \
  -maxdepth 1 -type f -print | sort
du -h src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k/*
```

Expected: exactly `manifest.json` and `model.pt`; no optimizer, RNG, or metrics
payload.

- [ ] **Step 4: Refactor the adapter and add the large explicit subclass**

In `src/garboid_pocketrocks/neural/tournament_bot.py`, retain the existing
public smoke names while replacing the file with:

```python
"""Frozen neural policies used by the standard local tournament."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pocketrocks import BotDecision, DecisionContext

from garboid_pocketrocks.adapters.public_history import PublicHistory
from garboid_pocketrocks.bots.base import BotSpec
from garboid_pocketrocks.knowledge import RulesetKnowledge

if TYPE_CHECKING:
    import torch

    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.neural.model import NeuralPolicy
    from garboid_pocketrocks.training.actions import ActionCodec

CHECKPOINTS_PATH = Path(__file__).with_name("checkpoints")
SMOKE_BOT_NAME = "vector_ppo_small_v1_g1500"
LARGE_BOT_NAME = "vector_ppo_large_v1_g350k"
SMOKE_CHECKPOINT_PATH = CHECKPOINTS_PATH / SMOKE_BOT_NAME
LARGE_CHECKPOINT_PATH = CHECKPOINTS_PATH / LARGE_BOT_NAME


@dataclass(frozen=True, slots=True)
class _Runtime:
    model: NeuralPolicy
    encoder: NeuralObservationEncoder
    codec: ActionCodec
    device: torch.device


@cache
def _runtime(checkpoint_path: Path) -> _Runtime:
    import torch

    from garboid_pocketrocks.neural.checkpoint import load_inference_checkpoint
    from garboid_pocketrocks.neural.encoding import NeuralObservationEncoder
    from garboid_pocketrocks.training.actions import ActionCodec
    from garboid_pocketrocks.training.bounds import EnvironmentBounds

    torch.set_num_threads(1)
    device = torch.device("cpu")
    loaded = load_inference_checkpoint(checkpoint_path, device=device)
    bounds = EnvironmentBounds(
        loaded.manifest.encoder_config.max_bid,
        loaded.manifest.encoder_config.max_hand_size,
    )
    codec = ActionCodec(bounds)
    return _Runtime(
        model=loaded.model,
        encoder=NeuralObservationEncoder(
            loaded.manifest.encoder_config,
            bounds,
            action_codec=codec,
        ),
        codec=codec,
        device=device,
    )


class _FrozenNeuralBrain:
    checkpoint_path: ClassVar[Path]

    def __init__(self, seed: int | None = None) -> None:
        del seed
        self._runtime = _runtime(self.checkpoint_path)

    def choose_decision(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
    ) -> BotDecision:
        del context, ruleset
        raise RuntimeError("frozen neural policy requires public history")

    def choose_decision_with_history(
        self,
        context: DecisionContext,
        ruleset: RulesetKnowledge,
        history: PublicHistory,
    ) -> BotDecision:
        import torch

        from garboid_pocketrocks.neural.encoding import batch_observations
        from garboid_pocketrocks.neural.policy import evaluate_masked_policy

        observation = self._runtime.encoder.encode(context, ruleset, history)
        batch = batch_observations((observation,), self._runtime.device)
        with torch.inference_mode():
            output = self._runtime.model(batch)
            selection = evaluate_masked_policy(
                output,
                batch,
                generator=None,
                deterministic=True,
            )
        return self._runtime.codec.decode(int(selection.actions[0].item()))


class VectorPpoSmallV1G1500Brain(_FrozenNeuralBrain):
    checkpoint_path = SMOKE_CHECKPOINT_PATH


class VectorPpoLargeV1G350kBrain(_FrozenNeuralBrain):
    checkpoint_path = LARGE_CHECKPOINT_PATH


VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC = BotSpec.for_simulation(
    SMOKE_BOT_NAME,
    VectorPpoSmallV1G1500Brain,
)
VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC = BotSpec.for_simulation(
    LARGE_BOT_NAME,
    VectorPpoLargeV1G350kBrain,
)
```

Import `ClassVar` from `typing`. Keep all Torch imports inside runtime paths so
the general bot registry remains importable without the neural extra.

- [ ] **Step 5: Run the focused neural tests**

Run:

```bash
uv run --extra neural pytest tests/neural/test_smoke_tournament_bot.py -q
uv run ruff check src/garboid_pocketrocks/neural/tournament_bot.py \
  tests/neural/test_smoke_tournament_bot.py
uv run ruff format --check src/garboid_pocketrocks/neural/tournament_bot.py \
  tests/neural/test_smoke_tournament_bot.py
```

Expected: all focused tests and checks pass.

- [ ] **Step 6: Commit the portable model and adapter**

```bash
git add src/garboid_pocketrocks/neural/tournament_bot.py \
  src/garboid_pocketrocks/neural/checkpoints/vector_ppo_large_v1_g350k \
  tests/neural/test_smoke_tournament_bot.py
git commit -m "feat: freeze large 350k neural policy"
```

### Task 2: Register the large bot in the standard tournament

**Files:**
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `tests/bots/test_registry.py`
- Modify: `tests/tournament/test_cli.py`

**Interfaces:**
- Consumes: `VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC` and `VectorPpoLargeV1G350kBrain`.
- Produces: a nine-bot `DEFAULT_TOURNAMENT_BOT_SPECS` and a registry entry selectable through `--bots vector_ppo_large_v1_g350k`.

- [ ] **Step 1: Update expected registry and default-field tests**

Append `"vector_ppo_large_v1_g350k"` to the expected tuples in
`tests/bots/test_registry.py` and to the curated default tuple in
`tests/tournament/test_cli.py`.

In `test_cli_runs_all_conditions_with_current_registry`, exclude both neural
models so this dependency-light CLI integration remains:

```python
"--exclude-bots",
"vector_ppo_small_v1_g1500,vector_ppo_large_v1_g350k",
```

- [ ] **Step 2: Run the registry tests and verify they fail**

Run:

```bash
uv run pytest tests/bots/test_registry.py tests/tournament/test_cli.py -q
```

Expected: tuple comparisons fail because the large bot is not registered.

- [ ] **Step 3: Export and register the large bot**

In `src/garboid_pocketrocks/bots/__init__.py`, import and export
`VectorPpoLargeV1G350kBrain`.

In `src/garboid_pocketrocks/bots/registry.py`, import
`VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC`, append it to `BOT_SPECS`, and append it
to `DEFAULT_TOURNAMENT_BOT_SPECS`:

```python
from garboid_pocketrocks.neural.tournament_bot import (
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
)

BOT_SPECS = (
    BotSpec.from_bot_class(RandomBot),
    AGGRESSIVE_HEURISTIC_BOT_SPEC,
    BALANCED_HEURISTIC_BOT_SPEC,
    PASSIVE_HEURISTIC_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
)

DEFAULT_TOURNAMENT_BOT_SPECS = (
    BOT_SPECS[0],
    AGGRESSIVE_HEURISTIC_V1_BOT_SPEC,
    BALANCED_HEURISTIC_V1_BOT_SPEC,
    PASSIVE_HEURISTIC_V1_BOT_SPEC,
    AGGRESSIVE_HEURISTIC_V2_BOT_SPEC,
    BALANCED_HEURISTIC_V2_BOT_SPEC,
    PASSIVE_HEURISTIC_V2_BOT_SPEC,
    VECTOR_PPO_SMALL_V1_G1500_BOT_SPEC,
    VECTOR_PPO_LARGE_V1_G350K_BOT_SPEC,
)
```

- [ ] **Step 4: Run registry and CLI tests**

Run:

```bash
uv run pytest tests/bots/test_registry.py tests/tournament/test_cli.py -q
uv run ruff check src/garboid_pocketrocks/bots tests/bots/test_registry.py \
  tests/tournament/test_cli.py
uv run ruff format --check src/garboid_pocketrocks/bots \
  tests/bots/test_registry.py tests/tournament/test_cli.py
```

Expected: all tests and checks pass.

- [ ] **Step 5: Commit tournament registration**

```bash
git add src/garboid_pocketrocks/bots/__init__.py \
  src/garboid_pocketrocks/bots/registry.py \
  tests/bots/test_registry.py tests/tournament/test_cli.py
git commit -m "feat: add large neural policy to default tournament"
```

### Task 3: Document and verify the standard field

**Files:**
- Modify: `README.md`
- Modify: `src/garboid_pocketrocks/neural/README.md`

**Interfaces:**
- Consumes: the nine-bot default registry.
- Produces: user-facing instructions naming both frozen neural checkpoints and the required neural extra.

- [ ] **Step 1: Update the tournament documentation**

In `README.md`, change the default-field description to state that it includes
the smoke checkpoint and the 349,860-game large checkpoint under its rounded
`vector_ppo_large_v1_g350k` name.

In `src/garboid_pocketrocks/neural/README.md`, document both immutable
tournament checkpoints, their exact training ages, and the unchanged command:

```bash
uv run --extra neural garboid-tournament
```

- [ ] **Step 2: Run documentation and full static checks**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Run focused and full tests**

Run:

```bash
uv run --extra neural pytest \
  tests/neural/test_smoke_tournament_bot.py \
  tests/bots/test_registry.py \
  tests/tournament/test_cli.py -q
uv run --extra neural pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md src/garboid_pocketrocks/neural/README.md
git commit -m "docs: explain large neural tournament policy"
```

### Task 4: Run and record the fixed-seed tournament

**Files:**
- Create: `docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-tournament.md`
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/ratings.csv`
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/summary.json`
- Create: `docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/report.html`

**Interfaces:**
- Consumes: the standard nine-bot tournament registry and fixed CLI defaults.
- Produces: reproducible leaderboard, runtime, throughput, comparison, and interactive artifacts.

- [ ] **Step 1: Run the full default tournament with wall-time measurement**

Run:

```bash
/usr/bin/time -lp uv run --extra neural garboid-tournament \
  --output-dir docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default
```

Expected: nine leaderboard rows, zero faults for both neural bots, and three
artifact files.

- [ ] **Step 2: Inspect the machine-readable result**

Run:

```bash
jq '{
  configuration: .configuration,
  model: .model,
  leaderboard: [.leaderboard[] | {
    bot_name,
    rank,
    pl_rating,
    games,
    outright_wins,
    mean_final_money,
    faults
  }]
}' docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/summary.json
```

Expected: the configuration matches the global constraints, both named neural
bots appear, and every bot has zero faults.

- [ ] **Step 3: Write the benchmark report from exact artifacts**

Create `docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-tournament.md`
with:

- the complete nine-row leaderboard copied from `ratings.csv`;
- the exact source commit and checkpoint parameter digest;
- wall time and `15_000 / wall_seconds` games per second;
- the 350k-versus-smoke rating, interval, win-rate, and mean-money comparison;
- an explicit statement whether the fixed-seed result supports improved play;
- links to `ratings.csv`, `summary.json`, and `report.html`.

- [ ] **Step 4: Verify benchmark artifacts**

Run:

```bash
test "$(wc -l < docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/ratings.csv)" -eq 10
jq -e '.leaderboard | length == 9' \
  docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/summary.json
jq -e '[.leaderboard[].faults] | add == 0' \
  docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/summary.json
test -s docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default/report.html
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Commit benchmark evidence**

```bash
git add docs/benchmarks/2026-07-30-vector-ppo-large-v1-g350k-tournament.md \
  docs/benchmarks/tournaments/2026-07-30-vector-ppo-large-v1-g350k-default
git commit -m "bench: rank large 350k neural policy"
```

- [ ] **Step 6: Run final branch verification**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run --extra neural pytest -q
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: static checks and the full suite pass, the committed diff has no
whitespace errors, and the worktree is clean.
