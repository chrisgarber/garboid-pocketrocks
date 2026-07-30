# Plackett–Luce Bot Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use braze-superpowers:subagent-driven-development (recommended) or braze-superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 10,000-game multiplayer bot tournament that fits tie-aware Plackett–Luce strengths and writes CSV, JSON, and illustrated HTML reports.

**Architecture:** A tournament scheduler will create explicit `GameJob` values for the existing Monte Carlo engine, balancing condition exposure, unique lineups, pairs, and seats. A separate analysis layer will fit a generalized Plackett–Luce model to whole multiplayer rankings, bootstrap complete games, compute descriptive metrics, and pass a frozen result to dependency-light artifact renderers.

**Tech Stack:** Python 3.14, dataclasses, NumPy, SciPy L-BFGS-B, existing Monte Carlo simulator, pytest, Ruff, mypy.

## Global Constraints

- Default to exactly 10,000 games across charts A–E and player counts 3, 4, and 5.
- Require distinct bot names and IDs and never repeat a bot ID within one lineup.
- Preserve deterministic jobs and results across worker counts for a fixed root seed.
- Model tied final-money ranks with the Davidson–Luce extension; never impose an arbitrary tie order.
- Use ghost pseudo-rankings with weight 0.5 and display ratings as `1500 + 400 * log10(worth / geometric_mean_worth)`.
- Bootstrap complete games, never pairwise fragments; bootstrap failure may remove intervals but must not suppress primary rankings.
- Default tournament fault handling to `record_and_pass`.
- Write only `ratings.csv`, `summary.json`, and `report.html`; overwrite only those known files.
- Follow TDD for every observable behavior.

---

### Task 1: Centralize simulator bot registration

**Files:**
- Create: `src/garboid_pocketrocks/bots/registry.py`
- Modify: `src/garboid_pocketrocks/bots/__init__.py`
- Modify: `src/garboid_pocketrocks/simulator/cli.py`
- Test: `tests/bots/test_registry.py`
- Test: `tests/simulator/test_cli.py`

**Interfaces:**
- Consumes: existing `BotSpec.from_bot_class` and the four live bot wrapper classes.
- Produces: `BOT_SPECS: tuple[BotSpec, ...]`, `BOT_SPECS_BY_NAME: Mapping[str, BotSpec]`, and `registered_bot_specs() -> tuple[BotSpec, ...]`.

- [ ] **Step 1: Write failing registry tests**

```python
def test_registered_bot_specs_have_unique_names_and_ids() -> None:
    specs = registered_bot_specs()
    assert tuple(spec.name for spec in specs) == (
        "random",
        "aggressive",
        "balanced",
        "passive",
    )
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.bot_id for spec in specs}) == len(specs)
    assert BOT_SPECS_BY_NAME == {spec.name: spec for spec in specs}


def test_simulator_cli_uses_shared_registry() -> None:
    assert simulator_cli._BOT_REGISTRY == BOT_SPECS_BY_NAME
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `uv run pytest tests/bots/test_registry.py tests/simulator/test_cli.py -q`

Expected: collection fails because `garboid_pocketrocks.bots.registry` does not exist.

- [ ] **Step 3: Add the registry and derive simulator lookup from it**

```python
# src/garboid_pocketrocks/bots/registry.py
BOT_SPECS = tuple(
    BotSpec.from_bot_class(bot_class)
    for bot_class in (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
    )
)


def _index_specs(specs: tuple[BotSpec, ...]) -> Mapping[str, BotSpec]:
    names = tuple(spec.name for spec in specs)
    ids = tuple(spec.bot_id for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("registered bot names must be unique")
    if len(set(ids)) != len(ids):
        raise ValueError("registered bot IDs must be unique")
    return MappingProxyType({spec.name: spec for spec in specs})


BOT_SPECS_BY_NAME = _index_specs(BOT_SPECS)


def registered_bot_specs() -> tuple[BotSpec, ...]:
    return BOT_SPECS
```

Replace the literal `_BOT_REGISTRY` in `simulator/cli.py` with
`_BOT_REGISTRY = BOT_SPECS_BY_NAME` and export the registry symbols from
`bots/__init__.py`.

- [ ] **Step 4: Run registry and simulator CLI tests**

Run: `uv run pytest tests/bots/test_registry.py tests/simulator/test_cli.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/bots tests/bots/test_registry.py src/garboid_pocketrocks/simulator/cli.py tests/simulator/test_cli.py
git commit -m "refactor: centralize simulator bot registry"
```

### Task 2: Add explicit-job execution to Monte Carlo

**Files:**
- Modify: `src/garboid_pocketrocks/simulator/monte_carlo.py`
- Test: `tests/simulator/test_monte_carlo.py`

**Interfaces:**
- Consumes: `MonteCarloConfig`, `GameJob`, `_execute_job`, and `_aggregate`.
- Produces: `MonteCarloRunner.run_jobs(config, jobs, workers=1) -> MonteCarloResult`; existing `run` delegates to it.

- [ ] **Step 1: Write failing tests for explicit jobs**

```python
def test_run_jobs_executes_a_valid_explicit_plan() -> None:
    config = _small_random_config(games=6)
    jobs = MonteCarloRunner.plan(config)
    assert MonteCarloRunner.run_jobs(config, jobs) == MonteCarloRunner.run(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda jobs: jobs[:-1], "job count"),
        (
            lambda jobs: (replace(jobs[0], game_index=99), *jobs[1:]),
            "game indices",
        ),
        (
            lambda jobs: (replace(jobs[0], root_seed=999), *jobs[1:]),
            "root seed",
        ),
    ),
)
def test_run_jobs_rejects_invalid_explicit_plans(mutate, message: str) -> None:
    config = _small_random_config(games=6)
    with pytest.raises(ValueError, match=message):
        MonteCarloRunner.run_jobs(config, tuple(mutate(MonteCarloRunner.plan(config))))
```

- [ ] **Step 2: Run the focused tests and verify `run_jobs` is absent**

Run: `uv run pytest tests/simulator/test_monte_carlo.py -q`

Expected: tests fail with `AttributeError: type object 'MonteCarloRunner' has no attribute 'run_jobs'`.

- [ ] **Step 3: Implement validation and delegate existing execution**

```python
class MonteCarloRunner:
    @staticmethod
    def run(
        config: MonteCarloConfig,
        *,
        workers: int = 1,
    ) -> MonteCarloResult:
        return MonteCarloRunner.run_jobs(
            config,
            MonteCarloRunner.plan(config),
            workers=workers,
        )

    @staticmethod
    def run_jobs(
        config: MonteCarloConfig,
        jobs: tuple[GameJob, ...],
        *,
        workers: int = 1,
    ) -> MonteCarloResult:
        _validate_jobs(config, jobs)
        completed = _execute_jobs(config, jobs, workers=workers)
        return _aggregate(config, completed)
```

`_validate_jobs` must require `len(jobs) == config.games`, indices exactly
`range(config.games)`, matching root seeds, configured player counts, supported
rulesets, lineup lengths, and bot IDs drawn from `config.bot_specs`.
`_execute_jobs` contains the current serial/process-pool logic.

- [ ] **Step 4: Run Monte Carlo tests**

Run: `uv run pytest tests/simulator/test_monte_carlo.py -q`

Expected: all tests pass, including worker-count equality.

- [ ] **Step 5: Commit**

```bash
git add src/garboid_pocketrocks/simulator/monte_carlo.py tests/simulator/test_monte_carlo.py
git commit -m "feat: execute explicit Monte Carlo job plans"
```

### Task 3: Build the deterministic tournament scheduler

**Files:**
- Create: `src/garboid_pocketrocks/tournament/__init__.py`
- Create: `src/garboid_pocketrocks/tournament/schedule.py`
- Create: `tests/tournament/__init__.py`
- Create: `tests/tournament/helpers.py`
- Create: `tests/tournament/test_schedule.py`

**Interfaces:**
- Consumes: `BotSpec`, `GameJob`, `MonteCarloConfig`, `FaultMode`, `WeightedRulesetSampler`, `derive_seed`, and `live_ruleset`.
- Produces:
  - `TournamentConfig`
  - `ConditionQuota`
  - `PairExposure`
  - `TournamentPlan`
  - `TournamentPlanner.plan(config) -> TournamentPlan`.

- [ ] **Step 1: Add five-bot test helpers and failing configuration tests**

```python
def five_random_specs() -> tuple[BotSpec, ...]:
    return tuple(
        BotSpec(f"random-{index}", f"random-{index}", RandomBot.build_brain) for index in range(5)
    )


def test_default_config_describes_ten_thousand_games_and_all_conditions() -> None:
    config = TournamentConfig(bot_specs=five_random_specs())
    assert config.games == 10_000
    assert config.player_counts == (3, 4, 5)
    assert config.charts == ("A", "B", "C", "D", "E")
    assert config.fault_mode is FaultMode.RECORD_AND_PASS


def test_five_player_tournament_requires_five_distinct_bots() -> None:
    with pytest.raises(ValueError, match="5 distinct"):
        TournamentConfig(bot_specs=five_random_specs()[:4])
```

- [ ] **Step 2: Run the tests and verify the tournament package is missing**

Run: `uv run pytest tests/tournament/test_schedule.py -q`

Expected: collection fails because `garboid_pocketrocks.tournament` does not exist.

- [ ] **Step 3: Implement frozen configuration and quota allocation**

```python
@dataclass(frozen=True, slots=True)
class TournamentConfig:
    bot_specs: tuple[BotSpec, ...]
    games: int = 10_000
    player_counts: tuple[int, ...] = (3, 4, 5)
    charts: tuple[str, ...] = tuple(VALUE_CHARTS)
    root_seed: int = 0
    fault_mode: FaultMode = FaultMode.RECORD_AND_PASS
    bootstrap_samples: int = 200


@dataclass(frozen=True, slots=True)
class ConditionQuota:
    chart: str
    player_count: int
    games: int
```

Validate unique names and IDs, requested charts and counts, enough distinct
bots, `games >= len(charts) * len(player_counts)`, and nonnegative bootstrap
samples. `_allocate_quotas` uses `divmod` over ordered chart/player cells.

- [ ] **Step 4: Add failing schedule determinism and balance tests**

```python
def test_default_plan_allocates_exactly_ten_thousand_games() -> None:
    plan = TournamentPlanner.plan(TournamentConfig(bot_specs=five_random_specs()))
    counts = tuple(quota.games for quota in plan.quotas)
    assert sum(counts) == 10_000
    assert max(counts) - min(counts) == 1
    assert len(plan.jobs) == 10_000


def test_plan_is_seeded_unique_and_balanced() -> None:
    config = TournamentConfig(
        bot_specs=tuple(
            BotSpec(f"bot-{index}", f"bot-{index}", RandomBot.build_brain) for index in range(8)
        ),
        games=150,
        root_seed=42,
    )
    first = TournamentPlanner.plan(config)
    assert first == TournamentPlanner.plan(config)
    assert first != TournamentPlanner.plan(replace(config, root_seed=43))
    assert [job.game_index for job in first.jobs] == list(range(150))
    assert len({job.seed for job in first.jobs}) == 150
    assert all(len({spec.bot_id for spec in job.lineup}) == job.player_count for job in first.jobs)
```

Add assertions that each condition's appearance range is at most one and every
bot's seat-count range is at most one within each player count.

- [ ] **Step 5: Implement greedy lineup selection and seat assignment**

Use count maps keyed by `(chart, player_count, bot_id)`, unordered bot-ID pair,
and `(player_count, bot_id, seat)`. For each lineup slot, choose the candidate
with minimum:

```python
(
    condition_appearances[chart, player_count, bot_id],
    sum(pair_appearances[pair(bot_id, selected_id)] for selected_id in selected),
    global_appearances[bot_id],
    derive_seed(root_seed, stable_namespace, game_index),
)
```

Enumerate lineup permutations and select the seat assignment minimizing the
documented max exposure, squared exposure, and seeded tie-break tuple. Build a
`MonteCarloConfig` using equally weighted selected live rulesets and return:

```python
@dataclass(frozen=True, slots=True)
class TournamentPlan:
    monte_carlo_config: MonteCarloConfig
    jobs: tuple[GameJob, ...]
    quotas: tuple[ConditionQuota, ...]
    pair_exposures: tuple[PairExposure, ...]
```

- [ ] **Step 6: Run scheduler tests**

Run: `uv run pytest tests/tournament/test_schedule.py -q`

Expected: all tests pass in under five seconds, including the 10,000-job plan.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/tournament tests/tournament
git commit -m "feat: plan balanced multiplayer bot tournaments"
```

### Task 4: Fit tie-aware Plackett–Luce strengths

**Files:**
- Modify: `pyproject.toml`
- Modify mechanically: `uv.lock`
- Create: `src/garboid_pocketrocks/tournament/rating.py`
- Modify: `src/garboid_pocketrocks/tournament/__init__.py`
- Create: `tests/tournament/test_rating.py`

**Interfaces:**
- Consumes: `GameSummary` and SciPy `optimize.minimize`.
- Produces:
  - `RankingObservation`
  - `PLBotRating`
  - `PLFitDiagnostics`
  - `PlackettLuceFit`
  - `observations_from_games`
  - `fit_plackett_luce`.

- [ ] **Step 1: Add SciPy dependency and refresh the lock**

Add `"scipy>=1.17"` beside NumPy in project dependencies.

Run: `uv lock`

Expected: `uv.lock` resolves a Python-3.14-compatible SciPy wheel.

- [ ] **Step 2: Write failing observation and rating transform tests**

```python
def test_observations_preserve_multiplayer_ties() -> None:
    game = game_summary(
        ("a", "b", "c", "d"),
        final_money=(30, 20, 20, 10),
        ranks=(1, 2, 2, 4),
    )
    assert observations_from_games((game,)) == (RankingObservation((("a",), ("b", "c"), ("d",))),)


def test_rating_transform_maps_ten_to_one_worth_to_four_hundred_points() -> None:
    ratings = _ratings_from_worths({"strong": 10.0, "weak": 1.0})
    assert ratings["strong"] - ratings["weak"] == pytest.approx(400.0)
    assert statistics.mean(ratings.values()) == pytest.approx(1500.0)
```

- [ ] **Step 3: Run rating tests and verify the module is missing**

Run: `uv run pytest tests/tournament/test_rating.py -q`

Expected: collection fails because `tournament.rating` does not exist.

- [ ] **Step 4: Implement observations and the generalized likelihood**

`RankingObservation.rank_groups` is an ordered tuple of sorted bot-ID tuples and
`weight` defaults to `1.0`. Validate that groups are nonempty and contain no
duplicate IDs.

Implement `_negative_log_likelihood(parameters, problem) -> tuple[float, NDArray]`.
At each rank stage enumerate combinations of remaining bot indexes from order
one through the observed maximum tie order. For subset `C`, compute:

```python
log_choice_weight = log_delta[len(C)] + mean(log_worth[index] for index in C)
```

Use `scipy.special.logsumexp` for the normalizer and accumulate the exact
analytic gradient. Stop before a final singleton stage because its probability
is one. Add weight-0.5 ghost win/loss pair rankings for every real bot.

- [ ] **Step 5: Add failing synthetic fit tests**

```python
def test_consistent_winner_has_highest_finite_rating() -> None:
    observations = (
        RankingObservation((("a",), ("b",), ("c",))),
        RankingObservation((("a",), ("c",), ("b",))),
        RankingObservation((("a",), ("b",), ("c",))),
    )
    fit = fit_plackett_luce(observations, ("a", "b", "c"))
    assert fit.ratings[0].bot_id == "a"
    assert all(math.isfinite(item.rating) and item.worth > 0 for item in fit.ratings)
    assert sum(item.worth for item in fit.ratings) == pytest.approx(1.0)


def test_tie_group_order_does_not_change_fit() -> None:
    first = fit_plackett_luce(
        (RankingObservation((("a", "b"), ("c",))),),
        ("a", "b", "c"),
    )
    second = fit_plackett_luce(
        (RankingObservation((("b", "a"), ("c",))),),
        ("a", "b", "c"),
    )
    assert first == second
    assert first.tie_prevalence[0].order == 2
```

Include undefeated, winless, bot-order permutation, higher-order tie,
non-finite input, and optimizer-failure cases.

- [ ] **Step 6: Implement L-BFGS-B fit and frozen result types**

Call `minimize(..., method="L-BFGS-B", jac=True, options={"maxiter": 1000, "gtol": 1e-8})`.
Bound each fitted `log_delta` to `(-12.0, 12.0)`. Reject unsuccessful or
non-finite results with `TournamentRatingError`.

Normalize real worths to sum one, center real log-worths, transform to PL
ratings, and sort by descending rating then bot ID. Store iterations, objective,
gradient norm, and pseudo-weight in diagnostics.

- [ ] **Step 7: Run rating tests and static checks**

Run: `uv run pytest tests/tournament/test_rating.py -q`

Expected: all tests pass.

Run: `uv run ruff check src/garboid_pocketrocks/tournament/rating.py tests/tournament/test_rating.py`

Expected: exit 0.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/garboid_pocketrocks/tournament tests/tournament/test_rating.py
git commit -m "feat: fit tie-aware Plackett-Luce bot ratings"
```

### Task 5: Compute tournament metrics and bootstrap intervals

**Files:**
- Create: `src/garboid_pocketrocks/tournament/analysis.py`
- Modify: `src/garboid_pocketrocks/tournament/__init__.py`
- Create: `tests/tournament/test_analysis.py`

**Interfaces:**
- Consumes: `MonteCarloResult`, `PlackettLuceFit`, and `fit_plackett_luce`.
- Produces:
  - `RatingInterval`
  - `TournamentBotRow`
  - `CalibrationBin`
  - `BootstrapSummary`
  - `TournamentAnalysis`
  - `analyze_tournament`
  - `bootstrap_rating_intervals`.

- [ ] **Step 1: Write failing descriptive-metric tests**

```python
def test_analysis_computes_normalized_finish_and_winning_money() -> None:
    result = monte_carlo_result(
        game_summary(("a", "b", "c"), final_money=(30, 20, 10), ranks=(1, 2, 3)),
        game_summary(("b", "a", "c"), final_money=(25, 25, 5), ranks=(1, 1, 3)),
    )
    analysis = analyze_tournament(result, fit_for(("a", "b", "c")))
    row = analysis.rows_by_id["a"]
    assert row.games == 2
    assert row.mean_normalized_finish == pytest.approx(0.75)
    assert row.mean_winning_money == pytest.approx(27.5)
```

Add tests for no wins (`None`), tied pair outcome 0.5, calibration bin totals,
condition grouping, and fault counts.

- [ ] **Step 2: Run tests and verify `analysis` is missing**

Run: `uv run pytest tests/tournament/test_analysis.py -q`

Expected: collection fails because `tournament.analysis` does not exist.

- [ ] **Step 3: Implement frozen rows and calibration**

Normalized finish is `(player_count - rank) / (player_count - 1)`. Include tied
firsts in winning-money samples. Calibration emits one oriented record for
every unordered pair, choosing the lexicographically first ID as the oriented
subject, with observed outcome 1, 0, or 0.5. Predicted probability is:

```python
worth_a / (worth_a + worth_b)
```

Bin by predicted probability in ten fixed `[0.0, 0.1), ..., [0.9, 1.0]`
intervals and store count, mean prediction, and mean observed result.

- [ ] **Step 4: Write failing deterministic bootstrap tests**

```python
def test_bootstrap_is_deterministic_and_resamples_whole_games() -> None:
    first = bootstrap_rating_intervals(games, bot_ids, samples=20, root_seed=42)
    second = bootstrap_rating_intervals(games, bot_ids, samples=20, root_seed=42)
    assert first == second
    assert first.requested == 20
    assert first.converged == 20
    assert tuple(interval.bot_id for interval in first.intervals) == bot_ids


def test_bootstrap_zero_samples_returns_no_intervals() -> None:
    assert bootstrap_rating_intervals(games, bot_ids, samples=0, root_seed=42).intervals == ()
```

- [ ] **Step 5: Implement bootstrap with convergence threshold**

For replicate index `i`, use `random.Random(derive_seed(root_seed, "bootstrap", i))`
to choose `len(games)` complete summaries with replacement, reconstruct
observations, and fit. Compute linear-interpolated 2.5% and 97.5% empirical
quantiles. If fewer than 90% converge, return no intervals and a warning rather
than raising.

- [ ] **Step 6: Run analysis tests**

Run: `uv run pytest tests/tournament/test_analysis.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/tournament tests/tournament/test_analysis.py
git commit -m "feat: analyze and bootstrap tournament rankings"
```

### Task 6: Render stable CSV, JSON, and HTML artifacts

**Files:**
- Create: `src/garboid_pocketrocks/tournament/reporting.py`
- Modify: `src/garboid_pocketrocks/tournament/__init__.py`
- Create: `tests/tournament/test_reporting.py`

**Interfaces:**
- Consumes: `TournamentConfig`, `TournamentPlan`, `TournamentAnalysis`, `PlackettLuceFit`, and `BootstrapSummary`.
- Produces: `TournamentArtifacts` and `write_tournament_artifacts(...) -> TournamentArtifacts`.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_artifacts_contain_stable_machine_data_and_three_svg_charts(tmp_path: Path) -> None:
    artifacts = write_tournament_artifacts(
        output_dir=tmp_path,
        overwrite=False,
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )
    assert artifacts.ratings_csv == tmp_path / "ratings.csv"
    payload = json.loads(artifacts.summary_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["configuration"]["games"] == config.games
    assert payload["leaderboard"][0]["pl_rating"] == fit.ratings[0].rating
    html = artifacts.report_html.read_text()
    assert html.count("<svg") == 3
    assert "PL rating leaderboard" in html
    assert "Rating versus mean winning money" in html
    assert "PL calibration" in html
```

Add tests for HTML escaping, `None` rendering as `n/a`, CSV ordering, warnings,
nonempty-directory refusal, and overwrite preserving unrelated files.

- [ ] **Step 2: Run tests and verify the reporting module is missing**

Run: `uv run pytest tests/tournament/test_reporting.py -q`

Expected: collection fails because `tournament.reporting` does not exist.

- [ ] **Step 3: Implement JSON and CSV renderers**

Use `dataclasses.asdict`, explicit schema/configuration dictionaries, stable
key ordering, and `csv.DictWriter`. Use `repr`-equivalent float serialization
without presentation rounding. Include pair exposure summary, tie parameters,
optimizer diagnostics, bootstrap metadata, leaderboard, condition statistics,
calibration, warnings, and artifact basenames.

- [ ] **Step 4: Implement accessible self-contained HTML and SVG**

Build escaped HTML with standard-library helpers. SVG 1 is a horizontal rating
dot/interval plot; SVG 2 is a labeled rating-versus-winning-money scatter; SVG
3 is a calibration plot with diagonal, binned points, and counts. Each SVG has
`role="img"`, `aria-labelledby`, a `<title>`, and `<desc>`. CSS must be embedded,
responsive, and print-friendly; no remote assets or scripts.

- [ ] **Step 5: Implement safe artifact replacement**

Refuse a nonempty output directory unless overwrite is true. On overwrite,
touch only the three known names. Render all contents first, write each to a
same-directory temporary file, `flush`, `fsync`, and `os.replace` it into place.
Clean up surviving temporary files on error.

- [ ] **Step 6: Run reporting tests**

Run: `uv run pytest tests/tournament/test_reporting.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/garboid_pocketrocks/tournament tests/tournament/test_reporting.py
git commit -m "feat: render tournament ranking reports"
```

### Task 7: Add the tournament service and CLI

**Files:**
- Create: `src/garboid_pocketrocks/tournament/runner.py`
- Create: `src/garboid_pocketrocks/tournament/cli.py`
- Modify: `src/garboid_pocketrocks/tournament/__init__.py`
- Modify: `pyproject.toml`
- Create: `tests/tournament/test_runner.py`
- Create: `tests/tournament/test_cli.py`

**Interfaces:**
- Consumes: registry, scheduler, `MonteCarloRunner.run_jobs`, rating, analysis,
  bootstrap, and reporting.
- Produces:
  - `TournamentRun`
  - `TournamentRunner.run(config, workers, output_dir, overwrite) -> TournamentRun`
  - `garboid-tournament` command.

- [ ] **Step 1: Write a failing small end-to-end runner test**

```python
def test_runner_is_identical_across_worker_counts(tmp_path: Path) -> None:
    config = TournamentConfig(
        bot_specs=five_random_specs(),
        games=15,
        bootstrap_samples=0,
        root_seed=42,
    )
    serial = TournamentRunner.run(
        config,
        workers=1,
        output_dir=tmp_path / "serial",
    )
    parallel = TournamentRunner.run(
        config,
        workers=2,
        output_dir=tmp_path / "parallel",
    )
    assert serial.plan == parallel.plan
    assert serial.monte_carlo_result == parallel.monte_carlo_result
    assert serial.fit == parallel.fit
    assert serial.analysis == parallel.analysis
    assert serial.artifacts.ratings_csv.read_text() == parallel.artifacts.ratings_csv.read_text()
```

- [ ] **Step 2: Run the test and verify the runner is missing**

Run: `uv run pytest tests/tournament/test_runner.py -q`

Expected: collection fails because `tournament.runner` does not exist.

- [ ] **Step 3: Implement the orchestration service**

`TournamentRunner.run` must plan, execute jobs, convert summaries to
observations, fit primary ratings, compute analysis, bootstrap, merge intervals
into report rows, write artifacts, and return all frozen intermediate results.
No CLI formatting logic belongs in the service.

- [ ] **Step 4: Write failing CLI parser and subprocess tests**

```python
def test_parser_defaults_to_full_tournament() -> None:
    args = _parser().parse_args(())
    assert args.games == 10_000
    assert args.players == (3, 4, 5)
    assert args.charts == ("A", "B", "C", "D", "E")
    assert args.bootstrap_samples == 200


def test_cli_reports_current_five_bot_preflight_error() -> None:
    completed = subprocess.run(
        ["uv", "run", "garboid-tournament", "--output-dir", "ignored"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "5 distinct bots" in completed.stderr
```

Also test `--bots`, `--exclude-bots`, comma parsers, and a programmatic
15-game five-bot CLI invocation by monkeypatching the registry.

- [ ] **Step 5: Implement CLI and entry point**

Add:

```toml
garboid-tournament = "garboid_pocketrocks.tournament.cli:main"
```

The CLI builds `TournamentConfig`, validates include/exclude names, calls the
runner, and prints columns `rank`, `bot`, `PL rating`, `95% interval`, `games`,
`win rate`, `mean money`, and `faults`. Convert domain errors to
`parser.error(...)`.

- [ ] **Step 6: Run tournament integration tests**

Run: `uv run pytest tests/tournament/test_runner.py tests/tournament/test_cli.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/garboid_pocketrocks/tournament tests/tournament
git commit -m "feat: add Plackett-Luce tournament command"
```

### Task 8: Document usage and verify the complete feature

**Files:**
- Modify: `README.md`
- Test: all tests and project quality checks.

**Interfaces:**
- Consumes: completed command and artifact schema.
- Produces: documented local workflow and fresh verification evidence.

- [ ] **Step 1: Update README with exact commands and interpretation**

Document the default full command, the current four-bot preflight behavior,
the temporary `--players 3,4` command, automatic inclusion of v2 registered
bots, PL worth versus 1500-centered display rating, output files, deterministic
seed/worker behavior, and the warning that one global worth averages over
charts, player counts, and opponent mixtures.

- [ ] **Step 2: Run focused tournament tests**

Run: `uv run pytest tests/tournament tests/simulator/test_monte_carlo.py tests/simulator/test_cli.py tests/bots/test_registry.py -q`

Expected: all focused tests pass.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Run lint**

Run: `uv run ruff check .`

Expected: exit 0 with no findings.

- [ ] **Step 5: Run formatting verification**

Run: `uv run ruff format --check .`

Expected: exit 0 with no files requiring formatting.

- [ ] **Step 6: Run strict type checking**

Run: `uv run mypy`

Expected: exit 0 with no errors.

- [ ] **Step 7: Verify the current-registry preflight and a programmatic 15-game tournament**

Run: `uv run garboid-tournament --output-dir /tmp/garboid-tournament-current`

Expected: exit 2 with a clear `5 distinct bots` message.

Run a test-backed five-spec tournament through `TournamentRunner` with 15
games, zero bootstrap samples, and a fresh temporary output directory.

Expected: 15 game summaries, all 15 condition cells, finite ratings, and three
nonempty artifacts containing three SVG charts.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: explain multiplayer bot tournaments"
```
