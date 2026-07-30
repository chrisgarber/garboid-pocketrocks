# Tournament Defaults and Bootstrap Scaling Design

## Goal

Make the bot comparator's default run representative, fast, and sustainable as
new bot versions are added. The default tournament will run 15,000 games,
retain the benchmark-selected batch size of 64, avoid duplicate public aliases,
and include `random` as a fixed baseline.

## Default bot roster

The general bot registry continues to contain every bot identity needed by
simulation and live integrations. A separate curated
`DEFAULT_TOURNAMENT_BOT_SPECS` tuple defines the comparator's default roster:

- `random`;
- every explicitly versioned simulation bot, currently the v1 and v2
  aggressive, balanced, and passive bots;
- no unsuffixed public bot aliases.

The comparator uses this tuple only when `--bots` is omitted. An explicit
`--bots` argument may still select any registered identity, including an
unsuffixed public alias.

The curated tuple is preferable to dynamic brain deduplication. Equivalent
brains can be exposed through different subclasses or factories, so runtime
fingerprinting would be brittle. Adding a version to the registry and the
default tournament tuple is an explicit, reviewable decision.

Tests will assert the exact current default roster and verify that unsuffixed
heuristic aliases are absent. Existing registry uniqueness checks remain in
place.

## Execution defaults

- Games: 15,000.
- Batch size: 64.
- Conditions: player counts 3, 4, and 5 across charts A-E.
- Bootstrap samples: retain 200 until the scaling benchmark determines whether
  a different value offers a better stability/cost tradeoff.

The CLI help, `TournamentConfig`, tests, and generated configuration metadata
must agree on these defaults.

## Bootstrap scaling experiment

Generate one deterministic 15,000-game result with root seed 42 and reuse its
ranking observations for every measurement. This separates bootstrap cost from
simulation cost and guarantees that differences between measurements come only
from resampling.

Measure:

1. schedule planning and batch simulation time;
2. primary Plackett-Luce fit time;
3. bootstrap time for 50, 100, 200, and 500 samples;
4. convergence count;
5. interval endpoints and widths for every bot.

Bootstrap samples use the existing deterministic derived seeds, so the smaller
measurements are prefixes of the larger experiment. Compare the 50-, 100-, and
200-sample intervals with the 500-sample reference. Select the smallest sample
count that preserves the substantive ranking conclusions and has acceptably
stable interval endpoints. Record timings and findings in a benchmark document.

## Expected scaling

Batch simulation should scale approximately linearly with game count once
worker utilization is saturated. Bootstrap work should scale approximately
with both game count and bootstrap sample count, although compiled choice-set
aggregation reduces the constant factor. The benchmark will report measured
rather than assumed proportions.

More games reduce sampling uncertainty from game outcomes at roughly the
inverse square root of the number of appearances. Increasing from 10,000 to
15,000 games should therefore narrow game-sampling error by about 18% in the
ideal independent-sample approximation. It is unlikely to reverse conclusions
where intervals are widely separated, but it can reorder bots whose intervals
overlap substantially.

## Verification and delivery

Use test-driven development for default roster and configuration changes:

1. add tests for the 15,000-game default and curated roster;
2. observe the expected failures;
3. implement the smallest registry and CLI changes;
4. run focused tests, the full suite, Ruff, mypy, and `git diff --check`.

After the benchmark and final 15,000-game report are complete, merge the
feature branch into local `main` without overwriting the existing uncommitted
`tests/neural/test_rollout.py` edit, then push `main` to `origin`.
