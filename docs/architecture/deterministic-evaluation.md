# Deterministic evaluation

## Decision

Simulation and evaluation derive all engine, lineup, seat, brain, sampling,
bootstrap, and worker seeds from an explicit root seed. Results must not
depend on scalar versus batch execution, batch packing, worker completion
order, or worker count.

Replays, game summaries, ratings, and reports use stable ordering and
serialization. Where byte identity is part of the current contract, tests
compare bytes rather than only parsed values.

Development and held-out evaluation are separate:

- development seeds and short runs may guide debugging and calibration;
- strategy, coefficients, model, checkpoint, roster, and evaluation
  configuration are frozen before a held-out run;
- held-out results are reported with the repository state, root seed, and
  complete configuration and are not reused for tuning.

## Consequences

- Scalar execution is the behavioral oracle for SDK batch execution.
- Parallel evaluation aggregates results in planned game order.
- Tournament confidence intervals resample complete games, preserving their
  multiplayer and tie structure.
- A training smoke proves mechanics and reproducibility, not playing strength.
- A changed generation receives a new fixed-seed comparison rather than a
  rewritten historical benchmark.
