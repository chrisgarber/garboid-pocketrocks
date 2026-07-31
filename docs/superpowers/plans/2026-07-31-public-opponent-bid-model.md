# Public opponent-bid model implementation plan

## Task 1: Deliver public history to live brains

- Add a single synchronous history-aware dispatch point in `bots/base.py`.
- Override the SDK raw-decision callback and parse `common_events` with the
  existing strict public-history adapter.
- Preserve the ordinary brain path byte-for-byte in behavior.
- Test exact history delivery, ordinary fallback, raw-capability reporting,
  malformed-frame failure, and hidden-field non-access.

## Task 2: Share the public turn phase

- Move the existing Issue #10 turn thresholds into a small public helper.
- Make diagnostic analysis call the shared helper.
- Pin turns 1/5/6/12/13 and invalid negative indexes in tests.
- Verify this refactor does not change existing reports.

## Task 3: Build the pure opponent model

- Add `heuristics/opponent_bids.py` with closed immutable input/output types.
- Pair only completed public turns and resolutions.
- Implement the deterministic public reference prior and sparse-history
  threshold.
- Build one smoothed discrete distribution per opponent.
- Compute exact public-tiebreak-aware winning probabilities for every legal
  bid under the documented independence approximation.
- Test PMF normalization, all input effects, sparse prior, history weighting,
  loan-expanded legal support, support clipping, deterministic bytes, zero-bid
  wins, tie order, malformed history, and forbidden-field independence.

## Task 4: Add an explicit balanced-v5 candidate

- Layer the opponent forecast over the unchanged balanced-v3 valuator.
- Keep reveal behavior unchanged.
- Route ordinary, history-aware, and explained calls through one internal
  choice so tracing cannot change behavior.
- Add a local-only candidate spec; do not register a live bot or latest alias.
- Test expected-surplus arithmetic for bid zero and positive bids, lower-bid tie resolution,
  legal-action safety, live/scalar/batch parity, and v1-v3 immutability.

## Task 5: Extend typed decision reports

- Add a closed opponent-aware explanation schema.
- Serialize and parse per-opponent distributions and every legal bid forecast.
- Validate probability normalization, action agreement, finite values, and
  selected-bid agreement.
- Update Markdown/JSON reporting with plain-English labels.
- Keep published benchmark diagnostics cohort-aggregated and fail closed below
  30 distinct games.

## Task 6: Select and freeze on development data

- Define a small, fixed grid for prior strength, minimum history, and history
  weights before evaluating it.
- Evaluate only the existing development corpus with fixed seeds and canonical
  opponents.
- Select by the existing deterministic development ranking.
- Freeze the balanced-v5 candidate, model configuration, v3 profile binding,
  corpus digest, report digests, repository commit, and safe aggregate
  diagnostics.
- Do not read or run the held-out corpus in this task.

## Task 7: Run the behavior-change safety matrix

- Run charts A-E for three, four, and five players.
- Require zero illegal actions, simulator faults, or model fallbacks caused by
  valid public inputs.
- Verify scalar, batch, and worker-count determinism.
- Record a privacy-safe aggregate summary.

## Task 8: Run the one-shot held-out gate

- Verify the candidate and all development evidence from disk before launch.
- Compare the exact frozen candidate with canonical balanced-v3 using the
  unchanged held-out paired corpus and bootstrap settings.
- Run diagnostics off and write the result once.
- If the interval includes or crosses zero, stop with v3 still released.

## Task 9: Release only after a pass

- On a strict held-out pass, add the explicit balanced-v5 released identity,
  preserve v1-v3, and advance only the balanced latest/unversioned wrapper.
- On failure, add no released identity and move no alias.
- Run the full default and neural test matrices, ruff, formatting, mypy,
  lockfile verification, CLI smoke checks, and independent review.
- Push stacked draft PRs with plain-English outcomes and links to Issue #13.
