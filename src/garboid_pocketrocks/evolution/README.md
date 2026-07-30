# Heuristic evolution

`garboid-evolve-heuristic` executes a versioned heuristic search manifest on
the fixed development corpus. Development results are tuning evidence, not a
promotion decision. A positive, complete, fault-free winner is only frozen as
an input to the separate held-out promotion gate.

Run a search from the repository root:

```bash
uv run garboid-evolve-heuristic \
  --manifest configs/evolution/balanced-v3-search-v1.json \
  --output-dir evolution-results/balanced-v3-search-v1
```

The manifest owns the search seed, generations, population and elite counts,
mutation rules, coefficient grids, initial profile, predecessor, and exact
development-corpus binding. The CLI exposes only execution controls:
`--development-corpus`, `--workers`, `--batch-size`, `--output-dir`, and
`--overwrite`. There is no resume mode, held-out input, or promotion switch.
Rerun the same immutable manifest from the beginning to reproduce a search.

## Exit codes

- `0`: the complete search found a positive development improvement and wrote
  `frozen-candidate.json` for later held-out evaluation.
- `1`: the complete search found no positive improvement, so nothing was
  frozen.
- `2`: invocation, operational, or evidence validation failed. A completed run
  with invalid evidence still writes its failure report.

## Evidence

Every successful artifact transaction writes:

- `search-manifest.json`
- `search-report.json`
- `candidate-evaluations.jsonl`
- `selection-log.jsonl`
- `development-games.jsonl`
- `development-corpus-snapshot.json`

`frozen-candidate.json` is present only for a frozen improvement. It records
the exact coefficients, repository commit, search and corpus digests, scores,
and hashes of its source report and evaluation log.

Artifacts are deterministic, finite JSON with terminal newlines. By default a
nonempty output directory is rejected. `--overwrite` replaces the known
artifact generation transactionally, preserves unrelated files, and removes a
stale `frozen-candidate.json` if the replacement run does not freeze a winner.
If replacement fails, the prior known artifact generation is restored.

The frozen candidate must still pass the separate
[promotion gate](../promotion/README.md) on held-out games before it can become
a released bot identity.
