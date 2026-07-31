# Fixed-bid search reproduction

Run these commands from this directory. The drivers preserve the exact
candidate manifest, rosters, seeds, field cap, and metrics used by the dated
benchmark report.

```bash
cd scripts/analysis/fixed_bid_search

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python -u _drive_fixed_search.py \
  --games 600 --seed 24701 --workers 8 \
  --output /tmp/fixed-screen-results.json

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python -u _refine_fixed_search.py \
  --games 1800 --workers 8 \
  --output /tmp/fixed-refine-results.json

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python _run_fixed_finalists.py \
  --games 7500 --seed 8675309 --workers 8

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python _run_fixed_finalists.py \
  --games 7500 --seed 13579 --workers 8
```

The screen and refinement use the same `fixed-search` identity for every
challenger so the schedule does not change with its values. Each field has the
original and one challenger among ten identities (20% fixed value). The final
driver pins the 11 identities that formed the curated field at selection time,
then adds the two finalists (3/13, or 23.1%, fixed value).
