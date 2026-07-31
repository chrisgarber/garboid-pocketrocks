# Fixed objective-overlay reproduction

Run these commands from this directory. The support module freezes all three
development configurations, including the selected private-heavy policy.

```bash
cd scripts/analysis/fixed_objective_overlay_search

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python run_overlay_experiment.py \
  --phase screen --games 6000 --seed 2026073101 --workers 8 \
  --output /tmp/overlay-screen-2026073101

UV_CACHE_DIR=/tmp/garboid-uv-cache uv run python run_overlay_experiment.py \
  --phase validate --games 15000 --seed 2026073102 --workers 8 \
  --output /tmp/overlay-validation-2026073102
```

The 16-bot development field counts all four fixed-family policies for a
conservative 25% share. The validation field contains the original fixed bot
and the frozen overlay among ten identities (20%). Both commands use all five
charts, all three supported player counts, and 200 bootstrap samples.
