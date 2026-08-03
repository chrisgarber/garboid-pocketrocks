#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repository_root"

training_root="artifacts/training/cold-mixed-mps-10h-v2"
tournament_root="artifacts/tournaments/cold-mixed-mps-10h-v2"

uv run --extra neural garboid-train train \
  --config configs/neural/cold-mixed-mps-3h-stage1-v2.json \
  --output-dir "$training_root/stage1-75"

uv run --extra neural python scripts/training/evaluate_neural_checkpoint.py \
  --checkpoint "$training_root/stage1-75/checkpoints/latest" \
  --inference-checkpoint "$training_root/stage1-75/inference" \
  --output-dir "$tournament_root/stage1-75" \
  --games 300 \
  --bootstrap-samples 0 \
  --seed 2026080211

uv run --extra neural garboid-train resume \
  --checkpoint "$training_root/stage1-75/checkpoints/latest" \
  --config configs/neural/cold-mixed-mps-4h-stage2-v2.json \
  --output-dir "$training_root/stage2-50"

uv run --extra neural python scripts/training/evaluate_neural_checkpoint.py \
  --checkpoint "$training_root/stage2-50/checkpoints/latest" \
  --inference-checkpoint "$training_root/stage2-50/inference" \
  --output-dir "$tournament_root/stage2-50" \
  --games 300 \
  --bootstrap-samples 0 \
  --seed 2026080212

uv run --extra neural garboid-train resume \
  --checkpoint "$training_root/stage2-50/checkpoints/latest" \
  --config configs/neural/cold-mixed-mps-3h-stage3-v2.json \
  --output-dir "$training_root/stage3-25"

uv run --extra neural python scripts/training/evaluate_neural_checkpoint.py \
  --checkpoint "$training_root/stage3-25/checkpoints/latest" \
  --inference-checkpoint "$training_root/stage3-25/inference" \
  --bot-name vector_ppo_large_v2_g1750k \
  --output-dir "$tournament_root/final" \
  --games 1500 \
  --bootstrap-samples 100 \
  --seed 2026080213
