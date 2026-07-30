"""Frozen neural policy used by the standard local tournament."""

from __future__ import annotations

from pathlib import Path

SMOKE_BOT_NAME = "vector_ppo_small_v1_g1500"
SMOKE_CHECKPOINT_PATH = Path(__file__).with_name("checkpoints") / SMOKE_BOT_NAME
