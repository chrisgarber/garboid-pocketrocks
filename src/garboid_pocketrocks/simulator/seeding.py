from __future__ import annotations

import hashlib


def derive_seed(root_seed: int, namespace: str, index: int) -> int:
    """Derive a stable independent seed for one simulation concern."""

    payload = f"{root_seed}:{namespace}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
