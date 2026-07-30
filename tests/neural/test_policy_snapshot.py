from __future__ import annotations

import io
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from garboid_pocketrocks.neural.config import (  # noqa: E402
    training_encoder_config,
    training_model_config,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.policy_snapshot import (  # noqa: E402
    load_policy_snapshots,
    snapshot_policies,
)


def _model(seed: int) -> NeuralPolicy:
    torch.manual_seed(seed)
    return NeuralPolicy(
        training_encoder_config(),
        training_model_config("small"),
    )


def test_policy_snapshots_are_sorted_eager_and_cpu_loadable() -> None:
    alpha = _model(31)
    zeta = _model(37)
    expected = {
        "alpha": {name: tensor.detach().clone() for name, tensor in alpha.state_dict().items()},
        "zeta": {name: tensor.detach().clone() for name, tensor in zeta.state_dict().items()},
    }

    snapshots = snapshot_policies({"zeta": zeta, "alpha": alpha})
    with torch.no_grad():
        for model in (alpha, zeta):
            for parameter in model.parameters():
                parameter.add_(1.0)
    loaded = load_policy_snapshots(snapshots)

    assert tuple(snapshot.identity for snapshot in snapshots) == ("alpha", "zeta")
    assert tuple(loaded) == ("alpha", "zeta")
    for identity, model in loaded.items():
        assert model.encoder_config == training_encoder_config()
        assert model.model_config == training_model_config("small")
        for name, tensor in model.state_dict().items():
            assert tensor.device == torch.device("cpu")
            assert torch.equal(tensor, expected[identity][name])


def test_policy_snapshot_loading_is_strict() -> None:
    snapshot = snapshot_policies({"current": _model(41)})[0]
    buffer = io.BytesIO()
    torch.save({}, buffer)

    with pytest.raises(RuntimeError):
        load_policy_snapshots((replace(snapshot, state_bytes=buffer.getvalue()),))
