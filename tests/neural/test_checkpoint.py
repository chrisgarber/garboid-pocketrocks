from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from pocketrocks import DecisionContext

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402
from torch import device as TorchDevice  # noqa: E402

from garboid_pocketrocks.adapters.public_history import (  # noqa: E402
    PublicAuctionResolved,
    PublicEventKind,
    PublicGameSetup,
    PublicHistory,
    PublicInformationRevealed,
    PublicTurnOpened,
)
from garboid_pocketrocks.knowledge import canonical_knowledge  # noqa: E402
from garboid_pocketrocks.neural.checkpoint import (  # noqa: E402
    CheckpointError,
    InferenceManifest,
    load_inference_checkpoint,
    parameter_digest,
    save_inference_checkpoint,
)
from garboid_pocketrocks.neural.config import (  # noqa: E402
    stage1_encoder_config,
    stage1_model_config,
)
from garboid_pocketrocks.neural.encoding import (  # noqa: E402
    NeuralBatch,
    NeuralObservationEncoder,
    batch_observations,
)
from garboid_pocketrocks.neural.model import NeuralPolicy  # noqa: E402
from garboid_pocketrocks.neural.policy import evaluate_masked_policy  # noqa: E402
from garboid_pocketrocks.training.bounds import EnvironmentBounds  # noqa: E402


def _model() -> NeuralPolicy:
    torch.manual_seed(71)
    model = NeuralPolicy(stage1_encoder_config(), stage1_model_config())
    model.eval()
    return model


def _manifest(model: NeuralPolicy) -> InferenceManifest:
    return InferenceManifest.create(
        model,
        repository_commit="0123456789abcdef",
        root_seed=42,
        completed_episodes=32,
        completed_updates=2,
    )


def _context() -> DecisionContext:
    return DecisionContext(
        request_id="checkpoint-fixture",
        deadline_at=0,
        received_at=0,
        decision_kind="submitBid",
        player_count=3,
        starting_cash=30,
        value_chart=canonical_knowledge(3).value_chart,
        objective_ids=(1, 2, 3, 4),
        current_action_id=1,
        current_resource_ids=(1, 0),
        cash_by_seat=(30, 27, 25),
        tiebreak_seat=2,
        won_resource_counts_by_seat=((1, 0, 0, 0, 0), (0, 2, 0, 0, 0), (0, 0, 3, 0, 0)),
        revealed_info_counts_by_seat=((0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)),
        owned_objective_ids_by_seat=((1,), (2,), (3,)),
        bot_seat=1,
        current_hand_suit_ids=(2, 5),
        legal_max_amount=7,
        revealable_count=2,
    )


def _history() -> PublicHistory:
    return (
        PublicGameSetup(
            kind=PublicEventKind.GAME_SETUP,
            player_count=3,
            starting_cash=30,
            value_chart=canonical_knowledge(3).value_chart,
            initial_tiebreak_seat=0,
            objective_ids=(1, 2, 3, 4),
        ),
        PublicTurnOpened(
            kind=PublicEventKind.TURN_OPENED,
            action_id=1,
            resource_ids=(1, 0),
        ),
        PublicAuctionResolved(
            kind=PublicEventKind.AUCTION_RESOLVED,
            bids_by_seat=(2, 5, 1),
        ),
        PublicInformationRevealed(
            kind=PublicEventKind.INFORMATION_REVEALED,
            seat=1,
            suit_id=3,
        ),
    )


def _fixture_batch() -> NeuralBatch:
    config = stage1_encoder_config()
    bounds = EnvironmentBounds(config.max_bid, config.max_hand_size)
    observation = NeuralObservationEncoder(config, bounds).encode(
        _context(),
        canonical_knowledge(3),
        _history(),
    )
    return batch_observations((observation,), torch.device("cpu"))


def _payload(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    )


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    (path / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state(path: Path) -> dict[str, Tensor]:
    loaded: object = torch.load(
        path / "model.pt",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    assert isinstance(loaded, dict)
    return cast(dict[str, Tensor], loaded)


def _rewrite_state(path: Path, state: dict[str, Tensor]) -> None:
    torch.save(state, path / "model.pt")
    payload = _payload(path)
    payload["model_sha256"] = _sha256(path / "model.pt")
    payload["parameter_digest"] = parameter_digest(state)
    _write_payload(path, payload)


def test_checkpoint_contains_only_sorted_manifest_and_model_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()

    save_inference_checkpoint(checkpoint, model, _manifest(model))

    assert {item.name for item in checkpoint.iterdir()} == {
        "manifest.json",
        "model.pt",
    }
    manifest_text = (checkpoint / "manifest.json").read_text(encoding="utf-8")
    payload = json.loads(manifest_text)
    assert manifest_text == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert set(payload) == {
        "schema_version",
        "encoder_schema_version",
        "repository_commit",
        "python_version",
        "torch_version",
        "numpy_version",
        "sdk_version",
        "encoder_config",
        "model_config",
        "action_space_size",
        "action_space_hash",
        "supported_ruleset_names",
        "supported_player_counts",
        "root_seed",
        "completed_episodes",
        "completed_updates",
        "model_sha256",
        "parameter_digest",
    }
    assert payload["schema_version"] == 1
    assert payload["encoder_schema_version"] == 1
    assert payload["repository_commit"] == "0123456789abcdef"
    assert all(
        payload[field]
        for field in ("python_version", "torch_version", "numpy_version", "sdk_version")
    )
    assert payload["encoder_config"] == json.loads(
        json.dumps(asdict(_manifest(model).encoder_config))
    )
    assert payload["action_space_size"] == 106
    assert len(payload["action_space_hash"]) == 64
    assert payload["supported_ruleset_names"] == ["live-A"]
    assert payload["supported_player_counts"] == [3]
    assert payload["root_seed"] == 42
    assert payload["completed_episodes"] == 32
    assert payload["completed_updates"] == 2
    assert payload["model_sha256"] == _sha256(checkpoint / "model.pt")
    assert payload["parameter_digest"] == parameter_digest(model.state_dict())

    state = _load_state(checkpoint)
    assert set(state) == set(model.state_dict())
    assert all(isinstance(value, Tensor) for value in state.values())


def test_reload_replays_fixture_bit_exactly_and_uses_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    batch = _fixture_batch()
    before_output = model(batch)
    before_selection = evaluate_masked_policy(
        before_output,
        batch,
        generator=None,
        deterministic=True,
    )
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    original_load = torch.load
    weights_only_calls: list[bool] = []

    def spy(
        path: str | Path,
        *,
        map_location: TorchDevice,
        weights_only: bool,
    ) -> object:
        weights_only_calls.append(weights_only)
        return cast(
            object,
            original_load(
                path,
                map_location=map_location,
                weights_only=weights_only,
            ),
        )

    monkeypatch.setattr(torch, "load", spy)
    loaded = load_inference_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    after_output = loaded.model(batch)
    after_selection = evaluate_masked_policy(
        after_output,
        batch,
        generator=None,
        deterministic=True,
    )

    assert weights_only_calls == [True]
    assert not loaded.model.training
    assert loaded.manifest.model_sha256 == _sha256(checkpoint / "model.pt")
    assert parameter_digest(loaded.model.state_dict()) == loaded.manifest.parameter_digest
    assert torch.equal(before_output.bid_logits, after_output.bid_logits)
    assert torch.equal(before_output.reveal_logits, after_output.reveal_logits)
    assert torch.equal(before_output.value, after_output.value)
    assert torch.equal(before_selection.actions, after_selection.actions)


def test_parameter_digest_is_order_independent_and_value_sensitive() -> None:
    first = torch.tensor((1.0, 2.0))
    second = torch.tensor((3.0,))
    forward = {"first": first, "second": second}
    reverse = {"second": second, "first": first}
    changed = {"first": first, "second": torch.tensor((4.0,))}

    assert parameter_digest(forward) == parameter_digest(reverse)
    assert parameter_digest(forward) != parameter_digest(changed)


def test_nonempty_checkpoint_destination_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "existing.txt").write_text("keep", encoding="utf-8")
    model = _model()

    with pytest.raises(CheckpointError, match="empty"):
        save_inference_checkpoint(checkpoint, model, _manifest(model))

    assert (checkpoint / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_save_rejects_noncanonical_model_dtype(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model().double()

    with pytest.raises(CheckpointError, match="dtype"):
        save_inference_checkpoint(checkpoint, model, _manifest(model))


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    model_path = checkpoint / "model.pt"
    model_path.write_bytes(model_path.read_bytes() + b"corrupt")

    with pytest.raises(CheckpointError, match="checksum"):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        pytest.param(
            lambda payload: payload.update(schema_version=2),
            "schema",
            id="checkpoint-schema",
        ),
        pytest.param(
            lambda payload: payload.update(encoder_schema_version=2),
            "encoder schema",
            id="encoder-schema",
        ),
        pytest.param(
            lambda payload: payload.update(action_space_hash="0" * 64),
            "action",
            id="action-hash",
        ),
        pytest.param(
            lambda payload: payload.update(supported_player_counts=[4]),
            "support",
            id="ruleset-support",
        ),
    ),
)
def test_manifest_compatibility_mismatches_are_rejected(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    payload = _payload(checkpoint)
    mutate(payload)
    _write_payload(checkpoint, payload)

    with pytest.raises(CheckpointError, match=message):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))


def test_nonfinite_model_weight_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    state = _load_state(checkpoint)
    name = next(name for name, value in state.items() if value.is_floating_point())
    damaged = state[name].clone()
    damaged.reshape(-1)[0] = torch.nan
    state[name] = damaged
    _rewrite_state(checkpoint, state)

    with pytest.raises(CheckpointError, match="nonfinite"):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize("change", ("dtype", "shape"))
def test_noncanonical_parameter_schema_is_rejected_before_state_loading(
    tmp_path: Path,
    change: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    state = _load_state(checkpoint)
    name = next(name for name, value in state.items() if value.ndim >= 2)
    if change == "dtype":
        state[name] = state[name].double()
    else:
        state[name] = state[name].reshape(-1)
    _rewrite_state(checkpoint, state)

    with pytest.raises(CheckpointError, match=change):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize("change", ("missing", "unexpected"))
def test_changed_parameter_names_are_rejected(
    tmp_path: Path,
    change: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    state = _load_state(checkpoint)
    if change == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.weight"] = torch.zeros(1)
    _rewrite_state(checkpoint, state)

    with pytest.raises(CheckpointError, match="parameter names"):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize("config_name", ("encoder_config", "model_config"))
@pytest.mark.parametrize("change", ("missing", "unknown"))
def test_nested_config_keys_must_match_schema_exactly(
    tmp_path: Path,
    config_name: str,
    change: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = _model()
    save_inference_checkpoint(checkpoint, model, _manifest(model))
    payload = _payload(checkpoint)
    nested = cast(dict[str, Any], payload[config_name])
    if change == "missing":
        nested.pop(next(iter(nested)))
    else:
        nested["unknown_field"] = 1
    _write_payload(checkpoint, payload)

    with pytest.raises(CheckpointError):
        load_inference_checkpoint(checkpoint, device=torch.device("cpu"))
