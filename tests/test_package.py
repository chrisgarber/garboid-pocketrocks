from importlib import import_module

import pytest

from garboid_pocketrocks import __version__, bots

NAMESPACES = (
    "garboid_pocketrocks.adapters",
    "garboid_pocketrocks.bots",
    "garboid_pocketrocks.simulator",
    "garboid_pocketrocks.training",
)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_planned_namespaces_are_importable() -> None:
    for namespace in NAMESPACES:
        assert import_module(namespace).__name__ == namespace


def test_bots_namespace_does_not_reexport_prebuilt_specs() -> None:
    assert not {name for name in vars(bots) if name.endswith("_BOT_SPEC")}


@pytest.mark.parametrize(
    "module_name",
    (
        "garboid_pocketrocks.rules",
        "garboid_pocketrocks.simulator.context",
        "garboid_pocketrocks.simulator.engine",
        "garboid_pocketrocks.simulator.events",
        "garboid_pocketrocks.simulator.model",
        "garboid_pocketrocks.simulator.sampling",
        "garboid_pocketrocks.simulator.setup",
    ),
)
def test_project_game_engine_modules_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        import_module(module_name)


def test_public_simulator_exports_only_the_sdk_engine_boundary() -> None:
    simulator = import_module("garboid_pocketrocks.simulator")

    assert hasattr(simulator, "SdkGameSession")
    assert not hasattr(simulator, "GameEngine")
    assert not hasattr(simulator, "RulesetVariationSampler")


if __name__ == "__main__":
    test_package_version()
    test_planned_namespaces_are_importable()
