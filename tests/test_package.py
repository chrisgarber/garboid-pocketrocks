from importlib import import_module

from garboid_pocketrocks import __version__

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


if __name__ == "__main__":
    test_package_version()
    test_planned_namespaces_are_importable()
