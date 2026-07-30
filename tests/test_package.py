from importlib import import_module
from importlib.resources import files

from garboid_pocketrocks import __version__, bots

NAMESPACES = (
    "garboid_pocketrocks.adapters",
    "garboid_pocketrocks.bots",
    "garboid_pocketrocks.bots.llm",
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


def test_llm_prompt_skill_is_packaged_as_a_separate_resource() -> None:
    skill = files("garboid_pocketrocks.bots.llm").joinpath(
        "skills",
        "pocketrocks",
        "SKILL.md",
    )

    assert skill.is_file()
    assert skill.read_text(encoding="utf-8").startswith("# PocketRocks decision skill")


if __name__ == "__main__":
    test_package_version()
    test_planned_namespaces_are_importable()
