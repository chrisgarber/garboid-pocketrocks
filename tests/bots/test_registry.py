from __future__ import annotations

from garboid_pocketrocks.bots.registry import (
    BOT_SPECS_BY_NAME,
    registered_bot_specs,
)


def test_registered_bot_specs_have_unique_names_and_ids() -> None:
    specs = registered_bot_specs()

    assert tuple(spec.name for spec in specs) == (
        "random",
        "aggressive",
        "balanced",
        "passive",
    )
    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.bot_id for spec in specs}) == len(specs)
    assert BOT_SPECS_BY_NAME == {spec.name: spec for spec in specs}
