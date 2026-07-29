from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping

import pytest

from garboid_pocketrocks.bots import (
    AggressiveHeuristicBot,
    BalancedHeuristicBot,
    PassiveHeuristicBot,
    RandomBot,
)
from garboid_pocketrocks.bots.launcher import (
    BOT_REGISTRY,
    BotFactory,
    BotRuntimeStopped,
    _bot_names,
    _parser,
    main,
    run_bots,
)


def test_registry_contains_every_live_wrapper_in_stable_order() -> None:
    assert tuple(BOT_REGISTRY) == ("random", "aggressive", "balanced", "passive")
    assert tuple(BOT_REGISTRY.values()) == (
        RandomBot,
        AggressiveHeuristicBot,
        BalancedHeuristicBot,
        PassiveHeuristicBot,
    )


def test_parser_selects_all_registered_bots_by_default() -> None:
    assert _parser().parse_args([]).bots == tuple(BOT_REGISTRY)


def test_bot_names_accepts_a_trimmed_subset_in_requested_order() -> None:
    assert _bot_names(" passive, aggressive ") == ("passive", "aggressive")


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("", "at least one bot name is required"),
        ("  ", "at least one bot name is required"),
        ("random,random", "duplicate bot name"),
        ("random,missing", "unknown bot name"),
    ),
)
def test_bot_names_rejects_invalid_selections(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        _bot_names(value)


class _ControlledBot:
    def __init__(
        self,
        *,
        started: asyncio.Event,
        cancelled: asyncio.Event,
        release: asyncio.Event,
        returns: bool,
    ) -> None:
        self._started = started
        self._cancelled = cancelled
        self._release = release
        self._returns = returns

    async def run_async(self) -> None:
        self._started.set()
        try:
            await self._release.wait()
            if not self._returns:
                await asyncio.Future[None]()
        except asyncio.CancelledError:
            self._cancelled.set()
            raise


def _factory(
    *,
    started: asyncio.Event,
    cancelled: asyncio.Event,
    release: asyncio.Event,
    returns: bool = False,
    constructions: list[str] | None = None,
    name: str = "",
) -> BotFactory:
    def build() -> _ControlledBot:
        if constructions is not None:
            constructions.append(name)
        return _ControlledBot(
            started=started,
            cancelled=cancelled,
            release=release,
            returns=returns,
        )

    return build


def test_run_bots_constructs_once_starts_concurrently_and_cancels_together() -> None:
    async def exercise() -> None:
        started = (asyncio.Event(), asyncio.Event())
        cancelled = (asyncio.Event(), asyncio.Event())
        releases = (asyncio.Event(), asyncio.Event())
        constructions: list[str] = []
        registry: Mapping[str, BotFactory] = {
            "first": _factory(
                started=started[0],
                cancelled=cancelled[0],
                release=releases[0],
                constructions=constructions,
                name="first",
            ),
            "second": _factory(
                started=started[1],
                cancelled=cancelled[1],
                release=releases[1],
                constructions=constructions,
                name="second",
            ),
        }

        launcher = asyncio.create_task(run_bots(("first", "second"), registry=registry))
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=1,
        )
        assert constructions == ["first", "second"]

        launcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launcher
        assert all(event.is_set() for event in cancelled)

    asyncio.run(exercise())


def test_runtime_return_cancels_siblings_and_identifies_stopped_bot() -> None:
    async def exercise() -> None:
        returning_started = asyncio.Event()
        sibling_started = asyncio.Event()
        returning_release = asyncio.Event()
        sibling_release = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        registry: Mapping[str, BotFactory] = {
            "returning": _factory(
                started=returning_started,
                cancelled=asyncio.Event(),
                release=returning_release,
                returns=True,
            ),
            "sibling": _factory(
                started=sibling_started,
                cancelled=sibling_cancelled,
                release=sibling_release,
            ),
        }

        launcher = asyncio.create_task(
            run_bots(("returning", "sibling"), registry=registry)
        )
        await asyncio.wait_for(
            asyncio.gather(returning_started.wait(), sibling_started.wait()),
            timeout=1,
        )
        returning_release.set()

        with pytest.raises(BotRuntimeStopped, match="returning") as raised:
            await launcher
        assert raised.value.bot_name == "returning"
        assert sibling_cancelled.is_set()

    asyncio.run(exercise())


def test_invalid_cli_input_does_not_construct_bots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    def fail_if_constructed() -> _ControlledBot:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid CLI input must not construct bots")

    monkeypatch.setitem(BOT_REGISTRY, "random", fail_if_constructed)

    with pytest.raises(SystemExit) as raised:
        main(["--bots", "missing"])

    assert raised.value.code == 2
    assert not constructed
