"""Refit the opponent-bid prior in ``heuristics/bid_priors.py``.

Captures decision traces over the registered field and prints the quantile table as
Python source, ready to paste into a **new** named prior. Released priors are
immutable, like any other coefficient set.

    PYTHONPATH=src uv run --extra neural python scripts/analysis/fit_bid_prior.py
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from garboid_pocketrocks.bots.registry import DEFAULT_TOURNAMENT_BOT_SPECS
from garboid_pocketrocks.heuristics.bid_priors import (
    ACTION_CLASSES,
    PHASES,
    action_class,
    phase_for,
    prior_key,
)
from garboid_pocketrocks.simulator import MonteCarloConfig, MonteCarloRunner

GAMES = 900
ROOT_SEED = 31337
BUCKETS = 25
WORKERS = 10


def quantiles(values: Sequence[float], buckets: int) -> tuple[float, ...]:
    if not values:
        return ()
    probabilities = np.linspace(0.0, 1.0, buckets)
    return tuple(float(q) for q in np.quantile(np.asarray(values), probabilities))


def collect(traces: Iterable[Any]) -> tuple[dict[str, list[float]], list[float]]:
    """Bid-as-fraction-of-cash samples, bucketed by action class and phase."""

    buckets: dict[str, list[float]] = {}
    everything: list[float] = []
    for trace in traces:
        context = trace.context
        if context.decision_kind != "submitBid":
            continue
        name = action_class(context.current_action_id)
        if name is None:
            continue
        cash = context.cash_by_seat[context.bot_seat]
        if cash <= 0:
            continue
        action = trace.selected_action
        amount = 0 if action.action_kind == "pass" else int(action.value or 0)
        fraction = min(1.5, amount / cash)
        buckets.setdefault(prior_key(name, phase_for(trace.turn_index)), []).append(fraction)
        everything.append(fraction)
    return buckets, everything


def render(values: Sequence[float]) -> str:
    body = ", ".join(f"{value:g}" for value in values)
    return "\n".join("        " + line for line in textwrap.wrap(body, 84))


def main() -> None:
    config = MonteCarloConfig(
        bot_specs=DEFAULT_TOURNAMENT_BOT_SPECS,
        games=GAMES,
        player_counts=(3, 4, 5),
        value_charts=("A", "B", "C", "D", "E"),
        objectives_enabled=(True,),
        root_seed=ROOT_SEED,
        capture_decision_traces=True,
    )
    result = MonteCarloRunner.run(config, workers=WORKERS)
    buckets, everything = collect(result.decision_traces)
    print(f"# fitted from {len(everything)} bidding decisions over {GAMES} games")
    print("_SAMPLES: dict[str, tuple[float, ...]] = {")
    for name in ACTION_CLASSES:
        for phase in PHASES:
            key = prior_key(name, phase)
            values = buckets.get(key, [])
            if values:
                print(f'    "{key}": (')
                print(render(quantiles(values, BUCKETS)))
                print("    ),")
    print("}")
    print("\n_FALLBACK: tuple[float, ...] = (")
    print(render(quantiles(everything, BUCKETS)))
    print(")")


if __name__ == "__main__":
    main()
