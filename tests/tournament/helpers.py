from __future__ import annotations

from garboid_pocketrocks.bots import BotSpec, RandomBot
from garboid_pocketrocks.simulator.monte_carlo import GameSummary
from garboid_pocketrocks.simulator.session import SessionScore


def random_specs(count: int = 5) -> tuple[BotSpec, ...]:
    return tuple(
        BotSpec(
            name=f"random-{index}",
            bot_id=f"random-{index}",
            brain_factory=RandomBot.build_brain,
        )
        for index in range(count)
    )


def game_summary(
    bot_ids: tuple[str, ...],
    *,
    final_money: tuple[int, ...],
    ranks: tuple[int, ...],
    game_index: int = 0,
    ruleset_name: str = "live-A",
) -> GameSummary:
    return GameSummary(
        game_index=game_index,
        root_seed=42,
        seed=game_index + 100,
        player_count=len(bot_ids),
        ruleset_name=ruleset_name,
        bot_names=bot_ids,
        bot_ids=bot_ids,
        scores=tuple(
            SessionScore(seat=seat, final_money=money, rank=rank)
            for seat, (money, rank) in enumerate(zip(final_money, ranks, strict=True))
        ),
        decision_counts=(0,) * len(bot_ids),
        fault_counts=(0,) * len(bot_ids),
    )
