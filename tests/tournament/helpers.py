from __future__ import annotations

from garboid_pocketrocks.bots import BotSpec, RandomBot


def random_specs(count: int = 5) -> tuple[BotSpec, ...]:
    return tuple(
        BotSpec(
            name=f"random-{index}",
            bot_id=f"random-{index}",
            brain_factory=RandomBot.build_brain,
        )
        for index in range(count)
    )
