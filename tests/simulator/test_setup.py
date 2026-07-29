from collections import Counter

from garboid_pocketrocks.rules import LIVE_RULESET
from garboid_pocketrocks.simulator.model import Phase
from garboid_pocketrocks.simulator.setup import build_setup


def test_setup_is_reproducible_and_conserves_cards() -> None:
    left = build_setup(LIVE_RULESET, player_count=3, seed=123)
    right = build_setup(LIVE_RULESET, player_count=3, seed=123)

    assert left == right
    assert left.state.phase is Phase.BIDDING
    all_cards = [
        *left.state.resource_deck,
        *left.state.visible_resources,
        *(card for player in left.state.players for card in player.private_hand),
    ]
    assert len(all_cards) == 30
    assert len({card.card_id for card in all_cards}) == 30
    assert set(Counter(card.suit for card in all_cards).values()) == {6}


def test_live_setup_counts_by_player_count() -> None:
    for players, cash, hand_size in ((3, 30, 5), (4, 25, 4), (5, 20, 3)):
        setup = build_setup(LIVE_RULESET, player_count=players, seed=7)
        assert len(setup.state.players) == players
        assert {player.cash for player in setup.state.players} == {cash}
        assert {len(player.private_hand) for player in setup.state.players} == {
            hand_size
        }
        assert len(setup.state.visible_resources) == 2
        assert len(setup.state.active_objective_ids) == 4
        assert len(setup.state.action_deck) == 29
