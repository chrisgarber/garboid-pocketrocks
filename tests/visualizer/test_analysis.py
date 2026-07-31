from __future__ import annotations

import json
from pathlib import Path

import pytest

from garboid_pocketrocks.visualizer.analysis import (
    BotInsightsEngine,
    TournamentInsightsEngine,
)
from garboid_pocketrocks.visualizer.loading import TournamentDataset


def _summary() -> dict[str, object]:
    rows = [
        {
            "rank": rank,
            "bot_id": bot,
            "bot_name": bot,
            "pl_rating": rating,
            "rating_interval_lower": rating - 10,
            "rating_interval_upper": rating + 10,
            "games": 1,
            "outright_wins": int(bot == "alpha"),
            "first_place_ties": 0,
            "mean_normalized_finish": float(bot == "alpha"),
            "mean_final_money": 30 if bot == "alpha" else 20,
            "mean_winning_money": 30 if bot == "alpha" else None,
            "faults": 0,
            "worth": 0.5,
            "log_worth": 0.0,
        }
        for rank, (bot, rating) in enumerate((("alpha", 1600), ("beta", 1400)), start=1)
    ]
    return {
        "schema_version": 1,
        "configuration": {
            "games": 1,
            "bots": [{"bot_id": row["bot_id"], "name": row["bot_name"]} for row in rows],
            "charts": ["A"],
        },
        "leaderboard": rows,
        "condition_statistics": [
            {
                "chart": "A",
                "player_count": 2,
                "bot_id": row["bot_id"],
                "games": 1,
                "outright_wins": row["outright_wins"],
                "first_place_ties": 0,
                "mean_normalized_finish": row["mean_normalized_finish"],
                "mean_final_money": row["mean_final_money"],
            }
            for row in rows
        ],
        "calibration": [],
        "pair_outcomes": 1,
    }


def _dataset(tmp_path: Path) -> TournamentDataset:
    traces = tmp_path / "decision-traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "bot_id": "alpha",
                "seat": 0,
                "turn_index": 0,
                "context": {
                    "decision_kind": "submitBid",
                    "cash_by_seat": [0, 10],
                    "legal_max_amount": 0,
                    "current_action_id": 1,
                },
                "selected_action": {"action_kind": "pass", "value": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    games = (
        {
            "game_index": 0,
            "bot_ids": ["alpha", "beta"],
            "scores": [
                {"seat": 0, "rank": 1, "final_money": 30},
                {"seat": 1, "rank": 2, "final_money": 20},
            ],
        },
    )
    details = (
        {
            "game_index": 0,
            "bot_ids": ["alpha", "beta"],
            "value_chart": [1, 2, 3, 4, 5],
            "turns": [
                {
                    "action": "Auction1",
                    "winner_seat": 0,
                    "paid": 1,
                    "bundle_suits": [3],
                    "claimed_objective_ids": [1],
                },
                {
                    "action": "Loan10",
                    "winner_seat": 0,
                    "paid": 2,
                    "bundle_suits": [],
                    "claimed_objective_ids": [],
                },
            ],
            "scores": [
                {
                    "seat": 0,
                    "cash": 4,
                    "items_value": 3,
                    "objectives_value": 10,
                    "investments_value": 5,
                    "loans_value": -10,
                    "total": 30,
                },
                {
                    "seat": 1,
                    "cash": 2,
                    "items_value": 2,
                    "objectives_value": 0,
                    "investments_value": 0,
                    "loans_value": 0,
                    "total": 20,
                },
            ],
        },
    )
    return TournamentDataset(tmp_path, _summary(), games, details, traces)


def test_tournament_engine_builds_directed_matchups(tmp_path: Path) -> None:
    report = TournamentInsightsEngine(_dataset(tmp_path)).build()

    alpha = next(row for row in report["matchups"] if row["bot_id"] == "alpha")
    beta = next(row for row in report["matchups"] if row["bot_id"] == "beta")

    assert alpha["score"] == 1.0
    assert beta["score"] == 0.0
    assert report["availability"] == {
        "game_summaries": True,
        "game_details": True,
        "decision_traces": True,
    }


def test_bot_engine_explains_objectives_profit_loans_and_cash(tmp_path: Path) -> None:
    alpha = BotInsightsEngine(_dataset(tmp_path)).build("alpha")

    assert alpha["opponents"][0]["score"] == 1.0
    assert alpha["objectives"][0]["game_rate"] == 1.0
    assert alpha["objectives"][0]["objectives_per_100_games"] == 100
    assert alpha["auction_profit"][0]["mean"] == 2
    assert alpha["loans"][0]["mean"] == 2
    assert alpha["loans"][0]["mean_upfront_liquidity"] == 8
    assert alpha["cash_by_phase"][0]["cash_zero_rate"] == 1.0
    assert alpha["cash_by_phase"][0]["hard_constrained_rate"] == 1.0


def test_unknown_bot_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown tournament bot"):
        BotInsightsEngine(_dataset(tmp_path)).build("missing")
