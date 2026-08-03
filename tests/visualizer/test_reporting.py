from __future__ import annotations

from garboid_pocketrocks.visualizer.reporting import render_insights_html


def test_report_is_self_contained_and_escapes_script_terminators() -> None:
    html = render_insights_html(
        {
            "configuration": {"games": 0, "bots": [], "charts": []},
            "leaderboard": [],
            "matchups": [],
            "conditions": [],
            "calibration": [],
            "pair_outcomes": 0,
            "availability": {
                "game_summaries": False,
                "game_details": False,
                "decision_traces": False,
            },
        },
        {"bot": {"bot_name": "</script><script>alert(1)</script>"}},
    )

    assert html.startswith("<!doctype html>")
    assert "https://" not in html
    assert "<\\/script><script>alert(1)<\\/script>" in html
    assert html.count("</script>") == 2
    assert "Profit on won resource auctions" in html
    assert "Winning prices for investments" in html
    assert "Net profit is always $5 for Invest $5 and $10 for Invest $10." in html
