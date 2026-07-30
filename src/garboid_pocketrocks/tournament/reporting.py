from __future__ import annotations

import csv
import html
import io
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from garboid_pocketrocks.tournament.analysis import (
    BootstrapSummary,
    TournamentAnalysis,
)
from garboid_pocketrocks.tournament.rating import PlackettLuceFit
from garboid_pocketrocks.tournament.schedule import (
    TournamentConfig,
    TournamentPlan,
)

_ARTIFACT_NAMES = ("ratings.csv", "summary.json", "report.html")


@dataclass(frozen=True, slots=True)
class TournamentArtifacts:
    ratings_csv: Path
    summary_json: Path
    report_html: Path


def write_tournament_artifacts(
    *,
    output_dir: Path,
    overwrite: bool,
    config: TournamentConfig,
    plan: TournamentPlan,
    fit: PlackettLuceFit,
    analysis: TournamentAnalysis,
    bootstrap: BootstrapSummary,
) -> TournamentArtifacts:
    validate_artifact_output_dir(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _summary_payload(
        config=config,
        plan=plan,
        fit=fit,
        analysis=analysis,
        bootstrap=bootstrap,
    )
    contents = {
        "ratings.csv": _render_csv(analysis, bootstrap),
        "summary.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
        "report.html": _render_html(config, fit, analysis, bootstrap),
    }
    for name, content in contents.items():
        _atomic_write(output_dir / name, content)
    return TournamentArtifacts(
        ratings_csv=output_dir / "ratings.csv",
        summary_json=output_dir / "summary.json",
        report_html=output_dir / "report.html",
    )


def validate_artifact_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _summary_payload(
    *,
    config: TournamentConfig,
    plan: TournamentPlan,
    fit: PlackettLuceFit,
    analysis: TournamentAnalysis,
    bootstrap: BootstrapSummary,
) -> dict[str, Any]:
    intervals = {interval.bot_id: interval for interval in bootstrap.intervals}
    pair_counts = tuple(exposure.games for exposure in plan.pair_exposures)
    leaderboard = []
    for row in analysis.rows:
        item = asdict(row)
        interval = intervals.get(row.bot_id)
        item["rating_interval_lower"] = interval.lower if interval is not None else None
        item["rating_interval_upper"] = interval.upper if interval is not None else None
        leaderboard.append(item)
    return {
        "schema_version": 1,
        "configuration": {
            "bots": [{"name": spec.name, "bot_id": spec.bot_id} for spec in config.bot_specs],
            "games": config.games,
            "player_counts": list(config.player_counts),
            "charts": list(config.charts),
            "root_seed": config.root_seed,
            "fault_mode": config.fault_mode.value,
            "batch_size": config.batch_size,
            "bootstrap_samples": config.bootstrap_samples,
        },
        "schedule": {
            "condition_quotas": [asdict(quota) for quota in plan.quotas],
            "pair_exposure": {
                "minimum": min(pair_counts, default=0),
                "median": float(statistics.median(pair_counts)) if pair_counts else 0.0,
                "maximum": max(pair_counts, default=0),
            },
        },
        "model": {
            "tie_prevalence": [asdict(item) for item in fit.tie_prevalence],
            "diagnostics": asdict(fit.diagnostics),
        },
        "bootstrap": asdict(bootstrap),
        "leaderboard": leaderboard,
        "condition_statistics": [asdict(item) for item in analysis.condition_statistics],
        "calibration": [asdict(item) for item in analysis.calibration],
        "pair_outcomes": analysis.pair_outcomes,
        "warnings": list(bootstrap.warnings),
        "artifacts": {
            "ratings_csv": "ratings.csv",
            "summary_json": "summary.json",
            "report_html": "report.html",
        },
    }


def _render_csv(
    analysis: TournamentAnalysis,
    bootstrap: BootstrapSummary,
) -> str:
    intervals = {interval.bot_id: interval for interval in bootstrap.intervals}
    fieldnames = (
        "rank",
        "bot_name",
        "bot_id",
        "worth",
        "log_worth",
        "pl_rating",
        "rating_interval_lower",
        "rating_interval_upper",
        "games",
        "outright_wins",
        "first_place_ties",
        "mean_normalized_finish",
        "mean_final_money",
        "mean_winning_money",
        "faults",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in analysis.rows:
        interval = intervals.get(row.bot_id)
        writer.writerow(
            {
                **asdict(row),
                "rating_interval_lower": interval.lower if interval else "",
                "rating_interval_upper": interval.upper if interval else "",
                "mean_winning_money": (
                    row.mean_winning_money if row.mean_winning_money is not None else ""
                ),
            }
        )
    return stream.getvalue()


def _render_html(
    config: TournamentConfig,
    fit: PlackettLuceFit,
    analysis: TournamentAnalysis,
    bootstrap: BootstrapSummary,
) -> str:
    del fit
    intervals = {interval.bot_id: interval for interval in bootstrap.intervals}
    warning_html = "".join(
        f'<p class="warning">{html.escape(warning)}</p>' for warning in bootstrap.warnings
    )
    table_rows = []
    for row in analysis.rows:
        interval = intervals.get(row.bot_id)
        interval_text = (
            f"{interval.lower:.1f}–{interval.upper:.1f}" if interval is not None else "n/a"
        )
        winning_money = (
            f"{row.mean_winning_money:.2f}" if row.mean_winning_money is not None else "n/a"
        )
        table_rows.append(
            "<tr>"
            f"<td>{row.rank}</td>"
            f'<th scope="row">{html.escape(row.bot_name)}</th>'
            f"<td>{row.pl_rating:.2f}</td>"
            f"<td>{interval_text}</td>"
            f"<td>{row.worth:.8g}</td>"
            f"<td>{row.games}</td>"
            f"<td>{row.outright_wins}</td>"
            f"<td>{row.first_place_ties}</td>"
            f"<td>{row.mean_normalized_finish:.3f}</td>"
            f"<td>{row.mean_final_money:.2f}</td>"
            f"<td>{winning_money}</td>"
            f"<td>{row.faults}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Garboid PocketRocks tournament</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0 auto; max-width: 1100px; padding: 1rem; }}
h1, h2 {{ line-height: 1.2; }}
.meta {{ color: #666; }}
.warning {{ border-left: .3rem solid #b45309; padding: .5rem .75rem; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid currentColor; padding: .4rem; text-align: right; }}
th[scope="row"] {{ text-align: left; }}
svg {{ display: block; height: auto; margin: 1rem 0 2rem; max-width: 100%; }}
.grid {{ stroke: currentColor; stroke-opacity: .18; }}
.mark {{ fill: #2563eb; stroke: #1e3a8a; }}
.interval {{ stroke: #2563eb; stroke-width: 2; }}
.axis {{ stroke: currentColor; stroke-width: 1; }}
.axis-label {{ fill: currentColor; font-size: 12px; }}
.label {{ fill: currentColor; font-size: 12px; }}
@media (prefers-color-scheme: dark) {{
  .meta {{ color: #aaa; }}
  .mark {{ fill: #60a5fa; stroke: #bfdbfe; }}
  .interval {{ stroke: #60a5fa; }}
}}
</style>
</head>
<body>
<h1>Garboid PocketRocks tournament</h1>
<p class="meta">{config.games:,} games · charts {", ".join(config.charts)} ·
players {", ".join(map(str, config.player_counts))} · seed {config.root_seed}</p>
{warning_html}
<div class="table-wrap">
<table>
<thead><tr><th>Rank</th><th>Bot</th><th>PL rating</th><th>95% interval</th>
<th>Worth</th><th>Games</th><th>Wins</th><th>Tied first</th>
<th>Mean finish</th><th>Mean money</th><th>Winning money</th><th>Faults</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody>
</table>
</div>
<h2>PL rating leaderboard</h2>
{_rating_svg(analysis, bootstrap)}
<h2>Rating versus mean winning money</h2>
{_money_svg(analysis)}
<h2>PL calibration</h2>
{_calibration_svg(analysis)}
</body>
</html>
"""


def _rating_svg(
    analysis: TournamentAnalysis,
    bootstrap: BootstrapSummary,
) -> str:
    width = 800
    height = 60 + 38 * len(analysis.rows)
    intervals = {interval.bot_id: interval for interval in bootstrap.intervals}
    values = [
        value
        for row in analysis.rows
        for value in (
            intervals[row.bot_id].lower if row.bot_id in intervals else row.pl_rating,
            intervals[row.bot_id].upper if row.bot_id in intervals else row.pl_rating,
        )
    ]
    low, high = _domain(values)
    marks = []
    for index, row in enumerate(analysis.rows):
        y = 35 + index * 38
        interval = intervals.get(row.bot_id)
        left = _scale(interval.lower if interval else row.pl_rating, low, high, 190, 760)
        right = _scale(interval.upper if interval else row.pl_rating, low, high, 190, 760)
        x = _scale(row.pl_rating, low, high, 190, 760)
        marks.append(
            f'<text class="label" x="5" y="{y + 4}">{html.escape(row.bot_name)}</text>'
            f'<line class="interval" x1="{left:.2f}" x2="{right:.2f}" y1="{y}" y2="{y}"/>'
            f'<circle class="mark" cx="{x:.2f}" cy="{y}" r="5"/>'
            f'<text class="label" x="{min(x + 8, 765):.2f}" y="{y + 4}">'
            f"{row.pl_rating:.0f}</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="rating-title rating-desc">'
        '<title id="rating-title">PL rating leaderboard</title>'
        '<desc id="rating-desc">Bot ratings with optional 95 percent bootstrap intervals.</desc>'
        + "".join(marks)
        + f'<line class="axis" x1="190" x2="760" y1="{height - 24}" y2="{height - 24}"/>'
        + f'<text x="475" y="{height - 5}" text-anchor="middle" class="axis-label">'
        "PL rating</text>" + "</svg>"
    )


def _money_svg(analysis: TournamentAnalysis) -> str:
    width, height = 800, 430
    rows = tuple(row for row in analysis.rows if row.mean_winning_money is not None)
    if not rows:
        marks = '<text class="label" x="20" y="40">No winning-money samples</text>'
    else:
        x_low, x_high = _domain([row.pl_rating for row in rows])
        money_values = [
            row.mean_winning_money for row in rows if row.mean_winning_money is not None
        ]
        y_low, y_high = _domain(money_values)
        mark_items: list[str] = []
        for row in rows:
            winning_money = row.mean_winning_money
            assert winning_money is not None
            mark_items.append(
                f'<circle class="mark" cx="{_scale(row.pl_rating, x_low, x_high, 70, 750):.2f}" '
                f'cy="{_scale(winning_money, y_low, y_high, 370, 35):.2f}" r="5"/>'
                f'<text class="label" '
                f'x="{_scale(row.pl_rating, x_low, x_high, 70, 750) + 8:.2f}" '
                f'y="{_scale(winning_money, y_low, y_high, 370, 35) + 4:.2f}">'
                f"{html.escape(row.bot_name)}</text>"
            )
        marks = "".join(mark_items)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="money-title money-desc">'
        '<title id="money-title">Rating versus mean winning money</title>'
        '<desc id="money-desc">Scatter plot comparing PL rating with final money '
        "in first-place finishes.</desc>"
        '<line class="axis" x1="70" x2="750" y1="370" y2="370"/>'
        '<line class="axis" x1="70" x2="70" y1="35" y2="370"/>'
        '<text x="410" y="410" text-anchor="middle" class="axis-label">PL rating</text>'
        '<text x="18" y="202" text-anchor="middle" transform="rotate(-90 18 202)" '
        'class="axis-label">Mean winning final money</text>' + marks + "</svg>"
    )


def _calibration_svg(analysis: TournamentAnalysis) -> str:
    width, height = 800, 430
    diagonal = '<line class="grid" x1="70" y1="370" x2="750" y2="35"/>'
    marks = "".join(
        (
            f'<circle class="mark" cx="{_scale(bucket.mean_prediction, 0, 1, 70, 750):.2f}" '
            f'cy="{_scale(bucket.observed_score, 0, 1, 370, 35):.2f}" '
            f'r="{min(12, 4 + math.sqrt(bucket.count)):.2f}"/>'
            f'<text class="label" '
            f'x="{_scale(bucket.mean_prediction, 0, 1, 70, 750) + 8:.2f}" '
            f'y="{_scale(bucket.observed_score, 0, 1, 370, 35) + 4:.2f}">'
            f"n={bucket.count}</text>"
        )
        for bucket in analysis.calibration
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="calibration-title calibration-desc">'
        '<title id="calibration-title">PL calibration</title>'
        '<desc id="calibration-desc">Predicted pairwise probability versus observed '
        "result, including ties as one half.</desc>"
        '<line class="axis" x1="70" x2="750" y1="370" y2="370"/>'
        '<line class="axis" x1="70" x2="70" y1="35" y2="370"/>'
        '<text x="410" y="410" text-anchor="middle" class="axis-label">'
        "Predicted pairwise score</text>"
        '<text x="18" y="202" text-anchor="middle" transform="rotate(-90 18 202)" '
        'class="axis-label">Observed pairwise score</text>' + diagonal + marks + "</svg>"
    )


def _domain(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return (low - 1.0, high + 1.0)
    padding = (high - low) * 0.05
    return (low - padding, high + padding)


def _scale(
    value: float,
    domain_low: float,
    domain_high: float,
    range_low: float,
    range_high: float,
) -> float:
    fraction = (value - domain_low) / (domain_high - domain_low)
    return range_low + fraction * (range_high - range_low)


def _atomic_write(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
