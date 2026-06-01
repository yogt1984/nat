"""Tournament reporting — rankings, telegram summary, markdown reports."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .db import TournamentDB

log = logging.getLogger("nat.tournament.report")

REPORTS_DIR = Path("reports")


def compute_rankings(db: TournamentDB, date: str, config: dict) -> list[dict]:
    """Compute composite rankings for all algorithms on a given date.

    Writes results to the rankings table and returns sorted list.
    """
    scoring = config.get("scoring", {})
    w_sharpe = scoring.get("sharpe_weight", 0.4)
    w_wr = scoring.get("win_rate_weight", 0.3)
    w_ic = scoring.get("ic_weight", 0.3)

    algo_names = db.get_all_algo_names()
    entries = []

    for algo in algo_names:
        sharpe_7d = db.compute_rolling_sharpe(algo, 7)
        sharpe_30d = db.compute_rolling_sharpe(algo, 30)
        win_rate_7d = db.compute_rolling_win_rate(algo, 7)

        # Normalize IC: use average across recent runs
        daily = db.get_daily_pnl(algo, last_n_days=7)
        if daily:
            avg_daily_bps = np.mean([r["daily_bps"] for r in daily])
        else:
            avg_daily_bps = 0.0

        # Composite: normalize each component to [0, 1] range approximately
        # Sharpe: clip to [-3, 3], scale to [0, 1]
        s_norm = (np.clip(sharpe_7d, -3, 3) + 3) / 6
        # Win rate: already [0, 1]
        wr_norm = np.clip(win_rate_7d, 0, 1)
        # IC proxy: use sign(daily_bps) as rough indicator
        ic_norm = np.clip(avg_daily_bps / 10, -1, 1) * 0.5 + 0.5

        composite = w_sharpe * s_norm + w_wr * wr_norm + w_ic * ic_norm

        entries.append({
            "algo_name": algo,
            "composite_score": round(float(composite), 4),
            "rolling_7d_sharpe": round(sharpe_7d, 2),
            "rolling_30d_sharpe": round(sharpe_30d, 2),
            "rolling_7d_win_rate": round(win_rate_7d, 3),
        })

    # Sort by composite descending
    entries.sort(key=lambda e: e["composite_score"], reverse=True)

    # Assign ranks and persist
    for i, entry in enumerate(entries):
        entry["rank"] = i + 1
        db.upsert_ranking(
            date=date,
            algo_name=entry["algo_name"],
            rank=entry["rank"],
            composite_score=entry["composite_score"],
            rolling_7d_sharpe=entry["rolling_7d_sharpe"],
            rolling_30d_sharpe=entry["rolling_30d_sharpe"],
            rolling_7d_win_rate=entry["rolling_7d_win_rate"],
        )

    return entries


def format_rankings_table(rankings: list[dict]) -> str:
    """Format rankings as an aligned text table."""
    if not rankings:
        return "No rankings available."

    lines = [
        f"{'Rank':>4}  {'Algorithm':<28} {'Score':>6} {'SR_7d':>6} {'SR_30d':>7} {'WR_7d':>6}",
        "-" * 68,
    ]
    for r in rankings:
        lines.append(
            f"{r['rank']:>4}  {r['algo_name']:<28} "
            f"{r['composite_score']:>6.3f} "
            f"{r['rolling_7d_sharpe']:>6.2f} "
            f"{r['rolling_30d_sharpe']:>7.2f} "
            f"{r['rolling_7d_win_rate']:>6.1%}"
        )
    return "\n".join(lines)


def format_telegram_summary(date: str, rankings: list[dict],
                            lifecycle_summary: dict,
                            n_evaluated: int) -> str:
    """Format a concise Telegram message."""
    lines = [f"NAT Tournament | {date}", "---"]

    # Top 3
    top = rankings[:3]
    if top:
        top_str = ", ".join(
            f"{r['algo_name']} (SR {r['rolling_7d_sharpe']:.1f})"
            for r in top
        )
        lines.append(f"Top 3: {top_str}")

    # Lifecycle events
    activated = lifecycle_summary.get("activated", [])
    if activated:
        lines.append(f"New active: {len(activated)} ({', '.join(activated[:3])})")

    promoted = lifecycle_summary.get("promoted", [])
    if promoted:
        lines.append(f"Promoted: {', '.join(promoted)}")

    degraded = lifecycle_summary.get("degraded", [])
    if degraded:
        lines.append(f"Probation: {', '.join(degraded)}")

    retired = lifecycle_summary.get("retired", [])
    if retired:
        lines.append(f"Retired: {', '.join(retired)}")

    recovered = lifecycle_summary.get("recovered", [])
    if recovered:
        lines.append(f"Recovered: {', '.join(recovered)}")

    # Totals
    statuses = {
        "active": len([r for r in rankings]),  # approximate
    }
    lines.append(f"Total ranked: {len(rankings)} | Evaluated today: {n_evaluated}")

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Send message to Telegram. Returns True on success."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.debug("Telegram not configured — skipping send")
        return False

    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [message[i:i + 4096] for i in range(0, len(message), 4096)]
    for chunk in chunks:
        try:
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
            }, timeout=10)
            if not resp.ok:
                log.warning("Telegram send failed: %s", resp.text)
                return False
        except Exception as e:
            log.warning("Telegram send error: %s", e)
            return False
    return True


def generate_markdown_report(db: TournamentDB, date: str,
                             rankings: list[dict],
                             lifecycle_summary: dict) -> str:
    """Generate a full markdown report."""
    lines = [
        f"# Tournament Report — {date}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()[:19]}Z",
        "",
        "## Rankings",
        "",
        "| Rank | Algorithm | Score | SR 7d | SR 30d | WR 7d |",
        "|------|-----------|-------|-------|--------|-------|",
    ]

    for r in rankings:
        lines.append(
            f"| {r['rank']} | {r['algo_name']} | {r['composite_score']:.3f} "
            f"| {r['rolling_7d_sharpe']:.2f} | {r['rolling_30d_sharpe']:.2f} "
            f"| {r['rolling_7d_win_rate']:.1%} |"
        )

    lines.extend(["", "## Lifecycle Events", ""])

    for event_type, algos in lifecycle_summary.items():
        if algos:
            lines.append(f"**{event_type.capitalize()}**: {', '.join(algos)}")

    # Per-algorithm status summary
    statuses = db.get_all_statuses()
    if statuses:
        lines.extend(["", "## Algorithm Status", ""])
        lines.append("| Algorithm | Status | Since | Days Tested | SR |")
        lines.append("|-----------|--------|-------|-------------|-----|")
        for s in statuses:
            lines.append(
                f"| {s['algo_name']} | {s['status']} | {s['since_date'] or '-'} "
                f"| {s['days_tested']} | {s['rolling_sharpe'] or 0:.2f} |"
            )

    return "\n".join(lines)


def save_report(date: str, content: str) -> Path:
    """Save markdown report to reports/ directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"tournament_{date}.md"
    path.write_text(content)
    log.info("Saved report to %s", path)
    return path
