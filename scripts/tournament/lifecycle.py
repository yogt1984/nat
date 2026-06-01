"""Algorithm lifecycle management — degradation and promotion state machine.

Status flow:
    candidate → active (after activation_days profitable days)
    active → probation (after probation_after_days consecutive underperformance)
    probation → retired (after retire_after_days consecutive probation)
    probation → active (if recovers for recovery_days)
    active → promoted (all promotion criteria met for min_days_active)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import TournamentDB

log = logging.getLogger("nat.tournament.lifecycle")


def run_lifecycle_checks(db: TournamentDB, config: dict) -> dict:
    """Run degradation and promotion checks on all algorithms.

    Returns summary: {promoted: [], degraded: [], retired: [], recovered: []}
    """
    deg = config.get("degradation", {})
    promo = config.get("promotion", {})

    probation_after = deg.get("probation_after_days", 5)
    retire_after = deg.get("retire_after_days", 14)
    min_sharpe_7d = deg.get("min_sharpe_7d", 0.0)
    min_win_rate_7d = deg.get("min_win_rate_7d", 0.40)
    activation_days = deg.get("activation_days", 3)
    recovery_days = deg.get("recovery_days", 3)

    min_days_active = promo.get("min_days_active", 14)
    min_sharpe_30d = promo.get("min_sharpe_30d", 1.0)
    min_win_rate_30d = promo.get("min_win_rate_30d", 0.50)
    min_symbols = promo.get("min_symbols_profitable", 2)

    summary = {"promoted": [], "degraded": [], "retired": [], "recovered": [], "activated": []}

    all_algos = db.get_all_algo_names()

    for algo_name in all_algos:
        status_row = db.get_status(algo_name)
        current_status = status_row["status"] if status_row else "candidate"
        days_tested = db.count_dates_tested(algo_name)
        sharpe_7d = db.compute_rolling_sharpe(algo_name, 7)
        sharpe_30d = db.compute_rolling_sharpe(algo_name, 30)
        win_rate_7d = db.compute_rolling_win_rate(algo_name, 7)

        source = status_row["source"] if status_row else "hand_coded"
        signal_id = status_row["signal_id"] if status_row else None

        daily_pnl = db.get_daily_pnl(algo_name, last_n_days=30)
        total_bps = sum(r["daily_bps"] for r in daily_pnl) if daily_pnl else 0.0

        if current_status == "candidate":
            # Activate after N profitable days
            recent = db.get_daily_pnl(algo_name, last_n_days=activation_days)
            profitable_days = sum(1 for r in recent if r["daily_bps"] > 0)
            if profitable_days >= activation_days:
                db.upsert_status(
                    algo_name=algo_name, status="active", source=source,
                    signal_id=signal_id, reason="activated",
                    rolling_sharpe=sharpe_7d, total_bps=total_bps,
                    days_tested=days_tested,
                )
                summary["activated"].append(algo_name)
                log.info("ACTIVATED: %s (profitable %d/%d days)",
                         algo_name, profitable_days, activation_days)
            else:
                db.upsert_status(
                    algo_name=algo_name, status="candidate", source=source,
                    signal_id=signal_id, rolling_sharpe=sharpe_7d,
                    total_bps=total_bps, days_tested=days_tested,
                )

        elif current_status == "active":
            # Check for degradation
            underperforming = (sharpe_7d < min_sharpe_7d and win_rate_7d < min_win_rate_7d)
            if underperforming and days_tested >= probation_after:
                # Count consecutive underperformance days
                consec = _consecutive_underperformance(daily_pnl)
                if consec >= probation_after:
                    db.upsert_status(
                        algo_name=algo_name, status="probation", source=source,
                        signal_id=signal_id,
                        reason=f"sharpe_7d={sharpe_7d:.2f}, win_rate_7d={win_rate_7d:.2f}",
                        rolling_sharpe=sharpe_7d, total_bps=total_bps,
                        days_tested=days_tested,
                    )
                    summary["degraded"].append(algo_name)
                    log.warning("PROBATION: %s (Sharpe=%.2f, WR=%.2f, consec=%d)",
                                algo_name, sharpe_7d, win_rate_7d, consec)
                    continue

            # Check for promotion
            if (days_tested >= min_days_active
                    and sharpe_30d >= min_sharpe_30d
                    and db.compute_rolling_win_rate(algo_name, 30) >= min_win_rate_30d
                    and _count_profitable_symbols(db, algo_name) >= min_symbols):
                db.upsert_status(
                    algo_name=algo_name, status="promoted", source=source,
                    signal_id=signal_id, reason="all promotion criteria met",
                    rolling_sharpe=sharpe_30d, total_bps=total_bps,
                    days_tested=days_tested,
                )
                summary["promoted"].append(algo_name)
                log.info("PROMOTED: %s (Sharpe_30d=%.2f, %d days)",
                         algo_name, sharpe_30d, days_tested)
                continue

            # Otherwise, update metrics
            db.upsert_status(
                algo_name=algo_name, status="active", source=source,
                signal_id=signal_id, rolling_sharpe=sharpe_7d,
                total_bps=total_bps, days_tested=days_tested,
            )

        elif current_status == "probation":
            # Check for recovery
            recent = db.get_daily_pnl(algo_name, last_n_days=recovery_days)
            profitable = sum(1 for r in recent if r["daily_bps"] > 0)
            if profitable >= recovery_days and sharpe_7d >= min_sharpe_7d:
                db.upsert_status(
                    algo_name=algo_name, status="active", source=source,
                    signal_id=signal_id, reason="recovered",
                    rolling_sharpe=sharpe_7d, total_bps=total_bps,
                    days_tested=days_tested,
                )
                summary["recovered"].append(algo_name)
                log.info("RECOVERED: %s (Sharpe=%.2f)", algo_name, sharpe_7d)
                continue

            # Check for retirement
            consec = _consecutive_underperformance(daily_pnl)
            if consec >= retire_after:
                db.upsert_status(
                    algo_name=algo_name, status="retired", source=source,
                    signal_id=signal_id,
                    reason=f"underperformed {consec} consecutive days",
                    rolling_sharpe=sharpe_7d, total_bps=total_bps,
                    days_tested=days_tested,
                )
                summary["retired"].append(algo_name)
                log.info("RETIRED: %s (underperformed %d days)", algo_name, consec)
                continue

            # Still in probation
            db.upsert_status(
                algo_name=algo_name, status="probation", source=source,
                signal_id=signal_id, rolling_sharpe=sharpe_7d,
                total_bps=total_bps, days_tested=days_tested,
            )

        # promoted and retired are terminal — don't update

    return summary


def _consecutive_underperformance(daily_pnl: list[dict]) -> int:
    """Count consecutive days with negative PnL from the end."""
    count = 0
    for row in reversed(daily_pnl):
        if row["daily_bps"] <= 0:
            count += 1
        else:
            break
    return count


def _count_profitable_symbols(db: TournamentDB, algo_name: str) -> int:
    """Count symbols where algo has positive total PnL over last 30 days."""
    count = 0
    for sym in ["BTC", "ETH", "SOL"]:
        runs = db.get_runs(algo_name, symbol=sym, last_n_days=30)
        if runs:
            total = sum(r["total_net_bps"] for r in runs)
            if total > 0:
                count += 1
    return count
