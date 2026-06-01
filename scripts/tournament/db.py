"""SQLite-backed performance database for the tournament engine.

Tracks every algorithm evaluation across time, maintains rolling
rankings, and stores algorithm lifecycle status.

Location: data/tournament/tournament.db (WAL mode, concurrent-reader safe).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("nat.tournament.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    algo_name       TEXT NOT NULL,
    algo_source     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,
    n_trades        INTEGER,
    total_net_bps   REAL,
    net_bps_per_trade REAL,
    win_rate        REAL,
    max_loss_bps    REAL,
    sharpe_daily    REAL,
    ic_mean         REAL,
    elapsed_s       REAL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(algo_name, symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_runs_algo ON runs(algo_name, date);
CREATE INDEX IF NOT EXISTS idx_runs_date ON runs(date);

CREATE TABLE IF NOT EXISTS rankings (
    date              TEXT NOT NULL,
    algo_name         TEXT NOT NULL,
    rank              INTEGER NOT NULL,
    composite_score   REAL,
    rolling_7d_sharpe REAL,
    rolling_30d_sharpe REAL,
    rolling_7d_win_rate REAL,
    PRIMARY KEY (date, algo_name)
);

CREATE TABLE IF NOT EXISTS algorithm_status (
    algo_name       TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    source          TEXT,
    signal_id       TEXT,
    since_date      TEXT,
    reason          TEXT,
    rolling_sharpe  REAL,
    total_bps       REAL,
    days_tested     INTEGER NOT NULL DEFAULT 0
);
"""


class TournamentDB:
    """SQLite store for tournament performance data.

    Thread-safety: each instance owns its own connection.
    WAL mode: concurrent readers never block writers.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path("data/tournament/tournament.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path), timeout=10, check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def insert_run(self, *, algo_name: str, algo_source: str, symbol: str,
                   date: str, n_trades: int, total_net_bps: float,
                   net_bps_per_trade: float, win_rate: float,
                   max_loss_bps: float, sharpe_daily: float = 0.0,
                   ic_mean: float = 0.0, elapsed_s: float = 0.0) -> None:
        """Insert or replace one evaluation result."""
        self._conn.execute(
            """INSERT OR REPLACE INTO runs
               (algo_name, algo_source, symbol, date, n_trades,
                total_net_bps, net_bps_per_trade, win_rate, max_loss_bps,
                sharpe_daily, ic_mean, elapsed_s, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (algo_name, algo_source, symbol, date, n_trades,
             total_net_bps, net_bps_per_trade, win_rate, max_loss_bps,
             sharpe_daily, ic_mean, elapsed_s,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_runs(self, algo_name: str, *, symbol: Optional[str] = None,
                 last_n_days: Optional[int] = None) -> list[dict]:
        """Get run history for an algorithm."""
        sql = "SELECT * FROM runs WHERE algo_name = ?"
        params: list = [algo_name]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if last_n_days:
            sql += " AND date >= date('now', ?)"
            params.append(f"-{last_n_days} days")
        sql += " ORDER BY date DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_evaluated_dates(self, algo_name: Optional[str] = None) -> set[str]:
        """Return set of dates already evaluated (optionally for one algo)."""
        if algo_name:
            rows = self._conn.execute(
                "SELECT DISTINCT date FROM runs WHERE algo_name = ?",
                (algo_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT date FROM runs",
            ).fetchall()
        return {r["date"] for r in rows}

    def get_all_algo_names(self) -> list[str]:
        """Return sorted list of all algorithm names that have runs."""
        rows = self._conn.execute(
            "SELECT DISTINCT algo_name FROM runs ORDER BY algo_name"
        ).fetchall()
        return [r["algo_name"] for r in rows]

    def get_daily_pnl(self, algo_name: str, *,
                      last_n_days: Optional[int] = None) -> list[dict]:
        """Aggregate net_bps per date across all symbols.

        When last_n_days is set, the window is anchored to the most recent
        date in the DB (not wall-clock 'now') so queries work on historical data.
        """
        sql = """
            SELECT date, SUM(total_net_bps) as daily_bps,
                   AVG(win_rate) as avg_win_rate,
                   SUM(n_trades) as total_trades
            FROM runs WHERE algo_name = ?
        """
        params: list = [algo_name]
        if last_n_days:
            sql += (" AND date >= date("
                    "(SELECT MAX(date) FROM runs WHERE algo_name = ?), ?)")
            params.extend([algo_name, f"-{last_n_days} days"])
        sql += " GROUP BY date ORDER BY date"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def upsert_ranking(self, *, date: str, algo_name: str, rank: int,
                       composite_score: float, rolling_7d_sharpe: float,
                       rolling_30d_sharpe: float,
                       rolling_7d_win_rate: float) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO rankings
               (date, algo_name, rank, composite_score,
                rolling_7d_sharpe, rolling_30d_sharpe, rolling_7d_win_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (date, algo_name, rank, composite_score,
             rolling_7d_sharpe, rolling_30d_sharpe, rolling_7d_win_rate),
        )
        self._conn.commit()

    def get_rankings(self, date: str) -> list[dict]:
        """Get rankings for a given date, ordered by rank."""
        rows = self._conn.execute(
            "SELECT * FROM rankings WHERE date = ? ORDER BY rank",
            (date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_rankings(self) -> list[dict]:
        """Get rankings from the most recent date."""
        row = self._conn.execute(
            "SELECT MAX(date) as d FROM rankings"
        ).fetchone()
        if not row or not row["d"]:
            return []
        return self.get_rankings(row["d"])

    # ------------------------------------------------------------------
    # Algorithm status
    # ------------------------------------------------------------------

    def get_status(self, algo_name: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM algorithm_status WHERE algo_name = ?",
            (algo_name,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_status(self, *, algo_name: str, status: str,
                      source: str = "hand_coded",
                      signal_id: Optional[str] = None,
                      reason: Optional[str] = None,
                      rolling_sharpe: Optional[float] = None,
                      total_bps: Optional[float] = None,
                      days_tested: int = 0) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        existing = self.get_status(algo_name)
        since = today
        if existing and existing["status"] == status:
            since = existing["since_date"] or today
        self._conn.execute(
            """INSERT OR REPLACE INTO algorithm_status
               (algo_name, status, source, signal_id, since_date,
                reason, rolling_sharpe, total_bps, days_tested)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (algo_name, status, source, signal_id, since, reason,
             rolling_sharpe, total_bps, days_tested),
        )
        self._conn.commit()

    def get_all_statuses(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM algorithm_status ORDER BY status, algo_name"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_status(self, status: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM algorithm_status WHERE status = ? ORDER BY algo_name",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Rolling metrics (computed from runs table)
    # ------------------------------------------------------------------

    def compute_rolling_sharpe(self, algo_name: str,
                               window_days: int = 7) -> float:
        """Annualized Sharpe from daily PnL over the last N days.

        Zero variance with negative mean returns -inf-capped Sharpe (-99).
        """
        rows = self.get_daily_pnl(algo_name, last_n_days=window_days)
        if len(rows) < 2:
            return 0.0
        import numpy as np
        daily = np.array([r["daily_bps"] for r in rows])
        mu = np.mean(daily)
        sigma = np.std(daily, ddof=1)
        if sigma < 1e-12:
            # Constant PnL: sign determines Sharpe
            if mu < 0:
                return -99.0
            elif mu > 0:
                return 99.0
            return 0.0
        return float(mu / sigma * np.sqrt(252))

    def compute_rolling_win_rate(self, algo_name: str,
                                 window_days: int = 7) -> float:
        """Average win rate over the last N days."""
        rows = self.get_daily_pnl(algo_name, last_n_days=window_days)
        if not rows:
            return 0.0
        return float(sum(1 for r in rows if r["daily_bps"] > 0) / len(rows))

    def count_dates_tested(self, algo_name: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT date) as n FROM runs WHERE algo_name = ?",
            (algo_name,),
        ).fetchone()
        return row["n"] if row else 0
