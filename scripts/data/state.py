"""SQLite-backed state store for all NAT research agents.

Replaces per-agent JSON files with a single ``data/nat.db`` database.
WAL mode enables concurrent reads (dashboard) during agent writes.
Every mutation is atomic — no more partial-write corruption on crash.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("nat.state")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_state (
    agent  TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (agent, key)
);

CREATE TABLE IF NOT EXISTS state_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent      TEXT NOT NULL,
    from_phase TEXT NOT NULL,
    to_phase   TEXT NOT NULL,
    msg        TEXT NOT NULL DEFAULT '',
    at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_agent ON state_history(agent, id DESC);

CREATE TABLE IF NOT EXISTS hypotheses (
    id             TEXT PRIMARY KEY,
    agent          TEXT NOT NULL,
    claim          TEXT NOT NULL,
    generator      TEXT NOT NULL,
    priority       REAL NOT NULL DEFAULT 0.0,
    test_protocol  TEXT NOT NULL DEFAULT '[]',
    thresholds     TEXT NOT NULL DEFAULT '{}',
    status         TEXT NOT NULL DEFAULT 'queued',
    failure_reason TEXT,
    parent_id      TEXT,
    results        TEXT,
    created        TEXT NOT NULL,
    completed      TEXT
);
CREATE INDEX IF NOT EXISTS idx_hyp_agent_status ON hypotheses(agent, status);

CREATE TABLE IF NOT EXISTS registry (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    agent            TEXT NOT NULL,
    hypothesis_id    TEXT,
    name             TEXT NOT NULL,
    features         TEXT NOT NULL DEFAULT '[]',
    regime_gate      TEXT,
    spectral_band    TEXT,
    extraction       TEXT NOT NULL DEFAULT 'raw',
    horizon_s        REAL NOT NULL DEFAULT 5.0,
    expected_ic      REAL NOT NULL DEFAULT 0.0,
    expected_ir      REAL NOT NULL DEFAULT 0.0,
    decay_halflife_s REAL NOT NULL DEFAULT 0.0,
    symbols          TEXT NOT NULL DEFAULT '["BTC","ETH","SOL"]',
    correlation_with TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'validated',
    discovery_date   TEXT,
    last_validated   TEXT,
    ic_history       TEXT NOT NULL DEFAULT '[]',
    latest_ic        REAL,
    decay_days       INTEGER NOT NULL DEFAULT 0,
    retired_reason   TEXT,
    retired_date     TEXT,
    paper_sharpe     REAL,
    paper_days_elapsed INTEGER,
    realized_ic      REAL,
    max_drawdown_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_reg_agent_status ON registry(agent, status);
CREATE INDEX IF NOT EXISTS idx_reg_hypothesis ON registry(hypothesis_id);

CREATE TABLE IF NOT EXISTS generator_stats (
    agent     TEXT NOT NULL,
    generator TEXT NOT NULL,
    attempts  INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, generator)
);

CREATE TABLE IF NOT EXISTS _migrations (
    name       TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

# JSON columns in the hypotheses table
_HYP_JSON_COLS = {"test_protocol", "thresholds", "results"}

# JSON columns in the registry table
_REG_JSON_COLS = {
    "features", "spectral_band", "symbols",
    "correlation_with", "ic_history",
}

# All registry columns in insertion order (excluding auto-increment id)
_REG_COLS = [
    "agent", "hypothesis_id", "name", "features", "regime_gate",
    "spectral_band", "extraction", "horizon_s", "expected_ic",
    "expected_ir", "decay_halflife_s", "symbols", "correlation_with",
    "status", "discovery_date", "last_validated", "ic_history",
    "latest_ic", "decay_days", "retired_reason", "retired_date",
    "paper_sharpe", "paper_days_elapsed", "realized_ic", "max_drawdown_pct",
]


class StateStore:
    """Single SQLite store for all agent state.

    Thread-safety: each instance owns its own connection.
    Atomicity: every public method runs in an implicit transaction.
    WAL mode: concurrent readers never block writers.
    """

    def __init__(self, db_path: Path):
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

    # -----------------------------------------------------------------------
    # Agent state
    # -----------------------------------------------------------------------

    def load_state(self, agent: str) -> dict:
        """Load all key-value pairs for *agent*. Returns empty dict if none."""
        rows = self._conn.execute(
            "SELECT key, value FROM agent_state WHERE agent = ?", (agent,)
        ).fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def save_state(self, agent: str, data: dict) -> None:
        """Persist full state dict (upserts every key)."""
        with self._conn:
            for key, value in data.items():
                if key == "history":
                    continue  # history stored in state_history table
                self._conn.execute(
                    "INSERT OR REPLACE INTO agent_state (agent, key, value) "
                    "VALUES (?, ?, ?)",
                    (agent, key, json.dumps(value, default=str)),
                )

    def append_history(self, agent: str, entry: dict) -> None:
        """Append a state-transition record."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO state_history (agent, from_phase, to_phase, msg, at) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent, entry["from"], entry["to"],
                 entry.get("msg", ""), entry["at"]),
            )

    def load_history(self, agent: str, limit: int = 200) -> list[dict]:
        """Return most recent *limit* history entries (oldest first)."""
        rows = self._conn.execute(
            "SELECT from_phase, to_phase, msg, at FROM state_history "
            "WHERE agent = ? ORDER BY id DESC LIMIT ?",
            (agent, limit),
        ).fetchall()
        return [
            {"from": r["from_phase"], "to": r["to_phase"],
             "msg": r["msg"], "at": r["at"]}
            for r in reversed(rows)
        ]

    # -----------------------------------------------------------------------
    # Hypotheses
    # -----------------------------------------------------------------------

    def load_hypotheses(self, agent: str) -> list[dict]:
        """Load all hypotheses for *agent*."""
        rows = self._conn.execute(
            "SELECT * FROM hypotheses WHERE agent = ?", (agent,)
        ).fetchall()
        return [self._row_to_hyp(r) for r in rows]

    def upsert_hypothesis(self, agent: str, h: dict) -> None:
        """Insert or update a hypothesis."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO hypotheses "
                "(id, agent, claim, generator, priority, test_protocol, "
                " thresholds, status, failure_reason, parent_id, results, "
                " created, completed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    h["id"], agent, h["claim"], h["generator"],
                    h.get("priority", 0.0),
                    json.dumps(h.get("test_protocol", []), default=str),
                    json.dumps(h.get("thresholds", {}), default=str),
                    h.get("status", "queued"),
                    h.get("failure_reason"),
                    h.get("parent_id"),
                    json.dumps(h["results"], default=str) if h.get("results") else None,
                    h["created"],
                    h.get("completed"),
                ),
            )

    def delete_hypothesis(self, hyp_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM hypotheses WHERE id = ?", (hyp_id,))

    @staticmethod
    def _row_to_hyp(row: sqlite3.Row) -> dict:
        d = dict(row)
        d.pop("agent", None)
        for col in _HYP_JSON_COLS:
            if d.get(col) is not None:
                d[col] = json.loads(d[col])
        return d

    # -----------------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------------

    def load_registry(self, agent: str) -> list[dict]:
        """Load all registered signals for *agent*."""
        rows = self._conn.execute(
            "SELECT * FROM registry WHERE agent = ?", (agent,)
        ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    # Default values for registry columns with NOT NULL constraints
    _REG_DEFAULTS = {
        "name": "",
        "features": "[]",
        "extraction": "raw",
        "horizon_s": 5.0,
        "expected_ic": 0.0,
        "expected_ir": 0.0,
        "decay_halflife_s": 0.0,
        "symbols": '["BTC","ETH","SOL"]',
        "correlation_with": "{}",
        "status": "validated",
        "ic_history": "[]",
        "decay_days": 0,
    }

    def append_signal(self, agent: str, sig: dict) -> None:
        """Add a new signal to the registry."""
        values = [agent]
        for col in _REG_COLS[1:]:  # skip 'agent' — already added
            v = sig.get(col)
            if v is None and col in self._REG_DEFAULTS:
                v = self._REG_DEFAULTS[col]
            elif col in _REG_JSON_COLS and v is not None:
                v = json.dumps(v, default=str)
            values.append(v)
        placeholders = ", ".join("?" * len(_REG_COLS))
        cols = ", ".join(_REG_COLS)
        with self._conn:
            self._conn.execute(
                f"INSERT INTO registry ({cols}) VALUES ({placeholders})",
                values,
            )

    def update_signal(self, agent: str, hypothesis_id: str,
                      updates: dict) -> None:
        """Update fields of a signal identified by hypothesis_id."""
        if not updates:
            return
        sets = []
        values = []
        for k, v in updates.items():
            sets.append(f"{k} = ?")
            if k in _REG_JSON_COLS and v is not None:
                v = json.dumps(v, default=str)
            values.append(v)
        values.extend([agent, hypothesis_id])
        with self._conn:
            self._conn.execute(
                f"UPDATE registry SET {', '.join(sets)} "
                "WHERE agent = ? AND hypothesis_id = ?",
                values,
            )

    def remove_signal(self, agent: str, hypothesis_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM registry WHERE agent = ? AND hypothesis_id = ?",
                (agent, hypothesis_id),
            )

    @staticmethod
    def _row_to_signal(row: sqlite3.Row) -> dict:
        d = dict(row)
        d.pop("id", None)   # auto-increment id is internal
        d.pop("agent", None)
        for col in _REG_JSON_COLS:
            if d.get(col) is not None:
                try:
                    d[col] = json.loads(d[col])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    # -----------------------------------------------------------------------
    # Generator stats
    # -----------------------------------------------------------------------

    def load_gen_stats(self, agent: str) -> dict[str, dict]:
        """Load generator stats as {name: {attempts, successes}}."""
        rows = self._conn.execute(
            "SELECT generator, attempts, successes "
            "FROM generator_stats WHERE agent = ?",
            (agent,),
        ).fetchall()
        return {
            r["generator"]: {"attempts": r["attempts"], "successes": r["successes"]}
            for r in rows
        }

    def save_gen_stats(self, agent: str, stats: dict[str, dict]) -> None:
        """Persist generator stats (upserts each generator)."""
        with self._conn:
            for gen, s in stats.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO generator_stats "
                    "(agent, generator, attempts, successes) "
                    "VALUES (?, ?, ?, ?)",
                    (agent, gen, s.get("attempts", 0), s.get("successes", 0)),
                )

    # -----------------------------------------------------------------------
    # Cross-agent queries
    # -----------------------------------------------------------------------

    def all_registries(self) -> dict[str, list[dict]]:
        """Return all signals grouped by agent."""
        rows = self._conn.execute("SELECT * FROM registry").fetchall()
        result: dict[str, list[dict]] = {}
        for r in rows:
            agent = r["agent"]
            result.setdefault(agent, []).append(self._row_to_signal(r))
        return result

    def all_states(self) -> dict[str, dict]:
        """Return all agent states."""
        rows = self._conn.execute("SELECT * FROM agent_state").fetchall()
        result: dict[str, dict] = {}
        for r in rows:
            result.setdefault(r["agent"], {})[r["key"]] = json.loads(r["value"])
        return result

    # -----------------------------------------------------------------------
    # Migration from JSON
    # -----------------------------------------------------------------------

    def migrate_from_json(self, agent: str,
                          state_path: Path | None = None,
                          hyp_path: Path | None = None,
                          reg_path: Path | None = None,
                          stats_path: Path | None = None) -> bool:
        """One-time import from JSON files. Returns True if anything imported."""
        migration_key = f"json_import_{agent}"
        existing = self._conn.execute(
            "SELECT 1 FROM _migrations WHERE name = ?", (migration_key,)
        ).fetchone()
        if existing:
            return False

        imported = False
        now = datetime.now(timezone.utc).isoformat()

        with self._conn:
            # Agent state
            if state_path and state_path.exists():
                try:
                    with open(state_path) as f:
                        data = json.load(f)
                    history = data.pop("history", [])
                    for k, v in data.items():
                        self._conn.execute(
                            "INSERT OR REPLACE INTO agent_state (agent, key, value) "
                            "VALUES (?, ?, ?)",
                            (agent, k, json.dumps(v, default=str)),
                        )
                    for entry in history:
                        self._conn.execute(
                            "INSERT INTO state_history "
                            "(agent, from_phase, to_phase, msg, at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (agent, entry.get("from", ""),
                             entry.get("to", ""),
                             entry.get("msg", ""),
                             entry.get("at", now)),
                        )
                    imported = True
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Migration: failed to read %s: %s", state_path, e)

            # Hypotheses
            if hyp_path and hyp_path.exists():
                try:
                    with open(hyp_path) as f:
                        hyps = json.load(f)
                    for h in hyps:
                        self._conn.execute(
                            "INSERT OR IGNORE INTO hypotheses "
                            "(id, agent, claim, generator, priority, "
                            " test_protocol, thresholds, status, "
                            " failure_reason, parent_id, results, "
                            " created, completed) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                h["id"], agent, h.get("claim", ""),
                                h.get("generator", "unk"),
                                h.get("priority", 0.0),
                                json.dumps(h.get("test_protocol", []), default=str),
                                json.dumps(h.get("thresholds", {}), default=str),
                                h.get("status", "queued"),
                                h.get("failure_reason"),
                                h.get("parent_id"),
                                json.dumps(h["results"], default=str) if h.get("results") else None,
                                h.get("created", now),
                                h.get("completed"),
                            ),
                        )
                    imported = True
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Migration: failed to read %s: %s", hyp_path, e)

            # Registry
            if reg_path and reg_path.exists():
                try:
                    with open(reg_path) as f:
                        signals = json.load(f)
                    for sig in signals:
                        values = [agent]
                        for col in _REG_COLS[1:]:
                            v = sig.get(col)
                            if v is None and col in self._REG_DEFAULTS:
                                v = self._REG_DEFAULTS[col]
                            elif col in _REG_JSON_COLS and v is not None:
                                v = json.dumps(v, default=str)
                            values.append(v)
                        placeholders = ", ".join("?" * len(_REG_COLS))
                        cols = ", ".join(_REG_COLS)
                        self._conn.execute(
                            f"INSERT INTO registry ({cols}) VALUES ({placeholders})",
                            values,
                        )
                    imported = True
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Migration: failed to read %s: %s", reg_path, e)

            # Generator stats
            if stats_path and stats_path.exists():
                try:
                    with open(stats_path) as f:
                        stats = json.load(f)
                    for gen, s in stats.items():
                        self._conn.execute(
                            "INSERT OR REPLACE INTO generator_stats "
                            "(agent, generator, attempts, successes) "
                            "VALUES (?, ?, ?, ?)",
                            (agent, gen,
                             s.get("attempts", 0), s.get("successes", 0)),
                        )
                    imported = True
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Migration: failed to read %s: %s", stats_path, e)

            # Mark migration done
            self._conn.execute(
                "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                (migration_key, now),
            )

        if imported:
            log.info("Migrated JSON state for agent %r to SQLite", agent)
        return imported

    # -----------------------------------------------------------------------
    # JSON export (backward compat for external tools)
    # -----------------------------------------------------------------------

    def export_json(self, agent: str, output_dir: Path) -> None:
        """Export agent state to JSON files matching the old format."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # agent_state.json
        state = self.load_state(agent)
        state["history"] = self.load_history(agent, limit=200)
        with open(output_dir / "agent_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

        # hypotheses.json
        hyps = self.load_hypotheses(agent)
        with open(output_dir / "hypotheses.json", "w") as f:
            json.dump(hyps, f, indent=2, default=str)

        # registry.json
        registry = self.load_registry(agent)
        with open(output_dir / "registry.json", "w") as f:
            json.dump(registry, f, indent=2)

        # generator_stats.json
        stats = self.load_gen_stats(agent)
        with open(output_dir / "generator_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
