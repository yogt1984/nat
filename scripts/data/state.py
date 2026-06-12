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

CREATE TABLE IF NOT EXISTS research_output (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    agent           TEXT NOT NULL,
    generator       TEXT,
    status          TEXT,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_ro_agent ON research_output(agent);
CREATE INDEX IF NOT EXISTS idx_ro_kind ON research_output(kind);
CREATE INDEX IF NOT EXISTS idx_ro_created ON research_output(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ro_status ON research_output(status);

CREATE TABLE IF NOT EXISTS budget (
    agent                    TEXT PRIMARY KEY,
    max_hypotheses_per_cycle INTEGER NOT NULL,
    compute_share            REAL NOT NULL DEFAULT 0.0,
    updated_at               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS directives (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    target_agent TEXT NOT NULL,
    action       TEXT NOT NULL,
    payload      TEXT,
    consumed     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dir_target ON directives(target_agent, consumed);

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
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply incremental schema migrations."""
        migrations = [
            ("add_data_version_to_hypotheses",
             "ALTER TABLE hypotheses ADD COLUMN data_version TEXT"),
            ("create_llm_calls", """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    tag TEXT NOT NULL DEFAULT '',
                    system TEXT NOT NULL,
                    user_msg TEXT NOT NULL,
                    response TEXT,
                    model TEXT NOT NULL DEFAULT '',
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    latency_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
             """),
            ("create_llm_calls_index",
             "CREATE INDEX IF NOT EXISTS idx_llm_agent ON llm_calls(agent, created_at DESC)"),
            ("create_arxiv_papers", """
                CREATE TABLE IF NOT EXISTS arxiv_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    published TEXT NOT NULL,
                    categories TEXT NOT NULL DEFAULT '[]',
                    ideas TEXT,
                    processed INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT,
                    created_at TEXT NOT NULL
                )
             """),
            ("create_arxiv_index",
             "CREATE INDEX IF NOT EXISTS idx_arxiv_processed ON arxiv_papers(processed)"),
            ("create_process_results", """
                CREATE TABLE IF NOT EXISTS process_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    process TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    n_tested INTEGER NOT NULL DEFAULT 0,
                    n_informative INTEGER NOT NULL DEFAULT 0,
                    top_feature TEXT,
                    top_metric TEXT,
                    top_value REAL,
                    json_path TEXT NOT NULL,
                    git_sha TEXT,
                    data_fingerprint TEXT,
                    created_at TEXT NOT NULL
                )
             """),
            ("create_process_results_index",
             "CREATE INDEX IF NOT EXISTS idx_process_results "
             "ON process_results(process, symbol, created_at DESC)"),
        ]
        for name, sql in migrations:
            already = self._conn.execute(
                "SELECT 1 FROM _migrations WHERE name = ?", (name,)
            ).fetchone()
            if already:
                continue
            try:
                with self._conn:
                    self._conn.execute(sql)
                    self._conn.execute(
                        "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                        (name, datetime.now(timezone.utc).isoformat()),
                    )
                log.info("Applied migration: %s", name)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    # Column already exists (e.g. from manual ALTER)
                    with self._conn:
                        self._conn.execute(
                            "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
                            (name, datetime.now(timezone.utc).isoformat()),
                        )
                else:
                    raise

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
                " created, completed, data_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    h.get("data_version"),
                ),
            )

    def delete_hypothesis(self, hyp_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM hypotheses WHERE id = ?", (hyp_id,))

    def check_provenance(self, current_version: str) -> list[dict]:
        """Return hypotheses whose data_version differs from *current_version*."""
        rows = self._conn.execute(
            "SELECT id, agent, claim, status, data_version "
            "FROM hypotheses "
            "WHERE data_version IS NOT NULL AND data_version != ?",
            (current_version,),
        ).fetchall()
        return [dict(r) for r in rows]

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
    # Budget
    # -----------------------------------------------------------------------

    def set_budget(self, agent: str, max_hypotheses: int,
                   compute_share: float) -> None:
        """Set the per-cycle budget for an agent (written by meta-agent)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO budget "
                "(agent, max_hypotheses_per_cycle, compute_share, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (agent, max_hypotheses, compute_share, now),
            )

    def get_budget(self, agent: str) -> dict | None:
        """Read the budget for *agent*. Returns None if not set."""
        row = self._conn.execute(
            "SELECT max_hypotheses_per_cycle, compute_share, updated_at "
            "FROM budget WHERE agent = ?",
            (agent,),
        ).fetchone()
        if row:
            return {
                "max_hypotheses_per_cycle": row["max_hypotheses_per_cycle"],
                "compute_share": row["compute_share"],
                "updated_at": row["updated_at"],
            }
        return None

    # -----------------------------------------------------------------------
    # Directives (meta-agent → research agents)
    # -----------------------------------------------------------------------

    def add_directive(self, target_agent: str, action: str,
                      payload: dict | None = None) -> int:
        """Queue a directive for *target_agent*. Returns the directive id."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO directives (target_agent, action, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (target_agent, action,
                 json.dumps(payload, default=str) if payload else None, now),
            )
            return cur.lastrowid

    def consume_directives(self, agent: str) -> list[dict]:
        """Read and mark consumed all pending directives for *agent*."""
        rows = self._conn.execute(
            "SELECT id, action, payload, created_at FROM directives "
            "WHERE target_agent = ? AND consumed = 0 ORDER BY id",
            (agent,),
        ).fetchall()
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        with self._conn:
            self._conn.execute(
                f"UPDATE directives SET consumed = 1 "
                f"WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
        return [
            {
                "id": r["id"],
                "action": r["action"],
                "payload": json.loads(r["payload"]) if r["payload"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # -----------------------------------------------------------------------
    # Research output
    # -----------------------------------------------------------------------

    def insert_research_output(self, record: dict, kind: str) -> None:
        """Insert a hypothesis or cycle record into research_output table."""
        record_id = record.get("id") or record.get("cycle_id")
        if not record_id:
            return
        agent = record.get("agent", "")
        generator = record.get("generator")
        status = record.get("status")
        created_at = (
            record.get("timestamps", {}).get("completed")
            or record.get("completed")
            or datetime.now(timezone.utc).isoformat()
        )
        schema_version = record.get("schema_version", 1)
        payload = json.dumps(record, default=str)

        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO research_output "
                "(id, kind, agent, generator, status, payload, created_at, schema_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (record_id, kind, agent, generator, status, payload,
                 created_at, schema_version),
            )

    def insert_process_result(self, row: dict) -> None:
        """Insert (or replace) one row into the process_results index.

        `row` keys mirror the table columns; full records live as JSON at
        `json_path` (data/research/processes/). See scripts/processes/.
        """
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO process_results "
                "(run_id, process, kind, symbol, timeframe, start_date, end_date, "
                " n_tested, n_informative, top_feature, top_metric, top_value, "
                " json_path, git_sha, data_fingerprint, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["run_id"], row["process"], row["kind"],
                    row.get("symbol"), row.get("timeframe"),
                    row.get("start_date"), row.get("end_date"),
                    row.get("n_tested", 0), row.get("n_informative", 0),
                    row.get("top_feature"), row.get("top_metric"), row.get("top_value"),
                    row["json_path"], row.get("git_sha"), row.get("data_fingerprint"),
                    row.get("created_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_process_results(
        self,
        process: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query the process_results index, newest first."""
        where_parts, params = [], []
        if process:
            where_parts.append("process = ?")
            params.append(process)
        if symbol:
            where_parts.append("symbol = ?")
            params.append(symbol)
        where = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        rows = self._conn.execute(
            f"SELECT * FROM process_results{where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def query_research_output(
        self,
        kind: str | None = None,
        agent: str | None = None,
        generator: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """Query research output with filters. Returns (items, total)."""
        where_parts = []
        params: list[Any] = []

        if kind:
            where_parts.append("kind = ?")
            params.append(kind)
        if agent:
            where_parts.append("agent = ?")
            params.append(agent)
        if generator:
            where_parts.append("generator = ?")
            params.append(generator)
        if status:
            where_parts.append("status = ?")
            params.append(status)

        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # Count total
        count_row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM research_output{where_clause}",
            params,
        ).fetchone()
        total = count_row["cnt"] if count_row else 0

        # Fetch page
        rows = self._conn.execute(
            f"SELECT payload FROM research_output{where_clause} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        items = [json.loads(r["payload"]) for r in rows]
        return items, total

    def get_research_output(self, record_id: str) -> dict | None:
        """Get a single research output record by ID."""
        row = self._conn.execute(
            "SELECT payload FROM research_output WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row:
            return json.loads(row["payload"])
        return None

    # -----------------------------------------------------------------------
    # Retention / cleanup
    # -----------------------------------------------------------------------

    def delete_old_research_output(self, max_age_days: int) -> int:
        """Delete research_output rows older than *max_age_days*. Returns count."""
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
            days=max_age_days,
        )
        cutoff_iso = cutoff.isoformat()
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM research_output WHERE created_at < ?",
                (cutoff_iso,),
            )
            return cur.rowcount

    @staticmethod
    def cleanup_old_json(research_dir: Path, max_age_days: int) -> int:
        """Remove hypothesis/cycle JSON files older than *max_age_days*.

        Reads each file's timestamp field to determine age. Returns count.
        """
        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
            days=max_age_days,
        )
        removed = 0
        for subdir in ("hypotheses", "cycles"):
            d = research_dir / subdir
            if not d.is_dir():
                continue
            for f in d.glob("*.json"):
                try:
                    record = json.loads(f.read_text())
                    ts = (
                        record.get("timestamps", {}).get("completed")
                        or record.get("completed")
                        or record.get("created_at")
                    )
                    if not ts:
                        continue
                    if datetime.fromisoformat(ts) < cutoff:
                        f.unlink()
                        removed += 1
                except (json.JSONDecodeError, OSError, ValueError):
                    continue
        return removed

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


# ---------------------------------------------------------------------------
# CLI: python -m data.state {status,export,migrate}
# ---------------------------------------------------------------------------

def _cli_status(db_path: Path) -> None:
    """Print all agents and their current phase."""
    store = StateStore(db_path)
    states = store.all_states()
    if not states:
        print("(no agents found in database)")
        return
    print(f"{'Agent':<20s} {'Phase':<12s} {'Cycles':>6s}  Last transition")
    print("-" * 65)
    for agent, data in sorted(states.items()):
        phase = data.get("phase", "?")
        cycles = data.get("cycle_count", "?")
        history = store.load_history(agent, limit=1)
        last = history[-1]["at"] if history else "—"
        print(f"{agent:<20s} {phase:<12s} {str(cycles):>6s}  {last}")
    store.close()


def _cli_export(db_path: Path, agent: str, output_dir: Path) -> None:
    store = StateStore(db_path)
    store.export_json(agent, output_dir)
    print(f"Exported {agent!r} to {output_dir}")
    store.close()


def _cli_migrate(db_path: Path, data_root: Path) -> None:
    """Run JSON→SQLite migration for all known agent directories."""
    store = StateStore(db_path)
    agents = {
        "microstructure": data_root / "agent",
        "medium_freq": data_root / "agent_mf",
        "macro": data_root / "agent_macro",
        "cascade": data_root / "agent_cascade",
        "meta": data_root / "agent_meta",
    }
    for agent, dir_path in agents.items():
        state_path = dir_path / "agent_state.json"
        if agent == "meta":
            state_path = dir_path / "meta_state.json"
        imported = store.migrate_from_json(
            agent,
            state_path=state_path if state_path.exists() else None,
            hyp_path=(dir_path / "hypotheses.json")
            if (dir_path / "hypotheses.json").exists() else None,
            reg_path=(dir_path / "registry.json")
            if (dir_path / "registry.json").exists() else None,
            stats_path=(dir_path / "generator_stats.json")
            if (dir_path / "generator_stats.json").exists() else None,
        )
        status = "migrated" if imported else "already done"
        print(f"  {agent:<20s} {status}")
    store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NAT SQLite state store CLI",
        prog="python -m data.state",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show all agent states")

    p_export = sub.add_parser("export", help="Export agent state to JSON")
    p_export.add_argument("agent", help="Agent name (e.g. microstructure)")
    p_export.add_argument("output_dir", help="Output directory")

    sub.add_parser("migrate", help="Migrate all JSON state files to SQLite")

    parser.add_argument("--db", default=None,
                        help="Path to nat.db (default: data/nat.db)")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    db = Path(args.db) if args.db else root / "data" / "nat.db"

    if args.command == "status":
        _cli_status(db)
    elif args.command == "export":
        _cli_export(db, args.agent, Path(args.output_dir))
    elif args.command == "migrate":
        _cli_migrate(db, root / "data")
    else:
        parser.print_help()
