"""Signal lifecycle state machine (plan T3).

The single source of truth for a signal's promotion state, persisted to `nat.db`.
Tables (`signal_lifecycle`, `lifecycle_history`) are created via the migration
framework in `scripts/data/state.py` — one migration framework only. Every insert
and transition is provenance-stamped (`git_sha`) and appended to `lifecycle_history`;
illegal transitions raise.

State machine (maps onto the maturity ladder in docs/contracts/README.md):

    DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING → LIVE → MONITORING → RETIRED
                     ↘ REJECTED ↙  (any pre-LIVE state can be rejected)

RETIRED and REJECTED are terminal. APPROVAL_PENDING → LIVE is the sole human gate.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data.state import StateStore

# --- states ----------------------------------------------------------------
DISCOVERED = "DISCOVERED"
VALIDATED = "VALIDATED"
PAPER_TRADING = "PAPER_TRADING"
APPROVAL_PENDING = "APPROVAL_PENDING"
LIVE = "LIVE"
MONITORING = "MONITORING"
RETIRED = "RETIRED"
REJECTED = "REJECTED"

STATES = {
    DISCOVERED, VALIDATED, PAPER_TRADING, APPROVAL_PENDING,
    LIVE, MONITORING, RETIRED, REJECTED,
}

# Allowed forward transitions. RETIRED / REJECTED are terminal (empty sets).
# reject() is permitted only pre-LIVE; a LIVE/MONITORING signal is retired, not rejected.
TRANSITIONS: dict[str, set[str]] = {
    DISCOVERED:       {VALIDATED, REJECTED},
    VALIDATED:        {PAPER_TRADING, REJECTED},
    PAPER_TRADING:    {APPROVAL_PENDING, REJECTED},
    APPROVAL_PENDING: {LIVE, REJECTED},
    LIVE:             {MONITORING, RETIRED},
    MONITORING:       {LIVE, RETIRED},
    RETIRED:          set(),
    REJECTED:         set(),
}


class IllegalTransition(Exception):
    """Raised when a transition is not permitted by TRANSITIONS."""


class UnknownSignal(KeyError):
    """Raised when operating on a signal_id absent from the lifecycle table."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provenance() -> dict:
    """git_sha + dirty + timestamp; prefers scripts/provenance.py (plan T2) when present."""
    ts = _now()
    try:  # T2's unified module, optional until it lands
        from provenance import get_provenance as _gp  # type: ignore
        p = _gp()
        p.setdefault("generated_at", ts)
        return p
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parent.parent
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        return {"git_sha": sha, "dirty": dirty, "generated_at": ts}
    except Exception:
        return {"git_sha": None, "dirty": False, "generated_at": ts}


class SignalLifecycle:
    """State-machine API over the `signal_lifecycle` / `lifecycle_history` tables."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "nat.db"
        # StateStore.__init__ runs _run_migrations(), creating our tables.
        self._store = StateStore(Path(db_path))
        self._conn = self._store._conn

    def close(self) -> None:
        self._store.close()

    # -- creation -----------------------------------------------------------
    def discover(self, signal_id: str, name: str = "", agent: str | None = None,
                 metadata: dict | None = None, msg: str = "") -> None:
        """Insert a new signal in DISCOVERED. Raises if signal_id already exists."""
        prov = _provenance()
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO signal_lifecycle "
                "(signal_id, name, agent, state, metadata, git_sha, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (signal_id, name, agent, DISCOVERED,
                 json.dumps(metadata or {}, default=str),
                 prov.get("git_sha"), now, now),
            )
            self._record(signal_id, "", DISCOVERED, msg or "discovered", prov, now)

    # -- transitions --------------------------------------------------------
    def _transition(self, signal_id: str, to_state: str, msg: str = "") -> None:
        row = self._conn.execute(
            "SELECT state FROM signal_lifecycle WHERE signal_id = ?", (signal_id,),
        ).fetchone()
        if row is None:
            raise UnknownSignal(signal_id)
        cur = row["state"]
        if to_state not in TRANSITIONS.get(cur, set()):
            raise IllegalTransition(f"{signal_id}: {cur} -> {to_state} not allowed")
        prov = _provenance()
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE signal_lifecycle SET state = ?, git_sha = ?, updated_at = ? "
                "WHERE signal_id = ?",
                (to_state, prov.get("git_sha"), now, signal_id),
            )
            self._record(signal_id, cur, to_state, msg, prov, now)

    def validate(self, signal_id: str, msg: str = "") -> None:
        self._transition(signal_id, VALIDATED, msg)

    def start_paper(self, signal_id: str, msg: str = "") -> None:
        self._transition(signal_id, PAPER_TRADING, msg)

    def request_approval(self, signal_id: str, msg: str = "") -> None:
        self._transition(signal_id, APPROVAL_PENDING, msg)

    def approve(self, signal_id: str, msg: str = "") -> None:
        """The sole human gate: APPROVAL_PENDING → LIVE."""
        self._transition(signal_id, LIVE, msg)

    def monitor(self, signal_id: str, msg: str = "") -> None:
        self._transition(signal_id, MONITORING, msg)

    def reject(self, signal_id: str, reason: str = "", msg: str = "") -> None:
        self._transition(signal_id, REJECTED, msg or reason)

    def retire(self, signal_id: str, reason: str = "", msg: str = "") -> None:
        self._transition(signal_id, RETIRED, msg or reason)

    # -- reads --------------------------------------------------------------
    def get_signal(self, signal_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM signal_lifecycle WHERE signal_id = ?", (signal_id,),
        ).fetchone()
        return self._row(row) if row else None

    def list_signals(self, state: str | None = None) -> list[dict]:
        if state is not None and state not in STATES:
            raise ValueError(f"unknown state {state!r}; valid: {sorted(STATES)}")
        if state:
            rows = self._conn.execute(
                "SELECT * FROM signal_lifecycle WHERE state = ? ORDER BY updated_at DESC",
                (state,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM signal_lifecycle ORDER BY updated_at DESC",
            ).fetchall()
        return [self._row(r) for r in rows]

    def history(self, signal_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT from_state, to_state, msg, git_sha, at FROM lifecycle_history "
            "WHERE signal_id = ? ORDER BY id",
            (signal_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- helpers ------------------------------------------------------------
    def _record(self, signal_id, from_state, to_state, msg, prov, now) -> None:
        self._conn.execute(
            "INSERT INTO lifecycle_history "
            "(signal_id, from_state, to_state, msg, git_sha, at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (signal_id, from_state, to_state, msg, prov.get("git_sha"), now),
        )

    @staticmethod
    def _row(row) -> dict:
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
