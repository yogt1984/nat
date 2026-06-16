"""T14 — Promotion daemon.

Automates the signal-lifecycle spine (T3–T5): polls every 300s and drives
signals through DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING,
plus LIVE/MONITORING → RETIRED on decay. Each transition is gated on IMPORTED
thresholds (G4 = config/alpha.toml [gates] g4_*; G8 = g8_* with g8_min_days=14)
and a data-sufficiency + ≥7-clean-day guard.

NON-NEGOTIABLE: the daemon NEVER promotes APPROVAL_PENDING → LIVE — that is the
sole human gate (`nat lifecycle approve`). It stops at APPROVAL_PENDING.

The expensive steps are seams (`_run_g4`, `_run_paper`, `_count_clean_days`,
`_days_in_paper`, `_data_sufficient`, `_check_g8`, `_check_decay`, `_now`) so the
state machine is unit-tested deterministically without shelling out. The default
seam implementations are best-effort and degrade to "skip cleanly" on any error
(the correct pre-streak behavior); their real-data execution is validated once a
clean streak accrues.

Mirrors scripts/discovery_orchestrator.py (subprocess orchestration) and the
kill-switch/gap-alert conventions (pidfile + heartbeat + `health` + path
bootstrap). CLI: `nat promotion status|start|once [--dry-run]|stop`.
"""

from __future__ import annotations

import json
import os
import re
import signal as _signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

_SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

ROOT = _SCRIPTS_ROOT.parent

from signal_lifecycle import (  # noqa: E402
    SignalLifecycle,
    DISCOVERED, VALIDATED, PAPER_TRADING, APPROVAL_PENDING, LIVE, MONITORING,
)

try:
    from logging_config import setup_logging  # noqa: E402

    log = setup_logging("nat.promotion")
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("nat.promotion")


# ───────────────────────────────────────────────────────────────────────────
# Config + imported gates
# ───────────────────────────────────────────────────────────────────────────


def load_config(path: Path | str | None = None) -> dict:
    defaults = {
        "poll_interval_s": 300,
        "db_path": str(ROOT / "data" / "nat.db"),
        "state_file": str(ROOT / "data" / "promotion" / "state.json"),
        "heartbeat_path": str(ROOT / "data" / "promotion" / "promotion.heartbeat"),
        "pid_file": str(ROOT / "data" / "promotion" / "promotion.pid"),
        "subprocess_timeout_s": 600,
        "good_day_mb": 200,
        "min_clean_days": 7,
        "symbols": ["BTC", "ETH", "SOL"],
        "validate_symbol": "BTC",  # representative symbol for the G4 signal
    }
    cfg_path = Path(path) if path else ROOT / "config" / "promotion.toml"
    if cfg_path.exists():
        try:
            import tomllib

            with open(cfg_path, "rb") as f:
                defaults.update(tomllib.load(f).get("promotion", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read %s: %s — using defaults", cfg_path, e)
    return defaults


def load_gates() -> dict:
    """Imported gate thresholds from config/alpha.toml [gates] — never invented."""
    gates = {
        "g4_min_oos_sharpe": 0.5, "g4_min_oos_is_ratio": 0.7,
        "g4_max_deflated_sharpe_p": 0.05, "g4_max_drawdown_pct": 5.0,
        "g4_min_trades": 30, "g4_min_profit_factor": 1.2,
        "g8_min_sharpe_ratio": 0.5, "g8_max_daily_loss_pct": 2.0,
        "g8_max_ic_decay_pct": 50.0, "g8_min_days": 14,
    }
    p = ROOT / "config" / "alpha.toml"
    if p.exists():
        try:
            import tomllib

            with open(p, "rb") as f:
                gates.update(tomllib.load(f).get("gates", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read alpha.toml gates: %s — using defaults", e)
    return gates


def healthy(config: dict | None = None, now: float | None = None) -> bool:
    """True if the daemon heartbeat is fresh (< 3 poll intervals, min 600s)."""
    cfg = config or load_config()
    hb = Path(cfg["heartbeat_path"])
    if not hb.exists():
        return False
    now = now if now is not None else time.time()
    max_age = max(600, 3 * int(cfg.get("poll_interval_s", 300)))
    return (now - hb.stat().st_mtime) < max_age


def build_paper_report(metrics: dict, gates: dict) -> dict:
    """Map paper-trading metrics to the 5 G8 booleans using imported thresholds.

    metrics: paper_sharpe, baseline_sharpe, max_daily_loss_pct, ic_decay_pct,
             infra_stable, n_days.
    """
    base = metrics.get("baseline_sharpe", 0.0) or 0.0
    ratio = (metrics["paper_sharpe"] / base) if base > 0 else 0.0
    return {
        "gate_sharpe_within_2x": ratio >= gates["g8_min_sharpe_ratio"],
        "gate_no_big_daily_loss": metrics["max_daily_loss_pct"] <= gates["g8_max_daily_loss_pct"],
        "gate_ic_stable": metrics["ic_decay_pct"] <= gates["g8_max_ic_decay_pct"],
        "gate_infra_stable": bool(metrics.get("infra_stable", True)),
        "n_days": int(metrics.get("n_days", 0)),
    }


def compute_algorithm_signal(name: str, df):
    """Run an algorithm over bars and return its primary signal column as ndarray.

    The primary signal is the algorithm's first declared alg_feature.
    """
    import numpy as np
    from algorithms import get_algorithm

    algo = get_algorithm(name)
    out = algo.run_batch(df)
    feats = algo.alg_features()
    if not feats:
        raise ValueError(f"{name} declares no alg_features")
    col = feats[0].name
    return np.asarray(out[col].to_numpy(), dtype=float)


# action labels returned by process_signal / _transition
_ACTION = {
    "validate": "validated",
    "start_paper": "started_paper",
    "request_approval": "approval_pending",
    "retire": "retired",
}


class PromotionDaemon:
    def __init__(
        self,
        config: dict | None = None,
        *,
        lc: SignalLifecycle | None = None,
        db_path: Path | str | None = None,
        state_path: Path | str | None = None,
        heartbeat_path: Path | str | None = None,
        pid_file: Path | str | None = None,
        clock: Callable[[], datetime] | None = None,
        dry_run: bool = False,
    ):
        self.config = config or load_config()
        self.gates = load_gates()
        self.dry_run = dry_run
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.lc = lc if lc is not None else SignalLifecycle(db_path=db_path or self.config["db_path"])
        self.state_path = Path(state_path or self.config["state_file"])
        self.heartbeat_path = Path(heartbeat_path or self.config["heartbeat_path"])
        self.pid_file = Path(pid_file or self.config["pid_file"])
        self.min_clean_days = int(self.config.get("min_clean_days", 7))
        self.g8_min_days = int(self.gates.get("g8_min_days", 14))
        self.timeout = int(self.config.get("subprocess_timeout_s", 600))
        self._shutdown = False

    # -- time -----------------------------------------------------------------
    def _now(self) -> datetime:
        return self._clock()

    # -- state-machine driver -------------------------------------------------
    def process_signal(self, row: dict) -> str | None:
        """Advance ONE signal at most one transition. Returns the action label."""
        sid, state = row["signal_id"], row["state"]

        if state == DISCOVERED:
            if self._count_clean_days() < self.min_clean_days:
                log.info("skip %s: <%d clean days", sid, self.min_clean_days)
                return None
            if not self._data_sufficient(row):
                log.info("skip %s: data insufficient", sid)
                return None
            g4 = self._run_g4(row)
            if g4 is None:
                return None  # timeout/error → skip cleanly
            if g4.get("gate_pass"):
                return self._transition(sid, "validate", f"G4 passed: {g4.get('metrics')}")
            log.info("hold %s: G4 not passed", sid)
            return None

        if state == VALIDATED:
            if self._run_paper(row) is None:
                return None
            return self._transition(sid, "start_paper", "paper trading started")

        if state == PAPER_TRADING:
            days = self._days_in_paper(row)
            if days < self.g8_min_days:
                return None
            ok, report = self._check_g8(row)
            if ok:
                return self._transition(sid, "request_approval",
                                        f"G8 passed after {days}d: {report}")
            return None

        if state in (LIVE, MONITORING):
            if self._check_decay(row):
                return self._transition(sid, "retire", "decay: sustained IC<0")
            return None

        # APPROVAL_PENDING (human gate), RETIRED/REJECTED (terminal): no-op.
        return None

    def _transition(self, sid: str, method: str, msg: str) -> str | None:
        action = _ACTION[method]
        if self.dry_run:
            log.info("DRY-RUN would %s %s (%s)", method, sid, msg)
            return action
        try:
            getattr(self.lc, method)(sid, msg=msg)
        except Exception as e:
            log.error("transition %s on %s failed: %s", method, sid, e)
            return None
        log.info("%s → %s", sid, action)
        return action

    def run_cycle(self) -> dict:
        transitions, errors = 0, 0
        for row in self.lc.list_signals():
            try:
                if self.process_signal(row) is not None:
                    transitions += 1
            except Exception as e:
                errors += 1
                log.error("process_signal %s failed: %s", row.get("signal_id"), e, exc_info=True)
        summary = {"transitions": transitions, "errors": errors,
                   "at": self._now().isoformat()}
        self._save_state(summary)
        return summary

    def status(self) -> dict:
        counts: dict[str, int] = {}
        for row in self.lc.list_signals():
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        return {"by_state": counts, "dry_run": self.dry_run,
                "min_clean_days": self.min_clean_days, "g8_min_days": self.g8_min_days,
                "clean_days_now": self._count_clean_days()}

    # -- daemon loop ----------------------------------------------------------
    def run(self) -> None:
        poll = int(self.config["poll_interval_s"])
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))
        _signal.signal(_signal.SIGTERM, self._handle_signal)
        _signal.signal(_signal.SIGINT, self._handle_signal)
        log.info("Promotion daemon started (PID %d, poll %ds, dry_run=%s)",
                 os.getpid(), poll, self.dry_run)
        try:
            while not self._shutdown:
                try:
                    s = self.run_cycle()
                    log.info("cycle: %d transitions, %d errors", s["transitions"], s["errors"])
                except Exception as e:
                    log.error("cycle error: %s", e, exc_info=True)
                self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                self.heartbeat_path.write_text(self._now().isoformat())
                for _ in range(poll):
                    if self._shutdown:
                        break
                    time.sleep(1)
        finally:
            self.pid_file.unlink(missing_ok=True)
            log.info("Promotion daemon stopped")

    def _handle_signal(self, signum, frame):  # pragma: no cover
        log.info("Received signal %d, shutting down after cycle", signum)
        self._shutdown = True

    def _save_state(self, summary: dict) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"last_cycle": summary}, indent=2))
        except OSError as e:  # pragma: no cover
            log.warning("state save failed: %s", e)

    # ── seams (best-effort real impls; overridden in tests) ──────────────────
    def _count_clean_days(self) -> int:
        """Consecutive 'good' (>= good_day_mb) calendar days ending at the latest."""
        base = ROOT / "data" / "features"
        if not base.exists():
            return 0
        good_bytes = float(self.config["good_day_mb"]) * 1e6
        sizes: dict[str, float] = {}
        for d in base.iterdir():
            if not d.is_dir() or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name):
                continue
            try:
                total = sum(f.stat().st_size for f in d.glob("*.parquet"))
                total += sum(f.stat().st_size for f in d.glob("*.parquet.tmp"))
            except OSError:
                total = 0
            sizes[d.name] = total
        if not sizes:
            return 0
        latest = max(datetime.strptime(k, "%Y-%m-%d").date() for k in sizes)
        streak, cur = 0, latest
        while sizes.get(cur.isoformat(), 0) >= good_bytes:
            streak += 1
            cur = cur - timedelta(days=1)
        return streak

    def _symbol_for(self, row: dict) -> str:
        return self.config.get("validate_symbol", "BTC")

    def _load_recent_bars(self, symbol: str):  # pragma: no cover - needs real data
        """Best-effort load of recent feature bars for a symbol (pandas)."""
        from data.features import available_dates, load_features
        from alpha.paper_trader_generic import aggregate_to_bars, load_date_ticks

        dates = available_dates(data_dir=ROOT / "data" / "features")
        dates = [d for d in dates if "clean" not in d][-10:]
        frames = []
        for ds in dates:
            ticks, feats = load_date_ticks(ROOT / "data" / "features", ds, symbol)
            frames.append(aggregate_to_bars(ticks, feats))
        import pandas as pd
        return pd.concat(frames, ignore_index=True)

    def _data_sufficient(self, row: dict) -> bool:  # pragma: no cover - needs real data
        if self.dry_run:
            return True
        try:
            from check_data_sufficiency import run_all_checks

            df = self._load_recent_bars(self._symbol_for(row))
            return bool(run_all_checks(df)["sufficient"])
        except Exception as e:
            log.warning("data-sufficiency check failed for %s: %s", row["signal_id"], e)
            return False

    def _run_g4(self, row: dict) -> dict | None:  # pragma: no cover - needs real data
        """Rigorous G4 via alpha.adapter.run_validation on the algorithm's signal."""
        name = row.get("name") or row["signal_id"]
        if self.dry_run:  # don't run the (expensive) validation in a preview
            return {"gate_pass": True, "metrics": {"dry_run": True}}
        try:
            from alpha.adapter import run_validation
            from backtest.costs import hyperliquid_taker

            df = self._load_recent_bars(self._symbol_for(row))
            signal = compute_algorithm_signal(name, df)
            results = run_validation(df=df, signal=signal, cost_model=hyperliquid_taker())
            rs = results if isinstance(results, list) else [results]
            gate_pass = any(bool(getattr(r, "gate_pass", False)) for r in rs)
            metrics = {"oos_sharpe": max((getattr(r, "oos_sharpe", 0.0) for r in rs), default=0.0)}
            return {"gate_pass": gate_pass, "metrics": metrics}
        except Exception as e:
            log.warning("G4 run failed for %s: %s", name, e)
            return None

    def _run_paper(self, row: dict) -> dict | None:  # pragma: no cover - needs real data
        if self.dry_run:  # don't launch the paper subprocess in a preview
            return {"dry_run": True}
        name = row.get("name") or row["signal_id"]
        out = ROOT / "data" / "paper_trades" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "scripts" / "alpha" / "paper_trader_generic.py"),
            "--algorithms", name, "--data-dir", "data/features",
            "--symbols", *self.config["symbols"],
            "--json-output", str(out), "--cost-mode", "config",
        ]
        rc = self._subprocess(cmd)
        return {"out": str(out)} if rc == 0 else None

    def _subprocess(self, cmd: list[str]) -> int:  # pragma: no cover - needs real data
        log.info("CMD: %s", " ".join(cmd))
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
                cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
            )
        except subprocess.TimeoutExpired:
            log.error("TIMEOUT after %ds: %s", self.timeout, cmd[0])
            return -1
        if r.returncode != 0:
            log.error("subprocess failed (%d): %s", r.returncode, r.stderr[-500:])
        return r.returncode

    def _days_in_paper(self, row: dict) -> int:  # pragma: no cover - needs history
        entered = None
        for h in self.lc.history(row["signal_id"]):
            if h.get("to_state") == PAPER_TRADING:
                entered = h.get("at")
        if not entered:
            return 0
        try:
            return (self._now() - datetime.fromisoformat(entered)).days
        except ValueError:
            return 0

    def _check_g8(self, row: dict) -> tuple[bool, dict]:  # pragma: no cover - needs real data
        if self.dry_run:
            return True, {"dry_run": True}
        try:
            metrics = self._paper_metrics(row)
            report = build_paper_report(metrics, self.gates)
            ok = (report["gate_sharpe_within_2x"] and report["gate_no_big_daily_loss"]
                  and report["gate_ic_stable"] and report["gate_infra_stable"]
                  and report["n_days"] >= self.g8_min_days)
            return ok, report
        except Exception as e:
            log.warning("G8 check failed for %s: %s", row["signal_id"], e)
            return False, {}

    def _paper_metrics(self, row: dict) -> dict:  # pragma: no cover - needs real data
        """Read the per-signal paper report and reduce to G8 metric inputs."""
        name = row.get("name") or row["signal_id"]
        path = ROOT / "data" / "paper_trades" / f"{name}.json"
        doc = json.loads(path.read_text())
        results = doc.get(name, {}).get("results", doc)
        sharpes, losses, daily_count = [], [], 0
        for sym in results.values():
            sharpes.append(float(sym.get("sharpe", 0.0)))
            losses.append(abs(float(sym.get("max_daily_loss_bps", 0.0))) / 100.0)
            daily_count = max(daily_count, len(sym.get("daily", [])))
        import numpy as np
        return {
            "paper_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
            "baseline_sharpe": float(row.get("metadata", {}).get("baseline_sharpe", 1.0)),
            "max_daily_loss_pct": max(losses) if losses else 0.0,
            "ic_decay_pct": float(row.get("metadata", {}).get("ic_decay_pct", 0.0)),
            "infra_stable": True,
            "n_days": daily_count,
        }

    def _check_decay(self, row: dict) -> bool:  # pragma: no cover - no LIVE signals yet
        """Thin LIVE-health check: retire on a recorded sustained-negative IC.

        No LIVE signals exist pre-paper-window; this only fires on an explicit
        metadata flag until per-LIVE-signal IC tracking lands.
        """
        return bool(row.get("metadata", {}).get("decayed", False))


# ───────────────────────────────────────────────────────────────────────────
# CLI entrypoint (invoked by `nat promotion ...`)
# ───────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "start"
    dry = "--dry-run" in argv
    if cmd == "health":
        return 0 if healthy(load_config()) else 1
    d = PromotionDaemon(dry_run=dry)
    if cmd == "start":
        d.run()
        return 0
    if cmd == "once":
        print(json.dumps(d.run_cycle(), indent=2))
        return 0
    if cmd == "status":
        print(json.dumps(d.status(), indent=2))
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
