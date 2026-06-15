"""T16 — Kill-switch daemon.

An independent process that polls realised PnL / IC every 60s and halts trading
when a risk threshold is breached. Thresholds are **imported, not invented**:
they come from ROADMAP Step 9 via :func:`alpha.deployer.evaluate_kill_switches`
(re-exported here) and must never be re-encoded in this file.

    daily loss  > 1%   -> halt_24h       auto-resume after 24h
    weekly DD   > 2%   -> halt_review     manual `nat risk resume --confirm`
    monthly DD  > 5%   -> kill_strategy   terminal; full pipeline re-run required
    IC < 0 for 5 days  -> halt            manual resume after investigation

Halt state is published atomically to ``data/risk/halt_state.json`` — the IPC
contract the signal bridge (T17) reads before every cycle. A provenance-stamped
audit line is appended to ``data/risk/halt_history.jsonl`` on every transition.

Data sources (decision: prefer the execution log, fall back to paper trades):
  * ``data/execution/daily_pnl.json``      — bridge output, already in percent
  * ``data/paper_trades/batch_report.json`` — paper trader, basis points -> percent

The IC gate uses a *real* Spearman rolling IC (signal vs forward return) computed
per day from the paper-trade records, matching ROADMAP semantics — not a
negative-PnL proxy.

CLI: ``nat risk status|resume [--confirm]|start|stop``.
"""

from __future__ import annotations

import json
import os
import signal as _signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats

# `nat risk start` runs this file directly, so sys.path[0] is scripts/risk/.
# Put the scripts/ root on the path so sibling packages (alpha, signal_lifecycle,
# tournament, logging_config) resolve regardless of how the daemon is launched —
# critical so the best-effort Telegram/lifecycle imports below don't silently
# no-op, which would make a halt fire without an alert.
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# Thresholds are imported from the single source of truth — see module docstring.
from alpha.deployer import evaluate_kill_switches  # noqa: F401,E402  (re-exported)

ROOT = _SCRIPTS_ROOT.parent

# ── logging (best-effort centralised setup) ────────────────────────────────
try:
    from logging_config import setup_logging, set_context

    log = setup_logging("nat.risk")
except Exception:  # pragma: no cover - fallback if logging_config unavailable
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("nat.risk")

    def set_context(**kwargs):  # type: ignore
        pass


# ── action severity & naming ───────────────────────────────────────────────
# deployer emits action="kill" for monthly DD; the spec/ROADMAP calls it
# "kill_strategy". Normalise to the spec name everywhere we persist state.
_ACTION_MAP = {"kill": "kill_strategy"}
_SEVERITY = {"halt_24h": 1, "halt": 2, "halt_review": 3, "kill_strategy": 4}
# Levels that auto-resume vs require a human:
_AUTO_RESUME_HOURS = {"halt_24h": 24}
_MANUAL_LEVELS = {"halt_review", "halt"}
_TERMINAL_LEVELS = {"kill_strategy"}


def _norm_action(action: str) -> str:
    return _ACTION_MAP.get(action, action)


def effective_level(levels: Iterable[str]) -> str | None:
    """Return the most severe level among *levels* (already spec-named), or None."""
    best = None
    best_sev = -1
    for lv in levels:
        sev = _SEVERITY.get(lv, 0)
        if sev > best_sev:
            best, best_sev = lv, sev
    return best


# ───────────────────────────────────────────────────────────────────────────
# Pure estimator helpers (planted-test pinned)
# ───────────────────────────────────────────────────────────────────────────


def max_drawdown_pct(returns: Sequence[float]) -> float:
    """Window-relative peak-to-trough drawdown (%), always >= 0.

    The equity curve is seeded at 0 (the window's opening level), so a window
    that only loses money reports a drawdown equal to its total loss. NaN/None
    entries are skipped.
    """
    arr = np.asarray(
        [r for r in returns if r is not None and np.isfinite(r)], dtype=float
    )
    if arr.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(arr)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def consecutive_negative(values: Sequence[float]) -> int:
    """Count trailing consecutive strictly-negative values.

    A NaN/None (missing measurement) breaks the streak — we do not treat absent
    data as a negative day.
    """
    count = 0
    for v in reversed(list(values)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            break
        if v < 0:
            count += 1
        else:
            break
    return count


def daily_ic(
    signal: Sequence[float], forward_return: Sequence[float], min_obs: int = 10
) -> float:
    """Spearman rank IC between *signal* and *forward_return* for one day.

    Matches the house method in ``alpha.screener.compute_rolling_ic``: drops NaN
    pairs, returns NaN below ``min_obs``, and guards constant arrays to 0.0.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_return, dtype=float)
    valid = ~(np.isnan(s) | np.isnan(r))
    s, r = s[valid], r[valid]
    if s.size < min_obs:
        return float("nan")
    if np.std(s) < 1e-15 or np.std(r) < 1e-15:
        return 0.0
    rho, _ = stats.spearmanr(s, r)
    return float(rho) if np.isfinite(rho) else float("nan")


def compute_kill_metrics(
    pnl_history: list[dict], ic_series: Sequence[float] | None = None
) -> dict:
    """Reduce raw PnL/IC history to the four metrics evaluate_kill_switches wants.

    All PnL values are percent. Returns kwargs for evaluate_kill_switches().
    """
    rets = [float(d.get("pnl_pct", 0.0)) for d in pnl_history]
    return {
        "daily_pnl_pct": rets[-1] if rets else 0.0,
        "weekly_dd_pct": max_drawdown_pct(rets[-7:]),
        "monthly_dd_pct": max_drawdown_pct(rets[-30:]),
        "ic_negative_days": consecutive_negative(list(ic_series or [])),
    }


# ───────────────────────────────────────────────────────────────────────────
# Halt-state persistence (the IPC contract T17 reads)
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class HaltState:
    halted: bool = False
    level: str | None = None
    reason: str = ""
    triggered: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    halted_at: str | None = None
    resume_at: str | None = None
    git_sha: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def read_halt_state(path: Path | str) -> HaltState:
    """Read halt state; an absent file means *not halted*."""
    p = Path(path)
    if not p.exists():
        return HaltState()
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return HaltState()
    fields = HaltState().to_dict().keys()
    return HaltState(**{k: data.get(k) for k in fields})


def write_halt_state(path: Path | str, state: HaltState) -> None:
    """Atomically publish halt state (write tmp + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    os.replace(tmp, p)


def _git_sha() -> str | None:
    try:
        from signal_lifecycle import _provenance

        return _provenance().get("git_sha")
    except Exception:
        return None


def healthy(config: dict | None = None, now: float | None = None) -> bool:
    """True if the daemon's heartbeat is fresh (< 3 poll intervals, min 180s).

    Used by the docker-compose healthcheck (`kill_switch.py health`) to detect a
    *hung* daemon, not merely a live process.
    """
    cfg = config or load_config()
    hb = Path(cfg["heartbeat_path"])
    if not hb.exists():
        return False
    now = now if now is not None else time.time()
    max_age = max(180, 3 * int(cfg.get("poll_interval_s", 60)))
    return (now - hb.stat().st_mtime) < max_age


# ───────────────────────────────────────────────────────────────────────────
# Data loaders
# ───────────────────────────────────────────────────────────────────────────


def load_config(path: Path | str | None = None) -> dict:
    """Load risk config (paths + poll interval). Thresholds are NOT here."""
    defaults = {
        "poll_interval_s": 60,
        "halt_state_path": str(ROOT / "data" / "risk" / "halt_state.json"),
        "audit_path": str(ROOT / "data" / "risk" / "halt_history.jsonl"),
        "exec_log": str(ROOT / "data" / "execution" / "daily_pnl.json"),
        "paper_report": str(ROOT / "data" / "paper_trades" / "batch_report.json"),
        "paper_trades_dir": str(ROOT / "data" / "paper_trades"),
        # OFF by default: paper total_net_bps is a sum of per-trade returns, not
        # a portfolio daily return; enabling it false-trips the drawdown gates.
        "use_paper_fallback": False,
        "db_path": str(ROOT / "data" / "nat.db"),
        "ic_min_obs": 10,
        "prometheus_port": 0,  # 0 disables the metrics HTTP server
        "pid_file": str(ROOT / "data" / "risk" / "kill_switch.pid"),
        "heartbeat_path": str(ROOT / "data" / "risk" / "kill_switch.heartbeat"),
    }
    cfg_path = Path(path) if path else ROOT / "config" / "risk.toml"
    if cfg_path.exists():
        try:
            import tomllib

            with open(cfg_path, "rb") as f:
                doc = tomllib.load(f)
            defaults.update(doc.get("risk", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read %s: %s — using defaults", cfg_path, e)
    return defaults


def load_pnl_history(
    exec_log: Path | str | None = None, paper_report: Path | str | None = None
) -> list[dict]:
    """Daily PnL history as ``[{date, pnl_pct}]`` sorted by date.

    Prefers the execution log (already a portfolio daily return, in percent).

    The paper-trade batch report is consulted ONLY when ``paper_report`` is
    passed explicitly — it is not the daemon's default source. Its
    ``total_net_bps`` is the *sum* of per-trade returns, not a portfolio daily
    return, so feeding it to the drawdown gates overstates losses and false-trips
    the kill-switch. The daemon gates this behind ``use_paper_fallback`` (default
    off); see :meth:`KillSwitch.check`.
    """
    cfg = load_config()
    exec_log = Path(exec_log or cfg["exec_log"])

    if exec_log.exists():
        try:
            data = json.loads(exec_log.read_text())
            return sorted(
                ({"date": d.get("date"), "pnl_pct": float(d.get("pnl_pct", 0.0))}
                 for d in data),
                key=lambda d: d["date"] or "",
            )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            log.warning("exec log unreadable (%s); falling back to paper trades", e)

    if paper_report is not None and Path(paper_report).exists():
        paper_report = Path(paper_report)
        try:
            doc = json.loads(paper_report.read_text())
            bps_by_date: dict[str, float] = {}
            for sym_block in doc.get("results", {}).values():
                for d in sym_block.get("daily", []):
                    date = d.get("date")
                    if date is None:
                        continue
                    bps_by_date[date] = bps_by_date.get(date, 0.0) + float(
                        d.get("total_net_bps", 0.0)
                    )
            return [
                {"date": date, "pnl_pct": bps / 100.0}  # 100 bps == 1%
                for date, bps in sorted(bps_by_date.items())
            ]
        except (json.JSONDecodeError, OSError, ValueError) as e:
            log.warning("paper report unreadable: %s", e)

    return []


def load_daily_ic_series(
    paper_trades_dir: Path | str | None = None, min_obs: int = 10
) -> list[float]:
    """Per-day Spearman IC (signal vs forward return) from paper-trade records.

    Reads ``{date}_{SYMBOL}.json`` files, pools all symbols' trades per day, and
    computes daily_ic over (signal_value, forward_return) where
    forward_return = (exit_price - entry_price) / entry_price * 1e4 (bps).
    Best-effort: returns ``[]`` if the directory is missing.
    """
    cfg = load_config()
    d = Path(paper_trades_dir or cfg["paper_trades_dir"])
    if not d.exists():
        return []
    by_date: dict[str, tuple[list[float], list[float]]] = {}
    for f in d.glob("*_*.json"):
        if f.name == "batch_report.json":
            continue
        try:
            trades = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(trades, list):
            continue
        for t in trades:
            date = t.get("date")
            entry = t.get("entry_price")
            exit_p = t.get("exit_price")
            sig = t.get("signal_value")
            if date is None or not entry or exit_p is None or sig is None:
                continue
            fwd = (exit_p - entry) / entry * 1e4
            sigs, fwds = by_date.setdefault(date, ([], []))
            sigs.append(float(sig))
            fwds.append(float(fwd))
    return [
        daily_ic(sigs, fwds, min_obs=min_obs)
        for _date, (sigs, fwds) in sorted(by_date.items())
    ]


# ───────────────────────────────────────────────────────────────────────────
# Prometheus (best-effort, optional)
# ───────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - exercised only when prometheus_client is installed
    from prometheus_client import Counter, Gauge, start_http_server

    _PROM_ACTIVE = Gauge(
        "nat_kill_switch_active", "Kill-switch active (1) per level", ["level"]
    )
    _PROM_TRIGGERS = Counter(
        "nat_kill_switch_triggers_total", "Kill-switch triggers", ["name"]
    )
    _PROM = True
except Exception:  # pragma: no cover
    _PROM = False
    start_http_server = None  # type: ignore


# ───────────────────────────────────────────────────────────────────────────
# Controller / daemon
# ───────────────────────────────────────────────────────────────────────────


class KillSwitch:
    """Evaluates kill-switch thresholds, publishes halt state, drives resume."""

    def __init__(
        self,
        config: dict | None = None,
        *,
        halt_path: Path | str | None = None,
        audit_path: Path | str | None = None,
        db_path: Path | str | None = "__default__",
        clock: Callable[[], datetime] | None = None,
        notify: bool = True,
    ):
        self.config = config or load_config()
        self.halt_path = Path(halt_path or self.config["halt_state_path"])
        self.audit_path = Path(audit_path or self.config["audit_path"])
        # db_path=None explicitly disables lifecycle retirement (used by tests);
        # the sentinel means "use the configured default".
        self.db_path = (
            self.config["db_path"] if db_path == "__default__" else db_path
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.notify = notify
        self._shutdown = False

    # -- time ---------------------------------------------------------------
    def _now(self) -> datetime:
        return self._clock()

    @staticmethod
    def _parse(ts: str) -> datetime:
        return datetime.fromisoformat(ts)

    # -- evaluation ---------------------------------------------------------
    def check(
        self,
        pnl_history: list[dict] | None = None,
        ic_series: Sequence[float] | None = None,
    ) -> HaltState:
        """One evaluation cycle. Loads data if not supplied; returns current state."""
        if pnl_history is None:
            # Paper-trade fallback is OFF by default — its total_net_bps is a
            # sum of per-trade returns, not a portfolio daily return, and would
            # false-trip the drawdown gates. Pre-T17 (no exec log) this means
            # no data -> no halt, which is correct: nothing is trading at the
            # portfolio level to protect yet.
            paper_report = (
                self.config["paper_report"]
                if self.config.get("use_paper_fallback")
                else None
            )
            pnl_history = load_pnl_history(self.config["exec_log"], paper_report)
        if ic_series is None:
            ic_series = load_daily_ic_series(
                self.config["paper_trades_dir"], self.config.get("ic_min_obs", 10)
            )

        metrics = compute_kill_metrics(pnl_history, ic_series)
        switches = evaluate_kill_switches(**metrics)
        triggered = [s for s in switches if s.triggered]
        new_level = effective_level(_norm_action(s.action) for s in triggered)

        now = self._now()
        prev = read_halt_state(self.halt_path)

        # 1. Auto-resume an expired halt_24h cooldown.
        if (
            prev.halted
            and prev.level == "halt_24h"
            and prev.resume_at
            and now >= self._parse(prev.resume_at)
        ):
            self._clear(prev, "auto-resume (24h cooldown elapsed)", now)
            prev = HaltState()

        # 2. Nothing newly triggered.
        if new_level is None:
            return prev  # halted manual/cooldown states stay; else not halted

        # 3. Already halted at >= severity — keep (never downgrade).
        if prev.halted and _SEVERITY.get(prev.level, 0) >= _SEVERITY[new_level]:
            return prev

        # 4. New halt or escalation.
        return self._raise(new_level, triggered, metrics, now)

    # -- transitions --------------------------------------------------------
    def _raise(self, level, triggered, metrics, now) -> HaltState:
        names = [s.name for s in triggered if _norm_action(s.action) == level] or [
            s.name for s in triggered
        ]
        reason = "; ".join(
            f"{s.name}={s.current_value:.3g} (>{s.threshold:g})" for s in triggered
        )
        resume_at = None
        if level in _AUTO_RESUME_HOURS:
            resume_at = (now + timedelta(hours=_AUTO_RESUME_HOURS[level])).isoformat()
        st = HaltState(
            halted=True,
            level=level,
            reason=reason,
            triggered=names,
            metrics=metrics,
            halted_at=now.isoformat(),
            resume_at=resume_at,
            git_sha=_git_sha(),
        )
        write_halt_state(self.halt_path, st)
        self._audit("halt", st, now)
        log.error("KILL-SWITCH HALT [%s]: %s", level, reason)

        if _PROM:  # pragma: no cover
            for lv in _SEVERITY:
                _PROM_ACTIVE.labels(level=lv).set(1.0 if lv == level else 0.0)
            for n in names:
                _PROM_TRIGGERS.labels(name=n).inc()

        if self.notify:
            self._telegram(
                f"🛑 *Kill-switch HALT* — `{level}`\n{reason}\n"
                f"at {st.halted_at}"
                + (f"\nauto-resume {resume_at}" if resume_at else "")
            )

        if level == "kill_strategy":
            self._retire_signals(reason)
        return st

    def _clear(self, prev: HaltState, reason: str, now: datetime) -> None:
        write_halt_state(self.halt_path, HaltState(halted=False))
        self._audit("resume", HaltState(halted=False, reason=reason, level=prev.level), now)
        log.info("Kill-switch resumed (%s): %s", prev.level, reason)
        if _PROM:  # pragma: no cover
            for lv in _SEVERITY:
                _PROM_ACTIVE.labels(level=lv).set(0.0)
        if self.notify:
            self._telegram(f"✅ *Kill-switch resumed* (was `{prev.level}`)\n{reason}")

    def resume(self, confirm: bool = False, now: datetime | None = None) -> tuple[bool, str]:
        """Apply resume rules. Returns (ok, message)."""
        st = read_halt_state(self.halt_path)
        if not st.halted:
            return False, "Trading is not halted — nothing to resume."
        if st.level in _TERMINAL_LEVELS:
            return (
                False,
                "kill_strategy is terminal: re-run the full pipeline from Step 1 "
                "before resuming. The CLI will not clear it.",
            )
        if st.level in _MANUAL_LEVELS and not confirm:
            return False, f"{st.level} requires manual confirmation — pass --confirm."
        if st.level == "halt_24h" and not confirm:
            return (
                False,
                f"halt_24h auto-resumes at {st.resume_at}; pass --confirm to clear now.",
            )
        self._clear(st, "manual resume via `nat risk resume --confirm`", now or self._now())
        return True, f"Resumed (cleared {st.level})."

    def get_status(self) -> dict:
        st = read_halt_state(self.halt_path)
        d = st.to_dict()
        d["pnl_source"] = (
            "exec_log" if Path(self.config["exec_log"]).exists() else "paper_trades"
        )
        return d

    # -- side effects (all best-effort) -------------------------------------
    def _telegram(self, message: str) -> None:
        try:
            from tournament.report import send_telegram

            send_telegram(message)
        except Exception as e:  # pragma: no cover
            log.debug("Telegram send skipped: %s", e)

    def _retire_signals(self, reason: str) -> None:
        if not self.db_path:
            return
        try:
            from signal_lifecycle import LIVE, MONITORING, SignalLifecycle

            lc = SignalLifecycle(db_path=self.db_path)
            for sig in lc.list_signals():
                if sig.get("state") in (LIVE, MONITORING):
                    lc.retire(sig["signal_id"], reason=f"kill_strategy: {reason}")
                    log.error("Retired LIVE signal %s (kill_strategy)", sig["signal_id"])
        except Exception as e:  # pragma: no cover
            log.warning("Lifecycle retirement skipped: %s", e)

    def _audit(self, event: str, st: HaltState, now: datetime) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "event": event,
            "at": now.isoformat(),
            "level": st.level,
            "reason": st.reason,
            "metrics": st.metrics,
            "git_sha": st.git_sha or _git_sha(),
        }
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- daemon -------------------------------------------------------------
    def run(self) -> None:
        poll = int(self.config["poll_interval_s"])
        pid_file = Path(self.config["pid_file"])
        heartbeat = Path(self.config["heartbeat_path"])
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))

        _signal.signal(_signal.SIGTERM, self._handle_signal)
        _signal.signal(_signal.SIGINT, self._handle_signal)

        port = int(self.config.get("prometheus_port", 0) or 0)
        if _PROM and port and start_http_server:  # pragma: no cover
            start_http_server(port)
            log.info("Prometheus metrics on :%d", port)

        log.info("Kill-switch daemon started (PID %d, poll %ds)", os.getpid(), poll)
        try:
            while not self._shutdown:
                try:
                    st = self.check()
                    if st.halted:
                        log.warning("HALT active: %s (%s)", st.level, st.reason)
                except Exception as e:
                    log.error("Kill-switch cycle error: %s", e, exc_info=True)
                # Heartbeat after every cycle (even on error) — the healthcheck
                # reads its freshness to distinguish a live daemon from a hung one.
                heartbeat.write_text(self._now().isoformat())
                for _ in range(poll):
                    if self._shutdown:
                        break
                    time.sleep(1)
        finally:
            pid_file.unlink(missing_ok=True)
            log.info("Kill-switch daemon stopped")

    def _handle_signal(self, signum, frame):  # pragma: no cover
        log.info("Received signal %d, shutting down after cycle", signum)
        self._shutdown = True


# ───────────────────────────────────────────────────────────────────────────
# CLI entrypoint (invoked by `nat risk start`)
# ───────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "start"
    ks = KillSwitch()
    if cmd == "start":
        ks.run()
        return 0
    if cmd == "status":
        print(json.dumps(ks.get_status(), indent=2))
        return 0
    if cmd == "resume":
        ok, msg = ks.resume(confirm="--confirm" in argv)
        print(msg)
        return 0 if ok else 1
    if cmd == "health":
        # Quiet, exit-code only — consumed by the docker-compose healthcheck.
        return 0 if healthy(ks.config) else 1
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
