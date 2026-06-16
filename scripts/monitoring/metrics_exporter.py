"""T18 — Prometheus metrics exporter (keystone for the Grafana dashboards).

Grafana can't scrape SQLite/JSON directly, so this exporter turns NAT's state
into Prometheus gauges, refreshed on an interval:

  * lifecycle funnel  — signal_lifecycle state counts (data/nat.db)
  * live P&L          — data/execution/daily_pnl.json (the bridge's rollup)
  * paper performance — per-signal sharpe/maxDD from data/oos_validation/state.json

The reductions (`lifecycle_counts`, `live_pnl`, `paper_metrics`, `collect`) are
pure and unit-tested without prometheus_client. The exposition layer
(`serve`: start_http_server + refresh loop + heartbeat) is guarded glue.

CLI: `once` (print a snapshot) | `start` (serve) | `health`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

ROOT = _SCRIPTS_ROOT.parent

try:
    from logging_config import setup_logging

    log = setup_logging("nat.metrics_exporter")
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("nat.metrics_exporter")


def load_config(path: Path | str | None = None) -> dict:
    defaults = {
        "port": 9094,
        "refresh_s": 15,
        "db_path": str(ROOT / "data" / "nat.db"),
        "daily_pnl_path": str(ROOT / "data" / "execution" / "daily_pnl.json"),
        "oos_state_path": str(ROOT / "data" / "oos_validation" / "state.json"),
        "heartbeat_path": str(ROOT / "data" / "monitoring" / "metrics_exporter.heartbeat"),
    }
    cfg_path = Path(path) if path else ROOT / "config" / "monitoring.toml"
    if cfg_path.exists():
        try:
            import tomllib

            with open(cfg_path, "rb") as f:
                defaults.update(tomllib.load(f).get("metrics_exporter", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read %s: %s — using defaults", cfg_path, e)
    return defaults


# ── reductions (pure, testable) ─────────────────────────────────────────────


def lifecycle_counts(db_path: Path | str) -> dict:
    """Signal counts by lifecycle state, via raw sqlite3 (no ORM/pandas import).

    {} if the DB or the signal_lifecycle table is absent/unreadable.
    """
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        return {}
    try:
        conn = sqlite3.connect(str(p))
        try:
            rows = conn.execute(
                "SELECT state, COUNT(*) FROM signal_lifecycle GROUP BY state"
            ).fetchall()
            return {state: count for state, count in rows}
        finally:
            conn.close()
    except sqlite3.Error as e:
        log.warning("lifecycle_counts failed: %s", e)
        return {}


def live_pnl(daily_pnl_path: Path | str) -> dict:
    """Aggregate the bridge's daily_pnl.json ([{date, pnl_pct}]). Zeros if absent."""
    zero = {"cum_pnl_pct": 0.0, "last_daily_pnl_pct": 0.0, "n_days": 0}
    p = Path(daily_pnl_path)
    if not p.exists():
        return zero
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return zero
    if not isinstance(data, list) or not data:
        return zero
    pnls = [float(d.get("pnl_pct", 0.0)) for d in data]
    return {
        "cum_pnl_pct": round(sum(pnls), 6),
        "last_daily_pnl_pct": pnls[-1],
        "n_days": len(pnls),
    }


def paper_metrics(oos_state_path: Path | str | None = None) -> dict:
    """Per-signal {sharpe, max_dd_bps, n_days} from the OOS-validation state JSON.

    Read directly (no viz dependency) — monitoring must not pull the terminal-viz
    rendering stack (pandas/matplotlib). Mirrors viz.approval.per_signal_risk's
    metric extraction.
    """
    p = Path(oos_state_path) if oos_state_path else (ROOT / "data" / "oos_validation" / "state.json")
    if not p.exists():
        return {}
    try:
        state = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, dict] = {}
    for name, blk in state.get("algos", {}).items():
        syms = blk.get("symbols", {})
        sym = syms.get("BTC") or next(iter(syms.values()), None)
        if not sym:
            continue
        m = sym.get("metrics", {})
        out[name] = {
            "sharpe": float(m.get("current_sharpe", 0.0)),
            "max_dd_bps": float(m.get("max_drawdown_bps", 0.0)),
            "n_days": int(m.get("n_days", 0)),
        }
    return out


def collect(config: dict | None = None) -> dict:
    cfg = config or load_config()
    return {
        "lifecycle": lifecycle_counts(cfg["db_path"]),
        "live_pnl": live_pnl(cfg["daily_pnl_path"]),
        "paper": paper_metrics(cfg.get("oos_state_path")),
    }


def healthy(config: dict | None = None, now: float | None = None) -> bool:
    cfg = config or load_config()
    hb = Path(cfg["heartbeat_path"])
    if not hb.exists():
        return False
    now = now if now is not None else time.time()
    return (now - hb.stat().st_mtime) < max(180, 3 * int(cfg.get("refresh_s", 15)))


# ── exposition (guarded — needs prometheus_client) ──────────────────────────


def _gauges():  # pragma: no cover - needs prometheus_client
    from prometheus_client import Gauge

    return {
        "lifecycle": Gauge("nat_lifecycle_signals", "Signals by lifecycle state", ["state"]),
        "cum_pnl": Gauge("nat_live_cum_pnl_pct", "Cumulative live P&L (percent)"),
        "last_pnl": Gauge("nat_live_last_daily_pnl_pct", "Most recent daily P&L (percent)"),
        "pnl_days": Gauge("nat_live_pnl_days", "Days of live P&L history"),
        "paper_sharpe": Gauge("nat_paper_sharpe", "Per-signal OOS/paper sharpe", ["signal"]),
        "paper_maxdd": Gauge("nat_paper_max_drawdown_bps", "Per-signal max drawdown (bps)", ["signal"]),
    }


def _apply(g, snap: dict) -> None:  # pragma: no cover - needs prometheus_client
    for state, n in snap["lifecycle"].items():
        g["lifecycle"].labels(state=state).set(n)
    g["cum_pnl"].set(snap["live_pnl"]["cum_pnl_pct"])
    g["last_pnl"].set(snap["live_pnl"]["last_daily_pnl_pct"])
    g["pnl_days"].set(snap["live_pnl"]["n_days"])
    for sig, m in snap["paper"].items():
        g["paper_sharpe"].labels(signal=sig).set(m["sharpe"])
        g["paper_maxdd"].labels(signal=sig).set(m["max_dd_bps"])


def serve(config: dict | None = None) -> None:  # pragma: no cover - needs prometheus_client
    import signal as _signal

    from prometheus_client import start_http_server

    cfg = config or load_config()
    g = _gauges()
    start_http_server(int(cfg["port"]))
    hb = Path(cfg["heartbeat_path"])
    hb.parent.mkdir(parents=True, exist_ok=True)
    refresh = int(cfg["refresh_s"])
    state = {"stop": False}

    def _handle(signum, frame):
        log.info("Received signal %d, stopping exporter", signum)
        state["stop"] = True

    _signal.signal(_signal.SIGTERM, _handle)
    _signal.signal(_signal.SIGINT, _handle)
    log.info("Metrics exporter on :%d (refresh %ds)", cfg["port"], refresh)
    while not state["stop"]:
        try:
            _apply(g, collect(cfg))
        except Exception as e:
            log.error("exporter refresh failed: %s", e, exc_info=True)
        hb.write_text(str(time.time()))
        for _ in range(refresh):
            if state["stop"]:
                break
            time.sleep(1)
    log.info("Metrics exporter stopped")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "start"
    if cmd == "health":
        return 0 if healthy(load_config()) else 1
    if cmd == "once":
        print(json.dumps(collect(load_config()), indent=2))
        return 0
    if cmd == "start":
        serve(load_config())
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
