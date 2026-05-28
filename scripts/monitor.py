"""
NAT Unified Terminal Monitor

Three-tab auto-refreshing dashboard:
  Tab 1 (Health): System components + OOS validation
  Tab 2 (Agent):  Research cycles, registry, generators, failures
  Tab 3 (Features): Live feature snapshot per symbol (requires Redis)

Usage:
  python scripts/monitor.py              # Default: health tab, 5s refresh
  python scripts/monitor.py --tab 2      # Start on agent tab
  python scripts/monitor.py --no-redis   # Skip Redis features tab
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent

from oos_terminal import (
    load_state as load_oos_state,
    render_performance_table,
    render_sparklines,
    render_alerts,
    sparkline,
    status_label,
    STATE_FILE as OOS_STATE_FILE,
)

DB_PATH = ROOT / "data" / "nat.db"
FEATURES_DIR = ROOT / "data" / "features"
AGENTS = ["microstructure", "medium_freq", "macro"]
SYMBOLS = ["BTC", "ETH", "SOL"]

# ── Globals ─────────────────────────────────────────────────────────────

_current_tab = 1
_shutdown = False


# ── System Health Polling ───────────────────────────────────────────────

def poll_system_health() -> dict:
    """Check component status: ingestor, redis, api, data freshness."""
    health = {}

    # Ingestor process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "target/release/ing"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n") if result.stdout.strip() else []
        if pids and pids[0]:
            health["ingestor"] = ("OK", f"PID {pids[0]}")
        else:
            health["ingestor"] = ("DOWN", "process not found")
    except Exception:
        health["ingestor"] = ("UNKNOWN", "pgrep failed")

    # Data freshness
    try:
        dates = sorted(d.name for d in FEATURES_DIR.iterdir()
                       if d.is_dir() and len(d.name) == 10)
        if dates:
            latest_dir = FEATURES_DIR / dates[-1]
            parquets = sorted(latest_dir.glob("*.parquet"))
            if parquets:
                age_s = time.time() - parquets[-1].stat().st_mtime
                if age_s < 120:
                    health["data"] = ("OK", f"{len(dates)} days, fresh ({int(age_s)}s ago)")
                else:
                    health["data"] = ("STALE", f"last write {int(age_s)}s ago")
            else:
                health["data"] = ("WARN", f"{len(dates)} days, no parquets today")
        else:
            health["data"] = ("DOWN", "no data directories")
    except Exception as e:
        health["data"] = ("ERROR", str(e)[:60])

    # Redis
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_timeout=2)
        r.ping()
        keys = r.keys("nat:latest:*")
        health["redis"] = ("OK", f"{len(keys)} feature keys")
    except ImportError:
        health["redis"] = ("N/A", "redis-py not installed")
    except Exception:
        health["redis"] = ("DOWN", "connection refused")

    # API server
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:3000/health", timeout=2)
        health["api"] = ("OK", "port 3000")
    except Exception:
        health["api"] = ("DOWN", "port 3000 not responding")

    # Agent processes
    for agent_name in AGENTS:
        short = agent_name.split("_")[0] if "_" in agent_name else agent_name[:5]
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"{agent_name.replace('_', '.')}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                health[f"agent_{short}"] = ("RUNNING", f"PID {result.stdout.strip().split()[0]}")
            else:
                health[f"agent_{short}"] = ("OFF", "not running")
        except Exception:
            health[f"agent_{short}"] = ("UNKNOWN", "")

    return health


# ── Agent Data Polling ──────────────────────────────────────────────────

def poll_agent_data() -> dict:
    """Read cycles, registry, generator stats, failure breakdown from SQLite."""
    if not DB_PATH.exists():
        return {"error": "nat.db not found"}

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    data = {}

    # Recent cycles
    try:
        rows = conn.execute(
            "SELECT payload FROM research_output WHERE kind = 'cycle' "
            "ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        data["cycles"] = [json.loads(r["payload"]) for r in rows]
    except Exception:
        data["cycles"] = []

    # Registry
    try:
        rows = conn.execute("SELECT * FROM registry ORDER BY expected_ic DESC").fetchall()
        signals = []
        for r in rows:
            d = dict(r)
            for col in ("features", "symbols", "correlation_with", "ic_history"):
                if d.get(col):
                    try:
                        d[col] = json.loads(d[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            signals.append(d)
        data["registry"] = signals
    except Exception:
        data["registry"] = []

    # Generator stats
    try:
        rows = conn.execute(
            "SELECT agent, generator, attempts, successes FROM generator_stats"
        ).fetchall()
        stats = {}
        for r in rows:
            key = r["generator"]
            stats[key] = {
                "agent": r["agent"],
                "attempts": r["attempts"],
                "successes": r["successes"],
                "weight": (r["successes"] + 1) / (r["attempts"] + 2),
            }
        data["gen_stats"] = stats
    except Exception:
        data["gen_stats"] = {}

    # Failure breakdown
    try:
        rows = conn.execute(
            "SELECT failure_reason, COUNT(*) as cnt FROM hypotheses "
            "WHERE status = 'failed' AND failure_reason IS NOT NULL "
            "GROUP BY failure_reason ORDER BY cnt DESC LIMIT 8"
        ).fetchall()
        data["failures"] = [(r["failure_reason"], r["cnt"]) for r in rows]
    except Exception:
        data["failures"] = []

    # Total counts
    try:
        total = conn.execute("SELECT COUNT(*) as c FROM hypotheses").fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) as c FROM hypotheses WHERE status = 'failed'"
        ).fetchone()["c"]
        registered = conn.execute("SELECT COUNT(*) as c FROM registry").fetchone()["c"]
        data["totals"] = {"total": total, "failed": failed, "registered": registered}
    except Exception:
        data["totals"] = {"total": 0, "failed": 0, "registered": 0}

    conn.close()
    return data


# ── Feature Polling ─────────────────────────────────────────────────────

def poll_features() -> dict:
    """Read latest feature snapshots from Redis."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_timeout=2, decode_responses=True)
        features = {}
        for sym in SYMBOLS:
            raw = r.get(f"nat:latest:{sym}")
            if raw:
                features[sym] = json.loads(raw)
        return features
    except Exception:
        return {}


# ── Rendering ───────────────────────────────────────────────────────────

def _status_style(status: str) -> str:
    if status in ("OK", "RUNNING"):
        return "bold green"
    if status in ("STALE", "WARN", "WATCH"):
        return "bold yellow"
    if status in ("DOWN", "ERROR", "FAIL"):
        return "bold red"
    return "dim"


def render_tab_bar() -> Text:
    tabs = ["1:Health", "2:Agent", "3:Features"]
    text = Text("NAT MONITOR                                        ")
    for i, tab in enumerate(tabs, 1):
        if i == _current_tab:
            text.append(f"[{tab}]", style="bold cyan underline")
        else:
            text.append(f" {tab} ", style="dim")
        text.append("  ")
    return text


def render_health_tab() -> Panel:
    """Tab 1: System health + OOS validation."""
    parts = []

    # System health table
    health = poll_system_health()
    table = Table(title="System Health", title_style="bold cyan",
                  show_header=True, header_style="bold", border_style="dim",
                  pad_edge=False)
    table.add_column("Component", min_width=16)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Details", min_width=30)

    labels = {
        "ingestor": "Ingestor",
        "data": "Data",
        "redis": "Redis",
        "api": "API Server",
        "agent_micro": "Micro Agent",
        "agent_mediu": "MF Agent",
        "agent_macro": "Macro Agent",
    }
    for key, label in labels.items():
        if key in health:
            status, detail = health[key]
            table.add_row(label, Text(status, style=_status_style(status)), detail)

    console = Console(file=None, force_terminal=True, width=80)
    parts.append(table)

    # OOS summary (compact version)
    oos = load_oos_state(OOS_STATE_FILE)
    if oos:
        oos_table = Table(title="OOS Validation", title_style="bold cyan",
                          show_header=True, header_style="bold", border_style="dim",
                          pad_edge=False)
        oos_table.add_column("Algo", min_width=18)
        oos_table.add_column("Sym", justify="center")
        oos_table.add_column("Sharpe", justify="right")
        oos_table.add_column("PnL", justify="right")
        oos_table.add_column("Status", justify="center")

        algos = oos.get("algos", {})
        for algo_name in ["3f", "jump_detector", "funding_reversion", "optimal_entry"]:
            algo = algos.get(algo_name, {})
            baseline = algo.get("baseline_sharpe", {})
            symbols = algo.get("symbols", {})
            for sym in ["BTC", "ETH", "SOL"]:
                sym_data = symbols.get(sym, {})
                metrics = sym_data.get("metrics", {})
                bl = baseline.get(sym, 0)
                sharpe = metrics.get("current_sharpe", 0)
                pnl = metrics.get("cumulative_pnl_bps", 0)
                degradation = metrics.get("degradation", False)
                s_style = "green" if sharpe > 0 else "red"
                p_style = "green" if pnl > 0 else "red"
                oos_table.add_row(
                    algo_name, sym,
                    Text(f"{sharpe:+.1f}", style=s_style),
                    Text(f"{pnl:+.0f}", style=p_style),
                    status_label(sharpe, bl, degradation),
                )
        parts.append(oos_table)

    group = Console(file=None, force_terminal=True, width=80)
    return parts


def render_agent_tab() -> list:
    """Tab 2: Agent research progress."""
    data = poll_agent_data()
    parts = []

    if "error" in data:
        return [Text(data["error"], style="red")]

    # Totals summary
    t = data.get("totals", {})
    summary = Text()
    summary.append(f"  Hypotheses: {t.get('total', 0)}  ", style="white")
    summary.append(f"Failed: {t.get('failed', 0)}  ", style="red")
    summary.append(f"Registered: {t.get('registered', 0)}", style="green")
    parts.append(Panel(summary, title="Summary", border_style="dim"))

    # Recent cycles
    cycles = data.get("cycles", [])
    if cycles:
        ct = Table(title="Recent Cycles", title_style="bold cyan",
                   show_header=True, header_style="bold", border_style="dim",
                   pad_edge=False)
        ct.add_column("Cycle", min_width=14)
        ct.add_column("Agent", min_width=10)
        ct.add_column("Tested", justify="right")
        ct.add_column("Registered", justify="right")
        ct.add_column("FDR Rej.", justify="right")
        ct.add_column("Duration", justify="right")

        for c in cycles[:8]:
            dur = c.get("duration_s", 0)
            dur_str = f"{int(dur // 60)}m{int(dur % 60):02d}s" if dur else "?"
            ct.add_row(
                c.get("cycle_id", "?")[:14],
                c.get("agent", "?"),
                str(c.get("n_tested", 0)),
                str(c.get("n_registered", 0)),
                str(c.get("n_fdr_rejected", 0)),
                dur_str,
            )
        parts.append(ct)

    # Registry
    registry = data.get("registry", [])
    if registry:
        rt = Table(title="Registered Signals", title_style="bold green",
                   show_header=True, header_style="bold", border_style="dim",
                   pad_edge=False)
        rt.add_column("Signal", min_width=16)
        rt.add_column("Agent", min_width=8)
        rt.add_column("Generator", min_width=12)
        rt.add_column("IC", justify="right")
        rt.add_column("Status", justify="center")

        for sig in registry[:10]:
            ic = sig.get("expected_ic", 0)
            ic_style = "green" if ic > 0.1 else "yellow" if ic > 0 else "red"
            status = sig.get("status", "?")
            s_style = "green" if status == "validated" else "yellow"
            rt.add_row(
                sig.get("hypothesis_id", "?")[:16],
                sig.get("agent", "?")[:8],
                sig.get("generator", "?"),
                Text(f"{ic:.3f}", style=ic_style),
                Text(status, style=s_style),
            )
        parts.append(rt)

    # Generator performance
    gen_stats = data.get("gen_stats", {})
    if gen_stats:
        gt = Table(title="Generator Performance", title_style="bold cyan",
                   show_header=True, header_style="bold", border_style="dim",
                   pad_edge=False)
        gt.add_column("Generator", min_width=14)
        gt.add_column("Attempts", justify="right")
        gt.add_column("Wins", justify="right")
        gt.add_column("Rate", justify="right")
        gt.add_column("Weight", justify="right")

        for name, gs in sorted(gen_stats.items(), key=lambda x: -x[1]["weight"]):
            att = gs["attempts"]
            wins = gs["successes"]
            rate = (wins / att * 100) if att else 0
            bar_len = int(gs["weight"] * 20)
            gt.add_row(
                name, str(att), str(wins),
                f"{rate:.1f}%",
                f"{gs['weight']:.3f}",
            )
        parts.append(gt)

    # Failure breakdown
    failures = data.get("failures", [])
    if failures:
        ft = Table(title="Failure Breakdown", title_style="bold red",
                   show_header=True, header_style="bold", border_style="dim",
                   pad_edge=False)
        ft.add_column("Reason", min_width=22)
        ft.add_column("Count", justify="right")
        ft.add_column("Bar", min_width=20)

        max_count = max(c for _, c in failures) if failures else 1
        for reason, count in failures:
            bar_len = int(count / max_count * 20)
            bar = "\u2588" * bar_len
            ft.add_row(reason[:22], str(count), Text(bar, style="red"))
        parts.append(ft)

    return parts


def render_features_tab() -> list:
    """Tab 3: Live feature snapshots."""
    features = poll_features()
    parts = []

    if not features:
        parts.append(Panel(
            "  Redis not available or no feature data.\n"
            "  Start with: make run  (ingestor must be writing to Redis)",
            title="Live Features",
            border_style="dim red",
        ))
        return parts

    for sym in SYMBOLS:
        if sym not in features:
            continue
        f = features[sym]
        ts = f.get("timestamp", "?")

        lines = []
        # Key metrics
        mid = f.get("raw_midprice", 0)
        spread = f.get("raw_spread_bps", 0)
        depth = f.get("raw_ask_depth_5", 0)
        entropy = f.get("entropy_book", 0)
        ofi = f.get("flow_ofi", 0)
        vwap = f.get("flow_vwap_deviation", 0)

        lines.append(f"  midprice: {mid:,.1f}    spread: {spread:.1f} bps    depth_5: {depth:,.0f}")
        lines.append(f"  entropy:  {entropy:.3f}      ofi: {ofi:+.4f}       vwap_dev: {vwap:+.4f}")

        # Optional fields
        funding = f.get("funding_rate", None)
        whale = f.get("whale_net_flow", None)
        if funding is not None:
            lines.append(f"  funding:  {funding:.4f}%    whale_flow: {whale:+.3f}" if whale else f"  funding:  {funding:.4f}%")

        parts.append(Panel("\n".join(lines), title=f"{sym}", border_style="cyan"))

    # OOS cumulative PnL chart via plotext
    oos = load_oos_state(OOS_STATE_FILE)
    if oos:
        try:
            import plotext as plt
            plt.clear_figure()
            plt.theme("dark")
            plt.title("Cumulative PnL (bps)")
            plt.xlabel("Day")

            algos = oos.get("algos", {})
            for algo_name in ["3f", "jump_detector"]:
                algo = algos.get(algo_name, {})
                for sym in ["BTC", "ETH"]:
                    sym_data = algo.get("symbols", {}).get(sym, {})
                    metrics = sym_data.get("metrics", {})
                    series = metrics.get("cumulative_pnl_series", [])
                    if len(series) > 2:
                        values = [p["cum_bps"] for p in series]
                        plt.plot(values, label=f"{algo_name} {sym}")

            plt.plotsize(76, 12)
            chart_str = plt.build()
            parts.append(Panel(chart_str, title="PnL Chart", border_style="dim green"))
        except Exception:
            pass

    return parts


# ── Main Loop ───────────────────────────────────────────────────────────

def _key_listener():
    """Background thread to listen for tab-switching keypresses."""
    global _current_tab, _shutdown
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not _shutdown:
            ch = sys.stdin.read(1)
            if ch == "1":
                _current_tab = 1
            elif ch == "2":
                _current_tab = 2
            elif ch == "3":
                _current_tab = 3
            elif ch in ("q", "\x03"):  # q or Ctrl+C
                _shutdown = True
                break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    global _current_tab, _shutdown

    parser = argparse.ArgumentParser(description="NAT Unified Terminal Monitor")
    parser.add_argument("--tab", type=int, default=1, choices=[1, 2, 3],
                        help="Starting tab (1=Health, 2=Agent, 3=Features)")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval seconds")
    parser.add_argument("--no-keys", action="store_true",
                        help="Disable keyboard listener (for non-interactive use)")
    args = parser.parse_args()
    _current_tab = args.tab

    signal.signal(signal.SIGINT, lambda *_: setattr(sys.modules[__name__], "_shutdown", True))

    console = Console()

    # Start keyboard listener
    if not args.no_keys and sys.stdin.isatty():
        key_thread = threading.Thread(target=_key_listener, daemon=True)
        key_thread.start()

    try:
        while not _shutdown:
            # Clear and render
            console.clear()
            console.print(render_tab_bar())
            console.print()

            if _current_tab == 1:
                parts = render_health_tab()
            elif _current_tab == 2:
                parts = render_agent_tab()
            else:
                parts = render_features_tab()

            for part in parts:
                console.print(part)
                console.print()

            # Footer
            now = datetime.now().strftime("%H:%M:%S")
            console.print(
                f"  [dim]Refreshing every {args.refresh}s | "
                f"Press 1/2/3 to switch tabs | q to quit | {now}[/dim]"
            )

            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown = True
        console.print("\n[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
