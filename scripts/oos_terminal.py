"""
OOS Validation Terminal Dashboard

Reads state from data/oos_validation/state.json and renders a rich terminal
dashboard with performance tables, rolling Sharpe sparklines, cumulative PnL,
and degradation alerts.

Usage:
  python scripts/oos_terminal.py                # One-shot render
  python scripts/oos_terminal.py --watch        # Re-render every 10s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data" / "oos_validation" / "state.json"

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def sparkline(values: list[float], width: int = 20) -> str:
    """Convert a list of floats to a Unicode sparkline string."""
    if not values:
        return ""
    recent = values[-width:]
    mn, mx = min(recent), max(recent)
    rng = mx - mn if mx != mn else 1.0
    return "".join(
        SPARK_BLOCKS[min(7, int((v - mn) / rng * 7.99))]
        for v in recent
    )


def status_label(current_sharpe: float, baseline_sharpe: float, degradation: bool) -> Text:
    """Return colored status indicator."""
    if baseline_sharpe <= 0:
        return Text("N/A", style="dim")
    ratio = current_sharpe / baseline_sharpe if baseline_sharpe else 0
    if degradation:
        return Text("DEGRAD", style="bold red")
    if ratio >= 0.7:
        return Text("OK", style="bold green")
    if ratio >= 0.4:
        return Text("WATCH", style="bold yellow")
    return Text("FAIL", style="bold red")


def render_performance_table(console: Console, state: dict) -> None:
    """Render the main algorithm performance table."""
    table = Table(
        title="Algorithm Performance",
        title_style="bold cyan",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
    )
    table.add_column("Algorithm", style="white", min_width=20)
    table.add_column("Sym", style="cyan", justify="center")
    table.add_column("Sharpe", justify="right")
    table.add_column("PnL (bps)", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("DD (bps)", justify="right")
    table.add_column("Days", justify="right")
    table.add_column("Status", justify="center")

    algos = state.get("algos", {})
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
            dd = metrics.get("max_drawdown_bps", 0)
            n_days = metrics.get("n_days", 0)
            degradation = metrics.get("degradation", False)

            # Average win rate across all days
            daily = sym_data.get("daily", [])
            if daily:
                wr = sum(d.get("win_rate", 0) for d in daily) / len(daily)
            else:
                wr = 0

            # Color Sharpe
            sharpe_style = "green" if sharpe > 0 else "red"
            pnl_style = "green" if pnl > 0 else "red"

            table.add_row(
                algo_name,
                sym,
                Text(f"{sharpe:+.1f}", style=sharpe_style),
                Text(f"{pnl:+.0f}", style=pnl_style),
                f"{wr:.0%}",
                f"{dd:+.0f}",
                str(n_days),
                status_label(sharpe, bl, degradation),
            )
        table.add_section()

    console.print(table)


def render_sparklines(console: Console, state: dict) -> None:
    """Render rolling Sharpe and cumulative PnL sparklines."""
    algos = state.get("algos", {})

    # Rolling Sharpe
    lines_sharpe = []
    for algo_name in ["3f", "jump_detector", "funding_reversion", "optimal_entry"]:
        algo = algos.get(algo_name, {})
        baseline = algo.get("baseline_sharpe", {})
        symbols = algo.get("symbols", {})

        for sym in ["BTC", "ETH", "SOL"]:
            sym_data = symbols.get(sym, {})
            metrics = sym_data.get("metrics", {})
            rolling = metrics.get("rolling_sharpe_7d", [])
            if not rolling:
                continue
            values = [r["sharpe"] for r in rolling]
            bl = baseline.get(sym, 0)
            current = values[-1] if values else 0
            spark = sparkline(values)
            label = f"  {algo_name:<20s} {sym}: {spark}  (base: {bl:.1f}, now: {current:+.1f})"
            lines_sharpe.append(label)

    if lines_sharpe:
        console.print(Panel(
            "\n".join(lines_sharpe),
            title="Rolling Sharpe (7d)",
            title_align="left",
            border_style="dim cyan",
        ))

    # Cumulative PnL
    lines_pnl = []
    for algo_name in ["3f", "jump_detector", "funding_reversion", "optimal_entry"]:
        algo = algos.get(algo_name, {})
        symbols = algo.get("symbols", {})

        for sym in ["BTC", "ETH", "SOL"]:
            sym_data = symbols.get(sym, {})
            metrics = sym_data.get("metrics", {})
            cum_series = metrics.get("cumulative_pnl_series", [])
            if not cum_series:
                continue
            values = [c["cum_bps"] for c in cum_series]
            total = values[-1] if values else 0
            spark = sparkline(values)
            style = "green" if total > 0 else "red"
            label = f"  {algo_name:<20s} {sym}: {spark}  [{total:+.0f} bps]"
            lines_pnl.append((label, style))

    if lines_pnl:
        text = Text()
        for label, style in lines_pnl:
            text.append(label + "\n", style=style)
        console.print(Panel(
            text,
            title="Cumulative PnL",
            title_align="left",
            border_style="dim green",
        ))


def render_alerts(console: Console, state: dict) -> None:
    """Render degradation alerts."""
    alerts = []
    algos = state.get("algos", {})

    for algo_name in ["3f", "jump_detector", "funding_reversion", "optimal_entry"]:
        algo = algos.get(algo_name, {})
        symbols = algo.get("symbols", {})
        for sym in ["BTC", "ETH", "SOL"]:
            sym_data = symbols.get(sym, {})
            metrics = sym_data.get("metrics", {})
            if metrics.get("degradation"):
                bl = algo.get("baseline_sharpe", {}).get(sym, 0)
                rolling = metrics.get("rolling_sharpe_7d", [])
                latest_7d = rolling[-1]["sharpe"] if rolling else 0
                alerts.append(
                    f"  {algo_name} {sym}: rolling 7d Sharpe below 50% of baseline "
                    f"for 3+ days (baseline {bl:.1f}, latest 7d: {latest_7d:+.1f})"
                )

    if alerts:
        console.print(Panel(
            "\n".join(alerts),
            title="Degradation Alerts",
            title_align="left",
            border_style="bold red",
        ))
    else:
        console.print(Panel(
            "  (none)",
            title="Alerts",
            title_align="left",
            border_style="dim",
        ))


def render_dashboard(state: dict) -> None:
    """Render the full terminal dashboard."""
    console = Console()
    console.print()

    last = state.get("last_updated", "unknown")
    console.print(
        f"[bold cyan]OOS VALIDATION DASHBOARD[/bold cyan]"
        f"              Last: {last[:19]}",
    )
    console.print()

    render_performance_table(console, state)
    console.print()
    render_sparklines(console, state)
    render_alerts(console, state)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="OOS Validation Terminal Dashboard")
    parser.add_argument("--state", type=str, default=str(STATE_FILE),
                        help="Path to state JSON")
    parser.add_argument("--watch", action="store_true",
                        help="Re-render every 10s")
    args = parser.parse_args()

    state_path = Path(args.state)

    if args.watch:
        try:
            while True:
                state = load_state(state_path)
                if state is None:
                    print(f"Waiting for state file: {state_path}")
                else:
                    # Clear screen and re-render
                    print("\033[2J\033[H", end="")
                    render_dashboard(state)
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        state = load_state(state_path)
        if state is None:
            print(f"No state file found at {state_path}")
            print("Run 'python scripts/oos_validate.py batch' first.")
            sys.exit(1)
        render_dashboard(state)


if __name__ == "__main__":
    main()
