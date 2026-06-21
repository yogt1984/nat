"""`nat oos|oos30|daily` — out-of-sample paper-trading validation for the winners.

daily = 6-hour OOS snapshot; oos30 = 30-day OOS validation across all winners;
oos = longitudinal window analysis over accumulated gauntlet daily P&L.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import (
    B, W, G, R,
    _py, _banner, _json_mode, _p,
)

WINNING_ALGOS = ["jump_detector", "funding_reversion", "optimal_entry"]


def cmd_daily(args):
    """Run daily 6-hour OOS paper trading snapshot."""
    _banner("Daily 6-Hour OOS Snapshot")
    sys.stdout.flush()

    data_dir = getattr(args, 'data_dir', 'data/features')
    date = getattr(args, 'date', None)
    min_hours = getattr(args, 'min_hours', 4.0)
    symbols = getattr(args, 'symbols', ['BTC', 'ETH', 'SOL'])

    cmd = f"scripts/alpha/paper_trader_daily.py --data-dir {data_dir} --min-hours {min_hours}"
    cmd += f" --symbols {' '.join(symbols)}"
    if date:
        cmd += f" --date {date}"
    if getattr(args, 'no_save', False):
        cmd += " --no-save"
    cost_mode = getattr(args, 'cost_mode', 'binance_vip9')
    cmd += f" --cost-mode {cost_mode}"

    r = _py(cmd)
    return r.returncode


def cmd_oos30(args):
    """Run 30-day OOS validation for all winning algorithms."""
    _banner("OOS30: 30-Day Out-of-Sample Validation")

    data_dir = getattr(args, 'data_dir', 'data/features')
    symbols = getattr(args, 'symbols', ['BTC', 'ETH', 'SOL'])
    sym_str = ' '.join(symbols)

    # Step 1: 3f liquidity signal (has its own paper trader)
    print(f"  {B}Step 1/3:{W} 3f liquidity signal (paper_trader.py)...\n")
    r = _py("scripts/alpha/paper_trader.py batch --save")
    if r.returncode != 0:
        _p("✗", R, "3f liquidity paper trader failed")
        return r.returncode
    _p("✓", G, "3f liquidity complete")

    # Step 2: Generic algorithms (jump_detector, funding_reversion, optimal_entry)
    algos = ' '.join(WINNING_ALGOS)
    print(f"\n  {B}Step 2/3:{W} Generic algorithms ({algos})...\n")
    r = _py(f"scripts/alpha/paper_trader_generic.py --algorithms {algos} "
            f"--data-dir {data_dir} --symbols {sym_str} --save")
    if r.returncode != 0:
        _p("✗", R, "Generic paper trader failed")
        return r.returncode
    _p("✓", G, "Generic algorithms complete")

    # Step 3: Surprise signal (has its own paper trader)
    print(f"\n  {B}Step 3/3:{W} Surprise signal (paper_trader_surprise.py)...\n")
    r = _py(f"scripts/alpha/paper_trader_surprise.py batch --data-dir {data_dir} "
            f"--symbols {sym_str} --save")
    if r.returncode != 0:
        _p("✗", R, "Surprise signal paper trader failed")
        return r.returncode
    _p("✓", G, "Surprise signal complete")

    print(f"\n  {'=' * 50}")
    _p("✓", G, "OOS30 validation complete. Reports saved to reports/")
    print()
    print(f"  Results:")
    print(f"    reports/mf_liquidity_updated.json    (3f liquidity)")
    print(f"    reports/algo_paper_trade_comparison.json (generic algos)")
    print(f"    reports/surprise_paper_trade.json    (surprise signal)")
    print()
    print(f"  Re-run periodically as new data accumulates.")
    print(f"  Target: ≥30 OOS days for statistical significance (currently 13).")
    print()
    return 0


def cmd_oos_window(args):
    """Longitudinal OOS validation over a trailing window of gauntlet daily P&L."""
    cmd = f"scripts/alpha/oos_window.py --window {getattr(args, 'window', '30d')}"
    symbols = getattr(args, 'symbols', None)
    if symbols:
        cmd += f" --symbols {' '.join(symbols)}"
    algos = getattr(args, 'algos', None)
    if algos:
        cmd += f" --algos {' '.join(algos)}"
    tf = getattr(args, 'train_frac', None)
    if tf is not None:
        cmd += f" --train-frac {tf}"
    if _json_mode(args):
        cmd += " --json"
    return _py(cmd).returncode


def register(sub):
    # ── oos30 (30-day OOS validation) ──
    oos_p = sub.add_parser('oos30', help='30-day OOS validation for winning algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Runs walk-forward paper trading for all 5 winning algorithms:
          1. 3f_liquidity   (spread+depth+vwap composite)
          2. jump_detector  (Lee-Mykland post-jump reversion)
          3. funding_reversion (funding rate mean-reversion)
          4. optimal_entry  (SPRT on Kalman innovations)
          5. surprise_signal (entropy regime transitions)

        Re-run periodically as the ingestor accumulates new data.
        Target: >=30 OOS days for statistical significance at alpha=0.05.

        Example:
          nat oos30                          # run all 5 on BTC/ETH/SOL
          nat oos30 --symbols BTC ETH       # subset of symbols
        """))
    oos_p.add_argument('--data-dir', default='data/features', help='Feature data directory')
    oos_p.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'], help='Symbols to test')
    oos_p.set_defaults(func=cmd_oos30)

    # ── oos (longitudinal window analysis over accumulated gauntlet P&L) ──
    oosw_p = sub.add_parser('oos',
        help='Longitudinal OOS validation over a trailing window of gauntlet P&L',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Analyzes the daily P&L the gauntlet already produced (reports/gauntlet_*.json)
        over a trailing window of trading days. Per (algorithm, symbol): annualized
        Sharpe, a daily-P&L walk-forward holdout (IS vs OOS -- the overfit signal),
        max drawdown, win rate, deflated Sharpe (multiple-testing adjusted), and
        cross-algorithm complementarity. Metrics only -- G4 gates live in the alpha
        pipeline. Run `nat gauntlet run --last N` first to (re)generate the daily
        P&L; coverage is reported honestly, never silently truncated.

        Example:
          nat gauntlet run --last 30        # accumulate/refresh 30 days of daily P&L
          nat oos --window 30d              # longitudinal verdict on the 5 deployables
          nat --json oos --window 14d       # machine-readable
        """))
    oosw_p.add_argument('--window', default='30d',
        help='Trailing window in trading days, e.g. 30d (default: 30d)')
    oosw_p.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'])
    oosw_p.add_argument('--algos', nargs='+', default=None,
        help='Algorithms to analyze (default: the 5 deployables)')
    oosw_p.add_argument('--train-frac', type=float, default=0.67,
        help='Walk-forward train fraction (default: 0.67)')
    oosw_p.set_defaults(func=cmd_oos_window)

    # ── daily (6-hour OOS snapshot) ──
    daily_p = sub.add_parser('daily', help='Daily 6-hour OOS snapshot for winning algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Runs all 5 winning algorithms on today's data (or --date) using
        the prior 3 days as training. Saves results as reports/6h__YYYY-MM-DD.json.
        Rolling 7-day and 30-day stats computed across all daily files.

        Run daily after ~6h of data collection for a quick health check.
        Individual days are noisy (~30 trades), but rolling stats converge.

        Example:
          nat daily                            # test latest date
          nat daily --date 2026-05-23          # specific date
          nat daily --min-hours 8              # require 8h minimum
        """))
    daily_p.add_argument('--date', type=str, default=None, help='Test date (YYYY-MM-DD). Default: latest.')
    daily_p.add_argument('--data-dir', default='data/features', help='Feature data directory')
    daily_p.add_argument('--min-hours', type=float, default=4.0, help='Minimum hours of data (default: 4)')
    daily_p.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'], help='Symbols to test')
    daily_p.add_argument('--no-save', action='store_true', help="Don't save report file")
    daily_p.add_argument('--cost-mode', choices=['binance_vip9', 'taker', 'maker', 'config'],
                         default='binance_vip9', help='Cost model (default: binance_vip9 = 1.61 bps RT)')
    daily_p.set_defaults(func=cmd_daily)


__all__ = ["WINNING_ALGOS", "cmd_daily", "cmd_oos30", "cmd_oos_window", "register"]
