"""`nat gauntlet` — multi-day OOS sweep across all algorithms."""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import _banner, _py


def cmd_gauntlet_run(args):
    """Run all algorithms across multiple OOS days with aggregate statistics."""
    _banner("Gauntlet: Multi-Day OOS Sweep")
    sys.stdout.flush()

    data_dir = getattr(args, 'data_dir', 'data/features')
    symbols = getattr(args, 'symbols', ['BTC', 'ETH', 'SOL'])
    cost_mode = getattr(args, 'cost_mode', 'binance_vip9')

    cmd = f"scripts/alpha/overnight_sweep.py run --data-dir {data_dir}"
    cmd += f" --symbols {' '.join(symbols)}"
    cmd += f" --cost-mode {cost_mode}"

    last = getattr(args, 'last', None)
    date_from = getattr(args, 'date_from', None)
    date_to = getattr(args, 'date_to', None)
    if last:
        cmd += f" --last {last}"
    if date_from:
        cmd += f" --from {date_from}"
    if date_to:
        cmd += f" --to {date_to}"
    if getattr(args, 'algos', None):
        cmd += f" --algos {' '.join(args.algos)}"
    if getattr(args, 'exclude_algos', None):
        cmd += f" --exclude-algos {' '.join(args.exclude_algos)}"

    r = _py(cmd)
    return r.returncode


def cmd_gauntlet_stop(args):
    """Stop a running gauntlet and print partial results."""
    r = _py("scripts/alpha/overnight_sweep.py stop")
    return r.returncode


def cmd_gauntlet_report(args):
    """Print the latest gauntlet report."""
    r = _py("scripts/alpha/overnight_sweep.py report")
    return r.returncode


def cmd_gauntlet_report_all(args):
    """Merge all gauntlet reports into a combined summary."""
    r = _py("scripts/alpha/overnight_sweep.py report_all")
    return r.returncode


def register(sub):
    # ── gauntlet (multi-day OOS sweep) ──
    gauntlet_p = sub.add_parser('gauntlet', help='Multi-day OOS sweep across all algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Runs all 19 algorithms (18 generic + 3f_liquidity) across multiple
        OOS dates with walk-forward z-score calibration. Produces a ranked
        summary table with cumulative PnL, Sharpe ratio, and win rate.

        Each date uses the prior 3 days as training. Dates with <4h of data
        are skipped automatically. Saves incrementally after each date.

        Example:
          nat gauntlet                           # all available dates
          nat gauntlet --last 7                  # last 7 days only
          nat gauntlet stop                      # kill + print partial results
          nat gauntlet report                    # print latest report
          nat gauntlet report_all                # merge all runs into one
        """))
    gauntlet_p.set_defaults(func=lambda a: gauntlet_p.print_help())
    gsub = gauntlet_p.add_subparsers(dest='gauntlet_cmd')

    # gauntlet run (also default when no subcommand)
    g_run = gsub.add_parser('run', help='Start the sweep')
    g_run.add_argument('--last', type=int, default=None,
                       help='Only test the last N dates')
    g_run.add_argument('--from', dest='date_from', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    g_run.add_argument('--to', dest='date_to', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    g_run.add_argument('--data-dir', default='data/features',
                       help='Feature data directory')
    g_run.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'],
                       help='Symbols to test')
    g_run.add_argument('--algos', nargs='+', default=None,
                       help='Only run these algorithms')
    g_run.add_argument('--exclude-algos', nargs='+', default=None,
                       help='Run all algorithms except these')
    g_run.add_argument('--cost-mode', choices=['binance_vip9', 'taker', 'maker'],
                       default='binance_vip9',
                       help='Cost model (default: binance_vip9 = 1.61 bps RT)')
    g_run.set_defaults(func=cmd_gauntlet_run)

    # gauntlet stop
    g_stop = gsub.add_parser('stop', help='Stop running gauntlet, print partial results')
    g_stop.set_defaults(func=cmd_gauntlet_stop)

    # gauntlet report
    g_report = gsub.add_parser('report', help='Print the latest gauntlet report')
    g_report.set_defaults(func=cmd_gauntlet_report)

    # gauntlet report_all
    g_all = gsub.add_parser('report_all', help='Merge all gauntlet runs into combined summary')
    g_all.set_defaults(func=cmd_gauntlet_report_all)


__all__ = ["cmd_gauntlet_run", "cmd_gauntlet_stop", "cmd_gauntlet_report",
           "cmd_gauntlet_report_all", "register"]
