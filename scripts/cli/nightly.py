"""`nat nightly` — overnight feature stats + algo performance report."""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import _banner, _py


def cmd_nightly_run(args):
    """Run the full overnight report pass (one-shot)."""
    _banner("Nightly Report")
    sys.stdout.flush()
    cmd = f"scripts/nightly_report.py run --last {getattr(args, 'last', 7)}"
    cmd += f" --data-dir {getattr(args, 'data_dir', 'data/features')}"
    symbols = getattr(args, 'symbols', None)
    if symbols:
        cmd += f" --symbols {' '.join(symbols)}"
    if getattr(args, 'quick', False):
        cmd += " --quick"
    if getattr(args, 'skip_gauntlet', False):
        cmd += " --skip-gauntlet"
    if getattr(args, 'full_gauntlet', False):
        cmd += " --full-gauntlet"
    if getattr(args, 'sections', None):
        cmd += f" --sections {args.sections}"
    r = _py(cmd)
    return r.returncode


def cmd_nightly_report(args):
    """Print the latest nightly report summary."""
    r = _py("scripts/nightly_report.py report")
    return r.returncode


def cmd_nightly_open(args):
    """Open the latest nightly HTML report."""
    r = _py("scripts/nightly_report.py open")
    return r.returncode


def register(sub):
    # ── nightly (overnight stats + performance report) ──
    nightly_p = sub.add_parser('nightly', help='Overnight feature stats + algo performance report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        One-shot overnight pass producing a self-contained HTML report:
        data health, feature NaN coverage + distributions + drift, NaN-wiring
        diagnostics (position tracker viability), gauntlet OOS sweep, and
        embedded plots. Saves incrementally — a crash still leaves a valid
        partial report.

        By default the gauntlet skips cascade_probability, which alone takes
        ~70 min/date (measured: 4065s of a 4297s date; every other algo runs
        in <45s). Default full pass: ~30-60 min. --full-gauntlet includes
        it (~1-1.5h per tested date).

        Output: reports/nightly/YYYY-MM-DD.html (+ .json sidecar)

        Example:
          nat nightly &                          # start before bed
          tmux new -d -s nightly 'nat nightly'   # or detached in tmux
          nat nightly --last 2 --quick --skip-gauntlet   # fast smoke run
          nat nightly --full-gauntlet            # include cascade_probability
          nat nightly report                     # morning summary in terminal
          nat nightly open                       # open HTML in browser
        """))
    nightly_p.add_argument('--last', type=int, default=7,
                           help='Lookback: last N available dates (default 7)')
    nightly_p.add_argument('--data-dir', default='data/features',
                           help='Feature data directory')
    nightly_p.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'],
                           help='Symbols to analyze')
    nightly_p.add_argument('--quick', action='store_true',
                           help='Coarser subsampling, never launch the screener')
    nightly_p.add_argument('--skip-gauntlet', action='store_true',
                           help='Use cached gauntlet results instead of running the sweep')
    nightly_p.add_argument('--full-gauntlet', action='store_true',
                           help='Include slow algos excluded by default (cascade_probability)')
    nightly_p.add_argument('--sections', type=str, default=None,
                           help='Comma list: health,wiring,features,gauntlet,viz')
    nightly_p.set_defaults(func=cmd_nightly_run)
    nsub = nightly_p.add_subparsers(dest='nightly_cmd')

    n_report = nsub.add_parser('report', help='Print latest nightly summary')
    n_report.set_defaults(func=cmd_nightly_report)

    n_open = nsub.add_parser('open', help='Open latest nightly HTML report')
    n_open.set_defaults(func=cmd_nightly_open)


__all__ = ["cmd_nightly_run", "cmd_nightly_report", "cmd_nightly_open", "register"]
