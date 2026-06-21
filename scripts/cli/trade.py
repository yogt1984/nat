"""`nat trade` — paper trade visualization."""

from __future__ import annotations

import sys

from cli.common import _banner, _py, _sym


def cmd_trade_viz(args):
    """Visualize paper trades — snapshot PNG per symbol per day."""
    _banner("PAPER TRADE VISUALIZATION")
    sys.stdout.flush()
    sym = _sym(args)
    date = getattr(args, 'date', None)
    date_range = getattr(args, 'date_range', None)
    extra = ""
    if date:
        extra += f" --date {date}"
    elif date_range:
        extra += f" --date-range {date_range[0]} {date_range[1]}"
    else:
        extra += " --latest"
    _py(f"scripts/trade_visualize.py{extra} --symbol {sym} -v")


def register(sub):
    tv_p = sub.add_parser('trade', help='Paper trade visualization')
    tv_p.set_defaults(func=cmd_trade_viz)
    tvsub = tv_p.add_subparsers(dest='subcmd')
    tvv = tvsub.add_parser('viz', help='Visualize paper trades (snapshot PNG)')
    tvv.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD)')
    tvv.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                     help='Date range (inclusive)')
    tvv.add_argument('--symbol', type=str, default='BTC', help='Symbol or "all"')
    tvv.set_defaults(func=cmd_trade_viz)


__all__ = ["cmd_trade_viz", "register"]
