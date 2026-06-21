"""`nat fetch` — fetch historical data from exchanges."""

from __future__ import annotations

from cli.common import _py


def cmd_fetch_candles(args):
    cmd = "scripts/data/fetch_candles.py"
    cmd += f" --symbol {' '.join(args.fetch_symbols)}"
    cmd += f" --interval {args.fetch_interval}"
    if args.fetch_start:
        cmd += f" --start {args.fetch_start}"
    else:
        cmd += f" --days {args.fetch_days}"
    _py(cmd)


def register(sub):
    # ── fetch ──
    fetch_p = sub.add_parser('fetch', help='Fetch historical data from exchanges')
    fetch_p.set_defaults(func=lambda a: fetch_p.print_help())
    ftsub = fetch_p.add_subparsers(dest='subcmd')
    fc = ftsub.add_parser('candles', help='Fetch OHLCV candles from Hyperliquid')
    fc.add_argument('--symbol', nargs='+', default=['BTC'], dest='fetch_symbols',
                    help='Symbols to fetch (default: BTC)')
    fc.add_argument('--interval', default='1m', dest='fetch_interval',
                    help='Candle interval (default: 1m)')
    fc.add_argument('--days', type=int, default=90, dest='fetch_days',
                    help='Days of history (default: 90)')
    fc.add_argument('--start', default=None, dest='fetch_start',
                    help='Start date (ISO, overrides --days)')
    fc.set_defaults(func=cmd_fetch_candles)


__all__ = ["cmd_fetch_candles", "register"]
