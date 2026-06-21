"""`nat audit` — backtest audit and parameter sweep tools."""

from __future__ import annotations

from cli.common import DATA_DEFAULT, _banner, _py, _data


def cmd_audit_aggregate(args):
    _banner("Audit: Aggregate Walk-Forward Results")
    _py(f"scripts/audit_aggregate.py {args.audit_dir}")

def cmd_audit_sweep(args):
    _banner("Audit: Parameter Sweep")
    cmd = f"scripts/audit_sweep.py --data-dir {_data(args)}"
    symbols = getattr(args, 'symbols', None)
    if symbols:
        cmd += " --symbols " + " ".join(symbols)
    timeframes = getattr(args, 'timeframes', None)
    if timeframes:
        cmd += " --timeframes " + " ".join(timeframes)
    if getattr(args, 'verbose', False):
        cmd += " -v"
    _py(cmd)


def register(sub):
    # ── audit ──
    aud_p = sub.add_parser('audit', help='Backtest audit and parameter sweep tools')
    aud_p.set_defaults(func=lambda a: aud_p.print_help())
    audsub = aud_p.add_subparsers(dest='subcmd')
    agg = audsub.add_parser('aggregate', help='Aggregate walk-forward backtest results')
    agg.add_argument('audit_dir', help='Directory containing walk_forward_*.json files')
    agg.set_defaults(func=cmd_audit_aggregate)
    asw = audsub.add_parser('sweep', help='Systematic parameter sweep across symbols/timeframes')
    asw.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    asw.add_argument('--symbols', nargs='+', default=None, help='Symbols to sweep (default: BTC ETH SOL)')
    asw.add_argument('--timeframes', nargs='+', default=None, help='Timeframes (default: 5min 15min 1h)')
    asw.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    asw.set_defaults(func=cmd_audit_sweep)


__all__ = ["cmd_audit_aggregate", "cmd_audit_sweep", "register"]
