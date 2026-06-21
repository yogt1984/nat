"""`nat profile` — regime / scalping feature profiling."""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import ROOT, DATA_DEFAULT, B, _p, _py, _banner, _sym, _data


def cmd_profile(args):
    """Run regime profiling on current data."""
    _p("...", B, "Running profiling pipeline...")
    print()
    r = _py("-m scripts.experiment.profiler")
    if r.returncode != 0:
        sys.path.insert(0, str(ROOT / "scripts"))
        from experiment.profiler import quick_profile
        from dataclasses import asdict
        import json
        print(json.dumps(asdict(quick_profile()), indent=2))


def cmd_profile_scalp(args):
    """Run scalping feature profiler."""
    _banner(f"Scalping profiler: {_sym(args)}")
    cmd = f"scripts/scalping_profiler.py profile --symbol {_sym(args)} --data-dir {_data(args)}"
    top = getattr(args, 'top', None)
    if top:
        cmd += f" --top {top}"
    if getattr(args, 'forward_test', False):
        cmd += " --forward-test"
    tf = getattr(args, 'timeframe', None)
    if tf:
        cmd += f" --timeframe {tf}"
    _py(cmd)


def register(sub):
    prof_p = sub.add_parser('profile', help='Profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Regime profiling of microstructure features using entropy and volatility
        measures to characterize market states.

        Key metrics:
          Permutation entropy:  H = -Σ pᵢ·ln(pᵢ) / ln(m!)
            where pᵢ = frequency of ordinal pattern i, m = embedding dimension
          Parkinson volatility: σ = ln(H/L) / (4·ln(2))^0.5 ≈ ln(H/L) / 1.663
          Book imbalance:       I = (bid_qty - ask_qty) / (bid_qty + ask_qty)

        Output: per-symbol regime statistics and feature distributions.

        Example:
          nat profile                    # full profiling
          nat profile scalp --symbol BTC # scalping features
        """))
    prof_p.set_defaults(func=cmd_profile)
    prof_sub = prof_p.add_subparsers(dest='subcmd')
    ps = prof_sub.add_parser('scalp', help='Scalping feature profiler (--symbol, --top, --forward-test)')
    ps.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ps.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ps.add_argument('--top', type=int, default=None, help='Show top N features')
    ps.add_argument('--forward-test', action='store_true', help='Run forward test')
    ps.add_argument('--timeframe', default=None, help='Bar timeframe (e.g. 5min, 15min)')
    ps.set_defaults(func=cmd_profile_scalp)


__all__ = ["cmd_profile", "cmd_profile_scalp", "register"]
