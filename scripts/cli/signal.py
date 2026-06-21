"""`nat signal` — signal existence / IC testing."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import _banner, _py, _sym


def cmd_signal_test(args):
    symbol = _sym(args)
    horizon = getattr(args, 'horizon', 3000)
    spread = getattr(args, 'spread_bps', 1.0)
    remove_leaky = getattr(args, 'remove_leaky', False)
    _banner(f"Signal existence test: {symbol}")
    cmd = f"scripts/phase1_signal_test.py --symbol {symbol} --horizon {horizon} --spread-bps {spread}"
    if remove_leaky:
        cmd += " --remove-leaky"
    _py(cmd)


def cmd_signal_test_all(args):
    horizon = getattr(args, 'horizon', 3000)
    spread = getattr(args, 'spread_bps', 1.0)
    _banner("Full signal sweep: all symbols + feature sets")
    for sym in ["BTC", "ETH", "SOL"]:
        for leaky in ([False, True] if sym == "BTC" else [False]):
            tag = f"{sym} -- {'leaky removed' if leaky else 'all features'}"
            print(f"\n  === {tag} ===\n")
            cmd = (f"scripts/phase1_signal_test.py --symbol {sym} "
                   f"--horizon {horizon} --spread-bps {spread}")
            if leaky:
                cmd += " --remove-leaky"
            _py(cmd)


def register(sub):
    sig_p = sub.add_parser('signal', help='Signal testing')
    sig_p.set_defaults(func=lambda a: sig_p.print_help())
    ssub = sig_p.add_subparsers(dest='subcmd')
    st = ssub.add_parser('test', help='Signal existence test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Tests whether individual features carry predictive information about
        future returns using Information Coefficient (IC) analysis.

        Mathematics:
          Forward return:  r_t(h) = (mid_{t+h} - mid_t) / mid_t
          IC:              Spearman rank correlation ρ(feature, r_t(h))
          Significance:    z = IC · √n,  p < 0.05 ⟹ |z| > 1.96
          Cost filter:     |IC| > spread_bps / (10000 · σ_r)

        Output: per-feature IC at the specified horizon, sorted by |IC|.
        Features passing both significance and cost filters are flagged.

        Example:
          nat signal test --symbol BTC --horizon 3000 --spread-bps 1.0
        """))
    st.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    st.add_argument('--horizon', type=int, default=3000, help='Forward horizon in ticks (1 tick=100ms)')
    st.add_argument('--spread-bps', type=float, default=1.0, help='Spread cost in basis points')
    st.add_argument('--remove-leaky', action='store_true', help='Remove forward-looking features')
    st.set_defaults(func=cmd_signal_test)
    sta = ssub.add_parser('test-all', help='Full symbol sweep')
    sta.add_argument('--horizon', type=int, default=3000, help='Forward horizon in ticks (1 tick=100ms)')
    sta.add_argument('--spread-bps', type=float, default=1.0, help='Spread cost in basis points')
    sta.set_defaults(func=cmd_signal_test_all)


__all__ = ["cmd_signal_test", "cmd_signal_test_all", "register"]
