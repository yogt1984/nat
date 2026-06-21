"""`nat alg1` — MF 3-feature liquidity signal (100min): backtest / paper / live."""

from __future__ import annotations

import argparse
import os
import textwrap

from cli.common import (
    B, W, R, Y,
    _py, _banner, _p,
)


def cmd_alg1(args):
    """Backtest + dry-run signal bridge."""
    _banner("ALG1: MF 3-Feature Liquidity Signal (100min)")
    print(f"  {B}Step 1:{W} Backtest on latest data...\n")
    r = _py("scripts/analysis/mf_liquidity_backtest.py --features both --save")
    if r.returncode != 0:
        return r.returncode
    cycle = getattr(args, 'cycle', 300)
    print(f"\n  {B}Step 2:{W} Signal bridge dry-run (cycle={cycle}s, Ctrl-C to stop)...\n")
    _py(f"scripts/execution/signal_bridge.py --mode dry-run --cycle {cycle}")


def cmd_alg1_paper(args):
    """Paper trader batch + watch."""
    _banner("ALG1: Paper Trading Mode")
    print(f"  {B}Step 1:{W} Batch replay...\n")
    r = _py("scripts/alpha/paper_trader.py batch --save")
    if r.returncode != 0:
        return r.returncode
    symbol = getattr(args, 'symbol', 'BTC')
    poll = getattr(args, 'poll', 300)
    print(f"\n  {B}Step 2:{W} Watch mode ({symbol}, poll={poll}s, Ctrl-C to stop)...\n")
    _py(f"scripts/alpha/paper_trader.py watch --symbol {symbol} --poll {poll}")


def cmd_alg1_live(args):
    """Live trading (requires HL_PRIVATE_KEY)."""
    if not os.environ.get("HL_PRIVATE_KEY"):
        _p("!", R, "HL_PRIVATE_KEY not set. Export your hex private key first.")
        return 1
    cycle = getattr(args, 'cycle', 300)
    _banner("ALG1: LIVE MODE")
    _p("!", Y, f"Placing real orders on Hyperliquid (cycle={cycle}s)")
    print()
    _py(f"scripts/execution/signal_bridge.py --mode live --cycle {cycle}")


def register(sub):
    # ── alg1 (MF 3-feature liquidity signal) ──
    a1_p = sub.add_parser('alg1', help='MF 3-feature liquidity signal (100min)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        First validated algorithm: 3-feature composite liquidity signal.

        Signal:
          composite = (z(spread_bps) + z(depth_5_std) + z(vwap_deviation_std)) / 3
          z-score params: walk-forward on last 3 completed dates
          Horizon: 100min (20 bars at 5min)
          Entry: composite < -1.0 (LONG) or > +1.0 (SHORT)

        Modes:
          nat alg1             Backtest latest data + dry-run signal bridge
          nat alg1 paper       Paper trader batch replay + watch mode
          nat alg1 live        Real orders on Hyperliquid (maker-only)

        Example:
          nat alg1 --cycle 60            # faster bridge cycles
          nat alg1 paper --symbol ETH    # watch ETH only
        """))
    a1_p.add_argument('--cycle', type=int, default=300, help='Signal bridge cycle seconds (default: 300)')
    a1_p.set_defaults(func=cmd_alg1)
    a1sub = a1_p.add_subparsers(dest='subcmd')
    a1paper = a1sub.add_parser('paper', help='Paper trader batch + watch')
    a1paper.add_argument('--symbol', default='BTC', help='Symbol for watch mode (default: BTC)')
    a1paper.add_argument('--poll', type=int, default=300, help='Watch poll interval seconds (default: 300)')
    a1paper.set_defaults(func=cmd_alg1_paper)
    a1live = a1sub.add_parser('live', help='LIVE orders on Hyperliquid (requires HL_PRIVATE_KEY)')
    a1live.add_argument('--cycle', type=int, default=300, help='Signal bridge cycle seconds (default: 300)')
    a1live.set_defaults(func=cmd_alg1_live)


__all__ = ["cmd_alg1", "cmd_alg1_paper", "cmd_alg1_live", "register"]
