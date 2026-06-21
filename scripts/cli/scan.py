"""`nat scan` — scalp edge scanner."""

from __future__ import annotations

import argparse
import subprocess
import textwrap

from cli.common import ROOT, PY, DATA_DEFAULT, _banner


def cmd_scan(args):
    """Run scalp edge scanner on collected data."""
    _banner("Scalp Edge Scanner")
    symbol = getattr(args, 'symbol', 'BTC')
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    tail = getattr(args, 'tail', None)
    cmd_parts = [PY, str(ROOT / "scripts" / "scalp_edge_scanner.py"),
                 "--symbol", symbol, "--data-dir", data_dir]
    if tail is not None:
        cmd_parts += ["--tail", str(tail)]
    subprocess.run(cmd_parts, cwd=str(ROOT))


def register(sub):
    scan_p = sub.add_parser('scan', help='Scalp edge scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Scans for scalping edges by analyzing bid/ask asymmetry and tail
        percentile behavior under different regime conditions.

        Mathematics:
          Edge = P(mid_price moves in predicted direction within horizon)
          Tail analysis:  P(edge > threshold | regime_quintile)
          Conditional IC: IC(feature, return | ent_book_shape ∈ Q_k)

        Output: ranked list of (feature, regime, horizon) combos with edge > cost.

        Example:
          nat scan --symbol BTC --tail 95
        """))
    scan_p.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    scan_p.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    scan_p.add_argument('--tail', type=int, default=None, help='Tail percentile threshold')
    scan_p.set_defaults(func=cmd_scan)


__all__ = ["cmd_scan", "register"]
