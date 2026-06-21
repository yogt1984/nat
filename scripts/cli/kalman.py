"""`nat kalman` — Kalman filter research (OU filter IC + drift analysis)."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import DATA_DEFAULT, _banner, _py, _data, _sym


def cmd_kalman_analysis(args):
    _banner(f"Kalman IC Analysis: {_sym(args)}")
    cmd = f"scripts/kalman/analysis.py --data-dir {_data(args)} --symbol {_sym(args)}"
    if getattr(args, 'all_symbols', False):
        cmd += " --all-symbols"
    report = getattr(args, 'json_report', None)
    if report:
        cmd += f" --json-report {report}"
    _py(cmd)

def cmd_kalman_drift(args):
    _banner(f"Kalman Drift Analysis: {_sym(args)}")
    cmd = f"scripts/kalman/drift_analysis.py --data-dir {_data(args)} --symbol {_sym(args)}"
    if getattr(args, 'all_symbols', False):
        cmd += " --all-symbols"
    report = getattr(args, 'json_report', None)
    if report:
        cmd += f" --json-report {report}"
    latency = getattr(args, 'latency_ticks', None)
    if latency:
        cmd += f" --latency-ticks {latency}"
    _py(cmd)


def register(sub):
    # ── kalman ──
    kal_p = sub.add_parser('kalman', help='Kalman filter research (OU filter IC + drift analysis)')
    kal_p.set_defaults(func=lambda a: kal_p.print_help())
    kalsub = kal_p.add_subparsers(dest='subcmd')
    ka = kalsub.add_parser('analysis', help='Phase 1: Kalman filter IC analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Evaluates OU Kalman filter on raw features. Computes IC at multiple
        horizons with regime gating (ent_book_shape < P30).

        Mathematics:
          OU process:  dx = θ(μ - x)dt + σ dW
          Kalman predict:  x̂⁻ = e^(-θΔt)·x̂,  P⁻ = e^(-2θΔt)·P + Q
          Kalman update:   K = P⁻/(P⁻ + R),  x̂ = x̂⁻ + K·(z - x̂⁻)

        Example:
          nat kalman analysis --symbol BTC --all-symbols
        """))
    ka.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ka.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ka.add_argument('--all-symbols', action='store_true', help='Run on all symbols')
    ka.add_argument('--json-report', default=None, help='Output JSON report path')
    ka.set_defaults(func=cmd_kalman_analysis)
    kd = kalsub.add_parser('drift', help='Phase 2: Drift analysis (latency-aware)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Analyzes signal drift and decay after Kalman filtering. Tests whether
        filtered signals remain predictive after realistic execution latency.

        Mathematics:
          Drift IC:      IC(t+lag) for lag ∈ {0, 1, 2, 5, 10} ticks
          Decay rate:    IC(lag) / IC(0) — measures signal half-life
          Latency cost:  IC loss from N ticks of execution delay

        Example:
          nat kalman drift --symbol BTC --latency-ticks 2
        """))
    kd.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    kd.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    kd.add_argument('--all-symbols', action='store_true', help='Run on all symbols')
    kd.add_argument('--json-report', default=None, help='Output JSON report path')
    kd.add_argument('--latency-ticks', type=int, default=2, help='Execution latency in ticks (default: 2)')
    kd.set_defaults(func=cmd_kalman_drift)


__all__ = ["cmd_kalman_analysis", "cmd_kalman_drift", "register"]
