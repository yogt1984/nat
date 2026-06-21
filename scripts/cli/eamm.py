"""`nat eamm` — entropy-aware market making (Avellaneda-Stoikov)."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import ROOT, PY, _exec, _sym


def cmd_eamm_run(args):
    symbol = _sym(args)
    horizon = getattr(args, 'horizon', 3000)
    mode = getattr(args, 'mode', 'regression')
    _exec(f"{PY} -m eamm.cli run --symbol {symbol} --horizon {horizon} --mode {mode}",
          cwd=ROOT / "scripts")


def cmd_eamm_regime(args):
    _exec(f"{PY} -m eamm.cli regime --symbol {_sym(args)} "
          f"--horizon {getattr(args, 'horizon', 3000)}",
          cwd=ROOT / "scripts")


def cmd_eamm_backtest(args):
    gamma = getattr(args, 'gamma', 0.1)
    q_max = getattr(args, 'q_max', 1.0)
    _exec(
        f"{PY} -m eamm.cli backtest --symbol {_sym(args)} "
        f"--horizon {getattr(args, 'horizon', 3000)} --gamma {gamma} --q-max {q_max}",
        cwd=ROOT / "scripts",
    )


def register(sub):
    eamm_p = sub.add_parser('eamm', help='EAMM market making')
    eamm_p.set_defaults(func=lambda a: eamm_p.print_help())
    easub = eamm_p.add_subparsers(dest='subcmd')
    er = easub.add_parser('run', help='Full EAMM pipeline')
    er.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    er.add_argument('--horizon', type=int, default=3000, help='Forward horizon in ticks')
    er.add_argument('--mode', default='regression', help='Pipeline mode (default: regression)')
    er.set_defaults(func=cmd_eamm_run)
    erg = easub.add_parser('regime', help='Regime analysis')
    erg.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    erg.add_argument('--horizon', type=int, default=3000, help='Forward horizon in ticks')
    erg.set_defaults(func=cmd_eamm_regime)
    ebt = easub.add_parser('backtest', help='Stateful backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Avellaneda-Stoikov entropy-aware market making backtest with
        inventory penalty and regime-dependent spread adjustment.

        Mathematics:
          Optimal spread:  δ = γ·σ² + (2/γ)·ln(1 + γ/κ)
          Inventory penalty: P&L_adj = P&L_raw - γ·q²
          Reservation price: r = s - q·γ·σ²
          Where: γ = risk aversion, σ² = mid-price variance,
                 κ = order arrival intensity, q = inventory

        Example:
          nat eamm backtest --symbol BTC --gamma 0.1 --q-max 1.0
        """))
    ebt.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ebt.add_argument('--horizon', type=int, default=3000, help='Forward horizon in ticks')
    ebt.add_argument('--gamma', type=float, default=0.1, help='Risk aversion parameter')
    ebt.add_argument('--q-max', type=float, default=1.0, help='Maximum inventory')
    ebt.set_defaults(func=cmd_eamm_backtest)


__all__ = ["cmd_eamm_run", "cmd_eamm_regime", "cmd_eamm_backtest", "register"]
