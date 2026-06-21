"""`nat spannung` — Spannung signal grid search + backtest/horizon/spectral/regime."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import _banner, _py, _data


def cmd_spannung(args):
    """Run Spannung offline grid search."""
    _banner("Spannung Grid Search")
    data = _data(args)
    sym = getattr(args, 'symbol', 'all')
    top = getattr(args, 'top', 20)
    cmd = f"scripts/spannung_grid.py --data-dir {data} --symbol {sym} --top {top}"
    horizons = getattr(args, 'horizons', None)
    if horizons:
        cmd += " --horizons " + " ".join(str(h) for h in horizons)
    output = getattr(args, 'output', None)
    if output:
        cmd += f" --output {output}"
    _py(cmd)


def cmd_spannung_backtest(args):
    """Run Spannung cost-aware backtest + regime gating."""
    _banner("Spannung Backtest + Regime Gating")
    data = _data(args)
    sym = getattr(args, 'symbol', 'all')
    cmd = f"scripts/spannung_backtest.py --data-dir {data} --symbol {sym}"
    horizon = getattr(args, 'horizon', None)
    if horizon:
        cmd += f" --horizon {horizon}"
    _py(cmd)


def cmd_spannung_horizon(args):
    """Run Spannung longer-horizon sweep (30s–15min bars)."""
    _banner("Spannung Horizon Sweep")
    data = _data(args)
    sym = getattr(args, 'symbol', 'all')
    cmd = f"scripts/spannung_horizon_sweep.py --data-dir {data} --symbol {sym}"
    output = getattr(args, 'output', None)
    if output:
        cmd += f" --output {output}"
    _py(cmd)


def cmd_spannung_spectral(args):
    """Run Spannung spectral analysis (PSD, coherence, ACF, band IC)."""
    _banner("Spannung Spectral Analysis")
    data = _data(args)
    sym = getattr(args, 'symbol', 'all')
    cmd = f"scripts/spannung_spectral.py --data-dir {data} --symbol {sym}"
    output = getattr(args, 'output', None)
    if output:
        cmd += f" --output {output}"
    _py(cmd)


def cmd_spannung_regime(args):
    """Run Spannung regime screener (systematic condition search)."""
    _banner("Spannung Regime Screener")
    data = _data(args)
    sym = getattr(args, 'symbol', 'all')
    cmd = f"scripts/spannung_regime_screener.py --data-dir {data} --symbol {sym}"
    output = getattr(args, 'output', None)
    if output:
        cmd += f" --output {output}"
    _py(cmd)


def register(sub):
    spn_p = sub.add_parser('spannung', help='Spannung signal grid search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Exhaustive grid search over (feature × horizon) pairs computing IC, t-stat,
        and cost-adjusted edge for scalping signal discovery.

        Mathematics:
          IC grid:       IC(f, h) = Spearman ρ(f_t, r_t(h))  for all (f, h) pairs
          t-statistic:   t = IC · √n
          Edge:          edge = |IC| × σ_r × √(h / dt)
          Cost filter:   edge > spread_bps / 10000

        Subcommands:
          backtest    Cost-aware backtest with regime gating on top signals
          horizon     Longer-horizon sweep (30s to 15min aggregated bars)
          spectral    PSD, cross-coherence, ACF, and frequency-band IC
          regime      Systematic regime condition search (quintile × Pareto)

        Example:
          nat spannung --symbol BTC --top 20 --horizons 10 50 100 300
        """))
    spn_p.add_argument('--data', type=str, default=None, help='Data dir (auto-detects latest)')
    spn_p.add_argument('--symbol', type=str, default='all', help='Symbol or "all" (default: all)')
    spn_p.add_argument('--horizons', type=int, nargs='+', default=None,
                       help='Forward horizons in ticks (1 tick=100ms, default: 10 50 100 300 600)')
    spn_p.add_argument('--top', type=int, default=20, help='Top N results to display')
    spn_p.add_argument('--output', type=str, default=None, help='Output directory')
    spn_p.set_defaults(func=cmd_spannung)
    spn_sub = spn_p.add_subparsers(dest='spn_subcmd')
    spn_bt = spn_sub.add_parser('backtest', help='Cost-aware backtest + regime gating')
    spn_bt.add_argument('--data', type=str, default=None, help='Data dir')
    spn_bt.add_argument('--symbol', type=str, default='all', help='Symbol or "all"')
    spn_bt.add_argument('--horizon', type=int, default=None, help='Horizon in ticks')
    spn_bt.set_defaults(func=cmd_spannung_backtest)
    spn_hz = spn_sub.add_parser('horizon', help='Longer-horizon sweep (30s–15min bars)')
    spn_hz.add_argument('--data', type=str, default=None, help='Data dir')
    spn_hz.add_argument('--symbol', type=str, default='all', help='Symbol or "all"')
    spn_hz.add_argument('--output', type=str, default=None, help='Output directory')
    spn_hz.set_defaults(func=cmd_spannung_horizon)
    spn_sp = spn_sub.add_parser('spectral', help='Spectral analysis (PSD, coherence, ACF, band IC)')
    spn_sp.add_argument('--data', type=str, default=None, help='Data dir')
    spn_sp.add_argument('--symbol', type=str, default='all', help='Symbol or "all"')
    spn_sp.add_argument('--output', type=str, default=None, help='Output directory')
    spn_sp.set_defaults(func=cmd_spannung_spectral)
    spn_rg = spn_sub.add_parser('regime', help='Systematic regime condition screener')
    spn_rg.add_argument('--data', type=str, default=None, help='Data dir')
    spn_rg.add_argument('--symbol', type=str, default='all', help='Symbol or "all"')
    spn_rg.add_argument('--output', type=str, default=None, help='Output directory')
    spn_rg.set_defaults(func=cmd_spannung_regime)


__all__ = ["cmd_spannung", "cmd_spannung_backtest", "cmd_spannung_horizon",
           "cmd_spannung_spectral", "cmd_spannung_regime", "register"]
