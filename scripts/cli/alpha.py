"""`nat alpha` — alpha research pipeline (Steps 2-9) + pipeline orchestrator."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import DATA_DEFAULT, _banner, _py, _sym, _data


# ── Alpha pipeline commands ──────────────────────────────────────────────────

def cmd_alpha_combine(args):
    """Run feature combination (Step 2)."""
    _banner(f"Feature Combination: {_sym(args)}")
    cmd = (f"scripts/alpha/combiner.py --data-dir {_data(args)} --symbol {_sym(args)} "
           f"--screen {getattr(args, 'screen', 'reports/alpha_screen.json')} "
           f"--top {getattr(args, 'top', 20)} "
           f"--max-corr {getattr(args, 'max_corr', 0.8)} "
           f"--timeframe {getattr(args, 'timeframe', '15min')} "
           f"--method {getattr(args, 'method', 'equal')} "
           f"--output {getattr(args, 'output', 'reports/alpha_combine.json')}")
    horizon = getattr(args, 'horizon', None)
    if horizon:
        cmd += f" --horizon {horizon}"
    _py(cmd)


def cmd_alpha_size(args):
    """Run cost-aware position sizing (Step 3)."""
    _banner("Cost-Aware Position Sizing")
    cmd = (f"scripts/alpha/position.py "
           f"--signal {args.signal} --ic {args.ic} --vol {args.vol} "
           f"--horizon-bars {getattr(args, 'horizon_bars', 16)} "
           f"--cost-multiplier {getattr(args, 'cost_multiplier', 1.5)} "
           f"--scale {getattr(args, 'scale', 1.0)} "
           f"--bar-minutes {getattr(args, 'bar_minutes', 15.0)} "
           f"--output {getattr(args, 'output', 'reports/alpha_position.json')}")
    _py(cmd)


def cmd_alpha_validate(args):
    """Run walk-forward validation (Step 4)."""
    _banner(f"Walk-Forward Validation: {_sym(args)}")
    cmd = (f"scripts/alpha/adapter.py "
           f"--signal {args.signal} --data-dir {_data(args)} --symbol {_sym(args)} "
           f"--timeframe {getattr(args, 'timeframe', '15min')} "
           f"--n-trials {getattr(args, 'n_trials', 1998)} "
           f"--entry-threshold {getattr(args, 'entry_threshold', 0.3)} "
           f"--n-splits {getattr(args, 'n_splits', 5)} "
           f"--embargo-bars {getattr(args, 'embargo_bars', 600)} "
           f"--output {getattr(args, 'output', 'reports/alpha_validation.json')}")
    directions = getattr(args, 'direction', ['long', 'short'])
    for d in directions:
        cmd += f" --direction {d}"
    _py(cmd)


def cmd_alpha_regime(args):
    """Run regime conditioning (Step 5)."""
    _banner(f"Regime Conditioning: {_sym(args)}")
    _py(f"scripts/alpha/regime_filter.py "
        f"--data {_data(args)} --symbol {_sym(args)} "
        f"--screen {getattr(args, 'screen', 'reports/alpha_screen.json')} "
        f"--timeframe {getattr(args, 'timeframe', '15min')} "
        f"--top-n {getattr(args, 'top_n', 10)} "
        f"--output {getattr(args, 'output', 'reports/alpha_regime.json')}"
        + (f" --model {args.model}" if getattr(args, 'model', None) else ""))


def cmd_alpha_multi_freq(args):
    """Run multi-frequency integration (Step 6)."""
    _banner(f"Multi-Frequency Integration: {_sym(args)}")
    _py(f"scripts/alpha/multi_freq.py "
        f"--data {_data(args)} --symbol {_sym(args)} "
        f"--timeframe {getattr(args, 'timeframe', '15min')} "
        f"--output {getattr(args, 'output', 'reports/alpha_multi_freq.json')}"
        + (f" --signal {args.signal}" if getattr(args, 'signal', None) else ""))


def cmd_alpha_portfolio(args):
    """Run portfolio assembly (Step 7)."""
    _banner("Portfolio Assembly")
    symbols = getattr(args, 'symbols', ['BTC', 'ETH', 'SOL'])
    sym_args = " ".join(f"--symbols {s}" for s in symbols) if symbols else ""
    _py(f"scripts/alpha/portfolio.py "
        f"--data {_data(args)} {sym_args} "
        f"--timeframe {getattr(args, 'timeframe', '15min')} "
        f"--output {getattr(args, 'output', 'reports/alpha_portfolio.json')}")


def cmd_alpha_paper(args):
    """Run paper trading simulation (Step 8)."""
    _banner(f"Paper Trading: {_sym(args)}")
    _py(f"scripts/alpha/paper_trader.py "
        f"--data {_data(args)} --symbol {_sym(args)} "
        f"--timeframe {getattr(args, 'timeframe', '15min')} "
        f"--backtest-sharpe {getattr(args, 'backtest_sharpe', 1.0)} "
        f"--backtest-ic {getattr(args, 'backtest_ic', 0.03)} "
        f"--output {getattr(args, 'output', 'reports/alpha_paper.json')}")


def cmd_alpha_deploy(args):
    """Run deployment readiness check or show status (Step 9)."""
    _banner("Deployment Status")
    extra = ""
    if getattr(args, 'check', False):
        extra += " --check"
    extra += f" --paper-report {getattr(args, 'paper_report', 'reports/alpha_paper.json')}"
    extra += f" --account-value {getattr(args, 'account_value', 10000.0)}"
    _py(f"scripts/alpha/deployer.py{extra}")


# ── Alpha pipeline orchestrator ─────────────────────────────────────────────

def cmd_alpha_pipeline_start(args):
    cfg = getattr(args, 'config', 'config/alpha.toml')
    _banner("Alpha Pipeline: Start")
    _py(f"scripts/alpha/alpha_pipeline.py --config {cfg} start")

def cmd_alpha_pipeline_resume(args):
    cfg = getattr(args, 'config', 'config/alpha.toml')
    extra = " --force-gate" if getattr(args, 'force_gate', False) else ""
    _py(f"scripts/alpha/alpha_pipeline.py --config {cfg} resume{extra}")

def cmd_alpha_pipeline_status(args):
    cfg = getattr(args, 'config', 'config/alpha.toml')
    _py(f"scripts/alpha/alpha_pipeline.py --config {cfg} status")

def cmd_alpha_pipeline_gates(args):
    cfg = getattr(args, 'config', 'config/alpha.toml')
    _py(f"scripts/alpha/alpha_pipeline.py --config {cfg} gates")

def cmd_alpha_pipeline_run_step(args):
    cfg = getattr(args, 'config', 'config/alpha.toml')
    _py(f"scripts/alpha/alpha_pipeline.py --config {cfg} run-step {args.step}")


def register(sub):
    alpha_p = sub.add_parser('alpha', help='Alpha research pipeline (Steps 2-9)')
    alpha_p.set_defaults(func=lambda a: alpha_p.print_help())
    asub = alpha_p.add_subparsers(dest='subcmd')
    ac = asub.add_parser('combine', help='Feature combination (Step 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Combines top IC features into a composite signal using correlation-aware
        deduplication and equal or IC-weighted averaging.

        Mathematics:
          Dedup:       drop feature j if |ρ(f_i, f_j)| > max_corr  (Spearman rank)
          Equal:       S = (1/N) Σ f_i
          IC-weighted: S = Σ (IC_i / Σ IC_j) · f_i

        Input:  Screen results JSON (from nat alpha screen or nat signal test)
        Output: Combined signal .npy + report JSON

        Example:
          nat alpha combine --symbol BTC --top 10 --method ic_weighted
        """))
    ac.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ac.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ac.add_argument('--screen', default='reports/alpha_screen.json', help='Screen results JSON')
    ac.add_argument('--horizon', default=None, help='Filter to specific horizon')
    ac.add_argument('--top', type=int, default=20, help='Top N features')
    ac.add_argument('--max-corr', type=float, default=0.8, help='Correlation dedup threshold')
    ac.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe (default: 15min)')
    ac.add_argument('--method', default='equal', choices=['equal', 'ic_weighted'], help='Combination method')
    ac.add_argument('--output', default='reports/alpha_combine.json', help='Output JSON path')
    ac.set_defaults(func=cmd_alpha_combine)
    az = asub.add_parser('size', help='Cost-aware position sizing (Step 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Computes optimal position size using Kelly criterion adjusted for costs.

        Mathematics:
          Kelly fraction: f* = IC / σ_r
          Cost-adjusted:  f_adj = f* × (1 - cost / edge)
          Position size:  q = f_adj × scale × account_value
          Edge estimate:  edge = IC × σ_r × √(horizon_bars / bar_minutes)

        Example:
          nat alpha size --signal signal.npy --ic 0.05 --vol 0.02
        """))
    az.add_argument('--signal', required=True, help='Signal .npy file')
    az.add_argument('--ic', type=float, required=True, help='Estimated IC')
    az.add_argument('--vol', type=float, required=True, help='Return volatility')
    az.add_argument('--horizon-bars', type=int, default=16, help='Holding period in bars')
    az.add_argument('--cost-multiplier', type=float, default=1.5, help='Cost safety multiplier')
    az.add_argument('--scale', type=float, default=1.0, help='Position scale factor')
    az.add_argument('--bar-minutes', type=float, default=15.0, help='Bar duration in minutes')
    az.add_argument('--output', default='reports/alpha_position.json', help='Output JSON path')
    az.set_defaults(func=cmd_alpha_size)
    av = asub.add_parser('validate', help='Walk-forward validation (Step 4)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Walk-forward cross-validation with permutation-based significance testing.

        Mathematics:
          CV:              5-fold time-series split with embargo bars between folds
          Embargo:         gap of N bars between train/test to prevent leakage
          Sharpe (OOS):    S = (μ_oos / σ_oos) × √(252 × bars_per_day)
          Significance:    permutation test (n_trials shuffles), p = rank(S_real) / n_trials
          Deflated Sharpe: adjusts for multiple testing: DSR = Φ(z_adj)
            where z_adj accounts for number of strategies tried

        Example:
          nat alpha validate --signal signal.npy --n-splits 5 --embargo-bars 600
        """))
    av.add_argument('--signal', required=True, help='Signal .npy file')
    av.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    av.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    av.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    av.add_argument('--n-trials', type=int, default=1998, help='Number of permutation trials')
    av.add_argument('--entry-threshold', type=float, default=0.3, help='Signal entry threshold')
    av.add_argument('--n-splits', type=int, default=5, help='Walk-forward splits')
    av.add_argument('--embargo-bars', type=int, default=600, help='Embargo bars between folds')
    av.add_argument('--direction', nargs='+', default=['long', 'short'], choices=['long', 'short'], help='Trade direction(s)')
    av.add_argument('--output', default='reports/alpha_validation.json', help='Output JSON path')
    av.set_defaults(func=cmd_alpha_validate)
    ar = asub.add_parser('regime', help='Regime conditioning (Step 5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Stratifies signal IC by regime variable quintiles to identify
        conditions where the signal is strongest.

        Mathematics:
          Regime partition: Q_k = {t : percentile_{k-1} < regime_t ≤ percentile_k}, k=1..5
          Per-quintile IC:  IC_k = Spearman ρ(signal, return | t ∈ Q_k)
          IC ratio:         IC_ratio = max_k(IC_k) / IC_baseline
          Gate:             only trade when regime ∈ best quintile(s)

        Example:
          nat alpha regime --symbol BTC --top-n 10
        """))
    ar.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ar.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ar.add_argument('--screen', default='reports/alpha_screen.json', help='Screen results JSON')
    ar.add_argument('--model', default=None, help='Regime GMM model JSON')
    ar.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    ar.add_argument('--top-n', type=int, default=10, help='Top N features to condition')
    ar.add_argument('--output', default='reports/alpha_regime.json', help='Output JSON path')
    ar.set_defaults(func=cmd_alpha_regime)
    amf = asub.add_parser('multi-freq', help='Multi-frequency integration (Step 6)')
    amf.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    amf.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    amf.add_argument('--signal', default=None, help='Pre-computed signal JSON')
    amf.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    amf.add_argument('--output', default='reports/alpha_multi_freq.json', help='Output JSON path')
    amf.set_defaults(func=cmd_alpha_multi_freq)
    ap = asub.add_parser('portfolio', help='Portfolio assembly (Step 7)')
    ap.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ap.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL'], help='Symbols to include')
    ap.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    ap.add_argument('--output', default='reports/alpha_portfolio.json', help='Output JSON path')
    ap.set_defaults(func=cmd_alpha_portfolio)
    apt = asub.add_parser('paper', help='Paper trading simulation (Step 8)')
    apt.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    apt.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    apt.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    apt.add_argument('--backtest-sharpe', type=float, default=1.0, help='Min backtest Sharpe to simulate')
    apt.add_argument('--backtest-ic', type=float, default=0.03, help='Min backtest IC to simulate')
    apt.add_argument('--output', default='reports/alpha_paper.json', help='Output JSON path')
    apt.set_defaults(func=cmd_alpha_paper)
    adp = asub.add_parser('deploy', help='Deployment status & readiness (Step 9)')
    adp.add_argument('--check', action='store_true', help='Run readiness check')
    adp.add_argument('--paper-report', default='reports/alpha_paper.json', help='Paper trading report JSON')
    adp.add_argument('--account-value', type=float, default=10000.0, help='Account value in USD')
    adp.set_defaults(func=cmd_alpha_deploy)
    # Alpha pipeline orchestrator (runs all 9 steps with quality gates)
    apl_start = asub.add_parser('pipeline-start', help='Start fresh alpha pipeline run (all 9 steps)')
    apl_start.add_argument('--config', default='config/alpha.toml', help='Alpha config TOML')
    apl_start.set_defaults(func=cmd_alpha_pipeline_start)
    apl_resume = asub.add_parser('pipeline-resume', help='Resume alpha pipeline from last phase')
    apl_resume.add_argument('--config', default='config/alpha.toml', help='Alpha config TOML')
    apl_resume.add_argument('--force-gate', action='store_true', help='Continue past GATE_FAILED')
    apl_resume.set_defaults(func=cmd_alpha_pipeline_resume)
    apl_status = asub.add_parser('pipeline-status', help='Alpha pipeline state + gate verdicts')
    apl_status.add_argument('--config', default='config/alpha.toml', help='Alpha config TOML')
    apl_status.set_defaults(func=cmd_alpha_pipeline_status)
    apl_gates = asub.add_parser('pipeline-gates', help='Detailed gate report with metrics')
    apl_gates.add_argument('--config', default='config/alpha.toml', help='Alpha config TOML')
    apl_gates.set_defaults(func=cmd_alpha_pipeline_gates)
    apl_step = asub.add_parser('pipeline-step', help='Run single alpha pipeline step (1-9)')
    apl_step.add_argument('step', type=int, help='Step number (1-9)')
    apl_step.add_argument('--config', default='config/alpha.toml', help='Alpha config TOML')
    apl_step.set_defaults(func=cmd_alpha_pipeline_run_step)


__all__ = [
    "cmd_alpha_combine", "cmd_alpha_size", "cmd_alpha_validate", "cmd_alpha_regime",
    "cmd_alpha_multi_freq", "cmd_alpha_portfolio", "cmd_alpha_paper", "cmd_alpha_deploy",
    "cmd_alpha_pipeline_start", "cmd_alpha_pipeline_resume", "cmd_alpha_pipeline_status",
    "cmd_alpha_pipeline_gates", "cmd_alpha_pipeline_run_step",
    "register",
]
