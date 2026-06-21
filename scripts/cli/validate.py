"""`nat validate` — validation suites (skeptical battery, regression test)."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import DATA_DEFAULT, _banner, _py, _data, _sym


def cmd_validate_skeptical(args):
    _banner("Running skeptical validation suite")
    output = getattr(args, 'output', 'reports/skeptical_validation')
    _py(f"scripts/exploration/skeptical_validation.py --data {_data(args)} --output {output}")


def cmd_validate_regression(args):
    _banner(f"Skeptical Regression Test: {_sym(args)}")
    cmd = (f"scripts/exploration/skeptical_regression_test.py --data-dir {_data(args)} "
           f"--symbol {_sym(args)} --horizon {getattr(args, 'horizon', 18000)}")
    report = getattr(args, 'json_report', None)
    if report:
        cmd += f" --json-report {report}"
    start = getattr(args, 'start_date', None)
    if start:
        cmd += f" --start-date {start}"
    end = getattr(args, 'end_date', None)
    if end:
        cmd += f" --end-date {end}"
    n_perm = getattr(args, 'n_permutations', None)
    if n_perm:
        cmd += f" --n-permutations {n_perm}"
    _py(cmd)


def register(sub):
    val_p = sub.add_parser('validate', help='Validation suites')
    val_p.set_defaults(func=lambda a: val_p.print_help())
    vsub = val_p.add_subparsers(dest='subcmd')
    vs = vsub.add_parser('skeptical', help='20+ statistical tests before investment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Comprehensive statistical validation suite (20+ tests) designed to catch
        overfitting, data-snooping, and spurious signals before capital deployment.

        Tests include:
          Runs test:       H0: sequence is random (Wald-Wolfowitz)
          Variance ratio:  VR(k) = Var(r_k) / (k·Var(r_1)),  H0: VR=1 (random walk)
          Ljung-Box:       Q = n(n+2) Σ (ρ_k²/(n-k)),  H0: no autocorrelation
          BDS test:        nonlinear dependence in residuals
          Hurst exponent:  H via R/S analysis,  H>0.5 = trending, H<0.5 = mean-reverting
          Deflated Sharpe: p-value adjusted for number of strategies tried
          Permutation:     empirical p-value from n_trials random shuffles

        Example:
          nat validate skeptical --data data/features --output reports/skeptical
        """))
    vs.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    vs.add_argument('--output', default='reports/skeptical_validation', help='Output directory')
    vs.set_defaults(func=cmd_validate_skeptical)
    vr = vsub.add_parser('regression', help='Skeptical regression signal test battery (10 tests)')
    vr.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    vr.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    vr.add_argument('--horizon', type=int, default=18000, help='Forward horizon in ticks (default: 18000)')
    vr.add_argument('--json-report', default=None, help='Output JSON report path')
    vr.add_argument('--start-date', default=None, help='Start date filter (YYYY-MM-DD)')
    vr.add_argument('--end-date', default=None, help='End date filter (YYYY-MM-DD)')
    vr.add_argument('--n-permutations', type=int, default=200, help='Permutation test iterations')
    vr.set_defaults(func=cmd_validate_regression)


__all__ = ["cmd_validate_skeptical", "cmd_validate_regression", "register"]
