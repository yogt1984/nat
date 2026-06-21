"""`nat backtest` — event-driven backtesting (strategies, ML predictions, algorithm signals)."""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import (
    ROOT, PY, DATA_DEFAULT, R, W, _banner, _py, _sh, _sym, _data,
    _json_mode, _output,
)


# ── Backtest commands ────────────────────────────────────────────────────────

def cmd_backtest(args):
    strategy = getattr(args, 'strategy', 'whale_flow_simple')
    _banner(f"Backtest: {strategy}")
    _py(f"scripts/run_backtest.py --data-dir {_data(args)} --symbol {_sym(args)} --strategy {strategy}")


def cmd_backtest_validate(args):
    strategy = getattr(args, 'strategy', 'whale_flow_simple')
    _banner(f"Walk-forward validation: {strategy}")
    _py(
        f"scripts/run_backtest.py --data-dir {_data(args)} --symbol {_sym(args)} "
        f"--strategy {strategy} --walk-forward"
    )


def _ml_args(args):
    pred = getattr(args, 'predictions', './predictions.parquet')
    entry = getattr(args, 'entry', 0.001)
    exit_ = getattr(args, 'exit', 0.0)
    return pred, entry, exit_


def cmd_backtest_ml(args):
    pred, entry, exit_ = _ml_args(args)
    _banner("ML model backtest")
    _py(
        f"scripts/run_backtest.py --data-dir {_data(args)} --symbol {_sym(args)} "
        f"--ml-predictions {pred} --ml-entry-threshold {entry} --ml-exit-threshold {exit_}"
    )


def cmd_backtest_ml_validate(args):
    pred, entry, exit_ = _ml_args(args)
    _banner("ML walk-forward validation")
    _py(
        f"scripts/run_backtest.py --data-dir {_data(args)} --symbol {_sym(args)} "
        f"--ml-predictions {pred} --ml-entry-threshold {entry} --ml-exit-threshold {exit_} "
        f"--walk-forward"
    )


def cmd_backtest_ml_quantile(args):
    pred = getattr(args, 'predictions', './predictions.parquet')
    entry_q = getattr(args, 'entry_q', 0.75)
    exit_q = getattr(args, 'exit_q', 0.50)
    _banner("ML backtest (quantile)")
    _py(
        f"scripts/run_backtest.py --data-dir {_data(args)} --symbol {_sym(args)} "
        f"--ml-predictions {pred} --ml-quantile "
        f"--ml-entry-threshold {entry_q} --ml-exit-threshold {exit_q} --walk-forward"
    )


def cmd_backtest_ml_tracked(args):
    pred, entry, exit_ = _ml_args(args)
    direction = getattr(args, 'direction', 'long')
    output = getattr(args, 'output', './backtest_results.json')
    _banner("ML backtest with tracking")
    _py(
        f"scripts/run_backtest_tracked.py "
        f"--ml-predictions {pred} --ml-entry-threshold {entry} --ml-exit-threshold {exit_} "
        f"--ml-direction {direction} --walk-forward --output {output}"
    )


def cmd_backtest_list(args):
    if _json_mode(args):
        # Parse strategy list from backtest module
        r = _sh(f"{PY} scripts/run_backtest.py --list-strategies")
        strategies = [l.strip().lstrip("- ") for l in r.stdout.strip().splitlines() if l.strip().startswith("-")]
        _output({"strategies": strategies}, args)
        return
    _py("scripts/run_backtest.py --list-strategies")


# ── Funding reversion backtest ──────────────────────────────────────────────

def cmd_backtest_funding(args):
    _banner("Backtest: Funding Rate Reversion")
    _py("scripts/backtest_funding_reversion.py")


def cmd_backtest_algorithm(args):
    """Run backtest using algorithm features as signals."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from algorithms.autodiscover import discover_all
    discover_all()
    from backtest.algorithm_strategy import run_algorithm_backtest

    alg_name = args.algorithm
    feature = getattr(args, 'feature', None)
    entry = getattr(args, 'entry_threshold', 0.3)
    exit_ = getattr(args, 'exit_threshold', -0.1)

    _banner(f"Algorithm backtest: {alg_name}")

    result = run_algorithm_backtest(
        algorithm_name=alg_name,
        data_dir=str(_data(args)),
        symbol=_sym(args),
        feature_col=feature,
        entry_threshold=entry,
        exit_threshold=exit_,
    )

    if _json_mode(args):
        _output(result, args)
        return

    if "error" in result:
        print(f"  {R}Error:{W} {result['error']}")
        return 1

    print(f"  Algorithm:   {result['algorithm']}")
    print(f"  Feature:     {result['feature']}")
    print(f"  Symbol:      {result['symbol']}")
    print(f"  Ticks:       {result['n_ticks']:,}")
    print(f"  Trades:      {result['n_trades']:,}")
    print(f"  Total ret:   {result['total_return_bps']:+.2f} bps")
    print(f"  Sharpe:      {result['sharpe']:+.3f}")
    print(f"  Max DD:      {result['max_drawdown_bps']:.2f} bps")
    print(f"  Win rate:    {result['win_rate']:.1%}")
    print(f"  Profit fac:  {result['profit_factor']:.3f}")
    print(f"  Avg/trade:   {result['avg_return_per_trade_bps']:+.4f} bps")
    print(f"  Total cost:  {result['cost_bps']:.2f} bps")


def register(sub):
    # ── backtest ──
    bt_p = sub.add_parser('backtest', help='Backtesting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Event-driven backtesting framework with walk-forward validation support.

        Metrics:
          Sharpe ratio:    S = (μ / σ) × √252
          Max drawdown:    MDD = min_t((equity_t - peak_t) / peak_t)
          Profit factor:   PF = Σ(winning_trades) / |Σ(losing_trades)|
          Win rate:        W = n_winners / n_total
          Calmar ratio:    C = annualized_return / |MDD|

        Walk-forward: trains on expanding window, tests on next fold.
        ML mode: uses pre-computed predictions from model scoring.

        Example:
          nat backtest --strategy whale_flow_simple --symbol BTC
          nat backtest ml --predictions predictions.parquet --entry 0.001
        """))
    bt_p.add_argument('--strategy', default='whale_flow_simple', help='Strategy name (default: whale_flow_simple)')
    bt_p.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    bt_p.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    bt_p.set_defaults(func=cmd_backtest)
    btsub = bt_p.add_subparsers(dest='subcmd')
    btsub.add_parser('validate', help='Walk-forward validation').set_defaults(func=cmd_backtest_validate)
    bml = btsub.add_parser('ml', help='ML predictions backtest')
    bml.add_argument('--predictions', default='./predictions.parquet', help='ML predictions parquet file')
    bml.add_argument('--entry', type=float, default=0.001, help='Entry signal threshold')
    bml.add_argument('--exit', type=float, default=0.0, help='Exit signal threshold')
    bml.set_defaults(func=cmd_backtest_ml)
    bmlv = btsub.add_parser('ml-validate', help='ML walk-forward')
    bmlv.add_argument('--predictions', default='./predictions.parquet', help='ML predictions parquet file')
    bmlv.add_argument('--entry', type=float, default=0.001, help='Entry signal threshold')
    bmlv.add_argument('--exit', type=float, default=0.0, help='Exit signal threshold')
    bmlv.set_defaults(func=cmd_backtest_ml_validate)
    bmlq = btsub.add_parser('ml-quantile', help='ML quantile thresholds')
    bmlq.add_argument('--predictions', default='./predictions.parquet', help='ML predictions parquet file')
    bmlq.add_argument('--entry-q', type=float, default=0.75, help='Entry quantile threshold')
    bmlq.add_argument('--exit-q', type=float, default=0.50, help='Exit quantile threshold')
    bmlq.set_defaults(func=cmd_backtest_ml_quantile)
    bmlt = btsub.add_parser('ml-tracked', help='ML with tracking')
    bmlt.add_argument('--predictions', default='./predictions.parquet', help='ML predictions parquet file')
    bmlt.add_argument('--entry', type=float, default=0.001, help='Entry signal threshold')
    bmlt.add_argument('--exit', type=float, default=0.0, help='Exit signal threshold')
    bmlt.add_argument('--direction', default='long', help='Trade direction (long/short)')
    bmlt.add_argument('--output', default='./backtest_results.json', help='Output results JSON')
    bmlt.set_defaults(func=cmd_backtest_ml_tracked)
    btsub.add_parser('list', help='List strategies').set_defaults(func=cmd_backtest_list)
    btsub.add_parser('funding', help='Funding rate reversion backtest').set_defaults(func=cmd_backtest_funding)
    balg = btsub.add_parser('algorithm', help='Backtest using algorithm features as signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Runs a registered algorithm on historical data, then backtests the
        resulting signal with configurable entry/exit thresholds and cost model.

        The algorithm feature is used as a continuous signal:
          Enter long  when signal > entry_threshold
          Enter short when signal < -entry_threshold
          Exit when |signal| < |exit_threshold| or sign flips

        Cost model: (maker_fee + taker_fee) / 2 per position change.

        Example:
          nat backtest algorithm --algorithm weighted_ofi --symbol BTC
          nat backtest algorithm --algorithm hawkes_intensity --feature alg_bid_ask_hawkes_imbalance
        """))
    balg.add_argument('--algorithm', required=True, help='Algorithm name')
    balg.add_argument('--feature', default=None, help='Feature column for signal (default: first)')
    balg.add_argument('--entry-threshold', type=float, default=0.3, help='Entry threshold (default: 0.3)')
    balg.add_argument('--exit-threshold', type=float, default=-0.1, help='Exit threshold (default: -0.1)')
    balg.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    balg.set_defaults(func=cmd_backtest_algorithm)


__all__ = [
    "cmd_backtest", "cmd_backtest_validate", "cmd_backtest_ml",
    "cmd_backtest_ml_validate", "cmd_backtest_ml_quantile", "cmd_backtest_ml_tracked",
    "cmd_backtest_list", "cmd_backtest_funding", "cmd_backtest_algorithm",
    "register",
]
