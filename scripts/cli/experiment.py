"""`nat experiment` — experiment tracking (list/get/compare/best/workflow/snapshot)."""

from __future__ import annotations

import os

from cli.common import DATA_DEFAULT, _py, _banner, _data


def cmd_experiment_list(args):
    stage = getattr(args, 'stage', None)
    cmd = "scripts/experiment_tracking.py list"
    if stage:
        cmd += f" --stage {stage}"
    _py(cmd)


def cmd_experiment_get(args):
    _py(f"scripts/experiment_tracking.py get {args.id}")


def cmd_experiment_compare(args):
    _py(f"scripts/experiment_tracking.py compare {args.ids}")


def cmd_experiment_best(args):
    metric = getattr(args, 'metric', 'sharpe_ratio')
    _py(f"scripts/experiment_tracking.py best --metric {metric}")


def cmd_experiment_workflow(args):
    snapshot = getattr(args, 'snapshot', 'baseline_30d')
    model_type = getattr(args, 'type', 'elasticnet')
    entry = getattr(args, 'entry', 0.001)
    exit_ = getattr(args, 'exit', 0.0)
    predictions = getattr(args, 'predictions', './predictions.parquet')
    output = getattr(args, 'output', './backtest_results.json')

    _banner("Complete ML workflow with tracking")
    print("  Step 1: Training model...")
    os.makedirs("models", exist_ok=True)
    _py(f"scripts/train_baseline.py --snapshot {snapshot} --model {model_type} --output-dir models")
    print("\n  Step 2: Scoring...")
    _py(f"scripts/score_data.py --model models/{model_type}_*.* --data {_data(args)} "
        f"--output {predictions} --evaluate")
    print("\n  Step 3: Backtest with tracking...")
    _py(f"scripts/run_backtest_tracked.py --ml-predictions {predictions} "
        f"--ml-entry-threshold {entry} --ml-exit-threshold {exit_} --walk-forward --output {output}")


def cmd_experiment_snapshot(args):
    name = args.name
    cmd = f"scripts/experiment_governance.py snapshot --data-dir {_data(args)} --name {name}"
    desc = getattr(args, 'description', None)
    if desc:
        cmd += f" --description \"{desc}\""
    _banner(f"Creating snapshot: {name}")
    _py(cmd)


def register(sub):
    exp_p = sub.add_parser('experiment', help='Experiment tracking')
    exp_p.set_defaults(func=lambda a: exp_p.print_help())
    esub = exp_p.add_subparsers(dest='subcmd')
    el = esub.add_parser('list', help='List experiments')
    el.add_argument('--stage', help='Filter by stage')
    el.set_defaults(func=cmd_experiment_list)
    eg = esub.add_parser('get', help='Get experiment details')
    eg.add_argument('--id', required=True, help='Experiment ID')
    eg.set_defaults(func=cmd_experiment_get)
    ec = esub.add_parser('compare', help='Compare experiments')
    ec.add_argument('--ids', required=True, help='Space-separated experiment IDs')
    ec.set_defaults(func=cmd_experiment_compare)
    eb = esub.add_parser('best', help='Find best')
    eb.add_argument('--metric', default='sharpe_ratio', help='Metric to rank by')
    eb.set_defaults(func=cmd_experiment_best)
    ew = esub.add_parser('workflow', help='Full ML workflow')
    ew.add_argument('--snapshot', default='baseline_30d', help='Dataset snapshot name')
    ew.add_argument('--type', default='elasticnet', help='Model type (elasticnet/lightgbm)')
    ew.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ew.add_argument('--entry', type=float, default=0.001, help='Entry signal threshold')
    ew.add_argument('--exit', type=float, default=0.0, help='Exit signal threshold')
    ew.add_argument('--predictions', default='./predictions.parquet', help='Predictions parquet path')
    ew.add_argument('--output', default='./backtest_results.json', help='Output results JSON')
    ew.set_defaults(func=cmd_experiment_workflow)
    esn = esub.add_parser('snapshot', help='Create dataset snapshot')
    esn.add_argument('--name', required=True, help='Snapshot name')
    esn.add_argument('--description', help='Snapshot description')
    esn.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    esn.set_defaults(func=cmd_experiment_snapshot)


__all__ = ["cmd_experiment_list", "cmd_experiment_get", "cmd_experiment_compare",
           "cmd_experiment_best", "cmd_experiment_workflow", "cmd_experiment_snapshot",
           "register"]
