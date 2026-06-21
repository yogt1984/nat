"""`nat model` — model training, scoring, and serving."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from cli.common import DATA_DEFAULT, R, W, _banner, _py, _exec, _p, _data, _sym


def cmd_model_train(args):
    snapshot = getattr(args, 'snapshot', 'baseline_30d')
    model_type = getattr(args, 'type', 'elasticnet')
    output_dir = getattr(args, 'output_dir', './models')
    _banner(f"Training baseline model: {model_type}")
    os.makedirs(output_dir, exist_ok=True)
    _py(f"scripts/train_baseline.py --snapshot {snapshot} --model {model_type} --output-dir {output_dir}")


def cmd_model_train_gmm(args):
    auto = getattr(args, 'auto', False)
    _banner("Training GMM regime classifier")
    os.makedirs("models", exist_ok=True)
    cmd = f"scripts/train_regime_gmm.py --data-dir {_data(args)} --output-dir models"
    if auto:
        cmd += " --auto-select"
    _py(cmd)


def cmd_model_train_hier(args):
    symbol = _sym(args)
    horizon = getattr(args, 'horizon', 60)
    dry_run = getattr(args, 'dry_run', False)
    start = getattr(args, 'start_date', None)
    l1_thresh = getattr(args, 'l1_threshold', 0.5)
    ablation = getattr(args, 'ablation', False)
    val_mode = getattr(args, 'validation_mode', 'walk_forward')
    _banner(f"Training Hierarchical Signal Combiner: {symbol}")
    os.makedirs("models/hierarchical_combiner", exist_ok=True)
    cmd = (f"scripts/train_hierarchical.py --symbol {symbol} "
           f"--data-dir {_data(args)} --horizon {horizon} "
           f"--l1-threshold {l1_thresh} --validation-mode {val_mode}")
    if dry_run:
        cmd += " --dry-run"
    if ablation:
        cmd += " --ablation"
    if start:
        cmd += f" --start-date {start}"
    _py(cmd)


def cmd_model_list(args):
    model_dir = getattr(args, 'model_dir', './models')
    _py(f"scripts/list_models.py --model-dir {model_dir}")


def cmd_model_score(args):
    model = getattr(args, 'model', 'models/latest.pkl')
    save = getattr(args, 'save', False)
    output = getattr(args, 'output', './predictions.parquet')
    cmd = f"scripts/score_data.py --model {model} --data {_data(args)} --evaluate"
    if save:
        cmd += f" --output {output}"
    _py(cmd)


def cmd_model_serve(args):
    try:
        import fastapi, uvicorn, pydantic  # noqa: F401
    except ImportError:
        _p("x", R, "Missing serving dependencies. Install with:")
        _p(" ", W, "pip install -r scripts/requirements-serving.txt")
        sys.exit(1)

    port = getattr(args, 'port', 8000)
    host = getattr(args, 'host', '0.0.0.0')
    cache = getattr(args, 'cache_size', 5)
    dev = getattr(args, 'dev', False)
    best = getattr(args, 'best', False)
    metric = getattr(args, 'metric', 'sharpe_ratio')

    if dev:
        _banner("Model serving API (dev mode)")
        _exec(f"uvicorn scripts.model_serving:app --reload --host {host} --port {port}")
    elif best:
        _banner(f"Serving best model by {metric}")
        _py(f"scripts/model_serving.py --host {host} --port {port} --serve-best "
            f"--metric {metric} --cache-size {cache}")
    else:
        _banner("Model serving API")
        _py(f"scripts/model_serving.py --host {host} --port {port} --cache-size {cache}")


def register(sub):
    model_p = sub.add_parser('model', help='Model training & serving')
    model_p.set_defaults(func=lambda a: model_p.print_help())
    msub = model_p.add_subparsers(dest='subcmd')
    mt = msub.add_parser('train', help='Train baseline model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Trains a supervised model to predict forward returns from features.

        Models:
          ElasticNet:  min (1/2n)·‖y - Xβ‖² + λ₁·‖β‖₁ + λ₂·‖β‖²₂
                       (L1 for sparsity + L2 for stability, α and l1_ratio via CV)
          LightGBM:    gradient boosted decision trees with leaf-wise growth
                       (handles non-linearity, feature interactions, missing values)

        Data: reads from dataset snapshots (pre-split train/val/test).
        Output: serialized model + feature importance + CV metrics.

        Example:
          nat model train --snapshot baseline_30d --type elasticnet
          nat model train --type lightgbm --output-dir models/
        """))
    mt.add_argument('--snapshot', default='baseline_30d', help='Dataset snapshot name')
    mt.add_argument('--type', default='elasticnet', help='Model type (elasticnet/lightgbm)')
    mt.add_argument('--output-dir', default='./models', help='Model output directory')
    mt.set_defaults(func=cmd_model_train)
    mg = msub.add_parser('train-gmm', help='Train GMM classifier')
    mg.add_argument('--auto', action='store_true', help='Auto BIC selection')
    mg.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    mg.set_defaults(func=cmd_model_train_gmm)
    mh = msub.add_parser('train-hier', help='Train hierarchical signal combiner')
    mh.add_argument('--symbol', default='BTC', help='Symbol (default: BTC)')
    mh.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    mh.add_argument('--horizon', type=int, default=60, help='Forward return horizon in bars (default: 60 = 5h)')
    mh.add_argument('--l1-threshold', type=float, default=0.5, help='L1 z-score gate threshold')
    mh.add_argument('--start-date', default=None, help='Earliest date (YYYY-MM-DD)')
    mh.add_argument('--dry-run', action='store_true', help='Evaluate only, no save')
    mh.add_argument('--ablation', action='store_true', help='Run layer ablation analysis (auto with --dry-run)')
    mh.add_argument('--validation-mode', choices=['walk_forward', 'purged_kfold'], default='walk_forward',
                    help='Validation method (default: walk_forward)')
    mh.set_defaults(func=cmd_model_train_hier)
    ml = msub.add_parser('list', help='List models')
    ml.add_argument('--model-dir', default='./models', help='Model directory')
    ml.set_defaults(func=cmd_model_list)
    ms = msub.add_parser('score', help='Score data')
    ms.add_argument('--model', default='models/latest.pkl', help='Model file path')
    ms.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ms.add_argument('--save', action='store_true', help='Save predictions to file')
    ms.add_argument('--output', default='./predictions.parquet', help='Predictions output path')
    ms.set_defaults(func=cmd_model_score)
    mv = msub.add_parser('serve', help='Model serving API')
    mv.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    mv.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    mv.add_argument('--cache-size', type=int, default=5, help='Model cache size')
    mv.add_argument('--dev', action='store_true', help='Hot-reload mode')
    mv.add_argument('--best', action='store_true', help='Serve best model')
    mv.add_argument('--metric', default='sharpe_ratio', help='Metric to select best model')
    mv.set_defaults(func=cmd_model_serve)


__all__ = ["cmd_model_train", "cmd_model_train_gmm", "cmd_model_train_hier",
           "cmd_model_list", "cmd_model_score", "cmd_model_serve", "register"]
