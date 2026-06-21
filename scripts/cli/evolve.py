"""`nat evolve` — evolutionary config optimization (Tier 3 — Optuna)."""

from __future__ import annotations

from cli.common import _py


def cmd_evolve_start(args):
    """Start evolutionary optimization."""
    cmd = (f"scripts/swarm/optuna_optimizer.py optimize "
           f"--study {args.study} --sampler {args.sampler} "
           f"--trials {args.trials} --jobs {args.jobs} "
           f"--symbol {args.symbol} --hours {args.hours} "
           f"--train-frac {args.train_frac}")
    if args.seed is not None:
        cmd += f" --seed {args.seed}"
    if args.storage:
        cmd += f" --storage {args.storage}"
    if args.timeout is not None:
        cmd += f" --timeout {args.timeout}"
    if args.no_guard_rails:
        cmd += " --no-guard-rails"
    _py(cmd)


def cmd_evolve_status(args):
    """Show Optuna study status."""
    cmd = f"scripts/swarm/optuna_optimizer.py status --study {args.study}"
    if args.storage:
        cmd += f" --storage {args.storage}"
    _py(cmd)


def cmd_evolve_best(args):
    """Show best configs from optimization."""
    cmd = (f"scripts/swarm/optuna_optimizer.py best "
           f"--study {args.study} --top {args.top}")
    if args.storage:
        cmd += f" --storage {args.storage}"
    if args.json:
        cmd += " --json"
    _py(cmd)


def cmd_evolve_pareto(args):
    """Show Pareto front (NSGA-II multi-objective studies)."""
    cmd = f"scripts/swarm/optuna_optimizer.py pareto --study {args.study}"
    if args.storage:
        cmd += f" --storage {args.storage}"
    if args.json:
        cmd += " --json"
    _py(cmd)


def cmd_evolve_export(args):
    """Export best config as usable TOML file."""
    cmd = (f"scripts/swarm/optuna_optimizer.py export "
           f"--study {args.study} --output {args.output}")
    if args.storage:
        cmd += f" --storage {args.storage}"
    _py(cmd)


def register(sub):
    ev_p = sub.add_parser('evolve', help='Evolutionary config optimization (Optuna)')
    ev_p.set_defaults(func=lambda a: ev_p.print_help())
    evsub = ev_p.add_subparsers(dest='subcmd')
    ev_start = evsub.add_parser('start', help='Start evolutionary optimization')
    ev_start.add_argument('--study', type=str, default='nat_evolve', help='Study name (default: nat_evolve)')
    ev_start.add_argument('--sampler', choices=['cma', 'tpe', 'nsga2'], default='cma',
                          help='Sampler: cma (continuous), tpe (mixed), nsga2 (multi-objective)')
    ev_start.add_argument('--trials', '-n', type=int, default=500, help='Number of trials (default: 500)')
    ev_start.add_argument('--jobs', type=int, default=1, help='Parallel trial threads (default: 1)')
    ev_start.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    ev_start.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    ev_start.add_argument('--symbol', type=str, default='BTC', help='Symbol (default: BTC)')
    ev_start.add_argument('--hours', type=int, default=720, help='Eval window hours (default: 720 = 30d)')
    ev_start.add_argument('--train-frac', type=float, default=0.667, help='Train fraction (default: 0.667)')
    ev_start.add_argument('--storage', type=str, default='', help='Optuna storage URL (default: SQLite)')
    ev_start.add_argument('--no-guard-rails', action='store_true', help='Disable overfit detection')
    ev_start.set_defaults(func=cmd_evolve_start)
    ev_st = evsub.add_parser('status', help='Show study status')
    ev_st.add_argument('--study', type=str, default='nat_evolve', help='Study name')
    ev_st.add_argument('--storage', type=str, default='', help='Optuna storage URL')
    ev_st.set_defaults(func=cmd_evolve_status)
    ev_best = evsub.add_parser('best', help='Show best configs')
    ev_best.add_argument('--study', type=str, default='nat_evolve', help='Study name')
    ev_best.add_argument('--storage', type=str, default='', help='Optuna storage URL')
    ev_best.add_argument('--top', type=int, default=5, help='Number of configs (default: 5)')
    ev_best.add_argument('--json', action='store_true', help='JSON output')
    ev_best.set_defaults(func=cmd_evolve_best)
    ev_pareto = evsub.add_parser('pareto', help='Pareto front (NSGA-II)')
    ev_pareto.add_argument('--study', type=str, default='nat_evolve', help='Study name')
    ev_pareto.add_argument('--storage', type=str, default='', help='Optuna storage URL')
    ev_pareto.add_argument('--json', action='store_true', help='JSON output')
    ev_pareto.set_defaults(func=cmd_evolve_pareto)
    ev_export = evsub.add_parser('export', help='Export best config as TOML')
    ev_export.add_argument('--study', type=str, default='nat_evolve', help='Study name')
    ev_export.add_argument('--storage', type=str, default='', help='Optuna storage URL')
    ev_export.add_argument('--output', type=str, default='config/evolved_algorithms.toml', help='Output path')
    ev_export.set_defaults(func=cmd_evolve_export)


__all__ = ["cmd_evolve_start", "cmd_evolve_status", "cmd_evolve_best",
           "cmd_evolve_pareto", "cmd_evolve_export", "register"]
