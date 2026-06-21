"""`nat swarm` — parameter sweep optimization (Tier 2 config swarm)."""

from __future__ import annotations

from cli.common import _py


def cmd_swarm_run(args):
    """Generate configs and evaluate in parallel."""
    _py(f"scripts/swarm/orchestrator.py run "
        f"--instances {args.instances} --hours {args.hours} "
        f"--symbol {args.symbol}"
        + (f" --seed {args.seed}" if args.seed is not None else "")
        + (f" --workers {args.workers}" if args.workers is not None else "")
        + (" --json" if args.json else ""))


def cmd_swarm_status(args):
    """Show swarm run status."""
    _py("scripts/swarm/orchestrator.py status")


def cmd_swarm_results(args):
    """Show top configs ranked by Sharpe."""
    _py(f"scripts/swarm/orchestrator.py results --top {args.top}"
        + (" --json" if args.json else ""))


def cmd_swarm_best(args):
    """Export best config as TOML."""
    _py(f"scripts/swarm/orchestrator.py best --export {args.export}")


def cmd_swarm_generate(args):
    """Generate configs only (no evaluation)."""
    _py(f"scripts/swarm/config_generator.py "
        f"--count {args.count} --output {args.output}"
        + (f" --seed {args.seed}" if args.seed is not None else ""))


def register(sub):
    sw_p = sub.add_parser('swarm', help='Parameter sweep optimization')
    sw_p.set_defaults(func=lambda a: sw_p.print_help())
    swsub = sw_p.add_subparsers(dest='subcmd')
    sw_run = swsub.add_parser('run', help='Generate configs and evaluate in parallel')
    sw_run.add_argument('--instances', '-n', type=int, default=16, help='Number of configs (default: 16)')
    sw_run.add_argument('--hours', type=int, default=24, help='Hours of data per eval (default: 24)')
    sw_run.add_argument('--symbol', type=str, default='BTC', help='Symbol (default: BTC)')
    sw_run.add_argument('--seed', type=int, default=None, help='Random seed')
    sw_run.add_argument('--workers', type=int, default=None, help='Max parallel workers')
    sw_run.add_argument('--json', action='store_true', help='JSON output')
    sw_run.set_defaults(func=cmd_swarm_run)
    sw_st = swsub.add_parser('status', help='Show swarm run status')
    sw_st.set_defaults(func=cmd_swarm_status)
    sw_res = swsub.add_parser('results', help='Show top configs ranked by Sharpe')
    sw_res.add_argument('--top', type=int, default=10, help='Number of results (default: 10)')
    sw_res.add_argument('--json', action='store_true', help='JSON output')
    sw_res.set_defaults(func=cmd_swarm_results)
    sw_best = swsub.add_parser('best', help='Export best config as TOML')
    sw_best.add_argument('--export', type=str, default='config/best_algorithms.toml', help='Output path')
    sw_best.set_defaults(func=cmd_swarm_best)
    sw_gen = swsub.add_parser('generate', help='Generate configs only (no evaluation)')
    sw_gen.add_argument('--count', '-n', type=int, default=16, help='Number of configs')
    sw_gen.add_argument('--output', type=str, default='data/swarm/configs', help='Output directory')
    sw_gen.add_argument('--seed', type=int, default=None, help='Random seed')
    sw_gen.set_defaults(func=cmd_swarm_generate)


__all__ = ["cmd_swarm_run", "cmd_swarm_status", "cmd_swarm_results",
           "cmd_swarm_best", "cmd_swarm_generate", "register"]
