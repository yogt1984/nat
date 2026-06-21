"""`nat exp` — week-long experiment runner."""

from __future__ import annotations

from cli.common import _py, _exec, _ensure_release


def cmd_exp_start(args):
    _ensure_release()
    _py("scripts/run_experiment.py start")


def cmd_exp_stop(args):
    _py("scripts/run_experiment.py stop")


def cmd_exp_status(args):
    _py("scripts/run_experiment.py status")


def cmd_exp_check(args):
    hours = getattr(args, 'hours', 24)
    _py(f"scripts/run_experiment.py check --hours {hours}")


def cmd_exp_midweek(args):
    _py("scripts/run_experiment.py midweek")


def cmd_exp_analyze(args):
    _py("scripts/run_experiment.py analyze")


def cmd_exp_dashboard(args):
    _py("scripts/run_experiment.py dashboard")


def cmd_exp_tunnel(args):
    _exec("cloudflared tunnel --url http://localhost:8050")


def register(sub):
    xp_p = sub.add_parser('exp', help='Experiment runner')
    xp_p.set_defaults(func=lambda a: xp_p.print_help())
    xsub = xp_p.add_subparsers(dest='subcmd')
    xsub.add_parser('start', help='Start ingestor in tmux').set_defaults(func=cmd_exp_start)
    xsub.add_parser('stop', help='Stop ingestor').set_defaults(func=cmd_exp_stop)
    xsub.add_parser('status', help='Health + data stats').set_defaults(func=cmd_exp_status)
    xc = xsub.add_parser('check', help='Daily validation')
    xc.add_argument('--hours', type=int, default=24, help='Hours of data to validate')
    xc.set_defaults(func=cmd_exp_check)
    xsub.add_parser('midweek', help='Full validation').set_defaults(func=cmd_exp_midweek)
    xsub.add_parser('analyze', help='End-of-experiment').set_defaults(func=cmd_exp_analyze)
    xsub.add_parser('dashboard', help='Show dashboard URL').set_defaults(func=cmd_exp_dashboard)
    xsub.add_parser('tunnel', help='Cloudflare tunnel').set_defaults(func=cmd_exp_tunnel)


__all__ = ["cmd_exp_start", "cmd_exp_stop", "cmd_exp_status", "cmd_exp_check",
           "cmd_exp_midweek", "cmd_exp_analyze", "cmd_exp_dashboard",
           "cmd_exp_tunnel", "register"]
