"""`nat tournament` — continuous algorithm testing engine."""

from __future__ import annotations

import argparse
import textwrap

from cli.common import _py


def cmd_tournament(args):
    """Route tournament subcommands to the daemon script."""
    subcmd = getattr(args, 'tournament_cmd', None)
    if not subcmd:
        return _py("scripts/tournament/daemon.py --help").returncode

    cmd_parts = ["scripts/tournament/daemon.py", subcmd]

    if subcmd == "history":
        algo = getattr(args, 'algorithm', None)
        if not algo:
            print("Usage: nat tournament history <algorithm> [--days N]")
            return 1
        cmd_parts.append(algo)
        days = getattr(args, 'days', 30)
        if days != 30:
            cmd_parts.extend(["--days", str(days)])

    elif subcmd == "compare":
        a = getattr(args, 'algo_a', None)
        b = getattr(args, 'algo_b', None)
        if not a or not b:
            print("Usage: nat tournament compare <algo_a> <algo_b>")
            return 1
        cmd_parts.extend([a, b])

    return _py(" ".join(cmd_parts)).returncode


def register(sub):
    # ── tournament (continuous algorithm testing) ──
    tour_p = sub.add_parser('tournament', help='Continuous algorithm testing engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Evaluates all algorithms (hand-coded + agent-discovered) on new data,
        tracks performance over time, and manages algorithm lifecycle.

        Example:
          nat tournament run               # single evaluation cycle
          nat tournament rankings           # current leaderboard
          nat tournament history jump_detector --days 14
          nat tournament compare optimal_entry funding_reversion
        """))
    tour_p.set_defaults(func=cmd_tournament)
    tsub = tour_p.add_subparsers(dest='tournament_cmd')

    tsub.add_parser('run', help='Run a single evaluation cycle').set_defaults(func=cmd_tournament)
    tsub.add_parser('start', help='Start background daemon').set_defaults(func=cmd_tournament)
    tsub.add_parser('stop', help='Stop daemon').set_defaults(func=cmd_tournament)
    tsub.add_parser('status', help='Show daemon state and DB stats').set_defaults(func=cmd_tournament)
    tsub.add_parser('rankings', help='Current leaderboard').set_defaults(func=cmd_tournament)

    t_hist = tsub.add_parser('history', help='Per-day history for one algorithm')
    t_hist.add_argument('algorithm', help='Algorithm name')
    t_hist.add_argument('--days', type=int, default=30, help='Number of days (default: 30)')
    t_hist.set_defaults(func=cmd_tournament)

    t_cmp = tsub.add_parser('compare', help='Head-to-head comparison of two algorithms')
    t_cmp.add_argument('algo_a', help='First algorithm')
    t_cmp.add_argument('algo_b', help='Second algorithm')
    t_cmp.set_defaults(func=cmd_tournament)

    tsub.add_parser('report', help='Generate markdown report').set_defaults(func=cmd_tournament)


__all__ = ["cmd_tournament", "register"]
