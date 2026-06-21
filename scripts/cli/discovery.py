"""`nat discovery` — alpha discovery orchestrator."""

from __future__ import annotations

from cli.common import _py


def cmd_discovery_start(args):
    _py("scripts/discovery_orchestrator.py --config config/discovery.toml start")


def cmd_discovery_once(args):
    _py("scripts/discovery_orchestrator.py --config config/discovery.toml once")


def cmd_discovery_status(args):
    _py("scripts/discovery_orchestrator.py --config config/discovery.toml status")


def cmd_discovery_stop(args):
    _py("scripts/discovery_orchestrator.py --config config/discovery.toml stop")


def register(sub):
    disc_p = sub.add_parser('discovery', help='Alpha discovery orchestrator')
    disc_p.set_defaults(func=lambda a: disc_p.print_help())
    dcsub = disc_p.add_subparsers(dest='subcmd')
    dcsub.add_parser('start', help='Launch discovery orchestrator').set_defaults(func=cmd_discovery_start)
    dcsub.add_parser('once', help='Run single sweep').set_defaults(func=cmd_discovery_once)
    dcsub.add_parser('status', help='Current state').set_defaults(func=cmd_discovery_status)
    dcsub.add_parser('stop', help='Stop orchestrator').set_defaults(func=cmd_discovery_stop)


__all__ = ["cmd_discovery_start", "cmd_discovery_once", "cmd_discovery_status",
           "cmd_discovery_stop", "register"]
