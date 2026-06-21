"""`nat screen` — alpha screening on collected data."""

from __future__ import annotations

import argparse
import subprocess

from cli.common import ROOT, PY, B, _p


def cmd_screen(args):
    """Run alpha screening on collected data."""
    _p("...", B, "Running alpha screener...")
    print()
    extra = getattr(args, 'screen_args', None) or []
    subprocess.run([PY, "-m", "scripts.alpha.screener"] + extra, cwd=str(ROOT))


def register(sub):
    screen_p = sub.add_parser('screen', help='Alpha screening')
    screen_p.add_argument('screen_args', nargs=argparse.REMAINDER)
    screen_p.set_defaults(func=cmd_screen)


__all__ = ["cmd_screen", "register"]
