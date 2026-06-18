"""Guard `nat help` against drifting from the live command tree.

The top-level `nat help` body is hand-curated (grouped sections, narrative lines,
per-command descriptions richer than the parser's ``help=`` strings, math context),
so it deliberately is NOT auto-generated — doing so would lose that value. Instead
this test enforces the one invariant that actually matters: every top-level command
group must be *mentioned* in `nat help` output as ``nat <group>``. That turns silent
drift (a new group shipped but never documented) into a loud, CI-visible failure.

When this fails: add a curated entry for the new group to ``cmd_help()`` in ``nat``.
Only add to ``EXEMPT`` if the group is genuinely meta (``help``) or greenfield/stub.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAT = ROOT / "nat"

# Groups intentionally without a curated help entry:
#   help  — the help command documenting itself is pointless
#   mesh  — empty stub (no description, no subcommands)
#   viz3d — interactive 3D feature surface, greenfield (see docs/STATE)
EXEMPT = {"help", "mesh", "viz3d"}


def _load_build_parser():
    """Exec the `nat` script as a module and return its build_parser()."""
    ns = {"__name__": "natmod", "__file__": str(NAT)}
    exec(compile(NAT.read_text(), str(NAT), "exec"), ns)
    return ns["build_parser"]


def _top_level_groups(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices.keys())
    return []


def test_every_command_group_documented_in_help():
    parser = _load_build_parser()()
    groups = _top_level_groups(parser)
    assert groups, "no top-level command groups found — parser changed shape?"

    help_text = subprocess.run(
        [sys.executable, str(NAT), "help"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout

    undocumented = [
        g for g in groups
        if g not in EXEMPT and f"nat {g}" not in help_text
    ]
    assert not undocumented, (
        "`nat help` is missing curated entries for: "
        f"{sorted(undocumented)}.\n"
        "Add a `nat <group> ...` line to cmd_help() in the `nat` script, "
        "or add the group to EXEMPT if it is meta/greenfield."
    )
