"""Regression oracle for the `nat` command tree.

`nat --json commands` reflects the live argparse tree (registration order, help,
args). This test pins it against a committed baseline so the CLI modularization
(splitting the monolith into scripts/cli/*) can be proven behavior-preserving:
the command tree must stay byte-identical at every step.

If you intentionally add/rename a command, regenerate the baseline:
    ./nat --json commands | python3 -c \
      "import sys,json;print(json.dumps(json.load(sys.stdin),indent=2,sort_keys=True))" \
      > scripts/tests/data/commands_snapshot.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_NAT = _ROOT / "nat"
_BASELINE = Path(__file__).resolve().parent / "data" / "commands_snapshot.json"


def _live_tree() -> dict:
    out = subprocess.run(
        [str(_NAT), "--json", "commands"],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, f"`nat --json commands` failed: {out.stderr[-500:]}"
    return json.loads(out.stdout)


def test_command_tree_matches_baseline():
    baseline = json.loads(_BASELINE.read_text())
    live = _live_tree()

    # Count first — fastest signal of an added/dropped command.
    assert live["count"] == baseline["count"], (
        f"command count changed: {baseline['count']} -> {live['count']}"
    )

    # Registration order + per-command help/args must be identical. Compare the
    # normalized forms; on mismatch, surface the first differing command.
    live_norm = json.loads(json.dumps(live, sort_keys=True))
    base_cmds = {c["name"]: c for c in baseline["commands"]}
    live_cmds = {c["name"]: c for c in live_norm["commands"]}

    assert set(live_cmds) == set(base_cmds), (
        "command names changed: "
        f"added={sorted(set(live_cmds) - set(base_cmds))[:10]} "
        f"removed={sorted(set(base_cmds) - set(live_cmds))[:10]}"
    )
    for name in base_cmds:
        assert live_cmds[name] == base_cmds[name], f"command '{name}' definition changed"

    # Registration order (the list order) — the reflection depends on it.
    assert [c["name"] for c in live_norm["commands"]] == \
           [c["name"] for c in baseline["commands"]], "command registration order changed"
