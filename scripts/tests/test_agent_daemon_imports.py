"""BUG-2 regression: agent daemon CLIs must not crash with
`ModuleNotFoundError: No module named 'logging_config'`.

Root cause: the editable install (`nat-research`) exposes *packages* (`agent`, `cli`, …)
via `packages.find`, but NOT the loose top-level module `scripts/logging_config.py`. So a
daemon launched the way `nat agent status` launches it —
`python scripts/agent/<x>.py status`, with `sys.path[0] = scripts/agent/` and no
`scripts/` on the path — could import `agent.base` but not `logging_config` (imported at
runtime inside each daemon's `cli_main`/`main`). Fixed by a `sys.path` bootstrap that puts
`scripts/` on the path in every daemon entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DAEMONS = ["daemon", "mf_daemon", "macro_daemon", "cascade_daemon", "meta_daemon"]


def _run_status(daemon: str) -> subprocess.CompletedProcess:
    script = ROOT / "scripts" / "agent" / f"{daemon}.py"
    # PYTHONPATH="" replicates the bare subprocess: scripts/ is NOT injected via the
    # environment, so the fix must come from the daemon's own sys.path bootstrap.
    env = {**os.environ, "PYTHONPATH": ""}
    return subprocess.run(
        [sys.executable, str(script), "status"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, env=env,
    )


@pytest.mark.parametrize("daemon", DAEMONS)
def test_daemon_status_has_no_logging_config_import_error(daemon):
    script = ROOT / "scripts" / "agent" / f"{daemon}.py"
    if not script.exists():
        pytest.skip(f"{script} not present")
    out = _run_status(daemon)
    combined = out.stdout + out.stderr
    assert "No module named 'logging_config'" not in combined, (
        f"{daemon}.py still cannot import logging_config:\n{combined[-1200:]}")
