"""Shared constants + helpers for the `nat` CLI (extracted from the monolith).

Every `cli/<domain>.py` module imports from here. `__all__` lists the
underscore-prefixed helpers/colors explicitly so `from cli.common import *`
(used by the `nat` entry shim and domain modules) actually re-exports them.

State note: these helpers are pure/stateless — all NAT runtime state lives on
the filesystem (pidfiles, gap_state.json, TOMLs), never in module globals.
"""

from __future__ import annotations

import json as _json
import os
import subprocess
import sys
from pathlib import Path

# scripts/ on path so `import nat_paths` resolves whether we're imported by the
# `nat` shim, a domain module, or a test importing cli.common directly.
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import nat_paths  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
# Program/binaries are install-relative (__file__-anchored, like the old script);
# data/config/log/report locations come from nat_paths (env → checkout → XDG).
ROOT = Path(__file__).resolve().parents[2]      # repo / install root
RUST = ROOT / "rust"
BIN_ING = RUST / "target" / "release" / "ing"

CONFIG_DIR = nat_paths.config_dir()
DATA_DEFAULT = nat_paths.features_dir()
REPORTS_DIR = nat_paths.reports_dir()
LOG_DIR = nat_paths.log_dir()
ING_CFG = CONFIG_DIR / "ing.toml"
PIPE_CFG = CONFIG_DIR / "pipeline.toml"
DEPLOY_HOST = os.environ.get("NAT_DEPLOY_HOST", "su-35")
DEPLOY_DIR = os.environ.get("NAT_DEPLOY_DIR", "~/nat")
PY = sys.executable

G, R, Y, B, W, BOLD = "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[0m", "\033[1m"

# Time-granularity → (overview bar tf, window minutes, fine page tf). Shared by
# the viz group and the test-capture flow.
_TF_MAP = {
    '1m':  ('1min',  1,  '2s'),
    '5m':  ('5min',  5,  '10s'),
    '15m': ('15min', 15, '30s'),
}

# ── Ops/gap state paths (shared by lifecycle ↔ ops/gap/service groups) ─────────
GAP_TMUX = "nat-gap-alert"
_OPS_DIR = nat_paths.state_dir("ops")
GAP_PIDFILE = _OPS_DIR / "gap_alert.pid"
GAP_STATE_FILE = _OPS_DIR / "gap_state.json"
PAUSE_FILE = _OPS_DIR / "ingestion_paused"
GAP_WATCHDOG_MARKER = "# nat-gap-watchdog"


def _set_ingestion_paused(paused: bool):
    """Write/clear the intentional-pause marker the gap daemon reads.

    Paused = an expected `nat stop`, so no page; cleared on `nat start`."""
    from datetime import datetime, timezone
    try:
        if paused:
            PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            PAUSE_FILE.write_text(datetime.now(timezone.utc).isoformat())
        else:
            PAUSE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def ensure_scripts_path():
    """Idempotently put scripts/ on sys.path (replaces the ~30 scattered inserts
    in handler bodies)."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sh(cmd, **kw):
    """Run shell command, capture output."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def _exec(cmd, cwd=None, env=None):
    """Run shell command in foreground, inheriting stdio."""
    # Flush our buffered output first so banners/prints appear *before* the
    # child's output. Without this, Python fully buffers stdout when it is a
    # pipe (e.g. `nat build | tail`), so the child writes to the terminal first
    # and our banner lands last — making a successful command look broken.
    sys.stdout.flush()
    sys.stderr.flush()
    # Propagate the resolved NAT_* locations so child scripts / the Rust ingestor
    # resolve the same data/config dirs regardless of their CWD.
    e = {**os.environ, **nat_paths.as_env(), **(env or {})}
    return subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, env=e)


def _cargo(args_str):
    return _exec(f"cargo {args_str}", cwd=RUST)


def _py(script_args, cwd=None):
    return _exec(f"{PY} {script_args}", cwd=cwd or ROOT)


def _pid():
    r = _sh("pgrep -x ing")
    return int(r.stdout.strip().split("\n")[0]) if r.returncode == 0 and r.stdout.strip() else None


def _p(icon, color, msg):
    print(f"  {color}{icon}{W}  {msg}")


def _banner(title):
    w = max(len(title) + 4, 50)
    print(f"  {'=' * w}")
    print(f"  {BOLD}{title}{W}")
    print(f"  {'=' * w}")
    print()


def _data(args):
    return getattr(args, 'data', None) or str(DATA_DEFAULT)


def _sym(args):
    return getattr(args, 'symbol', None) or 'BTC'


def _json_mode(args):
    return getattr(args, 'json', False)


def _output(data: dict, args, human_fn=None):
    """Output data as JSON or human-readable. If human_fn provided, call it for human output."""
    if _json_mode(args):
        print(_json.dumps(data, indent=2, default=str))
    elif human_fn:
        human_fn(data)


def _ensure_release():
    """Build release binaries if not present."""
    if not BIN_ING.exists():
        _p("...", Y, "Release binaries not found, building...")
        r = _cargo(
            "build --locked --release --bin ing --bin validate_api --bin validate_positions "
            "--bin validate_whales --bin validate_entropy --bin show_features --bin test_hypotheses"
        )
        if r.returncode != 0:
            _p("x", R, "Build failed")
            sys.exit(1)


def _dir_owner(path):
    """Best-effort username owning `path`."""
    try:
        import pwd
        return pwd.getpwuid(os.stat(path).st_uid).pw_name
    except Exception:
        return "?"


def _unwritable_data_dirs(bases=None):
    """Data dirs the ingestor must write but the current user CANNOT — the
    root-owned-dir silent-stall cause (Docker-as-root vs native-as-user). Returns
    [(path, owner)]. Pass `bases` to test against arbitrary dirs."""
    if bases is None:
        bases = [ROOT / "data" / "features", ROOT / "data" / "trades"]
    issues = []
    for base in bases:
        base = Path(base)
        if not base.exists():
            continue
        if not os.access(base, os.W_OK):
            issues.append((str(base), _dir_owner(base)))
        for d in sorted(base.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]")):
            if d.is_dir() and not os.access(d, os.W_OK):
                issues.append((str(d), _dir_owner(d)))
    return issues


__all__ = [
    # constants
    "ROOT", "RUST", "BIN_ING", "CONFIG_DIR", "DATA_DEFAULT", "REPORTS_DIR",
    "LOG_DIR", "ING_CFG", "PIPE_CFG", "DEPLOY_HOST", "DEPLOY_DIR", "PY",
    "G", "R", "Y", "B", "W", "BOLD", "nat_paths",
    # shared ops/viz state
    "_TF_MAP", "GAP_TMUX", "_OPS_DIR", "GAP_PIDFILE", "GAP_STATE_FILE",
    "PAUSE_FILE", "GAP_WATCHDOG_MARKER", "_set_ingestion_paused",
    # helpers
    "ensure_scripts_path", "_sh", "_exec", "_cargo", "_py", "_pid", "_p",
    "_banner", "_data", "_sym", "_json_mode", "_output", "_ensure_release",
    "_dir_owner", "_unwritable_data_dirs",
]
