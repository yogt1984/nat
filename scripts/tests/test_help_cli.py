"""Integration test for `nat help --grep` (NAT1) and group-level help (NAT2), plan T6.

Drives the real `nat` entrypoint via subprocess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

NAT = Path(__file__).resolve().parents[2] / "nat"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(NAT), *args],
                          capture_output=True, text=True, timeout=60)


# --- NAT1: nat help --grep ---

def test_grep_matches_name_and_help():
    r = _run("help", "--grep", "lifecycle")
    assert r.returncode == 0, r.stderr
    out = r.stdout.lower()
    assert "matching 'lifecycle'" in out and "nat lifecycle" in out


def test_grep_is_case_insensitive():
    assert "kalman" in _run("help", "--grep", "KALMAN").stdout.lower()


def test_grep_no_match_message():
    assert "no commands matching" in _run("help", "--grep", "zzqqxx").stdout.lower()


def test_help_grep_flag_documented():
    assert "--grep" in _run("help", "-h").stdout


# --- NAT2: group-level help ---

def test_group_help_is_scoped_not_argparse_usage():
    r = _run("alpha")
    assert r.returncode == 0, r.stderr
    assert "nat alpha" in r.stdout
    assert "nat alpha combine" in r.stdout          # subcommands listed
    assert "usage: nat alpha" not in r.stdout       # not the argparse dump


def test_group_help_fixes_bare_daemon_group():
    r = _run("agent")                               # bare `nat agent` used to crash
    assert r.returncode == 0
    assert "Traceback" not in (r.stdout + r.stderr)
    assert "research agent" in r.stdout.lower()


def test_subcommand_help_unchanged():
    assert "usage: nat alpha combine" in _run("alpha", "combine", "-h").stdout


def test_full_help_still_works():
    r = _run("help")
    assert r.returncode == 0 and "unified NAT research terminal" in r.stdout
