"""Integration test for the `nat lifecycle` CLI (plan T4).

Drives the real `nat` entrypoint via subprocess against a temp db, covering the
plan's acceptance criteria: seed -> 4 VALIDATED / 2 DISCOVERED, history carries
git_sha, --state filter, and the human-gate refuses an illegal promotion.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

NAT = Path(__file__).resolve().parents[2] / "nat"


def _run(db: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(NAT), "lifecycle", "--db", str(db), *args],
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def db(tmp_path):
    d = tmp_path / "nat.db"
    assert _run(d, "seed").returncode == 0
    return d


def test_seed_status_counts(db):
    r = _run(db, "status", "--json")
    assert r.returncode == 0, r.stderr
    by_state = json.loads(r.stdout)["by_state"]
    assert by_state == {"VALIDATED": 4, "DISCOVERED": 2}, by_state


def test_seed_is_idempotent(db):
    r = _run(db, "seed")
    assert "Seeded 0" in r.stdout, r.stdout


def test_history_carries_git_sha(db):
    r = _run(db, "history", "jump_detector", "--json")
    assert r.returncode == 0, r.stderr
    hist = json.loads(r.stdout)["history"]
    assert [h["to_state"] for h in hist] == ["DISCOVERED", "VALIDATED"]
    assert all(h["git_sha"] for h in hist), "a history row is missing git_sha"


def test_state_filter(db):
    r = _run(db, "list", "--state", "DISCOVERED", "--json")
    ids = {s["signal_id"] for s in json.loads(r.stdout)["signals"]}
    assert ids == {"hierarchical_combiner", "mean_reversion_detector"}, ids


def test_human_gate_refuses_illegal_promotion(db):
    # jump_detector is VALIDATED; VALIDATED -> LIVE is not a legal transition.
    r = _run(db, "approve", "jump_detector", "--confirm")
    assert r.returncode == 1
    assert "not allowed" in r.stdout.lower() or "cannot approve" in r.stdout.lower()


def test_approve_dry_run_does_not_mutate(db):
    _run(db, "approve", "jump_detector")          # no --confirm
    r = _run(db, "history", "jump_detector", "--json")
    hist = json.loads(r.stdout)["history"]
    assert [h["to_state"] for h in hist] == ["DISCOVERED", "VALIDATED"]  # unchanged
