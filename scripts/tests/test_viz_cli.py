"""Integration test for `nat viz features` (NAT4) + `nat viz algorithm` (NAT5), plan T7.

Subprocess-driven; tolerant of thin/no real data (asserts JSON shape, not values).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

NAT = Path(__file__).resolve().parents[2] / "nat"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(NAT), *args],
                          capture_output=True, text=True, timeout=120)


def test_viz_features_json_shape():
    r = _run("viz", "features", "--symbol", "BTC", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["symbol"] == "BTC" and isinstance(d["features"], list)


def test_viz_algorithm_json_shape():
    r = _run("viz", "algorithm", "jump_detector", "--symbol", "BTC", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)              # clean JSON: model-load noise is muted
    assert d["algorithm"] == "jump_detector" and isinstance(d["outputs"], list)


def test_viz_algorithm_unknown_returns_error():
    r = _run("viz", "algorithm", "definitely_not_an_algo", "--symbol", "BTC")
    assert r.returncode == 1
    assert "unknown algorithm" in r.stdout.lower()


def test_viz_group_help_lists_subcommands():
    r = _run("viz")
    assert "features" in r.stdout and "algorithm" in r.stdout
    assert "usage: nat viz" not in r.stdout
