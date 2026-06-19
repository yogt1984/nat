"""Planted contract test for the relocatable path resolver (scripts/nat_paths.py).

Written test-first (red) per the repo guardrail. Encodes the resolution precedence:
  explicit NAT_* env  →  source-checkout fallback (repo layout)  →  installed XDG.

The resolver must read os.environ at *call time* (so these monkeypatches take effect)
and anchor the install root via NAT_INSTALL_ROOT (a test/packaging hook) when set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

nat_paths = pytest.importorskip("nat_paths")

_ENV_KEYS = [
    "NAT_HOME", "NAT_DATA", "NAT_CONFIG", "NAT_REPORTS", "NAT_LOG",
    "NAT_INSTALL_ROOT", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known-empty NAT_*/XDG_* environment."""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


# ── source-checkout fallback (the dev default — must equal today's repo layout) ──

def test_source_checkout_uses_repo_layout(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "rust").mkdir(parents=True)
    (root / "rust" / "Cargo.toml").write_text("[workspace]\n")   # checkout marker
    (root / "scripts").mkdir()
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(root))

    assert nat_paths.home() == root
    assert nat_paths.features_dir() == root / "data" / "features"
    assert nat_paths.trades_dir() == root / "data" / "trades"
    assert nat_paths.config_dir() == root / "config"
    assert nat_paths.reports_dir() == root / "reports"
    assert nat_paths.log_dir() == root / "logs"
    assert nat_paths.db_path() == root / "data" / "nat.db"


def test_dot_git_is_also_a_checkout_marker(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(root))
    assert nat_paths.home() == root
    assert nat_paths.config_dir() == root / "config"


# ── NAT_HOME master override (applies to every location, including config) ──

def test_nat_home_is_master(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)          # even in a checkout, env wins
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(root))
    home = tmp_path / "nathome"
    monkeypatch.setenv("NAT_HOME", str(home))

    assert nat_paths.home() == home
    assert nat_paths.features_dir() == home / "data" / "features"
    assert nat_paths.config_dir() == home / "config"
    assert nat_paths.reports_dir() == home / "reports"
    assert nat_paths.log_dir() == home / "logs"


# ── granular overrides win over NAT_HOME ──

def test_granular_overrides_beat_nat_home(monkeypatch, tmp_path):
    monkeypatch.setenv("NAT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NAT_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("NAT_CONFIG", str(tmp_path / "c"))
    monkeypatch.setenv("NAT_REPORTS", str(tmp_path / "r"))

    assert nat_paths.features_dir() == tmp_path / "d" / "features"
    assert nat_paths.config_dir() == tmp_path / "c"
    assert nat_paths.reports_dir() == tmp_path / "r"
    # NAT_LOG not set → falls back under NAT_HOME
    assert nat_paths.log_dir() == tmp_path / "home" / "logs"


# ── installed mode (no checkout markers) → XDG / system ──

def test_installed_mode_uses_xdg(monkeypatch, tmp_path):
    root = tmp_path / "usr_lib_nat"               # no rust/ or .git → "installed"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(root))
    xdg_data = tmp_path / "xdg_data"
    xdg_cfg = tmp_path / "xdg_config"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_cfg))
    # No system /etc/nat in this test.
    monkeypatch.setattr(nat_paths, "_SYSTEM_CONFIG", tmp_path / "no_etc_nat", raising=False)

    assert nat_paths.home() == xdg_data / "nat"
    assert nat_paths.features_dir() == xdg_data / "nat" / "data" / "features"
    assert nat_paths.config_dir() == xdg_cfg / "nat"


def test_installed_mode_prefers_system_etc_nat_when_present(monkeypatch, tmp_path):
    root = tmp_path / "usr_lib_nat"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(root))
    etc_nat = tmp_path / "etc_nat"
    etc_nat.mkdir()
    monkeypatch.setattr(nat_paths, "_SYSTEM_CONFIG", etc_nat, raising=False)
    assert nat_paths.config_dir() == etc_nat


# ── as_env() exports a NAT_* dict consumable by child processes / Rust ing ──

def test_as_env_exports_absolute_paths(monkeypatch, tmp_path):
    home = tmp_path / "nathome"
    monkeypatch.setenv("NAT_HOME", str(home))
    env = nat_paths.as_env()

    assert env["NAT_HOME"] == str(home)
    assert env["NAT_DATA_DIR"] == str(home / "data" / "features")   # Rust ing reads this
    assert env["NAT_TRADE_DIR"] == str(home / "data" / "trades")
    assert env["NAT_CONFIG"] == str(home / "config")
    for v in env.values():
        assert Path(v).is_absolute()
