"""Planted test for the systemd unit-file renderer (ops.systemd_units).

Pins the unit contents so `nat service install` writes correct, relocatable
ExecStart/WorkingDirectory/Environment derived from nat_paths.
"""

from __future__ import annotations

import pytest

systemd_units = pytest.importorskip("ops.systemd_units")

_ENV_KEYS = ["NAT_HOME", "NAT_DATA", "NAT_CONFIG", "NAT_REPORTS", "NAT_LOG",
             "NAT_INSTALL_ROOT", "XDG_DATA_HOME", "XDG_CONFIG_HOME"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_ingestor_unit(monkeypatch, tmp_path):
    install = tmp_path / "usr_lib_nat"
    home = tmp_path / "nathome"
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(install))
    monkeypatch.setenv("NAT_HOME", str(home))

    units = systemd_units.render_units(python="/usr/bin/python3")
    ing = units["nat-ingestor.service"]

    assert f"WorkingDirectory={install / 'rust'}" in ing
    assert f"ExecStart={install / 'rust' / 'target' / 'release' / 'ing'} {home / 'config' / 'ing.toml'}" in ing
    assert f'Environment="NAT_DATA_DIR={home / "data" / "features"}"' in ing
    assert f'Environment="NAT_TRADE_DIR={home / "data" / "trades"}"' in ing
    assert 'Environment="RUST_LOG=info"' in ing
    assert "Restart=always" in ing
    assert "WantedBy=default.target" in ing


def test_gap_unit(monkeypatch, tmp_path):
    install = tmp_path / "usr_lib_nat"
    home = tmp_path / "nathome"
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(install))
    monkeypatch.setenv("NAT_HOME", str(home))

    units = systemd_units.render_units(python="/usr/bin/python3")
    gap = units["nat-gap-alert.service"]

    assert f"WorkingDirectory={install}" in gap
    assert f"ExecStart=/usr/bin/python3 {install / 'scripts' / 'ops' / 'gap_alert.py'} start" in gap
    assert f'Environment="NAT_CONFIG={home / "config"}"' in gap
    assert "Restart=always" in gap
    assert "WantedBy=default.target" in gap


def test_unit_dir_under_xdg_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert systemd_units.unit_dir() == tmp_path / "cfg" / "systemd" / "user"


def test_start_limit_interval_is_in_the_unit_section():
    """systemd parses StartLimitIntervalSec ONLY in [Unit]; in [Service] it is ignored
    with a warning, so `Restart=always` silently keeps the default rate limiter and a
    crash-looping daemon gives up after 5 tries in 10s. Observed in the journal when the
    XS-8 sampler units were installed on 2026-08-08."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ops.systemd_units import render_units

    for name, text in render_units().items():
        if "StartLimitIntervalSec" not in text:
            continue
        head = text.split("[Service]", 1)[0]
        assert "StartLimitIntervalSec" in head, (
            f"{name}: StartLimitIntervalSec sits after [Service] — systemd ignores it there")
