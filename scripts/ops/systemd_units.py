"""Render systemd `--user` unit files for the NAT ingestor + gap-alert daemon.

Pure (no host writes) so it is unit-testable. `nat service install` writes the
returned texts to ``~/.config/systemd/user/`` and enables them. Paths and env
come from ``nat_paths`` so the units are correct whether run from a dev checkout
or an installed prefix; the Rust ``ing`` reads ``NAT_DATA_DIR``/``NAT_TRADE_DIR``
from this env, so systemd-managed ingestion writes to the resolved data dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import nat_paths
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nat_paths

INGESTOR_UNIT = "nat-ingestor.service"
GAP_UNIT = "nat-gap-alert.service"


def _env_lines(extra: dict[str, str] | None = None) -> str:
    env = dict(nat_paths.as_env())
    if extra:
        env.update(extra)
    return "\n".join(f'Environment="{k}={v}"' for k, v in env.items())


def render_units(python: str | None = None) -> dict[str, str]:
    """Return {unit_filename: file_text} for the ingestor and gap-alert daemon."""
    py = python or sys.executable
    root = nat_paths.install_root()
    rust = root / "rust"
    bin_ing = rust / "target" / "release" / "ing"
    ing_cfg = nat_paths.config_dir() / "ing.toml"
    gap_script = root / "scripts" / "ops" / "gap_alert.py"

    ingestor = f"""\
[Unit]
Description=NAT Hyperliquid ingestor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={rust}
ExecStart={bin_ing} {ing_cfg}
{_env_lines({"RUST_LOG": "info"})}
Restart=always
RestartSec=5
# Never give up restarting (a crash-looping ingestor is still better than a dead one).
StartLimitIntervalSec=0

[Install]
WantedBy=default.target
"""

    gap = f"""\
[Unit]
Description=NAT data-gap alert daemon
After=network-online.target

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={py} {gap_script} start
{_env_lines()}
Restart=always
RestartSec=10
StartLimitIntervalSec=0

[Install]
WantedBy=default.target
"""

    return {INGESTOR_UNIT: ingestor, GAP_UNIT: gap}


def unit_dir() -> Path:
    """Where `nat service install` writes the units (~/.config/systemd/user)."""
    xdg = nat_paths._xdg("XDG_CONFIG_HOME", Path.home() / ".config")
    return xdg / "systemd" / "user"
