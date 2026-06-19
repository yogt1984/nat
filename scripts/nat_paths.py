"""nat_paths — single source of truth for where nat reads/writes on disk.

Makes the tool relocatable: the program (this file, the rest of ``scripts/``, and
the Rust binaries under ``rust/``) can live anywhere, while data / config / logs /
reports resolve from the environment with sensible fallbacks.

Resolution precedence per location:
  1. explicit env — ``NAT_HOME`` (master) and granular ``NAT_DATA`` / ``NAT_CONFIG``
     / ``NAT_REPORTS`` / ``NAT_LOG`` (which win over ``NAT_HOME``);
  2. source-checkout fallback — if the install root looks like a dev checkout
     (``rust/`` or ``.git`` present) use the repo layout (``<repo>/data`` …),
     so a development tree behaves exactly as it always has;
  3. installed mode — XDG: data under ``$XDG_DATA_HOME/nat`` (``~/.local/share/nat``),
     config under ``/etc/nat`` if present else ``$XDG_CONFIG_HOME/nat``.

Everything is read at call time so child processes / tests can override via env.
``install_root()`` (and the bundled binaries) stay anchored to the install
location — never XDG — because they ship with the program, not the user's data.
"""

from __future__ import annotations

import os
from pathlib import Path

# System-wide config dir a .deb installs to. Module-level so tests can patch it.
_SYSTEM_CONFIG = Path("/etc/nat")


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def install_root() -> Path:
    """Directory that holds ``scripts/`` and (in a checkout) ``rust/``.

    Honors ``NAT_INSTALL_ROOT`` (packaging/test hook); otherwise anchored to this
    file's location (``…/scripts/nat_paths.py`` → parent of ``scripts``)."""
    env = _env_path("NAT_INSTALL_ROOT")
    if env:
        return env
    return Path(__file__).resolve().parent.parent


def is_source_checkout() -> bool:
    # Markers present in a dev tree but NOT in an installed package: the .deb
    # ships rust/target/release/* (so bare `rust/` is a false positive) but never
    # `.git` or `rust/Cargo.toml`.
    root = install_root()
    return (root / ".git").exists() or (root / "rust" / "Cargo.toml").exists()


def _xdg(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def home() -> Path:
    """Writable base for data/reports/logs."""
    env = _env_path("NAT_HOME")
    if env:
        return env
    if is_source_checkout():
        return install_root()
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / "nat"


def data_root() -> Path:
    return _env_path("NAT_DATA") or (home() / "data")


def features_dir() -> Path:
    return data_root() / "features"


def trades_dir() -> Path:
    return data_root() / "trades"


def db_path() -> Path:
    return data_root() / "nat.db"


def state_dir(name: str) -> Path:
    return data_root() / name


def reports_dir() -> Path:
    return _env_path("NAT_REPORTS") or (home() / "reports")


def log_dir() -> Path:
    return _env_path("NAT_LOG") or (home() / "logs")


def config_dir() -> Path:
    """Where the ``*.toml`` configs live."""
    env = _env_path("NAT_CONFIG")
    if env:
        return env
    nat_home = _env_path("NAT_HOME")
    if nat_home:                       # NAT_HOME is master → config under it too
        return nat_home / "config"
    if is_source_checkout():
        return install_root() / "config"
    if _SYSTEM_CONFIG.exists():        # system package install
        return _SYSTEM_CONFIG
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / "nat"


def as_env() -> dict[str, str]:
    """NAT_* environment to hand to child processes (Python scripts and Rust `ing`).

    ``NAT_DATA_DIR`` / ``NAT_TRADE_DIR`` are the names the Rust ingestor reads."""
    return {
        "NAT_HOME": str(home()),
        "NAT_DATA": str(data_root()),
        "NAT_DATA_DIR": str(features_dir()),
        "NAT_TRADE_DIR": str(trades_dir()),
        "NAT_CONFIG": str(config_dir()),
        "NAT_REPORTS": str(reports_dir()),
        "NAT_LOG": str(log_dir()),
    }


def describe() -> dict[str, object]:
    """Resolved locations + how each base was chosen (for `nat config paths`)."""
    if _env_path("NAT_HOME"):
        src = "env:NAT_HOME"
    elif is_source_checkout():
        src = "source-checkout"
    else:
        src = "installed:xdg"
    return {
        "install_root": str(install_root()),
        "source": src,
        "home": str(home()),
        "data_root": str(data_root()),
        "features_dir": str(features_dir()),
        "trades_dir": str(trades_dir()),
        "config_dir": str(config_dir()),
        "reports_dir": str(reports_dir()),
        "log_dir": str(log_dir()),
        "db_path": str(db_path()),
    }
