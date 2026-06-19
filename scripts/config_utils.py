"""Centralized configuration helpers.

Provides load_symbols() — the single-source-of-truth reader for the
canonical symbol list in config/symbols.toml.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

log = logging.getLogger(__name__)

try:
    import nat_paths
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nat_paths

_SYMBOLS_PATH = nat_paths.config_dir() / "symbols.toml"
_AGENT_TOML_PATH = nat_paths.config_dir() / "agent.toml"

# Cache after first load (immutable at runtime)
_symbols_cache: list[str] | None = None


def load_dotenv(path: str | Path | None = None) -> bool:
    """Load KEY=VALUE pairs from a .env into os.environ (without overriding real env).

    Looks (in order) at the given path, then ``<install_root>/.env`` and
    ``<config_dir>/.env`` so secrets like TELEGRAM_* reach daemons launched by
    cron/tmux that don't inherit an interactive shell's environment. Returns True
    if a file was found. Best-effort and silent on malformed lines.
    """
    candidates = [Path(path)] if path else [
        nat_paths.install_root() / ".env",
        nat_paths.config_dir() / ".env",
    ]
    for c in candidates:
        if not c or not c.exists():
            continue
        try:
            for line in c.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    os.environ.setdefault(k, v)   # real env wins
            return True
        except OSError:
            return False
    return False


def load_symbols(path: str | Path | None = None) -> list[str]:
    """Load the canonical symbol list from config/symbols.toml.

    Args:
        path: Override path (for testing). Defaults to config/symbols.toml.

    Returns:
        List of symbol strings, e.g. ["BTC", "ETH", "SOL"].

    Raises:
        FileNotFoundError: if symbols.toml is missing.
        KeyError: if 'symbols' key is absent.
    """
    global _symbols_cache

    resolved = Path(path) if path else _SYMBOLS_PATH

    # Use cache only for the default path
    if path is None and _symbols_cache is not None:
        return list(_symbols_cache)

    with open(resolved, "rb") as f:
        data = tomllib.load(f)

    symbols = data["symbols"]
    if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
        raise ValueError(f"symbols.toml: 'symbols' must be a list of strings, got {type(symbols)}")

    if path is None:
        _symbols_cache = list(symbols)

    log.debug("Loaded %d symbols from %s", len(symbols), resolved)
    return list(symbols)


def load_cost_config(path: str | Path | None = None) -> dict[str, float]:
    """Load cost model defaults from config/agent.toml [defaults.costs].

    Returns:
        Dict with 'fee_bps' and 'slippage_bps' keys.
    """
    resolved = Path(path) if path else _AGENT_TOML_PATH

    with open(resolved, "rb") as f:
        data = tomllib.load(f)

    costs = data.get("defaults", {}).get("costs", {})
    return {
        "fee_bps": float(costs.get("fee_bps", 3.5)),
        "slippage_bps": float(costs.get("slippage_bps", 2.0)),
    }
