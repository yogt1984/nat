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

# Project root: two levels up from scripts/config_utils.py
_ROOT = Path(__file__).resolve().parent.parent

_SYMBOLS_PATH = _ROOT / "config" / "symbols.toml"
_AGENT_TOML_PATH = _ROOT / "config" / "agent.toml"

# Cache after first load (immutable at runtime)
_symbols_cache: list[str] | None = None


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
