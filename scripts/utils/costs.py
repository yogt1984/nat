"""Load trading costs from config/costs.toml (single source of truth)."""

import sys
from pathlib import Path

try:
    import nat_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nat_paths

_COSTS_PATH = nat_paths.config_dir() / "costs.toml"


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_costs() -> dict:
    """Return the full costs.toml dict. Falls back to defaults if file missing."""
    if not _COSTS_PATH.exists():
        return {
            "hyperliquid": {"taker_bps": 3.5, "maker_bps": 0.2, "round_trip_taker_bps": 7.0},
            "binance": {"vip9_round_trip_bps": 1.61},
        }
    return _load_toml(_COSTS_PATH)


def taker_bps() -> float:
    """Hyperliquid one-way taker fee in bps."""
    return load_costs().get("hyperliquid", {}).get("taker_bps", 3.5)


def maker_bps() -> float:
    """Hyperliquid one-way maker rebate in bps."""
    return load_costs().get("hyperliquid", {}).get("maker_bps", 0.2)


def round_trip_taker_bps() -> float:
    """Hyperliquid round-trip taker fee in bps."""
    return load_costs().get("hyperliquid", {}).get("round_trip_taker_bps", 7.0)


def slippage_bps() -> float:
    """Hyperliquid one-way slippage assumption in bps."""
    return load_costs().get("hyperliquid", {}).get("slippage_bps", 2.0)


def realistic_taker_rt_bps() -> float:
    """The honest all-in round-trip taker cost on the venue NAT trades:
    Hyperliquid RT fee + slippage both ways (~11 bps). THE default for every
    evaluation harness — the VIP9 tier below is explicit-opt-in only
    (Q4 kill gate, FINDINGS §4.6: defaulting to it created 5/5 false winners)."""
    return round_trip_taker_bps() + 2.0 * slippage_bps()


def binance_vip9_rt_bps() -> float:
    """Binance VIP9 round-trip fee in bps. EXPLICIT OPT-IN comparison tier only —
    never a harness default (wrong venue; see realistic_taker_rt_bps)."""
    return load_costs().get("binance", {}).get("vip9_round_trip_bps", 1.61)
