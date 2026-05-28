"""
Config schema validation for NAT TOML files.

Validates expected keys, types, and value ranges for each config file.
Exits non-zero if any errors are found.

Usage:
    python -m scripts.utils.validate_config
    # or: make validate-config
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

# ---------------------------------------------------------------------------
# Schema definitions: {section: {key: (type, required, validator?)}}
# validator is an optional callable(value) -> str|None (error message or None)
# ---------------------------------------------------------------------------


def _range(lo: float, hi: float):
    """Return a validator that checks lo <= value <= hi."""
    def check(v: Any) -> str | None:
        if isinstance(v, (int, float)) and not (lo <= v <= hi):
            return f"expected {lo}..{hi}, got {v}"
        return None
    return check


def _positive(v: Any) -> str | None:
    if isinstance(v, (int, float)) and v <= 0:
        return f"expected positive, got {v}"
    return None


def _non_negative(v: Any) -> str | None:
    if isinstance(v, (int, float)) and v < 0:
        return f"expected non-negative, got {v}"
    return None


def _one_of(*choices):
    def check(v: Any) -> str | None:
        if v not in choices:
            return f"expected one of {choices}, got {v!r}"
        return None
    return check


# Type aliases
S, I, F, B, L = str, int, float, bool, list

SCHEMAS: dict[str, dict[str, dict[str, tuple]]] = {
    "ing.toml": {
        "general": {
            "log_level": (S, True, _one_of("trace", "debug", "info", "warn", "error")),
            "data_dir": (S, True, None),
        },
        "websocket": {
            "url": (S, True, None),
            "reconnect_delay_ms": (I, True, _positive),
            "max_reconnect_delay_ms": (I, True, _positive),
            "ping_interval_ms": (I, True, _positive),
        },
        "symbols": {
            "assets": (L, True, None),
        },
        "features": {
            "emission_interval_ms": (I, True, _positive),
            "trade_buffer_seconds": (I, True, _positive),
            "book_levels": (I, True, _range(1, 50)),
            "price_buffer_size": (I, True, _positive),
        },
        "output": {
            "format": (S, True, _one_of("parquet")),
            "row_group_size": (I, True, _positive),
            "compression": (S, True, _one_of("zstd", "snappy", "gzip", "none")),
            "rotate_interval": (S, True, _one_of("1h", "1d")),
        },
        "redis": {
            "url": (S, True, None),
        },
        "dashboard": {
            "enabled": (B, True, None),
            "addr": (S, True, None),
        },
    },
    "costs.toml": {
        "hyperliquid": {
            "taker_bps": (F, True, _non_negative),
            "maker_bps": (F, True, None),
            "round_trip_taker_bps": (F, True, _non_negative),
            "slippage_bps": (F, True, _non_negative),
            "funding_interval_hours": (F, True, _positive),
        },
    },
    "it_engine.toml": {
        "it_engine": {
            "buffer_size": (I, True, _positive),
            "compute_interval_s": (I, True, _positive),
            "ksg_k": (I, True, _range(1, 50)),
            "horizons": (L, True, None),
            "stride_divisor": (I, True, _positive),
            "max_features_greedy": (I, True, _positive),
            "te_top_n": (I, True, _positive),
            "te_lag": (I, True, _positive),
            "te_order": (I, True, _positive),
            "te_method": (S, True, _one_of("linear", "ksg")),
        },
    },
    "alpha.toml": {
        "pipeline": {
            "data_dir": (S, True, None),
            "timeframe": (S, True, None),
            "primary_symbol": (S, True, None),
        },
        "screener": {
            "fdr_alpha": (F, True, _range(0, 1)),
            "min_ic": (F, True, _non_negative),
        },
        "combiner": {
            "top_n": (I, True, _positive),
            "max_corr": (F, True, _range(0, 1)),
        },
        "gates": {
            "g1_min_significant": (I, True, _non_negative),
            "g4_min_oos_sharpe": (F, True, None),
            "g4_max_drawdown_pct": (F, True, _positive),
        },
    },
    "agent.toml": {
        "defaults": {
            "max_cycle_runtime_s": (I, True, _positive),
        },
        "defaults.gates": {
            "fdr_q": (F, True, _range(0, 1)),
            "min_oos_dates": (I, True, _positive),
            "min_symbols": (I, True, _positive),
        },
        "agent": {
            "cycle_interval_s": (I, True, _positive),
            "max_experiments_per_cycle": (I, True, _positive),
        },
        "alerts": {
            "redis_url": (S, True, None),
            "channels": (L, True, None),
        },
    },
}


def _get_nested(data: dict, dotted_key: str) -> dict | None:
    """Traverse nested dict by dotted key like 'defaults.gates'."""
    parts = dotted_key.split(".")
    node = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


TYPE_NAMES = {S: "string", I: "int", F: "number", B: "bool", L: "array"}


def validate_file(filename: str, schema: dict[str, dict[str, tuple]]) -> list[str]:
    """Validate a single TOML file against its schema. Returns list of error strings."""
    path = CONFIG_DIR / filename
    errors: list[str] = []

    if not path.exists():
        errors.append(f"{filename}: file not found at {path}")
        return errors

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        errors.append(f"{filename}: failed to parse TOML: {e}")
        return errors

    for section_key, keys in schema.items():
        section = _get_nested(data, section_key)
        if section is None:
            errors.append(f"{filename}: missing section [{section_key}]")
            continue

        for key, (expected_type, required, validator) in keys.items():
            if key not in section:
                if required:
                    errors.append(f"{filename}: [{section_key}] missing required key '{key}'")
                continue

            value = section[key]

            # Type check (float also accepts int)
            if expected_type is F:
                if not isinstance(value, (int, float)):
                    type_name = TYPE_NAMES.get(expected_type, str(expected_type))
                    errors.append(
                        f"{filename}: [{section_key}].{key} expected {type_name}, "
                        f"got {type(value).__name__}"
                    )
                    continue
            elif not isinstance(value, expected_type):
                type_name = TYPE_NAMES.get(expected_type, str(expected_type))
                errors.append(
                    f"{filename}: [{section_key}].{key} expected {type_name}, "
                    f"got {type(value).__name__}"
                )
                continue

            # Value range / enum check
            if validator is not None:
                msg = validator(value)
                if msg:
                    errors.append(f"{filename}: [{section_key}].{key} {msg}")

    return errors


def main() -> int:
    all_errors: list[str] = []

    for filename, schema in SCHEMAS.items():
        errs = validate_file(filename, schema)
        all_errors.extend(errs)

    if all_errors:
        print(f"Config validation FAILED ({len(all_errors)} error(s)):\n")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"Config validation OK — {len(SCHEMAS)} files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
