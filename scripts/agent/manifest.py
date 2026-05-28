"""Data manifest builder — scans parquet files and reports availability."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from config_utils import load_symbols
except ImportError:
    from config_utils import load_symbols

log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
MANIFEST_PATH = Path("data/agent/manifest.json")
SYMBOLS = load_symbols()
ROWS_PER_HOUR = 36_000  # 10 Hz * 3600s


def scan_date_dir(date_dir: Path) -> dict[str, Any]:
    """Scan a single date directory for parquet metadata."""
    parquets = sorted(date_dir.glob("*.parquet"))
    valid = []
    corrupted = []
    total_size = 0

    for p in parquets:
        size = p.stat().st_size
        if size == 0:
            corrupted.append(p.name)
            continue
        # Quick magic byte check (PAR1)
        try:
            with open(p, "rb") as f:
                magic = f.read(4)
            if magic != b"PAR1":
                corrupted.append(p.name)
                continue
        except Exception:
            corrupted.append(p.name)
            continue
        valid.append(p.name)
        total_size += size

    return {
        "date": date_dir.name,
        "n_files": len(valid),
        "n_corrupted": len(corrupted),
        "corrupted": corrupted,
        "total_size_mb": round(total_size / 1e6, 1),
    }


def estimate_symbol_hours(date_info: dict, n_symbols: int = 3) -> float:
    """Rough estimate of hours per symbol from file count and size."""
    # Each parquet ≈ 10k rows (5.5 min for 3 symbols). Rough heuristic.
    # More accurate would be reading row counts, but that's slow.
    n_files = date_info["n_files"]
    if n_files == 0:
        return 0.0
    # Each file ≈ 10k rows, each symbol gets 1/3 of rows, 10 rows/sec
    rows_per_symbol = (n_files * 10_000) / n_symbols
    hours = rows_per_symbol / ROWS_PER_HOUR
    return round(hours, 1)


def build_manifest(
    features_dir: Path = FEATURES_DIR,
    output: Path = MANIFEST_PATH,
) -> dict:
    """Scan all date directories and build the manifest."""
    features_dir = Path(features_dir)
    if not features_dir.exists():
        log.warning("Features directory not found: %s", features_dir)
        return {"dates": {}, "symbols": {}, "updated": ""}

    date_dirs = sorted(
        [d for d in features_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()],
        key=lambda d: d.name,
    )

    dates = {}
    total_hours = 0.0
    for dd in date_dirs:
        info = scan_date_dir(dd)
        info["hours_per_symbol"] = estimate_symbol_hours(info)
        dates[dd.name] = info
        total_hours += info["hours_per_symbol"]

    # Per-symbol summary
    symbols = {}
    for sym in SYMBOLS:
        symbols[sym] = {
            "hours": total_hours,
            "dates": list(dates.keys()),
        }

    manifest = {
        "dates": dates,
        "symbols": symbols,
        "total_dates": len(dates),
        "total_hours_per_symbol": round(total_hours, 1),
        "updated": datetime.now(timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest: %d dates, ~%.1f hours/symbol", len(dates), total_hours)
    return manifest


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load the manifest from disk."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"dates": {}, "symbols": {}}
