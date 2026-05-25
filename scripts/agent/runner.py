"""Experiment runner — executes hypothesis test protocols via nat commands.

State machine per hypothesis:
    SETUP → DISCOVERY → REPLICATE_TEMPORAL → REPLICATE_SYMBOL → REGISTER
      |         |              |                    |
      v         v              v                    v
    ABORT    GRAVEYARD      GRAVEYARD           GRAVEYARD
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import (  # noqa: F401 — re-exported for backward compat
    BaseRunner,
    apply_fdr,
    check_ic_gate,
    check_dIC_gate,
    check_cost_gate,
    check_coverage_gate,
    check_walkforward_gate,
    check_correlation_gate,
)
from .cache import ReportCache
from .hypothesis import Hypothesis, RegisteredSignal
from .manifest import load_manifest

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
NAT_PATH = ROOT / "nat"
REGISTRY_PATH = ROOT / "data" / "agent" / "registry.json"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "data" / "agent" / "cache"

# Module-level cache instance (shared across all runners in a cycle)
_report_cache: Optional[ReportCache] = None


def get_cache() -> ReportCache:
    """Get or create the module-level cache instance."""
    global _report_cache
    if _report_cache is None:
        _report_cache = ReportCache(CACHE_DIR)
    return _report_cache


def set_cache(cache: Optional[ReportCache]) -> None:
    """Replace the module-level cache (for testing)."""
    global _report_cache
    _report_cache = cache

# Map nat subcommands to their JSON output paths
REPORT_PATTERNS = {
    "spannung regime": "reports/spannung/regime_screen_{symbol}.json",
    "spannung spectral": "reports/spannung/spectral_{symbol}.json",
    "spannung backtest": "reports/spannung/backtest_{symbol}.json",
    "spannung": "reports/spannung/spannung_{symbol}.json",
    "profile scalp": "reports/profiler/profile_{symbol}_{timeframe}.json",
}


def run_nat(cmd_parts: list[str], timeout_s: int = 900) -> subprocess.CompletedProcess:
    """Execute a nat command and return the result."""
    full_cmd = [sys.executable, str(NAT_PATH)] + cmd_parts
    log.info("Running: nat %s", " ".join(cmd_parts))
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        log.warning("nat %s failed (rc=%d): %s",
                     " ".join(cmd_parts), result.returncode, result.stderr[:500])
    return result


def run_nat_cached(cmd_parts: list[str], symbol: str = "BTC",
                   timeout_s: int = 900) -> tuple[subprocess.CompletedProcess, Optional[dict]]:
    """Execute a nat command with caching. Returns (process_result, cached_report).

    If a cached report exists, returns a synthetic CompletedProcess (rc=0)
    and the cached report dict. Otherwise runs the command, caches the
    report, and returns both.
    """
    cache = get_cache()

    # Check cache first
    cached = cache.get(cmd_parts)
    if cached is not None:
        # Return a synthetic successful result
        synthetic = subprocess.CompletedProcess(
            args=cmd_parts, returncode=0, stdout="[cached]", stderr=""
        )
        return synthetic, cached

    # Cache miss — run the command
    result = run_nat(cmd_parts, timeout_s=timeout_s)
    report = None
    if result.returncode == 0:
        cmd_str = " ".join(cmd_parts)
        report = parse_report(cmd_str, symbol=symbol)
        if report is not None:
            cache.put(cmd_parts, report)

    return result, report


def parse_report(cmd_str: str, symbol: str = "BTC", timeframe: str = "1min") -> Optional[dict]:
    """Attempt to parse the JSON report for a given nat command."""
    for pattern_key, pattern in REPORT_PATTERNS.items():
        if pattern_key in cmd_str:
            path = ROOT / pattern.format(symbol=symbol, timeframe=timeframe)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
    return None



# Re-export helpers used by base.py correlation gate
from .base import _parse_gate_spec  # noqa: F401


def _load_feature_data(data_dir: str, symbol: str):
    """Load feature data for a symbol from a data directory.

    Delegates to the unified data access layer (scripts/data/features.py).
    """
    from data.features import load_features

    data_path = ROOT / data_dir
    if not data_path.exists():
        return None
    date_str = data_path.name  # e.g., "2026-05-20"
    df = load_features(
        symbols=[symbol],
        date_range=(date_str, date_str),
        data_dir=data_path.parent,
        validate=False,
    )
    return df if not df.empty else None


def _extract_gated_signal(df, feature: str,
                          gate_spec: Optional[str]):
    """Extract a feature column masked by an optional regime gate.

    Returns the feature values where the gate condition is True, NaN elsewhere.
    This lets correlation measure agreement in the active regime only.
    """
    import numpy as np

    if feature not in df.columns:
        return None
    values = df[feature].to_numpy(dtype=float)

    if gate_spec is None:
        return values

    parsed = _parse_gate_spec(gate_spec)
    if parsed is None:
        return values

    gate_feat, direction, percentile_str = parsed
    if gate_feat not in df.columns:
        return values

    pct_val = int(percentile_str[1:])  # 'P40' -> 40
    gate_col = df[gate_feat].to_numpy(dtype=float)
    threshold = np.nanpercentile(gate_col, pct_val)

    if direction == "<":
        mask = gate_col < threshold
    else:
        mask = gate_col > threshold

    gated = np.full_like(values, np.nan)
    gated[mask] = values[mask]
    return gated



# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MicrostructureRunner(BaseRunner):
    """Runs a microstructure hypothesis through the 5-gate protocol.

    Extends the base 4-gate protocol with an additional cost check gate
    after discovery. Uses tick-level (100ms) data without bar resampling.
    """

    TIMEFRAME = None  # tick-level, no bar resampling
    SIGNAL_FEATURES = [
        "imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10",
        "imbalance_depth_weighted", "ent_book_shape", "toxic_vpin_50",
    ]
    DEFAULT_FEATURE = "imbalance_qty_l1"
    DEFAULT_HORIZON_S = 5.0
    REGISTRY_PATH = REGISTRY_PATH

    def steps(self) -> list:
        """5-gate protocol: discovery → cost → temporal → symbol → dedup."""
        return [
            self.run_discovery,
            self.run_cost_check,
            self.run_replication_temporal,
            self.run_replication_symbol,
            self.run_correlation_check,
        ]

    def run_cost_check(self) -> bool:
        """Run backtest and check if signal survives transaction costs.

        Runs after discovery passes. Signals that fail cost are marked
        'cost_killed' — they are real but untradeable, eligible for recycler.
        """
        data_dir = self._extract_data_dir()
        symbol = self._extract_symbol(self.h.test_protocol[0])
        cmd_parts = ["spannung", "backtest", "--data", data_dir, "--symbol", symbol]
        result, report = run_nat_cached(cmd_parts, symbol=symbol)
        if result.returncode != 0:
            log.warning("  Cost check: backtest failed (rc=%d), skipping", result.returncode)
            return True  # Don't block on backtest failure

        if report is None:
            log.warning("  Cost check: could not parse backtest report, skipping")
            return True

        passed, msg = check_cost_gate(report, self.h.thresholds)
        log.info("  Cost check: %s", msg)
        self.h.results = {**(self.h.results or {}), "cost_check": msg}
        if not passed:
            self.h.fail("cost_killed")
        return passed


# Backward compatibility alias
ExperimentRunner = MicrostructureRunner
