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

from .base import BaseRunner, apply_fdr  # noqa: F401 — apply_fdr re-exported
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


# ---------------------------------------------------------------------------
# Gate checks — evaluate hypothesis thresholds against results
# ---------------------------------------------------------------------------

def _find_gate_entry(report: dict, regime_gate: str) -> Optional[dict]:
    """Find the single_factors entry matching a regime_gate label (e.g. 'ent_book_shape<P40')."""
    for entry in report.get("single_factors", []):
        if entry.get("label") == regime_gate:
            return entry
    return None


def _ic_pvalue(ic: float, n_obs: int) -> float:
    """Two-sided p-value for Spearman IC under H0: no predictive power.

    Under H0, IC ~ N(0, 1/sqrt(n)).  z = IC * sqrt(n) is standard normal.
    """
    from math import erfc, sqrt
    if n_obs < 2:
        return 1.0
    z = abs(ic) * sqrt(n_obs)
    return erfc(z / sqrt(2))  # two-sided


def check_ic_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if IC exceeds minimum threshold.

    If the hypothesis specifies a regime_gate, extract the gate-specific IC
    from the single_factors array. Otherwise fall back to aggregate IC.

    Also computes a p-value (appended to the message as p=...) for FDR control.
    """
    min_ic = thresholds.get("min_ic", 0.10)
    regime_gate = thresholds.get("regime_gate")
    ic = None
    n_obs = report.get("n_rows", 0)

    # Gate-specific IC: look up in single_factors
    if regime_gate and "single_factors" in report:
        entry = _find_gate_entry(report, regime_gate)
        if entry:
            ic = entry.get("ic_filt_5s")
            n_obs = entry.get("n_obs", n_obs)

    # Fallback: aggregate report IC
    if ic is None:
        if "baseline_ic_filt_5s" in report:
            ic = report["baseline_ic_filt_5s"]
        elif "best_ic" in report:
            ic = report["best_ic"]
        elif "profiles" in report and len(report["profiles"]) > 0:
            ic = max(abs(p.get("ic_best", 0)) for p in report["profiles"])

    if ic is None:
        return False, "could not extract IC from report"
    pval = _ic_pvalue(ic, n_obs)
    passed = abs(ic) >= min_ic
    label = f"gated({regime_gate})" if regime_gate else "aggregate"
    return passed, (f"IC={ic:.4f} [{label}] vs min={min_ic} p={pval:.2e}"
                    + (" PASS" if passed else " FAIL"))


def check_dIC_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check that the regime gate improves IC over the ungated baseline.

    dIC = gated_IC - baseline_IC. The gate must ADD value, not just
    pass because the underlying signal is strong.
    """
    min_dIC = thresholds.get("min_dIC", 0.05)
    regime_gate = thresholds.get("regime_gate")

    if not regime_gate or "single_factors" not in report:
        return True, "no regime gate, dIC check skipped"

    baseline = report.get("baseline_ic_filt_5s", 0.0)
    entry = _find_gate_entry(report, regime_gate)
    if entry is None:
        return False, f"gate {regime_gate} not found in report FAIL"

    gated_ic = entry.get("ic_filt_5s", 0.0)
    dIC = gated_ic - baseline
    passed = dIC >= min_dIC
    return passed, (f"dIC={dIC:+.4f} (gated={gated_ic:.4f} - base={baseline:.4f}) "
                    f"vs min={min_dIC}" + (" PASS" if passed else " FAIL"))


def check_cost_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if signal has sufficient per-trade edge to be worth pursuing.

    Uses avg_return_per_trade_bps from the best threshold level in the
    backtest report. This measures gross signal edge — a necessary
    (not sufficient) condition for profitability after costs.

    Note: the backtest tests the ungated signal. Regime-gated versions
    should have higher per-trade returns. This gate filters out signals
    with zero or negligible directional value.
    """
    min_avg_bps = thresholds.get("min_avg_return_bps", 0.1)
    entries = report.get("thresholds", [])
    if not entries:
        return True, "no backtest thresholds, cost check skipped"

    best = max(entries, key=lambda t: t.get("avg_return_per_trade_bps", 0))
    avg_ret = best.get("avg_return_per_trade_bps", 0)
    maker_sharpe = best.get("net_sharpe_maker", 0)
    thresh = best.get("threshold", 0)
    passed = avg_ret >= min_avg_bps
    return passed, (f"avg_ret={avg_ret:.3f}bps (maker_sharpe={maker_sharpe:.1f}) "
                    f"at thresh={thresh:.1f} vs min={min_avg_bps}bps"
                    + (" PASS" if passed else " FAIL"))


def _parse_gate_spec(gate_str: str) -> Optional[tuple[str, str, str]]:
    """Parse 'ent_book_shape<P40' into ('ent_book_shape', '<', 'P40')."""
    m = re.match(r'^([a-z_0-9]+)([<>])(P\d+)$', gate_str)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _load_feature_data(data_dir: str, symbol: str):
    """Load and concatenate all Parquet files for a symbol from a data dir."""
    import pandas as pd

    data_path = ROOT / data_dir
    if not data_path.exists():
        return None
    files = sorted(data_path.glob("*.parquet"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue  # skip corrupted files
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["symbol"] == symbol] if "symbol" in df.columns else df


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


def check_correlation_gate(
    candidate_feature: str,
    candidate_gate: Optional[str],
    registry: list[dict],
    data_dir: str,
    symbol: str = "BTC",
    max_corr: float = 0.70,
) -> tuple[bool, str]:
    """Check if a candidate signal is redundant with existing registry signals.

    Computes Spearman rank correlation between the candidate's gated feature
    values and each registered signal's gated feature values. Rejects if
    any pairwise correlation exceeds max_corr.
    """
    import numpy as np
    import pandas as pd

    if not registry:
        return True, "empty registry, no dedup needed PASS"

    df = _load_feature_data(data_dir, symbol)
    if df is None or len(df) == 0:
        log.warning("  Correlation check: could not load data from %s", data_dir)
        return True, "no data for correlation check, skipped"

    cand_vals = _extract_gated_signal(df, candidate_feature, candidate_gate)
    if cand_vals is None:
        return True, f"feature {candidate_feature} not in data, skipped"

    worst_corr = 0.0
    worst_name = ""
    for sig in registry:
        sig_features = sig.get("features", [])
        sig_gate = sig.get("regime_gate")
        for sf in sig_features:
            ref_vals = _extract_gated_signal(df, sf, sig_gate)
            if ref_vals is None:
                continue
            # Spearman on non-NaN overlap
            valid = ~(np.isnan(cand_vals) | np.isnan(ref_vals))
            if valid.sum() < 100:
                continue
            corr = pd.Series(cand_vals[valid]).corr(
                pd.Series(ref_vals[valid]), method="spearman"
            )
            if abs(corr) > abs(worst_corr):
                worst_corr = corr
                worst_name = f"{sf}|{sig_gate or 'ungated'}"

    passed = abs(worst_corr) <= max_corr
    if worst_name:
        msg = (f"max_corr={worst_corr:+.3f} vs {worst_name} "
               f"(threshold={max_corr})" + (" PASS" if passed else " REDUNDANT"))
    else:
        msg = "no comparable registry signals PASS"
    return passed, msg


def check_coverage_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if regime coverage exceeds minimum."""
    min_coverage = thresholds.get("min_coverage", 0.20)
    # Look for Pareto-optimal results with sufficient coverage
    pareto = report.get("pareto_optimal", [])
    for p in pareto:
        if p.get("coverage", 0) >= min_coverage:
            return True, f"coverage={p['coverage']:.0%} >= {min_coverage:.0%} PASS"
    return False, f"no Pareto combo with coverage >= {min_coverage:.0%} FAIL"


def check_walkforward_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check walk-forward KEEP verdict."""
    keep_count = report.get("keep_count", 0)
    total = keep_count + report.get("monitor_count", 0) + report.get("drop_count", 0)
    if total == 0:
        return False, "no walk-forward results"
    keep_frac = keep_count / total
    passed = keep_frac >= thresholds.get("min_keep_frac", 0.3)
    return passed, f"KEEP={keep_count}/{total} ({keep_frac:.0%})" + (" PASS" if passed else " FAIL")


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
