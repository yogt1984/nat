"""15-Minute Smoke Test — fast validation of NAT ingestor data quality.

Runs a 5-phase pipeline (collect → validate → profile → cluster → report)
on 15 minutes of feature data to catch schema mismatches, NaN explosions,
emission failures, and corrupt writes before committing to hours of collection.

Usage:
    python3 scripts/15m_test.py run --data-dir data/features/2026-05-12-clean
    python3 scripts/15m_test.py run --live
    python3 scripts/15m_test.py run --data-dir data/features/2026-05-12-clean --skip-cluster -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Allow imports from scripts/
from cluster_pipeline.config import FEATURE_VECTORS, META_COLUMNS
from cluster_pipeline.loader import (
    get_duration_seconds,
    get_symbols,
    list_parquet_files,
    load_parquet,
    scan_schema,
)
from cluster_pipeline.preprocess import aggregate_bars

log = logging.getLogger("15m_test")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "pipeline.toml"
DEFAULT_OUTPUT = ROOT / "reports" / "smoke_test"
DEFAULT_DATA = ROOT / "data" / "features"

# Core vectors — these must have low NaN
CORE_VECTORS = [
    "entropy", "flow", "orderflow", "volatility", "trend",
    "illiquidity", "toxicity", "context", "derived",
]

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

DEFAULT_THRESHOLDS = {
    "live_duration_minutes": 15,
    "expected_symbols": ["BTC", "ETH", "SOL"],
    "expected_rate_per_sec": 10,
    "emission_rate_tolerance": 0.20,
    "cross_symbol_max_diff": 0.01,
    "nan_threshold_core": 0.05,
    "max_gap_seconds": 5.0,
    "warmup_seconds": 30,
    "optional_vectors": ["whale", "liquidation", "concentration"],
    "cluster_k": 4,
    "cluster_max_samples": 5000,
    "silhouette_floor": -0.1,
    "bar_timeframe": "1min",
}


def load_config(path: Path) -> dict:
    """Load [smoke_test] from pipeline.toml, fall back to defaults."""
    cfg = dict(DEFAULT_THRESHOLDS)
    if path.exists():
        with open(path, "rb") as f:
            toml = tomllib.load(f)
        if "smoke_test" in toml:
            cfg.update(toml["smoke_test"])
        else:
            log.warning("No [smoke_test] section in %s, using defaults", path)
    else:
        log.warning("Config %s not found, using defaults", path)
    return cfg


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    name: str
    passed: bool
    critical: bool
    message: str
    details: dict = field(default_factory=dict)
    timing_ms: float = 0.0


@dataclass
class PhaseResult:
    name: str
    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True
    gated: bool = False
    timing_s: float = 0.0


@dataclass
class SmokeTestReport:
    timestamp: str = ""
    data_dir: str = ""
    total_rows: int = 0
    symbols: list[str] = field(default_factory=list)
    phases: list[PhaseResult] = field(default_factory=list)
    overall_passed: bool = True
    critical_passed: bool = True


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def _vector_columns(df: pd.DataFrame, vector_name: str) -> list[str]:
    """Return columns from df that belong to a named feature vector."""
    vec = FEATURE_VECTORS.get(vector_name, {})
    known = set(vec.get("columns", []))
    prefixes = vec.get("prefixes", [])
    cols = []
    for c in df.columns:
        if c in known or any(c.startswith(p) for p in prefixes):
            if c not in cols:
                cols.append(c)
    return cols


def _core_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns belonging to core feature vectors."""
    cols = []
    for vec in CORE_VECTORS:
        for c in _vector_columns(df, vec):
            if c not in cols:
                cols.append(c)
    return cols


def _optional_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return numeric columns belonging to optional feature vectors."""
    cols = []
    for vec in cfg.get("optional_vectors", []):
        for c in _vector_columns(df, vec):
            if c not in cols:
                cols.append(c)
    return cols


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns except metadata."""
    meta = META_COLUMNS | {"__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta]


# --------------------------------------------------------------------------- #
#  Phase 1 — Collect
# --------------------------------------------------------------------------- #


def _fmt_duration(seconds: float) -> str:
    """Format seconds as compact human string: 14m34s, 1h34m4s, 45s."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m}m{sec}s" if sec else f"{h}h{m}m"
    return f"{m}m{sec}s" if sec else f"{m}m"


def _trim_to_window(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """Trim dataframe to the last N minutes of data."""
    ts_max = df["timestamp_ns"].max()
    ts_min = ts_max - int(window_minutes * 60 * 1e9)
    trimmed = df[df["timestamp_ns"] >= ts_min].copy()
    dur = (trimmed["timestamp_ns"].max() - trimmed["timestamp_ns"].min()) / 1e9
    log.info("Trimmed to last %d min: %d → %d rows (%.1f min actual)",
             window_minutes, len(df), len(trimmed), dur / 60)
    return trimmed


def phase_collect(cfg: dict, data_dir: Optional[Path], live: bool,
                  duration_s: Optional[int] = None) -> tuple[pd.DataFrame, Path]:
    """Load or wait for data. Returns (dataframe, data_dir_used)."""
    window = (duration_s / 60) if duration_s else cfg.get("live_duration_minutes", 15)

    if not live:
        if data_dir is None:
            raise ValueError("--data-dir required in offline mode")
        log.info("Loading data from %s", data_dir)
        df = load_parquet(str(data_dir))
        duration = get_duration_seconds(df)
        if duration < 300:
            log.warning("Only %.0fs of data (< 5 min)", duration)
        log.info("Loaded %d rows, %.1f min", len(df), duration / 60)
        # Auto-trim to last 15 minutes if data exceeds the window
        if duration > window * 60 * 1.5:
            df = _trim_to_window(df, window)
        return df, data_dir

    # Live mode: poll for data
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data_dir = DEFAULT_DATA / today
    duration_s = window * 60
    target_rows = int(
        cfg["expected_rate_per_sec"]
        * len(cfg["expected_symbols"])
        * duration_s
        * 0.8
    )

    log.info("Live mode: waiting for %d rows in %s (%.0f min)", target_rows, data_dir, window)
    start = time.time()
    timeout = duration_s + 300  # 5 min grace

    while time.time() - start < timeout:
        if data_dir.exists():
            try:
                info = scan_schema(str(data_dir))
                rows = info.get("total_rows", 0) if isinstance(info, dict) else 0
            except Exception:
                rows = 0
            if rows >= target_rows:
                log.info("Collected %d rows, loading...", rows)
                df = load_parquet(str(data_dir))
                df = _trim_to_window(df, window)
                return df, data_dir
        time.sleep(10)

    raise TimeoutError(f"Timed out waiting for {target_rows} rows after {timeout}s")


# --------------------------------------------------------------------------- #
#  Phase 2 — Validate (gate)
# --------------------------------------------------------------------------- #


def check_file_integrity(data_dir: Path) -> CheckResult:
    t0 = time.perf_counter()
    try:
        files = list(data_dir.glob("*.parquet"))
        if not files:
            return CheckResult("File Integrity", False, True, "No parquet files found",
                               timing_ms=(time.perf_counter() - t0) * 1000)
        bad = []
        for f in files:
            try:
                pq.read_metadata(f)
            except Exception as e:
                bad.append(f"{f.name}: {str(e)[:80]}")
        passed = len(bad) == 0
        msg = f"{len(files) - len(bad)}/{len(files)} files readable"
        if bad:
            msg += f" ({len(bad)} corrupt)"
        return CheckResult("File Integrity", passed, True, msg,
                           details={"corrupt": bad},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("File Integrity", False, True, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_schema(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        issues = []
        if "timestamp_ns" not in df.columns:
            issues.append("missing timestamp_ns")
        if "symbol" not in df.columns:
            issues.append("missing symbol")

        # Check each base vector has at least one matching column
        base_vectors = [v for v in FEATURE_VECTORS if v not in ("whale", "liquidation", "concentration")]
        for vec in base_vectors:
            found = len(_vector_columns(df, vec)) > 0
            if not found:
                issues.append(f"no columns for vector '{vec}'")

        n_features = len(_numeric_feature_cols(df))
        passed = len(issues) == 0
        msg = f"{n_features} features" if passed else f"Schema issues: {', '.join(issues)}"
        return CheckResult("Schema", passed, True, msg,
                           details={"issues": issues, "n_features": n_features},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Schema", False, True, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_emission_rate(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        expected = cfg["expected_rate_per_sec"]
        tol = cfg["emission_rate_tolerance"]
        lo, hi = expected * (1 - tol), expected * (1 + tol)

        rates = {}
        issues = []
        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym].sort_values("timestamp_ns")
            dur = (sub["timestamp_ns"].iloc[-1] - sub["timestamp_ns"].iloc[0]) / 1e9
            if dur > 0:
                rate = len(sub) / dur
                rates[sym] = round(rate, 1)
                if rate < lo or rate > hi:
                    issues.append(f"{sym}: {rate:.1f}/sec (expected {lo:.0f}-{hi:.0f})")
            else:
                issues.append(f"{sym}: zero duration")

        passed = len(issues) == 0
        msg = f"All symbols {lo:.0f}-{hi:.0f}/sec" if passed else "; ".join(issues)
        return CheckResult("Emission Rate", passed, True, msg,
                           details={"rates": rates},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Emission Rate", False, True, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_cross_symbol_consistency(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        counts = df["symbol"].value_counts()
        max_c, min_c = counts.max(), counts.min()
        diff = (max_c - min_c) / max_c if max_c > 0 else 0
        threshold = cfg["cross_symbol_max_diff"]
        passed = diff <= threshold
        msg = f"Row diff {diff:.2%} (threshold {threshold:.0%})"
        return CheckResult("Cross-Symbol Consistency", passed, False, msg,
                           details={"counts": counts.to_dict(), "diff": diff},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Cross-Symbol Consistency", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_nan_core(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        # Skip warmup period
        warmup_ns = int(cfg["warmup_seconds"] * 1e9)
        t_start = df["timestamp_ns"].min() + warmup_ns
        df_warm = df[df["timestamp_ns"] >= t_start]

        core_cols = _core_feature_cols(df_warm)
        if not core_cols:
            return CheckResult("NaN Core", False, True, "No core feature columns found",
                               timing_ms=(time.perf_counter() - t0) * 1000)

        threshold = cfg["nan_threshold_core"]
        bad = {}
        for c in core_cols:
            frac = df_warm[c].isna().mean()
            if frac > threshold:
                bad[c] = round(frac * 100, 1)

        passed = len(bad) == 0
        msg = f"All {len(core_cols)} core features <{threshold:.0%} NaN" if passed else f"{len(bad)} core features exceed {threshold:.0%} NaN"
        return CheckResult("NaN Core", passed, True, msg,
                           details={"violations": dict(list(bad.items())[:10]),
                                    "total_checked": len(core_cols)},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("NaN Core", False, True, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_nan_optional(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        opt_cols = _optional_feature_cols(df, cfg)
        if not opt_cols:
            return CheckResult("NaN Optional", True, False, "No optional columns found",
                               timing_ms=(time.perf_counter() - t0) * 1000)

        nan_fracs = {c: round(df[c].isna().mean() * 100, 1) for c in opt_cols}
        all_nan = sum(1 for v in nan_fracs.values() if v >= 99.9)
        msg = f"{all_nan}/{len(opt_cols)} optional features at 100% NaN (informational)"
        return CheckResult("NaN Optional", True, False, msg,
                           details={"nan_fracs": nan_fracs},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("NaN Optional", True, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_continuity(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        max_gap = cfg["max_gap_seconds"]
        total_gaps = 0
        worst = 0.0
        per_sym = {}

        for sym in df["symbol"].unique():
            sub = df[df["symbol"] == sym].sort_values("timestamp_ns")
            deltas = sub["timestamp_ns"].diff().dropna() / 1e9
            gaps = deltas[deltas > max_gap]
            total_gaps += len(gaps)
            if len(gaps) > 0:
                worst = max(worst, gaps.max())
                per_sym[sym] = {"count": int(len(gaps)), "max_s": round(float(gaps.max()), 1)}

        passed = total_gaps == 0
        msg = f"No gaps >{max_gap}s" if passed else f"{total_gaps} gaps >{max_gap}s (max {worst:.1f}s)"
        return CheckResult("Continuity", passed, False, msg,
                           details={"per_symbol": per_sym},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Continuity", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_sequence_monotonicity(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        issues = {}
        for sym in df["symbol"].unique():
            ts = df[df["symbol"] == sym]["timestamp_ns"].values
            non_mono = int(np.sum(np.diff(ts) <= 0))
            if non_mono > 0:
                issues[sym] = non_mono

        passed = len(issues) == 0
        msg = "All sequences monotonically increasing" if passed else f"{len(issues)} symbol(s) non-monotonic"
        return CheckResult("Sequence Monotonicity", passed, False, msg,
                           details={"issues": issues},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Sequence Monotonicity", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def phase_validate(df: pd.DataFrame, data_dir: Path, cfg: dict) -> PhaseResult:
    t0 = time.perf_counter()
    checks = [
        check_file_integrity(data_dir),
        check_schema(df),
        check_emission_rate(df, cfg),
        check_cross_symbol_consistency(df, cfg),
        check_nan_core(df, cfg),
        check_nan_optional(df, cfg),
        check_continuity(df, cfg),
        check_sequence_monotonicity(df),
    ]
    passed = all(c.passed for c in checks)
    gated = any(not c.passed and c.critical for c in checks)
    return PhaseResult("Validate", checks, passed, gated, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
#  Phase 3 — Profile
# --------------------------------------------------------------------------- #


def check_zero_variance(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        num_cols = _numeric_feature_cols(df)
        stds = df[num_cols].std()
        zero_var = stds[stds < 1e-10].index.tolist()
        frac = len(zero_var) / len(num_cols) if num_cols else 0
        passed = frac < 0.10
        msg = f"{len(zero_var)}/{len(num_cols)} features zero-variance ({frac:.0%})"
        return CheckResult("Zero Variance", passed, False, msg,
                           details={"zero_var_features": zero_var[:15], "fraction": frac},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Zero Variance", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_range(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        num_cols = _numeric_feature_cols(df)
        overflow = []
        for c in num_cols:
            mx = df[c].abs().max()
            if mx > 1e15:
                overflow.append((c, float(mx)))
        passed = len(overflow) == 0
        msg = "All features in range" if passed else f"{len(overflow)} features with |value| > 1e15"
        return CheckResult("Range Check", passed, False, msg,
                           details={"overflow": overflow[:10]},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Range Check", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_outlier_fraction(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        num_cols = _numeric_feature_cols(df)
        high_outlier = []
        for c in num_cols:
            col = df[c].dropna()
            if len(col) < 10:
                continue
            mean, std = col.mean(), col.std()
            if std > 0:
                frac = ((col - mean).abs() > 3 * std).mean()
                if frac > 0.05:
                    high_outlier.append((c, round(float(frac) * 100, 1)))

        frac_bad = len(high_outlier) / len(num_cols) if num_cols else 0
        passed = frac_bad < 0.20
        msg = f"{len(high_outlier)}/{len(num_cols)} features with >5% outliers ({frac_bad:.0%})"
        return CheckResult("Outlier Fraction", passed, False, msg,
                           details={"high_outlier": high_outlier[:10]},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Outlier Fraction", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_correlation_collapse(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        num_cols = _numeric_feature_cols(df)
        # Pick top 20 by variance
        variances = df[num_cols].var().dropna().sort_values(ascending=False)
        top20 = variances.head(20).index.tolist()
        if len(top20) < 3:
            return CheckResult("Correlation Collapse", True, False, "Too few features to check",
                               timing_ms=(time.perf_counter() - t0) * 1000)

        corr = df[top20].corr().abs()
        # Upper triangle only
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask)
        total_pairs = mask.sum()
        collapsed = (upper > 0.98).sum().sum()
        frac = collapsed / total_pairs if total_pairs > 0 else 0

        passed = frac < 0.80
        msg = f"{collapsed}/{total_pairs} pairs with |r|>0.98 ({frac:.0%})"
        return CheckResult("Correlation Collapse", passed, False, msg,
                           details={"fraction": frac, "collapsed_pairs": int(collapsed)},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Correlation Collapse", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_bar_aggregation(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        tf = cfg["bar_timeframe"]
        bars = aggregate_bars(df, tf)
        n_bars = len(bars)
        syms = bars["symbol"].nunique() if "symbol" in bars.columns else 0
        expected_cols = {"bar_start", "bar_end", "symbol", "tick_count"}
        missing = expected_cols - set(bars.columns)

        passed = n_bars > 0 and len(missing) == 0
        msg = f"{n_bars} bars at {tf}, {syms} symbols"
        if missing:
            msg += f" (missing cols: {missing})"
        return CheckResult("Bar Aggregation", passed, False, msg,
                           details={"n_bars": n_bars, "symbols": syms,
                                    "missing_cols": list(missing)},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Bar Aggregation", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def check_feature_completeness(df: pd.DataFrame) -> CheckResult:
    t0 = time.perf_counter()
    try:
        num_cols = _numeric_feature_cols(df)
        valid = sum(1 for c in num_cols if df[c].notna().mean() > 0.5)
        total = len(num_cols)
        frac = valid / total if total > 0 else 0
        msg = f"{valid}/{total} features with >50% valid data ({frac:.0%})"
        return CheckResult("Feature Completeness", True, False, msg,
                           details={"valid": valid, "total": total, "fraction": frac},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("Feature Completeness", True, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def phase_profile(df: pd.DataFrame, cfg: dict) -> PhaseResult:
    t0 = time.perf_counter()
    checks = [
        check_zero_variance(df),
        check_range(df),
        check_outlier_fraction(df),
        check_correlation_collapse(df),
        check_bar_aggregation(df, cfg),
        check_feature_completeness(df),
    ]
    passed = all(c.passed for c in checks)
    return PhaseResult("Profile", checks, passed, False, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
#  Phase 4 — Cluster
# --------------------------------------------------------------------------- #


def check_kmeans_cluster(df: pd.DataFrame, cfg: dict) -> CheckResult:
    t0 = time.perf_counter()
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Get entropy features
        num_set = set(df.select_dtypes(include=[np.number]).columns)
        ent_cols = [c for c in _vector_columns(df, "entropy") if c in num_set]

        if len(ent_cols) < 3:
            # Fallback to all numeric
            ent_cols = _numeric_feature_cols(df)[:30]

        # Sample for speed
        max_samples = cfg["cluster_max_samples"]
        if len(df) > max_samples:
            sample = df[ent_cols].sample(n=max_samples, random_state=42)
        else:
            sample = df[ent_cols]

        # Fill NaN with median
        X = sample.fillna(sample.median()).values

        # Remove any remaining NaN/inf
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]

        if len(X) < 100:
            return CheckResult("KMeans Cluster", False, False,
                               f"Only {len(X)} valid rows after cleaning",
                               timing_ms=(time.perf_counter() - t0) * 1000)

        k = cfg["cluster_k"]
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        # Check for degeneracy
        unique_labels = np.unique(labels)
        counts = np.bincount(labels)
        active_clusters = int((counts > 0).sum())

        if active_clusters < 2:
            return CheckResult("KMeans Cluster", False, False,
                               "Degenerate: all points in single cluster",
                               details={"active_clusters": active_clusters},
                               timing_ms=(time.perf_counter() - t0) * 1000)

        sil = float(silhouette_score(X, labels, sample_size=min(2000, len(X))))
        floor = cfg["silhouette_floor"]
        passed = sil > floor
        msg = f"Silhouette={sil:.3f} (floor={floor}), {active_clusters} active clusters"
        return CheckResult("KMeans Cluster", passed, False, msg,
                           details={"silhouette": sil, "active_clusters": active_clusters,
                                    "cluster_counts": counts.tolist(), "n_features": len(ent_cols)},
                           timing_ms=(time.perf_counter() - t0) * 1000)
    except Exception as e:
        return CheckResult("KMeans Cluster", False, False, f"Error: {e}",
                           timing_ms=(time.perf_counter() - t0) * 1000)


def phase_cluster(df: pd.DataFrame, cfg: dict) -> PhaseResult:
    t0 = time.perf_counter()
    checks = [check_kmeans_cluster(df, cfg)]
    passed = all(c.passed for c in checks)
    return PhaseResult("Cluster", checks, passed, False, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
#  Phase 5 — Report
# --------------------------------------------------------------------------- #


def _format_markdown(report: SmokeTestReport) -> str:
    lines = [
        f"# 15-Minute Smoke Test Report",
        f"Generated: {report.timestamp}",
        f"Data: {report.data_dir}",
        f"Rows: {report.total_rows:,} | Symbols: {', '.join(report.symbols)}",
        "",
    ]

    total_checks = sum(len(p.checks) for p in report.phases)
    passed_checks = sum(1 for p in report.phases for c in p.checks if c.passed)
    failed_checks = total_checks - passed_checks
    crit_failed = sum(1 for p in report.phases for c in p.checks if not c.passed and c.critical)

    status = "PASS" if report.overall_passed else "FAIL"
    lines.append(f"## Summary: **{status}**")
    lines.append(f"- Checks: {passed_checks}/{total_checks} passed, {failed_checks} failed ({crit_failed} critical)")
    total_time = sum(p.timing_s for p in report.phases)
    lines.append(f"- Total time: {total_time:.1f}s")
    lines.append("")

    for phase in report.phases:
        p_status = "PASS" if phase.passed else ("GATE FAIL" if phase.gated else "FAIL")
        lines.append(f"## Phase: {phase.name} ({p_status}) [{phase.timing_s:.1f}s]")
        lines.append("")
        lines.append("| Check | Status | Details |")
        lines.append("|-------|--------|---------|")
        for c in phase.checks:
            s = "PASS" if c.passed else "**FAIL**"
            crit = " [critical]" if c.critical and not c.passed else ""
            lines.append(f"| {c.name} | {s}{crit} | {c.message} |")
        lines.append("")

    # Timing table
    lines.append("## Timing")
    lines.append("| Phase | Duration |")
    lines.append("|-------|----------|")
    for phase in report.phases:
        lines.append(f"| {phase.name} | {phase.timing_s:.1f}s |")
    lines.append(f"| **Total** | **{total_time:.1f}s** |")
    lines.append("")

    return "\n".join(lines)


def _print_stdout(report: SmokeTestReport):
    """Print condensed pass/fail table to stdout."""
    total_checks = sum(len(p.checks) for p in report.phases)
    passed_checks = sum(1 for p in report.phases for c in p.checks if c.passed)

    overall = "\033[92mPASS\033[0m" if report.overall_passed else "\033[91mFAIL\033[0m"
    print(f"\n{'=' * 72}")
    print(f"  15-MINUTE SMOKE TEST — {overall}")
    print(f"  {report.total_rows:,} rows | {', '.join(report.symbols)} | {report.data_dir}")
    print(f"{'=' * 72}")

    for phase in report.phases:
        p_status = "\033[92mPASS\033[0m" if phase.passed else "\033[91mFAIL\033[0m"
        if phase.gated:
            p_status = "\033[91mGATE FAIL\033[0m"
        print(f"\n  [{p_status}] {phase.name} ({phase.timing_s:.1f}s)")
        for c in phase.checks:
            icon = "\033[92m+\033[0m" if c.passed else "\033[91m-\033[0m"
            crit = " [CRITICAL]" if c.critical and not c.passed else ""
            print(f"    [{icon}] {c.name}: {c.message}{crit}")

    total_time = sum(p.timing_s for p in report.phases)
    print(f"\n  {passed_checks}/{total_checks} checks passed in {total_time:.1f}s")
    print(f"{'=' * 72}\n")


def phase_report(report: SmokeTestReport, output_dir: Path) -> PhaseResult:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = output_dir / "smoke_test.md"
    md_path.write_text(_format_markdown(report))
    log.info("Markdown report: %s", md_path)

    # JSON
    json_path = output_dir / "smoke_test.json"
    json_path.write_text(json.dumps(asdict(report), indent=2, cls=_NumpyEncoder))
    log.info("JSON report: %s", json_path)

    # Stdout
    _print_stdout(report)

    print(f"  Reports: {md_path}")
    print(f"           {json_path}")

    return PhaseResult("Report", [], True, False, time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #


def _create_experiment_dir(output_dir: Path, used_dir: Path, df: pd.DataFrame,
                           report: SmokeTestReport) -> Path:
    """Create experiment subdirectory with 15m__ data snapshot and latest symlink."""
    exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"exp_{exp_ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Compute actual data duration for filename prefix
    actual_dur_s = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9
    dur_label = _fmt_duration(actual_dur_s)

    # Save data with duration prefix: e.g. 14m34s__20260512_192857.parquet
    data_path = exp_dir / f"{dur_label}__{exp_ts}.parquet"
    df.to_parquet(data_path)
    log.info("Saved %d rows to %s", len(df), data_path)

    # Write data reference pointing to local 15m__ file
    data_ref = {
        "data_file": str(data_path.resolve()),
        "source_dir": str(used_dir.resolve()),
        "rows": len(df),
        "symbols": report.symbols,
        "ts_min": int(df["timestamp_ns"].min()),
        "ts_max": int(df["timestamp_ns"].max()),
        "created": report.timestamp,
    }
    (exp_dir / "data_ref.json").write_text(json.dumps(data_ref, indent=2))

    # Update latest symlink
    latest = output_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(exp_dir.resolve())
    log.info("Experiment dir: %s", exp_dir)

    return exp_dir


def run_smoke_test(
    cfg: dict,
    data_dir: Optional[Path],
    live: bool,
    output_dir: Path,
    skip_cluster: bool,
    duration_s: Optional[int] = None,
) -> int:
    """Run all phases. Returns exit code (0/1/2)."""
    report = SmokeTestReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Phase 1: Collect
    try:
        df, used_dir = phase_collect(cfg, data_dir, live, duration_s)
    except Exception as e:
        log.error("Phase 1 (Collect) failed: %s", e)
        report.data_dir = str(data_dir or "live")
        report.overall_passed = False
        report.critical_passed = False
        phase_report(report, output_dir)
        return 2

    report.data_dir = str(used_dir)
    report.total_rows = len(df)
    report.symbols = sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else []

    # Create experiment directory with data reference
    exp_dir = _create_experiment_dir(output_dir, used_dir, df, report)

    # Phase 2: Validate (gate) — check against experiment dir (15m__ data, no corrupt source files)
    p2 = phase_validate(df, exp_dir, cfg)
    report.phases.append(p2)
    log.info("Phase 2 (Validate): %s [%.1fs]", "PASS" if p2.passed else "FAIL", p2.timing_s)

    if p2.gated:
        log.error("Gate tripped — skipping Profile and Cluster phases")
        report.overall_passed = False
        report.critical_passed = False
        report.phases.append(phase_report(report, exp_dir))
        return 2

    # Phase 3: Profile
    p3 = phase_profile(df, cfg)
    report.phases.append(p3)
    log.info("Phase 3 (Profile): %s [%.1fs]", "PASS" if p3.passed else "FAIL", p3.timing_s)

    # Phase 4: Cluster
    if not skip_cluster:
        p4 = phase_cluster(df, cfg)
        report.phases.append(p4)
        log.info("Phase 4 (Cluster): %s [%.1fs]", "PASS" if p4.passed else "FAIL", p4.timing_s)

    # Compute overall
    report.overall_passed = all(p.passed for p in report.phases)
    report.critical_passed = not any(p.gated for p in report.phases)

    # Phase 5: Report
    report.phases.append(phase_report(report, exp_dir))

    if not report.overall_passed:
        return 1
    return 0


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        prog="15m_test",
        description="15-Minute Smoke Test for NAT ingestor data",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the smoke test")
    run_p.add_argument("--data-dir", type=Path, default=None,
                       help="Path to parquet data directory (offline mode)")
    run_p.add_argument("--live", action="store_true",
                       help="Monitor for 15 min of fresh data from the ingestor")
    run_p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                       help="Report output directory")
    run_p.add_argument("--duration", type=int, default=None,
                       help="Trim window in seconds (default: 900 = 15 min)")
    run_p.add_argument("--skip-cluster", action="store_true",
                       help="Skip Phase 4 (clustering) for faster runs")
    run_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                       help="TOML config path")
    run_p.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    exit_code = run_smoke_test(
        cfg=cfg,
        data_dir=args.data_dir,
        live=args.live,
        output_dir=args.output,
        skip_cluster=args.skip_cluster,
        duration_s=args.duration,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
