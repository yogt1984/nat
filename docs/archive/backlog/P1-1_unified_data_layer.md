# P1-1: Unified Data Access Layer

**Priority**: Critical
**Effort**: 3-4 days
**Status**: Not started

---

## Problem

32 separate parquet-loading implementations exist across the codebase. Each reimplements file discovery, symbol filtering, column selection, and (in 7 cases) bar aggregation. Three different libraries are used interchangeably (`pyarrow.parquet`, `pandas.read_parquet`, `polars.read_parquet`). Only 1 of 32 loaders uses predicate pushdown for symbol filtering — the other 31 load all symbols then filter in memory.

One schema change (column rename, new feature) silently breaks N scripts. There is no single place to update.

---

## Current State: 32 Loaders Audited

### By loading library

| Library | Count | Files |
|---|---|---|
| `pyarrow.parquet.read_table` | 13 | paper_trader, mf_backtest, hypothesis_suite, signal_bridge, convolver_discovery, viz/loader, cluster_pipeline/loader, validate_data, skeptical_validation, experiment/metrics, signal_correlation, funding_carry, paper_trader_surprise |
| `pandas.read_parquet` | 11 | agent/runner, cascade_daemon, it_engine/daemon, it_multiday, it_multiday_ic, 15m_visualize, 15m_test, regime_labeler, direct_validation, data/macro |
| `polars.read_parquet` | 8 | backtest/data_loader, ml_strategy, train_baseline, analyze_clusters, explore_clusters, train_regime_gmm, score_data, eamm/cli |

### By feature used

| Feature | Count | Notes |
|---|---|---|
| Symbol filtering (post-load) | 26 | `df[df["symbol"] == sym]` — loads all symbols, filters after |
| Symbol filtering (pushdown) | 1 | Only `cluster_pipeline/loader.py` uses PyArrow filters |
| Column filtering | 12 | Mostly hardcoded subsets; only 2 are dynamic |
| Date range filtering | 7 | Inconsistent: some parse directory names, some use timestamp_ns |
| 5-min bar aggregation | 7 | All 7 reimplement the same floor-divide-group-agg logic |
| Schema validation | 2 | Only cluster_pipeline/loader and experiment/metrics |

### The 7 duplicated bar aggregation implementations

All follow the identical pattern but are copy-pasted:

| File | Function | Lines |
|---|---|---|
| `scripts/alpha/paper_trader.py` | `aggregate_to_bars()` | 125-143 |
| `scripts/alpha/paper_trader_generic.py` | `aggregate_to_bars()` | 206-232 |
| `scripts/alpha/paper_trader_surprise.py` | `aggregate_to_bars()` | ~120-140 |
| `scripts/execution/signal_bridge.py` | `aggregate_to_bars()` | ~142+ |
| `scripts/analysis/mf_liquidity_backtest.py` | `aggregate_to_bars()` | 77-104 |
| `scripts/analysis/mf_hypothesis_suite.py` | `aggregate_to_bars()` | 89-122 |
| `scripts/analysis/signal_correlation.py` | `make_bars_3f()` / `make_bars_algo()` | varies |

Logic: `bar_id = timestamp_ns // (bar_seconds * 1e9)`, group by bar_id, agg: midprice=last, spread=last, depth=std, vwap_dev=std, filter bars with < 10 ticks.

---

## Design

### New module: `scripts/data/features.py`

Single entry point for all parquet access. Internally uses PyArrow for predicate pushdown (symbol, timestamp range) and returns pandas DataFrames (the common denominator — 24 of 32 callers use pandas downstream).

```python
"""Unified feature data loader for NAT parquet files.

Every script that reads from data/features/ should use this module
instead of ad-hoc pyarrow/pandas/polars loading.
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features"


def load_features(
    symbols: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,  # ("2026-05-20", "2026-05-25")
    columns: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load tick-level features from parquet files.

    Uses PyArrow predicate pushdown for symbol filtering (avoids loading
    all symbols into memory then filtering). Date range filtering is done
    at directory level (skip directories outside range) and row level
    (timestamp_ns filter pushed down to row groups).

    Args:
        symbols: Filter to these symbols. None = all symbols.
        date_range: Inclusive date range as (start, end) ISO strings.
                    None = all available dates.
        columns: Columns to load. None = all columns.
                 "symbol" and "timestamp_ns" are always included.
        data_dir: Override default data/features/ directory.

    Returns:
        DataFrame sorted by timestamp_ns with requested columns.
    """
    ...


def load_bars(
    symbols: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
    columns: Optional[list[str]] = None,
    bar_seconds: int = 300,
    min_ticks: int = 10,
    agg_spec: Optional[dict[str, str]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load features aggregated to fixed-interval bars.

    Calls load_features() then aggregates. Default aggregation:
        - timestamp_ns: first
        - raw_midprice: last
        - raw_spread_bps: last
        - raw_ask_depth_5: std
        - flow_vwap_deviation: std
        - all other numeric columns: last

    Custom aggregation via agg_spec: {"col_name": "mean"|"std"|"last"|"first"|"sum"}

    Args:
        bar_seconds: Bar width in seconds. Default 300 (5 min).
        min_ticks: Minimum ticks per bar. Bars below this are dropped.
        agg_spec: Column-specific aggregation overrides.
    """
    ...


def available_dates(data_dir: Optional[Path] = None) -> list[str]:
    """Return sorted list of YYYY-MM-DD date strings with parquet data."""
    ...


def available_symbols(
    date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> list[str]:
    """Return sorted list of symbols present in data.

    If date is specified, only check that date's directory.
    """
    ...


def data_health(data_dir: Optional[Path] = None) -> dict:
    """Quick health check without loading full data.

    Returns:
        {
            "dates": ["2026-05-20", ...],
            "symbols": ["BTC", "ETH", "SOL"],
            "total_files": 42,
            "total_rows": 1234567,  # from parquet metadata (fast)
            "latest_timestamp": "2026-05-25T14:32:00",
            "freshness_seconds": 300,
            "warnings": ["ETH has 0 files for 2026-05-23"]
        }
    """
    ...
```

### New module: `scripts/data/schema.py`

```python
"""Feature schema validation.

Catches column renames, missing features, and data quality issues
at load time instead of at runtime NaN surprises.
"""

# Expected columns derived from Rust FeatureComputer::names_all()
BASE_COLUMNS = ["timestamp_ns", "symbol", "sequence_id"]

BASE_FEATURES = {
    "raw": ["raw_midprice", "raw_spread_bps", ...],       # 10
    "imbalance": ["imb_qty_l1", ...],                      # 8
    "flow": ["flow_trade_count_1s", ...],                  # 12
    ...
}

OPTIONAL_FEATURES = {
    "whale_flow": ["whale_net_flow_1h", ...],              # 12
    ...
}


def validate_columns(df_columns: list[str]) -> dict:
    """Check loaded columns against expected schema.

    Returns:
        {
            "missing_base": [...],     # columns expected but absent
            "missing_optional": [...], # optional columns absent (informational)
            "unexpected": [...],       # columns present but not in schema
            "valid": True/False
        }
    """
    ...


def validate_quality(df: pd.DataFrame) -> dict:
    """Data quality checks on a loaded DataFrame.

    Returns:
        {
            "nan_rates": {"col": 0.05, ...},  # per-column NaN rate
            "high_nan_cols": [...],            # > 50% NaN
            "constant_cols": [...],            # zero variance
            "row_count": 12345,
            "symbol_counts": {"BTC": 4000, "ETH": 4100, "SOL": 4245}
        }
    """
    ...
```

---

## Implementation Tasks

### Task 1: Create `scripts/data/__init__.py` and `scripts/data/features.py`

- [ ] Implement `load_features()` with PyArrow predicate pushdown for symbol and timestamp
- [ ] Implement date-range filtering at directory level (parse `YYYY-MM-DD` from dir names, skip dirs outside range)
- [ ] Ensure `columns` parameter always includes `timestamp_ns` and `symbol` (callers expect them)
- [ ] Handle missing files gracefully (empty DataFrame with correct schema, not crash)
- [ ] Implement `load_bars()` calling `load_features()` then aggregating
- [ ] Default agg spec: midprice=last, spread=last, depth=std, vwap_dev=std, others=last
- [ ] `min_ticks` filter: drop bars with fewer ticks
- [ ] Implement `available_dates()`, `available_symbols()`, `data_health()`
- [ ] Unit test with synthetic parquet fixture (3 dates, 2 symbols, 10 columns)

**Note**: `scripts/data/` directory already exists (contains `agent/` and `macro.py`). The new `features.py` sits alongside them. Ensure `__init__.py` doesn't break existing imports.

### Task 2: Create `scripts/data/schema.py`

- [ ] Extract expected column names from `rust/ing/src/features/mod.rs` `names_all()` (or from a sample parquet file)
- [ ] Implement `validate_columns()` — returns missing/unexpected columns
- [ ] Implement `validate_quality()` — NaN rates, constant columns, row counts
- [ ] Wire validation into `load_features()` (optional, `validate=True` by default on first load of a session)
- [ ] Unit test: load a parquet file, rename a column, verify validation catches it

### Task 3: Migrate the 7 bar-aggregation duplicates

These are the highest-value migrations because they eliminate the most duplicated logic.

- [ ] `scripts/alpha/paper_trader.py` — replace `load_date()` + `aggregate_to_bars()` with `load_bars(symbols=[symbol], date_range=(d, d), bar_seconds=300)`
- [ ] `scripts/alpha/paper_trader_generic.py` — replace `load_date_ticks()` + `aggregate_to_bars()` with `load_bars()` using custom `agg_spec`
- [ ] `scripts/alpha/paper_trader_surprise.py` — replace `load_date()` + `aggregate_to_bars()` with `load_bars()`
- [ ] `scripts/execution/signal_bridge.py` — replace `load_date()` + `aggregate_to_bars()` with `load_bars()`
- [ ] `scripts/analysis/mf_liquidity_backtest.py` — replace `load_date()` + `aggregate_to_bars()` with `load_bars()`
- [ ] `scripts/analysis/mf_hypothesis_suite.py` — replace `load_date()` + `aggregate_to_bars()` with `load_bars()`
- [ ] `scripts/analysis/signal_correlation.py` — replace `load_date_ticks()` + `make_bars_*()` with `load_bars()` using custom `agg_spec`

**Verification per file**: Run existing tests. For files without tests, verify output DataFrame shape and column dtypes match before/after migration on a real parquet date.

### Task 4: Migrate agent/runner loaders

- [ ] `scripts/agent/runner.py` `_load_feature_data()` — replace `pd.read_parquet` + post-filter with `load_features(symbols=[sym])`
- [ ] `scripts/agent/cascade_daemon.py` — replace rglob + last-3-dates logic with `load_features(date_range=...)`
- [ ] `scripts/it_engine/daemon.py` — replace rglob + post-filter with `load_features(symbols=[sym])`

### Task 5: Migrate analysis/research loaders

- [ ] `scripts/analysis/funding_carry.py` — replace `glob.glob` + full load with `load_features(symbols=[sym])`
- [ ] `scripts/analysis/convolver_discovery.py` — replace embedded glob with `load_features(symbols=[sym], columns=[...])`
- [ ] `scripts/viz/loader.py` — replace custom `load_data()` with `load_features()` (keep hours parameter as date_range conversion)
- [ ] `scripts/analysis/it_multiday.py`, `it_multiday_ic.py` — replace passed-in file list with `load_features(date_range=...)`

### Task 6: Migrate cluster/training loaders (lower priority)

These use Polars. Two options: (a) keep Polars callers converting from the pandas output, or (b) add a `load_features_polars()` variant. Option (a) is simpler and acceptable unless profiling shows conversion is a bottleneck.

- [ ] `scripts/cluster_pipeline/loader.py` — keep as-is initially (it's the most comprehensive loader, the new module borrows from it). Deprecate the direct import path, re-export via `scripts/data/features.py`
- [ ] `scripts/backtest/data_loader.py` — migrate or wrap
- [ ] `scripts/train_baseline.py` — migrate or wrap
- [ ] `scripts/analyze_clusters.py` — migrate or wrap
- [ ] `scripts/explore_clusters.py` — migrate or wrap
- [ ] `scripts/train_regime_gmm.py` — migrate or wrap
- [ ] `scripts/score_data.py` — migrate or wrap

### Task 7: Migrate remaining one-off loaders (lowest priority)

- [ ] `scripts/15m_visualize.py`, `scripts/15m_test.py`
- [ ] `scripts/validate_data.py` (uses pq for metadata — keep separate or add metadata mode to unified loader)
- [ ] `scripts/skeptical_validation.py`
- [ ] `scripts/experiment/metrics.py` (metadata-only path — add to `data_health()`)
- [ ] `scripts/eamm/cli.py`
- [ ] `exploration/validation/regime_labeler.py`, `direct_validation.py`

---

## Verification

### Per-migration verification

For each migrated file:
1. Run existing unit tests if they exist (`pytest scripts/tests/test_<module>.py`)
2. If no tests: load one real parquet date with both old and new code, assert `df.shape`, `df.dtypes`, and `df.head(5)` match
3. For bar-aggregation migrations: assert bar count, column set, and first/last bar values match

### Integration verification

After all Task 3 migrations (bar aggregation):
```bash
# Paper trader batch should produce identical trade logs
python scripts/alpha/paper_trader.py batch --save
diff data/paper_trades/batch_report.json data/paper_trades/batch_report_before.json

# Backtest should produce identical results
python scripts/analysis/mf_liquidity_backtest.py --symbol BTC
```

### Regression test

- [ ] Create `scripts/tests/test_data_features.py` with:
  - Synthetic parquet fixture (3 dates, 2 symbols, known values)
  - Test `load_features()` with all parameter combinations
  - Test `load_bars()` produces correct bar count and aggregation
  - Test schema validation catches missing/renamed columns
  - Test `available_dates()`, `available_symbols()`, `data_health()`
  - Test edge cases: empty date directory, single file, no matching symbol

---

## Migration Order

```
Task 1 (core loader)           ← build and test in isolation
    ↓
Task 2 (schema validation)     ← can parallel with Task 1
    ↓
Task 3 (7 bar-agg duplicates)  ← highest dedup value, migrate one-by-one
    ↓
Task 4 (agent runners)         ← next highest usage frequency
    ↓
Task 5 (analysis scripts)      ← medium frequency
    ↓
Task 6 (Polars callers)        ← low priority, wrap if needed
    ↓
Task 7 (one-offs)              ← lowest priority, migrate opportunistically
```

Tasks 1-3 are the critical path. Tasks 4-7 can be done incrementally over time. The old loading code doesn't need to be deleted immediately — just deprecated with a comment pointing to the new module.

---

## What This Does NOT Cover

- **Parquet writing** (ParquetWriter in Rust) — out of scope, that's well-contained in `rust/ing/src/output/writer.rs`
- **Non-feature data** (paper trade logs, state files, model files) — separate concern
- **Real-time streaming** (Redis pub/sub, WebSocket) — different data path entirely
- **Polars-native API** — deferred; pandas output with `.to_polars()` conversion is acceptable for now
