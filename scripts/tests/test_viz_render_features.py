"""Planted tests for `nat viz render --features` (feature-scoped panel grid).

Contract (docs/requirements/parquet_viz_validation.md §FR-A4):
  - --features resolves a category / named vector / comma-list / 'all'.
  - all-NaN (unavailable) columns are dropped; the grid caps to top-N by
    variance (legibility).
  - feature mode produces a PNG in both overview and page modes; an
    out-of-range page index raises.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from viz.feature_select import select_features, cap_by_variance

# 15m_visualize.py has a numeric prefix → spec-based import.
_spec = importlib.util.spec_from_file_location(
    "viz15m_feat", str(Path(__file__).parent.parent / "15m_visualize.py")
)
viz15m = importlib.util.module_from_spec(_spec)
sys.modules["viz15m_feat"] = viz15m
_spec.loader.exec_module(viz15m)

NS = 1_000_000_000
START = 1_781_532_000_000_000_000


def _frame(seconds: int = 600, symbol: str = "BTC") -> pd.DataFrame:
    n = seconds * 10
    ts = START + np.arange(n, dtype=np.int64) * (NS // 10)
    rng = np.linspace(0, 1, n)
    return pd.DataFrame({
        "timestamp_ns": ts,
        "symbol": symbol,
        "sequence_id": np.arange(n, dtype=np.int64),
        "raw_midprice": 60_000 + 100 * np.sin(rng * 6.28),
        "raw_spread": 0.5 + 0.1 * rng,
        "flow_volume": np.abs(np.sin(rng * 3.14)) * 10,
        "vol_realized_1m": 0.2 + 0.05 * rng,
        "whale_net_flow_1h": np.nan,  # all-NaN → must be dropped by cap_by_variance
    })


# ── selector unit tests (no matplotlib) ──────────────────────────────────── #

def test_comma_list_selection():
    df = _frame()
    assert select_features(df, "raw_midprice,raw_spread") == ["raw_midprice", "raw_spread"]


def test_all_selects_every_numeric_feature():
    df = _frame()
    cols = select_features(df, "all")
    assert "raw_midprice" in cols and "flow_volume" in cols
    for meta in ("timestamp_ns", "symbol", "sequence_id"):
        assert meta not in cols


def test_unknown_selector_raises():
    df = _frame()
    with pytest.raises(ValueError):
        select_features(df, "does_not_exist_xyz")


def test_cap_by_variance_drops_all_nan_and_caps():
    df = _frame()
    cols = ["raw_midprice", "raw_spread", "flow_volume", "whale_net_flow_1h"]
    kept, capped = cap_by_variance(df, cols, max_features=10)
    assert "whale_net_flow_1h" not in kept   # all-NaN dropped
    assert capped is False
    kept2, capped2 = cap_by_variance(df, cols, max_features=2)
    assert len(kept2) == 2 and capped2 is True


# ── integration: feature-grid rendering produces a PNG ───────────────────── #

def test_overview_feature_grid_writes_png(tmp_path):
    df = _frame()
    outs = viz15m.generate_visualization(
        df, "BTC", tmp_path, timeframe="1min", features="raw_midprice,raw_spread")
    assert len(outs) == 1 and outs[0].exists() and outs[0].stat().st_size > 0


def test_page_feature_grid_and_out_of_range(tmp_path):
    df = _frame()
    outs = viz15m.generate_visualization(
        df, "BTC", tmp_path, timeframe="10s", window_minutes=5, window_index=1,
        features="flow_volume,raw_midprice")
    assert len(outs) == 1 and outs[0].exists()
    with pytest.raises(IndexError):
        viz15m.generate_visualization(
            df, "BTC", tmp_path, timeframe="10s", window_minutes=5, window_index=999,
            features="flow_volume")
