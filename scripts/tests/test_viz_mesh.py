"""Planted tests for the 3D feature-surface (Capability C).

Contract (docs/requirements/parquet_viz_validation.md §FR-C):
  - build_surface() returns a Plotly figure with one Surface trace whose z has
    shape (n_features, n_time) matching the returned meta.
  - All-NaN (unavailable) feature columns are dropped, not plotted.
  - max_features caps the y-axis by variance.
  - The same paginated INDEX applies; out-of-range raises IndexError.
  - render() writes a non-empty self-contained HTML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

import viz_mesh

NS = 1_000_000_000
START = 1_781_532_000_000_000_000  # arbitrary non-midnight t0


def _frame(seconds: int = 600, symbol: str = "BTC") -> pd.DataFrame:
    n = seconds * 10  # 100 ms cadence
    ts = START + np.arange(n, dtype=np.int64) * (NS // 10)
    rng = np.linspace(0, 1, n)
    return pd.DataFrame({
        "timestamp_ns": ts,
        "symbol": symbol,
        "sequence_id": np.arange(n, dtype=np.int64),
        "raw_midprice": 60_000 + 100 * np.sin(rng * 6.28),
        "raw_spread": 0.5 + 0.1 * rng,
        "imbalance_qty_l1": np.cos(rng * 6.28),
        "flow_volume": np.abs(np.sin(rng * 3.14)) * 10,
        "vol_realized_1m": 0.2 + 0.05 * rng,
        "ent_tick_1m": 0.8 + 0.1 * np.cos(rng * 12.0),
        "whale_net_flow_1h": np.nan,          # all-NaN → must be dropped
        "liquidation_risk_above_1pct": np.nan,  # all-NaN → must be dropped
    })


def test_overview_surface_dims_match_meta():
    df = _frame()
    fig, meta = viz_mesh.build_surface(df, "BTC", tf="1m", index=None)
    assert len(fig.data) == 1
    surf = fig.data[0]
    assert surf.type == "surface"
    assert np.asarray(surf.z).shape == (meta["n_features"], meta["n_time"])
    # 1-min overview of 600 s → ~10 time bars
    assert meta["n_time"] >= 9
    assert meta["index"] is None


def test_all_nan_features_dropped():
    df = _frame()
    fig, meta = viz_mesh.build_surface(df, "BTC", tf="1m", index=None)
    assert "whale_net_flow_1h" not in meta["features"]
    assert "liquidation_risk_above_1pct" not in meta["features"]
    assert meta["n_features"] == 6  # the six populated feature columns


def test_max_features_caps_yaxis():
    df = _frame()
    fig, meta = viz_mesh.build_surface(df, "BTC", tf="1m", index=None, max_features=3)
    assert meta["capped"] is True
    assert meta["n_features"] == 3
    assert np.asarray(fig.data[0].z).shape[0] == 3


def test_page_mode_and_out_of_range():
    df = _frame()
    fig, meta = viz_mesh.build_surface(df, "BTC", tf="5m", index=1)
    assert meta["index"] == 1
    assert meta["n_time"] >= 2
    with pytest.raises(IndexError):
        viz_mesh.build_surface(df, "BTC", tf="5m", index=999)


def test_feature_category_selection():
    df = _frame()
    fig, meta = viz_mesh.build_surface(df, "BTC", tf="1m", index=None,
                                       features="raw_midprice,raw_spread")
    assert set(meta["features"]) == {"raw_midprice", "raw_spread"}


def test_render_writes_self_contained_html(tmp_path):
    df = _frame()
    # Patch the loader so render() uses our synthetic frame.
    orig_load = viz_mesh.load_parquet
    orig_syms = viz_mesh.get_symbols
    viz_mesh.load_parquet = lambda *a, **k: df
    viz_mesh.get_symbols = lambda *a, **k: ["BTC"]
    try:
        out = viz_mesh.render("ignored", "BTC", tf="1m", index=None,
                              output=tmp_path, date="2026-06-15")
    finally:
        viz_mesh.load_parquet = orig_load
        viz_mesh.get_symbols = orig_syms
    assert out.exists()
    text = out.read_text()
    assert out.stat().st_size > 50_000        # plotly.js embedded → self-contained
    assert "Plotly.newPlot" in text or "plotly" in text.lower()
