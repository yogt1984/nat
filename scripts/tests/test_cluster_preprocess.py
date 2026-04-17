"""
Skeptical tests for cluster_pipeline.preprocess — bar aggregation & preprocessing.

Tests cover:
  - Bar aggregation correctness (tick counts, OHLC, sums, means, slopes)
  - Multi-symbol handling
  - Multi-timeframe aggregation
  - NaN propagation and handling
  - Scaling (zscore, minmax, robust, none)
  - Vector matching on aggregated columns
  - Edge cases (single tick, single bar, empty bars, constant data)
  - Determinism and idempotency
  - Time alignment correctness
  - Summary and inspection utilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cluster_pipeline.config import FEATURE_VECTORS, META_COLUMNS, get_vector_columns
from cluster_pipeline.preprocess import (
    TIMEFRAMES,
    aggregate_bars,
    aggregate_multi_timeframe,
    bar_summary,
    list_bar_columns,
    preprocess,
    _build_agg_plan,
    _linear_slope,
    _match_vector_columns,
    _resolve_freq,
    _PRICE_COLUMNS,
    _SUM_COLUMNS,
    _ENTROPY_PREFIX,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic tick data
# ---------------------------------------------------------------------------


def _make_tick_df(
    n_ticks: int = 3000,
    symbols: list[str] | None = None,
    start_ns: int = 1_700_000_000_000_000_000,
    interval_ns: int = 100_000_000,  # 100ms
    seed: int = 42,
    include_entropy: bool = True,
    include_price: bool = True,
    include_flow: bool = True,
    include_whale: bool = True,
    include_trend: bool = True,
) -> pd.DataFrame:
    """Build a synthetic tick DataFrame mimicking the Rust ingestor output."""
    rng = np.random.default_rng(seed)

    if symbols is None:
        symbols = ["BTC"]

    frames = []
    for sym in symbols:
        ts = np.arange(start_ns, start_ns + n_ticks * interval_ns, interval_ns, dtype=np.int64)
        data = {"timestamp_ns": ts, "symbol": sym}

        if include_price:
            base_price = 50000.0 if sym == "BTC" else 3000.0
            data["raw_midprice"] = base_price + rng.normal(0, 10, n_ticks).cumsum()
            data["raw_microprice"] = data["raw_midprice"] + rng.normal(0, 0.5, n_ticks)
            data["raw_spread"] = np.abs(rng.normal(0.5, 0.1, n_ticks))
            data["raw_spread_bps"] = data["raw_spread"] / data["raw_midprice"] * 10000

        if include_entropy:
            for col in ["ent_tick_1s", "ent_tick_5s", "ent_tick_1m", "ent_permutation_returns_8",
                        "ent_zscore_1m", "ent_rate_of_change_5s"]:
                data[col] = rng.uniform(0, 1, n_ticks)

        if include_flow:
            data["flow_volume_1s"] = rng.exponential(100, n_ticks)
            data["flow_volume_5s"] = rng.exponential(500, n_ticks)
            data["flow_count_1s"] = rng.poisson(5, n_ticks).astype(float)
            data["flow_count_5s"] = rng.poisson(25, n_ticks).astype(float)
            data["flow_intensity"] = rng.uniform(0, 1, n_ticks)

        if include_whale:
            data["whale_net_flow_1h"] = rng.normal(0, 1000, n_ticks)
            data["whale_total_activity"] = rng.exponential(50, n_ticks)
            data["active_whale_count"] = rng.poisson(3, n_ticks).astype(float)
            data["whale_flow_intensity"] = rng.uniform(0, 1, n_ticks)

        if include_trend:
            data["trend_momentum_60"] = rng.normal(0, 0.01, n_ticks)
            data["trend_hurst_300"] = rng.uniform(0.3, 0.7, n_ticks)

        frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def tick_df():
    """Standard single-symbol tick data: 3000 ticks (~5 minutes at 100ms)."""
    return _make_tick_df(n_ticks=3000)


@pytest.fixture
def tick_df_multi():
    """Multi-symbol tick data: BTC + ETH, 6000 ticks each (~10 min)."""
    return _make_tick_df(n_ticks=6000, symbols=["BTC", "ETH"])


@pytest.fixture
def tick_df_long():
    """Long tick data: 36000 ticks (~1 hour) for testing 1h bars."""
    return _make_tick_df(n_ticks=36000)


@pytest.fixture
def tick_df_very_long():
    """Very long tick data: 180000 ticks (~5 hours) for testing 4h bars."""
    return _make_tick_df(n_ticks=180000)


# ---------------------------------------------------------------------------
# TestAggregateBasic — core aggregation behavior
# ---------------------------------------------------------------------------


class TestAggregateBasic:
    """Verify that aggregate_bars produces correct structure and values."""

    def test_returns_dataframe(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert isinstance(bars, pd.DataFrame)

    def test_has_meta_columns(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "bar_start" in bars.columns
        assert "bar_end" in bars.columns
        assert "symbol" in bars.columns
        assert "tick_count" in bars.columns

    def test_fewer_rows_than_ticks(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert len(bars) < len(tick_df)

    def test_tick_count_sums_to_total(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        # Total ticks should equal or be close to input (some may fall in partial bars)
        assert bars["tick_count"].sum() == len(tick_df)

    def test_tick_count_positive(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert (bars["tick_count"] > 0).all()

    def test_bar_start_monotonic(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert bars["bar_start"].is_monotonic_increasing

    def test_bar_end_after_bar_start(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert (bars["bar_end"] > bars["bar_start"]).all()

    def test_symbol_preserved(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert set(bars["symbol"].unique()) == {"BTC"}

    def test_all_timeframes_valid(self, tick_df):
        for tf in TIMEFRAMES:
            bars = aggregate_bars(tick_df, timeframe=tf)
            assert len(bars) >= 1, f"No bars for timeframe {tf}"

    def test_shorter_timeframe_more_bars(self, tick_df_long):
        bars_5m = aggregate_bars(tick_df_long, timeframe="5min")
        bars_15m = aggregate_bars(tick_df_long, timeframe="15min")
        assert len(bars_5m) > len(bars_15m)

    def test_custom_freq_string(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="2min")
        assert len(bars) >= 1

    def test_empty_df_raises(self):
        empty = pd.DataFrame({"timestamp_ns": [], "symbol": []})
        with pytest.raises(ValueError, match="empty"):
            aggregate_bars(empty, timeframe="5min")

    def test_missing_timestamp_raises(self):
        df = pd.DataFrame({"symbol": ["BTC"], "value": [1.0]})
        with pytest.raises(ValueError, match="timestamp"):
            aggregate_bars(df, timeframe="5min")


# ---------------------------------------------------------------------------
# TestAggregationRules — domain-specific aggregation correctness
# ---------------------------------------------------------------------------


class TestAggregationRules:
    """Verify category-specific aggregation rules from the spec."""

    def test_price_columns_ohlc(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        for price_col in ["raw_midprice", "raw_microprice"]:
            assert f"{price_col}_open" in bars.columns
            assert f"{price_col}_high" in bars.columns
            assert f"{price_col}_low" in bars.columns
            assert f"{price_col}_close" in bars.columns
            assert f"{price_col}_mean" in bars.columns

    def test_price_ohlc_values_correct(self, tick_df):
        """Verify OHLC values match manual computation for first bar."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        # Get ticks in first bar window
        dt = pd.to_datetime(tick_df["timestamp_ns"], unit="ns")
        bar_start = bars["bar_start"].iloc[0]
        bar_end = bars["bar_end"].iloc[0]
        mask = (dt >= bar_start) & (dt < bar_end)
        ticks_in_bar = tick_df.loc[mask, "raw_midprice"]

        assert bars["raw_midprice_open"].iloc[0] == pytest.approx(ticks_in_bar.iloc[0])
        assert bars["raw_midprice_close"].iloc[0] == pytest.approx(ticks_in_bar.iloc[-1])
        assert bars["raw_midprice_high"].iloc[0] == pytest.approx(ticks_in_bar.max())
        assert bars["raw_midprice_low"].iloc[0] == pytest.approx(ticks_in_bar.min())

    def test_entropy_has_mean_std_slope(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        for col in ["ent_tick_1s", "ent_tick_5s", "ent_tick_1m"]:
            assert f"{col}_mean" in bars.columns, f"Missing {col}_mean"
            assert f"{col}_std" in bars.columns, f"Missing {col}_std"
            assert f"{col}_slope" in bars.columns, f"Missing {col}_slope"

    def test_entropy_no_last_suffix(self, tick_df):
        """Entropy columns should NOT have _last suffix (spec says mean+slope only)."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        for col in bars.columns:
            if col.startswith("ent_") and col.endswith("_last"):
                pytest.fail(f"Entropy column {col} should not have _last suffix")

    def test_flow_volume_summed(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "flow_volume_1s_sum" in bars.columns
        assert "flow_volume_5s_sum" in bars.columns

    def test_flow_volume_sum_correct(self, tick_df):
        """Verify sum aggregation matches manual computation."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        dt = pd.to_datetime(tick_df["timestamp_ns"], unit="ns")
        bar_start = bars["bar_start"].iloc[0]
        bar_end = bars["bar_end"].iloc[0]
        mask = (dt >= bar_start) & (dt < bar_end)
        expected = tick_df.loc[mask, "flow_volume_1s"].sum()
        assert bars["flow_volume_1s_sum"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_flow_count_summed(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "flow_count_1s_sum" in bars.columns

    def test_whale_flow_summed(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "whale_net_flow_1h_sum" in bars.columns
        assert "whale_total_activity_sum" in bars.columns

    def test_default_features_have_mean_std_last(self, tick_df):
        """Non-special features (trend, etc.) get mean/std/last."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "trend_momentum_60_mean" in bars.columns
        assert "trend_momentum_60_std" in bars.columns
        assert "trend_momentum_60_last" in bars.columns

    def test_spread_treated_as_price(self, tick_df):
        """raw_spread is in _PRICE_COLUMNS, should get OHLC."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        assert "raw_spread_open" in bars.columns
        assert "raw_spread_high" in bars.columns

    def test_custom_agg_override(self, tick_df):
        """custom_aggs should override default rules."""
        bars = aggregate_bars(
            tick_df,
            timeframe="5min",
            custom_aggs={"trend_momentum_60": "max"},
        )
        assert "trend_momentum_60_custom" in bars.columns


# ---------------------------------------------------------------------------
# TestLinearSlope — OLS slope computation
# ---------------------------------------------------------------------------


class TestLinearSlope:
    """Verify the _linear_slope helper used for entropy bar slopes."""

    def test_constant_series_zero_slope(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        assert _linear_slope(s) == pytest.approx(0.0)

    def test_perfect_uptrend(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _linear_slope(s) == pytest.approx(1.0)

    def test_perfect_downtrend(self):
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _linear_slope(s) == pytest.approx(-1.0)

    def test_single_value_returns_zero(self):
        s = pd.Series([42.0])
        assert _linear_slope(s) == pytest.approx(0.0)

    def test_empty_series_returns_zero(self):
        s = pd.Series([], dtype=float)
        assert _linear_slope(s) == pytest.approx(0.0)

    def test_nan_values_ignored(self):
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        # After dropping NaN: [1, 3, 5] with x=[0,1,2] -> slope=2
        assert _linear_slope(s) == pytest.approx(2.0)

    def test_all_nan_returns_zero(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        assert _linear_slope(s) == pytest.approx(0.0)

    def test_two_points(self):
        s = pd.Series([10.0, 20.0])
        assert _linear_slope(s) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestMultiSymbol — per-symbol bar isolation
# ---------------------------------------------------------------------------


class TestMultiSymbol:
    """Verify symbols are aggregated independently."""

    def test_both_symbols_present(self, tick_df_multi):
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        assert set(bars["symbol"].unique()) == {"BTC", "ETH"}

    def test_symbol_bars_independent(self, tick_df_multi):
        """Each symbol should have its own bar sequence."""
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        btc_bars = bars[bars["symbol"] == "BTC"]
        eth_bars = bars[bars["symbol"] == "ETH"]
        # Same timeframe -> same number of bars
        assert len(btc_bars) == len(eth_bars)

    def test_tick_counts_per_symbol(self, tick_df_multi):
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        btc_total = bars[bars["symbol"] == "BTC"]["tick_count"].sum()
        eth_total = bars[bars["symbol"] == "ETH"]["tick_count"].sum()
        # Each symbol has 6000 ticks
        assert btc_total == 6000
        assert eth_total == 6000

    def test_no_cross_symbol_contamination(self, tick_df_multi):
        """BTC prices should be ~50k, ETH ~3k — they shouldn't mix."""
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        btc_mean = bars[bars["symbol"] == "BTC"]["raw_midprice_mean"].mean()
        eth_mean = bars[bars["symbol"] == "ETH"]["raw_midprice_mean"].mean()
        assert btc_mean > 10000, f"BTC mean {btc_mean} too low"
        assert eth_mean < 10000, f"ETH mean {eth_mean} too high"

    def test_no_symbol_column_still_works(self):
        """DataFrame without symbol column should still aggregate."""
        df = _make_tick_df(n_ticks=3000)
        df = df.drop(columns=["symbol"])
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) >= 1
        assert "symbol" in bars.columns  # should get "UNKNOWN"


# ---------------------------------------------------------------------------
# TestMultiTimeframe — aggregate_multi_timeframe
# ---------------------------------------------------------------------------


class TestMultiTimeframe:
    """Verify multi-timeframe aggregation."""

    def test_returns_all_timeframes(self, tick_df_long):
        result = aggregate_multi_timeframe(tick_df_long)
        assert set(result.keys()) == set(TIMEFRAMES.keys())

    def test_custom_timeframe_list(self, tick_df_long):
        result = aggregate_multi_timeframe(tick_df_long, timeframes=["5min", "1h"])
        assert set(result.keys()) == {"5min", "1h"}

    def test_coarser_has_fewer_bars(self, tick_df_long):
        result = aggregate_multi_timeframe(tick_df_long, timeframes=["5min", "15min"])
        assert len(result["5min"]) > len(result["15min"])

    def test_each_value_is_dataframe(self, tick_df_long):
        result = aggregate_multi_timeframe(tick_df_long)
        for tf, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{tf} not a DataFrame"


# ---------------------------------------------------------------------------
# TestPreprocess — NaN handling, scaling, feature selection
# ---------------------------------------------------------------------------


class TestPreprocess:
    """Verify the preprocess function."""

    def test_returns_tuple_of_three(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, meta = preprocess(bars)
        assert isinstance(X, np.ndarray)
        assert isinstance(cols, list)
        assert isinstance(meta, pd.DataFrame)

    def test_output_shape_matches_columns(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, meta = preprocess(bars)
        assert X.shape[1] == len(cols)
        assert X.shape[0] == len(meta)

    def test_no_nan_in_output(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars)
        assert not np.any(np.isnan(X))

    def test_no_inf_in_output(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars)
        assert not np.any(np.isinf(X))

    def test_zscore_properties(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars, scaler="zscore")
        # Each column should have mean ~0, std ~1
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)
        # Std should be close to 1 (with ddof=0)
        for s in stds:
            assert s == pytest.approx(1.0, abs=0.1) or s == pytest.approx(0.0, abs=1e-10)

    def test_minmax_range(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars, scaler="minmax")
        assert X.min() >= -1e-10
        assert X.max() <= 1.0 + 1e-10

    def test_robust_scaler(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars, scaler="robust")
        # Median of each column should be ~0
        medians = np.median(X, axis=0)
        np.testing.assert_allclose(medians, 0, atol=0.5)

    def test_no_scaler(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X_none, _, _ = preprocess(bars, scaler="none")
        # Raw values — no guarantee on range
        assert X_none.shape[0] > 0

    def test_invalid_scaler_raises(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        with pytest.raises(ValueError, match="Unknown scaler"):
            preprocess(bars, scaler="invalid")

    def test_nan_threshold_drops_columns(self):
        """Columns with too many NaNs should be dropped."""
        df = _make_tick_df(n_ticks=3000)
        # Inject NaN into one column
        df.loc[:, "ent_tick_1s"] = np.nan
        bars = aggregate_bars(df, timeframe="5min")
        X, cols, _ = preprocess(bars, nan_threshold=0.5)
        # Columns derived from ent_tick_1s should be dropped (they're all NaN)
        assert not any("ent_tick_1s" in c for c in cols), \
            "NaN-heavy columns should be dropped"

    def test_zero_variance_dropped(self):
        """Constant columns should be dropped."""
        df = _make_tick_df(n_ticks=3000)
        df["trend_momentum_60"] = 0.0  # constant
        bars = aggregate_bars(df, timeframe="5min")
        X, cols, _ = preprocess(bars)
        # trend_momentum_60_std is also zero (std of constant = 0)
        # At least the _std column should be dropped
        zero_var_cols = [c for c in cols if "trend_momentum_60" in c]
        for c in zero_var_cols:
            assert bars[c].std() > 0 or c not in cols

    def test_clip_sigma(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, _ = preprocess(bars, scaler="none", clip_sigma=2.0)
        # After clipping at 2 sigma, all values within range
        for j in range(X.shape[1]):
            col = X[:, j]
            mu = col.mean()
            sigma = col.std()
            if sigma > 1e-10:
                assert col.max() <= mu + 2.01 * sigma
                assert col.min() >= mu - 2.01 * sigma

    def test_meta_columns_correct(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        _, _, meta = preprocess(bars)
        assert "bar_start" in meta.columns
        assert "symbol" in meta.columns


# ---------------------------------------------------------------------------
# TestVectorMatching — preprocess with vector= parameter
# ---------------------------------------------------------------------------


class TestVectorMatching:
    """Verify vector matching on aggregated bar columns."""

    def test_entropy_vector_matches(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, _ = preprocess(bars, vector="entropy")
        # Should have entropy-related columns
        assert all("ent_" in c for c in cols)

    def test_trend_vector_matches(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, _ = preprocess(bars, vector="trend")
        assert all("trend_" in c for c in cols)

    def test_unknown_vector_raises(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        with pytest.raises(ValueError):
            preprocess(bars, vector="nonexistent_vector")

    def test_explicit_columns_override(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        target = [c for c in bars.columns if "ent_tick_1s" in c]
        X, cols, _ = preprocess(bars, columns=target)
        assert set(cols).issubset(set(target))

    def test_match_vector_columns_helper(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        matched = _match_vector_columns("entropy", bars.columns.tolist())
        assert len(matched) > 0
        assert all("ent_" in c for c in matched)


# ---------------------------------------------------------------------------
# TestNaNHandling — tricky NaN scenarios
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """Verify NaN handling edge cases."""

    def test_sparse_data_still_aggregates(self):
        """Ticks with lots of NaN features should still produce bars."""
        df = _make_tick_df(n_ticks=3000)
        rng = np.random.default_rng(99)
        # Set 30% of entropy values to NaN
        for col in ["ent_tick_1s", "ent_tick_5s", "ent_tick_1m"]:
            mask = rng.random(3000) < 0.3
            df.loc[mask, col] = np.nan
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) > 0

    def test_all_nan_column_becomes_nan_bar(self):
        """If a column is entirely NaN, the bar column should be NaN."""
        df = _make_tick_df(n_ticks=3000)
        df["ent_tick_1s"] = np.nan
        bars = aggregate_bars(df, timeframe="5min")
        assert bars["ent_tick_1s_mean"].isna().all()

    def test_preprocess_handles_nan_columns_gracefully(self):
        """Preprocess should drop NaN columns and still produce valid output."""
        df = _make_tick_df(n_ticks=3000)
        df["ent_tick_1s"] = np.nan
        df["ent_tick_5s"] = np.nan
        bars = aggregate_bars(df, timeframe="5min")
        X, cols, _ = preprocess(bars, nan_threshold=0.5)
        assert not np.any(np.isnan(X))
        assert len(cols) > 0

    def test_single_non_nan_value_in_bar(self):
        """Bar with only 1 valid tick should still produce values."""
        df = _make_tick_df(n_ticks=3000)
        # Set almost all ent_tick_1s to NaN except first tick
        df.loc[1:, "ent_tick_1s"] = np.nan
        bars = aggregate_bars(df, timeframe="5min")
        # First bar should have a non-NaN mean (the single value)
        assert not pd.isna(bars["ent_tick_1s_mean"].iloc[0])


# ---------------------------------------------------------------------------
# TestTimeAlignment — bar boundary correctness
# ---------------------------------------------------------------------------


class TestTimeAlignment:
    """Verify bar timestamps align to expected boundaries."""

    def test_5min_bars_align_to_5min(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        for ts in bars["bar_start"]:
            assert ts.minute % 5 == 0
            assert ts.second == 0

    def test_15min_bars_align_to_15min(self, tick_df_long):
        bars = aggregate_bars(tick_df_long, timeframe="15min")
        for ts in bars["bar_start"]:
            assert ts.minute % 15 == 0
            assert ts.second == 0

    def test_1h_bars_align_to_hour(self, tick_df_long):
        bars = aggregate_bars(tick_df_long, timeframe="1h")
        for ts in bars["bar_start"]:
            assert ts.minute == 0
            assert ts.second == 0

    def test_bar_end_equals_next_bar_start(self, tick_df_long):
        """Consecutive bars should be contiguous."""
        bars = aggregate_bars(tick_df_long, timeframe="15min")
        for i in range(len(bars) - 1):
            assert bars["bar_end"].iloc[i] == bars["bar_start"].iloc[i + 1]

    def test_bar_width_correct(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        for _, row in bars.iterrows():
            delta = row["bar_end"] - row["bar_start"]
            assert delta == pd.Timedelta(minutes=5)


# ---------------------------------------------------------------------------
# TestEdgeCases — boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_single_tick(self):
        """Single tick should produce one bar."""
        df = _make_tick_df(n_ticks=1)
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) == 1
        assert bars["tick_count"].iloc[0] == 1

    def test_two_ticks(self):
        df = _make_tick_df(n_ticks=2)
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) >= 1

    def test_very_few_bars_still_preprocessable(self):
        """Even 2 bars should be preprocessable."""
        df = _make_tick_df(n_ticks=6000)  # ~10 min -> 2 bars at 5min
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) >= 2
        X, cols, _ = preprocess(bars)
        assert X.shape[0] >= 2

    def test_constant_feature_across_all_ticks(self):
        """Constant feature should be dropped by preprocess (zero variance)."""
        df = _make_tick_df(n_ticks=3000)
        df["trend_momentum_60"] = 42.0
        df["trend_hurst_300"] = 42.0
        bars = aggregate_bars(df, timeframe="5min")
        X, cols, _ = preprocess(bars)
        # The _mean columns will be constant -> dropped
        # _std columns will be zero -> also constant -> dropped
        for c in cols:
            if "trend_momentum_60" in c or "trend_hurst_300" in c:
                # If it survived, it must have non-zero variance
                col_idx = cols.index(c)
                assert np.std(X[:, col_idx]) > 0

    def test_large_timestamp_values(self):
        """Future timestamps (year 2030+) should work fine."""
        future_ns = 1_900_000_000_000_000_000  # ~2030
        df = _make_tick_df(n_ticks=3000, start_ns=future_ns)
        bars = aggregate_bars(df, timeframe="5min")
        assert len(bars) >= 1

    def test_no_feature_columns_raises(self):
        """DataFrame with only meta columns should raise on preprocess."""
        df = pd.DataFrame({
            "timestamp_ns": np.arange(10) * 100_000_000 + 1_700_000_000_000_000_000,
            "symbol": "BTC",
        })
        # aggregate_bars needs at least some features to count ticks
        # Let's add one constant column
        df["val"] = 0.0
        bars = aggregate_bars(df, timeframe="5min")
        with pytest.raises(ValueError):
            preprocess(bars)  # zero-variance val_mean -> everything dropped


# ---------------------------------------------------------------------------
# TestResolveFreq — frequency string resolution
# ---------------------------------------------------------------------------


class TestResolveFreq:

    def test_known_timeframes(self):
        assert _resolve_freq("5min") == "5min"
        assert _resolve_freq("15min") == "15min"
        assert _resolve_freq("1h") == "1h"
        assert _resolve_freq("4h") == "4h"

    def test_raw_pandas_freq(self):
        assert _resolve_freq("10min") == "10min"
        assert _resolve_freq("2h") == "2h"

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError):
            _resolve_freq("invalid_freq_xyz")


# ---------------------------------------------------------------------------
# TestBuildAggPlan — aggregation plan structure
# ---------------------------------------------------------------------------


class TestBuildAggPlan:

    def test_price_column_gets_ohlc(self):
        plan = _build_agg_plan(["raw_midprice"])
        suffixes = [s for s, _ in plan["raw_midprice"]]
        assert "open" in suffixes
        assert "high" in suffixes
        assert "low" in suffixes
        assert "close" in suffixes

    def test_entropy_column_gets_slope(self):
        plan = _build_agg_plan(["ent_tick_1m"])
        suffixes = [s for s, _ in plan["ent_tick_1m"]]
        assert "slope" in suffixes
        assert "mean" in suffixes

    def test_sum_column_gets_sum(self):
        plan = _build_agg_plan(["flow_volume_1s"])
        suffixes = [s for s, _ in plan["flow_volume_1s"]]
        assert "sum" in suffixes
        assert "mean" not in suffixes

    def test_whale_sum_column(self):
        plan = _build_agg_plan(["whale_net_flow_1h"])
        suffixes = [s for s, _ in plan["whale_net_flow_1h"]]
        assert "sum" in suffixes

    def test_default_column_gets_mean_std_last(self):
        plan = _build_agg_plan(["trend_momentum_60"])
        suffixes = [s for s, _ in plan["trend_momentum_60"]]
        assert "mean" in suffixes
        assert "std" in suffixes
        assert "last" in suffixes


# ---------------------------------------------------------------------------
# TestBarSummary — inspection utilities
# ---------------------------------------------------------------------------


class TestBarSummary:

    def test_summary_structure(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        s = bar_summary(bars)
        assert "n_bars" in s
        assert "n_features" in s
        assert "symbols" in s
        assert "tick_count_stats" in s

    def test_summary_n_bars_correct(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        s = bar_summary(bars)
        assert s["n_bars"] == len(bars)

    def test_summary_symbols(self, tick_df_multi):
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        s = bar_summary(bars)
        assert set(s["symbols"]) == {"BTC", "ETH"}

    def test_list_bar_columns_structure(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        cols = list_bar_columns(bars)
        assert isinstance(cols, dict)
        # Should have at least mean and std
        assert "mean" in cols
        assert "std" in cols

    def test_list_bar_columns_slope_present(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        cols = list_bar_columns(bars)
        assert "slope" in cols, "Entropy slope columns should be categorized"


# ---------------------------------------------------------------------------
# TestDeterminism — same input -> same output
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_aggregate_deterministic(self, tick_df):
        bars1 = aggregate_bars(tick_df, timeframe="5min")
        bars2 = aggregate_bars(tick_df, timeframe="5min")
        pd.testing.assert_frame_equal(bars1, bars2)

    def test_preprocess_deterministic(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X1, cols1, _ = preprocess(bars, scaler="zscore")
        X2, cols2, _ = preprocess(bars, scaler="zscore")
        assert cols1 == cols2
        np.testing.assert_array_equal(X1, X2)

    def test_multi_timeframe_deterministic(self, tick_df_long):
        r1 = aggregate_multi_timeframe(tick_df_long, timeframes=["5min", "15min"])
        r2 = aggregate_multi_timeframe(tick_df_long, timeframes=["5min", "15min"])
        pd.testing.assert_frame_equal(r1["5min"], r2["5min"])
        pd.testing.assert_frame_equal(r1["15min"], r2["15min"])


# ---------------------------------------------------------------------------
# TestEndToEnd — full pipeline: ticks -> bars -> preprocess -> matrix
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests running the full preprocessing pipeline."""

    def test_full_pipeline_5min(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, meta = preprocess(bars, scaler="zscore")
        assert X.ndim == 2
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_full_pipeline_multi_symbol(self, tick_df_multi):
        bars = aggregate_bars(tick_df_multi, timeframe="5min")
        X, cols, meta = preprocess(bars, scaler="zscore")
        assert "symbol" in meta.columns
        assert set(meta["symbol"].unique()) == {"BTC", "ETH"}

    def test_full_pipeline_with_vector(self, tick_df):
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, cols, meta = preprocess(bars, vector="entropy", scaler="robust")
        assert all("ent_" in c for c in cols)
        assert X.shape[1] > 0

    def test_pipeline_preserves_bar_count(self, tick_df):
        """Number of rows should match unless bars are dropped."""
        bars = aggregate_bars(tick_df, timeframe="5min")
        X, _, meta = preprocess(bars, scaler="none", nan_threshold=1.0)
        assert X.shape[0] == len(bars)

    def test_pipeline_1h_on_long_data(self, tick_df_long):
        bars = aggregate_bars(tick_df_long, timeframe="1h")
        X, cols, meta = preprocess(bars, scaler="minmax")
        assert X.shape[0] >= 1
        assert X.min() >= -1e-10
        assert X.max() <= 1.0 + 1e-10

    def test_pipeline_all_timeframes(self, tick_df_very_long):
        """All 4 standard timeframes should produce valid outputs."""
        for tf in TIMEFRAMES:
            bars = aggregate_bars(tick_df_very_long, timeframe=tf)
            assert len(bars) >= 2, f"Need >=2 bars for {tf}, got {len(bars)}"
            X, cols, _ = preprocess(bars, scaler="zscore")
            assert X.shape[0] >= 1, f"No rows for {tf}"
            assert X.shape[1] >= 1, f"No columns for {tf}"
            assert not np.any(np.isnan(X)), f"NaN in {tf}"
