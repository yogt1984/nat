"""Unit tests for EAMM Context Feature Extractor."""

import numpy as np
import polars as pl
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.features import (
    extract_context,
    context_to_numpy,
    validate_context,
    CONTEXT_FEATURE_NAMES,
    CONTEXT_PARQUET_COLS,
    CONTEXT_FEATURE_COUNT,
    LN3,
)


def _make_context_df(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Create a synthetic DataFrame with all required context columns."""
    rng = np.random.default_rng(seed)
    data = {"timestamp_ns": list(range(n)), "symbol": ["BTC"] * n}
    for col in CONTEXT_PARQUET_COLS:
        if "ent_tick" in col:
            data[col] = rng.uniform(0, LN3, n).tolist()
        elif "ent_permutation" in col:
            data[col] = rng.uniform(0, 1, n).tolist()
        elif "vol_" in col:
            data[col] = rng.uniform(0, 0.001, n).tolist()
        elif "imbalance" in col:
            data[col] = rng.uniform(-1, 1, n).tolist()
        elif "toxic_vpin" in col:
            data[col] = rng.uniform(0, 1, n).tolist()
        elif "toxic_index" in col:
            data[col] = rng.uniform(0, 1, n).tolist()
        elif "flow_aggressor" in col:
            data[col] = rng.uniform(0, 1, n).tolist()
        elif "trend_hurst" in col:
            data[col] = rng.uniform(0, 1, n).tolist()
        elif "raw_spread_bps" in col:
            data[col] = rng.uniform(0.1, 2.0, n).tolist()
        else:
            data[col] = rng.standard_normal(n).tolist()
    return pl.DataFrame(data)


class TestFeatureCount:
    """Output must have exactly 19 context features."""

    def test_count(self):
        assert CONTEXT_FEATURE_COUNT == 19

    def test_output_columns(self):
        df = _make_context_df()
        ctx = extract_context(df)
        # Should have timestamp_ns + 19 features = 20 columns
        assert len(ctx.columns) == 20
        assert ctx.columns[0] == "timestamp_ns"
        for name in CONTEXT_FEATURE_NAMES:
            assert name in ctx.columns, f"Missing feature: {name}"


class TestFeatureNames:
    """Feature names must match the spec."""

    def test_names_match_spec(self):
        expected = [
            "H_tick_1s", "H_tick_5s", "H_tick_30s", "H_tick_1m",
            "H_perm_8", "H_perm_16", "H_perm_32",
            "VPIN_50", "toxic_index", "adverse_sel",
            "sigma_1m", "sigma_5m",
            "lambda_flow", "aggressor_5s",
            "I_l1", "I_l5",
            "mom_60", "hurst_300",
            "S_bps",
        ]
        assert CONTEXT_FEATURE_NAMES == expected


class TestNoNanInOutput:
    """After extraction, there should be no NaN values."""

    def test_clean_output(self):
        df = _make_context_df()
        ctx = extract_context(df)
        for name in CONTEXT_FEATURE_NAMES:
            assert ctx[name].is_nan().sum() == 0, f"NaN found in {name}"
            assert ctx[name].is_null().sum() == 0, f"Null found in {name}"

    def test_nan_replacement(self):
        """NaN values in source should be replaced with 0.0."""
        df = _make_context_df()
        # Inject NaN into one column
        vals = df["ent_tick_1s"].to_list()
        vals[0] = float("nan")
        vals[5] = float("nan")
        df = df.with_columns(pl.Series("ent_tick_1s", vals))

        with pytest.warns(UserWarning, match="NaN/null values"):
            ctx = extract_context(df)

        assert ctx["H_tick_1s"][0] == 0.0
        assert ctx["H_tick_1s"][5] == 0.0
        assert ctx["H_tick_1s"].is_nan().sum() == 0


class TestEntropyRange:
    """Entropy features should be within expected bounds."""

    def test_tick_entropy_range(self):
        df = _make_context_df()
        ctx = extract_context(df)
        for name in ["H_tick_1s", "H_tick_5s", "H_tick_30s", "H_tick_1m"]:
            arr = ctx[name].to_numpy()
            assert np.all(arr >= -0.01), f"{name} has values below 0"
            assert np.all(arr <= LN3 + 0.01), f"{name} has values above ln(3)"

    def test_permutation_entropy_range(self):
        df = _make_context_df()
        ctx = extract_context(df)
        for name in ["H_perm_8", "H_perm_16", "H_perm_32"]:
            arr = ctx[name].to_numpy()
            assert np.all(arr >= -0.01), f"{name} has values below 0"
            assert np.all(arr <= 1.01), f"{name} has values above 1"


class TestFeatureAlignment:
    """Timestamps must match between source and context."""

    def test_timestamp_alignment(self):
        df = _make_context_df()
        ctx = extract_context(df)
        assert (ctx["timestamp_ns"].to_numpy() == df["timestamp_ns"].to_numpy()).all()


class TestMissingColumns:
    """Should raise ValueError if required columns are missing."""

    def test_missing_column_raises(self):
        df = _make_context_df().drop("ent_tick_1s")
        with pytest.raises(ValueError, match="Missing required columns"):
            extract_context(df)


class TestContextToNumpy:
    """Test conversion to numpy array."""

    def test_shape(self):
        df = _make_context_df(n=50)
        ctx = extract_context(df)
        arr = context_to_numpy(ctx)
        assert arr.shape == (50, 19)
        assert arr.dtype == np.float64


class TestValidation:
    """Test the validation function."""

    def test_valid_data_no_issues(self):
        df = _make_context_df()
        ctx = extract_context(df)
        issues = validate_context(ctx)
        assert len(issues) == 0, f"Unexpected issues: {issues}"

    def test_out_of_range_detected(self):
        df = _make_context_df()
        ctx = extract_context(df)
        # Inject out-of-range value
        vals = ctx["H_perm_8"].to_list()
        vals[0] = 5.0  # Way above [0, 1]
        ctx = ctx.with_columns(pl.Series("H_perm_8", vals))
        issues = validate_context(ctx)
        assert "H_perm_8" in issues
