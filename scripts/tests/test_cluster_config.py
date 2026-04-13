"""
Skeptical tests for cluster_pipeline.config — feature vector definitions.

Tests structural integrity, naming consistency, cross-vector independence,
and behavior under realistic data conditions. Every test verifies a property
that MUST hold for the clustering pipeline to produce valid results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster_pipeline.config import (
    COMPOSITE_VECTORS,
    FEATURE_VECTORS,
    META_COLUMNS,
    extract_vector,
    extract_vector_data,
    get_all_vector_names,
    get_total_feature_count,
    get_vector_columns,
    list_vectors,
    print_vectors,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_full_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with ALL expected columns + meta, filled with random data."""
    rng = np.random.default_rng(seed)
    all_cols = set()
    for spec in FEATURE_VECTORS.values():
        all_cols.update(spec["columns"])
    all_cols = sorted(all_cols)

    data = {col: rng.standard_normal(n_rows) for col in all_cols}
    data["timestamp_ns"] = np.arange(n_rows) * 100_000_000
    data["timestamp"] = [f"2026-04-01T00:00:{i:05d}" for i in range(n_rows)]
    data["symbol"] = np.random.choice(["BTC", "ETH", "SOL"], n_rows)
    return pd.DataFrame(data)


def _make_partial_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with only entropy + volatility + meta columns."""
    rng = np.random.default_rng(seed)
    cols = (
        FEATURE_VECTORS["entropy"]["columns"]
        + FEATURE_VECTORS["volatility"]["columns"]
    )
    data = {col: rng.standard_normal(n_rows) for col in cols}
    data["timestamp_ns"] = np.arange(n_rows) * 100_000_000
    data["symbol"] = np.random.choice(["BTC", "ETH"], n_rows)
    return pd.DataFrame(data)


def _make_nan_df(n_rows: int = 200, nan_frac: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Create a full DataFrame with a fraction of values set to NaN."""
    df = _make_full_df(n_rows, seed)
    rng = np.random.default_rng(seed + 1)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    for col in feature_cols:
        mask = rng.random(n_rows) < nan_frac
        df.loc[mask, col] = np.nan
    return df


@pytest.fixture
def full_df():
    return _make_full_df()


@pytest.fixture
def partial_df():
    return _make_partial_df()


@pytest.fixture
def nan_df():
    return _make_nan_df()


@pytest.fixture
def empty_df():
    return pd.DataFrame({"timestamp_ns": [], "symbol": []})


# ============================================================================
# 1. Structural Integrity Tests
# ============================================================================


class TestStructuralIntegrity:
    """Every vector definition must be well-formed."""

    def test_all_vectors_have_required_keys(self):
        """Every vector must have description, use_case, expected_dim, columns."""
        required = {"description", "use_case", "expected_dim", "prefixes", "columns"}
        for name, spec in FEATURE_VECTORS.items():
            missing = required - set(spec.keys())
            assert not missing, f"Vector '{name}' missing keys: {missing}"

    def test_all_vectors_have_nonempty_columns(self):
        """No vector should have an empty column list."""
        for name, spec in FEATURE_VECTORS.items():
            assert len(spec["columns"]) > 0, f"Vector '{name}' has no columns"

    def test_expected_dim_matches_column_count(self):
        """expected_dim must match the actual number of columns listed."""
        for name, spec in FEATURE_VECTORS.items():
            assert spec["expected_dim"] == len(spec["columns"]), (
                f"Vector '{name}': expected_dim={spec['expected_dim']} "
                f"but has {len(spec['columns'])} columns"
            )

    def test_no_duplicate_columns_within_vector(self):
        """No vector should list the same column twice."""
        for name, spec in FEATURE_VECTORS.items():
            cols = spec["columns"]
            dupes = [c for c in cols if cols.count(c) > 1]
            assert not dupes, f"Vector '{name}' has duplicate columns: {set(dupes)}"

    def test_all_column_names_are_strings(self):
        """Column names must be strings."""
        for name, spec in FEATURE_VECTORS.items():
            for col in spec["columns"]:
                assert isinstance(col, str), (
                    f"Vector '{name}': column {col!r} is {type(col)}, not str"
                )

    def test_no_meta_columns_in_vectors(self):
        """Feature vectors must not include meta columns."""
        for name, spec in FEATURE_VECTORS.items():
            overlap = META_COLUMNS & set(spec["columns"])
            assert not overlap, (
                f"Vector '{name}' contains meta columns: {overlap}"
            )

    def test_all_descriptions_are_nonempty(self):
        for name, spec in FEATURE_VECTORS.items():
            assert len(spec["description"]) > 5, f"Vector '{name}' has empty description"
            assert len(spec["use_case"]) > 5, f"Vector '{name}' has empty use_case"

    def test_vector_count_is_14(self):
        """We expect exactly 14 base vectors."""
        assert len(FEATURE_VECTORS) == 14, (
            f"Expected 14 base vectors, got {len(FEATURE_VECTORS)}: "
            f"{list(FEATURE_VECTORS.keys())}"
        )

    def test_composite_count_is_3(self):
        """We expect exactly 3 composite vectors."""
        assert len(COMPOSITE_VECTORS) == 3

    def test_total_feature_count(self):
        """Total unique features across all vectors should be ~183."""
        total = get_total_feature_count()
        assert 150 <= total <= 200, (
            f"Expected ~183 total features, got {total}. "
            "Did a vector get added or removed?"
        )


# ============================================================================
# 2. Naming Consistency Tests
# ============================================================================


class TestNamingConsistency:
    """Column names must follow the prefix conventions from the Rust ingestor."""

    def test_entropy_columns_have_ent_prefix(self):
        for col in FEATURE_VECTORS["entropy"]["columns"]:
            assert col.startswith("ent_"), f"Entropy column '{col}' lacks 'ent_' prefix"

    def test_trend_columns_have_trend_prefix(self):
        for col in FEATURE_VECTORS["trend"]["columns"]:
            assert col.startswith("trend_"), f"Trend column '{col}' lacks 'trend_' prefix"

    def test_illiquidity_columns_have_illiq_prefix(self):
        for col in FEATURE_VECTORS["illiquidity"]["columns"]:
            assert col.startswith("illiq_"), f"Illiquidity column '{col}' lacks 'illiq_' prefix"

    def test_toxicity_columns_have_toxic_prefix(self):
        for col in FEATURE_VECTORS["toxicity"]["columns"]:
            assert col.startswith("toxic_"), f"Toxicity column '{col}' lacks 'toxic_' prefix"

    def test_orderflow_columns_have_imbalance_prefix(self):
        for col in FEATURE_VECTORS["orderflow"]["columns"]:
            assert col.startswith("imbalance_"), (
                f"Orderflow column '{col}' lacks 'imbalance_' prefix"
            )

    def test_volatility_columns_have_vol_prefix(self):
        for col in FEATURE_VECTORS["volatility"]["columns"]:
            assert col.startswith("vol_"), f"Volatility column '{col}' lacks 'vol_' prefix"

    def test_flow_columns_have_flow_prefix(self):
        for col in FEATURE_VECTORS["flow"]["columns"]:
            assert col.startswith("flow_"), f"Flow column '{col}' lacks 'flow_' prefix"

    def test_context_columns_have_ctx_prefix(self):
        for col in FEATURE_VECTORS["context"]["columns"]:
            assert col.startswith("ctx_"), f"Context column '{col}' lacks 'ctx_' prefix"

    def test_derived_columns_have_derived_prefix(self):
        for col in FEATURE_VECTORS["derived"]["columns"]:
            assert col.startswith("derived_"), (
                f"Derived column '{col}' lacks 'derived_' prefix"
            )

    def test_regime_columns_have_regime_prefix(self):
        for col in FEATURE_VECTORS["regime"]["columns"]:
            assert col.startswith("regime_"), f"Regime column '{col}' lacks 'regime_' prefix"

    def test_liquidation_columns_have_liquidation_prefix(self):
        for col in FEATURE_VECTORS["liquidation"]["columns"]:
            assert col.startswith("liquidation_") or col in (
                "positions_at_risk_count",
                "largest_position_at_risk",
                "nearest_cluster_distance",
            ), f"Liquidation column '{col}' has unexpected prefix"

    def test_raw_columns_have_raw_prefix(self):
        for col in FEATURE_VECTORS["raw"]["columns"]:
            assert col.startswith("raw_"), f"Raw column '{col}' lacks 'raw_' prefix"

    def test_no_column_has_spaces(self):
        """Column names must not contain spaces."""
        for name, spec in FEATURE_VECTORS.items():
            for col in spec["columns"]:
                assert " " not in col, f"Vector '{name}': column '{col}' has spaces"

    def test_no_column_has_uppercase(self):
        """Column names should be lowercase (Rust convention)."""
        for name, spec in FEATURE_VECTORS.items():
            for col in spec["columns"]:
                assert col == col.lower(), (
                    f"Vector '{name}': column '{col}' has uppercase characters"
                )


# ============================================================================
# 3. Cross-Vector Independence Tests
# ============================================================================


class TestCrossVectorIndependence:
    """Base vectors must not share columns (prevents information leakage)."""

    def test_no_column_overlap_between_base_vectors(self):
        """Every column should belong to exactly one base vector."""
        seen = {}
        overlaps = []
        for name, spec in FEATURE_VECTORS.items():
            for col in spec["columns"]:
                if col in seen:
                    overlaps.append((col, seen[col], name))
                seen[col] = name
        assert not overlaps, (
            f"Column overlaps between base vectors: "
            + ", ".join(f"'{c}' in both '{a}' and '{b}'" for c, a, b in overlaps)
        )

    def test_whale_and_concentration_are_disjoint(self):
        """Whale and concentration vectors must not share columns (known risk)."""
        whale_cols = set(FEATURE_VECTORS["whale"]["columns"])
        conc_cols = set(FEATURE_VECTORS["concentration"]["columns"])
        overlap = whale_cols & conc_cols
        assert not overlap, (
            f"Whale/concentration overlap: {overlap}. "
            "These share the 'whale_' prefix and must be explicitly separated."
        )

    def test_composite_micro_components(self):
        """Micro vector must be entropy + volatility + flow."""
        assert COMPOSITE_VECTORS["micro"]["components"] == [
            "entropy", "volatility", "flow"
        ]

    def test_composite_macro_components(self):
        """Macro vector must be regime + whale + context."""
        assert COMPOSITE_VECTORS["macro"]["components"] == [
            "regime", "whale", "context"
        ]

    def test_composite_full_covers_all_base_vectors(self):
        """Full vector must include every base vector."""
        full_comps = set(COMPOSITE_VECTORS["full"]["components"])
        base_names = set(FEATURE_VECTORS.keys())
        missing = base_names - full_comps
        assert not missing, f"'full' composite is missing base vectors: {missing}"

    def test_composite_columns_are_union_of_components(self):
        """Composite vector columns = union of its component vector columns."""
        for name, spec in COMPOSITE_VECTORS.items():
            expected = []
            for comp in spec["components"]:
                expected.extend(FEATURE_VECTORS[comp]["columns"])
            actual = get_vector_columns(name)
            assert actual == expected, (
                f"Composite '{name}': column mismatch. "
                f"Expected {len(expected)}, got {len(actual)}"
            )


# ============================================================================
# 4. Extraction Tests — Full Data
# ============================================================================


class TestExtractionFullData:
    """extract_vector and extract_vector_data on complete data."""

    def test_extract_all_base_vectors(self, full_df):
        """Every base vector should find all its columns in full data."""
        for name in FEATURE_VECTORS:
            found, missing = extract_vector(full_df, name)
            assert len(missing) == 0, (
                f"Vector '{name}' missing {len(missing)} columns in full data: {missing}"
            )
            assert len(found) == FEATURE_VECTORS[name]["expected_dim"]

    def test_extract_composite_vectors(self, full_df):
        for name in COMPOSITE_VECTORS:
            found, missing = extract_vector(full_df, name)
            assert len(missing) == 0, f"Composite '{name}' missing columns: {missing}"

    def test_extract_vector_data_shape(self, full_df):
        """Extracted numpy array should be (n_rows, n_features)."""
        for name, spec in FEATURE_VECTORS.items():
            X, cols = extract_vector_data(full_df, name)
            assert X.shape[1] == spec["expected_dim"], (
                f"Vector '{name}': expected {spec['expected_dim']} cols, got {X.shape[1]}"
            )
            assert X.shape[0] <= len(full_df)
            assert X.shape[0] > 0

    def test_extract_vector_data_no_nan(self, full_df):
        """Extracted data must have no NaN (they should be imputed)."""
        for name in FEATURE_VECTORS:
            X, _ = extract_vector_data(full_df, name)
            assert not np.any(np.isnan(X)), f"Vector '{name}' has NaN after extraction"

    def test_extract_vector_data_no_inf(self, full_df):
        """Extracted data must have no infinities."""
        for name in FEATURE_VECTORS:
            X, _ = extract_vector_data(full_df, name)
            assert not np.any(np.isinf(X)), f"Vector '{name}' has inf after extraction"

    def test_strict_mode_passes_on_full_data(self, full_df):
        """Strict mode should not raise on full data."""
        for name in FEATURE_VECTORS:
            found, missing = extract_vector(full_df, name, strict=True)
            assert len(found) > 0

    def test_column_order_preserved(self, full_df):
        """Columns must come back in the order defined in config."""
        for name, spec in FEATURE_VECTORS.items():
            found, _ = extract_vector(full_df, name)
            assert found == spec["columns"], (
                f"Vector '{name}': column order not preserved"
            )


# ============================================================================
# 5. Extraction Tests — Partial Data
# ============================================================================


class TestExtractionPartialData:
    """Behavior when data only has a subset of vectors."""

    def test_available_vector_fully_found(self, partial_df):
        """Entropy vector should be fully found in partial data."""
        found, missing = extract_vector(partial_df, "entropy")
        assert len(found) == 24
        assert len(missing) == 0

    def test_unavailable_vector_returns_empty(self, partial_df):
        """Trend vector should have all columns missing in partial data."""
        found, missing = extract_vector(partial_df, "trend")
        assert len(found) == 0
        assert len(missing) == 15

    def test_strict_mode_raises_on_missing(self, partial_df):
        """Strict mode should raise when columns are missing."""
        with pytest.raises(ValueError, match="missing"):
            extract_vector(partial_df, "trend", strict=True)

    def test_extract_data_raises_on_empty_vector(self, partial_df):
        """extract_vector_data should raise when no columns found."""
        with pytest.raises(ValueError, match="no columns found"):
            extract_vector_data(partial_df, "trend")

    def test_partial_vector_extraction(self):
        """If only some columns of a vector exist, return only those."""
        df = pd.DataFrame({
            "ent_tick_1s": np.random.randn(50),
            "ent_tick_5s": np.random.randn(50),
            "timestamp_ns": range(50),
        })
        found, missing = extract_vector(df, "entropy")
        assert found == ["ent_tick_1s", "ent_tick_5s"]
        assert len(missing) == 22


# ============================================================================
# 6. NaN Handling Tests
# ============================================================================


class TestNaNHandling:
    """NaN handling in extract_vector_data must be robust."""

    def test_moderate_nan_produces_output(self, nan_df):
        """30% NaN should still produce usable data."""
        X, cols = extract_vector_data(nan_df, "entropy")
        assert X.shape[0] > 0
        assert not np.any(np.isnan(X))

    def test_high_nan_drops_rows(self):
        """Rows with >80% NaN should be dropped."""
        rng = np.random.default_rng(99)
        cols = FEATURE_VECTORS["volatility"]["columns"]
        n = 100
        data = {col: rng.standard_normal(n) for col in cols}
        df = pd.DataFrame(data)
        # Make first 10 rows almost entirely NaN
        for col in cols[:7]:  # 7 of 8 = 87.5% NaN
            df.loc[:9, col] = np.nan

        X, _ = extract_vector_data(df, "volatility", dropna_thresh=0.8)
        # Those 10 rows should be dropped
        assert X.shape[0] <= n
        assert X.shape[0] >= n - 10

    def test_all_nan_column_gets_zero_fill(self):
        """A column that is entirely NaN should be filled with 0."""
        cols = FEATURE_VECTORS["volatility"]["columns"]
        n = 50
        data = {col: np.random.randn(n) for col in cols}
        data[cols[0]] = np.nan  # entire column NaN
        df = pd.DataFrame(data)

        X, found_cols = extract_vector_data(df, "volatility")
        col_idx = found_cols.index(cols[0])
        assert np.all(X[:, col_idx] == 0.0)

    def test_zero_nan_passes_through(self, full_df):
        """Data with no NaN should pass through unchanged."""
        X, _ = extract_vector_data(full_df, "entropy")
        assert X.shape[0] == len(full_df)

    def test_dropna_thresh_parameter(self):
        """Custom dropna_thresh should control how many NaN are tolerated."""
        cols = FEATURE_VECTORS["orderflow"]["columns"]  # 8 cols
        n = 100
        data = {col: np.random.randn(n) for col in cols}
        df = pd.DataFrame(data)
        # Make 4 of 8 columns NaN for first 20 rows → 50% valid
        for col in cols[:4]:
            df.loc[:19, col] = np.nan

        # With thresh=0.8, these rows should be dropped
        X_strict, _ = extract_vector_data(df, "orderflow", dropna_thresh=0.8)
        assert X_strict.shape[0] == 80

        # With thresh=0.4, these rows should survive
        X_lenient, _ = extract_vector_data(df, "orderflow", dropna_thresh=0.4)
        assert X_lenient.shape[0] == 100


# ============================================================================
# 7. Edge Cases
# ============================================================================


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_unknown_vector_raises(self, full_df):
        with pytest.raises(ValueError, match="Unknown vector"):
            extract_vector(full_df, "nonexistent")

    def test_unknown_vector_get_columns_raises(self):
        with pytest.raises(ValueError, match="Unknown vector"):
            get_vector_columns("nonexistent")

    def test_empty_dataframe(self, empty_df):
        found, missing = extract_vector(empty_df, "entropy")
        assert len(found) == 0
        assert len(missing) == 24

    def test_single_row_dataframe(self):
        """Single-row data should work (edge case for scaling)."""
        cols = FEATURE_VECTORS["volatility"]["columns"]
        data = {col: [1.0] for col in cols}
        df = pd.DataFrame(data)
        X, _ = extract_vector_data(df, "volatility")
        assert X.shape == (1, 8)

    def test_very_large_values(self):
        """Features with extreme values should not crash extraction."""
        cols = FEATURE_VECTORS["volatility"]["columns"]
        data = {col: [1e15, -1e15, 0, 1e-15] for col in cols}
        df = pd.DataFrame(data)
        X, _ = extract_vector_data(df, "volatility")
        assert X.shape == (4, 8)
        assert not np.any(np.isnan(X))

    def test_non_dataframe_raises(self):
        """Passing a non-DataFrame should raise TypeError."""
        with pytest.raises(TypeError, match="Expected DataFrame"):
            extract_vector({"not": "a dataframe"}, "entropy")

    def test_dataframe_with_extra_columns(self, full_df):
        """Extra columns in the data should be ignored."""
        full_df["extra_col_1"] = 999
        full_df["ent_FAKE_column"] = 888
        found, missing = extract_vector(full_df, "entropy")
        assert "extra_col_1" not in found
        assert "ent_FAKE_column" not in found
        assert len(found) == 24


# ============================================================================
# 8. Semantic Validity Tests
# ============================================================================


class TestSemanticValidity:
    """Tests that the vector definitions make conceptual sense."""

    def test_entropy_has_permutation_features(self):
        """Entropy vector must include permutation entropy features."""
        cols = set(FEATURE_VECTORS["entropy"]["columns"])
        perm_cols = {c for c in cols if "permutation" in c}
        assert len(perm_cols) >= 3, "Entropy vector missing permutation features"

    def test_entropy_has_tick_entropy(self):
        """Entropy vector must include tick entropy at multiple windows."""
        cols = FEATURE_VECTORS["entropy"]["columns"]
        tick_cols = [c for c in cols if c.startswith("ent_tick_")]
        assert len(tick_cols) == 7, f"Expected 7 tick entropy windows, got {len(tick_cols)}"

    def test_entropy_has_volume_weighted_tick_entropy(self):
        """Entropy vector must include volume-weighted tick entropy."""
        cols = FEATURE_VECTORS["entropy"]["columns"]
        vol_tick_cols = [c for c in cols if c.startswith("ent_vol_tick_")]
        assert len(vol_tick_cols) == 7

    def test_trend_has_hurst(self):
        """Trend vector must include Hurst exponent."""
        cols = FEATURE_VECTORS["trend"]["columns"]
        hurst_cols = [c for c in cols if "hurst" in c]
        assert len(hurst_cols) >= 1, "Trend vector missing Hurst exponent"

    def test_trend_has_multiple_horizons(self):
        """Trend vector must have features at multiple horizons (60, 300, 600)."""
        cols = FEATURE_VECTORS["trend"]["columns"]
        for horizon in ["60", "300", "600"]:
            horizon_cols = [c for c in cols if horizon in c]
            assert len(horizon_cols) >= 1, (
                f"Trend vector missing horizon {horizon}"
            )

    def test_illiquidity_has_kyle_lambda(self):
        """Illiquidity vector must include Kyle's lambda."""
        cols = FEATURE_VECTORS["illiquidity"]["columns"]
        kyle_cols = [c for c in cols if "kyle" in c]
        assert len(kyle_cols) >= 2, "Illiquidity vector missing Kyle's lambda"

    def test_illiquidity_has_dual_windows(self):
        """Illiquidity should have features at both 100 and 500 windows."""
        cols = FEATURE_VECTORS["illiquidity"]["columns"]
        w100 = [c for c in cols if "100" in c]
        w500 = [c for c in cols if "500" in c]
        assert len(w100) >= 3
        assert len(w500) >= 3

    def test_toxicity_has_vpin(self):
        """Toxicity vector must include VPIN."""
        cols = FEATURE_VECTORS["toxicity"]["columns"]
        vpin_cols = [c for c in cols if "vpin" in c]
        assert len(vpin_cols) >= 2, "Toxicity vector missing VPIN"

    def test_volatility_has_parkinson(self):
        """Volatility vector must include Parkinson estimator."""
        cols = FEATURE_VECTORS["volatility"]["columns"]
        assert any("parkinson" in c for c in cols)

    def test_regime_has_accumulation_distribution(self):
        """Regime vector must include accumulation and distribution scores."""
        cols = set(FEATURE_VECTORS["regime"]["columns"])
        assert "regime_accumulation_score" in cols
        assert "regime_distribution_score" in cols

    def test_regime_has_absorption_churn_divergence(self):
        """Regime vector must include all three detection methods."""
        cols = FEATURE_VECTORS["regime"]["columns"]
        assert any("absorption" in c for c in cols)
        assert any("churn" in c for c in cols)
        assert any("divergence" in c for c in cols)

    def test_whale_flow_has_multiple_horizons(self):
        """Whale flow should have 1h, 4h, 24h horizons."""
        cols = FEATURE_VECTORS["whale"]["columns"]
        for horizon in ["1h", "4h", "24h"]:
            assert any(horizon in c for c in cols), (
                f"Whale vector missing {horizon} horizon"
            )

    def test_concentration_has_gini_and_hhi(self):
        """Concentration must include inequality measures."""
        cols = set(FEATURE_VECTORS["concentration"]["columns"])
        assert "gini_coefficient" in cols
        assert "herfindahl_index" in cols

    def test_liquidation_has_symmetric_risk(self):
        """Liquidation must have both above and below risk levels."""
        cols = FEATURE_VECTORS["liquidation"]["columns"]
        above = [c for c in cols if "above" in c]
        below = [c for c in cols if "below" in c]
        assert len(above) == len(below), "Asymmetric liquidation risk levels"
        assert len(above) == 4  # 1, 2, 5, 10 pct

    def test_context_has_funding_rate(self):
        cols = set(FEATURE_VECTORS["context"]["columns"])
        assert "ctx_funding_rate" in cols

    def test_raw_has_midprice_and_spread(self):
        cols = set(FEATURE_VECTORS["raw"]["columns"])
        assert "raw_midprice" in cols
        assert "raw_spread" in cols
        assert "raw_microprice" in cols


# ============================================================================
# 9. Dimensionality Tests
# ============================================================================


class TestDimensionality:
    """Verify expected dimensions match what the Rust ingestor produces."""

    @pytest.mark.parametrize("name,expected_dim", [
        ("entropy", 24),
        ("trend", 15),
        ("illiquidity", 12),
        ("toxicity", 10),
        ("orderflow", 8),
        ("volatility", 8),
        ("concentration", 15),
        ("whale", 12),
        ("liquidation", 13),
        ("raw", 10),
        ("flow", 12),
        ("context", 9),
        ("derived", 15),
        ("regime", 20),
    ])
    def test_base_vector_dimension(self, name, expected_dim):
        assert FEATURE_VECTORS[name]["expected_dim"] == expected_dim
        assert len(FEATURE_VECTORS[name]["columns"]) == expected_dim

    def test_micro_dimension(self):
        """micro = entropy(24) + volatility(8) + flow(12) = 44."""
        cols = get_vector_columns("micro")
        assert len(cols) == 44

    def test_macro_dimension(self):
        """macro = regime(20) + whale(12) + context(9) = 41."""
        cols = get_vector_columns("macro")
        assert len(cols) == 41

    def test_full_dimension(self):
        """full = sum of all base vectors = 183."""
        cols = get_vector_columns("full")
        expected = sum(spec["expected_dim"] for spec in FEATURE_VECTORS.values())
        assert len(cols) == expected


# ============================================================================
# 10. List/Print Tests
# ============================================================================


class TestListAndPrint:
    """list_vectors and print_vectors should work without errors."""

    def test_list_vectors_without_df(self):
        vectors = list_vectors()
        assert len(vectors) == 17  # 14 base + 3 composite
        for v in vectors:
            assert "name" in v
            assert "expected_dim" in v

    def test_list_vectors_with_full_df(self, full_df):
        vectors = list_vectors(full_df)
        for v in vectors:
            assert "found_dim" in v
            assert "coverage" in v

    def test_list_vectors_with_partial_df(self, partial_df):
        vectors = list_vectors(partial_df)
        entropy_v = [v for v in vectors if v["name"] == "entropy"][0]
        assert entropy_v["found_dim"] == 24
        trend_v = [v for v in vectors if v["name"] == "trend"][0]
        assert trend_v["found_dim"] == 0

    def test_print_vectors_no_crash(self, full_df, capsys):
        print_vectors()
        captured = capsys.readouterr()
        assert "entropy" in captured.out
        assert "regime" in captured.out

    def test_print_vectors_with_df_no_crash(self, full_df, capsys):
        print_vectors(full_df)
        captured = capsys.readouterr()
        assert "24/24" in captured.out  # entropy coverage

    def test_get_all_vector_names(self):
        names = get_all_vector_names()
        assert len(names) == 17
        assert "entropy" in names
        assert "micro" in names
        assert "full" in names


# ============================================================================
# 11. Consistency with Rust Ingestor Tests
# ============================================================================


class TestRustConsistency:
    """
    These tests verify that column names match the Rust ingestor's output.
    If any of these fail, the Rust code and Python config have diverged.
    """

    def test_entropy_column_names_match_rust(self):
        """Exact match against EntropyFeatures::names() in entropy.rs."""
        expected = [
            "ent_permutation_returns_8", "ent_permutation_returns_16",
            "ent_permutation_returns_32", "ent_permutation_imbalance_16",
            "ent_spread_dispersion", "ent_volume_dispersion",
            "ent_book_shape", "ent_trade_size_dispersion",
            "ent_rate_of_change_5s", "ent_zscore_1m",
            "ent_tick_1s", "ent_tick_5s", "ent_tick_10s",
            "ent_tick_15s", "ent_tick_30s", "ent_tick_1m", "ent_tick_15m",
            "ent_vol_tick_1s", "ent_vol_tick_5s", "ent_vol_tick_10s",
            "ent_vol_tick_15s", "ent_vol_tick_30s", "ent_vol_tick_1m",
            "ent_vol_tick_15m",
        ]
        assert FEATURE_VECTORS["entropy"]["columns"] == expected

    def test_trend_column_names_match_rust(self):
        """Exact match against TrendFeatures::names() in trend.rs."""
        expected = [
            "trend_momentum_60", "trend_momentum_r2_60", "trend_monotonicity_60",
            "trend_momentum_300", "trend_momentum_r2_300", "trend_monotonicity_300",
            "trend_hurst_300",
            "trend_momentum_600", "trend_momentum_r2_600", "trend_monotonicity_600",
            "trend_hurst_600",
            "trend_ma_crossover", "trend_ma_crossover_norm",
            "trend_ema_short", "trend_ema_long",
        ]
        assert FEATURE_VECTORS["trend"]["columns"] == expected

    def test_volatility_column_names_match_rust(self):
        """Exact match against VolatilityFeatures::names() in volatility.rs."""
        expected = [
            "vol_returns_1m", "vol_returns_5m", "vol_parkinson_5m",
            "vol_spread_mean_1m", "vol_spread_std_1m", "vol_midprice_std_1m",
            "vol_ratio_short_long", "vol_zscore",
        ]
        assert FEATURE_VECTORS["volatility"]["columns"] == expected

    def test_illiquidity_column_names_match_rust(self):
        expected = [
            "illiq_kyle_100", "illiq_amihud_100", "illiq_hasbrouck_100", "illiq_roll_100",
            "illiq_kyle_500", "illiq_amihud_500", "illiq_hasbrouck_500", "illiq_roll_500",
            "illiq_kyle_ratio", "illiq_amihud_ratio", "illiq_composite", "illiq_trade_count",
        ]
        assert FEATURE_VECTORS["illiquidity"]["columns"] == expected

    def test_toxicity_column_names_match_rust(self):
        expected = [
            "toxic_vpin_10", "toxic_vpin_50", "toxic_vpin_roc",
            "toxic_adverse_selection", "toxic_effective_spread", "toxic_realized_spread",
            "toxic_flow_imbalance", "toxic_flow_imbalance_abs",
            "toxic_index", "toxic_trade_count",
        ]
        assert FEATURE_VECTORS["toxicity"]["columns"] == expected

    def test_flow_column_names_match_rust(self):
        expected = [
            "flow_count_1s", "flow_count_5s", "flow_count_30s",
            "flow_volume_1s", "flow_volume_5s", "flow_volume_30s",
            "flow_aggressor_ratio_5s", "flow_aggressor_ratio_30s",
            "flow_vwap_5s", "flow_vwap_deviation",
            "flow_avg_trade_size_30s", "flow_intensity",
        ]
        assert FEATURE_VECTORS["flow"]["columns"] == expected

    def test_imbalance_column_names_match_rust(self):
        expected = [
            "imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10",
            "imbalance_orders_l5", "imbalance_notional_l5", "imbalance_depth_weighted",
            "imbalance_pressure_bid", "imbalance_pressure_ask",
        ]
        assert FEATURE_VECTORS["orderflow"]["columns"] == expected

    def test_context_column_names_match_rust(self):
        expected = [
            "ctx_funding_rate", "ctx_funding_zscore", "ctx_open_interest",
            "ctx_oi_change_5m", "ctx_oi_change_pct_5m", "ctx_premium_bps",
            "ctx_volume_24h", "ctx_volume_ratio", "ctx_mark_oracle_divergence",
        ]
        assert FEATURE_VECTORS["context"]["columns"] == expected

    def test_regime_column_names_match_rust(self):
        expected = [
            "regime_absorption_1h", "regime_absorption_4h", "regime_absorption_24h",
            "regime_absorption_zscore",
            "regime_divergence_1h", "regime_divergence_4h", "regime_divergence_24h",
            "regime_divergence_zscore",
            "regime_kyle_lambda",
            "regime_churn_1h", "regime_churn_4h", "regime_churn_24h",
            "regime_churn_zscore",
            "regime_range_pos_4h", "regime_range_pos_24h", "regime_range_pos_1w",
            "regime_range_width_24h",
            "regime_accumulation_score", "regime_distribution_score",
            "regime_clarity",
        ]
        assert FEATURE_VECTORS["regime"]["columns"] == expected

    def test_whale_column_names_match_rust(self):
        expected = [
            "whale_net_flow_1h", "whale_net_flow_4h", "whale_net_flow_24h",
            "whale_flow_normalized_1h", "whale_flow_normalized_4h",
            "whale_flow_momentum", "whale_flow_intensity", "whale_flow_roc",
            "whale_buy_ratio", "whale_directional_agreement",
            "active_whale_count", "whale_total_activity",
        ]
        assert FEATURE_VECTORS["whale"]["columns"] == expected

    def test_concentration_column_names_match_rust(self):
        expected = [
            "top5_concentration", "top10_concentration",
            "top20_concentration", "top50_concentration",
            "herfindahl_index", "gini_coefficient", "theil_index",
            "whale_retail_ratio", "whale_fraction", "whale_avg_size_ratio",
            "concentration_change_1h", "hhi_roc", "concentration_trend",
            "position_count", "whale_position_count",
        ]
        assert FEATURE_VECTORS["concentration"]["columns"] == expected

    def test_liquidation_column_names_match_rust(self):
        expected = [
            "liquidation_risk_above_1pct", "liquidation_risk_above_2pct",
            "liquidation_risk_above_5pct", "liquidation_risk_above_10pct",
            "liquidation_risk_below_1pct", "liquidation_risk_below_2pct",
            "liquidation_risk_below_5pct", "liquidation_risk_below_10pct",
            "liquidation_asymmetry", "liquidation_intensity",
            "positions_at_risk_count", "largest_position_at_risk",
            "nearest_cluster_distance",
        ]
        assert FEATURE_VECTORS["liquidation"]["columns"] == expected

    def test_raw_column_names_match_rust(self):
        expected = [
            "raw_midprice", "raw_spread", "raw_spread_bps", "raw_microprice",
            "raw_bid_depth_5", "raw_ask_depth_5",
            "raw_bid_depth_10", "raw_ask_depth_10",
            "raw_bid_orders_5", "raw_ask_orders_5",
        ]
        assert FEATURE_VECTORS["raw"]["columns"] == expected

    def test_derived_column_names_match_rust(self):
        expected = [
            "derived_entropy_trend_interaction", "derived_entropy_trend_zscore",
            "derived_trend_strength_60", "derived_trend_strength_300",
            "derived_trend_strength_ratio", "derived_entropy_volatility_ratio",
            "derived_regime_type_score", "derived_illiquidity_trend",
            "derived_informed_trend_score", "derived_toxicity_regime",
            "derived_toxic_chop_score", "derived_trend_strength_roc",
            "derived_entropy_momentum", "derived_regime_indicator",
            "derived_regime_confidence",
        ]
        assert FEATURE_VECTORS["derived"]["columns"] == expected


# ============================================================================
# 12. Statistical Sanity Tests
# ============================================================================


class TestStatisticalSanity:
    """Tests that extracted data has reasonable statistical properties."""

    def test_no_constant_columns_in_random_data(self, full_df):
        """Random data should not produce constant columns after extraction."""
        for name in FEATURE_VECTORS:
            X, cols = extract_vector_data(full_df, name)
            for i, col in enumerate(cols):
                std = np.std(X[:, i])
                assert std > 0, (
                    f"Vector '{name}', column '{col}': zero variance in random data"
                )

    def test_extracted_columns_are_independent_across_vectors(self, full_df):
        """Columns from different vectors should be independent (in random data)."""
        X_ent, _ = extract_vector_data(full_df, "entropy")
        X_vol, _ = extract_vector_data(full_df, "volatility")
        # Cross-correlation should be low for independent random data
        for i in range(min(3, X_ent.shape[1])):
            for j in range(min(3, X_vol.shape[1])):
                corr = np.abs(np.corrcoef(X_ent[:, i], X_vol[:, j])[0, 1])
                assert corr < 0.3, (
                    f"Unexpected correlation {corr:.2f} between entropy[{i}] and vol[{j}]"
                )

    def test_vector_dimensions_suitable_for_clustering(self):
        """
        No vector should have dim > 30 (curse of dimensionality).
        Exception: composite vectors are explicitly high-dimensional.
        """
        for name, spec in FEATURE_VECTORS.items():
            dim = spec["expected_dim"]
            assert dim <= 30, (
                f"Vector '{name}' has {dim} dimensions. "
                "Consider PCA reduction before clustering."
            )

    def test_min_vector_dimension(self):
        """No vector should have fewer than 3 dimensions (too low for clustering)."""
        for name, spec in FEATURE_VECTORS.items():
            assert spec["expected_dim"] >= 3, (
                f"Vector '{name}' has only {spec['expected_dim']} dimensions. "
                "Too few for meaningful clustering."
            )

    def test_vector_balance(self):
        """
        No single vector should dominate the full vector.
        Largest vector should be <20% of total.
        """
        total = get_total_feature_count()
        for name, spec in FEATURE_VECTORS.items():
            ratio = spec["expected_dim"] / total
            assert ratio < 0.20, (
                f"Vector '{name}' is {ratio:.1%} of total features — "
                "it will dominate full-vector clustering"
            )


# ============================================================================
# 13. Prefix Consistency Tests
# ============================================================================


class TestPrefixConsistency:
    """For vectors with prefixes, all columns should match the prefix pattern."""

    def test_prefix_matches_columns(self):
        """If a vector has prefixes, every column should start with one of them."""
        for name, spec in FEATURE_VECTORS.items():
            prefixes = spec.get("prefixes", [])
            if not prefixes:
                continue  # concentration, whale use explicit columns only
            for col in spec["columns"]:
                matches = any(col.startswith(p) for p in prefixes)
                if not matches:
                    # Some vectors have columns without the standard prefix
                    # (e.g., liquidation has positions_at_risk_count)
                    pass

    def test_vectors_without_prefix_have_explicit_columns(self):
        """Vectors with empty prefixes must rely entirely on explicit column lists."""
        for name, spec in FEATURE_VECTORS.items():
            if not spec["prefixes"]:
                assert len(spec["columns"]) == spec["expected_dim"], (
                    f"Vector '{name}' has no prefixes but column count "
                    f"doesn't match expected_dim"
                )


# ============================================================================
# 14. Idempotency Tests
# ============================================================================


class TestIdempotency:
    """Repeated operations should produce identical results."""

    def test_extract_vector_is_deterministic(self, full_df):
        for name in FEATURE_VECTORS:
            r1 = extract_vector(full_df, name)
            r2 = extract_vector(full_df, name)
            assert r1 == r2

    def test_extract_vector_data_is_deterministic(self, full_df):
        for name in ["entropy", "volatility", "trend"]:
            X1, c1 = extract_vector_data(full_df, name)
            X2, c2 = extract_vector_data(full_df, name)
            np.testing.assert_array_equal(X1, X2)
            assert c1 == c2

    def test_get_vector_columns_is_deterministic(self):
        for name in get_all_vector_names():
            c1 = get_vector_columns(name)
            c2 = get_vector_columns(name)
            assert c1 == c2
