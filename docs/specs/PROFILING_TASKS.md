# NAT Profiling System — Implementation Tasks

**Version:** 2.0
**Date:** 2026-04-27
**Parent Document:** PROFILING_REQUIREMENTS.md
**Status:** Planning (revised after self-critique)

---

## Overview

This document defines every implementation task required to build the profiling system described in PROFILING_REQUIREMENTS.md. Tasks are organized into 9 phases, ordered by dependency. Each task specifies: what to build, where it lives, inputs/outputs, acceptance criteria, and what to test afterward.

The implementation target is Python (in `scripts/cluster_pipeline/`), with the online detector eventually ported to Rust once the research validates.

### Design Principles (v2 corrections)

1. **Selective derivation over exhaustive derivation.** Don't derive everything from everything. Select top features by variance first, then derive.
2. **Data-driven splits over naming-convention splits.** Use autocorrelation to classify slow/fast features, not column name parsing.
3. **Validate early, fail fast.** Test for structure existence before building hierarchy. Don't force GMM on uniform data.
4. **Block bootstrap over random bootstrap.** Preserve temporal structure in stability testing.
5. **Multi-horizon returns.** Never evaluate predictive quality at a single horizon.
6. **Built-in drift detection.** Not an afterthought — part of the online detector from day one.
7. **Spectral features are experimental.** Include them with a kill criterion, not as core pipeline.

---

## Phase 0: Data Collection Infrastructure

Everything downstream depends on having enough data. This phase runs in parallel with Phase 1 development.

### Task 0.1: Verify Ingestor Stability for Multi-Day Collection

**What:** Confirm the ingestor (`rust/target/release/ing`) can run for 7+ days without crash, memory leak, or silent data loss.

**Where:** No code changes. Operational verification.

**Steps:**
1. Start ingestor in tmux on su-35: `tmux new-session -d -s ingestor 'make run'`
2. Monitor daily:
   - `ls -lh data/features/$(date +%Y-%m-%d)/` — files growing, new hourly files appearing
   - `make pipeline_status` — row counts increasing
   - `ps aux | grep ing` — RSS memory stable (should stay under 200MB)
3. After 48h, run `make validate_data` — all 7 checks pass
4. After 7 days, confirm:
   - Continuous data with gaps < 5 seconds
   - No corrupted parquet files (schema consistent)
   - Row rate stable at ~30k rows/hr/symbol

**Acceptance criteria:**
- 7 consecutive days of clean data
- Zero corrupted files
- Continuity gaps < 5s between any consecutive rows per symbol
- Memory usage stable (no upward trend in RSS over 7 days)

**Testing:**
```bash
make validate_data              # Full 7-check validation
make scan_schema                # Schema consistency across all files
python3 -c "
from scripts.cluster_pipeline.loader import load_parquet
df = load_parquet('data/features')
print(f'Total rows: {len(df):,}')
print(f'Symbols: {df[\"symbol\"].nunique()}')
"
```

### Task 0.2: Quarantine Corrupted Files

**What:** Move known corrupted parquet files out of the data directory so they don't crash downstream processing.

**Where:** Filesystem operation.

**Steps:**
```bash
mkdir -p data/quarantine
python3 -c "
import pyarrow.parquet as pq
from pathlib import Path
bad = []
for f in Path('data/features').rglob('*.parquet'):
    try:
        pq.read_metadata(f)
    except:
        bad.append(f)
        print(f'CORRUPT: {f}')
print(f'Total corrupt: {len(bad)}')
"
# Move each to data/quarantine/
```

**Acceptance criteria:** `make scan_schema` runs without error on all remaining files.

### Task 0.3: Structural Break Detection (NEW)

**What:** Detect permanent distributional shifts in the raw data that would invalidate clustering across the break.

**File:** `scripts/cluster_pipeline/breaks.py` (new file)

**Function signature:**
```python
def detect_breaks(
    bars: pd.DataFrame,
    columns: List[str],
    method: str = "pelt",  # or "binseg"
    min_segment_length: int = 50,
    penalty: str = "bic",
) -> List[int]:
    """Returns indices of detected structural breaks."""
```

**Dependencies:** `ruptures` library.

**Implementation notes:**
- Run PELT (Pruned Exact Linear Time) change-point detection on the first 5 principal components of the raw bar features
- A break is a permanent shift (not a regime transition) — the feature distribution changes irreversibly
- If breaks detected: log warning, split data at break points, cluster each segment separately
- Minimum segment length prevents false positives on noise

**Acceptance criteria:**
- Correctly detects a mean-shift in synthetic data
- Does not fire on stationary data (false positive rate < 5%)
- Returns empty list for clean continuous data

**Tests:**
```
test_detects_mean_shift:
    Input: 500 bars at mean=0, then 500 bars at mean=3
    Assert: one break detected near index 500 (±10)

test_no_break_stationary:
    Input: 1000 bars from same distribution
    Assert: returns empty list

test_multiple_breaks:
    Input: 3 segments with different means
    Assert: 2 breaks detected

test_min_segment_respected:
    Input: break at index 10 (too short)
    Assert: not detected as break
```

---

## Phase 1: Derivative Generation Engine

The core new capability. Takes bar-aggregated features and produces the enriched derivative feature space.

### Task 1.0: Feature Selection Before Derivation (NEW)

**What:** Select the top-N most informative base features per vector before generating derivatives. This prevents the derivative explosion problem.

**File:** `scripts/cluster_pipeline/derivatives.py` (new file)

**Function signature:**
```python
def select_top_features(
    bars: pd.DataFrame,
    vector: str,
    max_features: int = 15,
    method: str = "variance",  # or "autocorrelation_range"
) -> List[str]:
    """
    Select most informative features from a vector.
    
    Methods:
      - "variance": top N by variance (most variable = most informative for clustering)
      - "autocorrelation_range": top N by range of autocorrelation across lags 1-30
        (features whose persistence varies most across time scales carry regime info)
    """
```

**Rationale:** 191 features × 5 derivatives × 3 windows = 2,800 derivatives. Most are noise. By selecting top 10-15 features first, we get ~150-300 derivatives — manageable and less noisy for PCA.

**Acceptance criteria:**
- Returns at most `max_features` column names
- All returned columns actually exist in bars
- High-variance features are consistently selected
- Constant or near-constant features are never selected

**Tests:**
```
test_max_features_respected:
    Input: bars with 50 columns, max_features=10
    Assert: returns exactly 10 columns

test_constant_excluded:
    Input: bars with one constant column
    Assert: constant column never in output

test_deterministic:
    Run twice, same result

test_variance_method_selects_highest:
    Manual check: compute variance of all columns
    Assert: selected columns are top N by variance
```

### Task 1.1: Temporal Derivative Generator

**What:** A function that takes a DataFrame of bar features and produces temporal derivatives for each feature column.

**File:** `scripts/cluster_pipeline/derivatives.py`

**Function signature:**
```python
def temporal_derivatives(
    bars: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 15, 30],
) -> pd.DataFrame:
```

**Derivatives to compute per column per window:**

| Derivative | Column name pattern | Formula |
|---|---|---|
| Velocity | `{col}_vel` | `f(t) - f(t-1)` |
| Acceleration | `{col}_accel` | `vel(t) - vel(t-1)` |
| Rolling z-score | `{col}_zscore_{w}` | `(f(t) - rolling_mean(w)) / rolling_std(w)` |
| Rolling slope | `{col}_slope_{w}` | OLS slope of `f` over window `w` |
| Rolling volatility | `{col}_rvol_{w}` | `rolling_std(f, w)` |

**Implementation notes:**
- Velocity and acceleration are window-independent (compute once)
- Z-score, slope, rvol are computed for each window in `windows`
- Rolling slope: use vectorized pandas: `df[col].rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])`
- Handle NaN: first `max(windows)` rows will have NaN derivatives. Don't drop them here — the caller decides.
- Handle division by zero in z-score: `np.where(std < 1e-10, 0.0, (val - mean) / std)`
- Output column naming must be deterministic and parseable back to (base_col, derivative_type, window)

**With Task 1.0 applied:** Input is ~10-15 selected columns, not all 191. Output is ~10-15 × (2 + 3×3) = ~110-165 columns. Manageable.

**Acceptance criteria:**
- Output DataFrame has exactly `n_cols * (2 + 3 * len(windows))` columns
- No inf values
- Velocity of a constant column is 0
- Z-score of a linear ramp with no noise has magnitude proportional to slope
- Output length equals input length (NaN-padded, not truncated)

**Tests (write in `scripts/tests/test_derivatives.py`):**

```
test_velocity_constant_is_zero:
    Input: column of all 5.0
    Assert: velocity column is all 0.0 (except first row NaN)

test_velocity_linear_ramp:
    Input: [0, 1, 2, 3, 4, 5]
    Assert: velocity is [NaN, 1, 1, 1, 1, 1]

test_acceleration_linear_is_zero:
    Input: [0, 1, 2, 3, 4, 5]
    Assert: acceleration is [NaN, NaN, 0, 0, 0, 0]

test_acceleration_quadratic:
    Input: [0, 1, 4, 9, 16, 25]  (t^2)
    Assert: acceleration is constant (= 2)

test_zscore_centered:
    Input: random normal data, window=20
    Assert: mean of z-score column ≈ 0 (within 0.1), std ≈ 1 (within 0.2)

test_slope_linear:
    Input: [0, 2, 4, 6, 8, 10], window=3
    Assert: slope is ≈ 2.0 for all rows after warmup

test_no_inf_values:
    Input: column with some zeros and some constant segments
    Assert: no inf in any output column

test_output_shape:
    Input: DataFrame with 5 columns, windows=[5, 15]
    Assert: output has 5 * (2 + 3*2) = 40 columns

test_nan_padding:
    Input: 100-row DataFrame, window=30
    Assert: first 29 rows of rolling derivatives are NaN
    Assert: row 30 onward has no NaN (assuming input has no NaN)
```

### Task 1.2: Cross-Feature Derivative Generator

**What:** Compute ratios, rolling correlations, and divergences between economically meaningful feature pairs.

**File:** `scripts/cluster_pipeline/derivatives.py`

**Function signature:**
```python
def cross_feature_derivatives(
    bars: pd.DataFrame,
    pairs: List[Dict],
    windows: List[int] = [5, 15, 30],
) -> pd.DataFrame:
```

**Pair definition format:**
```python
DEFAULT_CROSS_PAIRS = [
    {"a": "entropy_*_mean",    "b": "volatility_*_mean",      "ops": ["ratio", "corr"]},
    {"a": "orderflow_imbalance_*_mean", "b": "spread_*_mean", "ops": ["ratio"]},
    {"a": "whale_flow_*_sum",  "b": "volume_*_sum",           "ops": ["ratio"]},
    {"a": "toxicity_*_mean",   "b": "illiquidity_*_mean",     "ops": ["ratio", "corr"]},
    {"a": "entropy_*_mean",    "b": "trend_*_mean",           "ops": ["corr", "divergence"]},
]
```

**Note on pair selection (v2):** These pairs are starting hypotheses, not gospel. After first profiling run, check which cross-features load onto top PCA components. Pairs that never load > 0.05 on any top-20 PC should be removed. Pairs can also be auto-discovered by finding column pairs with highest mutual information — add this as a future enhancement, not a blocking requirement.

**Derivative types per pair:**

| Derivative | Column name | Formula |
|---|---|---|
| Ratio | `cross_{a}_{b}_ratio` | `a / (b + eps)`, clipped to [-100, 100] |
| Rolling correlation | `cross_{a}_{b}_corr_{w}` | `rolling_corr(a, b, w)` |
| Divergence | `cross_{a}_{b}_div_{w}` | `zscore(a, w) - zscore(b, w)` |

**Implementation notes:**
- Column matching: use `fnmatch` to resolve glob patterns against actual bar columns
- If a pair can't be resolved (column doesn't exist), skip silently with warning
- Limit to ~15-20 pairs maximum. More pairs = more noise.
- Ratio: `eps=1e-10` denominator, clip to [-100, 100]

**Acceptance criteria:**
- Pairs that can't be resolved are skipped with warning, not error
- Ratio of identical columns is ≈ 1.0
- Correlation of a column with itself is 1.0
- Divergence of identical columns is 0.0
- No inf values

**Tests:**

```
test_ratio_identical_is_one:
    Input: pair where a and b are the same column
    Assert: ratio ≈ 1.0

test_ratio_clipping:
    Input: b column with near-zero values
    Assert: no ratio exceeds [-100, 100]

test_correlation_with_self:
    Input: pair (col, col), window=10
    Assert: correlation is 1.0 for all rows past warmup

test_correlation_independent:
    Input: two independent random columns, window=50, n=1000
    Assert: mean |correlation| < 0.2

test_divergence_identical_is_zero:
    Input: pair (col, col)
    Assert: divergence is 0.0

test_unresolvable_pair_skipped:
    Input: pair referencing nonexistent column
    Assert: no error, warning logged, no output columns for that pair

test_output_deterministic:
    Run twice on same input
    Assert: outputs identical
```

### Task 1.3: Spectral Derivative Generator (EXPERIMENTAL)

**What:** Frequency-domain features for selected columns. **This is experimental — include with kill criterion.**

**File:** `scripts/cluster_pipeline/derivatives.py`

**Kill criterion:** After Phase 2 PCA, check if any spectral-derived column appears in the top-20 PCA loadings with |weight| > 0.05. If none do, remove spectral derivatives from the pipeline permanently. Do not keep dead features.

**Function signature:**
```python
def spectral_derivatives(
    bars: pd.DataFrame,
    columns: List[str],
    window: int = 30,
) -> pd.DataFrame:
```

**Derivatives per column (limit to 3-5 columns max, not all features):**

| Derivative | Column name | Formula |
|---|---|---|
| Low-frequency energy | `{col}_spec_low` | Sum of FFT power in bins 0..N//5 |
| High-frequency energy | `{col}_spec_high` | Sum of FFT power in bins 4*N//5..N |
| Spectral ratio | `{col}_spec_ratio` | low / (high + eps) |
| Dominant period | `{col}_spec_period` | `window / argmax(power[1:])` — skip DC component |

**Implementation notes:**
- Apply ONLY to: spread, volatility, and entropy (3 features). Not to everything.
- Use rolling window FFT: for each row, take the last `window` values, apply `np.fft.rfft`, compute power spectrum `|fft|^2`
- Dominant period: if max power < 2 × mean power, set to NaN (no dominant frequency)
- Total output: 3 features × 4 derivatives = 12 columns. Minimal overhead.

**Acceptance criteria:**
- Pure sine wave input recovers correct period
- White noise input has spectral ratio ≈ 1.0
- Constant input has all spectral derivatives = 0 (or NaN for period)
- No inf values

**Tests:**

```
test_sine_wave_period:
    Input: sin(2*pi*t/10) sampled at 30 points
    Assert: dominant period ≈ 10 (within ±1)

test_sine_wave_low_energy:
    Input: sin(2*pi*t/20) — low frequency
    Assert: spec_low > spec_high

test_white_noise_ratio:
    Input: np.random.randn(500)
    Assert: mean spectral ratio between 0.5 and 2.0

test_constant_input:
    Input: all 5.0
    Assert: low, high energy ≈ 0, ratio = 0 or NaN

test_output_length:
    Assert: output length == input length (NaN-padded for first window-1 rows)
```

### Task 1.4: Derivative Orchestrator

**What:** A single entry point that runs all derivative generators and returns the combined derivative DataFrame.

**File:** `scripts/cluster_pipeline/derivatives.py`

**Function signature:**
```python
def generate_derivatives(
    bars: pd.DataFrame,
    vector: str,
    max_base_features: int = 15,
    temporal_windows: List[int] = [5, 15, 30],
    spectral_window: int = 30,
    spectral_columns: Optional[List[str]] = None,
    cross_pairs: Optional[List[Dict]] = None,
    include_spectral: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
```

**Returns:**
- DataFrame with all derivative columns (temporal + cross + spectral)
- Metadata dict: `{"n_base_features": int, "base_features": List[str], "n_temporal": int, "n_cross": int, "n_spectral": int, "n_total": int, "warmup_rows": int}`

**Steps:**
1. Run `select_top_features(bars, vector, max_features=max_base_features)` — Task 1.0
2. Run `temporal_derivatives()` on selected columns — Task 1.1
3. Run `cross_feature_derivatives()` with pair config — Task 1.2
4. If `include_spectral`: run `spectral_derivatives()` on 3 selected columns — Task 1.3
5. Concatenate all derivative DataFrames column-wise
6. Report metadata

**Expected output size:** ~15 base × 11 temporal + ~15 cross + 12 spectral = ~192 total derivatives. This is tractable for PCA without the noise explosion.

**Acceptance criteria:**
- Calling with just `(bars, vector)` works with sensible defaults
- Metadata accurately reports column counts
- Column names are unique
- Total derivative count < 500 (not the 3000 from v1)

**Tests:**

```
test_orchestrator_default_config:
    Input: aggregate_bars(df, "15min"), vector="entropy"
    Assert: returns DataFrame with 100-300 columns, metadata populated

test_orchestrator_column_uniqueness:
    Assert: len(result.columns) == len(set(result.columns))

test_orchestrator_metadata_accuracy:
    Assert: metadata["n_total"] == len(result.columns)

test_orchestrator_reasonable_size:
    Assert: metadata["n_total"] < 500

test_orchestrator_no_nan_after_warmup:
    Assert: result.iloc[metadata["warmup_rows"]:].isna().sum().sum() < result.shape[1] * 0.01
```

---

## Phase 2: Dimensionality Reduction Pipeline

### Task 2.1: Variance and Correlation Filtering

**What:** Remove near-constant and redundant derivative columns before PCA.

**File:** `scripts/cluster_pipeline/reduction.py` (new file)

**Function signature:**
```python
def filter_derivatives(
    X: pd.DataFrame,
    variance_percentile: float = 10.0,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Dict]:
```

**Returns:**
- Filtered DataFrame
- Report dict: `{"n_input": int, "n_after_variance": int, "n_after_correlation": int, "dropped_variance": List[str], "dropped_correlation": List[str]}`

**Steps:**
1. Compute variance of each column
2. Drop columns below the `variance_percentile`-th percentile of variance
3. Compute pairwise correlation matrix of surviving columns
4. For each pair with |corr| > threshold, drop the column with lower variance
5. Use a greedy algorithm: sort pairs by |corr| descending, drop one at a time, recheck

**Acceptance criteria:**
- No surviving column pair has |corr| > threshold
- The highest-variance columns are never dropped
- Deterministic: same input → same output
- If all columns are identical, only one survives

**Tests:**

```
test_constant_column_dropped:
    Input: DataFrame with one constant column, rest random
    Assert: constant column in dropped_variance list

test_identical_columns_deduplicated:
    Input: DataFrame with col_a, col_b = col_a.copy()
    Assert: only one survives

test_independent_columns_preserved:
    Input: DataFrame of independent random columns
    Assert: all columns survive

test_greedy_triplet:
    Input: col_a, col_b (corr=0.99), col_c (corr with a=0.96, corr with b=0.97)
    Assert: only one of the three is dropped

test_report_counts_consistent:
    Assert: n_input == n_after_variance + len(dropped_variance)
```

### Task 2.2: PCA with Saved Basis (Regularized)

**What:** PCA reduction with configurable variance threshold, using Ledoit-Wolf shrinkage when `n_samples < 2 * n_features`.

**File:** `scripts/cluster_pipeline/reduction.py`

**Function signature:**
```python
@dataclass
class PCAResult:
    X_reduced: np.ndarray           # (n_samples, n_components)
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray          # (n_components, n_features)
    mean: np.ndarray                # (n_features,)
    std: np.ndarray                 # (n_features,) — for standardization
    column_names: List[str]
    loadings: Dict[int, List[Tuple[str, float]]]  # top 10 loadings per PC
    regularized: bool               # whether Ledoit-Wolf was used

def pca_reduce(
    X: np.ndarray,
    column_names: List[str],
    variance_threshold: float = 0.95,
    max_components: int = 50,
) -> PCAResult:
```

**v2 correction: Regularization.** When `n_samples < 2 * n_features`, use `sklearn.covariance.LedoitWolf` to estimate the covariance matrix before PCA. This prevents unstable components in the micro-state discovery where per-regime data is small.

**v2 correction: Lower max_components.** Changed from 80 to 50. With ~200 derivatives (after Task 1.0 feature selection), 50 components capturing 95% variance is generous. 80 was designed for the 3000-derivative explosion we're now avoiding.

**Steps:**
1. Standardize: subtract mean, divide by std (save both)
2. If `n_samples < 2 * n_features`: use Ledoit-Wolf covariance, set `regularized=True`
3. Fit PCA via eigendecomposition of covariance
4. Select n_components: smallest k where cumulative variance ≥ threshold, capped at max_components
5. Project: `X_reduced = (X - mean) / std @ components.T`
6. Compute loadings: for each PC, top 10 original columns by |weight|
7. **Spectral kill check:** If `include_spectral` was True, check if any `_spec_` column appears in top-20 loadings of any top-10 PC. Log result.

**Acceptance criteria:**
- Reconstruction MSE < (1 - variance_threshold) × total_variance
- Loadings are sorted by |weight| descending
- n_components ≤ max_components
- Regularized flag correctly set based on sample/feature ratio
- Spectral kill check result logged

**Tests:**

```
test_reconstruction_error:
    Input: random data, variance_threshold=0.95
    Assert: reconstruction MSE < 5% of total variance

test_perfect_reconstruction:
    Input: 10-column data, variance_threshold=1.0
    Assert: n_components = 10, MSE ≈ 0

test_low_rank_data:
    Input: 100 columns but only 3 independent
    Assert: n_components ≤ 5

test_regularization_triggered:
    Input: 50 samples, 100 features
    Assert: regularized == True

test_regularization_not_triggered:
    Input: 500 samples, 100 features
    Assert: regularized == False

test_loadings_sorted:
    For each PC: assert loadings are descending by |weight|

test_projection_on_new_data:
    Split data 80/20. Fit PCA on 80%.
    Project 20% using saved mean + std + components.
    Assert: no shape mismatch

test_deterministic:
    Run twice, assert X_reduced identical
```

### Task 2.3: Save/Load PCA Basis

**What:** Serialize and deserialize PCAResult for online detector reuse.

**File:** `scripts/cluster_pipeline/reduction.py`

**Function signature:**
```python
def save_pca_basis(result: PCAResult, path: Path) -> None:
def load_pca_basis(path: Path) -> PCAResult:
```

**Format:** NumPy `.npz` for arrays + JSON sidecar for metadata/column_names/loadings.

**Acceptance criteria:**
- Round-trip: `save → load → project` produces identical results
- File size < 10MB

**Tests:**

```
test_roundtrip:
    Fit PCA, save, load, project same data
    Assert: np.allclose(original, loaded)

test_metadata_preserved:
    Assert: column_names, n_components, regularized flag preserved
```

### Task 2.4: Full Reduction Pipeline

**What:** Single entry point: raw derivative DataFrame → reduced numpy array.

**File:** `scripts/cluster_pipeline/reduction.py`

**Function signature:**
```python
def reduce(
    derivatives: pd.DataFrame,
    variance_percentile: float = 10.0,
    correlation_threshold: float = 0.95,
    pca_variance: float = 0.95,
    pca_max_components: int = 50,
) -> Tuple[np.ndarray, PCAResult, Dict]:
```

**Returns:** `(X_reduced, pca_result, filter_report)`

**Tests:**

```
test_full_pipeline_integration:
    bars → generate_derivatives → reduce
    Assert: X_reduced.shape[1] < n_derivatives

test_pipeline_with_real_data:
    Use actual data from data/features/
    Assert: completes without error
    Assert: n_components between 5 and 50
```

---

## Phase 3: Hierarchical State Discovery

### Task 3.0: Structure Existence Test (NEW)

**What:** Before clustering, verify that non-trivial structure actually exists in the data. Don't force GMM on uniform noise.

**File:** `scripts/cluster_pipeline/hierarchy.py` (new file)

**Function signature:**
```python
@dataclass
class StructureTest:
    hopkins_statistic: float     # > 0.7 suggests clusters exist
    dip_test_p: float           # < 0.05 suggests multimodality on PC1
    has_structure: bool         # True if either test passes
    recommendation: str         # "proceed", "weak_structure", "no_structure"

def test_structure_existence(
    X_reduced: np.ndarray,
    significance: float = 0.05,
) -> StructureTest:
```

**Steps:**
1. Compute Hopkins statistic on PCA-reduced data (tests clustering tendency)
2. Run Hartigan dip test on first principal component (tests unimodality)
3. If Hopkins > 0.7 OR dip_test p < 0.05: structure likely exists
4. If both fail: recommend "no_structure" — stop pipeline, don't waste time on clustering

**Decision logic:**
- Hopkins > 0.7 AND dip p < 0.05 → "proceed" (strong evidence)
- Hopkins > 0.7 OR dip p < 0.05 → "weak_structure" (proceed with caution)
- Both fail → "no_structure" (stop, collect more data, or try different features)

**Acceptance criteria:**
- Correctly identifies clustered synthetic data as "proceed"
- Correctly identifies uniform random data as "no_structure"
- Fast: < 2 seconds for 2000 bars × 50 dimensions

**Tests:**

```
test_clustered_data:
    Input: 3 well-separated Gaussians
    Assert: has_structure == True, recommendation == "proceed"

test_uniform_data:
    Input: uniform random in hypercube
    Assert: has_structure == False, recommendation == "no_structure"

test_marginal_data:
    Input: 2 overlapping Gaussians (silhouette ~0.2)
    Assert: recommendation in ["weak_structure", "proceed"]
```

### Task 3.1: Macro Regime Discovery

**What:** Discover 2-4 broad market regimes using only slow-moving derivatives.

**File:** `scripts/cluster_pipeline/hierarchy.py`

**Function signature:**
```python
@dataclass
class RegimeResult:
    labels: np.ndarray
    k: int
    pca_result: PCAResult
    gmm_params: Dict
    quality: QualityReport
    stability: StabilityReport
    sweep: SweepResult
    centroid_profiles: pd.DataFrame
    self_transition_rate: float
    durations: Dict[int, List[int]]

def discover_macro_regimes(
    derivatives: pd.DataFrame,
    autocorrelation_threshold: float = 0.7,
    k_range: range = range(2, 6),
    pca_variance: float = 0.95,
    n_bootstrap: int = 30,
    block_size: int = 15,
    random_state: int = 42,
) -> RegimeResult:
```

**v2 correction: Data-driven slow/fast split.** Instead of filtering by column name convention (window ≥ 30), compute autocorrelation at lag=5 for each derivative column. Columns with autocorrelation > `autocorrelation_threshold` are "slow" (persistent features suitable for regime detection). Columns below are "fast."

This is better because:
- Velocity can be slow (trending features have persistent velocity)
- A rolling z-score with window=30 can still be fast (noisy feature)
- The split reflects actual temporal behavior, not naming convention

**v2 correction: Block bootstrap.** Use block bootstrap with `block_size` contiguous bars instead of random resampling. This preserves temporal autocorrelation and gives honest stability estimates.

**Steps:**
1. Compute autocorrelation at lag=5 for each derivative column
2. Select columns with autocorrelation > threshold as "slow"
3. Run reduction pipeline on slow derivatives only
4. **Run structure existence test (Task 3.0).** If "no_structure", return early with warning.
5. Run k_sweep over k_range
6. Select best k by BIC
7. Fit final GMM at best k
8. Compute quality with **block bootstrap** (not random)
9. Compute duration runs per regime
10. Build centroid profiles

**Acceptance criteria:**
- Self-transition rate > 0.8
- Block bootstrap ARI > 0.6
- At least 10% of bars in smallest regime
- Structure test passed before clustering

**Tests:**

```
test_synthetic_two_regimes:
    Generate 500 bars from N(0,I) and 500 from N(3,I), interleaved in blocks of 50
    Assert: discovers k=2, labels mostly correct
    Assert: STR > 0.9

test_autocorrelation_split:
    Create columns with known autocorrelation (one persistent, one noisy)
    Assert: persistent column selected as "slow", noisy excluded

test_block_bootstrap_preserves_order:
    Verify bootstrap samples are contiguous blocks, not random indices

test_minimum_cluster_size:
    Assert: smallest cluster ≥ 10% of total bars

test_no_structure_early_exit:
    Input: uniform random data
    Assert: returns early with warning, no clustering attempted

test_duration_computation:
    Labels: [0,0,0,1,1,0,0,0,0,1]
    Assert: durations[0] = [3, 4], durations[1] = [2, 1]
```

### Task 3.2: Micro State Discovery (Per Regime)

**What:** Within each macro regime, discover 2-5 finer-grained states using all derivatives.

**File:** `scripts/cluster_pipeline/hierarchy.py`

**Function signature:**
```python
@dataclass
class MicroStateResult:
    regime_id: int
    labels: np.ndarray
    k: int
    pca_result: PCAResult       # separate PCA per regime (regularized if needed)
    gmm_params: Dict
    quality: QualityReport
    stability: StabilityReport
    sweep: SweepResult
    centroid_profiles: pd.DataFrame

def discover_micro_states(
    derivatives: pd.DataFrame,
    macro_labels: np.ndarray,
    regime_id: int,
    k_range: range = range(2, 6),
    pca_variance: float = 0.95,
    n_bootstrap: int = 30,
    block_size: int = 10,
    random_state: int = 42,
) -> Optional[MicroStateResult]:  # Returns None if too few bars or no structure
```

**v2 correction: Regularized PCA.** Per-regime subsets may have few samples. PCA in Task 2.2 already handles this via Ledoit-Wolf when `n_samples < 2 * n_features`. This naturally applies here.

**v2 correction: Minimum sample check + structure test.** If a regime has < 100 bars, skip with warning. If structure test fails on the regime subset, skip — don't force clusters.

**Steps:**
1. Subset derivatives to rows where `macro_labels == regime_id`
2. If subset < 100 bars, return None with warning
3. Run full reduction pipeline on this subset (PCA fitted per regime)
4. Run structure existence test. If "no_structure", return None.
5. k_sweep, select best k, fit GMM
6. Quality + stability (block bootstrap)
7. Build centroid profiles

**Acceptance criteria:**
- Returns None for small regimes or regimes without structure
- Per-regime PCA basis is different from macro PCA basis
- Block bootstrap ARI > 0.5

**Tests:**

```
test_subset_correctness:
    Assert: only bars from specified regime used

test_separate_pca_per_regime:
    Run on regime 0 and regime 1
    Assert: PCA components are different

test_small_regime_returns_none:
    Input: regime with 50 bars
    Assert: returns None, warning logged

test_no_structure_returns_none:
    Input: uniform data within regime
    Assert: returns None

test_micro_within_macro:
    Generate: 2 macro clusters, each with 2 sub-clusters
    Assert: micro finds k=2 within each
```

### Task 3.3: Hierarchical Label Assembly

**What:** Combine macro + micro labels into unified system.

**File:** `scripts/cluster_pipeline/hierarchy.py`

**Function signature:**
```python
@dataclass
class HierarchicalLabels:
    macro_labels: np.ndarray
    micro_labels: np.ndarray          # global micro IDs
    composite_labels: np.ndarray      # "R0_S2"
    n_macro: int
    n_micro_per_regime: Dict[int, int]
    n_micro_total: int
    label_map: Dict[int, Tuple[int, int]]  # global_micro -> (regime, local_state)

def assemble_hierarchy(
    macro_result: RegimeResult,
    micro_results: Dict[int, Optional[MicroStateResult]],
) -> HierarchicalLabels:
```

**v2 note:** `micro_results` values can be None (regime too small or no structure). For regimes without micro states, assign all bars in that regime to a single micro state (regime itself is the finest granularity).

**Acceptance criteria:**
- Every bar has exactly one macro and one micro label
- Global micro IDs are contiguous
- Regimes without micro results get a single micro state

**Tests:**

```
test_global_ids_contiguous:
    Assert: set(micro_labels) == set(range(n_micro_total))

test_regime_without_micros:
    Input: regime 1 has None micro result
    Assert: all bars in regime 1 get same micro ID

test_composite_format:
    Assert: all labels match "R{int}_S{int}"

test_label_map_invertible:
    For each bar: label_map[micro_labels[i]] == (macro_labels[i], local_state)
```

### Task 3.4: Full Hierarchy Pipeline

**What:** Single entry point: raw data → ProfilingResult.

**File:** `scripts/cluster_pipeline/hierarchy.py`

**Function signature:**
```python
@dataclass
class ProfilingResult:
    hierarchy: HierarchicalLabels
    macro: RegimeResult
    micros: Dict[int, Optional[MicroStateResult]]
    derivatives_meta: Dict
    reduction_report: Dict
    bars: pd.DataFrame
    derivative_columns: List[str]
    breaks_detected: List[int]       # structural break indices (NEW)
    structure_test: StructureTest     # pre-clustering structure test (NEW)

def profile(
    df: pd.DataFrame,
    vector: str,
    timeframe: str,
    max_base_features: int = 15,
    temporal_windows: List[int] = [5, 15, 30],
    autocorrelation_threshold: float = 0.7,
    macro_k_range: range = range(2, 6),
    micro_k_range: range = range(2, 6),
    pca_variance: float = 0.95,
    include_spectral: bool = True,
    random_state: int = 42,
) -> ProfilingResult:
```

**Steps:**
1. Aggregate bars
2. Detect structural breaks (Task 0.3). If breaks found, use only the longest continuous segment.
3. Generate derivatives (Task 1.4, with feature selection)
4. Reduce (Task 2.4)
5. Test structure existence (Task 3.0)
6. Discover macro regimes (Task 3.1)
7. Discover micro states per regime (Task 3.2)
8. Assemble hierarchy (Task 3.3)

**Tests:**

```
test_end_to_end_smoke:
    Load real data, run profile("orderflow", "15min")
    Assert: returns ProfilingResult

test_structural_break_handled:
    Input: data with known break
    Assert: breaks_detected is non-empty, profiling uses longest segment

test_no_structure_graceful:
    Input: random noise data
    Assert: structure_test.has_structure == False, pipeline stops gracefully
```

---

## Phase 4: Transition Modeling

### Task 4.1: Empirical Transition Matrix

**What:** Compute transition probabilities between states at both hierarchical levels.

**File:** `scripts/cluster_pipeline/transitions.py` (new file)

**Function signature:**
```python
@dataclass
class TransitionModel:
    matrix: np.ndarray                # (k, k) row-stochastic
    state_names: List[str]
    self_transition_rates: Dict[int, float]
    row_entropy: Dict[int, float]
    most_likely_successor: Dict[int, int]
    mean_durations: Dict[int, float]
    duration_distributions: Dict[int, np.ndarray]

def empirical_transitions(
    labels: np.ndarray,
    state_names: Optional[List[str]] = None,
) -> TransitionModel:
```

**Steps:**
1. Count transitions: `T[i,j] = count(label[t]=i AND label[t+1]=j)`
2. Normalize rows: `T[i,:] /= sum(T[i,:])`
3. Self-transition rate: diagonal of T
4. Row entropy: `-sum(T[i,:] * log(T[i,:] + eps))`
5. Most likely successor: argmax of off-diagonal elements per row
6. Duration distributions: collect run lengths per state

**Acceptance criteria:**
- Each row sums to 1.0 (±1e-10)
- Duration mean ≈ 1 / (1 - self_transition_rate) for geometric distribution (report deviation)

**Tests:**

```
test_row_stochastic:
    Assert: all rows sum to 1.0

test_perfect_persistence:
    Labels: [0,0,0,0,0,1,1,1,1,1]
    Assert: T[0,0] = 0.8, T[0,1] = 0.2

test_alternating:
    Labels: [0,1,0,1,0,1]
    Assert: T[0,1] = 1.0, T[1,0] = 1.0

test_duration_distribution:
    Labels: [0,0,0,1,1,0,0]
    Assert: durations[0] = [3, 2], durations[1] = [2]

test_single_state:
    Labels: [0,0,0,0,0]
    Assert: matrix is [[1.0]], no crash
```

### Task 4.2: HMM Fitting (Optional)

**What:** Fit HMM for temporally smoothed state estimation.

**File:** `scripts/cluster_pipeline/transitions.py`

**Function signature:**
```python
@dataclass
class HMMResult:
    model: Any
    smoothed_labels: np.ndarray
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray
    log_likelihood: float
    bic: float
    convergence: bool

def fit_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int = 100,
    random_state: int = 42,
) -> HMMResult:
```

**When to skip:** If any regime has fewer than 200 bars, skip HMM.

**Acceptance criteria:**
- HMM converges
- Smoothed labels ARI > 0.5 vs GMM labels
- Transition matrix is diagonal-dominant

**Tests:**

```
test_hmm_on_synthetic:
    Generate 2-state HMM data with known parameters
    Assert: recovered parameters close to true

test_hmm_agrees_with_gmm:
    Assert: ARI(gmm, hmm) > 0.5

test_small_data_warning:
    Input: 50 bars
    Assert: graceful return or warning
```

---

## Phase 5: State Characterization

### Task 5.1: Centroid Profiling

**What:** For each state, identify defining features.

**File:** `scripts/cluster_pipeline/characterize.py` (new file)

**Function signature:**
```python
@dataclass
class StateProfile:
    state_id: int
    regime_id: int
    local_state_id: int
    n_bars: int
    centroid: Dict[str, float]
    top_elevated: List[Tuple[str, float]]
    top_suppressed: List[Tuple[str, float]]
    duration_mean: float
    duration_median: float
    duration_p90: float
    successor_probs: Dict[int, float]

def characterize_states(
    derivatives: pd.DataFrame,
    hierarchy: HierarchicalLabels,
    transition_model: TransitionModel,
) -> Dict[int, StateProfile]:
```

**Tests:**
```
test_centroid_is_mean:
    Assert: profile centroid == manual mean of derivatives where label==state

test_elevated_suppressed_disjoint:
    Assert: no overlap between top_elevated and top_suppressed

test_all_states_profiled:
    Assert: len(profiles) == n_micro_total
```

### Task 5.2: Entry and Exit Signatures

**What:** Average derivative trajectory before state entry/exit.

**File:** `scripts/cluster_pipeline/characterize.py`

**Function signature:**
```python
@dataclass
class TransitionSignature:
    state_id: int
    entry_trajectory: pd.DataFrame
    exit_trajectory: pd.DataFrame
    entry_count: int
    exit_count: int
    entry_std: pd.DataFrame
    exit_std: pd.DataFrame

def compute_signatures(
    derivatives: pd.DataFrame,
    labels: np.ndarray,
    state_id: int,
    lookback: int = 5,
) -> Optional[TransitionSignature]:  # None if < 5 events
```

**Tests:**
```
test_entry_count:
    Labels: [1,0,0,1,0,0,1,0]
    Assert: state 0 entry_count == 2

test_insufficient_events_returns_none:
    Input: state with only 2 entries, min_events=5
    Assert: returns None
```

### Task 5.3: Forward Return Profiling (Multi-Horizon)

**What:** Full return distribution at MULTIPLE horizons, not just 1-bar.

**File:** `scripts/cluster_pipeline/characterize.py`

**v2 correction:** Always test at multiple horizons including the state's mean duration. A macro regime shouldn't be evaluated on 1-bar returns — it should be evaluated at regime-duration scale.

**Function signature:**
```python
@dataclass
class ReturnProfile:
    state_id: int
    horizons: Dict[int, Dict]  # horizon_bars -> {mean, median, std, skew, kurtosis, p5, p95, sharpe, n}

def return_profile(
    labels: np.ndarray,
    prices: np.ndarray,
    state_id: int,
    horizons: List[int] = [1, 5, 10, 20],
    mean_duration: Optional[int] = None,  # auto-add state's mean duration as horizon
) -> ReturnProfile:
```

**v2 note:** If `mean_duration` provided, add it to horizons automatically. This ensures each state is evaluated at its natural timescale, not just arbitrary fixed horizons.

**Tests:**
```
test_return_computation:
    Assert: log(price[t+h] / price[t]) computed correctly

test_multiple_horizons:
    Assert: result has entries for all requested horizons

test_mean_duration_added:
    Input: mean_duration=8, horizons=[1, 5]
    Assert: result has horizons [1, 5, 8]
```

---

## Phase 6: Validation Framework

### Task 6.1: Unified Quality Gate Runner

**What:** Run Q1/Q2/Q3 gates with corrected methodology.

**File:** `scripts/cluster_pipeline/validate.py` (new file)

**Function signature:**
```python
@dataclass
class ValidationVerdict:
    q1_structural: Dict[str, Any]
    q2_predictive: Dict[str, Any]     # NOW includes multi-horizon results
    q3_operational: Dict[str, Any]
    overall: str                       # "GO", "PIVOT", "COLLECT", "DROP"
    per_state_verdicts: Dict[int, str]
    summary: str

def validate(
    profiling_result: ProfilingResult,
    prices: np.ndarray,
    q1_thresholds: Optional[Dict] = None,
    q2_thresholds: Optional[Dict] = None,
    q3_thresholds: Optional[Dict] = None,
) -> ValidationVerdict:
```

**Default thresholds:**
```python
Q1_DEFAULTS = {"silhouette": 0.25, "block_bootstrap_ari": 0.6, "temporal_ari": 0.5}
Q2_DEFAULTS = {"kruskal_p": 0.05, "eta_squared": 0.01, "min_sharpe": 0.3, "any_horizon": True}
Q3_DEFAULTS = {"macro_str": 0.8, "micro_str": 0.5, "min_duration": 3, "entry_lead": 1}
```

**v2 correction: Q2 `any_horizon` flag.** Q2 passes if Kruskal-Wallis is significant at ANY horizon in [1, 5, 10, 20, mean_duration], not just horizon=1. This fixes the horizon mismatch problem where persistent regimes don't predict 1-bar returns but do predict 20-bar returns.

**Decision logic:**
- Q1 fail → "DROP"
- Q1 pass + Q2 fail at all horizons → "COLLECT"
- Q1 pass + Q2 pass + Q3 fail → "PIVOT"
- All pass → "GO"

**Tests:**

```
test_all_pass_is_go:
    Assert: verdict == "GO"

test_q1_fail_is_drop:
    Mock: silhouette = 0.1
    Assert: verdict == "DROP"

test_q2_any_horizon:
    Mock: horizon=1 fails, horizon=10 passes
    Assert: Q2 passes (any_horizon=True)

test_q2_all_horizons_fail:
    Mock: all horizons insignificant
    Assert: verdict == "COLLECT"
```

### Task 6.2: Cross-Symbol Consistency Check

**What:** Run profiling independently per symbol and measure agreement.

**File:** `scripts/cluster_pipeline/validate.py`

**v2 correction: Define disagreement behavior.** When symbols disagree on regime, the system should:
- Report per-symbol states
- Compute majority-vote consensus regime
- If no majority (3 symbols, 3 different states), label as "mixed/uncertain"
- Track "agreement rate over time" as a signal quality metric

**Function signature:**
```python
@dataclass
class CrossSymbolResult:
    agreement_matrix: np.ndarray      # (n_symbols, n_symbols) pairwise ARI
    mean_agreement: float
    above_random: bool
    consensus_labels: np.ndarray      # majority-vote labels at aligned timestamps
    disagreement_rate: float          # fraction of bars with no majority

def cross_symbol_consistency(
    df: pd.DataFrame,
    vector: str,
    timeframe: str,
    symbols: List[str] = ["BTC", "ETH", "SOL"],
) -> CrossSymbolResult:
```

**Tests:**
```
test_identical_labels_ari_one:
    Assert: ARI = 1.0

test_random_labels_low_ari:
    Assert: mean ARI < 0.1

test_consensus_majority:
    Labels: BTC=0, ETH=0, SOL=1
    Assert: consensus = 0 (majority)

test_consensus_disagreement:
    Labels: BTC=0, ETH=1, SOL=2
    Assert: labeled as "uncertain", disagreement_rate increases
```

---

## Phase 7: Online Regime Detector

### Task 7.1: Rolling Derivative Buffer

**What:** Maintain fixed-size rolling buffer, compute derivatives incrementally.

**File:** `scripts/cluster_pipeline/online.py` (new file)

**Function signature:**
```python
class DerivativeBuffer:
    def __init__(self, max_window: int = 30, vector: str = "orderflow",
                 temporal_windows: List[int] = [5, 15, 30]):
        ...

    def update(self, bar: pd.Series) -> Optional[np.ndarray]:
        """Push bar, return derivative vector if buffer full, else None."""
        ...

    def reset(self) -> None:
        """Clear buffer (after gap or break detection)."""
        ...
```

**Tests:**
```
test_warmup_returns_none:
    Push 29 bars with max_window=30
    Assert: all None

test_warmup_plus_one:
    Push 30th bar
    Assert: returns vector

test_matches_batch:
    Push 100 bars sequentially, compare to batch generate_derivatives output
    Assert: last row matches (atol=1e-6)

test_memory_constant:
    Push 10000 bars
    Assert: internal size == max_window
```

### Task 7.2: Online Classifier with Drift Detection (UPDATED)

**What:** Classify derivative vectors with built-in drift monitoring.

**File:** `scripts/cluster_pipeline/online.py`

**v2 correction: Drift detection is built into the classifier, not a separate system.**

**Function signature:**
```python
@dataclass
class StateEstimate:
    macro_regime: int
    macro_confidence: float
    micro_state: int
    micro_confidence: float
    composite_label: str
    time_in_state: int
    likely_next_state: int
    transition_prob: float
    all_probabilities: Dict[int, float]
    drift_warning: bool              # NEW: True if confidence consistently low
    rolling_log_likelihood: float    # NEW: rolling avg of GMM log-likelihood

class OnlineClassifier:
    def __init__(self, profiling_result_path: Path):
        """Load saved PCA basis, GMM params, transition matrix, training log-likelihood stats."""
        ...

    def classify(self, derivative_vector: np.ndarray) -> StateEstimate:
        """Classify + track drift."""
        ...

    @property
    def drift_detected(self) -> bool:
        """True if rolling log-likelihood < training 10th percentile for 20+ bars."""
        ...
```

**Drift detection logic:**
- Track rolling GMM log-likelihood over last 50 classifications
- Compare to training-time distribution (save 10th, 50th percentile during profiling)
- If rolling avg drops below training 10th percentile for 20+ consecutive bars → `drift_warning = True`
- If drift persists for 100+ bars → log "DRIFT: re-profiling recommended"

**Tests:**
```
test_latency:
    Assert: mean < 1ms, p99 < 5ms

test_probabilities_sum_to_one:
    Assert: sum ≈ 1.0

test_matches_offline:
    Assert: ARI > 0.95 vs offline labels

test_drift_detection:
    Feed 100 bars from training distribution (no drift)
    Then feed 50 bars from shifted distribution
    Assert: drift_warning becomes True

test_no_false_drift:
    Feed 500 bars from training distribution
    Assert: drift_warning stays False
```

### Task 7.3: Detector Persistence (Save/Load)

**What:** Save/load everything for online classification.

**File:** `scripts/cluster_pipeline/online.py`

**Saved artifacts:**
- `pca_macro.npz`
- `pca_micro_{regime_id}.npz`
- `gmm_macro.json`
- `gmm_micro_{regime_id}.json`
- `transitions.json`
- `config.json` — vector, windows, column names, feature selection
- `metadata.json` — training data range, n_bars, verdict
- `training_stats.json` — log-likelihood percentiles for drift detection (NEW)

**Tests:**
```
test_roundtrip:
    Profile → save → load → classify
    Assert: identical labels

test_artifact_completeness:
    Assert: all expected files exist

test_drift_stats_saved:
    Assert: training_stats.json contains log_likelihood_p10 and log_likelihood_p50
```

---

## Phase 8: Visualization and Reporting

### Task 8.1: Hierarchical State Visualization

**What:** Core visual suite.

**File:** `scripts/visualize_profiling.py` (extend existing)

**Plots:**

1. **Hierarchy overview** (`plot_hierarchy_overview`): macro regime timeline + micro state timeline
2. **Derivative PCA scatter** (`plot_derivative_pca`): colored by state, loading arrows
3. **Entry/exit signature heatmaps** (`plot_signatures`)
4. **Forward return violin plots** (`plot_return_violins`): one per state per horizon
5. **Transition diagram** (`plot_transition_graph`): directed graph
6. **Drift monitor** (`plot_drift_dashboard`): rolling confidence + drift flags
7. **Structure test visualization** (`plot_structure_test`): Hopkins score + dip test result (NEW)

**Acceptance criteria:**
- Each plot saves to specified path without error
- Consistent color scheme across all plots

### Task 8.2: Automated Report Generator

**What:** Structured markdown report.

**File:** `scripts/cluster_pipeline/report.py` (new file)

**Report structure:**
```
# Profiling Report — {vector}@{timeframe}
## Data Summary
## Structural Breaks (if any)
## Feature Selection (which base features, why)
## Derivative Space (n_total, spectral kill check result)
## Structure Test (Hopkins, dip test, recommendation)
## Macro Regimes (k, centroids, STR, durations)
## Micro States (per regime)
## Transition Structure
## Predictive Quality (ALL horizons, not just 1-bar)
## Cross-Symbol Consistency
## Validation Verdict (Q1/Q2/Q3, overall)
## Drift Baseline (training log-likelihood stats)
## Recommendations
```

---

## Phase 9: Integration Testing

### Task 9.1: Synthetic Data Integration Test

**File:** `scripts/tests/test_integration_profiling.py`

```
test_integration_synthetic:
    Generate synthetic data with 2 macro regimes, 2 micro states each
    Run full profile() pipeline
    Assert: macro k == 2, ARI > 0.7

test_integration_no_structure:
    Generate uniform random data
    Run profile()
    Assert: structure_test.has_structure == False, pipeline stops gracefully

test_integration_structural_break:
    Generate data with mean shift at midpoint
    Assert: break detected, longest segment used

test_integration_real_data:
    Load actual data, run profile("orderflow", "15min")
    Assert: completes without crash

test_integration_save_load_classify:
    Profile → save → load → classify all bars
    Assert: online matches offline

test_integration_drift_detection:
    Profile on first half of data
    Classify second half (potentially different distribution)
    Assert: drift detection fires if distribution shifted
```

### Task 9.2: Performance Benchmarks

```
test_derivative_generation_time:
    Input: 2000 bars, 15 features (after selection), 3 windows
    Assert: < 5 seconds

test_reduction_time:
    Input: 2000 bars, 200 derivative columns
    Assert: < 3 seconds

test_macro_discovery_time:
    Input: 2000 bars post-reduction
    Assert: < 30 seconds (including block bootstrap)

test_full_pipeline_time:
    Input: 2000 bars
    Assert: full profile() < 90 seconds

test_online_classify_throughput:
    Assert: > 10,000 classifications per second
```

---

## Dependency Graph

```
Phase 0 (data collection + break detection) ────── runs in parallel ──────────
                                                                              │
Phase 1 (derivatives)                                                         │
  Task 1.0 (feature selection) — NEW                                          │
  Task 1.1 (temporal)                                                         │
  Task 1.2 (cross-feature)                                                    │
  Task 1.3 (spectral — EXPERIMENTAL)                                          │
  Task 1.4 (orchestrator) ← depends on 1.0, 1.1, 1.2, 1.3                   │
       │                                                                      │
Phase 2 (reduction)                                                           │
  Task 2.1 (filtering) ← depends on 1.4                                      │
  Task 2.2 (PCA, regularized) ← depends on 2.1                               │
  Task 2.3 (save/load PCA) ← depends on 2.2                                  │
  Task 2.4 (full pipeline) ← depends on 2.1, 2.2                             │
       │                                                                      │
Phase 3 (hierarchy)                                                           │
  Task 3.0 (structure test) — NEW, gate before clustering                     │
  Task 3.1 (macro regimes, autocorrelation split, block bootstrap)            │
  Task 3.2 (micro states, regularized, with structure gate)                   │
  Task 3.3 (label assembly)                                                   │
  Task 3.4 (full pipeline)                                                    │
       │                                                          requires data│
Phase 4 (transitions)                                                         │
  Task 4.1 (empirical) ← depends on 3.3                                      │
  Task 4.2 (HMM) ← optional                                                  │
       │                                                                      │
Phase 5 (characterization)                                                    │
  Task 5.1 (centroids) ← depends on 3.4                                      │
  Task 5.2 (entry/exit) ← depends on 3.4                                     │
  Task 5.3 (returns, MULTI-HORIZON) ← depends on 3.4                         │
       │                                                                      │
Phase 6 (validation)                                                          │
  Task 6.1 (quality gates, multi-horizon Q2) ← depends on 5.3                │
  Task 6.2 (cross-symbol, with consensus logic) ← depends on 3.4             │
       │                                                                      │
Phase 7 (online detector)                                                     │
  Task 7.1 (buffer) ← depends on 1.4                                         │
  Task 7.2 (classifier + DRIFT DETECTION) ← depends on 2.3, 3.4             │
  Task 7.3 (persistence) ← depends on 7.2                                    │
       │                                                                      │
Phase 8 (visualization + reporting)                                           │
  Task 8.1 (plots + structure test viz) ← depends on 3.4, 5.2                │
  Task 8.2 (report) ← depends on 5.1, 5.3, 6.1                             │
       │                                                                      │
Phase 9 (integration tests) ← depends on everything                          │
```

---

## Implementation Order Recommendation

**Week 1:** Tasks 1.0 → 1.1 → 1.2 → 1.3 → 1.4 (derivative engine with feature selection)
**Week 2:** Tasks 0.3 + 2.1 → 2.2 → 2.3 → 2.4 (break detection + reduction)
**Week 3:** Tasks 3.0 → 3.1 → 3.2 → 3.3 → 3.4 (hierarchy with structure gates)
**Week 4:** Tasks 4.1 → 5.1 → 5.2 → 5.3 → 6.1 → 6.2 (transitions + characterization + validation)
**Week 5:** Tasks 7.1 → 7.2 → 7.3 (online detector with drift)
**Week 6:** Tasks 8.1 → 8.2 → 9.1 → 9.2 (visualization, reporting, integration)

Task 4.2 (HMM) is optional — attempt after 2+ weeks of continuous data.

---

## Key v2 Changes Summary

| Original (v1) | Correction (v2) | Reason |
|---|---|---|
| Derive all 191 features | Select top 10-15 first (Task 1.0) | Prevents noise explosion |
| Spectral on all features | 3 features only, with kill criterion | Crypto has no periodicity |
| Column-name slow/fast split | Autocorrelation-based split | Data-driven, not arbitrary |
| Random bootstrap | Block bootstrap (block=10-20) | Preserves temporal structure |
| Q3 at 1-bar horizon only | Multi-horizon [1, 5, 10, 20, mean_duration] | Regimes predict at their own timescale |
| No pre-clustering check | Hopkins + dip test gate (Task 3.0) | Don't force structure on noise |
| PCA on small samples | Ledoit-Wolf regularization | Prevents unstable components |
| Drift detection as afterthought | Built into online classifier | Essential for production |
| Cross-symbol check unused | Consensus labels + disagreement rate | Actionable output |
| No break detection | PELT change-point detection (Task 0.3) | Don't cluster across structural shifts |
| 3000 derivatives before PCA | ~200 derivatives after selection | Tractable, less noise |
| max_components = 80 | max_components = 50 | Matches reduced derivative space |

---

## File Structure Summary

```
scripts/cluster_pipeline/
    derivatives.py          # Phase 1: feature selection + derivative generation
    reduction.py            # Phase 2: filtering + regularized PCA
    breaks.py               # Phase 0: structural break detection (NEW)
    hierarchy.py            # Phase 3: structure test + macro/micro discovery
    transitions.py          # Phase 4: transition modeling
    characterize.py         # Phase 5: state profiling
    validate.py             # Phase 6: quality gates + cross-symbol
    online.py               # Phase 7: real-time detector + drift
    report.py               # Phase 8: automated reporting
    preprocess.py           # existing — bar aggregation
    cluster.py              # existing — GMM, k_sweep, quality metrics
    loader.py               # existing — parquet loading

scripts/tests/
    test_derivatives.py     # Phase 1 tests
    test_reduction.py       # Phase 2 tests
    test_breaks.py          # Phase 0 tests (NEW)
    test_hierarchy.py       # Phase 3 tests
    test_transitions.py     # Phase 4 tests
    test_characterize.py    # Phase 5 tests
    test_validate.py        # Phase 6 tests
    test_online.py          # Phase 7 tests
    test_integration_profiling.py  # Phase 9
    test_performance.py     # Phase 9

scripts/visualize_profiling.py   # Phase 8 plots

data/detector/                   # Saved detector artifacts
    config.json
    pca_macro.npz
    pca_micro_0.npz
    gmm_macro.json
    transitions.json
    training_stats.json          # drift detection baseline (NEW)
    metadata.json
```
