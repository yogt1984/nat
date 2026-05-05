# NAT Profiling Pipeline — Mathematical Formulation

**Version:** 1.0  
**Date:** 2026-05-05  
**Scope:** Formal mathematical specification of the NAT profiling pipeline as both specified (PROFILING_TASKS.md v2.0) and implemented (`scripts/cluster_pipeline/`).  
**Status:** Ground-truth reference; discrepancies between spec and implementation are flagged explicitly.

---

## Notation Table

| Symbol | Type | Meaning |
|---|---|---|
| N | Z+ | Number of bars (time steps) in the dataset |
| D | Z+ | Number of raw base features after selection |
| P | Z+ | Number of derivative features |
| P' | Z+ | Number of derivative features after filtering |
| K_m | Z+ | Number of macro regimes |
| K_s(r) | Z+ | Number of micro states in macro regime r |
| K | Z+ | Total micro states: sum_{r} K_s(r) |
| f_i(t) | R | Value of base feature i at bar t, t in {0,...,N-1} |
| x(t) | R^D | Feature vector at bar t |
| d(t) | R^P | Derivative vector at bar t |
| z(t) | R^{P'} | Filtered, standardized derivative vector at bar t |
| y(t) | R^{n_components} | PCA-projected vector at bar t |
| r(t) | {0,...,K_m-1} | Macro regime label at bar t |
| s(t) | {0,...,K-1} | Global micro-state label at bar t |
| w | Z+ | Rolling window size (bars) |
| eps | R+ | Numerical stability floor (context-specific) |
| AC(x, lag) | R | Autocorrelation of series x at given lag |
| rho_ij | R | Pearson correlation between columns i and j |

---

## Formal Problem Statement

Let {x(t)}_{t=0}^{N-1} be a multivariate time series of market microstructure features from Hyperliquid perpetual futures, sampled at fixed bar intervals (e.g., 15 minutes). Each x(t) in R^{191} is a vector drawn from the NAT feature manifest.

**Goal:** Discover a hierarchical partition of the time index {0,...,N-1} into K_m macro regimes and K micro states such that:

1. Each partition is statistically non-trivial (Hopkins statistic > 0.7 or Hartigan dip p < 0.05).
2. States are stable under block-resampling perturbation (ARI > 0.6).
3. States predict forward log-returns at some horizon h in {1, 5, 10, 20} bars (Kruskal-Wallis H-test p < 0.05 with eta-squared > 0.01).
4. States are operationally persistent (self-transition rate > 0.8, mean duration >= 3 bars).

The pipeline also produces an online classifier that maps each new derivative vector d(t) to a state estimate in real time, with built-in distributional drift detection.

---

## Stage 0: Data Preprocessing

### 0.1 Bar Aggregation

Raw tick data (timestamp_ns, symbol, price, quantity, ...) is aggregated into fixed time bars via `aggregate_bars(df, timeframe)`. Each bar t corresponds to the interval [t * Delta, (t+1) * Delta) where Delta is the bar duration. The exact aggregation formulas (OHLCV, volume-weighted averages, entropy, imbalance, etc.) are defined in FEATURES.md and are the input domain for the profiling pipeline.

**Output:** DataFrame with N rows and ~191 numeric columns, indexed by bar open-time.

### 0.2 Structural Break Detection

**Spec (Task 0.3):** Apply PELT (Pruned Exact Linear Time) change-point detection to the first 5 PCA components of the raw bar features.

**Implementation (`hierarchy.py:_detect_breaks_safe`):**

The implementation does NOT use the first 5 PCA components as specified. Instead it operates directly on all numeric bar columns (after `fillna(0)`). The PELT penalty is the BIC penalty:

    pen = log(N) * n_features

using `ruptures.Pelt(model="rbf")`. The RBF kernel cost measures distributional change across all features jointly.

**PELT objective (Yao, 1988):**

    minimize_{b_1,...,b_m} sum_{l=0}^{m} C(y_{b_l + 1 : b_{l+1}}) + beta * m

where C is the RBF segment cost, beta = log(N) * n_features is the BIC penalty per breakpoint, and the number of breakpoints m is free. The minimum segment length constraint is min_size = 50 bars.

**Post-detection:** If breaks B = {b_1, ..., b_m} are detected, segment boundaries are:

    boundaries = [0, b_1, b_2, ..., b_m, N]

The longest segment [start, end) is selected:

    (start, end) = argmax_{consecutive pair in boundaries} (end - start)

The pipeline then proceeds using only bars[start:end].

**Discrepancy:** Spec says "first 5 principal components"; implementation uses all numeric columns. This means the break detector operates in higher-dimensional space and may have lower statistical power per dimension. The BIC penalty partially compensates by scaling with n_features.

**Fallback:** If `ruptures` is not installed or detection fails, returns [] (no breaks), and the full dataset is used. This is logged at INFO level.

---

## Stage 1: Feature Selection

### 1.1 Select Top Features (`derivatives.py:select_top_features`)

Given the aggregated bar DataFrame, candidate columns are identified by matching the vector name against bar column names via `_match_vector_columns(vector, columns)`. Near-constant columns (variance < 1e-10) are excluded before ranking.

**Method "variance" (default):**

    Selected = top-D columns by Var[f_i] in descending order, D = min(max_features, |candidates|)

where Var[f_i] = (1/N) * sum_{t=0}^{N-1} (f_i(t) - mean(f_i))^2.

The implementation uses `pandas.Series.var(skipna=True)` which computes the biased variance (ddof=0) by default.

**Method "autocorrelation_range":**

For each candidate column i, compute the autocorrelation at every lag l in {1,...,30}:

    AC_i(l) = [sum_{t=0}^{N-l-1} (f_i(t) - mu_i)(f_i(t+l) - mu_i)] / [(N - l) * sigma_i^2]

where mu_i = mean(f_i), sigma_i^2 = Var[f_i].

The informativeness score is:

    score_i = max_{l in 1..30} AC_i(l) - min_{l in 1..30} AC_i(l)

Features are ranked by score_i descending. This captures features whose persistence varies most across time scales.

**Output:** List of D column names, D <= max_base_features (default 15).

**Rationale for selection before derivation:** 191 features × (2 + 3 * 3) = 1,691 temporal derivatives alone. With D <= 15: 15 × 11 = 165 temporal + ~15 cross = ~180 total. A 10x reduction in dimensionality before PCA, reducing noise and computation.

---

## Stage 2: Derivative Generation

### 2.1 Temporal Derivatives (`derivatives.py:temporal_derivatives`)

For each selected base column f_i and each window w in W = {5, 15, 30}:

**Velocity (1st-order finite difference):**

    v_i(t) = f_i(t) - f_i(t-1),     t >= 1
    v_i(0) = NaN

**Acceleration (2nd-order finite difference):**

    a_i(t) = v_i(t) - v_i(t-1),     t >= 2
    a_i(0) = a_i(1) = NaN

**Rolling Z-score (window w):**

    mu_i(t, w) = (1/w) * sum_{s=t-w+1}^{t} f_i(s),              t >= w-1
    sigma_i(t, w) = sqrt[(1/(w-1)) * sum_{s=t-w+1}^{t} (f_i(s) - mu_i(t,w))^2]

    zscore_i(t, w) = { 0                                 if sigma_i(t, w) < 1e-10
                     { (f_i(t) - mu_i(t, w)) / sigma_i(t, w)    otherwise
                     { NaN                               if t < w-1

Note: `min_periods=w` is enforced, so the first w-1 rows are NaN. The rolling std uses ddof=1 (pandas default for `.std()`).

**Rolling Slope (OLS over window w):**

Define x_j = j for j in {0, 1, ..., w-1} (integer time index). The OLS slope of f_i over the window ending at t is the closed-form simple linear regression slope:

    b_i(t, w) = [w * sum_{j=0}^{w-1} j * f_i(t-w+1+j) - sum_x * sum_{j=0}^{w-1} f_i(t-w+1+j)] / denominator

where:
    sum_x = sum_{j=0}^{w-1} j = w(w-1)/2
    sum_x2 = sum_{j=0}^{w-1} j^2 = w(w-1)(2w-1)/6
    denominator = w * sum_x2 - sum_x^2

All three summations over x are constant for a fixed window width w and precomputed once per column. The sliding computation uses numpy dot product:

    sum_xy(t) = dot([0,1,...,w-1], f_i[t-w+1:t+1])

**Complexity:** O(N * w) per column per window due to the per-window dot product. The implementation uses a Python loop over window positions (lines 268-274 of derivatives.py), not a vectorized cumsum approach, so runtime is O(N * w * D * |W|).

Note: The spec suggests `df[col].rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])` as the implementation approach. The actual implementation uses a hand-coded closed-form formula that is mathematically equivalent but avoids the overhead of `np.polyfit` allocation per window. This is an optimization, not a discrepancy.

**Rolling Volatility (window w):**

    rvol_i(t, w) = sigma_i(t, w)     [same as rolling std defined above]
    rvol_i(t, w) = NaN               if t < w-1

**Column naming convention:**
- `{col}_vel` — velocity
- `{col}_accel` — acceleration
- `{col}_zscore_{w}` — z-score with window w
- `{col}_slope_{w}` — OLS slope with window w
- `{col}_rvol_{w}` — rolling std with window w

**Total temporal derivative columns per base feature:** 2 + 3 * |W| = 2 + 9 = 11 (for default W = {5, 15, 30}).

**Total temporal derivatives:** D * 11, where D <= 15.

**Warmup period:** max(W) bars. Rows 0..max(W)-2 have NaN in all window-dependent derivatives. The `profile()` function drops these rows after derivative generation.

**Inf handling:** After all derivatives are assembled, `replace([inf, -inf], 0.0)` is applied. This means a division-by-zero in any intermediate calculation is silently converted to 0.0 at the output stage.

### 2.2 Cross-Feature Derivatives (`derivatives.py:cross_feature_derivatives`)

For each pair (a, b) in DEFAULT_CROSS_PAIRS:

**Ratio:**

    ratio_{a,b}(t) = clip[a(t) / (b(t) + eps), -C, C]

where eps = 1e-10, C = 100.0. NaN is propagated if either a(t) or b(t) is NaN.

**Rolling Correlation (window w):**

    corr_{a,b}(t, w) = [sum_{s=t-w+1}^{t} (a(s) - mu_a(t,w)) * (b(s) - mu_b(t,w))] / [(w-1) * sigma_a(t,w) * sigma_b(t,w)]

This is the Pearson rolling correlation computed by `pandas.Series.rolling(w).corr()`. Returns NaN when t < w-1.

**Divergence (window w):**

    za(t, w) = { 0        if sigma_a(t, w) < 1e-10 or t < w-1
               { (a(t) - mu_a(t, w)) / sigma_a(t, w)      otherwise

    zb(t, w) = { 0        if sigma_b(t, w) < 1e-10 or t < w-1
               { (b(t) - mu_b(t, w)) / sigma_b(t, w)      otherwise

    div_{a,b}(t, w) = za(t, w) - zb(t, w)

**Default cross-pairs (as implemented; note the spec had different glob patterns):**

| Pattern a | Pattern b | Operations |
|---|---|---|
| `ent_*_mean` | `vol_*_mean` | ratio, corr |
| `imbalance_*_mean` | `raw_spread_*` | ratio |
| `whale_*_sum` | `flow_volume_*_sum` | ratio |
| `toxic_*_mean` | `illiq_*_mean` | ratio, corr |
| `ent_*_mean` | `trend_*_mean` | corr, divergence |

**Discrepancy:** The spec defines patterns like `entropy_*_mean`, `volatility_*_mean`, `orderflow_imbalance_*_mean`, `spread_*_mean`, `whale_flow_*_sum`, `volume_*_sum`, `toxicity_*_mean`, `illiquidity_*_mean`, `trend_*_mean`. The implementation uses shorter aliases: `ent_*_mean`, `vol_*_mean`, `imbalance_*_mean`, `raw_spread_*`, `whale_*_sum`, `flow_volume_*_sum`, `toxic_*_mean`, `illiq_*_mean`. These must match the actual bar column naming conventions defined in `config.py`.

**Column resolution:** `fnmatch.fnmatch(col, pattern)` for glob matching; exact match takes priority. If a pattern resolves to multiple columns, only the first match is used (the implementation uses `a_cols[0]`).

**Column naming:** `cross_{a_short}_{b_short}_{op}[_{w}]` where `_shorten_col()` strips trailing `_mean`, `_std`, `_last`, `_sum`, `_slope`, `_open`, `_high`, `_low`, `_close`.

### 2.3 Spectral Derivatives (Spec only, not implemented in generate_derivatives)

**Spec (Task 1.3) defines spectral features; these are NOT included in the current `generate_derivatives()` call.** The `include_spectral` parameter exists in `profile()` but is passed to `generate_derivatives()` which does not use it — the parameter is present but has no effect on output. The spectral derivative functions (`spectral_derivatives`) are not present in the current `derivatives.py`.

**Spec-defined spectral formulas (for reference):**

Let X_k = FFT{f_i(t-w+1:t+1)} (one-sided DFT of the w-length window ending at t).

    power[k] = |X_k|^2,    k = 0,...,w//2

    spec_low_i(t) = sum_{k=0}^{w//5} power[k]

    spec_high_i(t) = sum_{k=4*w//5}^{w//2} power[k]

    spec_ratio_i(t) = spec_low_i(t) / (spec_high_i(t) + eps)

    spec_period_i(t) = { NaN          if max(power[1:]) < 2 * mean(power)
                       { w / argmax_{k>=1}(power[k])    otherwise

These features are not computed in the current codebase.

### 2.4 Derivative Orchestrator (`derivatives.py:generate_derivatives`)

**Full pipeline:**

1. `base_features = select_top_features(bars, vector, max_features=D)` — D <= 15
2. `td = temporal_derivatives(bars, base_features, windows=W)` — D * 11 columns
3. `cd = cross_feature_derivatives(bars, DEFAULT_CROSS_PAIRS, windows=W)` — up to 5 pairs * (1 + |W| + |W|) columns = up to 5 * 7 = 35 columns
4. `combined = pd.concat([td, cd], axis=1)`

**Warmup rows:** max(W) = 30 (for default W = {5, 15, 30}).

**Discrepancy:** Spec says `generate_derivatives` returns `(DataFrame, Dict)`. Implementation returns a `DerivativeResult` dataclass with fields: `derivatives, n_base_features, base_features, n_temporal, n_cross, n_total, warmup_rows, metadata`. The profile() function accesses `deriv_result.derivatives` and other named fields.

---

## Stage 3: Dimensionality Reduction

### 3.1 Variance Filtering (`reduction.py:filter_derivatives`)

**Input:** Derivative DataFrame with P columns (NaN filled with 0.0 for computation).

**Step 1 — Variance threshold:**

    sigma_j^2 = Var[d_j]    for each column j

    v_threshold = percentile({sigma_j^2}, variance_percentile)

where `variance_percentile` = 10.0 (default), meaning the bottom 10% of columns by variance are dropped. Additionally, a hard floor eps_v = 1e-20 is enforced: any column with Var < 1e-20 is dropped regardless.

    Surviving_1 = {j : sigma_j^2 >= max(v_threshold, 1e-20)}

**Step 2 — Greedy correlation deduplication:**

Compute the pairwise Pearson correlation matrix C in R^{|Surviving_1| x |Surviving_1|}.

Collect all pairs (j, k) with j < k where |C_{jk}| > correlation_threshold (default 0.95):

    bad_pairs = {(j,k) : j < k, |C_{jk}| > 0.95}

Sort bad_pairs in descending order of |C_{jk}|. For each pair (j, k) in sorted order (most correlated first), if neither column has been dropped yet, drop the one with lower variance:

    drop min(j, k, key=lambda c: sigma_c^2)

**Output:** (filtered_df, report) where filtered_df retains only Surviving_2 columns.

**Determinism:** The greedy algorithm processes pairs in deterministic sorted order, giving reproducible outputs for the same input.

### 3.2 PCA with Ledoit-Wolf Regularization (`reduction.py:pca_reduce`)

**Input:** Array X in R^{N x P'} (NaN-filled, filtered).

**Step 1 — Standardization:**

    mu_j = (1/N) * sum_{t=0}^{N-1} X_{tj}

    sigma_j = sqrt[(1/N) * sum_{t=0}^{N-1} (X_{tj} - mu_j)^2]   (population std, ddof=0)

    sigma_j_safe = { 1.0        if sigma_j < 1e-20
                   { sigma_j    otherwise

    Z_{tj} = (X_{tj} - mu_j) / sigma_j_safe

**Step 2 — Covariance estimation:**

The regime determines whether regularization is applied:

    regularized = (N < 2 * P')

If NOT regularized (N >= 2 * P'):

    Sigma = (1/(N-1)) * Z^T Z        [standard sample covariance, via numpy.cov]

If regularized (N < 2 * P'):

    Sigma_LW = (1 - alpha) * S + alpha * mu_LW * I_P'

where S is the sample covariance, alpha is the Ledoit-Wolf shrinkage intensity (analytically estimated by Oracle Approximating Shrinkage), and mu_LW = trace(S)/P' is the shrinkage target (scaled identity). The exact formula for alpha comes from Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional covariance matrices," JMVA 88(2).

Practically: `sklearn.covariance.LedoitWolf().fit(Z).covariance_`.

**Step 3 — Eigendecomposition:**

    Sigma = V Lambda V^T

via `numpy.linalg.eigh` (symmetric eigendecomposition). Eigenvalues lambda_1 >= lambda_2 >= ... >= lambda_{P'} are sorted descending. Negative eigenvalues (numerical noise) are clamped to zero:

    lambda_j = max(lambda_j, 0)

    total_variance = sum_{j=1}^{P'} lambda_j

**Step 4 — Component selection:**

    explained_ratio_j = lambda_j / total_variance

    cumulative_k = sum_{j=1}^{k} explained_ratio_j

Select n_components = smallest k such that cumulative_k >= variance_threshold (default 0.95), capped at max_components (default 50). If threshold is not reached within max_components, use max_components.

Upper bound: n_components <= min(N, P', max_components).

**Step 5 — Projection:**

    components = V^T[:n_components, :]    (shape: n_components x P')

    Y = Z @ components^T                   (shape: N x n_components)

This is the standard PCA projection: Y_{ti} = sum_{j=1}^{P'} Z_{tj} * V_{ji}.

**Step 6 — Loadings:**

For each PC i in {0,...,n_components-1}, the loading is the component vector components[i] in R^{P'}. The top 10 features by absolute weight are stored:

    loadings[i] = top-10 of {(column_j, components[i,j]) : j=1..P'} by |components[i,j]|

**Output:** PCAResult with X_reduced = Y (shape N x n_components), saved mean mu, std sigma_safe, and component matrix V^T[:n_components,:].

### 3.3 Full Reduction Pipeline (`reduction.py:reduce`)

The full pipeline chains:

    (filtered_df, filter_report) = filter_derivatives(derivatives, variance_percentile=10, correlation_threshold=0.95)
    X = filtered_df.fillna(0.0).values
    pca_result = pca_reduce(X, filtered_df.columns, variance_threshold=0.95, max_components=50)
    return pca_result.X_reduced, pca_result, filter_report

NaN fill to 0.0 happens at this stage, after filtering. This means NaN-containing rows are not excluded; they are mapped to the column mean (which is 0 in standardized space) for PCA.

---

## Stage 4: Structure Existence Test

### 4.1 Hopkins Statistic (`hierarchy.py:_hopkins_statistic`)

The Hopkins statistic tests the null hypothesis H_0: data are generated uniformly at random. Under H_0, H ~ Beta(m, m) with mean 0.5.

**Setup:** Let X in R^{N x n_components} be the PCA-reduced data.

**Sample size:** m = clamp(floor(N * sample_ratio), 5, N//2), where sample_ratio = 0.1 (default).

**Random reference points:** Sample m points u_1,...,u_m uniformly from the data bounding box:

    [min_j(X_j), max_j(X_j)]^{n_components}

where any degenerate dimension (range < 1e-20) is replaced by range = 1.0.

**Nearest-neighbor distances:**

    w_i = nearest-neighbor distance from data point x_{idx_i} to its nearest DISTINCT data point:
          w_i = min_{t != idx_i} ||x_{idx_i} - x_t||_2

    u_i = nearest-neighbor distance from random point u_i to its nearest data point:
          u_i = min_{t in {0..N-1}} ||u_i - x_t||_2

**Hopkins statistic:**

    H = sum_{i=1}^{m} u_i / (sum_{i=1}^{m} u_i + sum_{i=1}^{m} w_i)

**Discrepancy from classical formulation:** The classical Hopkins statistic raises distances to the d-th power (where d = n_components):

    H_classical = sum u_i^d / (sum u_i^d + sum w_i^d)

The implementation uses raw L2 distances without the d-th power. This is explicitly noted in the code (lines 196-200 of hierarchy.py) as a numerical stability choice: in high dimensions (d > 5), the d-th power amplifies tiny differences and makes the statistic degenerate toward 0 or 1. The raw-distance formulation is a deliberate approximation.

**Degenerate case:** If sum(u) + sum(w) < 1e-30, returns H = 0.5 (no information).

**Interpretation:**
- H ~ 0.5: uniformly distributed data (no clustering)
- H > 0.7: strong clustering tendency
- Threshold: 0.7 (configurable via `hopkins_threshold`)

### 4.2 Hartigan Dip Test (`hierarchy.py:test_structure_existence`)

The dip test (Hartigan & Hartigan, 1985) tests H_0: the distribution of PC1 is unimodal.

    (dip_stat, dip_p) = diptest.diptest(X_reduced[:, 0])

where `X_reduced[:, 0]` is the first principal component. The p-value dip_p < significance (default 0.05) constitutes evidence of multimodality.

**Decision logic:**

    Let H = Hopkins statistic, p = dip_p, tau_H = 0.7, alpha = 0.05

    if H > tau_H AND p < alpha:  recommendation = "proceed"
    if H > tau_H XOR p < alpha:  recommendation = "weak_structure"
    if H <= tau_H AND p >= alpha: recommendation = "no_structure"

    has_structure = (H > tau_H OR p < alpha)

If `has_structure = False`, the pipeline returns early with a dummy result (all labels set to 0, k=0). No clustering is attempted.

---

## Stage 5: Macro Regime Discovery

### 5.1 Autocorrelation Split (`hierarchy.py:_autocorrelation_split`)

For each derivative column d_j, compute the lag-5 autocorrelation:

    AC_j(5) = [sum_{t=0}^{N-6} (d_j(t) - mu_j)(d_j(t+5) - mu_j)] / [(N - 5) * sigma_j^2]

where mu_j = mean(d_j), sigma_j^2 = Var[d_j]. If sigma_j^2 < 1e-20, AC_j(5) = 0.

**Slow columns:** S = {j : AC_j(5) > tau_ac}, where tau_ac = 0.7 (default `autocorrelation_threshold`).

If |S| < 2, a warning is issued and all columns are used.

Slow columns are sorted by AC_j(5) descending.

**Rationale:** Persistent (high-autocorrelation) features are better suited for detecting broad market regimes, since they vary slowly and their levels are meaningful across multiple bars. Fast features (low autocorrelation) capture transient dynamics better suited for micro-state detection.

**Discrepancy from spec:** The spec describes this as splitting into "slow" and "fast" feature sets, where fast features would be used for micro-state detection. The implementation uses slow features for macro detection but uses ALL features (not just fast) for micro-state detection. This is consistent with the spec's v2 correction note: "micro-states capture fast dynamics within a regime."

### 5.2 GMM k-Sweep (`hierarchy.py:_k_sweep_gmm`)

The GMM (Gaussian Mixture Model) likelihood model:

    p(y | theta) = sum_{k=1}^{K_m} pi_k * N(y ; mu_k, Sigma_k)

where pi_k are mixture weights, mu_k in R^{n_components} are component means, Sigma_k in R^{n_components x n_components} are full covariance matrices.

**Fitting:** Via EM algorithm (sklearn.mixture.GaussianMixture, n_init=3 for sweep, n_init=5 for final fit). `covariance_type="full"` means each component has its own unrestricted covariance matrix.

**BIC criterion:**

    BIC(k) = -2 * log L_hat(k) + k_params(k) * log(N)

where:
    log L_hat(k) = sum_{t=0}^{N-1} log p(y(t) | theta_hat_k)

    k_params(k) = k * (n_components + n_components*(n_components+1)/2) + (k-1)
                = k * (n_components + n_components*(n_components+1)/2 + 1) - 1

(number of free parameters: k means, k full covariance matrices, k-1 free weights).

**Best k:**

    k* = argmin_{k in k_range} BIC(k)

where k_range = range(2, 6) by default (i.e., k in {2, 3, 4, 5}).

### 5.3 Final GMM Fit

At best_k k*, re-fit GMM with n_init=5 (more restarts than the sweep, for better convergence):

    labels(t) = argmax_k pi_k * N(y(t) ; mu_k, Sigma_k)

This is the MAP assignment (hard assignment), equivalent to `gmm.fit_predict(X_reduced)`.

### 5.4 Quality Metrics (`hierarchy.py:_compute_quality`)

**Silhouette score:**

    a(t) = mean_{s: label(s)=label(t), s!=t} ||y(t) - y(s)||_2     [intra-cluster distance]

    b(t) = min_{k != label(t)} mean_{s: label(s)=k} ||y(t) - y(s)||_2  [nearest-cluster distance]

    sil(t) = (b(t) - a(t)) / max(a(t), b(t))

    silhouette = (1/N) * sum_{t=0}^{N-1} sil(t)

Range: [-1, 1]. Values near 1 indicate well-separated clusters.

**Minimum cluster fraction:**

    min_frac = min_k |{t : label(t) = k}| / N

This guards against degenerate solutions where one cluster is nearly empty.

### 5.5 Block Bootstrap Stability (`hierarchy.py:_block_bootstrap_stability`)

**Motivation:** Random bootstrap destroys temporal autocorrelation. Block bootstrap preserves it, giving honest stability estimates for temporally correlated data.

**Procedure:**

Let n_blocks = max(1, floor(N / block_size)). For each of n_bootstrap = 30 iterations:

1. Sample n_blocks block-start indices {s_1,...,s_{n_blocks}} uniformly from {0,...,N-block_size} with replacement.
2. Construct bootstrap sample indices: I = union_{l=1}^{n_blocks} {s_l, s_l+1, ..., s_l+block_size-1}, then trim to length N.
3. Construct X_boot = X_reduced[I, :].
4. Fit GMM(k=k*) on X_boot (n_init=1 for speed).
5. Compute ARI between reference labels on the same index set and bootstrap labels:

        boot_labels = gmm.fit_predict(X_boot)
        ref_labels = macro_labels[I]
        ARI_b = adjusted_rand_score(ref_labels, boot_labels)

**Adjusted Rand Index (Hubert & Arabie, 1985):**

    ARI = (RI - E[RI]) / (max(RI) - E[RI])

where RI (Rand Index) = fraction of pairs with consistent assignments. ARI = 1 means perfect agreement; ARI = 0 means random agreement.

**Stability report:**

    mean_ARI = (1/n_bootstrap) * sum_{b=1}^{n_bootstrap} ARI_b
    std_ARI  = std({ARI_b})

**Edge cases:** Bootstrap samples causing GMM convergence failure are silently skipped. If all bootstrap attempts fail, mean_ARI = 0, std_ARI = 0.

### 5.6 Self-Transition Rate (`hierarchy.py:_self_transition_rate`)

    STR = |{t : label(t) = label(t+1), t in {0,...,N-2}}| / (N - 1)

Range [0, 1]. STR = 1 means every consecutive pair shares the same label (maximally persistent). The operational threshold is STR > 0.8.

### 5.7 Duration Distributions (`hierarchy.py:_compute_durations`)

A "run" of regime k is a maximal consecutive sequence {t, t+1,...,t+L-1} where label(s) = k for all s in the run. Duration = L.

    durations[k] = [L : L is a run length of regime k in the label sequence]

This is computed by a single left-to-right scan of the label array, tracking the current label and current run count. This is O(N).

### 5.8 Centroid Profiles (`hierarchy.py:_centroid_profiles`)

For each regime k, the centroid profile is the mean of all derivative columns for bars assigned to that regime:

    centroid[k, j] = mean_{t : label(t)=k} d_j(t)

Stored as a DataFrame indexed by regime label, one column per derivative feature.

---

## Stage 6: Micro-State Discovery

### 6.1 Per-Regime Subsetting

For each macro regime r in {0,...,K_m-1}:

    regime_df = derivatives[macro_labels == r].reset_index(drop=True)
    n_regime = |{t : macro_labels(t) = r}|

If n_regime < min_bars (default 100), return None (regime too small). This threshold is hardcoded in the function signature as `min_bars=100`.

### 6.2 Per-Regime PCA

Apply the full reduction pipeline (filter_derivatives -> pca_reduce) to regime_df. Because n_regime < N, the Ledoit-Wolf condition (n_regime < 2 * P') is more likely to trigger, automatically regularizing the covariance estimate.

The regime-specific PCA basis {mu_r, sigma_r, V_r} is saved separately from the global macro PCA basis.

### 6.3 Structure Test and k-Sweep

Apply `test_structure_existence` to the regime-reduced data X_r in R^{n_regime x n_r_components}.

If `has_structure = False`: return None (no micro-state structure in this regime).

k-sweep and GMM fit proceed identically to the macro case:

    k_r* = argmin_{k in k_range, k < n_regime} BIC_r(k)

The k_range is filtered to ensure k < n_regime (cannot have more clusters than data points).

Final GMM: n_init=5, covariance_type="full", fit on X_r.

Block bootstrap stability uses block_size=10 (smaller than the macro block_size=15, appropriate for the smaller per-regime sample).

### 6.4 Output

MicroStateResult contains:
- `labels`: local micro-state labels in {0,...,k_r*-1}, indexed relative to the regime subset
- `k`: k_r* (number of micro states in this regime)
- `pca_result`: per-regime PCAResult
- `structure_test`: StructureTest for this regime's data

---

## Stage 7: Hierarchical Label Assembly

### 7.1 Global State ID Assignment (`hierarchy.py:assemble_hierarchy`)

Regimes are processed in sorted order {0, 1, ..., K_m-1}. Within each regime r, local states are processed in sorted order:

    global_id_start(r) = sum_{r'=0}^{r-1} K_s(r')

    global_id(r, local) = global_id_start(r) + local

where K_s(r) is the number of distinct local states in regime r (or 1 if micro_results[r] is None).

This gives a bijection `label_map: global_id -> (regime_id, local_state_id)`.

**Global micro-state label assignment:**

For each bar t:

1. Determine macro regime: r(t) = macro_labels[t].
2. If micro_results[r(t)] is not None, look up the local state for bar t:
   - The local state index is obtained by counting how many bars in regime r(t) precede bar t (tracked via `regime_bar_counters[rid]`).
   - local_id = micro_results[r(t)].labels[bar_idx]
3. If micro_results[r(t)] is None: local_id = 0.
4. s(t) = local_to_global[(r(t), local_id)]

**Composite label:** `"R{r(t)}_S{local_id}"` for all bars t.

**Invariants:**
- micro_labels[t] is always in {0,...,K-1}
- label_map is invertible: label_map[s(t)] = (r(t), local_id(t))
- Global IDs are contiguous: {0, 1, ..., K-1}

**Edge case:** Regimes not present in micro_results get a single global state. The code validates that all keys in micro_results reference valid regimes.

---

## Stage 8: Transition Modeling

### 8.1 Empirical Transition Matrix (`transitions.py:empirical_transitions`)

**Count matrix:**

    C[i, j] = |{t in {0,...,N-2} : label(t) = state_i AND label(t+1) = state_j}|

where state_i, state_j range over the unique states. States are mapped to matrix indices via a fixed ordering: unique_states sorted ascending.

**Row-normalization:**

    T[i, j] = C[i, j] / sum_j C[i, j]

**Degenerate row:** If state i appears only as the final bar (no outgoing transitions), row i has all-zero counts. In this case, the row is set to uniform: C[i, j] = 1/K for all j. This ensures T is row-stochastic.

**Row-stochasticity:** By construction, sum_j T[i,j] = 1 for all i.

### 8.2 Row Entropy

Shannon entropy of each transition row:

    H(i) = -sum_{j=0}^{K-1} T[i,j] * log(T[i,j] + eps),     eps = 1e-15

High entropy means transitions are nearly uniform (unpredictable successor). Low entropy (approaching 0) means almost certain self-transition.

### 8.3 Most Likely Successor

For state i, the most likely successor is the argmax of the off-diagonal row elements:

    successor(i) = argmax_{j != i} T[i,j]

Implemented by setting T[i,i] = -1 and taking argmax of the modified row.

### 8.4 Duration Distributions

Same run-length computation as in macro discovery (_compute_duration_distributions). For state s:

    mean_duration[s] = mean({L : L is a run of state s in the label sequence})

For a geometric distribution (Markov chain with constant self-transition probability p_s = T[s,s]), the mean duration is:

    E[L] = 1 / (1 - T[s,s])

The spec notes this relationship and asks for deviation to be reported. The implementation reports mean_durations but does not explicitly compute the geometric prediction or deviation.

---

## Stage 9: State Characterization

### 9.1 Centroid and Z-Score Elevation (`characterize.py:characterize_states`)

**Global statistics:**

    mu_g[j] = (1/N) * sum_{t=0}^{N-1} d_j(t)         [global mean of derivative j]
    sigma_g[j] = std({d_j(t)})                         [global std, ddof=1]

    sigma_g[j] = max(sigma_g[j], max(|mu_g[j]| * 1e-10, 1e-12))   [stability floor]

**State centroid:**

    centroid[s, j] = mean_{t: label(t)=s} d_j(t)

**Z-score of centroid vs global:**

    z[s, j] = (centroid[s, j] - mu_g[j]) / sigma_g[j]

**Top elevated features:** Top-10 columns by z[s, j] where z[s, j] > 1e-10.

**Top suppressed features:** Top-10 columns by z[s, j] (ascending, most negative) where z[s, j] < -1e-10.

### 9.2 Entry and Exit Signatures (`characterize.py:compute_signatures`)

**Entry events:** Times t where label(t) = s AND (t=0 OR label(t-1) != s).

**Entry trajectory for event at t:** derivatives[t-lookback:t] — the lookback bars BEFORE entering state s.

**Exit events:** Times t where label(t) = s AND (t=N-1 OR label(t+1) != s).

**Exit trajectory for event at t:** derivatives[t+1:t+1+lookback] — the lookback bars AFTER leaving state s.

**Mean and std trajectories:**

    entry_mean[tau, j] = mean over valid entry events of d_j(event - lookback + tau),   tau in {0,...,lookback-1}
    entry_std[tau, j]  = std over valid entry events

**Return condition:** Returns None if both entry_count < min_events AND exit_count < min_events (default min_events=5).

The time index of entry_trajectory is range(-lookback, 0); the time index of exit_trajectory is range(1, lookback+1).

### 9.3 Forward Return Profiling (`characterize.py:return_profile`)

**Log-return at horizon h for bar t:**

    r_h(t) = log(price(t+h)) - log(price(t))

For each state s and horizon h:

    valid_indices = {t : label(t) = s, t + h < N}
    returns_h(s) = {r_h(t) : t in valid_indices}

**Statistics per horizon:**

    mean_h(s) = E[returns_h(s)]
    median_h(s) = median(returns_h(s))
    std_h(s) = std(returns_h(s), ddof=1)
    p5_h(s)  = 5th percentile of returns_h(s)
    p95_h(s) = 95th percentile of returns_h(s)

**Skewness (sample):**

    m3 = (1/n) * sum (r - mean)^3
    skew = m3 / std^3 * n / ((n-1)(n-2)/n)

Note: This is NOT the standard unbiased Fisher-Pearson skewness (which uses n/((n-1)(n-2))). The formula as implemented multiplies by n/((n-1)(n-2)/n) = n^2/((n-1)(n-2)), which equals the adjusted Fisher-Pearson coefficient. For large n, the difference is negligible.

**Excess kurtosis:**

    kurt = (1/n) * sum ((r - mean)/std)^4 - 3

This is excess kurtosis (subtracts 3 from the normal-distribution value).

**Per-bar Sharpe:**

    sharpe_h(s) = mean_h(s) / std_h(s)     if std > 1e-15

This is NOT annualized. It is the per-bar mean-to-std ratio at horizon h.

**Auto-horizon:** If `mean_duration` is provided and not already in horizons, it is inserted into the sorted horizon list. This ensures each state is evaluated at its natural timescale.

---

## Stage 10: Validation Framework

### 10.1 Q1 — Structural Quality (`validate.py:_evaluate_q1`)

Q1 tests whether the discovered clusters are statistically real.

**Conditions (both must hold):**

    Q1_pass = (silhouette >= 0.25) AND (bootstrap_ARI >= 0.6)

where:
- silhouette: mean silhouette score of macro regime GMM (on macro PCA space)
- bootstrap_ARI: mean block-bootstrap ARI from macro discovery

**Failure consequence:** If Q1 fails, overall verdict is "DROP" — the clustering is not reproducible or separable enough to be trusted.

**Discrepancy from spec:** The spec lists Q1_DEFAULTS as including `"temporal_ari": 0.5`. This threshold is defined in Q1_DEFAULTS in the spec but is NOT evaluated in `_evaluate_q1()` in the implementation. Only `silhouette` and `block_bootstrap_ari` are checked. The `temporal_ari` field is absent from the implementation's Q1_DEFAULTS dict.

### 10.2 Q2 — Predictive Quality (`validate.py:_evaluate_q2`)

Q2 tests whether micro-state membership predicts forward log-returns.

**Kruskal-Wallis H-test:** At each horizon h in {1, 5, 10, 20}, test:

    H_0^h: the distribution of r_h(t) is the same across all micro states

The Kruskal-Wallis test (non-parametric one-way ANOVA on ranks):

    KW_H = [12 / (n_total * (n_total + 1))] * sum_s n_s * (R_bar_s - R_bar)^2

where n_s = number of observations in state s, n_total = sum n_s, R_bar_s = mean rank in state s, R_bar = (n_total+1)/2 (overall mean rank). Under H_0, KW_H ~ chi^2(K-1) asymptotically.

**Effect size (eta-squared):**

    eta_sq_h = KW_H / (n_total - 1)

**Horizon passes if:**

    pass_h = (p_value_h < 0.05) AND (eta_sq_h >= 0.01)

**any_horizon=True (default):**

    Q2_pass = any(pass_h for h in {1, 5, 10, 20})

**any_horizon=False:**

    Q2_pass = all(pass_h for h in {1, 5, 10, 20})

**Minimum group size:** Groups with fewer than 5 observations are excluded from the Kruskal-Wallis test.

**Failure consequence:** If Q1 passes but Q2 fails at all horizons, overall verdict is "COLLECT" — the clusters are structurally real but not yet predictive, suggesting more data is needed.

### 10.3 Q3 — Operational Quality (`validate.py:_evaluate_q3`)

Q3 tests whether the states are useful for trading.

**Conditions (both must hold):**

    Q3_pass = (STR >= 0.8) AND (mean_duration >= 3)

where:
- STR = macro self-transition rate
- mean_duration = mean run length across all macro regimes

**Discrepancy from spec:** The spec lists `"micro_str": 0.5` and `"entry_lead": 1` in Q3_DEFAULTS. The implementation defines these in the defaults dict but does NOT evaluate them in `_evaluate_q3()`. Only `macro_str` and `mean_duration` are checked. The micro self-transition rate and entry lead time are not computed.

**Failure consequence:** If Q1 and Q2 pass but Q3 fails, overall verdict is "PIVOT" — the states are real and predictive but need operational adjustment (e.g., increase bar duration to extend state persistence).

### 10.4 Decision Logic

The four-outcome decision tree:

    if NOT Q1_pass:          overall = "DROP"
    elif NOT Q2_pass:        overall = "COLLECT"
    elif NOT Q3_pass:        overall = "PIVOT"
    else:                    overall = "GO"

**Per-state verdict:** For each global micro state s, compute return_profile(labels, prices, state_id=s, horizons=[1,5,10,20]). If any horizon has |sharpe| >= 0.3 (with n >= 10 observations), the per-state verdict is "GO", otherwise "COLLECT".

### 10.5 Cross-Symbol Consistency (`validate.py:cross_symbol_consistency`)

Given label arrays {L_sym : sym in symbols} of equal length N:

**Pairwise ARI matrix:**

    agreement[i, j] = ARI(L_{sym_i}, L_{sym_j})

**Mean agreement:**

    mean_agreement = mean of off-diagonal elements of agreement matrix

    above_random = (mean_agreement > random_threshold)   [default threshold = 0.05]

**Majority-vote consensus at each bar t:**

    bar_labels = {L_{sym}[t] : sym in symbols}
    max_count = max frequency of any label in bar_labels
    if max_count > n_symbols / 2:   [strict majority]
        consensus[t] = mode(bar_labels)
    else:
        consensus[t] = -1   [uncertain]

    disagreement_rate = |{t : consensus[t] = -1}| / N

**Discrepancy from spec:** The spec's `cross_symbol_consistency` signature takes `(df, vector, timeframe, symbols)` and runs `profile()` per symbol internally. The implementation takes `(per_symbol_labels: Dict[str, np.ndarray])` — it expects pre-computed label arrays. The caller is responsible for running profile() per symbol and passing the resulting labels.

---

## Stage 11: Online Classification

### 11.1 Derivative Buffer (`online.py:DerivativeBuffer`)

A circular buffer of capacity max_window bars, implemented as `collections.deque(maxlen=max_window)`.

**Warmup condition:**

    is_warm = len(buffer) >= max_window

where max_window = max(temporal_windows) + 1 (minimum required for all derivatives to be valid at the final bar).

**Update procedure:**

On each bar push:
1. Append the bar's feature values (for the selected columns) to the deque.
2. If not warm: return None.
3. Build DataFrame from the deque contents.
4. Apply `temporal_derivatives(df, columns, windows)` to the full buffer.
5. Return the last row of the derivative DataFrame as a 1-D float64 array.

**Memory guarantee:** The deque has maxlen=max_window, so memory is O(max_window * D) regardless of how many bars have been pushed.

**Equivalence to batch computation:** The output of `update(bar_t)` equals the last row of `temporal_derivatives(df[t-max_window+1:t+1], columns, windows)`. This holds exactly for all derivatives (velocity uses the last 2 rows; z-score uses the last max(windows) rows).

### 11.2 Online Classifier (`online.py:OnlineClassifier`)

**Macro classification:**

Given derivative vector d in R^D:

    z_macro = (d - mu_macro) / sigma_macro_safe    [standardize using macro PCA parameters]
    y_macro = z_macro @ V_macro^T                  [project, shape: (n_macro_components,)]
    p_macro[k] = P(regime=k | y_macro; theta_macro)  [GMM posterior]
    r_hat = argmax_k p_macro[k]                    [MAP regime assignment]

where mu_macro, sigma_macro, V_macro are the saved macro PCA parameters, and theta_macro is the saved macro GMM.

**Micro classification (regime-conditional):**

    z_micro = (d - mu_r_hat) / sigma_r_hat_safe    [standardize using regime r_hat's PCA params]
    y_micro = z_micro @ V_r_hat^T                   [project into regime-specific PCA space]
    p_micro[l] = P(local_state=l | y_micro; theta_r_hat)  [regime-specific GMM posterior]
    l_hat = argmax_l p_micro[l]

    global_state = label_map_inverse[(r_hat, l_hat)]

**All-state probabilities (approximation):**

For global state g = (regime r, local state l):

    P(g) approx P(regime=r) / K_s(r)    [regime probability divided by number of micro states]

This is a factored approximation assuming uniform conditional distribution across micro states for non-current regimes. For the current regime, the actual micro GMM posteriors should be used; the implementation does not fully implement this (see note in `_compute_all_probabilities`, line 426-430).

**Discrepancy:** The spec states "all_probabilities: Dict[int, float]" should reflect true marginal probabilities. The implementation uses a uniform approximation for non-current regimes. For the current regime, it also uses a uniform approximation rather than the true micro GMM posterior (the micro GMM posteriors are computed but not propagated to all_probabilities).

### 11.3 Drift Detection

**Log-likelihood tracking:**

    LL(t) = log p(y_macro(t) | theta_macro)    [GMM log-likelihood for the projected macro vector]

computed via `gmm.score_samples(y_macro.reshape(1,-1))[0]`.

**Rolling average:**

    LL_rolling(t) = (1 / min(t+1, drift_window)) * sum_{s=max(0,t-drift_window+1)}^{t} LL(s)

implemented via `collections.deque(maxlen=drift_window)` with default drift_window = 50.

**Drift detection rule:**

    if LL_rolling(t) < training_ll_p10:
        bars_below_threshold += 1
    else:
        bars_below_threshold = 0

    drift_warning = (bars_below_threshold >= drift_consecutive)   [default drift_consecutive = 20]

where `training_ll_p10` = 10th percentile of per-sample GMM log-likelihoods on the training data.

**Severe drift:** If bars_below_threshold >= 100, a WARNING is logged: "DRIFT: re-profiling recommended".

**Training stats computation:** The training_ll_p10 and training_ll_p50 are stored in `ClassifierConfig` and persisted to `training_stats.json`. They must be computed separately (during profiling) by evaluating the macro GMM's score_samples on the training data and taking percentiles. The `profile()` function does NOT automatically compute these — this is the responsibility of the caller when building the ClassifierConfig.

---

## Stage 12: Persistence

### 12.1 PCA Basis (`reduction.py:save_pca_basis / load_pca_basis`)

**Saved files (given base path `p`):**
- `p.npz` — arrays: X_reduced, components, mean, std, explained_variance_ratio, cumulative_variance
- `p.json` — metadata: n_components, column_names, loadings (as list of [name, weight] pairs), regularized

**Round-trip invariant:** `load_pca_basis(path).components` is bitwise identical to the saved array (no precision loss from JSON).

### 12.2 Classifier Artifacts (`online.py:save_classifier / load_classifier`)

**Saved files in `output_dir/`:**

| File | Contents |
|---|---|
| `pca_macro.npz` | components, mean, std |
| `pca_micro_{r}.npz` | Per-regime: components, mean, std |
| `gmm_macro.npz` | means, covariances, weights, precisions_cholesky |
| `gmm_micro_{r}.npz` | Per-regime GMM parameters |
| `transitions.json` | Transition matrix (nested list), state_ids |
| `config.json` | label_map (str-keyed), n_regimes, regime_ids |
| `training_stats.json` | log_likelihood_p10, log_likelihood_p50 |
| `metadata.json` | Optional: training range, n_bars, verdict |

**GMM reconstruction:** The saved GMM parameters (means, covariances, weights, precisions_cholesky) are loaded and injected into a GaussianMixture object via attribute assignment after calling `GaussianMixture(n_components=k).fit(dummy_data)` to initialize the object structure. This is required because sklearn GaussianMixture does not support direct construction from parameters.

---

## Full Pipeline Data Flow

The complete `profile()` function executes these stages in order:

```
df (N rows, ~191 features)
    │
    ├─ [Optional] aggregate_bars(df, timeframe) → bars (N_bars rows)
    │
    ├─ _detect_breaks_safe(bars, numeric_cols) → breaks (list of int)
    │   └─ if breaks: bars = bars[longest_segment]
    │
    ├─ generate_derivatives(bars, vector, max_base_features=15, temporal_windows=[5,15,30])
    │   ├─ select_top_features(bars, vector, D=15) → base_features (D columns)
    │   ├─ temporal_derivatives(bars, base_features, W) → td (D*11 columns)
    │   └─ cross_feature_derivatives(bars, DEFAULT_CROSS_PAIRS, W) → cd (~15-35 columns)
    │   → DerivativeResult (derivatives: N x P DataFrame, warmup_rows: max(W))
    │
    ├─ Drop warmup rows: derivatives = derivatives[warmup_rows:]
    │   bars = bars[warmup_rows:]
    │
    ├─ discover_macro_regimes(derivatives, autocorrelation_threshold=0.7, k_range=range(2,6))
    │   ├─ _autocorrelation_split(derivatives, lag=5, threshold=0.7) → slow_cols
    │   ├─ reduce(slow_df) → (X_macro, pca_macro, filter_report)
    │   ├─ test_structure_existence(X_macro) → structure_test
    │   │   └─ if no structure: early_exit RegimeResult (k=0, all labels=0)
    │   ├─ _k_sweep_gmm(X_macro, k_range) → sweep (best_k by BIC)
    │   ├─ GaussianMixture(best_k).fit_predict(X_macro) → macro_labels
    │   ├─ _compute_quality(X_macro, macro_labels) → quality (silhouette, min_frac)
    │   └─ _block_bootstrap_stability(X_macro, macro_labels, best_k) → stability (mean_ARI)
    │   → RegimeResult
    │
    ├─ [if not early_exit] for r in range(macro.k):
    │   discover_micro_states(derivatives, macro_labels, r, k_range=range(2,6))
    │       ├─ subset to regime r (n_regime rows)
    │       ├─ if n_regime < 100: return None
    │       ├─ reduce(regime_subset) → (X_micro_r, pca_r, filter_r)
    │       ├─ test_structure_existence(X_micro_r)
    │       │   └─ if no structure: return None
    │       ├─ _k_sweep_gmm + GaussianMixture → micro_labels_r (local)
    │       └─ _compute_quality + _block_bootstrap_stability
    │   → Dict[int, Optional[MicroStateResult]]
    │
    └─ assemble_hierarchy(macro, micros)
        → HierarchicalLabels (macro_labels, micro_labels, composite_labels, label_map)

ProfilingResult (hierarchy, macro, micros, bars, derivative_columns, breaks_detected, structure_test)
```

---

## Computational Complexity Summary

| Stage | Complexity | Dominant Factor |
|---|---|---|
| Bar aggregation | O(N_ticks) | Pandas groupby |
| Break detection (PELT) | O(N log N) | PELT on numeric columns |
| Feature selection (variance) | O(N * 191) | Variance computation |
| Temporal derivatives | O(N * D * |W| * max(W)) | Rolling slope loop |
| Cross-feature derivatives | O(N * |pairs| * |W|) | Rolling correlation |
| Variance filter | O(P) | Variance computation |
| Correlation filter | O(P^2 + P^2 * log P) | Correlation matrix + sort |
| PCA (standardize + eig) | O(N * P + P^3) | Eigendecomposition |
| Hopkins statistic | O(N * log N + m * log N) | kNN queries |
| Dip test | O(N log N) | Sorting |
| GMM k-sweep (per k) | O(N * k * n_components^2 * n_init * n_iter) | EM iterations |
| Block bootstrap | O(n_bootstrap * N * k * n_components^2) | Per-bootstrap GMM fit |
| Transition matrix | O(N) | Single scan |
| State characterization | O(N * P') | Group-wise means |
| Return profiling | O(N * |horizons|) | Indexed lookups |
| Online classify (per bar) | O(P * n_macro_components + k_m * n_macro_components^2) | Matrix-vector products |

---

## Summary of Spec vs Implementation Discrepancies

| # | Location | Spec Says | Implementation Does |
|---|---|---|---|
| 1 | Break detection (Task 0.3) | Apply PELT to first 5 PCA components | Applies PELT directly to all numeric bar columns |
| 2 | Spectral derivatives (Task 1.3) | Include as experimental; kill criterion after PCA | Not implemented; `include_spectral` parameter is a no-op |
| 3 | `generate_derivatives` return type | Returns `(DataFrame, Dict)` | Returns `DerivativeResult` dataclass |
| 4 | Cross-pair glob patterns | `entropy_*_mean`, `volatility_*_mean`, etc. | Shorter aliases: `ent_*_mean`, `vol_*_mean`, etc. |
| 5 | Hopkins statistic | Classical formulation with d-th power of distances | Raw L2 distances (no d-th power), documented as intentional |
| 6 | Q1 thresholds | Includes `temporal_ari: 0.5` | Q1_DEFAULTS omits `temporal_ari`; not evaluated |
| 7 | Q3 thresholds | Includes `micro_str: 0.5`, `entry_lead: 1` | Q3_DEFAULTS includes these but `_evaluate_q3()` does not check them |
| 8 | `cross_symbol_consistency` signature | Takes `(df, vector, timeframe, symbols)`, runs profile internally | Takes `(per_symbol_labels: Dict[str, np.ndarray])`, expects pre-computed labels |
| 9 | `all_probabilities` in StateEstimate | Should reflect true marginal P(global_state) | Uses uniform approximation for non-current regimes |
| 10 | Training stats computation | Should be computed during profiling and saved automatically | Must be computed externally and injected into ClassifierConfig |
| 11 | Macro discovery: fast features for micro | Micro uses "fast" features (low autocorrelation) | Micro uses ALL features (no fast-only subsetting) |
| 12 | `_evaluate_q3` micro_str check | `micro_str` threshold defined in spec | Not evaluated in implementation; only macro_str and duration checked |
