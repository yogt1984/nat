# NAT Profiling Pipeline — Complete Mathematical Reference

**Version:** 1.0  
**Date:** 2026-05-06  
**Codebase:** `scripts/cluster_pipeline/`

---

## Notation Table

| Symbol | Meaning | Type / Domain |
|--------|---------|---------------|
| $t$ | Bar index | Integer, $t \in \{0, 1, \ldots, N-1\}$ |
| $N$ | Total number of bars | Integer, $N \geq 30$ |
| $n$ | Number of samples in a computation window | Integer |
| $p$ | Number of features (columns) | Integer |
| $k$ | Number of clusters / components | Integer, $k \in \{2, 3, 4, 5\}$ |
| $w$ | Rolling window size (in bars) | Integer, $w \in \{5, 15, 30\}$ |
| $f_j(t)$ | Value of feature $j$ at bar $t$ | $\mathbb{R}$ |
| $\mu, \sigma$ | Population mean and standard deviation | $\mathbb{R}$, $\sigma > 0$ |
| $\varepsilon$ | Numerical floor / epsilon | $\mathbb{R}^+$, typically $10^{-10}$ |
| $\mathbf{X}$ | Feature matrix | $\mathbb{R}^{N \times p}$ |
| $\mathbf{Z}$ | Standardised feature matrix | $\mathbb{R}^{N \times p}$, zero mean, unit variance |
| $\boldsymbol{\Sigma}$ | Covariance matrix | $\mathbb{R}^{p \times p}$, positive semi-definite |
| $\mathbf{V}_k$ | Top-$k$ PCA eigenvectors (rows) | $\mathbb{R}^{k \times p}$ |
| $\mathbf{X}_r$ | PCA-reduced matrix | $\mathbb{R}^{N \times k}$ |
| $\lambda_i$ | $i$-th eigenvalue (descending order) | $\mathbb{R}^+$ |
| $\ell(t)$ | State label at bar $t$ | Integer $\in \{0, \ldots, k-1\}$ |
| $P$ | Transition probability matrix | $\mathbb{R}^{k \times k}$, row-stochastic |
| $H(i)$ | Shannon row entropy for state $i$ | $\mathbb{R}^+$ (nats) |

All indices are zero-based. Bars correspond to 15-minute windows unless
otherwise stated. Logarithms are natural ($\ln$) throughout.

---

## Pipeline Overview

```
Raw 100ms ticks
       │
       ▼  Stage 1 — Preprocessing
  15-min bars  (N × p_raw)
       │
       ▼  Stage 2 — Derivative Generation
  Derivative matrix  (N × p')  where p' ≈ 260
       │
       ▼  Stage 3 — Dimensionality Reduction
  PCA-reduced matrix  (N × k)  where k ≈ 10–15
       │
       ▼  Stage 4 — Clustering
  Hierarchical state labels  (N,)
       │
       ▼  Stage 5 — Validation
  Quality verdict: GO / PIVOT / COLLECT / DROP
       │
       ▼  Stage 6 — Transition Matrix
  Row-stochastic P  (k × k)
```

---

## Stage 1: Preprocessing — Bar Aggregation and Scaling

### 1.1 Tick-to-Bar Aggregation

**Input:** DataFrame of 100ms tick rows, timestamp in nanoseconds since epoch.  
**Output:** DataFrame with one row per (symbol, 15-minute bar).

Each column $c$ from the raw tick stream is mapped to one or more aggregated
columns according to its semantic category:

| Category | Matching rule | Aggregations produced |
|----------|--------------|----------------------|
| Default | all other numeric columns | `mean`, `std`, `last` |
| Price | `raw_midprice`, `raw_microprice`, `raw_spread` | `open`, `high`, `low`, `close`, `mean` |
| Volume / count | prefix `flow_volume_`, `flow_count_` | `sum` |
| Whale flow | explicit column list (e.g. `whale_net_flow_*`) | `sum` |
| Entropy | prefix `ent_` | `mean`, `std`, `slope` |

**OLS Slope within a bar.**  
For entropy columns the bar-level *trend* of the feature within the bar is
captured as a linear regression slope. Let $y_0, y_1, \ldots, y_{M-1}$ be the
$M$ tick-level values inside the bar at integer times $x_i = i$. The OLS slope is

$$
\hat{\beta} = \frac{M \sum_{i=0}^{M-1} x_i y_i - \left(\sum_{i=0}^{M-1} x_i\right)\left(\sum_{i=0}^{M-1} y_i\right)}{M \sum_{i=0}^{M-1} x_i^2 - \left(\sum_{i=0}^{M-1} x_i\right)^2}
$$

Because $x_i = i$ the denominator is a fixed constant for a bar of $M$ ticks:

$$
\text{denom} = M \cdot \frac{M(M-1)(2M-1)}{6} - \left(\frac{M(M-1)}{2}\right)^2
$$

Output column name convention: `{original_col}_{agg_suffix}`, for example
`ent_tick_1m_slope`, `raw_midprice_open`.

### 1.2 NaN Handling

Let $\mathbf{X} \in \mathbb{R}^{N \times p}$ be the aggregated bar matrix. NaN
handling proceeds column-wise in three steps:

**Step 1 — Drop high-NaN columns.**

$$
\text{drop column } j \iff \frac{|\{t : X_{t,j} = \text{NaN}\}|}{N} > \tau_\text{nan}
$$

Default threshold: $\tau_\text{nan} = 0.5$.

**Step 2 — Drop near-constant columns.**

$$
\text{drop column } j \iff \widehat{\text{Var}}(X_{\cdot,j}) \leq \tau_\text{var}
$$

Default floor: $\tau_\text{var} = 10^{-10}$.

**Step 3 — Fill remaining NaN with column median.**

$$
X_{t,j} \leftarrow \tilde{X}_j \quad \text{for each remaining NaN entry}
$$

where $\tilde{X}_j = \text{median}(\{X_{t,j} : X_{t,j} \neq \text{NaN}\})$.
If the entire column is NaN, the fill value is 0.

### 1.3 Outlier Clipping

For each surviving column $j$, compute the column mean $\mu_j$ and standard
deviation $\sigma_j$ (computed after median fill), then clip:

$$
X_{t,j} \leftarrow \max\!\left(\mu_j - 5\sigma_j,\; \min\!\left(\mu_j + 5\sigma_j,\; X_{t,j}\right)\right)
$$

This eliminates values more than 5 standard deviations from the column mean.
The threshold of 5 was chosen to preserve genuine market extremes while
removing sensor noise and encoding errors.

### 1.4 Z-Score Normalisation

$$
\mathbf{Z} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

applied column-wise:

$$
Z_{t,j} = \frac{X_{t,j} - \mu_j}{\sigma_j}, \qquad \sigma_j = \sqrt{\frac{1}{N}\sum_{t=0}^{N-1}(X_{t,j} - \mu_j)^2}
$$

Zero-variance columns (after clipping) have $\sigma_j$ replaced by 1 to
prevent division by zero; their scaled values are identically 0.

**Output:** Standardised bar matrix $\mathbf{Z} \in \mathbb{R}^{N \times p_0}$
where $p_0 \leq 191$ is the number of columns that survived NaN and variance
filtering.

---

## Stage 2: Derivative Generation

The derivative stage converts feature *levels* into features capturing *dynamics*.
The motivation is that raw feature levels find structural separation but not
predictive states (Q3 failure mode); derivatives distinguish *how* features are
changing and *how features relate to each other*.

### 2.1 Base Feature Selection

Before generating derivatives, the top $D = 15$ features are selected from
a named feature vector (e.g. `"entropy"`, `"orderflow"`) by one of two methods.

**Method A — Variance ranking (default).**

$$
\text{rank}(j) = \widehat{\text{Var}}(f_j), \qquad
\text{select } \mathcal{F} = \text{top-}D\text{ by rank}
$$

**Method B — Autocorrelation range.**

For each feature $j$, compute the autocorrelation at lags $\ell = 1, \ldots, L$
(default $L = 30$):

$$
\rho_j(\ell) = \frac{\sum_{t=0}^{N-\ell-1}(f_j(t) - \bar{f}_j)(f_j(t+\ell) - \bar{f}_j)}{(N-\ell)\, \widehat{\text{Var}}(f_j)}
$$

Then rank by the *range* across lags:

$$
\text{rank}(j) = \max_\ell \rho_j(\ell) - \min_\ell \rho_j(\ell)
$$

Features whose persistence varies most across time scales carry the most regime
information. A feature with $\rho(1) = 0.95$ and $\rho(30) = 0.1$ is more
informative than one with $\rho(1) = 0.5$ and $\rho(30) = 0.45$.

**Dimension accounting:** $D = 15$ base features, 3 rolling windows
$w \in \{5, 15, 30\}$, 5 temporal derivative types. Temporal derivatives alone
produce $15 \times (2 + 3 \times 3) = 165$ columns. Adding cross-feature and
spectral derivatives brings the total to approximately $p' \approx 260$ columns.

### 2.2 Temporal Derivatives

For each selected feature $f = f_j$ and each bar $t$, the following
derivatives are computed. Notation: $f(t) \equiv f_j(t)$.

**Velocity (1st discrete difference).**

$$
v(t) = f(t) - f(t-1)
$$

Output column: `{col}_vel`. $v(0) = \text{NaN}$.

**Acceleration (2nd discrete difference).**

$$
a(t) = v(t) - v(t-1) = f(t) - 2f(t-1) + f(t-2)
$$

Output column: `{col}_accel`. $a(0) = a(1) = \text{NaN}$.

**Rolling z-score** at window $w$.

Let $\mu_w(t) = \frac{1}{w}\sum_{s=t-w+1}^{t} f(s)$ and
$\sigma_w(t) = \sqrt{\frac{1}{w-1}\sum_{s=t-w+1}^{t}(f(s)-\mu_w(t))^2}$ (Bessel-corrected).

$$
z_w(t) = \begin{cases}
\dfrac{f(t) - \mu_w(t)}{\sigma_w(t)} & \text{if } \sigma_w(t) \geq 10^{-10} \text{ and } t \geq w-1 \\
0 & \text{if } \sigma_w(t) < 10^{-10} \text{ and } t \geq w-1 \\
\text{NaN} & \text{if } t < w-1
\end{cases}
$$

Output column: `{col}_zscore_{w}`.

**Rolling OLS slope** at window $w$.

Within a rolling window ending at $t$, assign $x_i = i \in \{0, 1, \ldots, w-1\}$ and
$y_i = f(t - w + 1 + i)$. The closed-form OLS slope is:

$$
\hat{\beta}_w(t) = \frac{w \sum_{i=0}^{w-1} x_i y_i - \sum_{i=0}^{w-1} x_i \cdot \sum_{i=0}^{w-1} y_i}{w \sum_{i=0}^{w-1} x_i^2 - \left(\sum_{i=0}^{w-1} x_i\right)^2}
$$

Because $x_i = i$ is fixed, precompute constants $S_x = w(w-1)/2$,
$S_{x^2} = w(w-1)(2w-1)/6$, and $\text{denom} = w S_{x^2} - S_x^2$.
The denominator is constant across all windows of the same size, so only
$\sum x_i y_i$ and $\sum y_i$ vary per window.

$$
\hat{\beta}_w(t) = \frac{w \cdot \mathbf{x}^\top \mathbf{y}(t) - S_x \cdot \mathbf{1}^\top \mathbf{y}(t)}{\text{denom}}
$$

Output column: `{col}_slope_{w}`. NaN when $t < w - 1$ or any value in the
window is NaN.

**Rolling volatility** at window $w$.

$$
\sigma_w^\text{rvol}(t) = \sigma_w(t) = \sqrt{\frac{1}{w-1}\sum_{s=t-w+1}^{t}\left(f(s) - \mu_w(t)\right)^2}
$$

This is the same rolling standard deviation used in the z-score numerator.  
Output column: `{col}_rvol_{w}`. NaN when $t < w - 1$.

**Total temporal derivative columns per feature:**
$2 + 3|W| = 2 + 9 = 11$ columns (for $W = \{5, 15, 30\}$).

### 2.3 Spectral Derivatives

Spectral derivatives are defined on a rolling window of $w_s = 30$ bars.  
For each bar $t \geq w_s - 1$ and each selected feature $f$:

**Preprocessing.** Extract the detrended segment:

$$
\tilde{f}(s) = f(t - w_s + 1 + s) - \bar{f}_\text{seg}, \quad s = 0, \ldots, w_s - 1
$$

where $\bar{f}_\text{seg} = w_s^{-1} \sum_{s=0}^{w_s-1} f(t - w_s + 1 + s)$.

**One-sided power spectrum.** Apply the real-valued DFT (via `numpy.fft.rfft`):

$$
X[m] = \sum_{s=0}^{w_s-1} \tilde{f}(s)\, e^{-2\pi i m s / w_s}, \quad m = 0, 1, \ldots, \lfloor w_s/2 \rfloor
$$

$$
P[m] = |X[m]|^2
$$

The DC component $P[0]$ is excluded from all spectral features (it is zeroed
by mean-subtraction).

**Frequency band definitions:**

$$
m_\text{low} = 1, \ldots, \lfloor w_s/5 \rfloor - 1 \quad \text{(low-frequency band)}
$$

$$
m_\text{high} = \lfloor 4w_s/5 \rfloor, \ldots, \lfloor w_s/2 \rfloor \quad \text{(high-frequency band)}
$$

**Band powers:**

$$
p_\text{low}(t) = \sum_{m=1}^{\lfloor w_s/5 \rfloor - 1} P[m]
$$

$$
p_\text{high}(t) = \sum_{m=\lfloor 4w_s/5 \rfloor}^{\lfloor w_s/2 \rfloor} P[m]
$$

**Spectral ratio** (high value = low-frequency dominant = trending regime):

$$
r_\text{spec}(t) = \frac{p_\text{low}(t)}{p_\text{high}(t) + \varepsilon}, \quad \varepsilon = 10^{-10}
$$

**Dominant period** (bars per cycle of the strongest frequency):

$$
m^* = \underset{m \geq 1}{\operatorname{argmax}}\, P[m]
$$

$$
T^*(t) = \begin{cases}
w_s / m^* & \text{if } P[m^*] > 2 \cdot \overline{P}_{m \geq 1} \\
\text{NaN} & \text{otherwise}
\end{cases}
$$

where $\overline{P}_{m \geq 1} = \frac{1}{\lfloor w_s/2 \rfloor} \sum_{m=1}^{\lfloor w_s/2 \rfloor} P[m]$.
The factor-of-2 threshold prevents noise peaks from being labelled as dominant.

Output columns per feature: `{col}_spec_low_30`, `{col}_spec_high_30`,
`{col}_spec_ratio_30`, `{col}_spec_period_30`.

> **Implementation note:** As of the current codebase the `include_spectral`
> flag in `generate_derivatives()` is defined but the spectral derivatives
> *are* implemented in `spectral_derivatives()` and concatenated when
> `include_spectral=True` (default). The earlier recorded discrepancy that
> spectral derivatives were a no-op has been resolved.

### 2.4 Cross-Feature Derivatives

Cross-feature derivatives test economic relationships between pairs of feature
categories. Default pairs are:

| Pair (a, b) | Operations |
|------------|-----------|
| entropy × volume | ratio, rolling correlation |
| order imbalance × spread | ratio |
| whale flow × total flow volume | ratio |
| toxic flow × illiquidity | ratio, rolling correlation |
| entropy × trend | rolling correlation, divergence |

For each resolved pair $(a, b)$, the following are computed.

**Ratio** (instantaneous):

$$
r_{a,b}(t) = \text{clip}\!\left(\frac{a(t)}{b(t) + \varepsilon},\ -100,\ +100\right), \quad \varepsilon = 10^{-10}
$$

NaN when either series is NaN. Residual $\pm\infty$ values (numerically
possible despite $\varepsilon$) are set to 0.  
Output column: `cross_{a}_{b}_ratio`.

**Rolling Pearson correlation** at window $w$:

$$
\rho_w(t) = \frac{\sum_{s=t-w+1}^{t}(a(s)-\mu_{a,w})(b(s)-\mu_{b,w})}{\sqrt{\sum_{s}(a(s)-\mu_{a,w})^2 \cdot \sum_{s}(b(s)-\mu_{b,w})^2}}
$$

Computed via `pandas.Series.rolling.corr`. Output range: $[-1, 1]$. NaN during
warmup ($t < w - 1$).  
Output column: `cross_{a}_{b}_corr_{w}`.

**Divergence** at window $w$ (z-score difference):

$$
\delta_w(t) = z_{a,w}(t) - z_{b,w}(t)
$$

where $z_{a,w}(t)$ and $z_{b,w}(t)$ are the rolling z-scores of $a$ and $b$
respectively (as defined in §2.2). When $\sigma_w < 10^{-10}$ the z-score is
set to 0 to avoid division by zero; the NaN mask from the warmup period is
preserved.  
Output column: `cross_{a}_{b}_div_{w}`.

**Column name shortening:** Common aggregation suffixes (`_mean`, `_std`,
`_last`, `_sum`, `_slope`, OHLC suffixes) are stripped from column names
before constructing cross-derivative names to keep them readable.

**Total derivative output:** $p' \approx 260$ columns. The warmup period is
$\max(w_\text{max}, w_s) = 30$ bars, during which derivatives involving
rolling computations are NaN.

---

## Stage 3: Dimensionality Reduction

### 3.1 Pre-PCA Column Filtering

**Input:** Derivative matrix $\mathbf{D} \in \mathbb{R}^{N \times p'}$ (NaN replaced
by 0 for variance/correlation computation; original NaN structure preserved in
output).

**Step 1 — Variance percentile filter.**

Compute per-column variance $\sigma_j^2$ (with NaN treated as 0):

$$
\tau_v = \text{percentile}_{10}\!\left(\{\sigma_j^2\}_{j=1}^{p'}\right)
$$

Drop column $j$ if $\sigma_j^2 < \tau_v$ or $\sigma_j^2 < 10^{-20}$ (hard
floor regardless of percentile). This removes the bottom 10% of columns by
information content.

**Step 2 — Greedy correlation deduplication.**

For each pair $(j, k)$ with $j < k$, compute the absolute Pearson correlation
$|\rho_{jk}|$. Sort pairs by $|\rho_{jk}|$ descending. Process in this order:
for the first unvisited pair exceeding the threshold, drop the column with
lower variance.

$$
\text{If } |\rho_{jk}| > 0.95: \quad \text{drop } \underset{\ell \in \{j,k\}}{\operatorname{argmin}}\, \sigma_\ell^2
$$

This greedy pass is iterated once over all pairs sorted by correlation
(pairs involving already-dropped columns are skipped). The result is a set of
surviving columns with pairwise $|\rho| \leq 0.95$.

**Output:** Filtered derivative matrix $\mathbf{D}' \in \mathbb{R}^{N \times p''}$
where $p'' < p'$.

### 3.2 Standardisation

$$
Z_{t,j} = \frac{D'_{t,j} - \mu_j}{\sigma_j}, \qquad
\mu_j = \frac{1}{N}\sum_t D'_{t,j}, \quad
\sigma_j = \sqrt{\frac{1}{N}\sum_t (D'_{t,j} - \mu_j)^2}
$$

Zero-std columns (constant columns that survived filtering) use $\sigma_j = 1$.
NaN values in $\mathbf{D}'$ are filled with 0 before PCA (after standardisation
parameters are computed from non-NaN entries via `ddof=0`).

### 3.3 Covariance Estimation

**Standard covariance** (when $N \geq 2p''$):

$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{N-1}\mathbf{Z}^\top \mathbf{Z}
$$

(NumPy `cov` with `rowvar=False`, which uses $N-1$ denominator.)

**Ledoit-Wolf shrinkage** (when $N < 2p''$, i.e. samples are scarce relative
to features):

$$
\hat{\boldsymbol{\Sigma}}_\text{LW} = (1 - \alpha)\, \hat{\boldsymbol{\Sigma}}_\text{sample} + \alpha\, \mu_\text{LW}\, \mathbf{I}
$$

where $\alpha$ and $\mu_\text{LW}$ (the shrinkage intensity and shrinkage
target scale) are determined analytically by minimising the expected
Frobenius loss $\mathbb{E}\|\hat{\boldsymbol{\Sigma}}_\text{LW} - \boldsymbol{\Sigma}_\text{true}\|_F^2$
under the Oracle Approximating Shrinkage (OAS) estimator
(scikit-learn `LedoitWolf`).

The shrinkage coefficient $\alpha$ is closed-form under Ledoit & Wolf (2004):

$$
\alpha^* = \min\!\left(1,\; \frac{(n^{-1}\|\hat{\boldsymbol{\Sigma}}\|_F^2 + \text{tr}(\hat{\boldsymbol{\Sigma}})^2)}{(n+1-2/p'')(\|\hat{\boldsymbol{\Sigma}}\|_F^2 - \text{tr}(\hat{\boldsymbol{\Sigma}})^2/p'')}\right)
$$

Ledoit-Wolf prevents the estimated covariance from being ill-conditioned when
data is scarce, which would otherwise make PCA components unstable.

### 3.4 PCA via Eigendecomposition

**Eigendecomposition** of $\hat{\boldsymbol{\Sigma}} \in \mathbb{R}^{p'' \times p''}$:

$$
\hat{\boldsymbol{\Sigma}} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top
$$

where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_{p''})$
with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{p''} \geq 0$, and
$\mathbf{V}$ is the matrix of corresponding eigenvectors (columns).

Computed via `numpy.linalg.eigh` (symmetric solver). Eigenvalues numerically
below 0 (floating-point noise) are clamped to 0.

**Component selection.** Compute the explained variance ratio:

$$
r_i = \frac{\lambda_i}{\sum_{j=1}^{p''} \lambda_j}
$$

Select the smallest $k$ such that

$$
\sum_{i=1}^{k} r_i \geq \tau_\text{var} = 0.95
$$

capped at $k_\text{max} = \min(50, N, p'')$. In practice $k \approx 10$–15.

**Projection:**

$$
\mathbf{X}_r = \mathbf{Z}\, \mathbf{V}_k^\top \in \mathbb{R}^{N \times k}
$$

where $\mathbf{V}_k \in \mathbb{R}^{k \times p''}$ contains the top-$k$
eigenvectors as rows.

Each element $X_{r,t,i}$ is the score of bar $t$ on principal component $i$.
Principal component $i$ explains fraction $r_i$ of the total derivative
feature variance.

**Top loadings** per component: for each PC $i$, the top 10 features by
$|V_{k,i,j}|$ are stored as the *loadings* — the original derivative features
that most contribute to that component.

---

## Stage 4: Clustering

The clustering stage discovers hierarchical market states: broad *macro regimes*
(2–4 states driven by slow-moving features) and finer *micro states* within each
regime (driven by the full derivative set).

### 4.1 Structure Existence Tests

Before fitting any cluster model, two complementary tests check whether
non-trivial structure exists in $\mathbf{X}_r$.

#### Hopkins Statistic

The Hopkins statistic $H$ measures clustering tendency by comparing two types of
nearest-neighbour distances:

**Step 1.** Sample $m = \max(5, \min(\lfloor 0.1 N \rfloor, \lfloor N/2 \rfloor))$
random reference points $\{\mathbf{q}_i\}_{i=1}^m$ uniformly from the
axis-aligned bounding box of $\mathbf{X}_r$.

**Step 2.** Compute $u_i = d(\mathbf{q}_i, \text{NN}_\text{data}(\mathbf{q}_i))$,
the Euclidean distance from each reference point to its nearest data point.

**Step 3.** Sample $m$ data points $\{\mathbf{p}_i\}_{i=1}^m$ uniformly at
random from $\mathbf{X}_r$ without replacement. Compute
$w_i = d(\mathbf{p}_i, \text{NN}_\text{data \setminus \{\mathbf{p}_i\}}(\mathbf{p}_i))$,
the distance to the *second* nearest neighbour in the dataset (the nearest
neighbour of a data point in its own dataset is itself).

**Step 4.**

$$
H = \frac{\sum_{i=1}^m u_i}{\sum_{i=1}^m u_i + \sum_{i=1}^m w_i}
$$

Under uniform randomness, $H \approx 0.5$ in expectation. Under strong
clustering, $u_i \gg w_i$ (random points land in empty space; data points
cluster together), so $H \to 1$.

**Threshold:** $H > 0.7$ indicates clustered structure.

> **Note on the classical formulation.** The classical Hopkins (1954) statistic
> uses $d^k$ (the $k$-th power of distance, where $k$ is the dimensionality)
> to provide a chi-squared distribution under the null hypothesis. This
> implementation uses raw $L_2$ distances (power = 1) for numerical stability
> in high dimensions: the $d$-th power amplifies small distance differences
> exponentially, causing the statistic to degenerate toward 0 or 1 when
> $d > 5$. The threshold of 0.7 was empirically calibrated for this variant.

#### Hartigan Dip Test

The Silverman-Hartigan dip test is applied to the first principal component
$\mathbf{X}_{r,\cdot,0}$ (the direction of maximum variance).

The dip statistic $D$ measures the maximum difference between the empirical CDF
$F_n(x)$ and the best-fitting unimodal distribution $G(x)$:

$$
D = \sup_x |F_n(x) - G(x)|
$$

The $p$-value is computed against the asymptotic null distribution of $D$ under
unimodality. A small $p$-value rejects unimodality.

**Threshold:** $p < 0.05$ indicates multimodality.

**Combined decision:**

| Hopkins | Dip $p$ | Recommendation |
|---------|---------|---------------|
| $H > 0.7$ | $p < 0.05$ | `proceed` |
| $H > 0.7$ or $p < 0.05$ | — | `weak_structure` |
| both fail | — | `no_structure` |

If `no_structure`, regime discovery returns early with all labels set to 0
(a single state) and `early_exit=True`. No clustering is attempted.

### 4.2 Autocorrelation Split for Macro Regimes

The macro regime step operates only on *slow-moving* (persistent) features.
A feature $j$ is classified as slow if its lag-5 autocorrelation exceeds 0.7.

**Lag-$\ell$ autocorrelation:**

$$
\rho_j(\ell) = \frac{\sum_{t=0}^{N-\ell-1}(f_j(t) - \bar{f}_j)(f_j(t+\ell) - \bar{f}_j)}{(N - \ell)\, \widehat{\text{Var}}(f_j)}
$$

**Slow-feature set:**

$$
\mathcal{S} = \{j : \rho_j(5) > 0.7\}
$$

If $|\mathcal{S}| < 2$, all columns are used with a warning (insufficient slow
features).

The micro-state step uses the *full* derivative set (all columns of
$\mathbf{D}'$, not just slow columns), because micro states are defined
within a single macro regime and benefit from faster-moving features.

### 4.3 GMM k-Sweep with BIC

For each candidate number of components $k \in \{2, 3, 4, 5\}$, fit a
Gaussian Mixture Model with full covariance matrices:

$$
p(\mathbf{x}) = \sum_{i=1}^k \pi_i\, \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)
$$

Parameters $\{\pi_i, \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i\}$ are estimated
by the Expectation-Maximisation algorithm. With `n_init=5` random
initialisations, the best (highest log-likelihood) solution is retained.

**Bayesian Information Criterion:**

$$
\text{BIC}(k) = -2\ln \hat{L}(k) + k_\text{params}(k) \cdot \ln N
$$

where $\hat{L}(k)$ is the maximised log-likelihood and $k_\text{params}(k)$
is the number of free parameters:

$$
k_\text{params}(k) = k - 1 + k \cdot p'' + k \cdot \frac{p''(p''+1)}{2}
$$

($(k-1)$ mixing weights, $k \cdot p''$ mean parameters, $k \cdot p''(p''+1)/2$
unique covariance entries per component for full covariance type).

**Optimal cluster count:**

$$
k^* = \underset{k \in \{2,3,4,5\}}{\operatorname{argmin}}\; \text{BIC}(k)
$$

BIC penalises model complexity, favouring fewer components unless additional
components substantially improve the log-likelihood.

### 4.4 Cluster Quality Metrics

After fitting the final GMM at $k^*$ and obtaining hard assignments
$\ell(t) = \operatorname{argmax}_i \gamma_i(t)$:

**Silhouette score.** For each sample $t$ with label $\ell(t) = c$:

$$
a(t) = \frac{1}{|C_c| - 1} \sum_{s \in C_c, s \neq t} \|\mathbf{x}_t - \mathbf{x}_s\|_2
$$

$$
b(t) = \min_{c' \neq c} \frac{1}{|C_{c'}|} \sum_{s \in C_{c'}} \|\mathbf{x}_t - \mathbf{x}_s\|_2
$$

$$
s(t) = \frac{b(t) - a(t)}{\max(a(t),\, b(t))}
$$

Mean silhouette:

$$
\bar{s} = \frac{1}{N}\sum_{t=0}^{N-1} s(t) \in [-1, 1]
$$

$\bar{s} = 1$ means all samples are far closer to their own cluster than to
any other; $\bar{s} < 0$ means many samples are closer to a different cluster.

**Self-transition rate (STR):**

$$
\text{STR} = \frac{|\{t \in \{0,\ldots,N-2\} : \ell(t) = \ell(t+1)\}|}{N - 1}
$$

Fraction of consecutive bar pairs where the state does not change. High STR
indicates persistent, temporally coherent states.

**Duration distributions.** A *run* is a maximal contiguous sequence of bars
with the same label. Let $R_c$ be the multiset of run lengths for state $c$:

$$
\bar{d}_c = \frac{1}{|R_c|}\sum_{r \in R_c} r, \qquad \bar{d} = \frac{1}{\sum_c |R_c|}\sum_c \sum_{r \in R_c} r
$$

### 4.5 Block Bootstrap Stability

To measure how reproducible the clustering is, the GMM is re-fitted on 30
temporally-resampled datasets and compared to the reference labels via the
Adjusted Rand Index.

**Block resampling.** Divide the time series into blocks of $B = 15$ bars.
Sample $\lfloor N / B \rfloor$ block start indices uniformly with replacement
from $\{0, 1, \ldots, N - B\}$. Concatenate the selected blocks, trimming to
$N$ samples. This preserves within-block autocorrelation while breaking
between-block correlations.

**Adjusted Rand Index.** Given reference labels $\boldsymbol{\ell}^*$ and
bootstrap labels $\boldsymbol{\ell}^b$:

$$
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
$$

where $\text{RI} = (a + d) / \binom{N}{2}$ counts concordant pairs ($a$ =
same label in both, $d$ = different label in both).
ARI = 1 means perfect agreement; ARI $\approx$ 0 means random agreement.

$$
\overline{\text{ARI}} = \frac{1}{30}\sum_{b=1}^{30} \text{ARI}(\boldsymbol{\ell}^*, \boldsymbol{\ell}^b)
$$

### 4.6 Hierarchical Label Assembly

**Macro discovery** runs on the slow-feature subset $\mathcal{S}$ to produce
$k^*_\text{macro}$ regimes $\{0, 1, \ldots, k^*_\text{macro}-1\}$.

**Micro discovery** runs independently for each macro regime $c$. Let
$\mathcal{T}_c = \{t : \ell_\text{macro}(t) = c\}$ be the bars in that regime.
The full derivative matrix $\mathbf{D}'[\mathcal{T}_c, :]$ is passed through
the full pipeline (reduce → structure test → GMM k-sweep) to produce
$k^*_{\text{micro},c}$ micro states.

**Global micro labels.** Each micro state is given a global integer ID:

$$
\ell_\text{micro}(t) = \sum_{c'=0}^{c-1} k^*_{\text{micro},c'} + \ell_{\text{micro}|c}(t)
$$

where $c = \ell_\text{macro}(t)$ and $\ell_{\text{micro}|c}(t)$ is the local
micro label within macro regime $c$.

Total micro states: $K_\text{total} = \sum_{c=0}^{k^*_\text{macro}-1} k^*_{\text{micro},c}$.

---

## Stage 5: Validation — Quality Gates

Three sequential gates must pass for the pipeline to output actionable states.
Each gate tests a distinct property of the discovered states.

### Q1 — Structural Quality

Tests whether the clusters are statistically real (not a fitting artefact).

**Condition:**

$$
\bar{s} \geq 0.25 \quad \text{AND} \quad \overline{\text{ARI}} \geq 0.6
$$

- $\bar{s}$ is the mean silhouette score (§4.4)
- $\overline{\text{ARI}}$ is the mean block-bootstrap ARI (§4.5)

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Silhouette $\bar{s}$ | $\geq 0.25$ | Clusters are clearly separated in PCA space |
| Bootstrap ARI | $\geq 0.60$ | Re-fitting on resampled data recovers the same structure |

**Outcome if Q1 fails:** `DROP` — the clusters are not reproducible and should
not be used.

### Q2 — Predictive Quality

Tests whether market states predict forward returns (the core economic
hypothesis).

**Forward log-returns.** At bar $t$ and horizon $h \in \{1, 5, 10, 20\}$:

$$
r_h(t) = \ln P(t + h) - \ln P(t)
$$

where $P(t)$ is the close price of bar $t$.

**Kruskal-Wallis test.** Group forward returns by micro state label. The
Kruskal-Wallis $H$-statistic tests whether the return distributions are
identical across states (null hypothesis):

$$
H = \frac{12}{n(n+1)}\sum_{c=0}^{K-1} \frac{R_c^2}{n_c} - 3(n+1)
$$

where $n = \sum_c n_c$ is the total number of observations, $n_c = |\{t : \ell(t) = c, t + h < N\}|$ is
the sample count for state $c$ at horizon $h$, and $R_c = \sum_{t:\ell(t)=c} \text{rank}(r_h(t))$
is the sum of ranks for that state (ranks computed over all observations jointly).

Under the null, $H \sim \chi^2_{K-1}$.

**Effect size — eta-squared:**

$$
\eta^2 = \frac{H}{n - 1}
$$

This is the fraction of total rank variance explained by state membership.

**Pass condition at horizon $h$:**

$$
p_h < 0.05 \quad \text{AND} \quad \eta^2_h \geq 0.01
$$

Both conditions must hold: statistical significance (small $p$) and practical
significance (non-trivial effect size). The pipeline passes Q2 if this
condition holds for **any** $h \in \{1, 5, 10, 20\}$.

**Outcome if Q2 fails (with Q1 passing):** `COLLECT` — structure is real but
does not yet predict returns; more data may resolve it.

### Q3 — Operational Quality

Tests whether states are persistent enough to be tradeable.

**Condition:**

$$
\text{STR}_\text{macro} \geq 0.8 \quad \text{AND} \quad \bar{d} \geq 3
$$

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Self-transition rate | $\geq 0.80$ | Regime changes less than once every 5 bars on average |
| Mean duration $\bar{d}$ | $\geq 3$ bars | Average regime lasts at least 45 minutes (3 × 15 min) |

Both metrics are computed on the *macro* label sequence $\ell_\text{macro}(t)$.

**Outcome if Q3 fails (with Q1 and Q2 passing):** `PIVOT` — states predict
returns but switch too rapidly to trade; consider longer bar aggregation or
additional smoothing.

### Decision Matrix

$$
\text{verdict} = \begin{cases}
\texttt{DROP}    & \text{if Q1 fails} \\
\texttt{COLLECT} & \text{if Q1 passes, Q2 fails} \\
\texttt{PIVOT}   & \text{if Q1 and Q2 pass, Q3 fails} \\
\texttt{GO}      & \text{if all pass}
\end{cases}
$$

### Per-State Verdict

Each micro state $c$ independently receives a `GO` or `COLLECT` verdict based
on whether its Sharpe ratio exceeds $0.3$ at any horizon:

$$
\text{Sharpe}_h(c) = \frac{\bar{r}_h(c)}{\hat{\sigma}_h(c) \cdot \sqrt{N/h}}
$$

where $\bar{r}_h(c)$ and $\hat{\sigma}_h(c)$ are the sample mean and standard
deviation of forward returns $\{r_h(t) : \ell(t) = c\}$.

A state requires at least 10 observations at that horizon; otherwise it is
marked `COLLECT` by default.

---

## Stage 6: Transition Matrix

Given the final micro label sequence $\boldsymbol{\ell} = (\ell(0), \ell(1), \ldots, \ell(N-1))$
with $K_\text{total}$ unique states, the transition matrix is a complete
description of the Markov structure of regime dynamics.

### 6.1 Count Matrix

$$
T[i, j] = \left|\{t \in \{0, \ldots, N-2\} : \ell(t) = i \text{ and } \ell(t+1) = j\}\right|
$$

$T \in \mathbb{Z}_{\geq 0}^{K \times K}$ where $K = K_\text{total}$.

### 6.2 Row Normalisation

$$
P[i, j] = \frac{T[i, j]}{\sum_{j'=0}^{K-1} T[i, j']}
$$

$P$ is row-stochastic: $P[i, j] \geq 0$ and $\sum_j P[i, j] = 1$ for all $i$.

**Edge case:** If state $i$ has no outgoing transitions (only appears at the
last bar), the row $T[i, :]$ is set to the all-ones vector before
normalisation, yielding a uniform row $P[i, j] = 1/K$. This ensures $P$
remains row-stochastic even for terminal states.

### 6.3 Row Entropy

The Shannon entropy of the $i$-th row of $P$ measures the predictability of
the next state given that the current state is $i$:

$$
H(i) = -\sum_{j=0}^{K-1} P[i, j]\, \ln\!\left(P[i, j] + \varepsilon\right), \qquad \varepsilon = 10^{-15}
$$

The $\varepsilon$ prevents $\ln(0)$ for zero-probability transitions.

- $H(i) = 0$ nats: the next state is deterministic (STR = 1)
- $H(i) = \ln K$ nats: all transitions are equally likely (uniform row)

### 6.4 Most Likely Successor

For each state $i$, the most likely *next* state (excluding self-transition):

$$
j^*(i) = \underset{j \neq i}{\operatorname{argmax}}\; P[i, j]
$$

When $K = 1$, the only state succeeds itself by definition.

### 6.5 Duration Distributions

From the label sequence, extract all maximal contiguous runs of each state.
Let $R_c = (d_1^c, d_2^c, \ldots)$ be the vector of run lengths (in bars) for
state $c$:

$$
\bar{d}_c = \frac{1}{|R_c|} \sum_{r \in R_c} r, \quad
\bar{d}_c^2 = \frac{1}{|R_c|} \sum_{r \in R_c} r^2, \quad
\text{Var}(d_c) = \bar{d}_c^2 - (\bar{d}_c)^2
$$

Duration in calendar time: $\bar{d}_c \times 15$ minutes per bar.

**Geometric distribution comparison.** For a first-order Markov chain with
self-transition probability $p_{ii} = P[i, i]$, run lengths are geometrically
distributed:

$$
\Pr(d = k) = (1 - p_{ii})\, p_{ii}^{k-1}, \quad k = 1, 2, \ldots
$$

with mean $\mathbb{E}[d] = 1/(1 - p_{ii})$.

The observed $\bar{d}_c$ can be compared to $1/(1 - P[c,c])$ to diagnose
non-Markovian behaviour (e.g. state dependencies beyond one-step memory).

---

## Summary of Thresholds and Defaults

| Parameter | Value | Stage | Rationale |
|-----------|-------|-------|-----------|
| Bar timeframe | 15 min | 1 | Primary analysis horizon |
| NaN drop threshold $\tau_\text{nan}$ | 0.5 | 1 | Drop if >50% NaN |
| Variance floor (preprocessing) | $10^{-10}$ | 1 | Constant-column rejection |
| Outlier clip $\sigma$ | 5 | 1 | Preserve genuine extremes |
| Base features $D$ | 15 | 2 | Derivative explosion control |
| Rolling windows $W$ | $\{5, 15, 30\}$ | 2 | Multi-scale dynamics |
| Spectral window $w_s$ | 30 | 2 | Cycle detection |
| Ratio clip | 100 | 2 | Outlier cross-feature ratios |
| Variance percentile filter | 10th | 3 | Remove bottom 10% |
| Hard variance floor (reduction) | $10^{-20}$ | 3 | After percentile filter |
| Correlation dedup threshold | 0.95 | 3 | Near-perfect redundancy |
| PCA variance target $\tau_\text{var}$ | 0.95 | 3 | 95% variance captured |
| Max PCA components | 50 | 3 | Hard cap |
| LW regularisation condition | $N < 2p''$ | 3 | Scarce data regime |
| Hopkins threshold | 0.7 | 4 | Clustering tendency |
| Dip test significance | 0.05 | 4 | Multimodality |
| AC slow-feature threshold | 0.7 at lag 5 | 4 | Persistent features |
| GMM k range | $\{2, 3, 4, 5\}$ | 4 | Macro regimes |
| GMM n_init | 5 | 4 | EM initialisation robustness |
| Bootstrap iterations | 30 | 4 | Stability estimation |
| Bootstrap block size | 15 bars | 4 | Preserve autocorrelation |
| Q1 silhouette threshold | 0.25 | 5 | Structural quality |
| Q1 ARI threshold | 0.60 | 5 | Stability quality |
| Q2 Kruskal-Wallis $p$ | 0.05 | 5 | Predictive significance |
| Q2 $\eta^2$ (effect size) | 0.01 | 5 | Practical significance |
| Q2 return horizons $h$ | $\{1, 5, 10, 20\}$ | 5 | Multi-horizon test |
| Q2 pass mode | any horizon | 5 | At least one must pass |
| Q3 STR threshold | 0.80 | 5 | Regime persistence |
| Q3 min duration $\bar{d}$ | 3 bars | 5 | $\geq$ 45 min |
| Per-state Sharpe threshold | 0.30 | 5 | Tradeable edge |
| Entropy epsilon $\varepsilon$ | $10^{-15}$ | 6 | Avoid $\ln(0)$ |

---

## Known Implementation Notes

The following discrepancies between the specification and implementation are
documented for research reproducibility:

1. **Hopkins statistic.** The classical formulation raises distances to the
   $d$-th power (dimension). The implementation uses raw $L_2$ distances for
   numerical stability in high-dimensional PCA space. The threshold of 0.7
   was calibrated for this variant.

2. **Q1 temporal ARI.** A temporal ARI threshold of 0.5 is defined in the
   original spec but is not evaluated in `_evaluate_q1()`. Q1 checks only
   silhouette and block-bootstrap ARI.

3. **Q3 micro checks.** Per-micro-state self-transition rate and entry-lead
   metrics are defined in the spec but not checked in `_evaluate_q3()`.
   Q3 gates only on macro STR and macro mean duration.

4. **Block-break detection.** The spec describes PELT change-point detection
   on the first 5 PCA components. The implementation applies PELT to all
   numeric derivative columns. This is documented as an intentional deviation
   that provides more comprehensive break detection.

5. **Cross-symbol consistency.** `cross_symbol_consistency()` accepts
   pre-computed label dictionaries, not raw data. It does not call
   `profile()` internally.

6. **Non-current regime probabilities.** In `OnlineClassifier`, the
   `all_probabilities` field of `StateEstimate` uses a uniform approximation
   for non-current macro regimes (they are not re-evaluated at each step).

---

## References

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*,
  88(2), 365–411.
- Hopkins, B. and Skellam, J. G. (1954). A new method for determining the
  type of distribution of plant individuals. *Annals of Botany*, 18(2),
  213–227.
- Hartigan, J. A. and Hartigan, P. M. (1985). The dip test of unimodality.
  *Annals of Statistics*, 13(1), 70–84.
- Kruskal, W. H. and Wallis, W. A. (1952). Use of ranks in one-criterion
  variance analysis. *Journal of the American Statistical Association*,
  47(260), 583–621.
- McLachlan, G. J. and Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of
  Statistics*, 6(2), 461–464. (BIC criterion)
- Hubert, L. and Arabie, P. (1985). Comparing partitions. *Journal of
  Classification*, 2(1), 193–218. (Adjusted Rand Index)
