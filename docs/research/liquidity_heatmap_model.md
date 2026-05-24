# Liquidity Heatmap and Cascade Probability Model

**Status:** Design document / theoretical basis  
**Extends:** `features/illiquidity.rs`, `features/liquidation.rs`, `features/concentration.rs`  
**Hypothesis:** H3 validated (cascade lift > 2×); this model operationalises it as a real-time probability estimator.

---

## Notation Reference

| Symbol | Type | Domain | Meaning |
|--------|------|--------|---------|
| $t$ | time index | $\mathbb{Z}_{\geq 0}$ | Snapshot timestamp (100ms resolution) |
| $m_t$ | scalar | $(0, \infty)$ | Mid-price at time $t$: $(P^{\text{bid}}_t + P^{\text{ask}}_t)/2$ |
| $\delta$ | scalar | $(-R, R)$ | Log-return offset from mid, $\delta = \ln(P/m_t)$; $R = 0.10$ |
| $k$ | integer | $\{0, \ldots, K-1\}$ | Bin index along the $\delta$ axis |
| $\Delta$ | scalar | $(0, R/K]$ | Bin half-width (default $\Delta = 0.001$, i.e. 10 bps) |
| $K$ | integer | $\mathbb{Z}_{>0}$ | Number of bins; $K = \lceil 2R / \Delta \rceil = 200$ by default |
| $i$ | integer | $\{1, \ldots, N_t\}$ | Position index at snapshot $t$ |
| $V_i$ | scalar | $(0, \infty)$ | Notional value of position $i$ in USD |
| $\ell_i$ | scalar | $(0, \infty)$ | Liquidation price of position $i$ |
| $s_i$ | $\pm 1$ | $\{-1, +1\}$ | Side: $+1$ = long (liquidated by downward price move), $-1$ = short |
| $H(t, k)$ | scalar | $[0, \infty)$ | Heatmap value: USD liquidation mass in bin $k$ at time $t$ |
| $D^{\text{bid}}(r)$ | scalar | $[0, \infty)$ | Total bid-side book depth (USD) within $r\%$ of mid |
| $\lambda^{\text{Kyle}}$ | scalar | $[0, \infty)$ | Kyle's price impact coefficient (from `illiq_kyle_100`) |
| $\mathcal{I}$ | scalar | $[0, \infty)$ | Composite illiquidity score (`illiq_composite`) |
| $\sigma(\cdot)$ | function | $\mathbb{R} \to (0,1)$ | Logistic sigmoid: $\sigma(z) = 1/(1+e^{-z})$ |
| $T$ | integer | $\mathbb{Z}_{>0}$ | Prediction horizon in 100ms ticks |
| $\tau_{\text{cluster}}$ | scalar | $(0, \infty)$ | Cluster detection threshold in USD (default \$1M) |
| $\tau_V$ | scalar | $(0, \infty)$ | Minimum position value filter (default \$1,000) |

---

## 1. Liquidity Heatmap Construction

### 1.1 Price Axis Discretisation

The price axis is discretised into $K$ uniform bins indexed by $k \in \{0, \ldots, K-1\}$. Each bin corresponds to a log-return offset $\delta_k$ from the current mid-price $m_t$:

$$\delta_k = -R + \left(k + \tfrac{1}{2}\right)\Delta, \qquad k = 0, \ldots, K-1$$

where $R = 0.10$ (10% half-range) and $\Delta = 2R/K$. Bin $k$ covers the open interval

$$\mathcal{B}_k = \left(-R + k\Delta,\; -R + (k+1)\Delta\right).$$

The absolute price range of bin $k$ at time $t$ is

$$\mathcal{P}_k(t) = \left(m_t \cdot e^{-R + k\Delta},\; m_t \cdot e^{-R + (k+1)\Delta}\right).$$

Using log-returns (rather than arithmetic offsets) ensures bins are symmetric in percentage terms and the mapping is independent of the absolute price level.

### 1.2 Heatmap Function

The **liquidation USD density** at bin $k$, time $t$ is the total notional value of all positions whose liquidation price falls within $\mathcal{B}_k(t)$:

$$\boxed{H(t, k) = \sum_{i=1}^{N_t} V_i \cdot \mathbf{1}\!\left[\ell_i \in \mathcal{P}_k(t)\right] \cdot \mathbf{1}\!\left[V_i \geq \tau_V\right]}$$

**Variables:**
- $V_i$ — notional position value (USD), always positive
- $\ell_i$ — liquidation price of position $i$
- $\mathcal{P}_k(t)$ — price interval of bin $k$ at time $t$
- $\tau_V$ — minimum position value filter (removes dust positions)

**Computational complexity:** $O(N_t \cdot K)$ naively; $O(N_t \log K)$ with sorted position array and binary search assignment.

**Relation to existing features:** This generalises `liquidation_risk_above_Xpct` and `liquidation_risk_below_Xpct` in `liquidation.rs`. Those compute cumulative sums at four fixed thresholds (1%, 2%, 5%, 10%); $H(t,k)$ is the full spatial distribution from which those scalars can be recovered as:

$$\text{risk\_above\_}r\% = \sum_{k: \delta_k \in (0, \ln(1+r/100)]} H(t,k)$$

### 1.3 Four-Channel Tensor

Each snapshot yields a tensor $\mathbf{H}(t) \in \mathbb{R}^{K \times 4}$ with the following channels:

**Channel 1 — Liquidation USD Density**

$$\mathbf{H}_{k,1}(t) = H(t, k)$$

Raw mass per bin. Units: USD. This is the primary spatial representation.

**Channel 2 — Cumulative Liquidation Mass**

The cumulative mass running outward from the mid (bin $k_0 = K/2$) captures the potential **cascade chain**: if a cluster at distance $d_1$ liquidates and the price impact reaches the next cluster at $d_2 > d_1$, the cumulative mass up to $d_2$ represents the combined fuel available.

$$\mathbf{H}_{k,2}(t) = \begin{cases} \displaystyle\sum_{j=k}^{k_0} H(t, j) & k < k_0 \quad \text{(downward cumulation, long cascade)} \\[8pt] \displaystyle\sum_{j=k_0}^{k} H(t, j) & k \geq k_0 \quad \text{(upward cumulation, short cascade)} \end{cases}$$

Units: USD. Interpretation: $\mathbf{H}_{k,2}$ at bin $k$ gives the total liquidation mass between the mid and price level $\mathcal{P}_k$ — the amount that would be forced into the market if price reaches $k$.

**Channel 3 — Cluster Gradient**

The discrete spatial derivative of the heatmap identifies abrupt density spikes (dense clusters) versus diffuse distributions:

$$\mathbf{H}_{k,3}(t) = H(t, k) - H(t, k-1), \qquad k = 1, \ldots, K-1$$

with $\mathbf{H}_{0,3}(t) = 0$. Units: USD/bin. Sharp positive spikes in $|\mathbf{H}_{k,3}|$ indicate cluster boundaries. This is the discrete approximation of $\partial H / \partial \delta$ at resolution $\Delta$.

**Channel 4 — Temporal Change**

The migration velocity of liquidation mass is estimated at two timescales:

$$\mathbf{H}_{k,4}(t) = \alpha_1 \cdot \frac{H(t, k) - H(t - \Delta t_1, k)}{\Delta t_1} + \alpha_2 \cdot \frac{H(t, k) - H(t - \Delta t_2, k)}{\Delta t_2}$$

where $\Delta t_1 = 600$ ticks (1 min) and $\Delta t_2 = 3000$ ticks (5 min), $\alpha_1 = 0.7$, $\alpha_2 = 0.3$ (weights sum to 1; emphasise the finer timescale). Units: USD/tick.

Positive values indicate growing mass (positions being opened or migrating inward); negative values indicate dissolving mass. This channel serves as a **cluster migration velocity** field.

---

## 2. Spatial Feature Extraction

Eight scalar features are extracted from $\mathbf{H}(t)$ per snapshot. These are the microstructure analogue of physics observables on a density field.

### F1 — Nearest Cluster Distance

$$d_{\min}(t) = \min\left\{|\delta_k| : H(t,k) > \tau_{\text{cluster}}\right\}$$

where $|\delta_k| = |{-R + (k + \tfrac{1}{2})\Delta}|$ is the absolute log-return distance of bin $k$ from mid. Units: log-return (dimensionless, $\approx$ fraction for small values).

**Edge case:** if no bin exceeds $\tau_{\text{cluster}}$, return $R$ (heatmap range boundary). This matches the existing `nearest_cluster_distance` in `liquidation.rs` at coarser resolution (0.5% bins, fixed $\tau = \$1\text{M}$), which is subsumed here at finer resolution (10 bps).

**Complexity:** $O(K)$.

### F2 — Cluster Mass Ratio

Let $k^* = \arg\max_k H(t,k)$ be the bin with the highest liquidation mass. The cluster mass ratio normalises peak density by expected density:

$$\text{CMR}(t) = \frac{H(t, k^*)}{\bar{H}(t)}, \qquad \bar{H}(t) = \frac{1}{K} \sum_{k=0}^{K-1} H(t, k)$$

Range: $[1, N_t]$ (minimum 1 by definition; maximum $N_t$ if all mass is in one bin). Dimensionless. A value of 10 means the peak bin holds 10× the average bin density — a highly concentrated cluster.

**Relation to existing features:** Extends `largest_position_at_risk` (a single-position maximum) to a population-level concentration measure.

### F3 — Cascade Chain Length

The cascade chain length counts the number of **consecutive** bins above threshold within the critical window $[-5\%, +5\%]$ of mid (i.e., bins within $|\delta_k| \leq 0.05$):

$$L(t) = \max_{\text{consecutive runs}} \left|\left\{k : H(t,k) > \tau_{\text{chain}},\; |\delta_k| \leq 0.05\right\}\right|_{\text{consecutive}}$$

where $\tau_{\text{chain}} = \tau_{\text{cluster}}/10$ (one order of magnitude below cluster threshold, to catch sub-threshold bridge mass). Units: bins (multiply by $\Delta$ to convert to log-return units).

**Rationale (Brunnermeier & Pedersen 2009):** A cascade requires not just a dense cluster but a **chain** — consecutive price levels where forced sellers push price incrementally through successive liquidation zones. A long chain indicates that the liquidation spiral is geometrically feasible.

**Complexity:** $O(K)$ (single left-to-right scan).

### F4 — Asymmetric Cascade Potential

The directional imbalance of liquidation mass within the critical 5% window:

$$\text{ACP}(t) = \frac{\displaystyle\sum_{k: 0 < \delta_k \leq 0.05} H(t,k) - \displaystyle\sum_{k: -0.05 \leq \delta_k < 0} H(t,k)}{\displaystyle\sum_{k: |\delta_k| \leq 0.05} H(t,k) + \epsilon}$$

Range: $(-1, +1)$. $\epsilon = 1$ USD to prevent division by zero. 

- $\text{ACP} > 0$: more short liquidations above mid — upward cascade pressure (short squeeze)
- $\text{ACP} < 0$: more long liquidations below mid — downward cascade pressure (long cascade)

**Relation to existing features:** Generalises `liquidation_asymmetry` (ratio of 5% above vs below) to a normalised difference, preserving sign and bounding to $(-1,1)$.

### F5 — Absorption Capacity

The ratio of order-book depth to the downward liquidation mass pressure within 2% of mid:

$$A(t) = \frac{D^{\text{bid}}(0.02)}{\displaystyle\sum_{k: -0.02 \leq \delta_k < 0} H(t,k) + \epsilon}$$

where $D^{\text{bid}}(r)$ is the total USD depth on the bid side within $r$ of mid (computable from `raw_bid_depth_10` and the book's price levels). Range: $(0, \infty)$.

**Physical interpretation (Cont & Wagalath 2016):** When $A \gg 1$, the order book has sufficient depth to absorb forced sell flow without a cascade. When $A < 1$, liquidation volume exceeds available book depth — the necessary condition for price to gap through multiple levels, triggering the domino sequence.

**Directional note:** The formula above is stated for the downward direction. For the upward direction, substitute $D^{\text{ask}}(0.02)$ and the upward mass sum. In practice, compute both and use the minimum as the worst-case absorption metric.

### F6 — Cluster Velocity

The rate of change of the nearest cluster distance, estimated via backward finite difference:

$$v(t) = \frac{d_{\min}(t) - d_{\min}(t - \tau)}{\tau}$$

where $\tau = 600$ ticks (1 min) is the lookback. Units: log-return/tick ($\approx$ fraction/100ms). Negative $v$ means the nearest cluster is approaching mid.

**Interpretation:** A cluster approaching at velocity $v < 0$ in a thin market (high $\lambda^{\text{Kyle}}$) is the primary entry signal — the interaction term $v \cdot \mathcal{I}$ (Feature 2 in §3.2) captures this joint condition.

**Note:** The finite difference introduces a lag of $\tau/2 = 30$ seconds in expectation. This is acceptable for cascade prediction at horizons $T \geq 5$ min, but may be noisy for shorter horizons.

### F7 — Mass-Weighted Distance (Centre of Liquidation Gravity)

$$\bar{d}(t) = \frac{\displaystyle\sum_{k=0}^{K-1} |\delta_k| \cdot H(t,k)}{\displaystyle\sum_{k=0}^{K-1} H(t,k) + \epsilon}$$

Units: log-return (dimensionless). Range: $[0, R]$. This is the mean absolute distance of liquidation mass from mid, weighted by mass.

- Low $\bar{d}$: liquidation mass concentrated close to mid — high immediate risk
- High $\bar{d}$: liquidation mass distributed far from mid — lower immediate risk

**Relation to existing features:** Complements `positions_at_risk_count` (a count) and `liquidation_intensity` (intensity at fixed 5% window) with a continuous spatial summary of the full distribution.

### F8 — Heatmap Entropy

The Shannon entropy of the normalised liquidation mass distribution:

$$S(t) = -\sum_{k=0}^{K-1} p_k(t) \ln p_k(t), \qquad p_k(t) = \frac{H(t,k)}{\displaystyle\sum_{j=0}^{K-1} H(t,j) + \epsilon}$$

with the convention $0 \ln 0 = 0$. Range: $[0, \ln K]$. Low entropy = highly concentrated mass (one or few dominant clusters). High entropy = uniformly distributed mass.

**Relation to existing features:** The entropy features in `features/entropy.rs` (`ent_*`) operate on trade arrival sequences and price changes in time. $S(t)$ is a spatial entropy of the liquidation landscape — orthogonal to those time-series measures. Conceptually analogous to `ent_permutation_entropy` but over the price axis rather than the time axis.

---

## 3. Cascade Probability Model

### 3.1 Event Definition

A **cascade event** at time $t$ with horizon $T$ and thresholds $(X, Y)$ is defined as:

$$y_t = \mathbf{1}\!\left[|\Delta p_{t \to t+T}| > X\% \;\wedge\; \text{LiqVol}_{t \to t+T} > Y\right]$$

where:
- $\Delta p_{t \to t+T} = \ln(m_{t+T}/m_t)$ — log-return over horizon $T$
- $\text{LiqVol}_{t \to t+T}$ — total USD notional of forced liquidations in $(t, t+T]$, observable from Hyperliquid position updates
- $(X, Y)$ — calibrated jointly to achieve base rate $\approx 5\%$; preliminary values: $X = 3\%$, $Y = \$500\text{K}$, $T = 300$ ticks (5 min)

The conjunction requirement is critical: it distinguishes true cascades (price move driven by forced liquidation flow) from ordinary large moves. This is the same operational definition implicit in H3 validation.

### 3.2 Feature Vector

The cascade predictor uses an 8-dimensional feature vector augmented with three interaction terms:

$$\mathbf{x}(t) = \left[d_{\min},\; \text{CMR},\; L,\; \text{ACP},\; A^{-1},\; v,\; \bar{d},\; S,\; L \cdot A^{-1},\; v \cdot \mathcal{I},\; \text{ACP} \cdot \text{sgn}(f)\right]^{\top} \in \mathbb{R}^{11}$$

where $A^{-1} = 1/A$ (reciprocal absorption capacity, so that low absorption gives high feature value), $\mathcal{I}$ is `illiq_composite` from `illiquidity.rs`, and $f$ is the perpetual funding rate from `ctx_funding_rate`.

**Interaction term rationale:**

| Term | Interpretation |
|------|----------------|
| $L \cdot A^{-1}$ | Long chain with insufficient damping — fuel + no absorber |
| $v \cdot \mathcal{I}$ | Cluster approaching in thin market — impact amplified |
| $\text{ACP} \cdot \text{sgn}(f)$ | Directional alignment: ACP and funding rate both point the same way; positive funding + upward ACP amplifies short-squeeze risk |

### 3.3 Logistic Regression Model

$$P(\text{cascade at } t \mid \mathbf{x}(t)) = \sigma(\beta_0 + \boldsymbol{\beta}^{\top} \mathbf{x}(t)) = \frac{1}{1 + \exp(-\beta_0 - \boldsymbol{\beta}^{\top} \mathbf{x}(t))}$$

Parameters $(\beta_0, \boldsymbol{\beta}) \in \mathbb{R}^{12}$ are estimated by online logistic regression (SGD with exponential learning rate decay) to adapt to regime changes. The loss function is cross-entropy:

$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{n}\sum_{t=1}^{n}\left[y_t \ln \hat{p}_t + (1-y_t)\ln(1-\hat{p}_t)\right] + \frac{\lambda}{2}\|\boldsymbol{\beta}\|^2$$

with L2 regularisation $\lambda = 10^{-3}$ to prevent coefficient explosion on low-frequency cascade events.

**Feature normalisation:** All features are standardised online via running mean and variance (Welford's algorithm) before entering the model. This ensures $\boldsymbol{\beta}$ remains interpretable as relative effect sizes and prevents numerical instability.

**Why logistic (not, e.g., a deep model):** Cascade events are rare ($\approx 5\%$ base rate), data is non-stationary (changing leverage regimes), and interpretability of $\boldsymbol{\beta}$ is required for qualitative validation. A linear model in the log-odds space is the correct prior for a binary event with a clear physical mechanism. More complex models should only be attempted after the linear baseline is validated.

---

## 4. Cascade Physics: The Domino Mechanism

### 4.1 Single-Cluster Price Impact

Consider a cluster of mass $M_A = H(t, k_A)$ at log-distance $\delta_A > 0$ (above mid, short positions). If price reaches $m_t e^{\delta_A}$, these positions are force-liquidated as **market buy orders** of total USD value $M_A$.

The induced price impact, using Kyle's linear model (Kyle 1985), is:

$$\Delta p_A = \lambda^{\text{Kyle}} \cdot \frac{M_A}{m_t}$$

where $\lambda^{\text{Kyle}}$ is the price-impact coefficient estimated over the most recent 100 trades (`illiq_kyle_100`). Units: $\Delta p_A$ is in price units (USD per unit base).

### 4.2 Cascade Condition (Domino)

A second cluster of mass $M_B$ at distance $\delta_B > \delta_A$ is reached if and only if the price impact of cluster A's liquidation is sufficient to close the gap:

$$\boxed{\delta_B - \delta_A < \frac{\lambda^{\text{Kyle}} \cdot M_A}{m_t}}$$

If this condition holds, cluster B then liquidates, generating additional impact $\Delta p_B = \lambda^{\text{Kyle}} \cdot M_B / m_t$, and the cascade propagates. The total cascade price move is:

$$\Delta p_{\text{total}} \approx \lambda^{\text{Kyle}} \cdot \frac{1}{m_t} \sum_{j: \text{reached}} M_j$$

where the sum runs over all clusters reached by the self-reinforcing chain. This is the closed-form expression for cascade magnitude under the linear impact model.

**Key implication:** The cascade condition depends on the **spacing** between clusters (gap $\delta_B - \delta_A$) and the **cluster mass** $M_A$, mediated by $\lambda^{\text{Kyle}}$. The channel 3 (gradient) of the heatmap directly measures inter-cluster spacing by identifying bins where $H(t,k)$ jumps sharply — these jumps are the gaps between clusters.

### 4.3 Absorption Damping

The order book acts as a damper. During the cascade, each price increment must consume available bid depth $D^{\text{bid}}(\delta)$ before reaching the next cluster. The absorbed fraction of cluster A's impact is:

$$\eta = \min\!\left(1,\; \frac{D^{\text{bid}}(\delta_A)}{M_A}\right)$$

The **effective** price impact reaching cluster B is reduced to $\Delta p_A^{\text{eff}} = (1 - \eta) \cdot \Delta p_A$. Cascade is damped when $\eta \approx 1$, i.e., when $D^{\text{bid}} \gg M_A$, consistent with $A \gg 1$ in F5. This is the Brunnermeier & Pedersen (2009) liquidity spiral mechanism: thin markets amplify cascade magnitude because $\eta \to 0$.

### 4.4 Network Contagion Extension

When $N$ clusters are present at distances $\delta_1 < \delta_2 < \cdots < \delta_N$, the cascade can be modelled as a directed contagion graph: node $j$ is triggered if and only if

$$\sum_{i < j, \text{ triggered}} M_i \cdot \lambda^{\text{Kyle}} / m_t > \delta_j - \delta_1$$

This is analogous to the Caccioli et al. (2014) fire-sale network model, with positions playing the role of overlapping portfolios and price levels playing the role of asset prices. The cascade chain length F3 is the number of nodes reached in this contagion process, providing a direct operationalisation of the network contagion concept on the price axis.

---

## 5. Validation Protocol (5-Gate)

The cascade probability model must pass all five gates before deployment. Gates G1–G5 map onto the existing agent validation framework in `config/agent.toml`.

### Gate G1: Discriminative Power (Necessary)

**Test:** Compute AUC of $\hat{p}_t$ for predicting $y_t$ on held-out data.

$$\text{AUC} = P(\hat{p}_{t_+} > \hat{p}_{t_-} \mid y_{t_+} = 1, y_{t_-} = 0)$$

**Threshold:** AUC > 0.65.

**Lift test:** Compute $\text{lift}(q) = P(y = 1 \mid \hat{p} > q) / P(y = 1)$ at top decile ($q = 90\%$ quantile of $\hat{p}$). Threshold: lift > 2×.

**Rationale:** The existing H3 validation confirmed a cascade lift > 2× using only `nearest_cluster_distance`. The full model should comfortably exceed this baseline.

### Gate G2: Cost Awareness (Necessary)

Cascade regimes exhibit extreme bid-ask spreads. The net signal value must survive realistic execution costs:

$$\text{Net IC} = \text{IC}_{\text{gross}} - \frac{2 \cdot \bar{s}}{\sigma_r \cdot \sqrt{T}}$$

where $\bar{s}$ is the average spread during cascade events (expected to be 3–10× normal), $\sigma_r$ is return standard deviation over horizon $T$, and the term accounts for round-trip transaction cost as a fraction of signal volatility.

**Threshold:** Net IC > 0.02.

**Implementation:** During cascade events, use conservative spread estimates from the 90th percentile of observed spreads, not the average.

### Gate G3: Temporal Stability (Walk-Forward)

Split the dataset into rolling windows of 30 days each. For each window, train on the previous 90 days and test on the current 30 days. Compute AUC per window.

**Threshold:** Median AUC over windows > 0.60, and fewer than 20% of windows have AUC < 0.55.

**Rationale:** Cascade dynamics change with the leverage regime (e.g., high-leverage periods post-airdrop vs. normal periods). The model must be robust across regimes, not just well-calibrated in-sample.

### Gate G4: Cross-Symbol Validity

Estimate separate models for BTC, ETH, SOL. Compare $\boldsymbol{\beta}$ vectors and AUC.

**Threshold:** All three symbols achieve Gate G1. Pairwise cosine similarity of $\boldsymbol{\beta}$ vectors > 0.6 (consistent factor structure across symbols).

**Rationale:** If the model is data-mining a symbol-specific artefact, the feature weights will diverge across symbols. Consistent $\boldsymbol{\beta}$ provides evidence for a shared mechanism.

### Gate G5: Orthogonality to Existing Signals

Regress the cascade probability $\hat{p}_t$ on the existing MF liquidity signal (`spread_depth_composite`):

$$\hat{p}_t = \gamma_0 + \gamma_1 \cdot \text{spread\_depth}_t + \varepsilon_t$$

**Threshold:** $R^2 < 0.30$ (less than 30% of variance explained by existing signal), confirming that the cascade model adds information beyond the baseline spread+depth strategy.

**Implementation:** Use partial IC (information coefficient conditioned on the existing signal) as the primary diagnostic.

---

## 6. Connection to Existing Features

### 6.1 Subsumed Features

The following existing features are special cases of heatmap quantities and are effectively subsumed (the heatmap computes a richer version):

| Existing Feature | Module | Subsumed By |
|-----------------|--------|-------------|
| `liquidation_risk_above_{1,2,5,10}pct` | `liquidation.rs` | $H(t,k)$ partial sums over fixed windows |
| `liquidation_risk_below_{1,2,5,10}pct` | `liquidation.rs` | $H(t,k)$ partial sums over fixed windows |
| `nearest_cluster_distance` | `liquidation.rs` | F1 at finer resolution (10 bps vs 50 bps) |
| `largest_position_at_risk` | `liquidation.rs` | Replaced by F2 (population-level peak, not single position) |

**Note:** Subsumed does not mean deprecated. The coarser fixed-threshold features remain computationally cheap and useful as standalone inputs to other models. They should be retained in the feature vector.

### 6.2 Complementary Features

These existing features are **not subsumed** — they capture orthogonal dimensions and should be used as inputs to the cascade model:

| Existing Feature | Module | Relationship to Heatmap |
|-----------------|--------|------------------------|
| `illiq_kyle_100` / `illiq_composite` | `illiquidity.rs` | Mediates price impact per unit mass (the $\lambda$ in cascade condition §4.2); enters interaction term $v \cdot \mathcal{I}$ |
| `illiq_roll_100` / `illiq_roll_500` | `illiquidity.rs` | Effective bid-ask spread; bounds the minimum move before cascade begins |
| `liquidation_asymmetry` | `liquidation.rs` | Directional bias; enters interaction $\text{ACP} \cdot \text{sgn}(f)$ — the existing scalar is the sign of ACP |
| `liquidation_intensity` | `liquidation.rs` | Overall leverage stress; conditions model on market-wide risk appetite |
| `herfindahl_index` / `gini_coefficient` | `concentration.rs` | Concentration of **positions** (who holds what); complements concentration of **liquidation mass** (where are the triggers) |
| `whale_retail_ratio` | `concentration.rs` | Whale flow is the primary source of large cluster mass; high ratio preconditions the heatmap for concentrated clusters |
| `concentration_change_1h` | `concentration.rs` | Whale accumulation/distribution changes the heatmap over 1h; provides a temporal lead |
| `ctx_funding_rate` | `context.rs` | Enters interaction $\text{ACP} \cdot \text{sgn}(f)$; funding rate aligns directional pressure with position-side pressure |

### 6.3 Feature Integration Summary

The cascade probability $\hat{p}_t$ consumes:

- 8 spatial features from the heatmap (F1–F8), computed fresh each snapshot
- 3 interaction terms derived from the above
- 3 conditioning features from existing modules: $\mathcal{I}$ (`illiq_composite`), $f$ (`ctx_funding_rate`), and optionally $A_{\text{book}}$ (`raw_bid_depth_10` + price levels)

Total input dimension to the logistic model: 11 (post-interaction) + 3 existing = 14 features.

---

## 7. Implementation Notes

### 7.1 Incremental Heatmap Update

On each snapshot, the heatmap need not be rebuilt from scratch. Since position updates arrive as diffs (position opened/closed/resized), the update rule is:

$$H(t, k) = H(t-1, k) + \sum_{i \in \text{new}} V_i \cdot \mathbf{1}[\ell_i \in \mathcal{P}_k(t)] - \sum_{i \in \text{closed}} V_i \cdot \mathbf{1}[\ell_i \in \mathcal{P}_{k'}(t-1)]$$

However, mid-price drift also shifts $\mathcal{P}_k(t)$ — a 0.1% price move shifts all bins. At 10 bps bin width this means bins shift by approximately 1 bin per 10 bps of price movement. The safe implementation is a full rebuild every snapshot using the current $m_t$ as reference; incremental updates with bin-shift correction are an optimisation for a second pass.

### 7.2 Numerical Stability

- $H(t,k)$ can span 5 orders of magnitude (USD 1K to USD 100M per bin). Use `f64` throughout; never cast to `f32`.
- The entropy F8 requires $\sum H > 0$. Guard with $\epsilon = 1$ USD added to the denominator.
- The logistic model input $\boldsymbol{\beta}^{\top}\mathbf{x}$ should be clipped to $[-20, +20]$ before applying $\sigma(\cdot)$ to avoid underflow/overflow.
- The finite-difference velocity F6 will produce NaN for the first $\tau = 600$ ticks after startup. Initialise to $v = 0$ and exclude from the model until the buffer is warm.

### 7.3 Memory Budget

Storing $\mathbf{H}(t)$ for 4 channels × 200 bins × `f64` (8 bytes) = 6.4 KB per snapshot. Two snapshots needed for Channel 4 (1-min lag) = 12.8 KB minimum. Five-minute history requires 3000 snapshots × 6.4 KB = 19.2 MB — well within budget.

---

## 8. References

1. **Kyle, A.S. (1985)** — Continuous auctions and insider trading. *Econometrica*, 53(6), 1315–1335.  
   *Foundation for the linear price impact model $\Delta p = \lambda \cdot q$ used in §4.1–4.3.*

2. **Amihud, Y. (2002)** — Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.  
   *Defines the $|\text{return}|/\text{volume}$ illiquidity measure implemented in `illiquidity.rs`.*

3. **Cont, R. & Wagalath, L. (2016)** — Fire sales forensics: measuring endogenous risk. *Mathematical Finance*, 26(4), 835–866.  
   *Formalises the fire-sale feedback loop underlying §4.1–4.3; absorption capacity interpretation in §4.3.*

4. **Brunnermeier, M.K. & Pedersen, L.H. (2009)** — Market liquidity and funding liquidity. *Review of Financial Studies*, 22(6), 2201–2238.  
   *Liquidity spiral mechanism: thin markets amplify forced liquidation cascades (§4.3 and F5).*

5. **Caccioli, F., Shrestha, M., Moore, C. & Farmer, J.D. (2014)** — Stability analysis of financial contagion due to overlapping portfolios. *Journal of Banking & Finance*, 46, 233–245.  
   *Network contagion model mapped to the cascade chain formulation in §4.4.*

6. **Hasbrouck, J. (2009)** — Trading costs and returns for US equities: estimating effective costs from daily data. *Journal of Finance*, 64(3), 1445–1477.  
   *Permanent price impact estimator implemented in `illiquidity.rs`; provides complementary impact estimate to Kyle's $\lambda$.*

7. **Roll, R. (1984)** — A simple implicit measure of the effective bid-ask spread in an efficient market. *Journal of Finance*, 39(4), 1127–1139.  
   *Autocovariance-based spread estimator; establishes the floor on price increments before cascade onset.*
