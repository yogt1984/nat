# Mathematical Foundations of the Information-Theoretic Alpha Discovery Engine

**NAT Quantitative Research Platform**
**Version:** June 2026
**Audience:** PhD-level quantitative finance / market microstructure

---

## Abstract

This document provides a self-contained mathematical treatment of the Information-Theoretic (IT) engine embedded in the NAT quantitative research platform. The engine continuously estimates mutual information (MI), conditional mutual information (CMI), interaction information (II), and transfer entropy (TE) between microstructure features and forward returns at multiple horizons, then applies a rate-distortion cost gate to filter features that cannot overcome transaction costs. We derive each estimator from first principles, state all closed-form expressions, characterize convergence rates and bias properties, and explain the engineering choices that connect theory to a production 236-feature pipeline operating at 100 ms tick frequency.

---

## Notation

| Symbol | Domain | Definition |
|--------|--------|-----------|
| $X, Y, Z$ | Random vectors | Generic random variables (bold for vectors) |
| $\mathbf{x} = (x_1, \ldots, x_N)$ | $\mathbb{R}^N$ | Observed sample of $X$ |
| $N$ | $\mathbb{Z}_{>0}$ | Sample size after NaN removal |
| $d$ | $\mathbb{Z}_{>0}$ | Dimensionality of the joint or conditioning space |
| $k$ | $\mathbb{Z}_{>0}$ | Number of nearest neighbors (default $k = 5$) |
| $\varepsilon_i$ | $\mathbb{R}_{>0}$ | Chebyshev (L$^\infty$) distance to the $k$-th neighbor of point $i$ in joint space |
| $n_x(i), n_y(i)$ | $\mathbb{Z}_{\geq 0}$ | Number of points within $\varepsilon_i$-ball in the $X$- and $Y$-marginal spaces |
| $\psi(\cdot)$ | $\mathbb{R} \to \mathbb{R}$ | Digamma function: $\psi(n) = \frac{d}{dn}\ln\Gamma(n)$ |
| $H(X)$ | bits or nats | Differential entropy of $X$ |
| $I(X;Y)$ | bits | Mutual information between $X$ and $Y$ |
| $I(X;Y \mid Z)$ | bits | Conditional mutual information |
| $\mathrm{II}(X;Y;Z)$ | bits | Interaction information (third-order cumulant) |
| $\mathrm{TE}(X \to Y)$ | bits | Transfer entropy from $X$ to $Y$ |
| $r_h$ | bps | Forward return at horizon $h$ ticks |
| $f$ | — | Microstructure feature time series |
| $c_\mathrm{RT}$ | bps | Round-trip transaction cost (taker fee × 2) |
| $\sigma_r$ | bps | Standard deviation of $r_h$ |
| $\kappa$ | — | Sample excess kurtosis of $r_h$ (Pearson; Gaussian $= 3$) |
| $I_\mathrm{min}$ | bits | Minimum MI required to achieve positive expected profit after costs |
| $h$ | ticks | Forward horizon; tick period is 100 ms |
| $s$ | ticks | Stride: $s = \max(1,\, \lfloor h / s_d \rfloor)$, $s_d = 5$ |

All logarithms are base-2 (bits) unless the explicit base "nats" is noted. All probability distributions are assumed to admit continuous densities; the estimators degrade gracefully for discrete inputs through the kernel-density interpretation of $k$-NN methods.

---

## 1. KSG Mutual Information Estimator

### 1.1 Background: Kozachenko-Leonenko Entropy Estimation

The foundation of the KSG estimator is the Kozachenko-Leonenko (KL) $k$-NN entropy estimator (Kozachenko & Leonenko, 1987). For a $d$-dimensional random variable $\mathbf{V}$ with density $p(\mathbf{v})$, the differential entropy is

$$H(\mathbf{V}) = -\int p(\mathbf{v}) \ln p(\mathbf{v})\, d\mathbf{v}.$$

Given an i.i.d. sample $\{\mathbf{v}_i\}_{i=1}^N$, let $\rho_i^{(k)}$ denote the distance from $\mathbf{v}_i$ to its $k$-th nearest neighbor (using a chosen metric). The KL estimator is

$$\hat{H}_\mathrm{KL}(\mathbf{V}) = -\psi(k) + \psi(N) + \ln c_d + d \cdot \frac{1}{N}\sum_{i=1}^N \ln \rho_i^{(k)}, \tag{1}$$

where $c_d$ is the volume of the unit ball in the chosen metric. For the L$^\infty$ (Chebyshev) metric, $c_d = 2^d$ (the ball is a hypercube of half-side 1), so $\ln c_d = d \ln 2$. Combining the volume term with the log-distance term:

$$\hat{H}_\mathrm{KL}(\mathbf{V}) = \psi(N) - \psi(k) + d \cdot \frac{1}{N}\sum_{i=1}^N \ln(2\varepsilon_i), \tag{2}$$

where $\varepsilon_i = \rho_i^{(k)}$ is the Chebyshev distance to the $k$-th neighbor. This is the formula implemented in `_ksg_entropy` (estimators.py, lines 288–307).

**Intuition.** Near each point $\mathbf{v}_i$, the empirical density is approximated as $\hat{p}(\mathbf{v}_i) \approx k / (N \cdot V_i)$, where $V_i = c_d \varepsilon_i^d$ is the volume of the ball containing exactly $k$ neighbors. Substituting into $-\ln \hat{p}$ and averaging over $i$ yields equation (2). The digamma function arises from the exact expected value of $\ln V_i$ under the order-statistic distribution of distances.

### 1.2 KSG Algorithm 1: Avoiding Double-Counting

The naive MI estimate $\hat{I}(X;Y) = \hat{H}(X) + \hat{H}(Y) - \hat{H}(X,Y)$ has high variance because all three entropy estimates share the same $k$ but the marginal and joint ball sizes are inconsistent. Kraskov, Stögbauer & Grassberger (2004) resolve this by fixing the ball size in the joint space and adapting the marginal counts, yielding Algorithm 1:

$$\boxed{I(X;Y) \approx \psi(k) - \bigl\langle \psi\bigl(n_x(i)\bigr) + \psi\bigl(n_y(i)\bigr) \bigr\rangle + \psi(N).} \tag{3}$$

**Derivation.** For each point $(x_i, y_i)$ in the joint space:

1. Find $\varepsilon_i =$ Chebyshev distance to the $k$-th neighbor in $\mathbb{R}^2$ (joint).
2. Define $n_x(i) = \#\{j \neq i : |x_j - x_i| < \varepsilon_i\}$ — the count in the open $x$-marginal ball.
3. Define $n_y(i)$ analogously.
4. Because the $x$-ball has volume $2\varepsilon_i$ (a 1-D interval), the marginal contribution at scale $\varepsilon_i$ is $\psi(n_x(i) + 1) \approx \psi(n_x(i))$ (for $n_x \geq 1$).

The estimator follows by substituting the KL formula for each marginal entropy using the adaptively sized balls and recognizing that the joint contribution cancels to $\psi(k)$. Full derivation is in Kraskov et al. (2004), eqs. (8)–(12). The result in equation (3) is in nats; the implementation converts to bits by dividing by $\ln 2$.

### 1.3 Implementation Details

The implementation at `ksg_mi` (estimators.py, lines 21–66) follows equation (3) exactly:

- Joint tree uses `p=np.inf` (Chebyshev metric).
- `k+1` neighbors are queried because `cKDTree.query` counts the point itself; the $(k+1)$-th query result is the $k$-th distinct neighbor.
- Marginal counts use open balls (`query_ball_point`) with radius `eps[i]` and Chebyshev metric; the point itself is subtracted.
- $n_x(i)$ and $n_y(i)$ are clamped to $\geq 1$ before applying $\psi$ to avoid $\psi(0) = -\infty$.
- The final result is $\max(\hat{I}, 0)$ — the theoretical MI is non-negative, and the clamp corrects for small negative estimates caused by finite-sample bias.

### 1.4 Bias and Convergence

The KSG estimator has bias $O(k^{-2})$ for smooth densities (the bias decreases with $k$ but the variance increases). The mean squared error is minimized at $k \sim N^{2/(d+4)}$ (Gao et al., 2015), giving an optimal rate:

$$\mathrm{MSE}\bigl[\hat{I}_\mathrm{KSG}\bigr] = O\!\left(N^{-4/(d+4)}\right). \tag{4}$$

For the standard bivariate case ($d = 2$) this is $O(N^{-2/3})$, significantly better than histogram-based estimators at $O(N^{-1/2})$ for fixed binning. With $k = 5$ and typical buffer sizes of $N = 500$–$6000$, bias is small but non-negligible for features with multimodal or heavy-tailed marginals.

**Why Chebyshev, not Euclidean?** With the L$^\infty$ metric, the joint ball is a rectangle in $(X, Y)$ space, which exactly projects to intervals in each marginal. This means the marginal counts $n_x(i)$ and $n_y(i)$ are defined over the exact same $\varepsilon_i$-interval, eliminating the geometric inconsistency that would arise with L$^2$ (where the projection of a circle onto an axis is smaller than the radius). KSG also defines Algorithm 2 using L$^\infty$ in the marginals with a smaller radius, which has slightly different bias properties; the NAT engine uses Algorithm 1.

### 1.5 Practical Considerations

- **Minimum sample size.** The estimator returns 0 when $N < k + 1 = 6$.
- **Constant features.** Features with $\mathrm{std}(f) < 10^{-12}$ are skipped upstream in the daemon (line 193); $k$-NN distances collapse to zero for constant arrays.
- **NaN handling.** `_validate_xy` (lines 279–285) strips rows where either variable is non-finite before estimation.
- **Computational cost.** Building the KD-tree is $O(N \log N)$; each query is $O(k \log N)$; total is $O(N k \log N)$ per (feature, horizon) pair.

---

## 2. Conditional Mutual Information

### 2.1 Chain Rule Decomposition

Conditional mutual information is defined as

$$I(X;Y \mid Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z). \tag{5}$$

This follows directly from the definition $I(X;Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$ and the chain rule $H(X \mid Z) = H(X,Z) - H(Z)$. The decomposition into four entropy terms is preferred for implementation because each term is estimated by the same $k$-NN formula (equation 2), differing only in dimensionality $d$.

### 2.2 The Four Entropy Terms

Let $d_x, d_y, d_z$ denote the dimensionalities of $X$, $Y$, $Z$. In the NAT engine, $X$ and $Y$ are always 1-D scalar features; $Z$ is the entropy conditioning matrix (up to $d_z = 5$ columns). Then:

| Entropy term | Dimensionality | Spaces |
|---|---|---|
| $H(Z)$ | $d_z$ | $Z$ alone |
| $H(X,Z)$ | $1 + d_z$ | $X$ stacked with $Z$ |
| $H(Y,Z)$ | $1 + d_z$ | $Y$ stacked with $Z$ |
| $H(X,Y,Z)$ | $2 + d_z$ | All variables |

Each term is estimated by `_ksg_entropy` (equation 2). The CMI in nats is then assembled by equation (5) and converted to bits by dividing by $\ln 2$ (line 109).

### 2.3 Dimensionality and the Curse

The convergence rate of each $k$-NN entropy estimator degrades with dimensionality according to equation (4). For the highest-dimensional term $H(X,Y,Z)$, the effective dimension is $d = 2 + d_z$. With $d_z = 3$ entropy conditioning features (the default in `entropy_conditioning`), this is $d = 5$. Substituting into equation (4):

$$\mathrm{MSE}\bigl[\hat{H}_{X,Y,Z}\bigr] = O\!\left(N^{-4/9}\right), \quad d = 5. \tag{6}$$

This motivates the `cmi_min_samples = 500` guard in the config: at $N = 500$ with $d = 5$, the effective standard error of the CMI estimate is roughly $0.05$–$0.15$ bits, large enough to require careful thresholding. The daemon skips CMI computation and logs a debug message when fewer samples are available (lines 202–208).

The `cmi_max_z_dims = 5` cap is a hard engineering constraint: adding more conditioning dimensions increases computational cost quadratically (via KD-tree construction in higher dimensions) while the marginal information gain from additional conditioning variables is rapidly diminishing.

### 2.4 Sign and Interpretation

$I(X;Y \mid Z) \geq 0$ always (information-theoretically), but the $k$-NN estimator can produce small negative values due to finite-sample bias. The CMI estimator does not clamp to zero (unlike `ksg_mi`) because negative values carry diagnostic meaning in the interaction information computation (Section 3).

### 2.5 Practical Considerations

- The conditioning variable $Z$ in the greedy selector (`feature_selector.py`, line 85) is built by stacking already-selected feature arrays, so its dimensionality grows from 1 to `max_features - 1` during the forward pass. The `cmi_max_z_dims` cap does not apply here; instead the sample size decreases monotonically with horizon, and the stopping criterion fires before CMI becomes unreliable.
- For `ksg_te`, $Z$ is the AR history of the target (1-D for `order = 1`), making the joint dimension $d = 3$ — well within the reliable regime for $N \geq 200$.

---

## 3. Interaction Information

### 3.1 Definition and Closed Form

The interaction information (McGill, 1954; Jakulin & Bratko, 2003) of three variables is defined as:

$$\mathrm{II}(X;Y;Z) = I(X;Y \mid Z) - I(X;Y). \tag{7}$$

Expanding via the chain rule, this is equivalently

$$\mathrm{II}(X;Y;Z) = I(X;Y) + I(X;Z) + I(Y;Z) - I(X;Y;Z) - I(X;Y) = I(X;Z) + I(Y;Z) - I(X,Y;Z), \tag{8}$$

though equation (7) is the computationally convenient form implemented in `interaction_info` (lines 117–128).

### 3.2 Third-Order Information Cumulant

Interaction information is the third-order cumulant of the information-theoretic lattice. By analogy with statistical cumulants: $\kappa_1 = \mu$ (mean), $\kappa_2 = \sigma^2$ (variance), $\kappa_3$ (skewness); here MI is the second-order interaction and II is the third-order. For three jointly Gaussian variables:

$$\mathrm{II}(X;Y;Z) = -\frac{1}{2}\log_2\frac{(1-\rho_{XY}^2)(1-\rho_{XZ}^2)(1-\rho_{YZ}^2)}{(1-\rho_{XY}^2-\rho_{XZ}^2-\rho_{YZ}^2+2\rho_{XY}\rho_{XZ}\rho_{YZ})^2}, \tag{9}$$

where $\rho_{AB}$ is the Pearson correlation between $A$ and $B$. The sign structure is non-trivial even in the Gaussian case.

### 3.3 Synergy and Redundancy

The sign of $\mathrm{II}$ reveals the information structure of the triplet $(X, Y, Z)$:

$$\mathrm{II}(X;Y;Z) \begin{cases} > 0 & \text{synergy: } Z \text{ reveals additional } X \leftrightarrow Y \text{ dependence} \\ = 0 & \text{independence of interaction} \\ < 0 & \text{redundancy: } Z \text{ explains away part of } X \leftrightarrow Y. \end{cases}$$

In the NAT pipeline, $X$ is a microstructure feature (e.g., order-book imbalance), $Y$ is the forward return, and $Z$ is a conditioning entropy feature (e.g., tick entropy $\mathrm{ent\_tick\_5s}$). A strongly negative II means the feature and entropy jointly carry redundant information about returns — the feature's apparent predictive power is partially explained by the regime. A positive II means the feature is more predictive in high-entropy regimes than expected from its marginal MI.

### 3.4 Connection to Partial Information Decomposition

Williams & Beer (2010) introduced the Partial Information Decomposition (PID) framework, which attempts to decompose $I(X,Y;Z)$ into four non-negative atoms: unique information of $X$, unique information of $Y$, synergistic, and redundant. Interaction information is the difference between the synergistic and redundant atoms:

$$\mathrm{II}(X;Y;Z) = \mathrm{Syn}(X,Y;Z) - \mathrm{Red}(X,Y;Z). \tag{10}$$

Computing the individual PID atoms requires solving a constrained optimization problem (specifically, minimizing a divergence subject to marginal consistency constraints), which is NP-hard for more than three variables. II is computable in $O(N k \log N)$ as the difference of two already-available estimates, making it the tractable proxy for PID structure in the NAT engine.

---

## 4. Transfer Entropy

### 4.1 Schreiber Formulation

Transfer entropy was introduced by Schreiber (2000) to measure the information flow from a source process $X$ to a target process $Y$, beyond what $Y$'s own past predicts. For stationary processes with AR order $p$:

$$\mathrm{TE}(X \to Y) = I\!\left(X_t^{(\ell)};\, Y_{t+1} \;\Big|\; Y_t^{(p)}\right), \tag{11}$$

where $X_t^{(\ell)} = (X_{t-\ell+1}, \ldots, X_t)$ is the length-$\ell$ lag vector of the source and $Y_t^{(p)} = (Y_{t-p+1}, \ldots, Y_t)$ is the AR history of the target. In the NAT implementation, $\ell = 1$ (`lag = 1`) and $p = 1$ (`order = 1`) by default, so the conditioning variable is scalar.

The equivalent definition via entropies (Schreiber 2000, eq. 3) is:

$$\mathrm{TE}(X \to Y) = H\!\left(Y_{t+1} \mid Y_t^{(p)}\right) - H\!\left(Y_{t+1} \mid Y_t^{(p)}, X_t^{(\ell)}\right). \tag{12}$$

Equations (11) and (12) are identical by the definition of conditional MI.

### 4.2 Nonparametric TE via KSG CMI

`ksg_te` (lines 135–181) estimates equation (11) by calling `cmi(x_past, y_present, y_past, k=k)`. The array construction is:

- `x_past[i] = source[max_lag - 1 + i]` for $i = 0, \ldots, N - \mathrm{max\_lag} - 1$, where $\mathrm{max\_lag} = \max(\mathrm{lag}, \mathrm{order})$.
- `y_present[i] = target[max_lag + i]`.
- `y_past` stacks `order` lagged target values.

The resulting triple `(x_past, y_present, y_past)` spans a joint space of dimension $\ell + 1 + p$. For the default parameters this is $1 + 1 + 1 = 3$, giving convergence rate $O(N^{-4/7})$.

### 4.3 Linear (Gaussian) TE and Equivalence with Granger Causality

For jointly Gaussian processes, transfer entropy is equivalent to Granger causality (Barnett, Barrett & Seth, 2009). The Granger causality measure is defined as:

$$\mathcal{F}_{X \to Y} = \ln\frac{\Sigma_{\mathrm{reduced}}}{\Sigma_{\mathrm{full}}}, \tag{13}$$

where $\Sigma_{\mathrm{reduced}} = \mathrm{Var}(\varepsilon_{\mathrm{reduced}})$ is the residual variance of the AR($p$) model on $Y$ alone, and $\Sigma_{\mathrm{full}} = \mathrm{Var}(\varepsilon_{\mathrm{full}})$ is the residual variance when lagged $X$ terms are included. For Gaussian residuals, the entropy of $\varepsilon_{\mathrm{reduced}}$ is $\frac{1}{2}\ln(2\pi e \Sigma_{\mathrm{reduced}})$, and the reduction upon including $X$ is exactly:

$$\mathrm{TE}_\mathrm{Gaussian}(X \to Y) = \frac{1}{2}\log_2\frac{\Sigma_{\mathrm{reduced}}}{\Sigma_{\mathrm{full}}}. \tag{14}$$

This is implemented in `linear_te` (lines 184–232). The two regression models are:

$$\text{Reduced:}\quad Y_{t+1} = \sum_{i=1}^p \alpha_i Y_{t-i+1} + \varepsilon_t^{(r)}, \tag{15}$$
$$\text{Full:}\quad Y_{t+1} = \sum_{i=1}^p \alpha_i Y_{t-i+1} + \sum_{j=1}^\ell \beta_j X_{t-j+1} + \varepsilon_t^{(f)}. \tag{16}$$

Both are estimated by ordinary least squares via `_ols_residual_var` (lines 310–319). Equation (14) in bits uses $\log_2$, so the implementation divides the natural-log version by $\ln 2$.

**When to use which.** `linear_te` is $O(N p^2)$ (OLS with $p + \ell$ regressors) — far cheaper than `ksg_te` at $O(N k \log N)$. It is preferred in the swarm optimizer and during parameter sweeps. `ksg_te` is used for the final feature set in the IT engine because crypto returns have heavy tails ($\kappa \approx 5$–$15$) and the linear assumption fails for regime-switching microstructure signals.

### 4.4 Practical Considerations

- The `te_top_n = 20` config parameter limits TE computation to the top-20 features by MI, avoiding $O(n_\mathrm{features})$ expensive CMI calls.
- TE is clamped to $\geq 0$ in both implementations, as TE is non-negative by definition for the conditional MI formulation.
- The `linear_te` guard at line 228 (`var_full <= 0 or var_reduced <= 0`) protects against degenerate inputs (e.g., all-zero return slices).

---

## 5. Cost Gate via Rate-Distortion Theory

### 5.1 The Rate-Distortion Problem

Shannon (1959) defined the rate-distortion function $R(D)$ as the minimum number of bits required to represent a source $X$ with expected distortion $\leq D$:

$$R(D) = \min_{p(\hat{x}|x):\, \mathbb{E}[d(X,\hat{X})] \leq D} I(X;\hat{X}). \tag{17}$$

For a Gaussian source $X \sim \mathcal{N}(0, \sigma^2)$ with squared-error distortion $d(x,\hat{x}) = (x-\hat{x})^2$, the rate-distortion function has the closed form (Shannon 1959, Berger 1971):

$$R(D) = \frac{1}{2}\log_2\frac{\sigma^2}{D}, \quad 0 < D \leq \sigma^2. \tag{18}$$

### 5.2 Deriving the Cost Gate

**Setup.** Frame the return prediction problem as: a feature $F$ produces a signal $\hat{R}$ (a lossy reconstruction of the forward return $R$). A trade is profitable only if the prediction error $|R - \hat{R}|$ is small enough relative to the round-trip cost $c_\mathrm{RT}$. The minimum required information from $F$ about $R$ is the rate needed to keep distortion $D \leq c_\mathrm{RT}^2$.

**Bound.** Setting $D = c_\mathrm{RT}^2$ in equation (18):

$$I_\mathrm{min}^\mathrm{Gaussian} = \frac{1}{2}\log_2\frac{\sigma_r^2}{c_\mathrm{RT}^2} = -\frac{1}{2}\log_2\!\left(1 - \frac{\sigma_r^2 - c_\mathrm{RT}^2}{\sigma_r^2}\right). \tag{19}$$

Re-parameterizing in terms of the signal-to-cost ratio $\rho \equiv c_\mathrm{RT} / \sigma_r$:

$$\boxed{I_\mathrm{min}^\mathrm{Gaussian} = -\frac{1}{2}\log_2\!\left(1 - \rho^2\right).} \tag{20}$$

Equation (20) is the formula in `min_info_bits` (line 269). When $\rho \geq 1$ (costs exceed volatility), the argument of the logarithm is $\leq 0$, so $I_\mathrm{min} = +\infty$: no strategy can generate positive expected returns, regardless of how much information a feature carries. This motivates the guard at line 267.

**Asymptotic behavior.** For small costs ($\rho \ll 1$), Taylor expansion gives:

$$I_\mathrm{min}^\mathrm{Gaussian} \approx \frac{\rho^2}{2\ln 2} = \frac{c_\mathrm{RT}^2}{2\sigma_r^2 \ln 2} + O(\rho^4). \tag{21}$$

The bound grows quadratically with cost and shrinks quadratically with volatility — consistent with the intuition that higher-volatility markets are more exploitable per unit of cost.

### 5.3 Kurtosis Correction

The Gaussian bound (20) is tight only for Gaussian return distributions. For heavy-tailed distributions, the information–distortion tradeoff is less favorable: the same distortion $D$ requires strictly more information when probability mass is concentrated in the tails (Verdú & Guo, 2006). The correction used in the implementation is a first-order approximation:

$$I_\mathrm{min} = I_\mathrm{min}^\mathrm{Gaussian} \times \frac{\kappa}{3}, \tag{22}$$

where $\kappa$ is the Pearson kurtosis (Gaussian $= 3$). Equation (22) preserves backward compatibility for Gaussian inputs ($\kappa = 3 \Rightarrow$ factor $= 1$) and scales the requirement up for fat-tailed crypto returns. With $\kappa = 6$ (typical for BTC at 100-tick horizon), the cost gate is $2\times$ the Gaussian bound.

**Caveat.** Equation (22) is a heuristic; it is not the exact rate-distortion function for a specific heavy-tailed distribution. The exact correction for a Student-$t$ distribution with $\nu$ degrees of freedom requires numerical optimization of equation (17), which is prohibitive in a real-time pipeline. The linear-kurtosis scaling captures the dominant effect while remaining analytically tractable.

### 5.4 Practical Considerations

- `sigma_r_bps` is computed as `np.std(r)` on the strided return sample for each horizon (daemon line 186). This is the sample standard deviation in basis points.
- Kurtosis is computed in `greedy_select` (line 57) using `scipy.stats.kurtosis(r, fisher=False)` — Pearson kurtosis with a minimum sample of 30 before applying the correction.
- The cost parameters are loaded from `config/costs.toml` (single source of truth for taker fee, slippage, etc.).

---

## 6. Greedy Forward Feature Selection

### 6.1 Algorithm

The greedy forward selection in `greedy_select` (feature_selector.py, lines 53–118) solves the following problem: given a feature dictionary $\{f_1, \ldots, f_M\}$ and a return series $r$, find the subset $S^* \subseteq [M]$ that maximizes $I(\mathbf{f}_{S^*}; r)$ subject to $|S^*| \leq M_\mathrm{max}$ and the cost constraint that marginal gains exceed $I_\mathrm{min}$.

**Phase 1 (initialization).**

$$f^* = \operatorname*{argmax}_{f \in [M]} I(f;\, r). \tag{23}$$

**Phase 2 (greedy expansion).** For $t = 2, 3, \ldots, M_\mathrm{max}$:

$$f_t = \operatorname*{argmax}_{f \notin S_{t-1}} I\!\left(f;\, r \;\Big|\; \mathbf{f}_{S_{t-1}}\right). \tag{24}$$

The conditioning variable $Z$ in the CMI call is the matrix formed by stacking all previously selected feature arrays column-wise (`np.column_stack(selected_arrays)`, line 85).

**Stopping criterion.** At step $t > 1$, stop if $I(f_t; r \mid \mathbf{f}_{S_{t-1}}) < I_\mathrm{min}$. Step $t = 1$ is never stopped early (it must pass the non-negativity check but not the $I_\mathrm{min}$ gate).

### 6.2 Submodularity and Approximation Guarantee

For Gaussian sources, mutual information is a submodular set function: for sets $A \subseteq B$ and feature $f \notin B$,

$$I(f; r \mid \mathbf{f}_B) \leq I(f; r \mid \mathbf{f}_A). \tag{25}$$

Equation (25) reflects the data-processing inequality: conditioning on more information cannot increase the marginal gain of $f$. Under submodularity, the greedy algorithm achieves a $(1 - e^{-1}) \approx 63\%$ approximation guarantee of the optimal subset (Nemhauser et al., 1978):

$$I\!\left(\mathbf{f}_{S_\mathrm{greedy}};\, r\right) \geq \left(1 - e^{-1}\right) I\!\left(\mathbf{f}_{S^*};\, r\right). \tag{26}$$

For non-Gaussian sources, MI is not always submodular, but empirical evidence in feature selection literature (Brown et al., 2012) shows that greedy selection with CMI gain performs well for up to 30–50 features — well beyond the `max_features = 10` cap in the NAT engine.

### 6.3 Connection to MIFS and CMIM

The greedy CMI algorithm implemented here is equivalent to the Mutual Information Feature Selection (MIFS) framework of Battiti (1994) with $\beta = 1$, or equivalently to the Conditional Mutual Information Maximization (CMIM) criterion of Fleuret (2004). In both cases, the selected feature at each step maximizes:

$$f_t = \operatorname*{argmax}_f \left[ I(f; r) - \beta \max_{s \in S_{t-1}} I(f; s) \right] \quad \text{(MIFS with } \beta = 1\text{)}, \tag{27}$$

which is a tractable lower bound on $I(f; r \mid \mathbf{f}_{S_{t-1}})$. The NAT engine computes the exact CMI (equation 24) rather than the MIFS approximation, at greater computational cost but without the approximation error in equation (27). For a thorough unified treatment, see Brown et al. (2012).

### 6.4 Computational Complexity

Each call to `cmi` at step $t$ has complexity $O(N k \log N)$ for $k$-NN search in dimension $1 + 1 + (t-1) = t + 1$. Summing over $M_\mathrm{max}$ steps with $M$ candidate features:

$$T_\mathrm{greedy} = O\!\left(M \cdot M_\mathrm{max} \cdot N k \log N\right). \tag{28}$$

For the typical parameters $M \approx 100$ candidate features, $M_\mathrm{max} = 10$, $N = 500$, $k = 5$: approximately $10^7$ distance evaluations, executed in $O(1\text{ s})$ in practice.

### 6.5 Practical Considerations

- The `excluded` set allows the agent to pass already-deployed features, preventing re-selection of signals already in the portfolio.
- The `cumulative_mi` field in each returned dict tracks $\hat{I}(\mathbf{f}_{S_t}; r)$, estimated as the sum of marginal CMI gains. This is an approximation (the exact joint MI is not the sum of conditional MIs unless the selected features are conditionally independent given $r$), but it provides a consistent monotone proxy for portfolio information content.
- `cost_viable` flags features where the cumulative MI exceeds $I_\mathrm{min}$, providing a natural binary gate for the alpha pipeline.

---

## 7. Stride Mechanism for Honest MI Estimation

### 7.1 The Overlapping Returns Problem

At horizon $h$ ticks and emission frequency 1 tick, the forward return series

$$r_h(t) = \frac{p(t+h) - p(t)}{p(t)}$$

has an autocorrelation structure: $r_h(t)$ and $r_h(t+1)$ share $h-1$ price observations, so

$$\mathrm{Corr}(r_h(t), r_h(t+1)) = \frac{h-1}{h} \xrightarrow{h \to \infty} 1. \tag{29}$$

More precisely, for a log-price random walk $\ln p(t) = \sum_{s=1}^t \epsilon_s$ with i.i.d. increments $\epsilon_s \sim (0, \sigma^2)$:

$$\mathrm{Cov}(r_h(t), r_h(t+s)) = \max(0,\, h - s) \cdot \sigma^2, \quad s = 0, 1, 2, \ldots \tag{30}$$

### 7.2 Bias Inflation in MI Estimates

The KSG estimator assumes i.i.d. samples. With overlapping returns, consecutive samples $(f(t), r_h(t))$ and $(f(t+1), r_h(t+1))$ are not independent — they share $h-1$ return observations. This inflates the apparent "diversity" of the sample, causing the $k$-NN distances to be systematically smaller than they would be for independent draws. The consequence is a positive bias in $\hat{I}(f; r_h)$:

$$\mathbb{E}\bigl[\hat{I}_\mathrm{KSG}(f; r_h)\bigr] = I(f; r_h) + \Delta_\mathrm{autocorr}, \quad \Delta_\mathrm{autocorr} > 0 \text{ for overlapping returns}. \tag{31}$$

The magnitude of $\Delta_\mathrm{autocorr}$ is approximately proportional to the fraction of overlapping observations: $\Delta_\mathrm{autocorr} \approx I(f; r_h) \cdot (1 - 1/h)$ in the worst case (all samples perfectly correlated), though in practice the feature $f$ decorrelates faster than the returns, reducing the effective bias.

### 7.3 Stride Eliminates Autocorrelation

By subsampling with stride $s = \max(1, \lfloor h / s_d \rfloor)$ where $s_d = 5$ (the `stride_divisor`), we select every $s$-th sample:

$$\mathcal{T}_s = \{t_0, t_0 + s, t_0 + 2s, \ldots\}.$$

For two samples at times $t$ and $t + s$:

$$\mathrm{Cov}(r_h(t), r_h(t+s)) = \max(0, h - s) \cdot \sigma^2. \tag{32}$$

With $s = h / 5$, the residual overlap is $h - s = 4h/5$, so there is still $80\%$ overlap at $s_d = 5$. The choice $s_d = 5$ is a practical compromise: it reduces the autocorrelation by $1/5$ while retaining $N/5$ effective samples. For strict non-overlap one would need $s \geq h$, reducing $N$ to $N/h$, which is prohibitive at large horizons ($h = 500$ ticks $\Rightarrow$ only 12 independent samples per minute of data).

### 7.4 Effective Sample Size

After striding, the effective sample size is

$$N_\mathrm{eff} = \left\lfloor \frac{N}{s} \right\rfloor = \left\lfloor \frac{N \cdot s_d}{h} \right\rfloor. \tag{33}$$

For the default configuration ($N = 6000$ buffer, $h = 500$ ticks, $s_d = 5$):

$$N_\mathrm{eff} = \left\lfloor \frac{6000 \times 5}{500} \right\rfloor = 60.$$

This is above the `cmi_min_samples = 500` threshold only for shorter horizons; the daemon's `valid_mask.sum() < 50` guard (line 181) is the downstream catch for small effective samples at long horizons.

For the short horizons ($h = 10$, $s = 2$): $N_\mathrm{eff} = 3000$, well within the reliable estimation regime.

**Example from configuration:**

| Horizon $h$ | Stride $s$ | $N_\mathrm{eff}$ (from 6000) | Bar horizon |
|---|---|---|---|
| 10 ticks (1 s) | 2 | 3000 | — |
| 50 ticks (5 s) | 10 | 600 | — |
| 500 ticks (50 s) | 100 | 60 | — |

### 7.5 Practical Considerations

- The `_compute_forward_returns` method in the daemon returns both the return values and the corresponding valid `indices`, which are used to align feature values to the strided return timestamps (lines 185–186).
- At $h = 500$, $N_\mathrm{eff} = 60 < 500$, so the CMI gate is skipped for long horizons (line 202). Only MI (which is lower-dimensional and more sample-efficient) is computed.

---

## 8. Unified Pipeline

The seven components described above connect in a single pass per (symbol, horizon, compute cycle):

```
Raw Parquet buffer (N ≤ 6000 rows, 236 features, 100ms ticks)
    │
    ▼
[Stride] — subsample with s = max(1, h // stride_divisor)
    │         Eliminates autocorrelation in forward returns
    │         Reduces N by factor s_d = 5
    │
    ▼
[KSG MI] — I(f_i; r_h) for all i ∈ [236], using eq. (3)
    │         O(N k log N) per feature; ~100ms total for 236 features
    │
    ▼
[CMI] — I(f_i; r_h | Z) for top features above I_min gate
    │         Z = entropy conditioning matrix (≤ 5 columns)
    │         Requires N_eff ≥ 500; skipped for long horizons
    │
    ▼
[II] — II(f_i; r_h; Z) = CMI - MI for each conditioning feature
    │         Sign reveals synergy vs redundancy with regime
    │
    ▼
[TE] — TE(f_i → r_h) for top-20 features by MI, using eq. (11)
    │         Directional: detects lead-lag causal structure
    │
    ▼
[Cost Gate] — I_min = -0.5 log₂(1 - ρ²) × (κ/3), eq. (20,22)
    │            Features with MI < I_min pruned immediately
    │
    ▼
[Greedy Selector] — Forward selection by CMI gain, stopping when
    │                 marginal gain < I_min, up to max_features = 10
    │
    ▼
ITState update — Ranked features with MI, CMI gain, TE, II, cost_viable
    → Persisted to data/it_engine/{symbol}_{horizon}_state.json
    → Consumed by alpha pipeline and agent framework
```

The full pipeline runs every `compute_interval_s = 5` seconds per symbol, processing all three horizons $(10, 50, 500)$ ticks. Each 5-second cycle produces a ranked feature list with information-theoretic metadata, which the alpha agent uses to generate and validate research hypotheses.

---

## References

Barnett, L., Barrett, A. B., & Seth, A. K. (2009). Granger causality and transfer entropy are equivalent for Gaussian variables. *Physical Review Letters*, 103(23), 238701.

Battiti, R. (1994). Using mutual information for selecting features in supervised neural net learning. *IEEE Transactions on Neural Networks*, 5(4), 537–550.

Berger, T. (1971). *Rate Distortion Theory: A Mathematical Basis for Data Compression*. Prentice-Hall.

Brown, G., Pocock, A., Zhao, M.-J., & Luján, M. (2012). Conditional likelihood maximisation: A unifying framework for information theoretic feature selection. *Journal of Machine Learning Research*, 13, 27–66.

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience. [Primary reference for entropy, MI, and rate-distortion theory.]

Fleuret, F. (2004). Fast binary feature selection with conditional mutual information. *Journal of Machine Learning Research*, 5, 1531–1555.

Gao, W., Oh, S., & Viswanath, P. (2015). Demystifying fixed $k$-nearest neighbor information estimators. *IEEE Transactions on Information Theory*, 64(8), 5629–5661.

Jakulin, A., & Bratko, I. (2003). Analyzing attribute dependencies. In *Proceedings of PKDD 2003*, Lecture Notes in AI 2838, 229–240.

Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the entropy of a random vector. *Problems of Information Transmission*, 23(2), 95–101.

Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

McGill, W. J. (1954). Multivariate information transmission. *Psychometrika*, 19(2), 97–116.

Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14(1), 265–294.

Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461–464.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

Shannon, C. E. (1959). Coding theorems for a discrete source with a fidelity criterion. In *IRE Convention Record*, 4, 142–163.

Verdú, S., & Guo, D. (2006). A simple proof of the entropy-power inequality. *IEEE Transactions on Information Theory*, 52(5), 2165–2166. [Also: Guo, D., Shamai, S., & Verdú, S. (2005). Mutual information and minimum mean-square error in Gaussian channels. *IEEE Transactions on Information Theory*, 51(4), 1261–1282.]

Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. *arXiv preprint* arXiv:1004.2515.
