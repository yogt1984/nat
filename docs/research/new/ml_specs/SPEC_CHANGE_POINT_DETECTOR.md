# Specification: Change-Point Detector (#5)

Early-warning system for regime transitions using two complementary statistical methods: CUSUM for fast mean-shift detection and Bayesian Online Change-Point Detection for probabilistic inference over regime duration.

---

## Mathematical Formulation

### CUSUM (Page 1954)

Bilateral cumulative sum on standardized imbalance. Let x_t = (imbalance_t - mu) / sigma where mu, sigma are estimated from a rolling calibration window.

**Update rules:**

```
S+_t = max(0, S+_{t-1} + x_t - delta)     (positive shift detector)
S-_t = max(0, S-_{t-1} - x_t - delta)     (negative shift detector)
```

where delta is the drift parameter (allowance). An alarm fires when S+_t > h or S-_t > h (threshold h). On alarm, the corresponding accumulator resets to 0 and regime_age resets to 0.

**Composite signal:** `cusum_signal = max(S+, |S-|) * sign`

### Bayesian OCD (Adams & MacKay 2007)

Maintains a run-length distribution P(r_t | x_{1:t}) where r_t is the number of bars since the last change-point.

**Growth step:** For each run length r, compute predictive probability under a Student-t distribution derived from the Normal-Inverse-Gamma conjugate prior:

```
pi(r) = P(x_t | r_t = r)  ~  t_{2*alpha_r}(mu_r, beta_r*(kappa_r+1)/(alpha_r*kappa_r))
```

**Joint distribution update:**

```
P(r_t = 0) = sum_r P(r_{t-1} = r) * H * pi(r)        (change-point)
P(r_t = r+1) = P(r_{t-1} = r) * (1-H) * pi(r)        (growth)
```

where H is the constant hazard rate (prior probability of change-point at any bar).

**NIG posterior update** for each surviving run length:

```
kappa_{r+1} = kappa_r + 1
mu_{r+1} = (kappa_r * mu_r + x) / kappa_{r+1}
alpha_{r+1} = alpha_r + 0.5
beta_{r+1} = beta_r + kappa_r * (x - mu_r)^2 / (2 * kappa_{r+1})
```

Run-length distribution is truncated at `max_run_length` and renormalized to prevent unbounded memory growth.

---

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `cusum_threshold` | 5.0 | [2, 15] | CUSUM alarm threshold h |
| `cusum_drift` | 0.05 | [0.01, 0.5] | CUSUM allowance delta |
| `hazard_rate` | 0.005 | [0.001, 0.05] | Bayesian OCD prior P(change) per bar |
| `max_run_length` | 500 | [100, 2000] | Truncation for run-length array |
| `calibration_window` | 200 | [50, 500] | Rolling window for CUSUM normalization |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_cpd_cusum_signal` | (-inf, inf) | 100 | CUSUM composite: max(S+, \|S-\|) * sign |
| `alg_cpd_run_length` | [0, inf) | 100 | Expected bars since last change-point |
| `alg_cpd_change_prob` | [0, 1] | 100 | P(change-point at current bar) |
| `alg_cpd_regime_age` | [0, inf) | 0 | Bars since last CUSUM alarm |

---

## Input Features

| Column | Source |
|--------|--------|
| `imbalance_qty_l1_mean` | L1 order book imbalance (bar mean) |
| `vol_returns_5m_last` | 5-minute return volatility (bar last) |
| `ent_tick_1m_mean` | 1-minute tick entropy (bar mean) |

---

## References

- Page, E.S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100-115.
- Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." *arXiv:0710.3742*.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/change_point_detector.py` |
| Config | `config/algorithms.toml` → `[change_point_detector]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` → `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` → `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_change_point_unit.py` |
| Integration tests | `scripts/algorithms/tests/test_change_point_integration.py` |
