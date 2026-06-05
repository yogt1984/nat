# Specification: Mean-Reversion / False-Breakout Detector (#2)

Predicts whether price will revert toward EMA over the next 20 bars (100 min) using LightGBM. Active only in high-entropy (ranging) regimes where mean-reversion dominates. Complementary to MomentumContinuation, which operates in low-entropy (trending) regimes.

---

## Thesis

In high-entropy microstructure states, price displacement from EMA tends to revert rather than persist. This detector measures displacement via z-score and predicts reversion probability using LightGBM trained on structural features. The contrarian signal shorts when price is above EMA and longs when below.

---

## Z-Score Formula

```
z(t) = (midprice(t) - EMA_15m(t)) / (vol_5m(t) * midprice(t))
```

Where:
- `midprice(t)` = bar midprice mean (`raw_midprice_mean`)
- `EMA_15m(t)` = 15-minute exponential moving average (`mf_ema_15m_last`)
- `vol_5m(t)` = 5-minute return volatility (`vol_returns_5m_last`)

The denominator normalizes displacement by both volatility and price level, making z-score comparable across assets and regimes.

---

## Signal Construction

```
signal(t) = -sign(z(t)) * P(reversion | X(t))    if gate(t) = 1 and P > thresh
           = 0                                     otherwise
```

The signal is contrarian: positive z-score (above EMA) produces negative signal (short), and vice versa. Gated by entropy floor and probability threshold.

---

## Entropy Gate (Inverted)

Unlike MomentumContinuation which gates on `ent < ceiling` (active in trending), this detector gates on `ent > floor` (active in ranging):

```
gate(t) = 1{ent_tick_1m(t) > entropy_threshold}
```

Default `entropy_threshold = 0.70`. This ensures the model operates only when the market is in a regime where mean-reversion is the dominant dynamic.

---

## Feature Set

| # | Column | Description |
|---|--------|-------------|
| 1 | `vol_returns_5m_last` | 5-minute return volatility |
| 2 | `ent_tick_1m_mean` | 1-minute tick entropy |
| 3 | `trend_hurst_300_mean` | Hurst exponent (persistence) |
| 4 | `imbalance_qty_l1_mean` | L1 order imbalance |
| 5 | `toxic_vpin_50_mean` | Volume-synchronized PIN |
| 6 | `zscore` | Computed displacement (not from data) |

Plus `raw_midprice_mean` and `mf_ema_15m_last` for z-score computation (not direct model features).

---

## LightGBM Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_leaves` | 15 | Shallow trees prevent overfitting on noisy labels |
| `n_estimators` | 100 | Early stopping trims to best iteration |
| `learning_rate` | 0.05 | Conservative step size |
| `feature_fraction` | 0.8 | Regularization via feature subsampling |
| `bagging_fraction` | 0.8 | Row subsampling per round |
| `early_stopping` | 20 rounds | Stop if validation AUC plateaus |

---

## Training Pipeline

1. Load parquet, aggregate to 5-min bars
2. Compute z-score from raw midprice, EMA, volatility
3. Compute forward returns: `fwd_ret[t] = mid[t+20] / mid[t] - 1`
4. Binary reversion label: `1 if sign(zscore) != sign(fwd_return)` (price moved against displacement)
5. Walk-forward validation: 4 expanding folds, 100-bar embargo
6. SHAP feature importance: flag features with normalized gain < 0.02
7. Save via `model_io.save_lightgbm_model()`

---

## SHAP Protocol

After each walk-forward fold, gain-based feature importance is computed and averaged. Features below 0.02 normalized importance are flagged for removal. This prevents the model from relying on noise features that happen to correlate in-sample.

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `model_path` | `models/mean_reversion_detector` | `[mean_reversion_detector].model_path` | Model directory |
| `entropy_threshold` | 0.70 | `[mean_reversion_detector].entropy_threshold` | Entropy floor for gate |
| `zscore_threshold` | 2.0 | `[mean_reversion_detector].zscore_threshold` | Max abs zscore |
| `reversion_prob_thresh` | 0.65 | `[mean_reversion_detector].reversion_prob_thresh` | Min P(reversion) for signal |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_mr_signal` | [-1, 1] | 50 | Contrarian reversion signal |
| `alg_mr_probability` | [0, 1] | 50 | Model P(reversion) |
| `alg_mr_zscore` | (-inf, inf) | 0 | Price displacement z-score |
| `alg_mr_entropy_gate` | {0, 1} | 0 | 1 if entropy above floor |

---

## Decision Gate

| Metric | FAIL | PASS |
|--------|------|------|
| OOS AUC | < 0.52 | >= 0.52 |

---

## References

- Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the US Equities Market." *Quantitative Finance*, 10(7), 761-782.
- Cont, R., Stoikov, S., & Talreja, R. (2010). "A Stochastic Model for Order Book Dynamics." *Operations Research*, 58(3), 549-563.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/mean_reversion_detector.py` |
| Training | `scripts/train_mean_reversion.py` |
| Config | `config/algorithms.toml` -> `[mean_reversion_detector]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` -> `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` -> `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_mean_reversion_unit.py` |
| Training tests | `scripts/tests/test_train_mean_reversion.py` |
| Integration tests | `scripts/algorithms/tests/test_mean_reversion_integration.py` |
