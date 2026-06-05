# Specification: Regime-Conditioned LightGBM (#7)

Per-regime LightGBM ensemble that routes predictions through regime-specific models. Each regime group uses a tailored feature subset capturing the dominant dynamics of that regime. Falls back to a global model when regime confidence is low or a per-regime model is unavailable.

---

## Thesis

Within-regime prediction edge exceeds global-model edge. Trending regimes are best predicted by momentum/persistence features; ranging regimes by mean-reversion/imbalance features; volatile regimes by toxicity/entropy features. A single global model averages over these distinct dynamics, diluting signal.

---

## Routing Architecture

```
Input tick -> read alg_rsm_regime_last, alg_rsm_confidence_last
  |
  +-- confidence < threshold? ---------> global model
  +-- regime in {2, 3} (TRENDING)? ----> trending model
  +-- regime in {0, 1} (ACCUM/DIST)? --> ranging model
  +-- regime in {4, 5} (RANGE/NOISE)? -> volatile model
  +-- no per-regime model? ------------> global model (fallback)
```

---

## Per-Regime Feature Sets

| Group | Regime IDs | Features |
|-------|-----------|----------|
| Trending | 2, 3 | trend_hurst_300_mean, whale_net_flow_4h_sum, regime_accumulation_score_mean, trend_momentum_300_mean |
| Ranging | 0, 1 | ent_tick_1m_mean, vol_returns_5m_last, imbalance_qty_l1_mean, mf_bb_pctb_5m_last |
| Volatile | 4, 5 | vol_returns_5m_last, toxic_vpin_50_mean, ent_tick_1m_mean |
| Global | all | Union of all above (8 features) |

---

## Training Protocol

1. Load bars, compute regime labels via RegimeStateMachine
2. For each group with count >= `min_samples_regime`:
   - Extract bars belonging to that regime group
   - Train LightGBM (regression) on group-specific features
   - Walk-forward: 4 folds, embargo=100
   - Save as `model_{group}.txt`
3. Train global fallback on all data with all features
4. If a regime group has too few samples, its bars are covered by the global model

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `model_dir` | `models/regime_conditioned_lgbm` | `[regime_conditioned_lgbm].model_dir` | Model directory |
| `confidence_threshold` | 0.60 | `[regime_conditioned_lgbm].confidence_threshold` | Min RSM confidence for routing |
| `min_samples_regime` | 500 | `[regime_conditioned_lgbm].min_samples_regime` | Min samples per group for training |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_rlgbm_signal` | [-1, 1] | 50 | Directional signal (tanh-scaled) |
| `alg_rlgbm_predicted_return` | (-inf, inf) | 50 | Raw predicted forward return |
| `alg_rlgbm_regime_used` | {0..5} | 0 | Regime model used (5=global) |
| `alg_rlgbm_regime_confidence` | [0, 1] | 0 | RSM confidence for routing decision |

---

## References

- Gu, S., Kelly, B. & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273.
- Nystrup, P., Hansen, B.W. & Madsen, H. (2017). "Dynamic Allocation or Diversification: A Regime-Based Approach to Multiple Assets." *Journal of Portfolio Management*, 44(2), 62-73.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/regime_conditioned_lgbm.py` |
| Training | `scripts/train_regime_lgbm.py` |
| Config | `config/algorithms.toml` -> `[regime_conditioned_lgbm]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` -> `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` -> `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_regime_lgbm_unit.py` |
| Integration tests | `scripts/algorithms/tests/test_regime_lgbm_integration.py` |
