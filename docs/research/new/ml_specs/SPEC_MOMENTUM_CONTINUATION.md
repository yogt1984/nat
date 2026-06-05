# Specification: Momentum Continuation Classifier (#1)

Predicts whether current price momentum will continue over the next 20 bars (100 min) using logistic regression on 7 microstructure features. Active only in low-entropy regimes where trend persistence is highest.

---

## Thesis

In low-entropy regimes, informed flow creates persistent momentum that is predictable via microstructure features. The entropy gate suppresses signals during high-randomness regimes where momentum is noise. Whale flow confirmation and volatility context improve directional accuracy beyond simple trend-following.

---

## Feature Set

| # | Column Name | Description |
|---|-------------|-------------|
| 1 | `ent_tick_1m_mean` | 1-minute tick entropy (also used for gate) |
| 2 | `ent_permutation_returns_16_mean` | Permutation entropy of 16-tick return windows |
| 3 | `trend_hurst_300_mean` | Hurst exponent over 300-tick window |
| 4 | `toxic_vpin_50_mean` | Volume-synchronized PIN (50-bucket) |
| 5 | `whale_net_flow_4h_sum` | Net large-order flow over 4 hours |
| 6 | `regime_accumulation_score_mean` | Accumulation/distribution regime score |
| 7 | `vol_returns_5m_last` | 5-minute return volatility |

---

## Model

- **Type:** `StandardScaler` + `LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')`
- **Output:** P(continuation) in [0, 1]
- **Signal mapping:**
  - P > p_long (0.6): `signal = (P - 0.5) * 2 * gate`
  - P < p_short (0.4): `signal = -(0.5 - P) * 2 * gate`
  - Otherwise (dead zone): `signal = 0`
- **Entropy gate:** If `ent_tick_1m_mean >= entropy_ceiling` (0.85), `gate = 0` and signal forced to 0
- **No model loaded:** signal=0.0, confidence=0.5 (neutral, not NaN)

---

## Training

```bash
python scripts/train_momentum.py --symbol BTC --data-dir data/features
python scripts/train_momentum.py --symbol BTC --data-dir data/features --dry-run
```

Protocol:
1. Load parquet, aggregate to 5-min bars
2. Forward returns: `fwd_ret[t] = midprice_mean[t+20] / midprice_mean[t] - 1`
3. Binary labels: 1 if fwd_ret > 0, else 0
4. Expanding-window walk-forward validation (4 folds, 100-bar embargo)
5. Decision gate: OOS AUC < 0.52 = FAIL
6. Final model trained on all data, saved via `model_io.save_sklearn_model()`

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `model_path` | `models/momentum_continuation` | — | Directory for saved models |
| `entropy_ceiling` | 0.85 | `[momentum_continuation].entropy_ceiling` | Gate threshold |
| `p_long` | 0.6 | `[momentum_continuation].p_long` | Long signal threshold |
| `p_short` | 0.4 | `[momentum_continuation].p_short` | Short signal threshold |
| `C` | 1.0 | Training CLI `--C` | Regularization strength |
| `embargo` | 100 | Training CLI `--embargo` | Bars between train/test folds |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_mc_signal` | [-1, 1] | 50 | Signed continuation signal (0 if gated) |
| `alg_mc_confidence` | [0, 1] | 50 | Raw P(continuation) from LogReg |
| `alg_mc_entropy_gate` | {0, 1} | 0 | 1 if entropy below ceiling |

---

## Differences from Original Spec

| Aspect | Spec (ml_algorithms.txt) | Implementation |
|--------|--------------------------|----------------|
| Features | 9 features | 7 features (dropped 2 redundant) |
| C | 0.1 | 1.0 (less regularization) |
| entropy_ceiling | 0.30 | 0.85 (more permissive gate) |

---

## References

- Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). "Time Series Momentum." *Journal of Financial Economics*, 104(2), 228-250.
- Bouchaud, J.P. et al. (2004). "Fluctuations and Response in Financial Markets." *Quantitative Finance*, 4(2), 176-190.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/momentum_continuation.py` |
| Training | `scripts/train_momentum.py` |
| Models | `models/momentum_continuation/` |
| Config | `config/algorithms.toml` → `[momentum_continuation]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` → `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` → `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_momentum_unit.py` |
| Training tests | `scripts/tests/test_train_momentum.py` |
| Integration tests | `scripts/algorithms/tests/test_momentum_integration.py` |
