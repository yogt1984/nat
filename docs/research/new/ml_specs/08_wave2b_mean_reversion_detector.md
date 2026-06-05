# 08 — Wave 2b: Mean-Reversion / False-Breakout Detector (#2)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 5.2 (lines 868–910)
Spec: `docs/research/new/ml_algorithms.txt`, Section 2 (lines 610–1015)

**Status: NOT STARTED.** Gated on Wave 1 decision gate (Case A or B).

---

### Task 1: Implement algorithm class

**Read first:** `scripts/algorithms/momentum_continuation.py` (as pattern), `docs/research/new/ml_algorithms.txt` Section 2.2

**Create:** `scripts/algorithms/mean_reversion_detector.py` (~200 LOC)

```python
@register
class MeanReversionDetector(MicrostructureAlgorithm):
    bar_level = True
```

Implementation requirements:
1. `__init__` with defaults: `model_path="models/mean_reversion_detector"`, `entropy_threshold=0.70`, `zscore_threshold=2.0`, `reversion_prob_thresh=0.65`
2. Load LightGBM model via `model_io.get_latest_model()` + `load_lightgbm_model()` if exists
3. `required_columns()`: `vol_returns_5m_last`, `ent_tick_1m_mean`, `trend_hurst_300_mean`, `imbalance_qty_l1_mean`, `toxic_vpin_50_mean`, `raw_midprice_mean`, `mf_ema_15m_last`
4. `step()`: compute z-score internally `(midprice - ema) / (vol * midprice)`, entropy gate INVERTED (active when ent > threshold), signal = `-sign(zscore) * P(reversion)` when conditions met
5. 4 output features: `alg_mr_signal`, `alg_mr_probability`, `alg_mr_zscore`, `alg_mr_entropy_gate`
6. No-model fallback: all NaN except `alg_mr_zscore` (computed from raw data) and `alg_mr_entropy_gate`
7. `run_batch()`: vectorized path

---

### Task 2: Implement training script

**Create:** `scripts/train_mean_reversion.py` (~180 LOC)

1. Load parquet, aggregate to 5-min bars
2. Compute z-score, forward returns (20 bars), binary reversion label
3. Feature matrix: 6–8 features (z-score, range_pos, whale_divergence, toxicity, etc.)
4. LightGBM with `num_leaves=15`, `n_estimators=100`, early stopping patience=20
5. Walk-forward: 4 folds, embargo=100
6. Post-training: SHAP feature importance. Drop features with importance < 0.02.
7. Save via `model_io.save_lightgbm_model()`
8. Decision gate: OOS AUC < 0.52 = FAIL

CLI: `python scripts/train_mean_reversion.py --symbol BTC --data-dir data/features [--dry-run]`

**Create:** `models/mean_reversion_detector/` directory.

---

### Task 3: Add config and paper trader wiring

**Modify:** `config/algorithms.toml`:
```toml
[mean_reversion_detector]
model_path = "models/mean_reversion_detector"
entropy_threshold = 0.70
zscore_threshold = 2.0
reversion_prob_thresh = 0.65
```

**Modify:** `paper_trader_generic.py` — add ALGO_CONFIG entry (`primary="alg_mr_signal"`, `polarity="high_long"`, `bar_agg="mean"`).

**Modify:** `paper_trader_daily.py` — add to DAILY_ALGOS.

---

### Task 4: Write unit tests

**Create:** `scripts/algorithms/tests/test_mean_reversion_unit.py` (~120 LOC)

```python
def test_entropy_gate_inverted():
    """ent=0.50 (< 0.70) -> gate=0 (blocked). ent=0.80 -> gate=1 (active)."""

def test_zscore_computation():
    """midprice=67500, ema=67400, vol=0.0015 -> zscore ≈ +0.99."""

def test_signal_is_contrarian():
    """Positive zscore + high P(reversion) -> negative signal (short)."""

def test_signal_range():
    """alg_mr_signal always in [-1, 1]."""

def test_no_model_returns_nan_for_signal():
    """No model: signal=NaN, probability=NaN, but zscore and gate computed."""

def test_nan_input_propagation():
    """NaN in required column -> all outputs NaN."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_mean_reversion_unit.py -v`

---

### Task 5: Write training pipeline tests

**Create:** `scripts/tests/test_train_mean_reversion.py` (~80 LOC)

```python
def test_reversion_label_binary():
    """Labels are 0 or 1 only."""

def test_zscore_no_lookahead():
    """z-score at bar t uses only data up to t."""

def test_lgbm_fits_without_error():
    """LightGBM trains on 2000 synthetic bars without warnings."""

def test_shap_feature_drop():
    """Feature with importance < 0.02 is flagged for removal."""
```

**Verification:** `cd scripts && python -m pytest tests/test_train_mean_reversion.py -v`

---

### Task 6: Write integration test

**Create:** `scripts/algorithms/tests/test_mean_reversion_integration.py` (~40 LOC)

```python
def test_complementarity_with_momentum():
    """Run both MR and MC on same data. Spearman of primary signals < 0.3."""
```

---

### Task 7: Create specification document

**Create:** `docs/research/new/ml_specs/SPEC_MEAN_REVERSION_DETECTOR.md` (~100 lines)

Sections: Thesis (high-entropy reversion), z-score formula, feature set, LightGBM config, SHAP protocol, parameters, output features, references (Avellaneda & Lee 2010, Cont et al. 2010).

---

### Task 8: Update README.md

Add to Algorithm Catalog: `| 31 | mean_reversion_detector | LightGBM false-breakout detector | Avellaneda & Lee (2010) |`
