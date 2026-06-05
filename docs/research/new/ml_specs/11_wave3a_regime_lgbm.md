# 11 — Wave 3a: Regime-Conditioned LightGBM (#7)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 7.1 (lines 1069–1117)
Spec: `docs/research/new/ml_algorithms.txt`, Section 7 (lines 2457–2690)

**Status: NOT STARTED.** Gated on Wave 2 decision gate (Case A or B).

---

### Task 1: Implement algorithm class

**Read first:** `scripts/algorithms/mean_reversion_detector.py` (LightGBM pattern), `scripts/algorithms/regime_state_machine.py` (regime labels)

**Create:** `scripts/algorithms/regime_conditioned_lgbm.py` (~200 LOC)

```python
@register
class RegimeConditionedLGBM(MicrostructureAlgorithm):
    bar_level = True
```

1. `__init__`: `model_dir="models/regime_conditioned_lgbm"`, `confidence_threshold=0.60`, `min_samples_regime=500`
2. Load N+1 models (per-regime + global) from model_dir via `model_io.get_latest_model()`
3. `required_columns()`: union of per-regime feature sets + `alg_rsm_regime_last` (from #4) or `gmm_label_last` (fallback)
4. `step()`: read regime label, check confidence, route to `model_{regime}` or `model_global`
5. 4 outputs: `alg_rlgbm_signal`, `alg_rlgbm_predicted_return`, `alg_rlgbm_regime_used`, `alg_rlgbm_regime_confidence`
6. No models loaded: all NaN
7. Missing regime label: use global model, set `regime_used=5`

Per-regime feature subsets (define as class constants):
- Trending (states 1,3): `trend_hurst_300_mean`, `whale_net_flow_4h_sum`, `regime_accumulation_score_mean`, `trend_momentum_300_mean`
- Ranging (states 0,2): `ent_tick_1m_mean`, `vol_returns_5m_last`, `imbalance_qty_l1_mean`, `mf_bb_pctb_5m_last`
- Volatile (states 4,5): `vol_returns_5m_last`, `toxic_vpin_50_mean`, `ent_tick_1m_mean`
- Global: union of all features

---

### Task 2: Implement training script

**Create:** `scripts/train_regime_lgbm.py` (~180 LOC)

1. Load bars, add regime labels (from RSM algorithm or GMM)
2. For each regime R with count >= `min_samples_regime`:
   - Extract bars where regime == R
   - Train LightGBM(`num_leaves=15`, `n_estimators=100`, early_stopping=20) on regime-specific features
   - Save as `model_R.txt`
3. Train global fallback on all data with all features
4. Save via `model_io.save_lightgbm_model()` with metadata per model
5. Walk-forward: 4 folds, embargo=100, per-regime evaluation

CLI: `python scripts/train_regime_lgbm.py --symbol BTC --data-dir data/features`

**Create:** `models/regime_conditioned_lgbm/` directory.

---

### Task 3: Add config and paper trader wiring

**Modify:** `config/algorithms.toml`:
```toml
[regime_conditioned_lgbm]
model_dir = "models/regime_conditioned_lgbm"
confidence_threshold = 0.60
min_samples_regime = 500
```

**Modify:** `paper_trader_generic.py` — ALGO_CONFIG: `primary="alg_rlgbm_signal"`, `polarity="high_long"`, `bar_agg="last"`.
**Modify:** `paper_trader_daily.py` — add to DAILY_ALGOS.

---

### Task 4: Write unit tests

**Create:** `scripts/algorithms/tests/test_regime_lgbm_unit.py` (~120 LOC)

```python
def test_regime_dispatch():
    """Regime=1 uses trending model, regime=4 uses volatile model."""

def test_global_fallback():
    """Low confidence (0.3) routes to global model."""

def test_missing_regime_label():
    """NaN regime label uses global model, regime_used=5."""

def test_no_models_returns_nan():
    """No models loaded: all outputs NaN."""

def test_signal_range():
    """alg_rlgbm_signal in [-1, 1]."""

def test_per_regime_features_differ():
    """Trending and ranging models use different feature subsets."""

def test_rare_regime_uses_global():
    """Regime with <500 training samples uses global model."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_regime_lgbm_unit.py -v`

---

### Task 5: Write integration test

**Create:** `scripts/algorithms/tests/test_regime_lgbm_integration.py` (~40 LOC)

```python
def test_outperforms_global():
    """Per-regime dispatch produces higher correlation with forward returns
    than global-only model. (Soft check: log result, don't hard-fail.)"""
```

---

### Task 6: Create specification document

**Create:** `docs/research/new/ml_specs/SPEC_REGIME_CONDITIONED_LGBM.md` (~100 lines)

Sections: Thesis (within-regime edge > global edge), routing architecture, per-regime feature sets, training protocol (per-regime volume check, fallback merge), parameters, outputs, references (Gu et al. 2020, Nystrup et al. 2017).

---

### Task 7: Update README.md

Add: `| 33 | regime_conditioned_lgbm | Per-regime LightGBM ensemble | Gu, Kelly & Xiu (2020) |`
