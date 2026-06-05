# 09 — Wave 2c: Meta-Labeling System (#3)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 5.3 (lines 912–1013)
Spec: `docs/research/new/ml_algorithms.txt`, Section 3 (lines 1021–1462)

**Status: NOT STARTED.** Gated on Wave 1 (Case A only). Skip if Case B.

---

### Task 1: Implement preprocessing pipeline

**Read first:** `scripts/algorithms/runner.py` (AlgorithmRunner), `scripts/alpha/paper_trader_generic.py` (for how base algo signals are generated)

**Create:** `scripts/build_meta_training_data.py` (~150 LOC)

Reusable function `build_meta_training_data(data_dir, symbols, base_algos)`:

1. For each date/symbol parquet file, load raw ticks
2. Run `AlgorithmRunner` for each of the 5 winners: `jump_detector`, `3f_liquidity`, `optimal_entry`, `funding_reversion`, `surprise_signal`
3. Merge tick features + algorithm outputs into one DataFrame
4. Aggregate ALL columns to 5-min bars via `aggregate_bars()`
5. Filter bars where at least one base signal magnitude > 0.01
6. Compute triple-barrier labels (De Prado method):
   - Upper barrier: entry_price * (1 + profit_target_bps/10000)
   - Lower barrier: entry_price * (1 - stop_loss_bps/10000)
   - Time barrier: max_holding_bars
   - Label: 1 if upper hit first, 0 if lower hit first
7. Return `(bar_df, labels, meta_feature_matrix)`

CLI: `python scripts/build_meta_training_data.py --data-dir data/features --symbol BTC --output data/meta_training/`

---

### Task 2: Implement algorithm class

**Create:** `scripts/algorithms/meta_labeling.py` (~150 LOC)

```python
@register
class MetaLabeling(MicrostructureAlgorithm):
    bar_level = True
```

1. `__init__`: `model_path="models/meta_labeling"`, `meta_threshold=0.55`
2. `required_columns()`: 10 market state features (non-directional): `ent_tick_1m_mean`, `ent_rate_of_change_5s_mean`, `toxic_vpin_10_mean`, `toxic_index_mean`, `conc_hhi_last`, `whale_directional_agreement_last`, `vol_returns_5m_mean`, `vol_ratio_short_long_last`, `regime_clarity_last`, `raw_spread_bps_mean`
3. `step()`: load state features, predict P(success), output meta_probability, meta_side (sign of highest base signal), meta_size (probability * scaling)
4. 3 outputs: `alg_meta_probability`, `alg_meta_side`, `alg_meta_size`
5. No-model fallback: probability=0.5, side=0, size=0

---

### Task 3: Implement training script

**Create:** `scripts/train_meta_labeling.py` (~150 LOC)

1. Call `build_meta_training_data()` to get features + triple-barrier labels
2. 10 market state features (from required_columns)
3. StandardScaler + LogisticRegression(C=1.0)
4. Purged K-fold (K=5): out-of-fold predictions only, embargo=100
5. Save via `model_io.save_sklearn_model()`

CLI: `python scripts/train_meta_labeling.py --symbol BTC --data-dir data/features`

**Create:** `models/meta_labeling/` directory.

---

### Task 4: Add config and paper trader wiring

**Modify:** `config/algorithms.toml`:
```toml
[meta_labeling]
model_path = "models/meta_labeling"
meta_threshold = 0.55
profit_target_bps = 5.0
stop_loss_bps = 10.0
max_holding_bars = 100
```

**Modify:** `paper_trader_generic.py` — ALGO_CONFIG: `primary="alg_meta_probability"`, `polarity="high_long"`, `bar_agg="last"`.

**Modify:** `paper_trader_daily.py` — add to DAILY_ALGOS.

---

### Task 5: Write unit tests

**Create:** `scripts/algorithms/tests/test_meta_labeling_unit.py` (~100 LOC)

```python
def test_probability_range():
    """alg_meta_probability always in [0, 1]."""

def test_no_model_returns_neutral():
    """No model: probability=0.5, side=0, size=0."""

def test_triple_barrier_upper_hit():
    """Price path hitting upper barrier first -> label=1."""

def test_triple_barrier_lower_hit():
    """Price path hitting lower barrier first -> label=0."""

def test_triple_barrier_time_expiry():
    """Neither barrier hit -> label = sign(exit return)."""

def test_purged_no_leakage():
    """For K=5 fold split with embargo=10, no test index within 10 of any train index."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_meta_labeling_unit.py -v`

---

### Task 6: Create specification document

**Create:** `docs/research/new/ml_specs/SPEC_META_LABELING.md` (~100 lines)

Sections: Thesis (precision filter, not direction), triple-barrier method, 10 state features, anti-leakage protocol, parameters, outputs, references (De Prado 2018 Ch.3).

---

### Task 7: Update README.md

Add to Algorithm Catalog: `| 32 | meta_labeling | De Prado meta-label precision filter | De Prado (2018) |`
