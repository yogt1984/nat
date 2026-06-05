# 12 — Wave 3b: Nearest-Neighbor State Retrieval (#6)

Source: `docs/research/new/ml_implementation_plan.txt`, Section 7.2 (lines 1119–1151)
Spec: `docs/research/new/ml_algorithms.txt`, Section 6 (lines 2184–2451)

**Status: NOT STARTED.** Gated on Wave 2 decision gate (Case A only).

---

### Task 1: Implement algorithm class

**Read first:** `docs/research/new/ml_algorithms.txt` Section 6.2 (distance metric, aggregation)

**Create:** `scripts/algorithms/knn_retrieval.py` (~200 LOC)

```python
@register
class KNNRetrieval(MicrostructureAlgorithm):
    bar_level = True
```

1. `__init__`: `k=20`, `buffer_size=5000`, `time_decay_halflife=500`, `refit_interval=100`, `min_buffer=100`, `cost_threshold_bps=2.0`, `win_rate_threshold=0.60`
2. Internal state: ring buffer of `(feature_vector, forward_return, timestamp)` tuples, Ledoit-Wolf covariance + Cholesky whitening matrix, KD-tree (rebuilt every `refit_interval` bars)
3. `required_columns()`: 7 bar features + `raw_midprice_mean` (for forward return computation)
4. `step()`:
   - If buffer < `min_buffer`: return NaN
   - Whiten query via stored Cholesky
   - Find K nearest in KD-tree (Euclidean on whitened space)
   - Apply time-decay weights: `w_i = exp(-ln(2) * age_i / halflife)`
   - Compute weighted expected_return, win_rate, confidence
   - Generate signal if |expected_return| > cost_threshold AND win_rate > threshold
   - Store current (features, fwd_return when known) into buffer
5. 4 outputs: `alg_knn_signal`, `alg_knn_expected_return`, `alg_knn_win_rate`, `alg_knn_confidence`
6. `reset()`: clear buffer, KD-tree, covariance
7. `run_batch()`: iterate rows (KNN is inherently sequential due to growing buffer)

No training script. No model directory. Buffer-based.

---

### Task 2: Add config and paper trader wiring

**Modify:** `config/algorithms.toml`:
```toml
[knn_retrieval]
k = 20
buffer_size = 5000
time_decay_halflife = 500
refit_interval = 100
cost_threshold_bps = 2.0
win_rate_threshold = 0.60
```

**Modify:** `paper_trader_generic.py` — ALGO_CONFIG: `primary="alg_knn_signal"`, `polarity="high_long"`, `bar_agg="last"`.
**Modify:** `paper_trader_daily.py` — add to DAILY_ALGOS.

---

### Task 3: Write unit tests

**Create:** `scripts/algorithms/tests/test_knn_unit.py` (~130 LOC)

```python
def test_empty_buffer_returns_nan():
    """Before min_buffer observations, all outputs NaN."""

def test_known_neighbor_retrieval():
    """Insert 10 known pairs. Query exact match -> nearest is itself, outcome matches."""

def test_mahalanobis_vs_euclidean():
    """On correlated features (rho=0.8), Mahalanobis neighbors differ from Euclidean
    for at least 30% of K neighbors."""

def test_time_decay_weighting():
    """Recent neighbors (age=10) have higher weight than old (age=1000)."""

def test_buffer_size_cap():
    """After buffer_size+100 insertions, buffer length == buffer_size."""

def test_signal_range():
    """alg_knn_signal in [-1, 1]."""

def test_win_rate_range():
    """alg_knn_win_rate in [0, 1]."""

def test_refit_interval():
    """KD-tree and covariance refit every refit_interval bars."""

def test_cost_threshold_gate():
    """Expected return below cost_threshold -> signal=0."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_knn_unit.py -v`

---

### Task 4: Write integration test

**Create:** `scripts/algorithms/tests/test_knn_integration.py` (~50 LOC)

```python
def test_run_batch_on_bar_df(make_bar_df):
    """run_batch() on 400 bars. After min_buffer, outputs become finite."""

def test_complementarity(make_bar_df):
    """Spearman of knn_signal vs mc_signal and mr_signal both < 0.4."""
```

**Verification:** `cd scripts && python -m pytest algorithms/tests/test_knn_integration.py -v`

---

### Task 5: Create specification document

**Create:** `docs/research/new/ml_specs/SPEC_KNN_RETRIEVAL.md` (~90 lines)

Sections: Thesis (non-parametric, adaptive), Mahalanobis formula, Ledoit-Wolf shrinkage, KD-tree rebuild protocol, time-decay weighting, parameters, outputs, references (Cover & Hart 1967, Mahalanobis 1936, Ledoit & Wolf 2004).

---

### Task 6: Update README.md

Add: `| 34 | knn_retrieval | Mahalanobis nearest-neighbor retrieval | Cover & Hart (1967) |`
