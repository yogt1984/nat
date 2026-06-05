# ML Algorithm Verification Matrix

Maps testable properties to algorithms. An `x` means the property is tested; `-` means not applicable; `inv` means inverted logic.

## Property Matrix

| Property | CPD | MC | RSM | MR | Meta | RLGBM | KNN |
|----------|:---:|:--:|:---:|:--:|:----:|:-----:|:---:|
| ABC contract (step/reset/features) | x | x | x | x | x | x | x |
| bar_level = True | x | x | x | x | x | x | x |
| Bar suffix check | x | x | x | x | x | x | x |
| Warmup < 1000 bars | x | x | x | x | x | x | x |
| No-model graceful | x | x | x | x | x | x | x |
| run_batch shape | x | x | x | x | x | x | x |
| Model roundtrip | - | x | - | x | x | x | - |
| Entropy gating | - | x | - | inv | - | - | - |
| Regime dispatch | - | - | x | - | - | x | - |
| CUSUM correctness | x | - | - | - | - | - | - |
| Bayesian run-length | x | - | - | - | - | - | - |
| Buffer management | - | - | - | - | - | - | x |
| KD-tree rebuild | - | - | - | - | - | - | x |
| Mahalanobis distance | - | - | - | - | - | - | x |
| Time-decay weighting | - | - | - | - | - | - | x |
| Signal range [-1,1] | x | x | x | x | - | x | x |
| Win rate range [0,1] | - | - | - | - | - | - | x |
| Cost threshold gate | - | - | - | - | - | - | x |
| Triple-barrier labels | - | - | - | - | x | - | - |
| Purged K-fold no-leak | - | - | - | - | x | - | - |
| Complementarity | - | - | - | x | - | x | x |

**Legend:**
- CPD = change_point_detector
- MC = momentum_continuation
- RSM = regime_state_machine
- MR = mean_reversion_detector
- Meta = meta_labeling
- RLGBM = regime_conditioned_lgbm
- KNN = knn_retrieval

---

## Per-Phase Verification Commands

All commands run from `scripts/` directory.

### Phase 1 — Smoke Tests
```bash
python -m pytest algorithms/tests/test_algorithms.py -v
```
Tests ABC contract for all 27+ registered algorithms (ML and non-ML).

### Phase 2 — ML-Specific Parametrized Tests
```bash
python -m pytest algorithms/tests/test_ml_algorithms.py -v
```
Auto-discovers `bar_level=True` algorithms. Tests: bar_level flag, warmup bounds, column suffix naming, no-model graceful, run_batch output shape.

### Phase 3 — Constraint Validation
```bash
python validate_all_algorithms.py
```
Checks registry consistency, feature name prefixes, required column availability.

### Phase 4 — Model Persistence
```bash
python -m pytest tests/test_model_persistence.py -v
```
Roundtrip save/load for sklearn and LightGBM. Metadata completeness. Latest model selection.

### Phase 5 — Algorithm Unit Tests
```bash
python -m pytest algorithms/tests/test_change_point_unit.py -v
python -m pytest algorithms/tests/test_momentum_unit.py -v
python -m pytest algorithms/tests/test_regime_sm_unit.py -v
python -m pytest algorithms/tests/test_mean_reversion_unit.py -v
python -m pytest algorithms/tests/test_meta_labeling_unit.py -v
python -m pytest algorithms/tests/test_regime_lgbm_unit.py -v
python -m pytest algorithms/tests/test_knn_unit.py -v
```

### Phase 6 — Integration Tests
```bash
python -m pytest algorithms/tests/test_mean_reversion_integration.py -v
python -m pytest algorithms/tests/test_regime_lgbm_integration.py -v
python -m pytest algorithms/tests/test_knn_integration.py -v
```

### Phase 7 — Training & Infrastructure Tests
```bash
python -m pytest tests/test_train_momentum.py -v
python -m pytest tests/test_train_mean_reversion.py -v
python -m pytest tests/test_deferred_triggers.py -v
python -m pytest tests/test_wave2_gate.py -v
```

### Full Suite
```bash
bash scripts/run_ml_verification.sh
```

---

## CI Integration Guidance

| Phase | When to Run | Duration |
|-------|-------------|----------|
| 1 — Smoke | Every commit | ~10s |
| 2 — ML parametrized | Every commit | ~7s |
| 3 — Constraint validation | Every commit | ~2s |
| 4 — Model persistence | Every commit | ~3s |
| 5 — Unit tests | Every commit | ~15s |
| 6 — Integration tests | Nightly / pre-merge | ~30s |
| 7 — Training tests | Nightly / pre-merge | ~20s |

Phases 1-5 are fast enough for every commit. Phases 6-7 can run nightly or as a pre-merge gate.
