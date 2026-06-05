# ML Algorithm Developer Guide

Reference for implementing new ML algorithms in the NAT framework.
Derived from `docs/research/new/ml_implementation_plan.txt` Section 0.

---

## 1. Registry Contract

The `@register` decorator calls `cls()` with **no arguments** to extract the algorithm name:

```python
def register(cls):
    instance = cls()  # NO arguments
    _REGISTRY[instance.name()] = cls
    return cls
```

Every `__init__` must have defaults for ALL parameters. If a constructor requires a positional argument, registration silently fails and the algorithm vanishes from `list_algorithms()`.

```python
# Correct
def __init__(self, model_path="models/momentum_continuation", entropy_ceiling=0.85):

# Wrong — crashes at import time
def __init__(self, model_path):
```

---

## 2. Bar-Level Algorithms

ML algorithms operate on 5-min bars, not raw 100ms ticks. Set `bar_level = True` and the runner handles aggregation transparently via `aggregate_bars()` from `scripts/cluster_pipeline/preprocess.py`.

```python
@register
class MyMLAlgorithm(MicrostructureAlgorithm):
    bar_level = True
    bar_timeframe = "5min"
```

`required_columns()` must use bar-aggregated suffixes (`_mean`, `_std`, `_last`, `_sum`, `_slope`), not raw tick names.

```python
# Correct
def required_columns(self):
    return ["ent_tick_1m_mean", "trend_hurst_300_mean", "vol_returns_5m_last"]

# Wrong — these are raw tick column names
def required_columns(self):
    return ["ent_tick_1m", "trend_hurst_300", "vol_returns_5m"]
```

Suffix rules (determined by column prefix in `preprocess.py`):
- Price columns: `_open`, `_high`, `_low`, `_close`, `_mean`
- Entropy columns: `_mean`, `_std`, `_slope`
- Flow columns: `_sum`
- Others: `_mean`, `_std`, `_last`

---

## 3. walk_forward_validation() Takes Polars

`scripts/backtest/walk_forward.py` uses Polars DataFrames, not pandas. Training scripts must convert:

```python
import polars as pl
df_pl = pl.from_pandas(df_pd)
result = walk_forward_validation(df_pl, strategy, cost_model, ...)
```

Returns `WalkForwardResult` with `in_sample_sharpe`, `out_of_sample_sharpe`, `oos_is_ratio`, `is_valid`.

---

## 4. Config Loading

`config/algorithms.toml` is **not** auto-injected into constructors. The algorithm must explicitly load its config or accept kwargs:

```python
import toml
cfg = toml.load("config/algorithms.toml").get("momentum_continuation", {})
self._threshold = cfg.get("entropy_ceiling", 0.85)
```

Convention: constructor defaults are production values. Config overrides are optional.

---

## 5. model_io Conventions

`scripts/utils/model_io.py` provides:

| Function | Purpose |
|----------|---------|
| `save_sklearn_model(model, scaler, metadata, output_dir)` | Save scikit-learn model (joblib, compress=3) |
| `save_lightgbm_model(model, metadata, output_dir)` | Save LightGBM native `.txt` |
| `get_latest_model(model_dir, model_type=None)` | Get most recent model by timestamp |
| `load_sklearn_model(path)` | Returns `(model, scaler, metadata)` |

Load at `__init__` time. If no model exists, `step()` returns all NaN — this is expected for freshly registered algorithms.

```python
path = model_io.get_latest_model("models/momentum_continuation")
if path is not None:
    self._model, self._scaler, self._meta = model_io.load_sklearn_model(path)
else:
    self._model = None  # step() returns NaN
```

Metadata sidecar JSON includes: `model_type`, `model_name`, `feature_names`, `hyperparameters`, `performance_metrics`, `training_date`, `snapshot_name`.

---

## 6. autodiscover.py

Any `.py` file in `scripts/algorithms/` is auto-imported at discovery time. Import failures are **silently swallowed**. Always verify after implementing:

```bash
cd scripts && python -c "
from algorithms.autodiscover import discover_all
discover_all()
from algorithms.registry import list_algorithms
print(list_algorithms())
"
```

If your algorithm isn't in the list, check for import errors by importing the module directly.

---

## 7. AlgorithmFeature

Frozen dataclass with three fields:

```python
@dataclass(frozen=True)
class AlgorithmFeature:
    name: str          # Must start with 'alg_'
    warmup: int = 0    # In bars for bar_level=True, ticks otherwise
    description: str = ""
```

For bar-level algorithms, `warmup` is in number of **bars** (not ticks), because `run_batch()` operates on the bar-level DataFrame after runner aggregation.

---

## 8. Testing

New algorithms are auto-discovered by `test_algorithm_smoke.py` which runs 11 parametrized tests on every registered algorithm. Must pass all:

```bash
# Single algorithm
cd scripts && python -m pytest tests/test_algorithm_smoke.py -k my_algorithm -v

# All algorithms
cd scripts && python -m pytest tests/test_algorithm_smoke.py -v

# Constraint validation
python scripts/validate_all_algorithms.py
```

After smoke tests pass, evaluate on real data:

```bash
nat algorithm evaluate --algorithm my_algorithm --symbol BTC
```

---

## Checklist for New ML Algorithms

1. All `__init__` params have defaults
2. `bar_level = True` set as class attribute
3. `required_columns()` uses `_mean`/`_std`/`_last`/`_sum`/`_slope` suffixes
4. `alg_features()` names start with `alg_`
5. `step()` returns exactly the keys from `alg_features()`
6. No-model case returns NaN (not crash)
7. Config loaded explicitly from `algorithms.toml`
8. Training script converts to Polars before `walk_forward_validation()`
9. `validate_all_algorithms.py` passes
10. `test_algorithm_smoke.py -k <name>` passes all 11 tests
