# Wave 0 Changelog — Infrastructure Prerequisites

Implemented on branch `feat/ml-wave0-wave1`.

---

## What Changed

| File | Change |
|------|--------|
| `scripts/algorithms/base.py` | Added `bar_level: bool = False` and `bar_timeframe: str = "5min"` class attributes to `MicrostructureAlgorithm` |
| `scripts/algorithms/runner.py` | Added bar aggregation dispatch in `run_on_dataframe()`: if `algo.bar_level` is True, calls `aggregate_bars()` before `run_batch()`, forward-fills results to tick-level index |
| `scripts/utils/online.py` | NEW — `WelfordNormalizer` class extracted for reuse by ML algorithms |
| `scripts/algorithms/cascade_probability.py` | Changed to import `WelfordNormalizer` from `utils.online` instead of inline definition |
| `scripts/algorithms/tests/conftest.py` | Added `make_bar_df()`, `make_forward_returns()`, `make_labeled_df()` fixtures |

## Why

All ML algorithms operate on 5-min bars, not raw 100ms ticks. Without centralized bar dispatch:
- Every ML algorithm would copy-paste `aggregate_bars()` calls in `run_batch()`
- Tick-to-bar column renaming would be duplicated ~10 times
- Forward-fill logic for downstream compatibility would be inconsistent

Centralizing in `runner.py` means ML algorithms simply declare `bar_level = True` and receive pre-aggregated bars.

## How to Use

```python
@register
class MyMLAlgorithm(MicrostructureAlgorithm):
    bar_level = True              # runner aggregates ticks to 5-min bars
    bar_timeframe = "5min"        # default, can be overridden

    def required_columns(self):
        return ["ent_tick_1m_mean", "vol_returns_5m_last"]  # bar-suffixed names
```

The runner handles:
1. Detecting `bar_level = True` + `timestamp_ns` in DataFrame
2. Calling `aggregate_bars(df, timeframe=algo.bar_timeframe)`
3. Passing aggregated bars to `run_batch()`
4. Forward-filling results to tick-level index

## Backward Compatibility

All existing tick-level algorithms are unaffected. The default `bar_level = False` preserves the original code path. Verified by `test_bar_level_dispatch.py::test_existing_tick_algos_unaffected`.
