# Step 14: Online Cosine Similarity Scoring

## Context

Production deployment. The online algorithm scores 100ms tick streams against
the discovered kernel library, producing 8 features per tick. This is the
matched filter (Turin 1960): project current price shape onto each kernel.

## Architecture

```
100ms ticks -> aggregate 600 ticks -> 60s micro-candle
    -> decompose into 4 channels
    -> normalize W-candle sliding window by ATR
    -> cosine similarity against each kernel
    -> IC-weighted multi-channel aggregation
    -> 8 output features
```

File: `scripts/algorithms/convolver.py` (306 lines)
Class: `Convolver(MicrostructureAlgorithm)`, registered via `@register`

## Cosine Similarity

```
s = <x, k> / (||x|| * ||k||)
```

Since kernels are unit-norm: s = <x, k> / ||x||. Returns 0 if ||x|| < 1e-12.

File: `scripts/algorithms/convolver_kernels.py:176-192`
Function: `cosine_similarity(x, kernel) -> float`

## Multi-Channel Aggregation

Per event type e, aggregate across channels c and kernels j:

```
S^(e)(t) = sum_c  w_c * sum_j  IC_j * s^(e,c)_j(t)
```

Channel weights w_c default to 0.25 each (equal weighting).
IC weighting gives more influence to higher-IC kernels.

File: `scripts/algorithms/convolver_kernels.py:195-225`
Function: `score_all_kernels(channels, library) -> dict[str, float]`

## Output Features (8)

| Feature | Range | Description |
|---------|-------|-------------|
| alg_conv_breakout_bull | [-1, 1] | Bullish breakout similarity |
| alg_conv_breakout_bear | [-1, 1] | Bearish breakout similarity |
| alg_conv_turtle_bull | [-1, 1] | Bullish turtle soup similarity |
| alg_conv_turtle_bear | [-1, 1] | Bearish turtle soup similarity |
| alg_conv_trap_bull | [-1, 1] | Bull trap similarity |
| alg_conv_trap_bear | [-1, 1] | Bear trap similarity |
| alg_conv_best_score | [0, 1] | max(|score|) across all 6 |
| alg_conv_best_pattern | {0..5} | Index: 0=breakout_bull 1=breakout_bear 2=turtle_bull 3=turtle_bear 4=trap_bull 5=trap_bear |

## Warmup

(W + atr_period) * candle_ticks = (20 + 14) * 600 = 20,400 ticks (~34 minutes).
Output is NaN during warmup.

## Required Columns

`raw_midprice`, `flow_volume_1s` — both from the Rust ingestor feature vector.

## Dependencies

Library: `numpy`, `pandas` (for run_batch DataFrame interface)
Data: kernel library at `models/convolver_kernels.{npz,json}`
Fallback: `make_fallback_kernels()` if no library exists

## Verification

```bash
# Smoke test
pytest scripts/tests/test_algorithm_smoke.py -k convolver

# Evaluate on real data
nat algorithm evaluate --algorithm convolver --symbol BTC
```

```python
from algorithms.convolver import Convolver
algo = Convolver(fallback_to_analytical=True)
assert algo.name() == "convolver"
assert len(algo.alg_features()) == 8
assert "raw_midprice" in algo.required_columns()
```
