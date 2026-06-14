# Step 3: ATR Computation and Window Normalization

## Context

Stage [3] of the pipeline. Before SVD, each W-candle window must be normalized
to remove absolute price level and volatility scale. A $100K BTC pattern and a
$3K ETH pattern with the same shape must produce identical normalized windows.

## ATR Computation

Average True Range (Wilder 1978). Smoothed measure of bar-to-bar volatility.

```
TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
ATR_t = (1/N) * sum_{j=0}^{N-1} TR_{t-j}
```

File: `scripts/algorithms/convolver_kernels.py:132-151`
Function: `compute_atr(H, L, C, period=14) -> np.ndarray`

Returns array of same length as input. First `period` values are NaN (warmup).

## Window Normalization

Shape-preserving: center by mean, scale by ATR at window endpoint.

```
x_norm = (x - mean(x)) / ATR_t
```

File: `scripts/algorithms/convolver_kernels.py:159-168`
Function: `normalize_window(window, atr) -> np.ndarray`

Two effects:
1. Center by mean — removes absolute level
2. Scale by ATR — a 2% move in low-vol vs high-vol produces different magnitudes

Edge case: ATR <= 0 or non-finite returns zero vector. This row gets excluded
from the event matrix by the validity check in `build_event_matrix()`.

## Dependencies

Library: `numpy` (np.nanmean, np.isfinite, np.zeros_like)
Prior step: candle decomposition (step 2) produces the channel arrays to normalize

## Verification

```python
import numpy as np
from algorithms.convolver_kernels import compute_atr, normalize_window

H = np.array([102, 104, 103, 105, 106, 104, 107, 105, 108, 106,
              109, 107, 110, 108, 111])  # length 15
L = H - 2.0
C = H - 1.0

atr = compute_atr(H, L, C, period=14)
assert np.all(np.isnan(atr[:13]))       # first 13 values are NaN
assert np.isfinite(atr[14])             # 15th value computed

window = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
normed = normalize_window(window, atr=2.0)
assert abs(normed.mean()) < 1e-10       # centered
assert abs(np.std(normed) - np.std(window)/2.0) < 1e-10  # scaled by ATR

# Edge case: zero ATR returns zero vector
assert np.all(normalize_window(window, atr=0.0) == 0.0)
```
