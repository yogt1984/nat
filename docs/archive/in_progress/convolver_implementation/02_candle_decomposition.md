# Step 2: Candle Decomposition

## Context

Stage [1] of the pipeline. Raw OHLCV has redundant axes (O and C both encode
level). Decompose into 4 orthogonal signal channels that isolate direction,
rejection, and participation.

## Input

5 arrays of equal length: O (open), H (high), L (low), C (close), V (volume).

## Output

Dict of 4 channels, each same length as input:

| Channel | Formula | Captures | Sign |
|---------|---------|----------|------|
| body | C - O | Direction / momentum | signed (+bullish) |
| upper_wick | H - max(C, O) | Upside rejection | >= 0 always |
| lower_wick | min(C, O) - L | Downside rejection | >= 0 always |
| volume | V | Participation intensity | >= 0 always |

## Implementation

File: `scripts/algorithms/convolver_kernels.py:102-124`
Function: `decompose_candles(O, H, L, C, V) -> dict[str, np.ndarray]`

```python
return {
    "body": C - O,
    "upper_wick": H - np.maximum(C, O),
    "lower_wick": np.minimum(C, O) - L,
    "volume": V.copy(),
}
```

## Why These 4 (Not Raw OHLCV)

- Body isolates directional signal from level noise
- Wicks capture rejection / exhaustion independently of body direction
- Volume is structurally independent of price action
- BTC discovery confirmed: trap_bull volume channel has EVR=26.5%, SI=2.74 —
  volume patterns during traps are stereotyped and distinct from price channels

## Dependencies

Library: `numpy` (np.maximum, np.minimum — element-wise, not np.max/np.min)

## Verification

```python
import numpy as np
from algorithms.convolver_kernels import decompose_candles

O = np.array([100.0, 101.0, 99.0])
H = np.array([102.0, 103.0, 101.0])
L = np.array([99.0,  100.0, 98.0])
C = np.array([101.0, 100.0, 100.0])
V = np.array([50.0,  60.0,  40.0])

ch = decompose_candles(O, H, L, C, V)
assert ch["body"][0] == 1.0          # bullish: C > O
assert ch["body"][1] == -1.0         # bearish: C < O
assert np.all(ch["upper_wick"] >= 0) # always non-negative
assert np.all(ch["lower_wick"] >= 0) # always non-negative
assert np.array_equal(ch["volume"], V)
```
