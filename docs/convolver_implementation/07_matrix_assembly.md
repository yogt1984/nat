# Step 7: Event-Aligned Matrix Assembly

## Context

Stage [4] of the pipeline. For each (event_type, channel) pair, stack all
normalized W-candle windows into matrix X. This is the input to SVD.

## Input

- `channel_data`: 1D array of one channel (e.g. body values for all candles)
- `event_mask`: boolean array marking event occurrences
- `W`: window width (default 20 candles)
- `atr`: ATR array (same length as channel_data)

## Output

- `X`: shape (n_events, W) — each row is a normalized W-candle window
- `event_indices`: 1D int array of candle indices where events occurred

## Implementation

File: `scripts/analysis/convolver_discovery.py:167-207`
Function: `build_event_matrix(channel_data, event_mask, W, atr) -> (X, indices)`

```python
indices = np.where(event_mask)[0]
valid = indices[indices >= W - 1]                           # need W-1 lookback
valid = valid[np.isfinite(atr[valid]) & (atr[valid] > 0)]  # need valid ATR

rows = []
for idx in valid:
    window = channel_data[idx - W + 1 : idx + 1]           # W candles ending at event
    normed = normalize_window(window, atr[idx])
    rows.append(normed)

return np.stack(rows), np.array(good_idx)
```

## Validity Filters

Three conditions must hold for an event to enter the matrix:
1. Event index >= W-1 (sufficient lookback for full window)
2. ATR at event bar is finite and positive
3. Window contains no NaN values

Rows failing any filter are silently excluded.

## Matrix Dimensions

Typical sizes for BTC 60s data:

| Event Type | Events (12K candles) | Matrix Shape |
|-----------|---------------------|-------------|
| breakout_bull | ~418 | (418, 20) |
| trap_bull | ~136 | (136, 20) |
| trap_bear | ~172 | (172, 20) |

Each (event_type, channel) pair produces one matrix. Total matrices per run:
6 event types x 4 channels = 24.

## Dependencies

Prior steps: candle decomposition (step 2), ATR normalization (step 3),
event detection (steps 4-6).

## Verification

```python
import numpy as np
from analysis.convolver_discovery import build_event_matrix
from algorithms.convolver_kernels import compute_atr

# 30 candles, events at indices 20 and 25
channel = np.random.randn(30)
mask = np.zeros(30, dtype=bool); mask[[20, 25]] = True
H = np.abs(np.random.randn(30)) + 100; L = H - 2; C = H - 1
atr = compute_atr(H, L, C, period=14)

X, idx = build_event_matrix(channel, mask, W=20, atr=atr)
assert X.shape[1] == 20              # W columns
assert X.shape[0] <= 2               # at most 2 events
assert len(idx) == X.shape[0]        # indices match rows
assert np.all(np.abs(X.mean(axis=1)) < 1e-10)  # each row centered
```
