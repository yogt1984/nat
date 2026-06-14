# Step 4: Breakout Event Detection

## Context

Stage [2a] of the pipeline. Breakouts are Donchian channel breaks with volume
confirmation. They are the most common event type but carry the LEAST forward
information — zero breakout kernels survived BH-FDR in BTC discovery (May 2026).

## Breakout Bull

```
C_t > max(H_{t-N}, ..., H_{t-1})  AND  V_t > vol_mult * median(V_{t-N}, ..., V_{t-1})
```

Price closes above the N-bar high with volume exceeding 1.5x rolling median.

File: `scripts/analysis/convolver_discovery.py:80-87`
Function: `_detect_breakout_bull(H, L, C, V, N, vol_mult) -> np.ndarray`

TA origin: Donchian channel breakout (Donchian 1960). Foundation of turtle
trading systems.

## Breakout Bear

```
C_t < min(L_{t-N}, ..., L_{t-1})  AND  V_t > vol_mult * median(V_{t-N}, ..., V_{t-1})
```

File: `scripts/analysis/convolver_discovery.py:90-97`
Function: `_detect_breakout_bear(H, L, C, V, N, vol_mult) -> np.ndarray`

## Why Volume Filter

Without it, thin-market spikes (gap opens, single large orders) register as
breakouts. These lack broad participation and carry no directional information.
The 1.5x median threshold keeps only breaks with genuine order flow.

## Parameters

| Param | Default | Role |
|-------|---------|------|
| N | 20 | Lookback window for Donchian channel |
| vol_mult | 1.5 | Volume confirmation multiplier |

## Dependencies

Library: `numpy`
Helpers: `_rolling_max()`, `_rolling_min()`, `_rolling_median()` at
`convolver_discovery.py:56-77`

## Empirical Result

BTC (May 2026): breakout_bull 418 events, breakout_bear 522 events in 12K candles.
Rate: 34-43 per 1000 candles (~50-62/day at 60s).
BH-FDR survivors: **0 out of 158 breakout candidates**. Breakout patterns are the
most widely traded systematic strategy in crypto — their information content is
arbitraged away.

## Verification

```python
import numpy as np
# Construct a clear breakout: price jumps above 20-bar high with high volume
H = np.concatenate([np.full(25, 100.0), [110.0]])
L = H - 2.0
C = H - 1.0
V = np.concatenate([np.full(25, 10.0), [20.0]])  # 2x median vol

from analysis.convolver_discovery import _detect_breakout_bull
mask = _detect_breakout_bull(H, L, C, V, N=20, vol_mult=1.5)
assert mask[25] == True   # last bar is a breakout
assert mask[24] == False  # bar before is not
```
