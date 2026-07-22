# Step 6: Trap Event Detection

## Context

Stage [2c] of the pipeline. Traps are confirmed false breakouts: a volume-
confirmed breakout occurs, then price reverses by more than 1 ATR within K bars.
Traps carry the STRONGEST forward information — highest rolling SVD stability
(rho=0.941) and volume channel carries independent structural signal.

## Bull Trap

```
breakout_bull(t) = True  AND  C_{t+K} < C_t - alpha * ATR_t
```

File: `scripts/analysis/convolver_discovery.py:116-128`
Function: `_detect_bull_trap(H, L, C, V, N, K, alpha, atr, vol_mult) -> np.ndarray`

Breakout longs get trapped. Stop-loss selling + reversal recognition creates a
positive feedback loop driving price below the failed breakout level.

## Bear Trap

```
breakout_bear(t) = True  AND  C_{t+K} > C_t + alpha * ATR_t
```

File: `scripts/analysis/convolver_discovery.py:131-143`
Function: `_detect_bear_trap(H, L, C, V, N, K, alpha, atr, vol_mult) -> np.ndarray`

## Lookahead Warning

Traps are the ONLY event type with a lookahead (K bars). The trap at bar t is
confirmed at bar t+K. In offline discovery: all data available, no issue. In
online scoring: the convolver scores the W-candle window ending at t+K, which
contains only past data at detection time.

## Parameters

| Param | Default | Role |
|-------|---------|------|
| N | 20 | Donchian lookback (inherited from breakout) |
| vol_mult | 1.5 | Volume confirmation (inherited from breakout) |
| K | 3 | Lookahead bars for reversal confirmation |
| alpha | 1.0 | ATR-scaled reversal threshold |

## Dependencies

Prior steps: breakout detection (step 4), ATR computation (step 3).
Trap detection internally calls `_detect_breakout_bull/bear`.

## Empirical Results (BTC, May 2026)

Rate: 11-14 per 1000 candles (~16-21/day at 60s). **Bottleneck** for SVD.
trap_bull volume: EVR=26.5%, SI=2.74 — stereotyped volume pattern.
trap_bull rolling stability: rho=0.941 (highest of all 6 event types).
BH-FDR: trap kernels survive.

## Sample Size Constraint

Traps limit discovery quality. Minimum data for robust SVD:
- 100 events per (event, channel): needs ~31K candles (~22 days)
- 200 OOS events for IC: needs ~39K candles (~27 days)
- Full analysis: `docs/research/new/convolver_data_analysis.txt`

## Verification

```python
import numpy as np
from algorithms.convolver_kernels import compute_atr
from analysis.convolver_discovery import _detect_bull_trap

H = np.concatenate([np.full(25, 100.0), [110.0, 109.0, 108.0, 104.0]])
L = H - 2.0; C = H - 1.0
V = np.concatenate([np.full(25, 10.0), [20.0, 10.0, 10.0, 10.0]])
atr = compute_atr(H, L, C, period=14)

mask = _detect_bull_trap(H, L, C, V, N=20, K=3, alpha=1.0, atr=atr, vol_mult=1.5)
# Bar 25 is breakout; bar 28 confirms reversal > 1*ATR below breakout close
```
