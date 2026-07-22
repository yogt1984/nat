# Step 5: Turtle Soup Event Detection

## Context

Stage [2b] of the pipeline. Turtle soup events are false range breaks: price
penetrates a new N-bar extreme intrabar but closes back inside. They exploit
the weakness in breakout systems and carry moderate forward information.

TA origin: Raschke & Connors, *Street Smarts* (1995).

## Turtle Soup Bull

```
L_t < min(L_{t-N}, ..., L_{t-1})  AND  C_t > min(L_{t-N}, ..., L_{t-1})
```

Low undercuts the N-bar low (looks like a bearish breakout starting), but close
recovers above it (the break fails).

File: `scripts/analysis/convolver_discovery.py:100-105`
Function: `_detect_turtle_soup_bull(H, L, C, N) -> np.ndarray`

## Turtle Soup Bear

```
H_t > max(H_{t-N}, ..., H_{t-1})  AND  C_t < max(H_{t-N}, ..., H_{t-1})
```

High exceeds the N-bar high but close falls back below. Traps breakout longs.

File: `scripts/analysis/convolver_discovery.py:108-112`
Function: `_detect_turtle_soup_bear(H, L, C, N) -> np.ndarray`

## No Volume Requirement

Unlike breakouts, turtle soup events deliberately have NO volume filter. False
breakouts are characterized by low follow-through — requiring volume would filter
out exactly what this detector targets.

## Mechanical Forward Information

When price breaks to a new N-bar low, breakout systems enter short. If the bar
closes back above the previous low, those shorts are trapped. Their stop-losses
sit above the failed break level. As stops trigger, buying pressure creates a
mean-reversion impulse. Reversal strength depends on aggregate trapped position.

## Parameters

| Param | Default | Role |
|-------|---------|------|
| N | 20 | Lookback for N-bar high/low reference |

## Empirical Results (BTC, May 2026)

Rate: 20-24 per 1000 candles (~29-34/day at 60s).
turtle_bull rolling SVD stability: rho = 0.920 (shapes temporally consistent).
BH-FDR: turtle kernels survive. Walk-forward: < 0.50 (data volume limitation).

## Verification

```python
import numpy as np
from analysis.convolver_discovery import _detect_turtle_soup_bull

# Bar that undercuts 20-bar low intrabar but closes above it
L = np.concatenate([np.full(25, 100.0), [98.0]])   # new low at bar 25
H = np.concatenate([np.full(25, 105.0), [104.0]])
C = np.concatenate([np.full(25, 103.0), [101.0]])   # closes above prev low

mask = _detect_turtle_soup_bull(H, L, C, N=20)
assert mask[25] == True   # false break detected
assert mask[24] == False   # no event before
```
