# Step 10: Benjamini-Hochberg FDR Correction

## Context

Stage [7] of the pipeline. Second overfitting safeguard. With 6 event types x
4 channels x ~5 SVD components = ~120 tests, uncorrected p-values
produce many false positives. BH-FDR controls the expected false discovery
proportion at q=0.05.

## Algorithm

Benjamini & Hochberg (1995):

```
1. Sort all m p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m)
2. Find the largest rank j where p_(j) <= j * q / m
3. Reject all hypotheses with rank <= j
```

File: `scripts/analysis/convolver_discovery.py:233-251`
Function: `_bh_fdr(pvalues, q=0.05) -> np.ndarray` (boolean rejection mask)

## Input

Array of p-values from all IC tests across the entire discovery run.
One p-value per (event_type, channel, component, horizon) combination.

## Output

Boolean mask: True = kernel survives, False = rejected as likely false discovery.

## Implementation

```python
sorted_idx = np.argsort(pvalues)
sorted_p = pvalues[sorted_idx]
thresholds = np.arange(1, m + 1) * q / m

reject = sorted_p <= thresholds
if not np.any(reject):
    return np.zeros(m, dtype=bool)  # nothing survives

j_star = np.max(np.where(reject)[0])
result = np.zeros(m, dtype=bool)
result[sorted_idx[:j_star + 1]] = True
return result
```

## Empirical Result

BTC discovery (May 2026): 158 IC-passing candidates -> 6 BH-FDR survivors.
All 6 are turtle/trap types. Zero breakout kernels survived.

## Dependencies

Library: `numpy` (np.argsort, np.where)
Prior step: IC gate (step 9) provides the p-values

## Verification

```python
import numpy as np
from analysis.convolver_discovery import _bh_fdr

# 5 tests: 2 real (low p), 3 noise (high p)
pvals = np.array([0.001, 0.003, 0.2, 0.5, 0.8])
mask = _bh_fdr(pvals, q=0.05)
assert mask[0] == True and mask[1] == True    # real signals survive
assert mask[2] == False and mask[3] == False  # noise rejected

# All high p-values: nothing survives
mask_null = _bh_fdr(np.array([0.3, 0.5, 0.7, 0.9]), q=0.05)
assert not np.any(mask_null)

# Empty input
mask_empty = _bh_fdr(np.array([]), q=0.05)
assert len(mask_empty) == 0
```
