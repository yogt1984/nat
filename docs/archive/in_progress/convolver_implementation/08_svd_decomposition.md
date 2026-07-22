# Step 8: SVD Decomposition and EVR Truncation

## Context

Stage [5] of the pipeline. Thin SVD on the event matrix discovers characteristic
shapes. This is return-blind — no forward return information is used. This is the
first overfitting safeguard.

## Input

Event matrix X from step 7, shape (n_events, W).

## Output

List of `(ConvolverKernel, pvalue)` tuples for components passing the IC gate.
Each kernel's `.kernel` attribute is a unit-norm vector in R^W.

## SVD

```
X = U S V^T

U in R^{n x K}   per-event loadings
S in R^{K x K}   singular values (importance ordering)
V in R^{W x K}   right singular vectors = discovered shapes
```

File: `scripts/analysis/convolver_discovery.py:254-299`
Function: `discover_kernels(X, forward_returns, event_type, channel, ...)`

Call: `np.linalg.svd(X, full_matrices=False)` — thin SVD via LAPACK `dgesdd`.
O(n * W^2) complexity. For n=200-600, W=20: sub-millisecond.

## EVR Truncation

Not all components carry signal. Keep first K where cumulative explained variance
ratio reaches threshold:

```
EVR_k = sigma_k^2 / sum(sigma_j^2)
K = min{ k : sum_{j=1}^{k} EVR_j >= 0.80 }
```

Default threshold: 80%. Typically K=3-6 for W=20. Discards noise components
explaining < 2% variance each.

## Stereotypy Index

```
SI = sigma_1 / sigma_2
```

Measures shape consistency across events. SI > 3: highly stereotyped. SI ~ 1:
heterogeneous.

BTC discovery: turtle_bull body SI=3.12, trap_bull volume SI=2.74.

## Key Property

V_k (right singular vectors) are the discovered shapes. SVD finds them purely
from event-aligned candle shapes — it never sees forward returns. The IC gate
(step 9) decides which shapes predict returns. This separation prevents data
snooping.

## Dependencies

Library: `numpy` (np.linalg.svd, np.cumsum, np.searchsorted)
Prior step: matrix assembly (step 7)

## Verification

```python
import numpy as np

# Synthetic: 100 events, each is noisy version of a known shape
true_shape = np.sin(np.linspace(0, np.pi, 20))
X = np.outer(np.random.randn(100), true_shape) + 0.1 * np.random.randn(100, 20)

U, S, Vt = np.linalg.svd(X, full_matrices=False)
evr = S**2 / (S**2).sum()

assert evr[0] > 0.5                 # first component dominant
cos_sim = abs(np.dot(Vt[0], true_shape / np.linalg.norm(true_shape)))
assert cos_sim > 0.9                # recovers planted shape
assert S[0] / S[1] > 2.0           # high stereotypy
```
