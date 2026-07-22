# Step 9: Information Coefficient Gate

## Context

Stage [6] of the pipeline. For each SVD component surviving EVR truncation,
test whether its per-event loadings predict forward returns. This bridges
the return-blind SVD shapes with actual return predictability.

## Computation

```
IC_k = corr(U_k * sigma_k, r_forward)
```

Where:
- U_k * sigma_k: scaled loadings (how strongly each event projects onto shape k)
- r_forward: log-return over horizon h following each event

File: `scripts/analysis/convolver_discovery.py:215-230`
Function: `_compute_ic(loadings, returns) -> (ic, pvalue)`

## Threshold

Retain components with |IC| >= 0.03. Deliberately lenient — the BH-FDR
correction (step 10) handles the multiple testing burden.

## Significance Test

t-test under H0: IC = 0:

```
t = IC * sqrt((n - 2) / (1 - IC^2))
p = 2 * (1 - CDF_t(|t|, df=n-2))
```

P-values are collected across ALL (event_type, channel, component)
combinations and passed to BH-FDR as a batch.

Minimum sample: n < 30 returns (ic=0.0, pval=1.0) to avoid spurious estimates.

## Horizon

The discovery pipeline uses the PRIMARY horizon only (first element of the
horizons list, default h=10 candles = 10 minutes at 60s). Forward returns:

```
r_forward[t] = log(C[t+h] / C[t])
```

The horizons list [10, 50, 100] is accepted as config but only `horizons[0]`
is used for IC computation (see `convolver_discovery.py:622`). Multi-horizon
testing is a future extension.

## Dependencies

Library: `scipy.stats.t.cdf` for p-value computation
Prior step: SVD decomposition (step 8)

Why scipy over normal approximation: for n < 100 (sparse trap events),
the t-distribution correction matters. Normal approx underestimates p-values
for small samples, inflating false discovery rates.

## Verification

```python
import numpy as np
from scipy import stats

# Known positive IC: loadings correlate with returns
np.random.seed(42)
n = 200
loadings = np.random.randn(n)
returns = 0.3 * loadings + 0.9 * np.random.randn(n)  # IC ~ 0.3

ic = np.corrcoef(loadings, returns)[0, 1]
t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-15))
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

assert 0.2 < ic < 0.4          # plausible IC
assert p_val < 0.01             # significant
assert abs(ic) >= 0.03          # passes threshold

# Edge case: uncorrelated
returns_null = np.random.randn(n)
ic_null = np.corrcoef(loadings, returns_null)[0, 1]
assert abs(ic_null) < 0.15     # no signal
```
