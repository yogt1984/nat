# Step 12: Analytical Basis Dictionary

## Context

A fixed set of 10 unit-norm parametric functions. Serves two roles:
1. Interpretability diagnostic — cosine similarity reveals if SVD shapes are
   novel or match known analytical forms
2. Fallback kernels — 6 analytical kernels for smoke tests when no SVD library
   exists (first run, CI)

## The 10 Basis Functions

All defined on tau in [0, 1], discretized to W points, L2-normalized.

| Family | Name | Formula | Shape |
|--------|------|---------|-------|
| Polynomial | phi_q1 | tau^2 | Quadratic ramp up |
| | phi_q2 | (tau - 0.5)^2 | Curvature (U-shape) |
| | phi_q3 | (1 - tau)^2 | Quadratic ramp down |
| Cubic | phi_c1 | tau^3 | Steep ramp up |
| | phi_c2 | (tau - 0.5)^3 | S-curve / turning point |
| | phi_c3 | (1 - tau)^3 | Steep ramp down |
| | phi_s1 | 3*tau^2 - 2*tau^3 | Smoothstep |
| Exponential | phi_e1 | exp(-a*tau) | Decay spike |
| | phi_e2 | exp(-a*(1-tau)) | Climax spike |
| | phi_b1 | exp(-a*(tau-0.5)^2) | Gaussian bump |

Parameter a=5.0 (decay rate, configurable in [3, 8]).

File: `scripts/algorithms/convolver_kernels.py:233-277`
Function: `analytical_basis(W, a=5.0) -> dict[str, np.ndarray]`

## Alignment Check

```
alignment(v, phi) = |cos(v, phi)| = |<v, phi>| / (||v|| * ||phi||)
```

File: `scripts/algorithms/convolver_kernels.py:280-297`
Function: `basis_alignment(v, basis_dict) -> dict[str, float]`

- \> 0.85: strong alignment (SVD shape matches a known analytical form)
- < 0.50: novel shape (data-driven, not explainable by simple parametrics)

## Fallback Kernels

File: `scripts/algorithms/convolver_kernels.py:305-340`
Function: `make_fallback_kernels(W=20, a=5.0) -> KernelLibrary`

6 kernels (one per event type), body channel only:
- breakout_bull: phi_s1 (smoothstep ramp)
- breakout_bear: -phi_s1
- turtle_bull: phi_c2 (S-curve turning point)
- turtle_bear: -phi_c2
- trap_bull: phi_e2 - phi_e1 (spike then reversal)
- trap_bear: -(phi_e2 - phi_e1)

Used by `convolver.py` when `fallback_to_analytical=True` and no `.npz` exists.

## Verification

```python
import numpy as np
from algorithms.convolver_kernels import analytical_basis, basis_alignment

basis = analytical_basis(W=20, a=5.0)
assert len(basis) == 10

# All unit-norm
for name, vec in basis.items():
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-10, f"{name} not unit norm"

# Self-alignment is 1.0
for name, vec in basis.items():
    align = basis_alignment(vec, basis)
    assert abs(align[name] - 1.0) < 1e-10

# Random vector has low alignment with all basis functions
rnd = np.random.randn(20); rnd /= np.linalg.norm(rnd)
align = basis_alignment(rnd, basis)
assert max(align.values()) < 0.9  # unlikely to match any basis closely
```
