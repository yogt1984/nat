# Step 11: Kernel Library Persistence

## Context

Stage [8] of the pipeline. Surviving kernels are saved in dual format for both
fast numeric loading and human-readable inspection. Follows NAT convention of
JSON state files + binary data.

## Output Format

**`.npz` file** — kernel vectors stacked as 2D NumPy array (n_kernels x W).
Compressed via `np.savez_compressed()`. Typical size: ~2 KB.

**`.json` file** — all metadata: event_type, channel, component_idx, IC,
ic_pvalue, EVR per kernel, plus library-level: window_width, atr_period,
channel_weights, discovery_date, train_end, horizons, n_kernels.

## Save

File: `scripts/algorithms/convolver_kernels.py:348-399`
Function: `save_kernel_library(library: KernelLibrary, path: Path) -> None`

Accepts path with or without extension. Creates `<path>.npz` and `<path>.json`.
Parent directory is created if missing.

## Load

File: `scripts/algorithms/convolver_kernels.py:402-455`
Function: `load_kernel_library(path: Path) -> KernelLibrary`

Reconstructs full KernelLibrary with numpy kernel vectors + metadata.
Raises FileNotFoundError if either .npz or .json is missing.

## Data Structures

```python
@dataclass(frozen=True)
class ConvolverKernel:
    event_type: str        # "breakout_bull", "turtle_bear", etc.
    channel: str           # "body", "upper_wick", "lower_wick", "volume"
    component_idx: int     # SVD component index k (0-based)
    kernel: np.ndarray     # v_k in R^W, unit norm
    ic: float              # signed IC at primary horizon
    ic_pvalue: float       # p-value from t-test
    evr: float             # explained variance ratio
```

## Why .npz + .json (Not Alternatives)

| Format | Why Not |
|--------|---------|
| HDF5 / h5py | Requires C library dependency. Overkill for 6-20 vectors. |
| Parquet | Schema overhead exceeds data size for small dense arrays. |
| Pickle | Security risk (arbitrary code execution on load). |
| Custom binary | `.npz` is standard, portable, compressed, zero-effort. |

## Dependencies

Library: `numpy` (np.savez_compressed, np.load), `json` (dump, load)
Data: `ConvolverKernel`, `KernelLibrary` dataclasses

## Verification

```python
from pathlib import Path
from algorithms.convolver_kernels import (
    save_kernel_library, load_kernel_library, make_fallback_kernels
)

lib = make_fallback_kernels(W=20)
save_kernel_library(lib, Path("/tmp/test_kernels"))

assert Path("/tmp/test_kernels.npz").exists()
assert Path("/tmp/test_kernels.json").exists()

loaded = load_kernel_library(Path("/tmp/test_kernels"))
assert len(loaded.kernels) == len(lib.kernels)
assert loaded.window_width == 20
for orig, load in zip(lib.kernels, loaded.kernels):
    assert orig.event_type == load.event_type
    assert np.allclose(orig.kernel, load.kernel)
```
