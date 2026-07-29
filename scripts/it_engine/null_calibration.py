"""PROC-12: null-calibration for information-theoretic estimators.

Every IT estimator carries finite-sample bias — KSG has a spurious ~0.07-bit floor —
so a raw estimate alone cannot answer "is this edge real?". This module wraps an
estimator call with a permutation null: the label is shuffled (or circularly shifted)
`n_shuffles` times to break dependence while preserving its marginal, and every finding
is reported as **bits-above-null**, a **z-score**, and an empirical **p-value** rather
than raw bits.

It is the estimator-honesty gate the whole IT / process discovery layer depends on
(spec: docs/specs/process_layer.md §12). Inject `null_calibrate` once around
`ksg_mi` / `cmi` / `ksg_te` and gate findings with `NullResult.informative`.

Reproducibility: the RNG (a seed or numpy Generator) is passed IN — never a global RNG —
so seeded runs are bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

import numpy as np

# estimator(x, y) -> scalar information (bits/nats — units follow the estimator).
Estimator = Callable[[np.ndarray, np.ndarray], float]

# Config defaults (mirrored in config/it_engine.toml [null_calibration]).
DEFAULT_N_SHUFFLES = 200
DEFAULT_NULL_Z_THRESHOLD = 3.0
DEFAULT_I_MIN = 0.0


@dataclass(frozen=True)
class NullResult:
    """Null-calibrated verdict for one estimator call."""
    raw_bits: float          # the estimate on the real (feature, label) pair
    null_mean: float         # mean of the shuffled-label null distribution
    null_std: float          # std of the null (ddof=1)
    bits_above_null: float   # raw_bits - null_mean (the bias-corrected signal)
    z: float                 # (raw - null_mean) / null_std
    p: float                 # empirical one-sided p (fraction of null >= raw), +1-smoothed
    n_shuffles: int

    def informative(
        self,
        i_min: float = DEFAULT_I_MIN,
        z_threshold: float = DEFAULT_NULL_Z_THRESHOLD,
    ) -> bool:
        """Real iff it clears BOTH the effect-size floor and the significance threshold."""
        return self.bits_above_null >= i_min and self.z >= z_threshold

    def to_dict(self) -> dict:
        return asdict(self)


def _as_rng(rng: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def null_calibrate(
    estimator: Estimator,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    method: str = "shuffle",
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> NullResult:
    """Permutation-null calibration of ``estimator(x, y)``.

    Parameters
    ----------
    estimator : callable (x, y) -> float
        Any information estimator (e.g. ``lambda a, b: ksg_mi(a, b, k=5)``).
    x, y : arrays of equal length. `y` (the label/target) is the one shuffled.
    n_shuffles : number of null draws (>= 1).
    method : "shuffle"  -> i.i.d. permutation of y (breaks ALL dependence), or
             "circular" -> random circular shift of y (preserves its autocorrelation,
                           the honest null for serially-correlated targets).
    rng : seed or numpy Generator (passed in for reproducibility).

    Returns
    -------
    NullResult with raw_bits, null_mean/std, bits_above_null, z, p, n_shuffles.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    if method not in ("shuffle", "circular"):
        raise ValueError(f"unknown method {method!r}; use 'shuffle' or 'circular'")

    gen = _as_rng(rng)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"x and y length mismatch: {len(x)} vs {len(y)}")
    n = len(y)

    raw = float(estimator(x, y))

    null = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        if method == "circular":
            shift = int(gen.integers(1, n)) if n > 1 else 0
            y_null = np.roll(y, shift)
        else:
            y_null = gen.permutation(y)
        null[i] = float(estimator(x, y_null))

    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if n_shuffles > 1 else 0.0
    bits_above_null = raw - null_mean
    z = bits_above_null / null_std if null_std > 1e-12 else 0.0
    # +1 smoothing (Phipson & Smyth 2010): p is never exactly 0; p >= 1/(n_shuffles+1).
    p = float((np.count_nonzero(null >= raw) + 1) / (n_shuffles + 1))

    return NullResult(
        raw_bits=raw,
        null_mean=null_mean,
        null_std=null_std,
        bits_above_null=bits_above_null,
        z=z,
        p=p,
        n_shuffles=n_shuffles,
    )


def load_null_config(path: Optional[str] = None) -> dict:
    """Load the ``[null_calibration]`` block from config/it_engine.toml.

    Returns a dict with n_shuffles, null_z_threshold, i_min, method — falling back to
    the module defaults if the file or section is absent.
    """
    defaults = {
        "n_shuffles": DEFAULT_N_SHUFFLES,
        "null_z_threshold": DEFAULT_NULL_Z_THRESHOLD,
        "i_min": DEFAULT_I_MIN,
        "method": "shuffle",
    }
    try:
        from pathlib import Path
        try:
            import tomllib
        except ModuleNotFoundError:  # py < 3.11
            import tomli as tomllib  # type: ignore
        if path is None:
            try:
                import nat_paths
                cfg_path = nat_paths.config_dir() / "it_engine.toml"
            except Exception:
                cfg_path = Path(__file__).resolve().parents[2] / "config" / "it_engine.toml"
        else:
            cfg_path = Path(path)
        if not cfg_path.exists():
            return defaults
        with open(cfg_path, "rb") as fh:
            section = tomllib.load(fh).get("null_calibration", {})
        return {**defaults, **{k: section[k] for k in defaults if k in section}}
    except Exception:
        return defaults
