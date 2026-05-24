"""
Convolver Kernel Library — Shared Math and Persistence
=======================================================

Pure math module for the convolver pattern detection system.
Used by both the offline SVD discovery pipeline (convolver_discovery.py)
and the online production algorithm (convolver.py).

See docs/convolver_method.tex for full mathematical specification.

Contents:
  Data structures:  ConvolverKernel, KernelLibrary
  Candle math:      decompose_candles, compute_atr, normalize_window
  Scoring:          cosine_similarity, score_all_kernels
  Basis functions:  analytical_basis, basis_alignment
  Persistence:      save_kernel_library, load_kernel_library
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "breakout_bull",
    "breakout_bear",
    "turtle_bull",
    "turtle_bear",
    "trap_bull",
    "trap_bear",
]

CHANNELS = ["body", "upper_wick", "lower_wick", "volume"]

# ---------------------------------------------------------------------------
# Data structures (tex §8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvolverKernel:
    """One discovered kernel: a unit-norm vector with metadata.

    Attributes:
        event_type:    e.g. "breakout_bull", "turtle_bear"
        channel:       e.g. "body", "upper_wick", "lower_wick", "volume"
        component_idx: SVD component index k (0-based)
        kernel:        v_k in R^W, unit norm
        ic:            signed IC at primary horizon
        ic_pvalue:     p-value from t-test under H0: IC=0
        evr:           explained variance ratio sigma_k^2 / sum(sigma^2)
    """

    event_type: str
    channel: str
    component_idx: int
    kernel: np.ndarray
    ic: float
    ic_pvalue: float
    evr: float


@dataclass
class KernelLibrary:
    """Full set of discovered kernels with metadata.

    Attributes:
        kernels:         list of ConvolverKernel
        window_width:    W (candles in pattern window)
        atr_period:      N_atr for ATR computation
        channel_weights: per-channel aggregation weights {channel: w}
        discovery_date:  ISO timestamp of discovery run
        train_end:       last candle timestamp in training set
        horizons:        forward-return horizons tested
    """

    kernels: list[ConvolverKernel]
    window_width: int
    atr_period: int
    channel_weights: dict[str, float] = field(
        default_factory=lambda: {c: 0.25 for c in CHANNELS}
    )
    discovery_date: str = ""
    train_end: str = ""
    horizons: list[int] = field(default_factory=lambda: [10, 50, 100])


# ---------------------------------------------------------------------------
# Candle decomposition (tex §3, eq 2-5)
# ---------------------------------------------------------------------------


def decompose_candles(
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose OHLCV candles into 4 signal channels.

    Args:
        O, H, L, C: price arrays (same length)
        V: volume array

    Returns:
        {"body": C-O, "upper_wick": H-max(C,O),
         "lower_wick": min(C,O)-L, "volume": V}
    """
    return {
        "body": C - O,
        "upper_wick": H - np.maximum(C, O),
        "lower_wick": np.minimum(C, O) - L,
        "volume": V.copy(),
    }


# ---------------------------------------------------------------------------
# ATR computation (tex §3, eq 6)
# ---------------------------------------------------------------------------


def compute_atr(
    H: np.ndarray, L: np.ndarray, C: np.ndarray, period: int = 14
) -> np.ndarray:
    """Vectorized Average True Range.

    ATR_t = (1/period) * sum_{j=0}^{period-1} TR_{t-j}
    TR_t  = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)

    Returns array of same length as inputs; first `period` values are NaN.
    """
    n = len(H)
    tr = np.empty(n)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))

    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1 : i + 1])
    return atr


# ---------------------------------------------------------------------------
# Window normalization (tex §3, eq 8)
# ---------------------------------------------------------------------------


def normalize_window(window: np.ndarray, atr: float) -> np.ndarray:
    """Normalize a W-length window: subtract mean, divide by ATR.

    Shape-preserving normalization that removes level and scale.
    Returns zero vector if ATR <= 0.
    """
    if atr <= 0 or not np.isfinite(atr):
        return np.zeros_like(window)
    centered = window - np.nanmean(window)
    return centered / atr


# ---------------------------------------------------------------------------
# Cosine similarity scoring (tex §9, eq 19)
# ---------------------------------------------------------------------------


def cosine_similarity(x: np.ndarray, kernel: np.ndarray) -> float:
    """Cosine similarity between a normalized window and a unit-norm kernel.

    s = <x, k> / (||x|| * ||k||)

    Since kernels are unit-norm (SVD right singular vectors), simplifies to:
    s = <x, k> / ||x||

    Returns 0.0 if ||x|| is near zero.
    """
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        return 0.0
    k_norm = np.linalg.norm(kernel)
    if k_norm < 1e-12:
        return 0.0
    return float(np.dot(x, kernel) / (x_norm * k_norm))


def score_all_kernels(
    channels: dict[str, np.ndarray],
    library: KernelLibrary,
) -> dict[str, float]:
    """Score one window against all kernels, aggregate across channels.

    For each event type e:
      S^(e)(t) = sum_c w_c * sum_j IC_j * s^(e,c)_j(t)

    Args:
        channels: {channel_name: normalized_window} for the current position
        library:  KernelLibrary with kernels and channel_weights

    Returns:
        {event_type: composite_score} for each event type
    """
    scores: dict[str, float] = {et: 0.0 for et in EVENT_TYPES}

    for kernel in library.kernels:
        ch = kernel.channel
        if ch not in channels:
            continue
        window = channels[ch]
        if len(window) != len(kernel.kernel):
            continue
        sim = cosine_similarity(window, kernel.kernel)
        w = library.channel_weights.get(ch, 0.25)
        # IC-weighted contribution (tex §9, eq 21)
        scores[kernel.event_type] += w * kernel.ic * sim

    return scores


# ---------------------------------------------------------------------------
# Analytical basis functions (tex §12, eq 25-34)
# ---------------------------------------------------------------------------


def analytical_basis(W: int, a: float = 5.0) -> dict[str, np.ndarray]:
    """Generate the 10-element discretized analytical basis dictionary.

    All functions are defined on normalized tau_bar in [0, 1], discretized
    to W points. Each basis is L2-normalized to unit norm.

    Families:
      Polynomial (3): quadratic ramps and curvature
      Cubic (4):      S-curves, turning points, smoothstep
      Exponential (3): spikes, bumps, climax

    Args:
        W: window width (number of discrete points)
        a: exponential decay parameter, a in [3, 8]

    Returns:
        {name: unit_norm_vector} for 10 basis functions
    """
    tau = np.linspace(0, 1, W)

    raw = {
        # Polynomial family
        "phi_q1": tau**2,
        "phi_q2": (tau - 0.5) ** 2,
        "phi_q3": (1 - tau) ** 2,
        # Cubic family
        "phi_c1": tau**3,
        "phi_c2": (tau - 0.5) ** 3,
        "phi_c3": (1 - tau) ** 3,
        "phi_s1": 3 * tau**2 - 2 * tau**3,  # smoothstep
        # Exponential family
        "phi_e1": np.exp(-a * tau),
        "phi_e2": np.exp(-a * (1 - tau)),
        "phi_b1": np.exp(-a * (tau - 0.5) ** 2),  # Gaussian bump
    }

    # L2-normalize each basis
    normalized = {}
    for name, vec in raw.items():
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            normalized[name] = vec / norm
        else:
            normalized[name] = vec
    return normalized


def basis_alignment(
    v: np.ndarray, basis_dict: dict[str, np.ndarray]
) -> dict[str, float]:
    """Cosine similarity between an SVD-discovered vector and each analytical basis.

    Values > 0.85 indicate strong alignment (interpretable shape).
    Values < 0.5 indicate the data-driven shape is novel.

    Args:
        v: discovered basis vector (unit norm, length W)
        basis_dict: output of analytical_basis(W)

    Returns:
        {basis_name: |cos(v, phi)|}
    """
    return {
        name: abs(cosine_similarity(v, phi)) for name, phi in basis_dict.items()
    }


# ---------------------------------------------------------------------------
# Fallback kernels from analytical basis (for smoke tests)
# ---------------------------------------------------------------------------


def make_fallback_kernels(W: int = 20, a: float = 5.0) -> KernelLibrary:
    """Construct a minimal kernel library from analytical basis functions.

    Used when no SVD-discovered kernels exist (e.g., first run, smoke tests).
    Creates 6 kernels (one per event type) from the body channel only:
      breakout_bull:  smoothstep phi_s1 (gradual ramp up)
      breakout_bear: -phi_s1 (gradual ramp down)
      turtle_bull:    cubic phi_c2 (dip then reversal)
      turtle_bear:   -phi_c2 (spike then reversal)
      trap_bull:      phi_e2 followed by decline (spike at end then drop)
      trap_bear:     -trap_bull
    """
    basis = analytical_basis(W, a)

    # Trap kernel: ramp up then sharp reversal (phi_e2 - phi_e1)
    trap_raw = basis["phi_e2"] - basis["phi_e1"]
    trap_norm = np.linalg.norm(trap_raw)
    trap_kernel = trap_raw / trap_norm if trap_norm > 1e-12 else trap_raw

    kernels = [
        ConvolverKernel("breakout_bull", "body", 0, basis["phi_s1"], 0.02, 0.1, 0.5),
        ConvolverKernel("breakout_bear", "body", 0, -basis["phi_s1"], -0.02, 0.1, 0.5),
        ConvolverKernel("turtle_bull", "body", 0, basis["phi_c2"], 0.02, 0.1, 0.3),
        ConvolverKernel("turtle_bear", "body", 0, -basis["phi_c2"], -0.02, 0.1, 0.3),
        ConvolverKernel("trap_bull", "body", 0, trap_kernel, 0.01, 0.2, 0.2),
        ConvolverKernel("trap_bear", "body", 0, -trap_kernel, -0.01, 0.2, 0.2),
    ]

    return KernelLibrary(
        kernels=kernels,
        window_width=W,
        atr_period=14,
        channel_weights={"body": 1.0, "upper_wick": 0.0, "lower_wick": 0.0, "volume": 0.0},
        discovery_date=datetime.now(timezone.utc).isoformat(),
        train_end="analytical_fallback",
    )


# ---------------------------------------------------------------------------
# Persistence (.npz + .json, follows cluster_pipeline/reduction.py pattern)
# ---------------------------------------------------------------------------


def save_kernel_library(library: KernelLibrary, path: Path) -> None:
    """Save kernel library to disk as .npz (arrays) + .json (metadata).

    Creates:
      <path>.npz  - kernel vectors stacked as 2D array
      <path>.json - all metadata + per-kernel attributes
    """
    path = Path(path)
    if path.suffix in (".npz", ".json"):
        path = path.with_suffix("")

    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    # Ensure parent directory exists
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Stack all kernel vectors into one 2D array
    if library.kernels:
        kernel_matrix = np.stack([k.kernel for k in library.kernels])
    else:
        kernel_matrix = np.empty((0, library.window_width))

    np.savez_compressed(npz_path, kernels=kernel_matrix)

    # Build JSON metadata
    kernel_meta = []
    for k in library.kernels:
        kernel_meta.append(
            {
                "event_type": k.event_type,
                "channel": k.channel,
                "component_idx": k.component_idx,
                "ic": float(k.ic),
                "ic_pvalue": float(k.ic_pvalue),
                "evr": float(k.evr),
            }
        )

    metadata = {
        "window_width": library.window_width,
        "atr_period": library.atr_period,
        "channel_weights": library.channel_weights,
        "discovery_date": library.discovery_date,
        "train_end": library.train_end,
        "horizons": library.horizons,
        "n_kernels": len(library.kernels),
        "kernels": kernel_meta,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_kernel_library(path: Path) -> KernelLibrary:
    """Load kernel library from .npz + .json files.

    Args:
        path: base path (e.g. Path("models/convolver_kernels")).
              Accepts .npz, .json, or no extension.

    Returns:
        Reconstructed KernelLibrary.

    Raises:
        FileNotFoundError: if either file is missing.
    """
    path = Path(path)
    if path.suffix in (".npz", ".json"):
        path = path.with_suffix("")

    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Kernel .npz not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Kernel .json not found: {json_path}")

    data = np.load(npz_path)
    kernel_matrix = data["kernels"]

    with open(json_path) as f:
        meta = json.load(f)

    kernels = []
    for i, km in enumerate(meta["kernels"]):
        kernels.append(
            ConvolverKernel(
                event_type=km["event_type"],
                channel=km["channel"],
                component_idx=km["component_idx"],
                kernel=kernel_matrix[i],
                ic=km["ic"],
                ic_pvalue=km["ic_pvalue"],
                evr=km["evr"],
            )
        )

    return KernelLibrary(
        kernels=kernels,
        window_width=meta["window_width"],
        atr_period=meta["atr_period"],
        channel_weights=meta.get("channel_weights", {c: 0.25 for c in CHANNELS}),
        discovery_date=meta.get("discovery_date", ""),
        train_end=meta.get("train_end", ""),
        horizons=meta.get("horizons", [10, 50, 100]),
    )
