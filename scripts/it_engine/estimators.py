"""
Information-theoretic estimators for alpha discovery.

Implements:
  - KSG mutual information (Kraskov-Stögbauer-Grassberger, 2004)
  - Conditional mutual information  I(X;Y|Z)
  - Interaction information  II(X;Y;Z) = I(X;Y|Z) - I(X;Y)
  - Linear transfer entropy  TE(X→Y)
  - Cost threshold  I_min(fee, σ_r)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


# ---------------------------------------------------------------------------
# KSG Mutual Information  (Algorithm 1 from Kraskov et al. 2004)
# ---------------------------------------------------------------------------

def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate I(X;Y) in bits using KSG Algorithm 1.

    Uses Chebyshev (max-norm) distance for the joint space ε-ball,
    then counts marginal neighbors within that radius.

    Parameters
    ----------
    x, y : 1-D arrays of shape (N,)
    k    : number of nearest neighbors (default 5)

    Returns
    -------
    Mutual information in bits (≥ 0 by construction after bias correction).
    """
    x, y = _validate_xy(x, y)
    n = len(x)
    if n < k + 1:
        return 0.0

    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x[:, None])
    tree_y = cKDTree(y[:, None])

    # k+1 because query includes the point itself
    dd, _ = tree_xy.query(xy, k=[k + 1], p=np.inf)
    eps = dd[:, 0]  # Chebyshev distance to k-th neighbor in joint space

    # Count marginal neighbors strictly within eps (open ball)
    # cKDTree.query_ball_point with p=inf gives Chebyshev ball
    nx = np.array([
        len(tree_x.query_ball_point([xi], r=ei, p=np.inf)) - 1
        for xi, ei in zip(x, eps)
    ])
    ny = np.array([
        len(tree_y.query_ball_point([yi], r=ei, p=np.inf)) - 1
        for yi, ei in zip(y, eps)
    ])

    # Clamp to at least 1 to avoid digamma(0)
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi_nats = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    return max(float(mi_nats / np.log(2)), 0.0)


# ---------------------------------------------------------------------------
# Conditional Mutual Information  I(X;Y|Z)
# ---------------------------------------------------------------------------

def cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 5) -> float:
    """Estimate I(X;Y|Z) in bits using the KSG-based CMI estimator.

    Uses the identity:
        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    Each entropy term is estimated via the KSG k-NN approach:
        H(V) ≈ digamma(N) - digamma(k) + d·<log(2ε)>

    Parameters
    ----------
    x, y : 1-D arrays of shape (N,)
    z    : 1-D or 2-D array of shape (N,) or (N, d_z)
    k    : number of nearest neighbors

    Returns
    -------
    Conditional mutual information in bits.
    """
    x, y = _validate_xy(x, y)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    n = len(x)
    if n < k + 1:
        return 0.0

    xz = np.column_stack([x[:, None], z])
    yz = np.column_stack([y[:, None], z])
    xyz = np.column_stack([x[:, None], y[:, None], z])

    h_xz = _ksg_entropy(xz, k)
    h_yz = _ksg_entropy(yz, k)
    h_xyz = _ksg_entropy(xyz, k)
    h_z = _ksg_entropy(z, k)

    cmi_bits = (h_xz + h_yz - h_xyz - h_z) / np.log(2)
    return float(cmi_bits)


# ---------------------------------------------------------------------------
# Interaction Information  II(X;Y;Z)
# ---------------------------------------------------------------------------

def interaction_info(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     k: int = 5) -> float:
    """Compute interaction information II(X;Y;Z) = I(X;Y|Z) - I(X;Y).

    Positive → synergy  (conditioning on Z reveals more X↔Y dependence).
    Negative → redundancy (Z shares information with the X↔Y link).

    Returns value in bits.
    """
    mi_xy = ksg_mi(x, y, k=k)
    cmi_xyz = cmi(x, y, z, k=k)
    return cmi_xyz - mi_xy


# ---------------------------------------------------------------------------
# Linear Transfer Entropy  TE(source → target)
# ---------------------------------------------------------------------------

def ksg_te(source: np.ndarray, target: np.ndarray,
           lag: int = 1, order: int = 1, k: int = 5) -> float:
    """Nonparametric transfer entropy TE(source→target) via KSG CMI.

    Uses the identity:
        TE(X→Y) = I(X_past; Y_present | Y_past)

    where I(·;·|·) is estimated by the KSG-based CMI estimator.

    Parameters
    ----------
    source, target : 1-D arrays of shape (N,)
    lag   : how many lags of source to include (default 1)
    order : AR order for target history (default 1)
    k     : number of nearest neighbors for KSG (default 5)

    Returns
    -------
    Transfer entropy in bits. Clamped to ≥ 0.
    """
    source = np.asarray(source, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = len(target)
    max_lag = max(order, lag)
    if n <= max_lag + 1:
        return 0.0

    # X_past: lagged source values (may be multi-column if lag > 1)
    x_past = np.column_stack([
        source[max_lag - i - 1: n - i - 1] for i in range(lag)
    ])
    if x_past.ndim == 1:
        x_past = x_past.ravel()
    elif x_past.shape[1] == 1:
        x_past = x_past.ravel()

    # Y_present: target at current time
    y_present = target[max_lag:]

    # Y_past: lagged target values (conditioning variable)
    y_past = np.column_stack([
        target[max_lag - i - 1: n - i - 1] for i in range(order)
    ])

    # TE = CMI(X_past; Y_present | Y_past)
    te_bits = cmi(x_past, y_present, y_past, k=k)
    return max(te_bits, 0.0)


def linear_te(source: np.ndarray, target: np.ndarray,
              lag: int = 1, order: int = 1) -> float:
    """Estimate linear (Gaussian) transfer entropy TE(source→target) in bits.

    Uses the autoregressive formulation:
        TE = 0.5 × log(var(ε_reduced) / var(ε_full))

    where ε_reduced residuals come from AR(order) on target history only,
    and ε_full includes lagged source terms.

    Parameters
    ----------
    source, target : 1-D arrays of shape (N,)
    lag   : how many lags of source to include (default 1)
    order : AR order for target history (default 1)

    Returns
    -------
    Transfer entropy in bits. Clamped to ≥ 0.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = len(target)
    max_lag = max(order, lag)
    if n <= max_lag + 1:
        return 0.0

    # Build regression matrices
    y = target[max_lag:]  # response

    # Reduced model: target history only
    X_reduced = np.column_stack([
        target[max_lag - i - 1: n - i - 1] for i in range(order)
    ])

    # Full model: target history + source lags
    X_source = np.column_stack([
        source[max_lag - i - 1: n - i - 1] for i in range(lag)
    ])
    X_full = np.column_stack([X_reduced, X_source])

    var_reduced = _ols_residual_var(X_reduced, y)
    var_full = _ols_residual_var(X_full, y)

    if var_full <= 0 or var_reduced <= 0:
        return 0.0

    te_nats = 0.5 * np.log(var_reduced / var_full)
    return max(float(te_nats / np.log(2)), 0.0)


# ---------------------------------------------------------------------------
# Cost Threshold in Bits
# ---------------------------------------------------------------------------

def min_info_bits(fee_rt_bps: float, sigma_r_bps: float,
                  kurtosis: float = 3.0) -> float:
    """Minimum MI (bits) needed to overcome transaction costs.

    Base formula (Gaussian returns):
        I_min = -0.5 × log₂(1 - (fee_RT / σ_r)²)

    Kurtosis correction: the Gaussian rate-distortion bound underestimates
    the information needed for leptokurtic distributions (crypto returns
    typically have kurtosis 5-15). Scale by κ/3 so that fat-tailed returns
    require proportionally more information to overcome costs.

    If fee ≥ σ_r, returns inf (no amount of information suffices).

    Parameters
    ----------
    fee_rt_bps  : round-trip fee in basis points
    sigma_r_bps : return standard deviation in basis points at the target horizon
    kurtosis    : sample kurtosis (Fisher=False, i.e. Gaussian=3.0). Default 3.0
                  preserves backward-compatible Gaussian behavior.

    Returns
    -------
    Minimum mutual information in bits.
    """
    if sigma_r_bps <= 0:
        return float('inf')
    ratio = fee_rt_bps / sigma_r_bps
    if abs(ratio) >= 1.0:
        return float('inf')
    gaussian_bits = -0.5 * np.log2(1.0 - ratio ** 2)
    # Scale by kurtosis ratio: heavier tails → higher information requirement
    kurt_factor = max(kurtosis, 3.0) / 3.0
    return gaussian_bits * kurt_factor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_xy(x, y):
    """Ensure x, y are float64 1-D arrays of equal length, drop NaN pairs."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    assert len(x) == len(y), f"x and y must have same length: {len(x)} vs {len(y)}"
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _ksg_entropy(data: np.ndarray, k: int) -> float:
    """KSG-style differential entropy estimate in nats.

    H(X) ≈ digamma(N) - digamma(k) + d × <log(2ε_i)>
    where ε_i is the Chebyshev distance to the k-th neighbor.
    """
    if data.ndim == 1:
        data = data[:, None]
    n, d = data.shape
    if n < k + 1:
        return 0.0

    tree = cKDTree(data)
    dd, _ = tree.query(data, k=[k + 1], p=np.inf)
    eps = dd[:, 0]

    # Avoid log(0) — replace zero distances with smallest positive
    eps = np.maximum(eps, np.finfo(float).tiny)

    return float(digamma(n) - digamma(k) + d * np.mean(np.log(2.0 * eps)))


def _ols_residual_var(X: np.ndarray, y: np.ndarray) -> float:
    """OLS residual variance using numpy least squares."""
    if X.shape[0] <= X.shape[1]:
        return 0.0
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    resid = y - X @ coef
    return float(np.var(resid))
