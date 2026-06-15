"""Realized higher moments (F5 in feature_algorithm_gaps.md).

Rolling realized skewness / excess-kurtosis of returns at 5m–1h windows, plus a
signed-jump (realized-semivariance) asymmetry. These are **conditioning /
tail-risk** features, not alpha on their own: realized skewness predicts the
cross-section of short-horizon returns (Amaya, Christoffersen, Jacobs & Vasquez
2015), and the up/down semivariance split (Barndorff-Nielsen, Kinnebrock &
Shephard 2010) separates "good" from "bad" volatility — both useful to gate the
jump_detector and to feed the daily/macro agents.

Derived purely from ``raw_midprice`` + ``timestamp_ns`` — no ingestor or schema
changes (the design rule in feature_algorithm_gaps.md).

Features (prefix ``rm_``):

    rm_skew_5m / _15m / _1h    Rolling skewness of grid log-returns
    rm_kurt_5m / _15m / _1h    Rolling excess kurtosis (Fisher; 0 = Gaussian)
    rm_signed_jump_1h          (RS+ − RS−)/(RS+ + RS−) over 1h, in [−1, 1];
                               <0 = downside-dominated ("bad") variance

Construction (per symbol, 24/7 crypto):

    Returns are taken on a 10-second resampled grid — noise-robust enough to
    estimate third/fourth moments (naive 100ms returns are pure microstructure
    noise, which dominates higher moments far worse than variance). Gap cells are
    NaN so no return spans a data gap; rolling windows are time-based with a
    coverage-based ``min_periods`` guard. Each window value is shifted one grid
    step and forward-filled back onto the tick index — strictly causal, no
    lookahead — exactly like the HAR-RV and settlement-clock features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRID_S = 10.0  # 10-second return grid (noise-robust for higher moments)
WINDOWS_S = {"5m": 300, "15m": 900, "1h": 3600}
# minimum return samples before a moment estimate is trusted (skew/kurt are
# noisy on tiny samples); also scaled by min_coverage of the window's cells.
_MIN_SAMPLES = {"5m": 10, "15m": 30, "1h": 90}

REALIZED_MOMENTS_FEATURES = [
    "rm_skew_5m", "rm_skew_15m", "rm_skew_1h",
    "rm_kurt_5m", "rm_kurt_15m", "rm_kurt_1h",
    "rm_signed_jump_1h",
]


def _grid_moments(
    sub: pd.DataFrame, *, mid_col: str, ts_col: str, min_coverage: float,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Rolling realized moments on the 10s grid for one sorted symbol.

    Returns (grid frame with one column per feature, tick DatetimeIndex).
    """
    ts = pd.to_datetime(sub[ts_col], unit="ns", utc=True).dt.tz_localize(None)
    tick_index = pd.DatetimeIndex(ts)
    px = pd.Series(sub[mid_col].to_numpy(dtype=np.float64), index=tick_index)
    r = np.log(px.resample(f"{int(GRID_S)}s").last()).diff()

    out = pd.DataFrame(index=r.index)
    cells_per_s = 1.0 / GRID_S
    for name, win_s in WINDOWS_S.items():
        window = pd.Timedelta(seconds=win_s)
        min_periods = max(_MIN_SAMPLES[name], int(min_coverage * win_s * cells_per_s))
        roll = r.rolling(window, min_periods=min_periods)
        out[f"rm_skew_{name}"] = roll.skew()
        out[f"rm_kurt_{name}"] = roll.kurt()      # pandas kurt = excess (Fisher)

    # Signed-jump / realized-semivariance asymmetry over 1h.
    win_1h = pd.Timedelta(seconds=WINDOWS_S["1h"])
    mp_1h = max(_MIN_SAMPLES["1h"], int(min_coverage * WINDOWS_S["1h"] * cells_per_s))
    up = (r.clip(lower=0.0) ** 2).rolling(win_1h, min_periods=mp_1h).sum()
    dn = (r.clip(upper=0.0) ** 2).rolling(win_1h, min_periods=mp_1h).sum()
    tot = up + dn
    out["rm_signed_jump_1h"] = ((up - dn) / tot).where(tot > 0)
    return out, tick_index


def compute_realized_moments(
    df: pd.DataFrame,
    *,
    mid_col: str = "raw_midprice",
    ts_col: str = "timestamp_ns",
    symbol_col: str = "symbol",
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the rm_* realized-moment columns appended.

    Strictly causal (each value uses only past grid returns). Per-symbol when a
    ``symbol`` column is present; otherwise the whole frame is one series.
    """
    if ts_col not in df.columns:
        raise KeyError(f"missing timestamp column {ts_col!r}")
    if mid_col not in df.columns:
        raise KeyError(f"missing price column {mid_col!r}")

    out = df.copy()
    for col in REALIZED_MOMENTS_FEATURES:
        out[col] = np.nan

    groups = (
        out.groupby(symbol_col, sort=False).indices.items()
        if symbol_col in out.columns
        else [(None, np.arange(len(out)))]
    )
    for _, pos in groups:
        pos = np.asarray(pos)
        sub = out.iloc[pos].sort_values(ts_col)
        grid, tick_index = _grid_moments(
            sub, mid_col=mid_col, ts_col=ts_col, min_coverage=min_coverage
        )
        sorted_pos = pos[np.argsort(out[ts_col].iloc[pos].to_numpy(), kind="stable")]
        for col in REALIZED_MOMENTS_FEATURES:
            causal = grid[col].shift(1).reindex(tick_index, method="ffill")
            out.iloc[sorted_pos, out.columns.get_loc(col)] = causal.to_numpy()
    return out
