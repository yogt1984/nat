"""
Convolver Discovery — Event-Aligned SVD for Pattern Kernel Discovery
=====================================================================

Offline research tool that discovers data-driven pattern kernels:

  1. Load tick parquet → aggregate to candles
  2. Detect 6 event types (breakout, turtle soup, trap) analytically
  3. Extract W-candle windows around events, normalize by ATR
  4. SVD to discover characteristic shapes per (event_type, channel)
  5. IC gate: keep components whose loadings predict forward returns
  6. BH-FDR correction across all tests
  7. Walk-forward validation + rolling stability
  8. Save kernel library (.npz + .json) + JSON report

Usage:
  python scripts/analysis/convolver_discovery.py \\
      --data-dir data/features --symbol BTC \\
      --candle-freq 60s --window 20 --save

See docs/convolver_method.tex for full mathematical specification.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

# Ensure scripts/ is on path

from algorithms.convolver_kernels import (
    EVENT_TYPES,
    CHANNELS,
    ConvolverKernel,
    KernelLibrary,
    analytical_basis,
    basis_alignment,
    compute_atr,
    decompose_candles,
    normalize_window,
    save_kernel_library,
)


# ---------------------------------------------------------------------------
# Event detection (tex §4) — private to this module
# ---------------------------------------------------------------------------


def _rolling_max(arr: np.ndarray, N: int) -> np.ndarray:
    """Rolling max of arr over the previous N values (excludes current)."""
    out = np.full(len(arr), np.nan)
    for i in range(N, len(arr)):
        out[i] = np.max(arr[i - N : i])
    return out


def _rolling_min(arr: np.ndarray, N: int) -> np.ndarray:
    """Rolling min of arr over the previous N values (excludes current)."""
    out = np.full(len(arr), np.nan)
    for i in range(N, len(arr)):
        out[i] = np.min(arr[i - N : i])
    return out


def _rolling_median(arr: np.ndarray, N: int) -> np.ndarray:
    """Rolling median over the previous N values (excludes current)."""
    out = np.full(len(arr), np.nan)
    for i in range(N, len(arr)):
        out[i] = np.median(arr[i - N : i])
    return out


def _detect_breakout_bull(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, N: int, vol_mult: float,
) -> np.ndarray:
    """Bull breakout: C_t > max(H_{t-N:t}) AND V_t > vol_mult * median(V_{t-N:t})."""
    prev_high = _rolling_max(H, N)
    med_vol = _rolling_median(V, N)
    return (C > prev_high) & (V > vol_mult * med_vol)


def _detect_breakout_bear(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    V: np.ndarray, N: int, vol_mult: float,
) -> np.ndarray:
    """Bear breakout: C_t < min(L_{t-N:t}) AND V_t > vol_mult * median(V_{t-N:t})."""
    prev_low = _rolling_min(L, N)
    med_vol = _rolling_median(V, N)
    return (C < prev_low) & (V > vol_mult * med_vol)


def _detect_turtle_soup_bull(
    H: np.ndarray, L: np.ndarray, C: np.ndarray, N: int,
) -> np.ndarray:
    """Bull turtle soup: L_t < min(L_{t-N:t}) AND C_t > min(L_{t-N:t})."""
    prev_low = _rolling_min(L, N)
    return (L < prev_low) & (C > prev_low)


def _detect_turtle_soup_bear(
    H: np.ndarray, L: np.ndarray, C: np.ndarray, N: int,
) -> np.ndarray:
    """Bear turtle soup: H_t > max(H_{t-N:t}) AND C_t < max(H_{t-N:t})."""
    prev_high = _rolling_max(H, N)
    return (H > prev_high) & (C < prev_high)


def _detect_bull_trap(
    H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
    N: int, K: int, alpha: float, atr: np.ndarray, vol_mult: float,
) -> np.ndarray:
    """Bull trap: breakout_bull at t, then C_{t+K} < C_t - alpha * ATR_t."""
    breakout = _detect_breakout_bull(H, L, C, V, N, vol_mult)
    n = len(C)
    trap = np.zeros(n, dtype=bool)
    for i in range(n - K):
        if breakout[i] and np.isfinite(atr[i]) and atr[i] > 0:
            if C[i + K] < C[i] - alpha * atr[i]:
                trap[i] = True
    return trap


def _detect_bear_trap(
    H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
    N: int, K: int, alpha: float, atr: np.ndarray, vol_mult: float,
) -> np.ndarray:
    """Bear trap: breakout_bear at t, then C_{t+K} > C_t + alpha * ATR_t."""
    breakout = _detect_breakout_bear(H, L, C, V, N, vol_mult)
    n = len(C)
    trap = np.zeros(n, dtype=bool)
    for i in range(n - K):
        if breakout[i] and np.isfinite(atr[i]) and atr[i] > 0:
            if C[i + K] > C[i] + alpha * atr[i]:
                trap[i] = True
    return trap


def detect_all_events(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
    atr: np.ndarray,
    N: int = 20, vol_mult: float = 1.5, K: int = 3, alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Detect all 6 event types. Returns {event_type: boolean_mask}."""
    return {
        "breakout_bull": _detect_breakout_bull(H, L, C, V, N, vol_mult),
        "breakout_bear": _detect_breakout_bear(H, L, C, V, N, vol_mult),
        "turtle_bull": _detect_turtle_soup_bull(H, L, C, N),
        "turtle_bear": _detect_turtle_soup_bear(H, L, C, N),
        "trap_bull": _detect_bull_trap(H, L, C, V, N, K, alpha, atr, vol_mult),
        "trap_bear": _detect_bear_trap(H, L, C, V, N, K, alpha, atr, vol_mult),
    }


# ---------------------------------------------------------------------------
# Matrix assembly (tex §5)
# ---------------------------------------------------------------------------


def build_event_matrix(
    channel_data: np.ndarray,
    event_mask: np.ndarray,
    W: int,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack normalized W-candle windows into event-aligned matrix.

    Args:
        channel_data: 1D array of one channel (e.g. body values)
        event_mask: boolean mask of event occurrences
        W: window width
        atr: ATR array (same length as channel_data)

    Returns:
        (X, event_indices) where:
          X: shape (n_events, W) — normalized event windows
          event_indices: 1D array of event bar indices
    """
    indices = np.where(event_mask)[0]
    # Filter: need W-1 bars before event + valid ATR
    valid = indices[indices >= W - 1]
    valid = valid[np.isfinite(atr[valid]) & (atr[valid] > 0)]

    if len(valid) == 0:
        return np.empty((0, W)), np.array([], dtype=int)

    rows = []
    good_idx = []
    for idx in valid:
        window = channel_data[idx - W + 1 : idx + 1]
        if len(window) != W or np.any(~np.isfinite(window)):
            continue
        normed = normalize_window(window, atr[idx])
        rows.append(normed)
        good_idx.append(idx)

    if not rows:
        return np.empty((0, W)), np.array([], dtype=int)

    return np.stack(rows), np.array(good_idx, dtype=int)


# ---------------------------------------------------------------------------
# SVD + IC gate (tex §6-7)
# ---------------------------------------------------------------------------


def _compute_ic(loadings: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
    """Pearson IC with t-test p-value.

    Returns (ic, pvalue). Returns (0, 1) if insufficient data.
    """
    mask = np.isfinite(loadings) & np.isfinite(returns)
    n = mask.sum()
    if n < 30:
        return 0.0, 1.0
    r = float(np.corrcoef(loadings[mask], returns[mask])[0, 1])
    if not np.isfinite(r):
        return 0.0, 1.0
    # t-statistic under H0: IC=0
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-15))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    return r, float(p_val)


def _bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean mask of rejections."""
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    thresholds = np.arange(1, m + 1) * q / m

    # Find largest j where p_(j) <= j*q/m
    reject = sorted_p <= thresholds
    if not np.any(reject):
        return np.zeros(m, dtype=bool)

    j_star = np.max(np.where(reject)[0])
    result = np.zeros(m, dtype=bool)
    result[sorted_idx[: j_star + 1]] = True
    return result


def discover_kernels(
    X: np.ndarray,
    forward_returns: np.ndarray,
    event_type: str,
    channel: str,
    evr_threshold: float = 0.80,
    ic_threshold: float = 0.03,
) -> list[tuple[ConvolverKernel, float]]:
    """SVD decomposition + IC gate for one (event_type, channel) pair.

    Args:
        X: event-aligned matrix (n_events, W)
        forward_returns: returns per event (n_events,)
        event_type, channel: labels
        evr_threshold: cumulative EVR for truncation
        ic_threshold: minimum |IC| for retention

    Returns:
        List of (ConvolverKernel, pvalue) tuples for components passing IC gate.
        BH-FDR correction is applied across ALL calls externally.
    """
    n_events, W = X.shape
    if n_events < 10:
        return []

    # Thin SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T  # columns are basis shapes

    # Explained variance ratios
    s2 = S**2
    total_var = s2.sum()
    if total_var < 1e-15:
        return []
    evr = s2 / total_var
    cum_evr = np.cumsum(evr)

    # Truncation: keep components up to evr_threshold
    K = int(np.searchsorted(cum_evr, evr_threshold) + 1)
    K = min(K, len(S))

    candidates = []
    for k in range(K):
        loadings = U[:, k] * S[k]  # scaled loadings
        ic, pval = _compute_ic(loadings, forward_returns)
        if abs(ic) >= ic_threshold:
            kernel_vec = V[:, k]
            # Ensure unit norm (should be by SVD, but be safe)
            norm = np.linalg.norm(kernel_vec)
            if norm > 1e-12:
                kernel_vec = kernel_vec / norm

            ck = ConvolverKernel(
                event_type=event_type,
                channel=channel,
                component_idx=k,
                kernel=kernel_vec,
                ic=ic,
                ic_pvalue=pval,
                evr=float(evr[k]),
            )
            candidates.append((ck, pval))

    return candidates


# ---------------------------------------------------------------------------
# Walk-forward validation (tex §10)
# ---------------------------------------------------------------------------


def walk_forward_validate(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
    params: dict,
    train_ratio: float = 0.60,
    horizons: list[int] | None = None,
) -> dict:
    """Walk-forward kernel discovery: train on first portion, test on rest.

    Returns dict with IS/OOS IC per surviving kernel, IC decay ratios.
    """
    if horizons is None:
        horizons = [10, 50, 100]

    n = len(O)
    split = int(n * train_ratio)
    W = params.get("window_width", 20)
    N = params.get("range_lookback", 20)
    atr_period = params.get("atr_period", 14)
    primary_horizon = horizons[0]

    atr_full = compute_atr(H, L, C, atr_period)

    # --- Training phase ---
    O_tr, H_tr, L_tr, C_tr, V_tr = O[:split], H[:split], L[:split], C[:split], V[:split]
    atr_tr = atr_full[:split]

    events_tr = detect_all_events(O_tr, H_tr, L_tr, C_tr, V_tr, atr_tr, N=N,
                                  vol_mult=params.get("volume_multiplier", 1.5),
                                  K=params.get("trap_confirmation_lag", 3),
                                  alpha=params.get("trap_atr_threshold", 1.0))

    channels_tr = decompose_candles(O_tr, H_tr, L_tr, C_tr, V_tr)

    # Forward returns (log-return at primary horizon)
    fwd_tr = np.full(split, np.nan)
    if primary_horizon < split:
        fwd_tr[:-primary_horizon] = np.log(C_tr[primary_horizon:] / C_tr[:-primary_horizon])

    # Discover kernels on training data
    all_candidates = []
    for et in EVENT_TYPES:
        mask = events_tr[et]
        for ch in CHANNELS:
            X, idx = build_event_matrix(channels_tr[ch], mask, W, atr_tr)
            if X.shape[0] < params.get("min_events", 100):
                continue
            fwd_events = fwd_tr[idx]
            candidates = discover_kernels(
                X, fwd_events, et, ch,
                evr_threshold=params.get("evr_threshold", 0.80),
                ic_threshold=params.get("ic_threshold", 0.03),
            )
            all_candidates.extend(candidates)

    # BH-FDR across all candidates
    if not all_candidates:
        return {"n_candidates": 0, "n_surviving": 0, "kernels": []}

    pvals = np.array([pv for _, pv in all_candidates])
    fdr_mask = _bh_fdr(pvals, q=params.get("fdr_q", 0.05))

    surviving = [ck for (ck, _), keep in zip(all_candidates, fdr_mask) if keep]

    # --- Validation phase ---
    O_te, H_te, L_te, C_te, V_te = O[split:], H[split:], L[split:], C[split:], V[split:]
    atr_te = compute_atr(H_te, L_te, C_te, atr_period)

    events_te = detect_all_events(O_te, H_te, L_te, C_te, V_te, atr_te, N=N,
                                  vol_mult=params.get("volume_multiplier", 1.5),
                                  K=params.get("trap_confirmation_lag", 3),
                                  alpha=params.get("trap_atr_threshold", 1.0))

    channels_te = decompose_candles(O_te, H_te, L_te, C_te, V_te)

    fwd_te = np.full(len(O_te), np.nan)
    if primary_horizon < len(O_te):
        fwd_te[:-primary_horizon] = np.log(C_te[primary_horizon:] / C_te[:-primary_horizon])

    results = []
    for ck in surviving:
        et, ch = ck.event_type, ck.channel
        mask = events_te[et]
        X_oos, idx_oos = build_event_matrix(channels_te[ch], mask, W, atr_te)
        if X_oos.shape[0] < 20:
            oos_ic, oos_pval = 0.0, 1.0
        else:
            # Score each OOS event window against this kernel
            scores_oos = np.array([
                float(np.dot(X_oos[i], ck.kernel) / (np.linalg.norm(X_oos[i]) + 1e-12))
                for i in range(X_oos.shape[0])
            ])
            oos_ic, oos_pval = _compute_ic(scores_oos, fwd_te[idx_oos])

        decay_ratio = oos_ic / (ck.ic + 1e-12) if abs(ck.ic) > 1e-6 else 0.0
        results.append({
            "event_type": et,
            "channel": ch,
            "component": ck.component_idx,
            "is_ic": float(ck.ic),
            "is_pvalue": float(ck.ic_pvalue),
            "oos_ic": float(oos_ic),
            "oos_pvalue": float(oos_pval),
            "ic_decay_ratio": float(decay_ratio),
            "evr": float(ck.evr),
            "robust": abs(decay_ratio) >= 0.5 and np.sign(oos_ic) == np.sign(ck.ic),
        })

    return {
        "n_candidates": len(all_candidates),
        "n_surviving_fdr": len(surviving),
        "n_tested_oos": len(results),
        "primary_horizon": primary_horizon,
        "train_candles": split,
        "test_candles": n - split,
        "kernels": results,
    }


# ---------------------------------------------------------------------------
# Rolling stability (tex §10, eq 24)
# ---------------------------------------------------------------------------


def rolling_stability(
    O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray,
    params: dict,
    roll_length: int = 5000,
    stride: int = 2000,
) -> dict:
    """Rolling re-discovery to check kernel temporal stability.

    Returns mean cosine stability per event type.
    """
    n = len(O)
    W = params.get("window_width", 20)
    N = params.get("range_lookback", 20)
    atr_period = params.get("atr_period", 14)

    atr_full = compute_atr(H, L, C, atr_period)

    # Collect first right singular vectors per window
    first_svs: dict[str, list[np.ndarray]] = {et: [] for et in EVENT_TYPES}

    for start in range(0, n - roll_length + 1, stride):
        end = start + roll_length
        O_w, H_w, L_w, C_w, V_w = O[start:end], H[start:end], L[start:end], C[start:end], V[start:end]
        atr_w = atr_full[start:end]

        events_w = detect_all_events(O_w, H_w, L_w, C_w, V_w, atr_w, N=N)
        channels_w = decompose_candles(O_w, H_w, L_w, C_w, V_w)

        for et in EVENT_TYPES:
            mask = events_w[et]
            X, _ = build_event_matrix(channels_w["body"], mask, W, atr_w)
            if X.shape[0] < 30:
                continue
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            first_svs[et].append(Vt[0])  # first right singular vector

    # Compute pairwise cosine stability
    stability = {}
    for et in EVENT_TYPES:
        vecs = first_svs[et]
        if len(vecs) < 2:
            stability[et] = {"n_windows": len(vecs), "mean_stability": np.nan}
            continue
        cos_sims = []
        for i in range(len(vecs) - 1):
            cs = abs(float(np.dot(vecs[i], vecs[i + 1])))
            cos_sims.append(cs)
        stability[et] = {
            "n_windows": len(vecs),
            "mean_stability": float(np.mean(cos_sims)),
            "min_stability": float(np.min(cos_sims)),
            "stabilities": [float(x) for x in cos_sims],
        }

    return stability


# ---------------------------------------------------------------------------
# Tick-to-candle aggregation
# ---------------------------------------------------------------------------


def aggregate_ticks_to_candles(
    df: "pd.DataFrame",
    candle_ticks: int = 600,
    symbol: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate tick-level DataFrame to OHLCV arrays.

    Args:
        df: DataFrame with 'raw_midprice' and 'flow_volume_1s' columns
        candle_ticks: ticks per candle
        symbol: optional symbol filter

    Returns:
        (O, H, L, C, V) as numpy arrays
    """
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]

    mid = df["raw_midprice"].values.astype(np.float64)
    vol_col = "flow_volume_1s" if "flow_volume_1s" in df.columns else None

    if vol_col:
        vol = df[vol_col].values.astype(np.float64)
        vol = np.where(np.isfinite(vol), vol, 0.0)
    else:
        vol = np.ones(len(mid))

    n_candles = len(mid) // candle_ticks
    usable = n_candles * candle_ticks

    mid_r = mid[:usable].reshape(n_candles, candle_ticks)
    vol_r = vol[:usable].reshape(n_candles, candle_ticks)

    O = mid_r[:, 0]
    H = np.nanmax(mid_r, axis=1)
    L = np.nanmin(mid_r, axis=1)
    C = mid_r[:, -1]
    V = np.nansum(vol_r, axis=1)

    return O, H, L, C, V


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def _run_single_symbol(
    symbol: str,
    args: argparse.Namespace,
    params: dict,
) -> tuple[list[ConvolverKernel], dict]:
    """Run full discovery pipeline for one symbol.

    Returns (surviving_kernels, report_dict).
    """
    # --- Load data ---
    print(f"\n{'='*60}")

    ohlcv_file = getattr(args, "ohlcv_file", None)
    if ohlcv_file is not None:
        # Direct OHLCV parquet (from fetch_candles.py)
        import pandas as pd
        print(f"Loading pre-fetched OHLCV from {ohlcv_file}...")
        ohlcv = pd.read_parquet(ohlcv_file)
        O = ohlcv["open"].values.astype(np.float64)
        H = ohlcv["high"].values.astype(np.float64)
        L = ohlcv["low"].values.astype(np.float64)
        C = ohlcv["close"].values.astype(np.float64)
        V = ohlcv["volume"].values.astype(np.float64)
        n_candles = len(O)
        print(f"Loaded {n_candles:,} candles directly from OHLCV parquet")
    else:
        # Tick parquet (from Rust ingestor)
        print(f"Loading data from {args.data_dir} for {symbol}...")
        try:
            from cluster_pipeline.loader import load_parquet
            df = load_parquet(
                args.data_dir,
                symbols=[symbol],
                columns=["raw_midprice", "flow_volume_1s", "symbol", "timestamp_ns"],
                max_memory_mb=args.max_memory_mb,
            )
        except ImportError:
            import pandas as pd
            import pyarrow.parquet as pq
            data_path = Path(args.data_dir)
            files = sorted(data_path.glob("**/*.parquet"))
            dfs = []
            for f in files:
                t = pq.read_table(f, columns=["raw_midprice", "flow_volume_1s", "symbol"])
                d = t.to_pandas()
                d = d[d["symbol"] == symbol]
                if len(d) > 0:
                    dfs.append(d)
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        print(f"Loaded {len(df):,} ticks")

        # --- Aggregate to candles ---
        O, H, L, C, V = aggregate_ticks_to_candles(df, args.candle_ticks, symbol)
        n_candles = len(O)
        print(f"Aggregated to {n_candles:,} candles ({args.candle_ticks} ticks each)")

    if n_candles < args.window + args.atr_period + args.min_events:
        print(f"ERROR: Not enough candles for {symbol}. Need at least "
              f"{args.window + args.atr_period + args.min_events}.")
        return [], {}

    # --- ATR ---
    atr = compute_atr(H, L, C, args.atr_period)

    # --- Detect events ---
    events = detect_all_events(O, H, L, C, V, atr,
                               N=args.lookback,
                               vol_mult=params["volume_multiplier"],
                               K=params["trap_confirmation_lag"],
                               alpha=params["trap_atr_threshold"])

    print("\nEvent counts:")
    for et in EVENT_TYPES:
        count = int(events[et].sum())
        print(f"  {et:20s}: {count:6d}" + (" (< min_events)" if count < args.min_events else ""))

    # --- Channel decomposition ---
    channels = decompose_candles(O, H, L, C, V)

    # --- Forward returns at primary horizon ---
    primary_h = args.horizons[0]
    fwd_returns = np.full(n_candles, np.nan)
    if primary_h < n_candles:
        fwd_returns[:-primary_h] = np.log(C[primary_h:] / C[:-primary_h])

    # --- SVD discovery per (event_type, channel) ---
    print(f"\nRunning SVD discovery (horizon={primary_h} candles)...")
    all_candidates: list[tuple[ConvolverKernel, float]] = []
    svd_report: dict[str, dict] = {}

    for et in EVENT_TYPES:
        mask = events[et]
        n_events = int(mask.sum())
        svd_report[et] = {"n_events": n_events, "channels": {}}

        if n_events < args.min_events:
            continue

        for ch in CHANNELS:
            X, idx = build_event_matrix(channels[ch], mask, args.window, atr)
            if X.shape[0] < args.min_events:
                svd_report[et]["channels"][ch] = {"n_windows": X.shape[0], "skipped": True}
                continue

            fwd_events = fwd_returns[idx]

            # Full SVD for reporting
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            s2 = S**2
            total = s2.sum()
            evr = (s2 / total).tolist() if total > 0 else []
            stereotypy = float(S[0] / S[1]) if len(S) > 1 and S[1] > 0 else float("inf")

            candidates = discover_kernels(X, fwd_events, et, ch,
                                          args.evr_threshold, args.ic_threshold)

            # Alignment diagnostic
            basis_dict = analytical_basis(args.window)
            alignments = {}
            for ck, _ in candidates:
                alignments[f"k{ck.component_idx}"] = basis_alignment(ck.kernel, basis_dict)

            svd_report[et]["channels"][ch] = {
                "n_windows": X.shape[0],
                "singular_values_top5": [float(x) for x in S[:5]],
                "evr_top5": evr[:5],
                "stereotypy_index": stereotypy,
                "n_candidates": len(candidates),
                "candidates": [
                    {"k": ck.component_idx, "ic": float(ck.ic),
                     "pval": float(pv), "evr": float(ck.evr)}
                    for ck, pv in candidates
                ],
                "alignments": {k: {b: round(v, 3) for b, v in al.items()}
                               for k, al in alignments.items()},
            }

            all_candidates.extend(candidates)

    # --- BH-FDR correction ---
    if all_candidates:
        pvals = np.array([pv for _, pv in all_candidates])
        fdr_mask = _bh_fdr(pvals, q=args.fdr_q)
        surviving = [ck for (ck, _), keep in zip(all_candidates, fdr_mask) if keep]
    else:
        surviving = []

    print(f"\nSVD candidates: {len(all_candidates)}")
    print(f"Surviving FDR (q={args.fdr_q}): {len(surviving)}")

    if surviving:
        print("\nSurviving kernels:")
        print(f"  {'Event Type':20s} {'Channel':12s} {'k':>3s} {'IC':>8s} {'p-val':>8s} {'EVR':>6s}")
        print(f"  {'-'*20} {'-'*12} {'-'*3} {'-'*8} {'-'*8} {'-'*6}")
        for ck in surviving:
            print(f"  {ck.event_type:20s} {ck.channel:12s} {ck.component_idx:3d} "
                  f"{ck.ic:+8.4f} {ck.ic_pvalue:8.4f} {ck.evr:6.3f}")

    # --- Walk-forward validation ---
    print("\nRunning walk-forward validation...")
    wf_result = walk_forward_validate(O, H, L, C, V, params,
                                       train_ratio=0.60, horizons=args.horizons)
    n_robust = sum(1 for k in wf_result.get("kernels", []) if k.get("robust"))
    print(f"Walk-forward: {wf_result.get('n_surviving_fdr', 0)} IS → "
          f"{wf_result.get('n_tested_oos', 0)} OOS → {n_robust} robust")

    # --- Rolling stability ---
    print("\nRunning rolling stability check...")
    stab = rolling_stability(O, H, L, C, V, params)
    for et, s in stab.items():
        ms = s.get("mean_stability", float("nan"))
        nw = s.get("n_windows", 0)
        if nw >= 2:
            status = "STABLE" if ms >= 0.70 else "UNSTABLE"
            print(f"  {et:20s}: rho={ms:.3f} ({nw} windows) [{status}]")

    # --- Build and save kernel library ---
    if args.save and surviving:
        # IC-weighted channel weights
        ch_ic_sum: dict[str, float] = {c: 0.0 for c in CHANNELS}
        for ck in surviving:
            ch_ic_sum[ck.channel] += abs(ck.ic)
        total_ic = sum(ch_ic_sum.values())
        if total_ic > 0:
            ch_weights = {c: v / total_ic for c, v in ch_ic_sum.items()}
        else:
            ch_weights = {c: 0.25 for c in CHANNELS}

        library = KernelLibrary(
            kernels=surviving,
            window_width=args.window,
            atr_period=args.atr_period,
            channel_weights=ch_weights,
            discovery_date=datetime.now(timezone.utc).isoformat(),
            train_end=f"candle_{n_candles}",
            horizons=args.horizons,
        )

        out_path = Path(args.output_dir) / f"convolver_kernels_{symbol}"
        save_kernel_library(library, out_path)
        print(f"\nKernel library saved to {out_path}.npz + .json")

    # --- Build report ---
    report = {
        "title": "Convolver Discovery: Event-Aligned SVD",
        "generated": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "n_ticks": len(df) if ohlcv_file is None else n_candles,
        "n_candles": n_candles,
        "candle_ticks": args.candle_ticks,
        "params": params,
        "horizons": args.horizons,
        "event_counts": {et: int(events[et].sum()) for et in EVENT_TYPES},
        "svd_report": svd_report,
        "n_candidates": len(all_candidates),
        "n_surviving_fdr": len(surviving),
        "surviving_kernels": [
            {"event_type": ck.event_type, "channel": ck.channel,
             "k": ck.component_idx, "ic": float(ck.ic),
             "pval": float(ck.ic_pvalue), "evr": float(ck.evr)}
            for ck in surviving
        ],
        "walk_forward": wf_result,
        "rolling_stability": stab,
    }

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"convolver_discovery_{symbol}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    return surviving, report


def _cross_symbol_similarity(
    all_kernels: dict[str, list[ConvolverKernel]],
) -> None:
    """Print cross-symbol cosine similarity matrix for shared kernel types."""
    symbols = sorted(all_kernels.keys())
    if len(symbols) < 2:
        return

    print(f"\n{'='*60}")
    print("Cross-Symbol Kernel Similarity")
    print(f"{'='*60}")

    # Group kernels by (event_type, channel, component_idx)
    by_key: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for sym, kernels in all_kernels.items():
        for ck in kernels:
            key = (ck.event_type, ck.channel, ck.component_idx)
            by_key.setdefault(key, {})[sym] = ck.kernel

    # Compute pairwise similarity for shared keys
    shared = {k: v for k, v in by_key.items() if len(v) >= 2}
    if not shared:
        print("  No shared kernel types across symbols.")
        return

    print(f"\n  {'Kernel':40s}", end="")
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1:]:
            print(f"  {s1}-{s2:>5s}", end="")
    print()
    print(f"  {'-'*40}", end="")
    for i in range(len(symbols)):
        for _ in symbols[i + 1:]:
            print(f"  {'-'*9}", end="")
    print()

    similarities = []
    for key in sorted(shared.keys()):
        kernels_map = shared[key]
        et, ch, k = key
        label = f"{et}/{ch}/k{k}"
        print(f"  {label:40s}", end="")
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                if s1 in kernels_map and s2 in kernels_map:
                    cos_sim = abs(float(np.dot(kernels_map[s1], kernels_map[s2])))
                    similarities.append(cos_sim)
                    print(f"  {cos_sim:9.3f}", end="")
                else:
                    print(f"  {'n/a':>9s}", end="")
        print()

    if similarities:
        mean_sim = float(np.mean(similarities))
        print(f"\n  Mean cross-symbol similarity: {mean_sim:.3f}")
        if mean_sim >= 0.7:
            print("  -> HIGH universality: pooled discovery justified")
        elif mean_sim >= 0.4:
            print("  -> MODERATE universality: per-symbol kernels recommended")
        else:
            print("  -> LOW universality: patterns are symbol-specific")


def main():
    parser = argparse.ArgumentParser(
        description="Convolver Discovery: Event-Aligned SVD for Pattern Kernel Discovery"
    )
    parser.add_argument("--data-dir", default="data/features",
                        help="Path to tick parquet directory")
    parser.add_argument("--ohlcv-file", default=None,
                        help="Pre-fetched OHLCV parquet (bypasses tick aggregation)")
    parser.add_argument("--symbol", default=None,
                        help="Single symbol to analyze (backward compat)")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (e.g. BTC,ETH,SOL)")
    parser.add_argument("--candle-ticks", type=int, default=600,
                        help="Ticks per candle (600 = 60s at 100ms)")
    parser.add_argument("--window", type=int, default=20,
                        help="W: candles in pattern window")
    parser.add_argument("--lookback", type=int, default=20,
                        help="N: range lookback for event detection")
    parser.add_argument("--atr-period", type=int, default=14,
                        help="ATR period")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50, 100],
                        help="Forward-return horizons (in candles)")
    parser.add_argument("--evr-threshold", type=float, default=0.80,
                        help="Cumulative EVR for SVD truncation")
    parser.add_argument("--ic-threshold", type=float, default=0.03,
                        help="Minimum |IC| for kernel retention")
    parser.add_argument("--fdr-q", type=float, default=0.05,
                        help="BH-FDR significance level")
    parser.add_argument("--min-events", type=int, default=100,
                        help="Minimum events per type for SVD")
    parser.add_argument("--save", action="store_true",
                        help="Save kernel library to models/convolver_kernels")
    parser.add_argument("--output-dir", default="models",
                        help="Directory for kernel library output")
    parser.add_argument("--max-memory-mb", type=float, default=2000.0,
                        help="Max memory for parquet loading")
    args = parser.parse_args()

    # Resolve symbol list
    if args.symbols:
        symbol_list = [s.strip() for s in args.symbols.split(",")]
    elif args.symbol:
        symbol_list = [args.symbol]
    else:
        symbol_list = ["BTC"]

    params = {
        "window_width": args.window,
        "range_lookback": args.lookback,
        "atr_period": args.atr_period,
        "volume_multiplier": 1.5,
        "trap_confirmation_lag": 3,
        "trap_atr_threshold": 1.0,
        "evr_threshold": args.evr_threshold,
        "ic_threshold": args.ic_threshold,
        "fdr_q": args.fdr_q,
        "min_events": args.min_events,
    }

    all_kernels: dict[str, list[ConvolverKernel]] = {}
    for symbol in symbol_list:
        surviving, _ = _run_single_symbol(symbol, args, params)
        all_kernels[symbol] = surviving

    # Cross-symbol comparison
    if len(symbol_list) > 1:
        _cross_symbol_similarity(all_kernels)


if __name__ == "__main__":
    main()
