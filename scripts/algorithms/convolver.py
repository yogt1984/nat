"""
Convolver Algorithm — Pattern Kernel Similarity Scoring
========================================================

Online production algorithm that scores 100ms tick streams against
SVD-discovered pattern kernels. Produces 8 features measuring how
closely recent price action resembles breakout, turtle soup, and
trap patterns.

Architecture:
  100ms ticks → aggregate to micro-candles (candle_ticks ticks each)
    → decompose into 4 channels (body, wick, volume)
    → normalize W-candle sliding window by ATR
    → cosine similarity against each kernel in the library
    → multi-channel IC-weighted aggregation → 8 output features

Scores update once per candle completion and are held constant between
candle boundaries.

Falls back to analytical basis kernels when no SVD-discovered library
exists (allows registration and smoke tests without prior discovery run).

See docs/convolver_method.tex for full mathematical specification.

Output Features (8):
  alg_conv_breakout_bull   [-1,1]  bullish breakout similarity
  alg_conv_breakout_bear   [-1,1]  bearish breakout similarity
  alg_conv_turtle_bull     [-1,1]  bullish turtle soup similarity
  alg_conv_turtle_bear     [-1,1]  bearish turtle soup similarity
  alg_conv_trap_bull       [-1,1]  bull trap similarity
  alg_conv_trap_bear       [-1,1]  bear trap similarity
  alg_conv_best_score      [0,1]   max |score| across all 6
  alg_conv_best_pattern    {0..5}  index of pattern with max |score|
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register
from .convolver_kernels import (
    EVENT_TYPES,
    CHANNELS,
    KernelLibrary,
    compute_atr,
    cosine_similarity,
    decompose_candles,
    load_kernel_library,
    make_fallback_kernels,
    normalize_window,
    score_all_kernels,
)


@register
class Convolver(MicrostructureAlgorithm):
    """Pattern kernel convolver — cosine similarity against SVD-discovered templates."""

    def __init__(
        self,
        kernel_path: str = "models/convolver_kernels",
        candle_ticks: int = 600,
        fallback_to_analytical: bool = True,
    ):
        """
        Args:
            kernel_path: path to .npz/.json kernel library (no extension)
            candle_ticks: ticks per micro-candle (600 = 60s at 100ms)
            fallback_to_analytical: use analytical basis if no library found
        """
        # Load or create kernel library
        kp = Path(kernel_path)
        npz = kp.with_suffix(".npz") if kp.suffix != ".npz" else kp
        if npz.exists():
            self._library = load_kernel_library(kp)
        elif fallback_to_analytical:
            self._library = make_fallback_kernels(W=20)
        else:
            raise FileNotFoundError(f"No kernel library at {kernel_path}")

        self._candle_ticks = candle_ticks
        self._W = self._library.window_width
        self._atr_period = self._library.atr_period

        # Total candles needed: W (window) + atr_period (for ATR warmup)
        self._buf_len = self._W + self._atr_period

        # Warmup in ticks: enough candles for ATR + full window
        self._warmup_ticks = self._buf_len * candle_ticks

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize/reset all mutable state."""
        # Tick accumulation for current candle
        self._tick_count = 0
        self._candle_open = np.nan
        self._candle_high = -np.inf
        self._candle_low = np.inf
        self._candle_close = np.nan
        self._candle_volume = 0.0

        # Candle ring buffers
        self._O_buf: list[float] = []
        self._H_buf: list[float] = []
        self._L_buf: list[float] = []
        self._C_buf: list[float] = []
        self._V_buf: list[float] = []

        # Cached output scores (held constant within a candle)
        self._cached: dict[str, float] = {f.name: np.nan for f in self.alg_features()}

    def name(self) -> str:
        return "convolver"

    def alg_features(self) -> list[AlgorithmFeature]:
        warmup = self._warmup_ticks if hasattr(self, "_warmup_ticks") else 20400
        return [
            AlgorithmFeature(f"alg_conv_{et}", warmup=warmup,
                             description=f"Kernel similarity: {et}")
            for et in EVENT_TYPES
        ] + [
            AlgorithmFeature("alg_conv_best_score", warmup=warmup,
                             description="Max |score| across all event types"),
            AlgorithmFeature("alg_conv_best_pattern", warmup=warmup,
                             description="Index of event type with max |score|"),
        ]

    def required_columns(self) -> list[str]:
        return ["raw_midprice", "flow_volume_1s"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        mid = tick.get("raw_midprice", np.nan)
        vol = tick.get("flow_volume_1s", np.nan)

        if not np.isfinite(mid):
            return dict(self._cached)

        # Accumulate tick into current candle
        if self._tick_count == 0:
            self._candle_open = mid
            self._candle_high = mid
            self._candle_low = mid
            self._candle_volume = 0.0
        else:
            if mid > self._candle_high:
                self._candle_high = mid
            if mid < self._candle_low:
                self._candle_low = mid

        self._candle_close = mid
        if np.isfinite(vol):
            self._candle_volume += vol
        self._tick_count += 1

        # Candle complete?
        if self._tick_count >= self._candle_ticks:
            self._emit_candle()
            self._tick_count = 0

        return dict(self._cached)

    def _emit_candle(self) -> None:
        """Push completed candle to buffers and recompute scores."""
        self._O_buf.append(self._candle_open)
        self._H_buf.append(self._candle_high)
        self._L_buf.append(self._candle_low)
        self._C_buf.append(self._candle_close)
        self._V_buf.append(self._candle_volume)

        # Trim to buffer length
        if len(self._O_buf) > self._buf_len:
            self._O_buf.pop(0)
            self._H_buf.pop(0)
            self._L_buf.pop(0)
            self._C_buf.pop(0)
            self._V_buf.pop(0)

        # Need enough candles for ATR + full window
        if len(self._O_buf) < self._buf_len:
            return

        # Convert to arrays for math
        O = np.array(self._O_buf)
        H = np.array(self._H_buf)
        L = np.array(self._L_buf)
        C = np.array(self._C_buf)
        V = np.array(self._V_buf)

        # ATR at the last position
        atr_arr = compute_atr(H, L, C, self._atr_period)
        atr = atr_arr[-1]
        if not np.isfinite(atr) or atr <= 0:
            return

        # Decompose the last W candles into channels
        channels_raw = decompose_candles(
            O[-self._W:], H[-self._W:], L[-self._W:], C[-self._W:], V[-self._W:]
        )

        # Normalize each channel window
        channels_norm = {
            ch: normalize_window(data, atr) for ch, data in channels_raw.items()
        }

        # Score against all kernels
        scores = score_all_kernels(channels_norm, self._library)

        # Update cached output
        for et in EVENT_TYPES:
            self._cached[f"alg_conv_{et}"] = scores.get(et, 0.0)

        # Best score / pattern
        abs_scores = [(abs(scores.get(et, 0.0)), i) for i, et in enumerate(EVENT_TYPES)]
        best_abs, best_idx = max(abs_scores, key=lambda x: x[0])
        self._cached["alg_conv_best_score"] = best_abs
        self._cached["alg_conv_best_pattern"] = float(best_idx)

    def reset(self) -> None:
        self._reset_state()

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override: aggregate ticks to candles, then score all windows."""
        import pandas as pd

        mid = df["raw_midprice"].values.astype(np.float64)
        vol = df["flow_volume_1s"].values.astype(np.float64)
        vol = np.where(np.isfinite(vol), vol, 0.0)

        n_ticks = len(df)
        ct = self._candle_ticks

        # --- Step 1: Aggregate ticks to candles ---
        n_candles = n_ticks // ct
        if n_candles < self._buf_len:
            # Not enough data for even one full scoring window
            result = pd.DataFrame(
                {f.name: np.full(n_ticks, np.nan) for f in self.alg_features()},
                index=df.index,
            )
            return result

        # Reshape for efficient aggregation
        usable = n_candles * ct
        mid_trim = mid[:usable].reshape(n_candles, ct)
        vol_trim = vol[:usable].reshape(n_candles, ct)

        candle_O = mid_trim[:, 0]
        candle_H = np.nanmax(mid_trim, axis=1)
        candle_L = np.nanmin(mid_trim, axis=1)
        candle_C = mid_trim[:, -1]
        candle_V = np.nansum(vol_trim, axis=1)

        # --- Step 2: ATR over all candles ---
        atr_arr = compute_atr(candle_H, candle_L, candle_C, self._atr_period)

        # --- Step 3: Decompose into channels ---
        channels_full = decompose_candles(candle_O, candle_H, candle_L, candle_C, candle_V)

        # --- Step 4: Score each window position ---
        n_features = len(self.alg_features())
        feat_names = self.feature_names
        candle_scores = np.full((n_candles, n_features), np.nan)

        W = self._W
        start = self._atr_period + W - 1  # first position with full ATR + window

        for i in range(start, n_candles):
            atr = atr_arr[i]
            if not np.isfinite(atr) or atr <= 0:
                continue

            # Extract and normalize W-candle window per channel
            channels_norm = {}
            for ch in CHANNELS:
                window = channels_full[ch][i - W + 1 : i + 1]
                channels_norm[ch] = normalize_window(window, atr)

            scores = score_all_kernels(channels_norm, self._library)

            for j, et in enumerate(EVENT_TYPES):
                candle_scores[i, j] = scores.get(et, 0.0)

            abs_scores = [abs(scores.get(et, 0.0)) for et in EVENT_TYPES]
            candle_scores[i, 6] = max(abs_scores)
            candle_scores[i, 7] = float(np.argmax(abs_scores))

        # --- Step 5: Broadcast candle-level scores to tick-level ---
        tick_scores = np.full((n_ticks, n_features), np.nan)
        for ci in range(n_candles):
            t_start = ci * ct
            t_end = t_start + ct
            tick_scores[t_start:t_end, :] = candle_scores[ci, :]

        result = pd.DataFrame(tick_scores, columns=feat_names, index=df.index)

        # NaN-out warmup
        warmup = self.warmup
        if warmup > 0 and warmup < n_ticks:
            result.iloc[:warmup] = np.nan

        return result
