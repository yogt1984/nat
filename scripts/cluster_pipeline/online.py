"""
Online regime detection for NAT profiling pipeline.

Phase 7: Rolling derivative buffer and online classification.

Usage:
    from cluster_pipeline.online import DerivativeBuffer

    buf = DerivativeBuffer(columns=["feat_a", "feat_b"], temporal_windows=[5, 15, 30])
    for bar in bar_stream:
        vec = buf.update(bar)
        if vec is not None:
            # classify vec...
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cluster_pipeline.derivatives import temporal_derivatives

logger = logging.getLogger(__name__)


class DerivativeBuffer:
    """
    Fixed-size rolling buffer that computes temporal derivatives incrementally.

    Maintains a deque of max_window bars. After warmup (max_window bars pushed),
    each update() call returns the derivative vector for the most recent bar.

    The derivative vector is the last row of temporal_derivatives() applied to
    the buffer contents — equivalent to batch computation on the full history
    but with constant memory.

    Args:
        columns: Feature columns to derive (pre-selected, e.g. from select_top_features).
        temporal_windows: Window sizes for z-score, slope, rvol.
            Default [5, 15, 30].
        max_window: Buffer size. Must be >= max(temporal_windows) + 1.
            Default: max(temporal_windows) + 1 (minimum needed for valid derivatives).

    Raises:
        ValueError: if columns is empty, temporal_windows is empty,
            or max_window < max(temporal_windows) + 1.
    """

    def __init__(
        self,
        columns: List[str],
        temporal_windows: Optional[List[int]] = None,
        max_window: Optional[int] = None,
    ):
        if not columns:
            raise ValueError("columns must be non-empty")

        if temporal_windows is None:
            temporal_windows = [5, 15, 30]

        if not temporal_windows:
            raise ValueError("temporal_windows must be non-empty")

        self._columns = list(columns)
        self._temporal_windows = list(temporal_windows)

        min_required = max(temporal_windows) + 1
        if max_window is None:
            max_window = min_required

        if max_window < min_required:
            raise ValueError(
                f"max_window ({max_window}) must be >= "
                f"max(temporal_windows) + 1 = {min_required}"
            )

        self._max_window = max_window
        self._buffer: deque = deque(maxlen=max_window)
        self._n_pushed = 0

    @property
    def max_window(self) -> int:
        """Buffer capacity."""
        return self._max_window

    @property
    def columns(self) -> List[str]:
        """Feature columns being derived."""
        return list(self._columns)

    @property
    def temporal_windows(self) -> List[int]:
        """Temporal window sizes."""
        return list(self._temporal_windows)

    @property
    def n_pushed(self) -> int:
        """Total bars pushed since creation/reset."""
        return self._n_pushed

    @property
    def is_warm(self) -> bool:
        """True if buffer has enough data to produce derivatives."""
        return len(self._buffer) >= self._max_window

    def update(self, bar: pd.Series) -> Optional[np.ndarray]:
        """
        Push a bar into the buffer. Return derivative vector if warm, else None.

        The bar must contain all columns specified at construction time.

        Args:
            bar: A pandas Series representing one aggregated bar.
                Must contain all self.columns as index entries.

        Returns:
            1-D numpy array of derivative values (last row of temporal_derivatives),
            or None if the buffer hasn't accumulated enough bars yet (warmup).

        Raises:
            ValueError: if bar is missing required columns.
        """
        missing = [c for c in self._columns if c not in bar.index]
        if missing:
            raise ValueError(f"Bar missing columns: {missing[:5]}")

        # Extract only the needed columns
        values = {col: float(bar[col]) for col in self._columns}
        self._buffer.append(values)
        self._n_pushed += 1

        if not self.is_warm:
            return None

        # Build DataFrame from buffer and compute derivatives
        df = pd.DataFrame(list(self._buffer))
        derivatives = temporal_derivatives(
            df, columns=self._columns, windows=self._temporal_windows
        )

        # Return the last row as a flat array
        last_row = derivatives.iloc[-1].values.astype(np.float64)
        return last_row

    def reset(self) -> None:
        """Clear the buffer (e.g., after a gap or break detection)."""
        self._buffer.clear()
        self._n_pushed = 0

    def derivative_names(self) -> List[str]:
        """
        Return the column names of the derivative vector, in order.

        Useful for mapping the output of update() to named features.
        """
        # Build a dummy DataFrame to get column names
        dummy = pd.DataFrame(
            np.zeros((self._max_window, len(self._columns))),
            columns=self._columns,
        )
        derivatives = temporal_derivatives(
            dummy, columns=self._columns, windows=self._temporal_windows
        )
        return derivatives.columns.tolist()
