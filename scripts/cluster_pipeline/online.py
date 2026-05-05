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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Task 7.2: Online Classifier with Drift Detection
# ---------------------------------------------------------------------------


@dataclass
class StateEstimate:
    """Classification result for a single derivative vector."""

    macro_regime: int
    macro_confidence: float
    micro_state: int  # global micro-state ID
    micro_confidence: float
    composite_label: str  # e.g. "R0_S2"
    time_in_state: int  # consecutive bars in current micro state
    likely_next_state: int  # most probable successor state
    transition_prob: float  # probability of that transition
    all_probabilities: Dict[int, float]  # global state → probability
    drift_warning: bool  # True if rolling log-likelihood below training baseline
    rolling_log_likelihood: float  # rolling average GMM log-likelihood


@dataclass
class ClassifierConfig:
    """Configuration for building an OnlineClassifier from profiling artifacts."""

    # Macro PCA projection
    macro_pca_components: np.ndarray  # (n_components, n_features)
    macro_pca_mean: np.ndarray  # (n_features,)
    macro_pca_std: np.ndarray  # (n_features,)

    # Macro GMM (fitted sklearn object)
    macro_gmm: Any  # GaussianMixture

    # Per-regime micro PCA + GMM
    micro_pca_components: Dict[int, np.ndarray]  # regime → (n_comp, n_feat)
    micro_pca_mean: Dict[int, np.ndarray]  # regime → (n_feat,)
    micro_pca_std: Dict[int, np.ndarray]  # regime → (n_feat,)
    micro_gmm: Dict[int, Any]  # regime → GaussianMixture

    # Label map: global_micro_id → (regime_id, local_micro_id)
    label_map: Dict[int, tuple]

    # Transition model (matrix indexed by global micro state)
    transition_matrix: np.ndarray  # (n_states, n_states)
    state_ids: List[int]  # ordered global state IDs corresponding to matrix rows

    # Training log-likelihood stats for drift detection
    training_ll_p10: float  # 10th percentile of per-sample log-likelihood
    training_ll_p50: float  # 50th percentile


class OnlineClassifier:
    """
    Classify derivative vectors into hierarchical states with drift detection.

    Uses the macro GMM to determine regime, then the regime-specific micro GMM
    to determine micro state. Tracks rolling log-likelihood for drift detection.

    Args:
        config: ClassifierConfig with all trained model parameters.
        drift_window: Number of recent log-likelihoods to average for drift.
        drift_consecutive: Bars below threshold before drift_warning fires.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        drift_window: int = 50,
        drift_consecutive: int = 20,
    ):
        self._config = config
        self._drift_window = drift_window
        self._drift_consecutive = drift_consecutive

        # State tracking
        self._current_state: Optional[int] = None
        self._time_in_state: int = 0
        self._ll_history: deque = deque(maxlen=drift_window)
        self._bars_below_threshold: int = 0

        # Build reverse label map: (regime, local) → global
        self._reverse_map: Dict[tuple, int] = {}
        for global_id, (regime, local) in config.label_map.items():
            self._reverse_map[(regime, local)] = global_id

        # State ID to index in transition matrix
        self._state_to_idx = {s: i for i, s in enumerate(config.state_ids)}

    def classify(self, derivative_vector: np.ndarray) -> StateEstimate:
        """
        Classify a derivative vector into hierarchical state.

        Args:
            derivative_vector: 1-D array from DerivativeBuffer.update().

        Returns:
            StateEstimate with regime, state, confidence, and drift info.

        Raises:
            ValueError: if vector has wrong dimensionality.
        """
        vec = np.asarray(derivative_vector, dtype=np.float64)
        if vec.ndim != 1:
            raise ValueError(f"Expected 1-D vector, got shape {vec.shape}")

        cfg = self._config

        # ----- Step 1: Macro classification -----
        # Standardize + PCA project
        vec_std = (vec - cfg.macro_pca_mean) / np.where(
            cfg.macro_pca_std > 1e-12, cfg.macro_pca_std, 1.0
        )
        macro_projected = vec_std @ cfg.macro_pca_components.T  # (n_macro_components,)

        # GMM predict
        macro_probs = cfg.macro_gmm.predict_proba(
            macro_projected.reshape(1, -1)
        )[0]
        macro_regime = int(np.argmax(macro_probs))
        macro_confidence = float(macro_probs[macro_regime])

        # Log-likelihood for drift detection
        macro_ll = float(
            cfg.macro_gmm.score_samples(macro_projected.reshape(1, -1))[0]
        )

        # ----- Step 2: Micro classification (within predicted regime) -----
        if macro_regime in cfg.micro_gmm:
            micro_mean = cfg.micro_pca_mean[macro_regime]
            micro_std = cfg.micro_pca_std[macro_regime]
            micro_components = cfg.micro_pca_components[macro_regime]
            micro_gmm = cfg.micro_gmm[macro_regime]

            vec_micro_std = (vec - micro_mean) / np.where(
                micro_std > 1e-12, micro_std, 1.0
            )
            micro_projected = vec_micro_std @ micro_components.T

            micro_probs = micro_gmm.predict_proba(
                micro_projected.reshape(1, -1)
            )[0]
            local_state = int(np.argmax(micro_probs))
            micro_confidence = float(micro_probs[local_state])

            # Map to global state ID
            global_state = self._reverse_map.get(
                (macro_regime, local_state), 0
            )
        else:
            # Regime has no micro model (single state)
            local_state = 0
            micro_confidence = 1.0
            global_state = self._reverse_map.get((macro_regime, 0), 0)

        # ----- Step 3: Time-in-state tracking -----
        if global_state == self._current_state:
            self._time_in_state += 1
        else:
            self._current_state = global_state
            self._time_in_state = 1

        # ----- Step 4: Transition prediction -----
        likely_next, trans_prob = self._predict_next(global_state)

        # ----- Step 5: All probabilities (global state space) -----
        all_probs = self._compute_all_probabilities(
            macro_probs, macro_regime, cfg
        )

        # ----- Step 6: Drift detection -----
        self._ll_history.append(macro_ll)
        rolling_ll = float(np.mean(list(self._ll_history)))

        if rolling_ll < cfg.training_ll_p10:
            self._bars_below_threshold += 1
        else:
            self._bars_below_threshold = 0

        drift_warning = self._bars_below_threshold >= self._drift_consecutive

        if self._bars_below_threshold >= 100:
            logger.warning("DRIFT: re-profiling recommended")

        # ----- Build composite label -----
        composite = f"R{macro_regime}_S{local_state}"

        return StateEstimate(
            macro_regime=macro_regime,
            macro_confidence=macro_confidence,
            micro_state=global_state,
            micro_confidence=micro_confidence,
            composite_label=composite,
            time_in_state=self._time_in_state,
            likely_next_state=likely_next,
            transition_prob=trans_prob,
            all_probabilities=all_probs,
            drift_warning=drift_warning,
            rolling_log_likelihood=rolling_ll,
        )

    @property
    def drift_detected(self) -> bool:
        """True if rolling log-likelihood below training p10 for consecutive bars."""
        return self._bars_below_threshold >= self._drift_consecutive

    @property
    def bars_below_threshold(self) -> int:
        """Number of consecutive bars with rolling LL below training p10."""
        return self._bars_below_threshold

    def reset_drift(self) -> None:
        """Reset drift counter (e.g., after re-profiling)."""
        self._bars_below_threshold = 0
        self._ll_history.clear()

    def _predict_next(self, current_state: int) -> tuple:
        """Predict most likely next state from transition matrix."""
        if current_state not in self._state_to_idx:
            return (current_state, 0.0)

        idx = self._state_to_idx[current_state]
        row = self._config.transition_matrix[idx, :]

        # Exclude self-transition for "next state"
        row_no_self = row.copy()
        row_no_self[idx] = 0.0
        total = row_no_self.sum()

        if total < 1e-12:
            return (current_state, 0.0)

        best_idx = int(np.argmax(row_no_self))
        best_state = self._config.state_ids[best_idx]
        best_prob = float(row_no_self[best_idx])

        return (best_state, best_prob)

    def _compute_all_probabilities(
        self, macro_probs: np.ndarray, current_regime: int, cfg: ClassifierConfig
    ) -> Dict[int, float]:
        """Compute probability for each global state."""
        all_probs: Dict[int, float] = {}

        for global_id, (regime, local) in cfg.label_map.items():
            # P(global_state) = P(regime) * P(local | regime)
            regime_prob = float(macro_probs[regime]) if regime < len(macro_probs) else 0.0

            if regime in cfg.micro_gmm and regime == current_regime:
                # We already computed micro probs for current regime
                # For other regimes, approximate with uniform
                # This is a simplification — full inference would require
                # projecting into each regime's PCA space
                micro_gmm = cfg.micro_gmm[regime]
                n_micro = micro_gmm.n_components
                # For current regime, we have the actual micro probs
                all_probs[global_id] = regime_prob / n_micro  # placeholder
            else:
                n_micro = sum(1 for _, (r, _) in cfg.label_map.items() if r == regime)
                if n_micro > 0:
                    all_probs[global_id] = regime_prob / n_micro
                else:
                    all_probs[global_id] = 0.0

        # Normalize
        total = sum(all_probs.values())
        if total > 1e-12:
            for k in all_probs:
                all_probs[k] /= total

        return all_probs
