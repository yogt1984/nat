"""Bridge between agent-discovered RegisteredSignals and the algorithm framework.

Reads validated signals from the agent registry (SQLite) and wraps each
as a MicrostructureAlgorithm so the tournament engine can paper-trade them
alongside hand-coded algorithms.

Usage:
    from algorithms.signal_adapter import load_signal_algorithms
    signal_algos = load_signal_algorithms()  # list[SignalAlgorithm]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AlgorithmFeature, MicrostructureAlgorithm

log = logging.getLogger("nat.algorithms.signal_adapter")

# Agent names in the state store
_AGENT_NAMES = ["microstructure", "medium_frequency", "macro"]

# Default DB path (same as agent system)
_DEFAULT_DB = Path("data/nat.db")


def _parse_regime_gate(gate_str: str) -> Optional[dict]:
    """Parse 'feature < PNN' or 'feature > PNN' into a dict.

    Returns: {"feature": str, "op": "<"|">", "percentile": float} or None.
    """
    if not gate_str:
        return None
    m = re.match(r"(\w+)\s*([<>])\s*P(\d+(?:\.\d+)?)", gate_str.strip())
    if not m:
        return None
    return {
        "feature": m.group(1),
        "op": m.group(2),
        "percentile": float(m.group(3)),
    }


class SignalAlgorithm(MicrostructureAlgorithm):
    """Algorithm auto-generated from a RegisteredSignal spec.

    Emits two features:
      - alg_sig_{id}_raw   : the raw signal feature value
      - alg_sig_{id}_gated : signal masked by regime gate (NaN when gated out)
    """

    def __init__(self, signal_spec: dict):
        self.spec = signal_spec
        hyp_id = signal_spec.get("hypothesis_id", "unknown")
        self._id = hyp_id[:8] if len(hyp_id) >= 8 else hyp_id
        self._signal_name = f"sig_{self._id}"
        self._features_in = signal_spec.get("features", [])
        self._primary_feature = self._features_in[0] if self._features_in else "unknown"
        self._gate_spec = _parse_regime_gate(signal_spec.get("regime_gate") or "")
        self._horizon_s = signal_spec.get("horizon_s", 5.0)
        self._expected_ic = signal_spec.get("expected_ic", 0.0)
        self._source_agent = signal_spec.get("_source_agent", "unknown")

        # Percentile threshold — computed from training data in run_batch
        self._gate_threshold: Optional[float] = None

    def name(self) -> str:
        return self._signal_name

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature(
                name=f"alg_{self._signal_name}_raw",
                warmup=100,
                description=f"Raw signal from {self._primary_feature}",
            ),
            AlgorithmFeature(
                name=f"alg_{self._signal_name}_gated",
                warmup=100,
                description=f"Regime-gated signal from {self._primary_feature}",
            ),
        ]

    def required_columns(self) -> list[str]:
        cols = list(self._features_in)
        if self._gate_spec:
            gate_feat = self._gate_spec["feature"]
            if gate_feat not in cols:
                cols.append(gate_feat)
        return cols

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        raw_name = f"alg_{self._signal_name}_raw"
        gated_name = f"alg_{self._signal_name}_gated"

        val = tick.get(self._primary_feature, np.nan)
        if not np.isfinite(val):
            return {raw_name: np.nan, gated_name: np.nan}

        gated = val
        if self._gate_spec and self._gate_threshold is not None:
            gate_val = tick.get(self._gate_spec["feature"], np.nan)
            if np.isfinite(gate_val):
                if self._gate_spec["op"] == "<":
                    if gate_val >= self._gate_threshold:
                        gated = np.nan
                else:
                    if gate_val <= self._gate_threshold:
                        gated = np.nan
            else:
                gated = np.nan

        return {raw_name: val, gated_name: gated}

    def reset(self) -> None:
        self._gate_threshold = None

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized path — compute regime gate threshold from data."""
        raw_name = f"alg_{self._signal_name}_raw"
        gated_name = f"alg_{self._signal_name}_gated"

        if self._primary_feature not in df.columns:
            n = len(df)
            return pd.DataFrame(
                {raw_name: np.full(n, np.nan), gated_name: np.full(n, np.nan)},
                index=df.index,
            )

        raw = df[self._primary_feature].values.astype(float).copy()

        # Apply regime gate
        gated = raw.copy()
        if self._gate_spec and self._gate_spec["feature"] in df.columns:
            gate_vals = df[self._gate_spec["feature"]].values.astype(float)
            finite_mask = np.isfinite(gate_vals)
            pct = self._gate_spec["percentile"]
            if finite_mask.sum() > 20:
                threshold = np.nanpercentile(gate_vals[finite_mask], pct)
                if self._gate_spec["op"] == "<":
                    mask = gate_vals >= threshold
                else:
                    mask = gate_vals <= threshold
                gated[mask] = np.nan
                gated[~finite_mask] = np.nan

        result = pd.DataFrame(
            {raw_name: raw, gated_name: gated},
            index=df.index,
        )

        # NaN-out warmup
        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan

        return result

    @property
    def polarity(self) -> str:
        """Infer signal polarity from expected IC sign."""
        return "high_long" if self._expected_ic >= 0 else "low_long"

    @property
    def source_agent(self) -> str:
        return self._source_agent

    @property
    def signal_id(self) -> str:
        return self.spec.get("hypothesis_id", "")


def load_signal_algorithms(db_path: Optional[Path] = None) -> list[SignalAlgorithm]:
    """Load all non-retired signals from agent registries and wrap as algorithms.

    Returns a list of SignalAlgorithm instances ready for paper trading.
    """
    from data.state import StateStore

    db = db_path or _DEFAULT_DB
    if not db.exists():
        log.info("No agent state DB at %s — no signal algorithms to load", db)
        return []

    store = StateStore(db)
    algos = []
    seen_ids = set()

    for agent_name in _AGENT_NAMES:
        try:
            registry = store.load_registry(agent_name)
        except Exception as e:
            log.warning("Failed to load registry for %s: %s", agent_name, e)
            continue

        for sig in registry:
            status = sig.get("status", "validated")
            if status == "retired":
                continue

            hyp_id = sig.get("hypothesis_id", "")
            if not hyp_id or hyp_id in seen_ids:
                continue
            seen_ids.add(hyp_id)

            features = sig.get("features", [])
            if not features:
                continue

            sig["_source_agent"] = agent_name
            try:
                algo = SignalAlgorithm(sig)
                algos.append(algo)
            except Exception as e:
                log.warning("Failed to create SignalAlgorithm for %s: %s", hyp_id, e)

    log.info("Loaded %d signal algorithms from %d agent registries",
             len(algos), len(_AGENT_NAMES))
    return algos
