"""Behavior tests for LF1 funding-settlement (contract conformance is covered
by test_algorithm_smoke.py since it's @register-ed)."""

import numpy as np
import pandas as pd
import pytest

from algorithms.funding_settlement import (
    FundingSettlement, _settlement_proximity, _SETTLE_PERIOD_S,
)


def test_proximity_peaks_at_8h_marks():
    for hh in (0, 8, 16):
        t = pd.Timestamp(f"2026-05-01 {hh:02d}:00:00", tz="UTC").value
        assert _settlement_proximity(t, 3600.0) == pytest.approx(1.0, abs=1e-9)
    far = pd.Timestamp("2026-05-01 04:00:00", tz="UTC").value     # 4h from 00 & 08
    assert _settlement_proximity(far, 3600.0) < 0.02


def _df(premium, start="2026-05-01 07:50:00", step_s=1.0):
    n = len(premium)
    ts = pd.Timestamp(start, tz="UTC").value + (np.arange(n) * int(step_s * 1e9))
    return pd.DataFrame({"timestamp_ns": ts, "symbol": "BTC",
                         "ctx_premium_bps": np.asarray(premium, dtype=float)})


def test_fades_elevated_premium_near_mark():
    algo = FundingSettlement(z_window=100, prox_hours=1.0)
    prem = np.zeros(700)
    prem[600:] = 20.0                       # premium spikes right at the 08:00 mark (tick 600)
    out = algo.run_batch(_df(prem))
    near = out.iloc[610:650]
    assert (near["alg_settlement_premium_z"] > 0).mean() > 0.8      # elevated premium
    assert (near["alg_settlement_signal"] < 0).mean() > 0.8         # → fade (negative)


def test_signal_decays_far_from_any_mark():
    algo = FundingSettlement(z_window=100, prox_hours=0.5)
    n = 400
    ts = pd.Timestamp("2026-05-01 04:00:00", tz="UTC").value + np.arange(n) * int(1e9)
    df = pd.DataFrame({"timestamp_ns": ts, "symbol": "BTC",
                       "ctx_premium_bps": np.r_[np.zeros(200), np.full(200, 30.0)]})
    out = algo.run_batch(df)
    assert out["alg_settlement_proximity"].max() < 0.05            # 4h from any mark
    assert out["alg_settlement_signal"].abs().max() < 0.05         # proximity gate kills it


def test_nan_in_nan_out_step():
    algo = FundingSettlement()
    algo.reset()
    out = algo.step({"ctx_premium_bps": float("nan"), "timestamp_ns": float("nan")})
    assert all(np.isnan(v) for v in out.values())
