"""Planted (Level-1) test for GAP-01 `vwap_reversion` — RED before the algo exists.

The ungated baseline: fade the z-scored deviation from micro-VWAP. It is the control
that GAP-03's VPIN gate is measured against (docs/GAP__26_7_26.md/01_vwap_reversion.md).

Contract encoded here:
  - Recovery: fading a mean-reverting (OU) deviation is PROFITABLE (positive IC).
  - Polarity trap: fading a pure TREND LOSES (the fade is wrong when price continues).
  - NaN discipline + interface conformance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from algorithms.vwap_reversion import VwapReversion


def _ou_deviation(seed: int, *, n: int = 6000, phi: float = 0.9, gamma: float = 0.5):
    """Mean-reverting AR(1) deviation; forward return reverts it (fading should win)."""
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        d[t] = phi * d[t - 1] + shocks[t]
    fwd = -gamma * d + 0.3 * rng.standard_normal(n)  # high deviation -> reverts down
    return pd.DataFrame({"flow_vwap_deviation": d}), fwd


def _trend_deviation(seed: int, *, n: int = 6000, slope: float = 0.02, drift: float = 0.5):
    """Monotonically growing deviation; price CONTINUES (fading should lose)."""
    rng = np.random.default_rng(seed)
    d = slope * np.arange(n) + 0.1 * rng.standard_normal(n)
    fwd = drift * slope + 0.1 * rng.standard_normal(n)  # positive drift — trend persists
    return pd.DataFrame({"flow_vwap_deviation": d}), fwd


def _ic_and_pnl(df, fwd):
    algo = VwapReversion(z_window=96, k_entry=0.5)
    sig = algo.run_batch(df)["alg_vwaprev_signal"].to_numpy()
    ok = np.isfinite(sig) & np.isfinite(fwd)
    active = ok & (sig != 0.0)
    # IC undefined on a constant signal (e.g. fading a pure trend => all -1).
    if active.sum() > 50 and np.std(sig[active]) > 1e-12:
        ic = spearmanr(sig[active], fwd[active]).statistic
    else:
        ic = 0.0
    pnl = float(np.nanmean(np.where(ok, sig * fwd, np.nan)))
    return ic, pnl


class TestContract:
    def test_name_and_required_columns(self):
        a = VwapReversion()
        assert a.name() == "vwap_reversion"
        assert a.required_columns() == ["flow_vwap_deviation"]

    def test_step_keys(self):
        a = VwapReversion()
        keys = {f.name for f in a.alg_features()}
        assert keys == {"alg_vwaprev_z", "alg_vwaprev_signal"}
        out = a.step({"flow_vwap_deviation": 0.5})
        assert set(out.keys()) == keys

    def test_nan_input_yields_all_nan(self):
        a = VwapReversion()
        out = a.step({"flow_vwap_deviation": np.nan})
        assert all(np.isnan(v) for v in out.values())


class TestPolarity:
    def test_fading_reverting_deviation_wins(self):
        df, fwd = _ou_deviation(seed=11)
        ic, pnl = _ic_and_pnl(df, fwd)
        assert ic > 0.15, f"fading an OU deviation should be predictive (IC={ic:.3f})"
        assert pnl > 0, f"fading a reverting deviation should be profitable (PnL={pnl:.4f})"

    def test_fading_a_trend_loses(self):
        # Polarity trap: on a persistent trend the fade is systematically wrong.
        df, fwd = _trend_deviation(seed=11)
        _, pnl = _ic_and_pnl(df, fwd)
        assert pnl < 0, f"fading a trend must lose (PnL={pnl:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
