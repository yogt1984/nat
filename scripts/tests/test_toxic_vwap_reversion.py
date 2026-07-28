"""Planted (Level-1) test for GAP-03 `toxic_vwap_reversion` — RED before the algo exists.

Thesis (docs/GAP__26_7_26.md/03): a deviation from micro-VWAP has two generators —
liquidity noise (reverts → fade it, profit) and informed flow (continues → stand aside).
VPIN is the classifier: fading only in LOW-toxicity states should lift the reversion
signal's IC and net PnL. A VPIN that does NOT separate the two generators must NOT help
(null control — the local PROC-12 discipline).

Written test-first: `algorithms.toxic_vwap_reversion` does not exist yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from algorithms.toxic_vwap_reversion import ToxicVwapReversion


# --------------------------------------------------------------------------- #
# Two-generator synthetic process
# --------------------------------------------------------------------------- #
def _two_generator(seed: int, *, n: int = 6000, vpin_correlated: bool = True):
    """Build (df, forward_return).

    deviation d ~ N(0,1). ~30% of bars are 'informed' (deviation CONTINUES next bar),
    the rest are 'noise' (deviation REVERTS). When vpin_correlated, informed bars carry
    systematically higher VPIN (top of the range) so a percentile gate can separate them;
    in the null case VPIN is independent of the tag.
    """
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(n)
    informed = rng.random(n) < 0.30

    eps = 0.3 * rng.standard_normal(n)
    fwd = np.where(informed, +1.0 * d, -0.5 * d) + eps  # continue vs revert

    if vpin_correlated:
        vpin = np.where(informed, rng.uniform(0.60, 1.0, n), rng.uniform(0.0, 0.60, n))
    else:
        vpin = rng.uniform(0.0, 1.0, n)  # independent of the generator

    spread = rng.uniform(1.0, 3.0, n)  # calm — spread gate never fires

    df = pd.DataFrame({
        "flow_vwap_deviation": d,
        "toxic_vpin_50": vpin,
        "raw_spread_bps": spread,
    })
    return df, fwd


def _signal_ic_and_pnl(df, fwd, **kwargs):
    """Run the algo over df; return (IC over active bars, mean gross PnL over all bars)."""
    algo = ToxicVwapReversion(w_z=96, w_p=96, k_entry=0.5,
                              size_by_toxicity=False, **kwargs)
    out = algo.run_batch(df)
    sig = out["alg_txvr_signal"].to_numpy()
    valid = np.isfinite(sig) & np.isfinite(fwd)
    active = valid & (sig != 0.0)
    ic = spearmanr(sig[active], fwd[active]).statistic if active.sum() > 50 else 0.0
    pnl = float(np.nanmean(np.where(valid, sig * fwd, np.nan)))
    return ic, pnl


# --------------------------------------------------------------------------- #
# Contract conformance
# --------------------------------------------------------------------------- #
class TestContract:
    def test_name_and_required_columns(self):
        a = ToxicVwapReversion()
        assert a.name() == "toxic_vwap_reversion"
        assert a.required_columns() == ["flow_vwap_deviation", "toxic_vpin_50", "raw_spread_bps"]

    def test_step_returns_exactly_declared_features(self):
        a = ToxicVwapReversion()
        keys = {f.name for f in a.alg_features()}
        assert keys == {"alg_txvr_z", "alg_txvr_gate", "alg_txvr_signal"}
        out = a.step({"flow_vwap_deviation": 0.1, "toxic_vpin_50": 0.2, "raw_spread_bps": 1.5})
        assert set(out.keys()) == keys

    def test_nan_input_yields_all_nan(self):
        a = ToxicVwapReversion()
        out = a.step({"flow_vwap_deviation": np.nan, "toxic_vpin_50": 0.2, "raw_spread_bps": 1.5})
        assert all(np.isnan(v) for v in out.values())

    def test_gate_closes_on_high_toxicity(self):
        # Warm up on low-VPIN history, then a high-VPIN tick must close the gate.
        a = ToxicVwapReversion(w_p=50, theta_pct=0.70)
        for _ in range(60):
            a.step({"flow_vwap_deviation": 0.0, "toxic_vpin_50": 0.1, "raw_spread_bps": 1.5})
        hot = a.step({"flow_vwap_deviation": 3.0, "toxic_vpin_50": 0.99, "raw_spread_bps": 1.5})
        assert hot["alg_txvr_gate"] == 0.0
        assert hot["alg_txvr_signal"] == 0.0  # gate closed → stand aside


# --------------------------------------------------------------------------- #
# Planted thesis: the gate lifts IC/PnL only when VPIN separates the generators
# --------------------------------------------------------------------------- #
class TestVpinGateThesis:
    def test_gate_lifts_ic_and_pnl_when_vpin_informative(self):
        df, fwd = _two_generator(seed=7, vpin_correlated=True)
        ic_gated, pnl_gated = _signal_ic_and_pnl(df, fwd, theta_pct=0.70, spread_pct_max=1.0)
        ic_ungated, pnl_ungated = _signal_ic_and_pnl(df, fwd, theta_pct=1.0, spread_pct_max=1.0)

        assert ic_gated > 0.05, f"gated signal should be predictive (IC={ic_gated:.3f})"
        assert ic_gated > ic_ungated + 0.03, (
            f"VPIN gate must lift IC: gated={ic_gated:.3f} vs ungated={ic_ungated:.3f}")
        assert pnl_gated > pnl_ungated, (
            f"gate must improve gross PnL: gated={pnl_gated:.4f} vs ungated={pnl_ungated:.4f}")

    def test_gate_neutral_when_vpin_uninformative(self):
        # Null control: VPIN independent of the generator → the gate cannot help.
        df, fwd = _two_generator(seed=7, vpin_correlated=False)
        ic_gated, _ = _signal_ic_and_pnl(df, fwd, theta_pct=0.70, spread_pct_max=1.0)
        ic_ungated, _ = _signal_ic_and_pnl(df, fwd, theta_pct=1.0, spread_pct_max=1.0)
        assert ic_gated <= ic_ungated + 0.02, (
            f"uninformative VPIN must not lift IC: gated={ic_gated:.3f} vs ungated={ic_ungated:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
