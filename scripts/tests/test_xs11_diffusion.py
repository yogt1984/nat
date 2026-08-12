"""Planted tests for the XS-11 diffusion study (pre-registered driver).

The planted scenario is family 6's own prediction: persistent per-symbol drift
in the WIDE (illiquid) tercile — trailing return then ranks forward return by
construction — against pure noise in the tight/mid terciles. The estimator must
find the effect exactly where it was planted, nowhere else, and the verdict
machinery must refuse to promote what it cannot power.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from exploration.xs11_diffusion_study import (
    ABSENT_IC_BAND,
    bh_fdr,
    cell_ics,
    liquidity_terciles,
    momentum_signal,
    run_study,
    summarize_cell,
    tradability,
    verdict,
)

RNG = np.random.RandomState(7)


def _panel(n_hours=2200, n_syms=30, drift_terc="wide"):
    """Symbols S00..S29, spreads 1..30 bps -> terciles of 10. Drift where planted."""
    syms = [f"S{i:02d}" for i in range(n_syms)]
    hs = pd.Series(np.arange(1.0, n_syms + 1.0), index=syms)
    terc = liquidity_terciles(hs)
    mu = np.zeros(n_syms)
    planted = terc[terc == drift_terc].index if drift_terc else []
    for j, s in enumerate(syms):
        if s in planted:
            mu[j] = (j - n_syms + 5.5) * 2e-5  # distinct persistent drifts
    noise = RNG.normal(0, 2e-4, size=(n_hours, n_syms))
    logp = np.cumsum(noise, axis=0) + np.outer(np.arange(n_hours), mu)
    prices = pd.DataFrame(100.0 * np.exp(logp), columns=syms)
    return prices, hs, terc


class TestConstruction:
    def test_terciles_split_evenly(self):
        hs = pd.Series(np.arange(1.0, 10.0), index=list("ABCDEFGHI"))
        terc = liquidity_terciles(hs)
        assert (terc.value_counts() == 3).all()
        assert terc["A"] == "tight" and terc["I"] == "wide"

    def test_signal_skips_the_reversal_window(self):
        """A price jump inside the skip window must not move the signal."""
        prices, _, _ = _panel(400, 6, drift_terc=None)
        base = momentum_signal(prices, 300, 168)
        jumped = prices.copy()
        jumped.iloc[290:300, 0] *= 1.5          # inside the 24 h skip
        after = momentum_signal(jumped, 300, 168)
        assert after["S00"] == pytest.approx(base["S00"])

    def test_rebalances_do_not_overlap(self):
        prices, _, _ = _panel(2200, 12, drift_terc=None)
        ics = cell_ics(prices, list(prices.columns), 168, 168)
        idx = [r["t_idx"] for r in ics]
        assert all(b - a >= 168 for a, b in zip(idx, idx[1:]))
        assert len(idx) >= 10                    # ~12 non-overlapping weeks

    def test_perfectly_stable_ic_is_not_undecidable(self):
        """se=0 with nonzero mean is a perfect signal, not a zero t."""
        s = summarize_cell([{"ic": 0.8, "fwd_dispersion": 0.01}] * 5)
        assert s["p_value"] == 0.0
        assert verdict(s, fdr_pass=True) == "present"


class TestVerdictMachinery:
    def test_bh_fdr_hand_case(self):
        flags = bh_fdr([0.001] + [0.9] * 11)
        assert flags[0] is True and sum(flags) == 1

    def test_absent_requires_a_powered_null(self):
        tight = {"mean_ic": 0.001, "ci_lo": -0.02, "ci_hi": 0.02}
        wide_ci = {"mean_ic": 0.02, "ci_lo": -0.2, "ci_hi": 0.24}
        assert verdict(tight, fdr_pass=False) == "absent"
        assert verdict(wide_ci, fdr_pass=False) == "undecidable"
        assert abs(wide_ci["ci_hi"]) > ABSENT_IC_BAND  # why it stays open

    def test_tradability_hand_arithmetic(self):
        cell = {"mean_ic": 0.5, "mean_fwd_dispersion": 0.02}
        t = tradability(dict(cell), tercile_rt_cost_bps=20.0)
        assert t["expected_move_bps"] == pytest.approx(100.0)  # 0.5 x 200 bps
        assert t["mc_ratio"] == pytest.approx(5.0) and t["tradeable"]
        assert not tradability(dict(cell), 50.0)["tradeable"]


class TestPlantedStudy:
    def test_effect_found_where_planted_and_nowhere_else(self):
        prices, hs, _ = _panel(drift_terc="wide")
        result = run_study(prices, hs)
        by = {(c["window_h"], c["horizon_h"], c["tercile"]): c
              for c in result["cells"]}
        wide_1w = by[(168, 168, "wide")]
        assert wide_1w["verdict"].startswith("real")
        assert wide_1w["mean_ic"] > 0.3
        # noise terciles must never be promoted (undecidable/absent both fine)
        for (w, h, terc), c in by.items():
            if terc != "wide":
                assert not c["verdict"].startswith("real"), (w, h, terc, c)

    def test_pure_noise_promotes_nothing(self):
        prices, hs, _ = _panel(drift_terc=None)
        result = run_study(prices, hs)
        assert not any(c["verdict"].startswith("real") for c in result["cells"])
