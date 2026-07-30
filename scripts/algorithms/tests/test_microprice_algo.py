"""Planted test for HF1 — the registered `microprice` algorithm.

The Stoikov micro-price is the fair-value anchor of the maker line: its deviation from
mid is the expected short-horizon mid move in bps — exactly the `center` adjustment the
GAP-04 maker sim quotes around, and the primitive A4/HF5 build on. The feature existed
(`scripts/features/microprice.py`, live `raw_microprice`) but no registered algorithm
consumed it (TASKS HF1).

Contract:
  (a) analytic mode: alg_mp_dev_bps == (I − 0.5) · spread_bps exactly; antisymmetric
      under I → 1 − I; zero at balance;
  (b) planted predictiveness: by construction mid moves toward the weighted mid →
      dev sign predicts the next-tick mid move (positive IC);
  (c) fitted mode: a provided mp_micro_adj_bps column (the fitted Stoikov g*)
      overrides the analytic fallback;
  (d) NaN inputs → NaN outputs, state frozen across the gap;
  (e) EXACT step/run_batch parity (rtol=1e-9) — the BUG-5 lesson, enforced from birth;
  (f) registered, alg_ prefix, step returns exactly the declared keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.microprice import Microprice
from algorithms.registry import get_algorithm

KEYS = ("alg_mp_dev_bps", "alg_mp_dev_ema", "alg_mp_signal")


def _tick(mid=100.0, imb=0.5, spread=2.0, mp_adj=None):
    t = {"raw_midprice": mid, "imbalance_qty_l1": imb, "raw_spread_bps": spread}
    if mp_adj is not None:
        t["mp_micro_adj_bps"] = mp_adj
    return t


def _df(n=2000, seed=0, coupling=0.6, spread=2.0):
    """Planted: imbalance drives the NEXT tick's mid move toward the weighted mid."""
    rng = np.random.default_rng(seed)
    imb = np.clip(0.5 + 0.25 * rng.standard_normal(n), 0.01, 0.99)
    mid = np.empty(n)
    mid[0] = 100.0
    for t in range(1, n):
        dev_bps = (imb[t - 1] - 0.5) * spread          # analytic microprice deviation
        noise = 0.3 * rng.standard_normal()
        mid[t] = mid[t - 1] * (1.0 + (coupling * dev_bps + noise) * 1e-4)
    return pd.DataFrame({
        "raw_midprice": mid,
        "imbalance_qty_l1": imb,
        "raw_spread_bps": np.full(n, spread),
    })


class TestAnalyticDeviation:
    def test_exact_formula_and_antisymmetry(self):
        alg = Microprice()
        for _ in range(50):                            # past warmup for the z machinery
            alg.step(_tick())
        out_hi = alg.step(_tick(imb=0.8, spread=4.0))
        assert out_hi["alg_mp_dev_bps"] == pytest.approx((0.8 - 0.5) * 4.0)

        alg2 = Microprice()
        for _ in range(50):
            alg2.step(_tick())
        out_lo = alg2.step(_tick(imb=0.2, spread=4.0))
        assert out_lo["alg_mp_dev_bps"] == pytest.approx(-out_hi["alg_mp_dev_bps"])

    def test_balanced_book_is_zero(self):
        alg = Microprice()
        for _ in range(50):
            alg.step(_tick())
        out = alg.step(_tick(imb=0.5, spread=3.0))
        assert out["alg_mp_dev_bps"] == pytest.approx(0.0)


class TestPlantedPredictiveness:
    def test_dev_predicts_next_tick_move(self):
        df = _df(seed=1)
        out = Microprice().run_batch(df)
        dev = out["alg_mp_dev_bps"].to_numpy()
        mid = df["raw_midprice"].to_numpy()
        fwd = np.full(len(df), np.nan)
        fwd[:-1] = np.log(mid[1:] / mid[:-1])
        mask = np.isfinite(dev) & np.isfinite(fwd)
        assert mask.sum() > 1000
        from scipy.stats import spearmanr
        ic, _ = spearmanr(dev[mask], fwd[mask])
        assert ic > 0.15, f"planted coupling must be recovered, IC={ic:.3f}"

    def test_signal_z_tracks_dev(self):
        df = _df(seed=2)
        out = Microprice().run_batch(df)
        dev = out["alg_mp_dev_bps"].to_numpy()
        z = out["alg_mp_signal"].to_numpy()
        mask = np.isfinite(dev) & np.isfinite(z)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(dev[mask], z[mask])
        assert rho > 0.8


class TestFittedOverride:
    def test_mp_column_overrides_analytic(self):
        df = _df(n=500, seed=3)
        df["mp_micro_adj_bps"] = 0.42                  # fitted g* says +0.42 bps everywhere
        out = Microprice().run_batch(df)
        dev = out["alg_mp_dev_bps"].to_numpy()
        assert np.allclose(dev[np.isfinite(dev)], 0.42)

    def test_step_honors_fitted_value(self):
        alg = Microprice()
        for _ in range(50):
            alg.step(_tick())
        out = alg.step(_tick(imb=0.9, spread=4.0, mp_adj=-0.7))
        assert out["alg_mp_dev_bps"] == pytest.approx(-0.7)


class TestNanContract:
    def test_nan_input_nan_output(self):
        alg = Microprice()
        for _ in range(50):
            alg.step(_tick())
        out = alg.step(_tick(imb=np.nan))
        assert all(np.isnan(v) for v in out.values())

    def test_state_frozen_across_gap(self):
        alg = Microprice()
        for _ in range(100):
            alg.step(_tick(imb=0.7))
        before = alg.step(_tick(imb=0.7))
        for _ in range(5):
            alg.step(_tick(imb=np.nan))                # gap: no state updates
        after = alg.step(_tick(imb=0.7))
        assert after["alg_mp_signal"] == pytest.approx(
            before["alg_mp_signal"], rel=1e-6
        )


class TestStepBatchParity:
    def test_exact_parity_with_gaps(self):
        df = _df(seed=4)
        df.loc[df.index[300:320], "imbalance_qty_l1"] = np.nan   # planted gap
        batch = Microprice().run_batch(df)
        inst = Microprice()
        step = pd.DataFrame([
            inst.step({k: float(row[k]) for k in
                       ("raw_midprice", "imbalance_qty_l1", "raw_spread_bps")})
            for _, row in df.iterrows()
        ])
        warmup = Microprice().warmup
        for col in KEYS:
            b = batch[col].to_numpy()[warmup:]
            s = step[col].to_numpy()[warmup:]
            assert np.array_equal(np.isnan(b), np.isnan(s)), f"NaN pattern differs: {col}"
            mask = np.isfinite(b) & np.isfinite(s)
            assert mask.sum() > 1000, col
            np.testing.assert_allclose(b[mask], s[mask], rtol=1e-9, atol=1e-12,
                                       err_msg=col)


class TestRegistryContract:
    def test_registered_and_keys(self):
        from algorithms import discover_all
        discover_all()
        alg = get_algorithm("microprice")
        assert isinstance(alg, Microprice)
        feats = [f.name for f in alg.alg_features()]
        assert set(feats) == set(KEYS)
        assert all(f.startswith("alg_") for f in feats)
        out = Microprice().step(_tick())
        assert set(out.keys()) == set(KEYS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
