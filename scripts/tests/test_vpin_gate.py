"""HF4 extraction — planted tests for the standalone `vpin_gate` unit.

§4.5 validated the VPIN gate directionally (Sharpe improved 3/3 as a veto inside
`toxic_vwap_reversion`) but the gate itself was never a registered unit — nothing else
on the maker path (HF1 quoting, A4-gated mean reversion) could consume it without
dragging in the taker fade it was welded to. This extracts the permission half.

What the tests plant:

- **parity with the donor**: on an identical stream and identical parameters, the
  standalone gate must equal `toxic_vwap_reversion`'s internal `alg_txvr_gate` bit for
  bit — an extraction that drifts from the validated original is a new, unvalidated
  unit wearing its name;
- **hand-computed percentiles**: the max-rank "fraction of window <= x" convention is
  asserted on planted values, because two off-by-one rank conventions both look
  plausible and only one matches the §4.5 record;
- **the veto composes**: gate closes on toxic VPIN and on blown spread independently;
- **NaN discipline**: NaN in → NaN out for every output, and the NaN tick must not
  contaminate the rolling buffers (the next clean tick sees the same window a
  NaN-free stream would have shown it);
- **step ≡ run_batch** past warmup on clean streams, per the dispatch contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from algorithms.toxic_vwap_reversion import ToxicVwapReversion  # noqa: E402
from algorithms.vpin_gate import VpinGate  # noqa: E402


def _tick(vpin: float, spread: float = 1.0) -> dict:
    return {"toxic_vpin_50": vpin, "raw_spread_bps": spread}


def _stream(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "toxic_vpin_50": rng.uniform(0.1, 0.9, n),
        "raw_spread_bps": rng.uniform(0.5, 5.0, n),
        "flow_vwap_deviation": rng.normal(0, 1e-3, n),
    })


class TestPercentileAndVeto:
    def test_hand_computed_percentile(self):
        g = VpinGate(w_p=4, theta_pct=0.70, spread_pct_max=0.90)
        for v in (0.1, 0.2, 0.3):
            g.step(_tick(v))
        out = g.step(_tick(0.25))
        # window is now [0.1, 0.2, 0.3, 0.25]; frac <= 0.25 is 3/4
        assert out["alg_vping_pct"] == 0.75

    def test_gate_closes_on_toxic_vpin(self):
        # spreads must VARY: the max-rank percentile of a constant window is 1.0,
        # so a flat spread stream keeps the spread veto permanently closed (donor
        # semantics, conservative) — a planted fixture has to avoid that trap.
        g = VpinGate(w_p=8, theta_pct=0.70, spread_pct_max=0.90)
        for i, v in enumerate(np.linspace(0.1, 0.5, 8)):
            g.step(_tick(v, spread=1.0 + 0.1 * i))
        assert g.step(_tick(0.05, spread=1.2))["alg_vping_gate"] == 1.0  # calmest
        assert g.step(_tick(0.99, spread=1.2))["alg_vping_gate"] == 0.0  # most toxic

    def test_gate_closes_on_blown_spread_alone(self):
        g = VpinGate(w_p=8, theta_pct=0.70, spread_pct_max=0.90)
        for _ in range(8):
            g.step(_tick(0.2, spread=1.0))
        out = g.step(_tick(0.01, spread=50.0))     # calm VPIN, blown spread
        assert out["alg_vping_pct"] < 0.70
        assert out["alg_vping_gate"] == 0.0

    def test_size_is_toxicity_scaled_permission(self):
        g = VpinGate(w_p=4, theta_pct=0.99, spread_pct_max=0.99)
        for i, v in enumerate((0.1, 0.2, 0.3)):
            g.step(_tick(v, spread=1.0 + 0.1 * i))
        out = g.step(_tick(0.15, spread=1.05))     # pct = 2/4 = 0.5, gate open
        assert out["alg_vping_gate"] == 1.0
        assert out["alg_vping_size"] == 0.5


class TestNaNDiscipline:
    def test_nan_in_nan_out_all_outputs(self):
        g = VpinGate(w_p=4)
        g.step(_tick(0.2))
        out = g.step(_tick(np.nan))
        assert all(np.isnan(v) for v in out.values())

    def test_nan_tick_does_not_contaminate_buffers(self):
        a, b = VpinGate(w_p=4), VpinGate(w_p=4)
        for v in (0.1, 0.2, 0.3):
            a.step(_tick(v))
            b.step(_tick(v))
        a.step(_tick(np.nan))                      # only a sees the NaN
        assert a.step(_tick(0.25)) == b.step(_tick(0.25))


class TestBatchAndParity:
    def test_step_equals_run_batch_past_warmup(self):
        df = _stream(700, seed=1)
        g = VpinGate(w_p=64)
        batch = g.run_batch(df)
        g.reset()
        rows = [g.step({c: df[c].iloc[i] for c in g.required_columns()})
                for i in range(len(df))]
        for name in g.feature_names:
            s = np.array([r[name] for r in rows])[g.warmup:]
            bv = batch[name].to_numpy()[g.warmup:]
            assert np.allclose(s, bv, equal_nan=True, atol=1e-12), name

    def test_parity_with_donor_gate(self):
        """The extraction must reproduce §4.5's validated gate exactly."""
        df = _stream(900, seed=2)
        donor = ToxicVwapReversion(w_z=16, w_p=64, theta_pct=0.70,
                                   spread_pct_max=0.90)
        gate = VpinGate(w_p=64, theta_pct=0.70, spread_pct_max=0.90)
        d_out = donor.run_batch(df)["alg_txvr_gate"].to_numpy()
        g_out = gate.run_batch(df)["alg_vping_gate"].to_numpy()
        warm = max(donor.warmup, gate.warmup)
        assert np.array_equal(d_out[warm:], g_out[warm:]), \
            "standalone gate drifted from the validated donor"

    def test_warmup_blanked_in_batch(self):
        df = _stream(200, seed=3)
        g = VpinGate(w_p=64)
        batch = g.run_batch(df)
        assert batch.iloc[:g.warmup].isna().all().all()
        assert batch.iloc[g.warmup + 1:].notna().any().any()
