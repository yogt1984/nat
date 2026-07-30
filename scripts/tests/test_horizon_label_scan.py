"""Planted test for PROC-7 — the horizon/label MI-surface meta-process.

`horizon_label_scan` sweeps (horizon h × barrier geometry g × regime bucket z):
for each (h, g) it derives triple-barrier labels, then measures each feature's
null-calibrated conditional MI against that label per Z-bucket (PROC-6/12), and
BH-corrects the whole surface (PROC-13). The argmax cell — always carrying its
BH q — names the best (target, horizon, regime) triple.

Planted construction: a feature drives the price ONLY (a) in the low-Z regime,
(b) over the next 4 bars, (c) strongly enough to hit a wide barrier. Therefore
the surface must peak at exactly (h=4, wide geometry, bucket 0):
  - h=12 dilutes the 4-bar drift with 8 extra bars of noise/other events;
  - a tight (0.5σ) barrier is hit by the first bar of noise -> label ~ coin flip;
  - high-Z buckets carry no drive at all.

Contract:
  (a) the scan's argmax is the planted cell and every neighboring cell is lower;
  (b) an all-null scan yields zero discoveries after BH (never a lucky argmax);
  (c) the argmax is never surfaced without its q-value;
  (d) seeded runs are bit-for-bit reproducible;
  (e) grid caps are logged, never silent;
  (f) unconditional mode (no Z) still produces one cell per (feature, h, g).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processes.base import ProcessContext
from processes.horizon_label_scan import HorizonLabelScanProcess
from processes.registry import get_process


def _ctx():
    return ProcessContext(
        symbol="BTC", timeframe="15min", price_col="raw_midprice",
        horizons={"unused": 1}, costs={},
    )


def _planted_bars(seed=0, n=2000, regime_q=0.25, drive_h=4, d=0.0018, sigma=0.001):
    """Feature f_good moves price over the NEXT drive_h bars, only when z < regime_q."""
    rng = np.random.default_rng(seed)
    f_good = rng.standard_normal(n)
    f_noise = rng.standard_normal(n)
    z = rng.uniform(0.0, 1.0, n)
    r = sigma * rng.standard_normal(n)
    drive = np.where(z < regime_q, d * f_good, 0.0)
    for k in range(1, drive_h + 1):
        r[k:] += drive[:-k]
    price = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame({
        "raw_midprice": price, "f_good": f_good, "f_noise": f_noise, "cond_z": z,
    })


def _scan(bars, **over):
    params = dict(
        features=["f_"], conditioning=["cond_z"],
        horizons=[4, 12], geometries=[[2.0, 2.0], [0.5, 0.5]],
        vol_window=48, n_buckets=4, n_shuffles=40,
        min_bucket_obs=80, seed=1,
    )
    params.update(over)
    proc = HorizonLabelScanProcess(**params)
    return proc.evaluate(bars, _ctx())


class TestPlantedOptimum:
    def test_argmax_is_the_planted_cell(self):
        res = _scan(_planted_bars(seed=0))
        surface = res.summary["surface"]
        top = surface["argmax"]
        assert top is not None
        assert top["feature"] == "f_good"
        assert top["horizon"] == "4bar"
        assert top["extras"]["pt_mult"] == 2.0
        assert top["extras"]["bucket"] == 0            # the low-Z regime
        # Never surfaced bare. (Surviving BH on this 32-cell grid would need
        # n_shuffles ≳ m/alpha ≈ 640 — the focused-grid test below covers survival.)
        assert top["q_value"] is not None

    def test_planted_cell_survives_fdr_on_focused_grid(self):
        # Small family (4 cells) + deep null (200 shuffles -> p floor 1/201) so the
        # planted regime can actually clear BH: q = 0.005 * 4 / 1 = 0.02 <= 0.05.
        res = _scan(
            _planted_bars(seed=0),
            features=["f_good"], horizons=[4], geometries=[[2.0, 2.0]],
            n_shuffles=200,
        )
        surf = res.summary["surface"]
        assert surf["n_discoveries"] >= 1
        top = surf["argmax"]
        assert top["extras"]["bucket"] == 0
        assert top["q_value"] <= 0.05

    def test_neighbors_are_lower(self):
        res = _scan(_planted_bars(seed=0))
        cells = {
            (f.horizon, f.extras["pt_mult"], f.extras["bucket"]): f.value
            for f in res.findings if f.feature == "f_good"
        }
        peak = cells[("4bar", 2.0, 0)]
        for key, v in cells.items():
            if key != ("4bar", 2.0, 0):
                assert v < peak, (key, v, peak)

    def test_noise_feature_never_beats_planted(self):
        res = _scan(_planted_bars(seed=0))
        best_noise = max(f.value for f in res.findings if f.feature == "f_noise")
        peak = max(f.value for f in res.findings if f.feature == "f_good")
        assert peak > best_noise

    def test_horizon_profile_prefers_short(self):
        res = _scan(_planted_bars(seed=0))
        prof = res.summary["surface"]["horizon_profile"]
        assert prof["4bar"] > prof["12bar"]


class TestAllNullFdr:
    def test_pure_noise_scan_yields_zero_discoveries(self):
        rng = np.random.default_rng(11)
        n = 1200
        bars = pd.DataFrame({
            "raw_midprice": 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(n))),
            "f_a": rng.standard_normal(n),
            "f_b": rng.standard_normal(n),
            "cond_z": rng.uniform(0.0, 1.0, n),
        })
        res = _scan(bars)
        assert res.summary["surface"]["n_discoveries"] == 0
        assert not any(f.informative for f in res.findings)
        # every cell still carries its correction
        assert all(f.p_adjusted is not None for f in res.findings)


class TestDeterminism:
    def test_same_seed_same_surface(self):
        bars = _planted_bars(seed=3)
        r1 = _scan(bars, seed=9)
        r2 = _scan(bars, seed=9)
        v1 = [(f.feature, f.horizon, f.extras["pt_mult"], f.extras["bucket"], f.value, f.p_value)
              for f in r1.findings]
        v2 = [(f.feature, f.horizon, f.extras["pt_mult"], f.extras["bucket"], f.value, f.p_value)
              for f in r2.findings]
        assert v1 == v2
        assert r1.summary["surface"]["argmax"] == r2.summary["surface"]["argmax"]


class TestGridBounds:
    def test_truncation_is_logged_not_silent(self):
        bars = _planted_bars(seed=4)
        # cap so only one feature fits: cells/feature = 2h * 2g * 4 buckets = 16
        res = _scan(bars, max_cells=16)
        surf = res.summary["surface"]
        assert surf["features_truncated"], "dropped features must be recorded"
        assert len(res.features_tested) == 1
        tested_and_dropped = set(res.features_tested) | set(surf["features_truncated"])
        assert tested_and_dropped == {"f_good", "f_noise"}


class TestUnconditionalMode:
    def test_no_conditioning_gives_one_cell_per_h_g(self):
        bars = _planted_bars(seed=5)
        res = _scan(bars, conditioning=[])
        cells = [(f.feature, f.horizon, f.extras["pt_mult"]) for f in res.findings]
        assert len(cells) == len(set(cells))            # exactly one cell per (feat, h, g)
        assert all(f.extras["bucket"] is None for f in res.findings)
        # the planted (4bar, wide) cell still wins for f_good
        top = res.summary["surface"]["argmax"]
        assert (top["feature"], top["horizon"], top["extras"]["pt_mult"]) == ("f_good", "4bar", 2.0)


class TestContract:
    def test_registered(self):
        p = get_process("horizon_label_scan")
        assert isinstance(p, HorizonLabelScanProcess)
        assert p.name() == "horizon_label_scan"

    def test_findings_shape(self):
        res = _scan(_planted_bars(seed=6))
        assert res.findings
        for f in res.findings:
            assert f.metric == "cond_mi_bits"
            assert f.p_value is not None
            for key in ("pt_mult", "sl_mult", "bucket", "z", "raw_bits", "n"):
                assert key in f.extras, key

    def test_missing_conditioning_column_errors_cleanly(self):
        bars = _planted_bars(seed=7).drop(columns=["cond_z"])
        res = _scan(bars)
        assert res.findings == []
        assert res.summary.get("error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
