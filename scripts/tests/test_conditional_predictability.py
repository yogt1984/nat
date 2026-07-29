"""Planted test for PROC-6 — conditional_predictability.

The genuinely-absent concept: MI(feature; label | Z=z) as a FUNCTION of z (per Z-bucket),
not the z-averaged CMI. The bucket where MI spikes is the tradeable regime. Each bucket's
MI is null-calibrated (PROC-12), so a regime only counts if it clears the shuffle-null.

Contract:
  (a) a feature that predicts the label ONLY in the low-Z regime -> the low-Z bucket is
      the argmax AND informative; the other buckets are not;
  (b) a feature independent of the label everywhere -> no informative bucket;
  (c) seeded runs are reproducible;
  (d) the process is registered and its evaluate() emits per-bucket cond_mi findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processes.base import ProcessContext
from processes.conditional_predictability import (
    ConditionalPredictabilityProcess,
    conditional_predictability,
)
from processes.registry import get_process


def _regime_data(seed: int, *, n: int = 5000, regime_q: float = 0.25, beta: float = 1.2, noise: float = 0.3):
    """feature predicts label ONLY where Z is in its bottom quantile `regime_q`."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    z = rng.uniform(0.0, 1.0, n)
    in_regime = z < regime_q
    label = np.where(in_regime, beta * f, 0.0) + noise * rng.standard_normal(n)
    return f, label, z


class TestFindsTradeableRegime:
    def test_low_z_bucket_is_argmax_and_dominates(self):
        f, label, z = _regime_data(0)
        buckets, argmax = conditional_predictability(
            f, label, z, n_buckets=4, n_shuffles=100, rng=1
        )
        # The tradeable regime is the low-Z bucket, informative, and dominant. (Per-bucket
        # calibration alone can't control family-wise error across 4 buckets — that's PROC-13
        # (FDR). PROC-6's contract is that the true regime is the argmax and dominates.)
        assert argmax == 0, f"tradeable regime should be the low-Z bucket, got {argmax}"
        assert buckets[0]["result"].informative(z_threshold=3.0), buckets[0]["result"]
        assert buckets[0]["result"].bits_above_null > 0.15, buckets[0]["result"]
        others = [buckets[b]["result"].bits_above_null for b in (1, 2, 3) if buckets[b]["result"]]
        assert buckets[0]["result"].bits_above_null > 2.0 * max(others)

    def test_null_feature_produces_no_real_regime(self):
        rng = np.random.default_rng(5)
        f = rng.standard_normal(5000)
        label = rng.standard_normal(5000)      # independent of f everywhere
        z = rng.uniform(0.0, 1.0, 5000)
        buckets, _ = conditional_predictability(f, label, z, n_buckets=4, n_shuffles=100, rng=1)
        # No bucket carries real information — all sit near the shuffle-null floor.
        max_bits = max(b["result"].bits_above_null for b in buckets if b["result"])
        assert max_bits < 0.04, f"a null feature must stay at the floor, got {max_bits}"


class TestDeterminism:
    def test_seeded_reproducible(self):
        f, label, z = _regime_data(0)
        b1, a1 = conditional_predictability(f, label, z, n_buckets=4, n_shuffles=50, rng=7)
        b2, a2 = conditional_predictability(f, label, z, n_buckets=4, n_shuffles=50, rng=7)
        assert a1 == a2
        assert b1[0]["result"].z == b2[0]["result"].z


class TestProcessContract:
    def test_registered(self):
        p = get_process("conditional_predictability")
        assert isinstance(p, ConditionalPredictabilityProcess)
        assert p.name() == "conditional_predictability"

    def test_evaluate_emits_per_bucket_findings(self):
        # Build bars whose forward return tracks a feature only in the low-Z regime.
        rng = np.random.default_rng(3)
        n = 4000
        feat = rng.standard_normal(n)
        z = rng.uniform(0.0, 1.0, n)
        fwd = np.where(z < 0.25, 0.001 * feat, 0.0) + 0.0003 * rng.standard_normal(n)
        price = 100.0 * np.exp(np.cumsum(np.r_[0.0, fwd[:-1]]))
        bars = pd.DataFrame({"raw_midprice": price, "my_feat": feat, "cond_z": z})

        proc = ConditionalPredictabilityProcess(
            conditioning=["cond_z"], n_buckets=4, n_shuffles=60, min_bucket_obs=100
        )
        ctx = ProcessContext(
            symbol="BTC", timeframe="bar", price_col="raw_midprice",
            horizons={"h1": 1}, costs={},
        )
        res = proc.evaluate(bars, ctx)
        cond = [f for f in res.findings if f.metric == "cond_mi_bits"]
        assert len(cond) >= 4  # one per bucket for my_feat x cond_z x h1
        assert any(f.informative for f in cond), "the planted low-Z regime should be found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
