"""Planted test for PROC-12 — the null-calibration layer (the trust foundation).

Every IT estimator has finite-sample bias (KSG has a spurious ~0.07-bit floor), so raw
bits can't answer "is this real?". null_calibrate() shuffles the label to build a null
distribution and reports bits-above-null / z / p instead. The contract:

  (a) pure noise  -> bits_above_null ≈ 0, z ≈ 0, p ≈ 0.5, NOT informative (floor cancels);
  (b) planted edge -> large bits-above-null, large z, small p, informative;
  (c) seeded shuffles are reproducible (seed passed in, never a global RNG).
"""

from __future__ import annotations

import numpy as np
import pytest

from it_engine.estimators import ksg_mi
from it_engine.null_calibration import NullResult, load_null_config, null_calibrate


def _est(x, y):
    return ksg_mi(x, y, k=5)


class TestKillsTheFloor:
    def test_pure_noise_not_informative(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(800)
        y = rng.standard_normal(800)  # independent of x
        r = null_calibrate(_est, x, y, n_shuffles=100, rng=rng)
        # The raw KSG bits may be a small positive floor, but the null carries the SAME
        # floor -> bits_above_null ≈ 0, z small, p large -> not a real edge.
        assert abs(r.bits_above_null) < 0.03, r
        assert abs(r.z) < 3.0, r
        assert r.p > 0.05, r
        assert not r.informative(z_threshold=3.0)


class TestDetectsPlantedEdge:
    def test_planted_edge_informative(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(800)
        y = x + 0.4 * rng.standard_normal(800)  # strong, known dependence
        r = null_calibrate(_est, x, y, n_shuffles=100, rng=rng)
        assert r.bits_above_null > 0.1, r
        assert r.z > 5.0, r
        assert r.p < 0.05, r
        assert r.informative(z_threshold=3.0)

    def test_circular_method_also_detects(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(800)
        y = x + 0.4 * rng.standard_normal(800)
        r = null_calibrate(_est, x, y, n_shuffles=100, method="circular", rng=rng)
        assert r.informative(z_threshold=3.0)


class TestDeterminism:
    def test_seeded_reproducible(self):
        rng_x = np.random.default_rng(9)
        x = rng_x.standard_normal(500)
        y = x + rng_x.standard_normal(500)
        r1 = null_calibrate(_est, x, y, n_shuffles=50, rng=7)
        r2 = null_calibrate(_est, x, y, n_shuffles=50, rng=7)
        assert (r1.raw_bits, r1.null_mean, r1.z, r1.p) == (r2.raw_bits, r2.null_mean, r2.z, r2.p)


class TestInformativeGate:
    def test_requires_both_bits_and_z(self):
        r = NullResult(raw_bits=0.10, null_mean=0.05, null_std=0.01,
                       bits_above_null=0.05, z=5.0, p=0.001, n_shuffles=100)
        assert r.informative(i_min=0.0, z_threshold=3.0)
        assert not r.informative(i_min=0.10, z_threshold=3.0)   # bits below i_min
        assert not r.informative(i_min=0.0, z_threshold=6.0)    # z below threshold

    def test_p_value_floor(self):
        # +1 smoothing: p is never 0, bounded below by 1/(n_shuffles+1).
        rng = np.random.default_rng(3)
        x = rng.standard_normal(600)
        y = x + 0.3 * rng.standard_normal(600)
        r = null_calibrate(_est, x, y, n_shuffles=100, rng=rng)
        assert r.p >= 1.0 / (100 + 1) - 1e-12


class TestConfig:
    def test_loads_null_config_from_toml(self):
        cfg = load_null_config()
        assert cfg["n_shuffles"] == 200
        assert cfg["null_z_threshold"] == 3.0
        assert cfg["method"] in ("shuffle", "circular")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
