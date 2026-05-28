"""
Tests for information-theoretic estimators.

Validates against analytical results for known distributions:
  - KSG MI vs bivariate Gaussian analytical MI
  - CMI on conditional independence (X⊥Y|Z → CMI ≈ 0)
  - Linear TE on AR(1) causal process
  - Cost threshold formula
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.it_engine.estimators import (
    ksg_mi,
    cmi,
    interaction_info,
    linear_te,
    min_info_bits,
)


# ---------------------------------------------------------------------------
# KSG Mutual Information
# ---------------------------------------------------------------------------

class TestKSGMI:
    """Test KSG MI against bivariate Gaussian analytical result."""

    def _analytical_mi_gaussian(self, rho: float) -> float:
        """MI of bivariate Gaussian: I(X;Y) = -0.5 log₂(1 - ρ²)"""
        return -0.5 * np.log2(1 - rho ** 2)

    @pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
    def test_bivariate_gaussian(self, rho):
        """KSG MI should be within 20% of analytical MI for bivariate Gaussian."""
        np.random.seed(42)
        n = 5000
        cov = [[1, rho], [rho, 1]]
        xy = np.random.multivariate_normal([0, 0], cov, size=n)
        x, y = xy[:, 0], xy[:, 1]

        estimated = ksg_mi(x, y, k=5)
        analytical = self._analytical_mi_gaussian(rho)

        # Allow 20% tolerance (KSG has finite-sample bias)
        assert estimated > 0, "MI should be non-negative"
        assert abs(estimated - analytical) / analytical < 0.20, (
            f"rho={rho}: estimated={estimated:.4f}, analytical={analytical:.4f}"
        )

    def test_independent_variables(self):
        """MI of independent variables should be near zero."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        y = np.random.randn(n)

        estimated = ksg_mi(x, y, k=5)
        assert estimated < 0.05, f"MI of independent vars should be ~0, got {estimated}"

    def test_nan_handling(self):
        """Should handle NaN values gracefully."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        x[::10] = np.nan  # 10% NaN

        estimated = ksg_mi(x, y, k=5)
        assert np.isfinite(estimated)

    def test_small_sample(self):
        """Should return 0 for too-small samples."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        assert ksg_mi(x, y, k=5) == 0.0


# ---------------------------------------------------------------------------
# Conditional MI
# ---------------------------------------------------------------------------

class TestCMI:
    """Test conditional mutual information."""

    def test_conditional_independence(self):
        """If X⊥Y|Z, then I(X;Y|Z) ≈ 0."""
        np.random.seed(42)
        n = 3000
        z = np.random.randn(n)
        x = z + np.random.randn(n) * 0.5  # X depends on Z
        y = z + np.random.randn(n) * 0.5  # Y depends on Z, independent of X given Z

        cmi_val = cmi(x, y, z, k=5)
        # Should be small (not exactly 0 due to estimation noise)
        assert abs(cmi_val) < 0.15, (
            f"CMI for conditional independence should be ~0, got {cmi_val}"
        )

    def test_synergy(self):
        """If X and Z are jointly needed to predict Y, CMI > MI."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        z = np.random.randn(n)
        y = x * z + np.random.randn(n) * 0.3  # XOR-like: need both X and Z

        mi_xy = ksg_mi(x, y, k=5)
        cmi_xyz = cmi(x, y, z, k=5)

        # Synergy: conditioning on Z should increase information
        assert cmi_xyz > mi_xy, (
            f"Expected synergy: CMI={cmi_xyz:.4f} > MI={mi_xy:.4f}"
        )


# ---------------------------------------------------------------------------
# Interaction Information
# ---------------------------------------------------------------------------

class TestInteractionInfo:
    """Test interaction information = CMI - MI."""

    def test_synergy_positive(self):
        """Synergistic relationship should give positive II."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        z = np.random.randn(n)
        y = x * z + np.random.randn(n) * 0.3

        ii = interaction_info(x, y, z, k=5)
        assert ii > 0, f"Synergy should give positive II, got {ii}"

    def test_redundancy_negative(self):
        """Redundant relationship should give negative II."""
        np.random.seed(42)
        n = 3000
        z = np.random.randn(n)
        x = z + np.random.randn(n) * 0.1  # X ≈ Z
        y = z + np.random.randn(n) * 0.5

        ii = interaction_info(x, y, z, k=5)
        assert ii < 0, f"Redundancy should give negative II, got {ii}"


# ---------------------------------------------------------------------------
# Linear Transfer Entropy
# ---------------------------------------------------------------------------

class TestLinearTE:
    """Test linear transfer entropy on AR processes."""

    def test_causal_direction(self):
        """TE should be positive in causal direction, ~0 in reverse."""
        np.random.seed(42)
        n = 5000
        x = np.random.randn(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + np.random.randn() * 0.5

        te_xy = linear_te(x, y, lag=1, order=1)  # X→Y (causal)
        te_yx = linear_te(y, x, lag=1, order=1)  # Y→X (non-causal)

        assert te_xy > 0.01, f"Causal TE should be positive, got {te_xy}"
        assert te_xy > te_yx * 2, (
            f"Causal TE={te_xy:.4f} should dominate reverse TE={te_yx:.4f}"
        )

    def test_independent_processes(self):
        """TE between independent processes should be ~0."""
        np.random.seed(42)
        n = 3000
        x = np.random.randn(n)
        y = np.random.randn(n)

        te = linear_te(x, y, lag=1, order=1)
        assert te < 0.01, f"TE of independent processes should be ~0, got {te}"


# ---------------------------------------------------------------------------
# Cost Threshold
# ---------------------------------------------------------------------------

class TestCostThreshold:
    """Test minimum information threshold for cost viability."""

    def test_binance_vip9_50min(self):
        """Binance VIP9 at 50min horizon: I_min should be small (~0.002 bits)."""
        i_min = min_info_bits(fee_rt_bps=1.61, sigma_r_bps=30.0)
        assert 0.001 < i_min < 0.01, f"Expected ~0.002 bits, got {i_min}"

    def test_high_fee_low_vol(self):
        """When fee ≥ σ, should return inf."""
        i_min = min_info_bits(fee_rt_bps=10.0, sigma_r_bps=5.0)
        assert i_min == float('inf')

    def test_zero_fee(self):
        """Zero fee → zero information needed."""
        i_min = min_info_bits(fee_rt_bps=0.0, sigma_r_bps=30.0)
        assert i_min == 0.0

    def test_monotonic_in_fee(self):
        """Higher fees should require more information."""
        i1 = min_info_bits(fee_rt_bps=1.0, sigma_r_bps=30.0)
        i2 = min_info_bits(fee_rt_bps=3.0, sigma_r_bps=30.0)
        i3 = min_info_bits(fee_rt_bps=7.0, sigma_r_bps=30.0)
        assert i1 < i2 < i3


class TestOverlapBias:
    """Verify that strided MI is not inflated by overlapping returns."""

    def test_strided_independent_near_zero(self):
        """For independent data, strided MI should stay near zero."""
        np.random.seed(42)
        n = 5000
        x = np.random.randn(n)
        y = np.random.randn(n)  # independent

        # Overlapping: create pseudo-returns with horizon overlap
        horizon = 500
        r_overlap = np.cumsum(y)
        r_overlap = r_overlap[horizon:] - r_overlap[:n - horizon]
        f_overlap = x[:n - horizon]

        # Strided
        stride = 100
        idx = np.arange(0, n - horizon, stride)
        r_strided = r_overlap[idx]
        f_strided = f_overlap[idx]

        mi_overlap = ksg_mi(f_overlap, r_overlap, k=5)
        mi_strided = ksg_mi(f_strided, r_strided, k=5)

        # Both should be near zero, but strided is a more honest estimate
        assert mi_strided < 0.1, f"Strided MI too high: {mi_strided}"

    def test_strided_correlated_lower_than_overlap(self):
        """For correlated data, strided MI should be lower than overlapping MI."""
        np.random.seed(123)
        n = 6000
        horizon = 500
        # Correlated: x drives returns
        x = np.random.randn(n)
        prices = np.cumsum(0.01 * x + np.random.randn(n) * 0.1)
        r_all = (prices[horizon:] - prices[:n - horizon]) / (1 + np.abs(prices[:n - horizon])) * 100
        f_all = x[:n - horizon]

        mi_overlap = ksg_mi(f_all, r_all, k=5)

        stride = 100
        idx = np.arange(0, n - horizon, stride)
        mi_strided = ksg_mi(f_all[idx], r_all[idx], k=5)

        # Strided should be lower — overlap inflates the estimate
        assert mi_strided <= mi_overlap + 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
