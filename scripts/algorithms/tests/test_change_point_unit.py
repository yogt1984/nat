"""Unit tests for ChangePointDetector algorithm."""

import numpy as np
import pytest

from algorithms.change_point_detector import ChangePointDetector


@pytest.fixture
def cpd():
    return ChangePointDetector()


def _make_tick(imb: float, vol: float = 0.001, ent: float = 0.5) -> dict:
    return {
        "imbalance_qty_l1_mean": imb,
        "vol_returns_5m_last": vol,
        "ent_tick_1m_mean": ent,
    }


def test_cusum_zero_on_constant_input(cpd):
    """Feed 200 identical ticks. CUSUM signal stays near 0. No alarms."""
    results = []
    for _ in range(200):
        r = cpd.step(_make_tick(0.0))
        results.append(r)

    # After calibration stabilizes, CUSUM should be near 0
    late_signals = [r["alg_cpd_cusum_signal"] for r in results[50:]]
    assert all(abs(s) < 1.0 for s in late_signals), "CUSUM should be near 0 on constant input"


def test_cusum_detects_mean_shift(cpd):
    """Feed 200 ticks at mean=0, then 200 at mean=2.0.
    CUSUM alarm fires within 50 bars of shift."""
    # Stable phase
    for _ in range(200):
        cpd.step(_make_tick(0.0))

    # Record regime_age before shift
    pre_shift_age = cpd.step(_make_tick(0.0))["alg_cpd_regime_age"]

    # Shift phase — strong mean shift
    alarm_fired = False
    for i in range(200):
        r = cpd.step(_make_tick(2.0))
        if r["alg_cpd_regime_age"] < 5 and i > 5:
            alarm_fired = True
            break

    assert alarm_fired, "CUSUM should detect mean shift within 50 bars"


def test_bayesian_run_length_grows_on_stable(cpd):
    """Feed 150 stable ticks. Expected run length > 50."""
    rng = np.random.default_rng(42)
    for _ in range(150):
        cpd.step(_make_tick(rng.normal(0, 0.1)))

    r = cpd.step(_make_tick(0.05))
    assert r["alg_cpd_run_length"] > 50, f"Expected RL > 50, got {r['alg_cpd_run_length']}"


def test_bayesian_run_length_drops_on_change(cpd):
    """Feed 150 stable + 50 shifted ticks.
    Expected run length should drop significantly after the shift."""
    rng = np.random.default_rng(42)

    # Stable phase — run length grows
    for _ in range(150):
        cpd.step(_make_tick(rng.normal(0, 0.1)))
    pre_shift_rl = cpd.step(_make_tick(rng.normal(0, 0.1)))["alg_cpd_run_length"]

    # Shifted phase — run length should drop
    min_rl = float("inf")
    for _ in range(50):
        r = cpd.step(_make_tick(rng.normal(3.0, 0.1)))
        min_rl = min(min_rl, r["alg_cpd_run_length"])

    assert min_rl < pre_shift_rl * 0.5, (
        f"Expected run length to drop after shift: pre={pre_shift_rl:.1f}, min_post={min_rl:.1f}"
    )


def test_regime_age_resets_on_alarm():
    """After CUSUM fires, alg_cpd_regime_age resets to 0."""
    cpd = ChangePointDetector(cusum_threshold=3.0, cusum_drift=0.01)

    # Build up calibration on stable data
    for _ in range(100):
        cpd.step(_make_tick(0.0))

    # Hit with extreme values to trigger alarm
    age_reset = False
    prev_age = cpd.step(_make_tick(0.0))["alg_cpd_regime_age"]

    for _ in range(200):
        r = cpd.step(_make_tick(5.0))
        if r["alg_cpd_regime_age"] < prev_age:
            age_reset = True
            break
        prev_age = r["alg_cpd_regime_age"]

    assert age_reset, "regime_age should reset after CUSUM alarm"


def test_run_length_capped():
    """After many ticks, run_length_probs array never exceeds max_run_length."""
    max_rl = 50
    cpd = ChangePointDetector(max_run_length=max_rl)
    rng = np.random.default_rng(42)

    for _ in range(200):
        cpd.step(_make_tick(rng.normal(0, 0.1)))

    assert len(cpd._rl_probs) <= max_rl, f"RL array length {len(cpd._rl_probs)} exceeds max {max_rl}"


def test_nan_input_returns_nan(cpd):
    """If imbalance_qty_l1_mean is NaN, all outputs are NaN."""
    r = cpd.step(_make_tick(float("nan")))
    assert np.isnan(r["alg_cpd_cusum_signal"])
    assert np.isnan(r["alg_cpd_run_length"])
    assert np.isnan(r["alg_cpd_change_prob"])
    assert np.isnan(r["alg_cpd_regime_age"])
