"""Unit tests for RegimeStateMachine algorithm."""

import numpy as np
import pytest

from algorithms.regime_state_machine import (
    RegimeStateMachine,
    ACCUMULATION, DISTRIBUTION, TRENDING_UP, TRENDING_DOWN,
    RANGING, VOLATILE_NOISE,
)


@pytest.fixture
def rsm():
    return RegimeStateMachine()


def _make_tick(**overrides) -> dict:
    """Create a neutral tick dict with all required columns."""
    tick = {
        "vol_returns_5m_last": 0.001,
        "trend_hurst_300_mean": 0.50,
        "ent_tick_1m_mean": 0.50,
        "whale_net_flow_4h_sum": 0.0,
        "toxic_vpin_50_mean": 0.40,
        "regime_accumulation_score_mean": 0.50,
    }
    tick.update(overrides)
    return tick


def _warmup(rsm, n=30):
    """Feed neutral ticks to build vol buffer."""
    for _ in range(n):
        rsm.step(_make_tick())


def test_accumulation_detection(rsm):
    """High accum score, positive whale, low entropy -> ACCUMULATION."""
    _warmup(rsm)
    # Force enough bars at volatile noise to allow transition
    for _ in range(10):
        rsm.step(_make_tick(
            regime_accumulation_score_mean=0.85,
            whale_net_flow_4h_sum=5000.0,
            ent_tick_1m_mean=0.25,
            trend_hurst_300_mean=0.40,
        ))
    r = rsm.step(_make_tick(
        regime_accumulation_score_mean=0.85,
        whale_net_flow_4h_sum=5000.0,
        ent_tick_1m_mean=0.25,
        trend_hurst_300_mean=0.40,
    ))
    assert r["alg_rsm_regime"] == float(ACCUMULATION)


def test_volatile_noise_detection(rsm):
    """Very high vol, high entropy, high toxicity -> VOLATILE_NOISE."""
    _warmup(rsm)
    for _ in range(10):
        r = rsm.step(_make_tick(
            vol_returns_5m_last=0.1,  # Very high relative to median
            ent_tick_1m_mean=0.85,
            toxic_vpin_50_mean=0.90,
        ))
    assert r["alg_rsm_regime"] == float(VOLATILE_NOISE)


def test_regime_is_integer(rsm):
    """alg_rsm_regime always in {0,1,2,3,4,5} across 500 ticks."""
    rng = np.random.default_rng(42)
    valid_states = {float(i) for i in range(6)}

    for _ in range(500):
        tick = _make_tick(
            vol_returns_5m_last=rng.normal(0, 0.01),
            trend_hurst_300_mean=rng.uniform(0.3, 0.7),
            ent_tick_1m_mean=rng.uniform(0, 1),
            whale_net_flow_4h_sum=rng.normal(0, 2000),
            toxic_vpin_50_mean=rng.uniform(0, 1),
            regime_accumulation_score_mean=rng.uniform(0, 1),
        )
        r = rsm.step(tick)
        assert r["alg_rsm_regime"] in valid_states


def test_confidence_range(rsm):
    """alg_rsm_confidence in [0, 1] always."""
    rng = np.random.default_rng(42)
    for _ in range(300):
        tick = _make_tick(
            vol_returns_5m_last=rng.normal(0, 0.01),
            trend_hurst_300_mean=rng.uniform(0.3, 0.7),
            ent_tick_1m_mean=rng.uniform(0, 1),
            whale_net_flow_4h_sum=rng.normal(0, 2000),
            toxic_vpin_50_mean=rng.uniform(0, 1),
            regime_accumulation_score_mean=rng.uniform(0, 1),
        )
        r = rsm.step(tick)
        assert 0.0 <= r["alg_rsm_confidence"] <= 1.0


def test_trade_allowed_zero_in_noise(rsm):
    """When state=VOLATILE_NOISE, alg_rsm_trade_allowed=0."""
    _warmup(rsm)
    for _ in range(10):
        r = rsm.step(_make_tick(
            vol_returns_5m_last=0.1,
            ent_tick_1m_mean=0.85,
            toxic_vpin_50_mean=0.90,
        ))
    assert r["alg_rsm_regime"] == float(VOLATILE_NOISE)
    assert r["alg_rsm_trade_allowed"] == 0.0


def test_min_duration_hysteresis():
    """State doesn't change for min_duration bars even if scores shift."""
    rsm = RegimeStateMachine(min_duration=10)

    # Warmup to build vol buffer
    for _ in range(20):
        rsm.step(_make_tick())

    # Drive into accumulation (transition happens, regime_age resets to 0)
    accum_tick = _make_tick(
        regime_accumulation_score_mean=0.85,
        whale_net_flow_4h_sum=5000.0,
        ent_tick_1m_mean=0.25,
        trend_hurst_300_mean=0.40,
    )
    for _ in range(12):
        rsm.step(accum_tick)
    # Now in ACCUMULATION with regime_age ~ 12

    # Force transition to trending_up -> regime_age resets to 0
    trend_tick = _make_tick(
        trend_hurst_300_mean=0.70,
        whale_net_flow_4h_sum=5000.0,
        ent_tick_1m_mean=0.15,
    )
    for _ in range(12):
        rsm.step(trend_tick)
    # regime_age is now ~12, transition allowed

    # Force ANOTHER transition to accumulation -> age resets to 0
    rsm.step(accum_tick)
    # age is now 0 or 1

    # IMMEDIATELY switch to noise — hysteresis should hold accumulation
    noise_tick = _make_tick(
        vol_returns_5m_last=0.1,
        ent_tick_1m_mean=0.85,
        toxic_vpin_50_mean=0.90,
    )

    held_count = 0
    for i in range(5):
        r = rsm.step(noise_tick)
        if r["alg_rsm_regime"] != float(VOLATILE_NOISE):
            held_count += 1

    # With min_duration=10, regime should hold for these 5 bars
    assert held_count == 5, f"Expected 5 held bars, got {held_count}"


def test_regime_age_increments(rsm):
    """Transition risk decreases as regime age grows."""
    _warmup(rsm)
    risks = []
    for _ in range(20):
        r = rsm.step(_make_tick())
        risks.append(r["alg_rsm_transition_risk"])
    # Risk should generally decrease (exponential decay)
    assert risks[-1] < risks[0], "Transition risk should decay with age"


def test_tiebreak_favors_noise():
    """When two states tie at max score, higher index (VOLATILE_NOISE) wins."""
    rsm = RegimeStateMachine(min_duration=1)
    _warmup(rsm, 30)

    # Create a tick where VOLATILE_NOISE and RANGING both score equally
    # RANGING: ent > 0.60 (yes), hurst < 0.45 (yes), |whale| < 500 (yes) = 3
    # VOLATILE_NOISE: vol > 2*median (yes), vpin > 0.80 (yes), ent > 0.70 (yes) = 3
    for _ in range(10):
        r = rsm.step(_make_tick(
            vol_returns_5m_last=0.1,       # high vol
            trend_hurst_300_mean=0.35,     # anti-persistent -> RANGING
            ent_tick_1m_mean=0.75,         # high entropy -> both
            whale_net_flow_4h_sum=100.0,   # low whale -> RANGING
            toxic_vpin_50_mean=0.85,       # high vpin -> VOLATILE_NOISE
            regime_accumulation_score_mean=0.30,
        ))
    # VOLATILE_NOISE has higher index -> should win tie
    assert r["alg_rsm_regime"] == float(VOLATILE_NOISE)


def test_all_six_states_reachable():
    """Over varied synthetic data, all 6 states appear at least once."""
    rsm = RegimeStateMachine(min_duration=1)  # Allow fast transitions
    seen = set()

    # Accumulation
    for _ in range(20):
        r = rsm.step(_make_tick(regime_accumulation_score_mean=0.9, whale_net_flow_4h_sum=5000, ent_tick_1m_mean=0.2))
        seen.add(r["alg_rsm_regime"])

    # Distribution
    for _ in range(20):
        r = rsm.step(_make_tick(regime_accumulation_score_mean=0.1, whale_net_flow_4h_sum=-5000, ent_tick_1m_mean=0.2))
        seen.add(r["alg_rsm_regime"])

    # Trending up
    for _ in range(20):
        r = rsm.step(_make_tick(trend_hurst_300_mean=0.7, whale_net_flow_4h_sum=5000, ent_tick_1m_mean=0.15))
        seen.add(r["alg_rsm_regime"])

    # Trending down
    for _ in range(20):
        r = rsm.step(_make_tick(trend_hurst_300_mean=0.7, whale_net_flow_4h_sum=-5000, ent_tick_1m_mean=0.15))
        seen.add(r["alg_rsm_regime"])

    # Ranging
    for _ in range(20):
        r = rsm.step(_make_tick(ent_tick_1m_mean=0.75, trend_hurst_300_mean=0.35, whale_net_flow_4h_sum=100))
        seen.add(r["alg_rsm_regime"])

    # Volatile noise
    for _ in range(20):
        r = rsm.step(_make_tick(vol_returns_5m_last=0.1, ent_tick_1m_mean=0.85, toxic_vpin_50_mean=0.90))
        seen.add(r["alg_rsm_regime"])

    expected = {float(i) for i in range(6)}
    assert seen == expected, f"Missing states: {expected - seen}"


def test_nan_input_returns_nan(rsm):
    """NaN in any required column -> all outputs NaN."""
    r = rsm.step(_make_tick(vol_returns_5m_last=float("nan")))
    assert np.isnan(r["alg_rsm_regime"])
    assert np.isnan(r["alg_rsm_confidence"])
    assert np.isnan(r["alg_rsm_transition_risk"])
    assert np.isnan(r["alg_rsm_trade_allowed"])
