"""Spectral process contracts on constructed tick-level signals:
known-period sinusoid recovered, OU half-life within 2x of analytic,
band-limited planted predictor flagged with persistence."""

import numpy as np
import pandas as pd
import pytest

from processes import get_process
from processes.base import ProcessContext
from processes.synthetic import make_ou_series

FS = 10.0  # spannung machinery is built around 10 Hz ticks
N = 20000  # 2000 s


def _ctx():
    return ProcessContext(
        symbol="SYN", timeframe="tick", price_col="raw_midprice",
        horizons={"1s": 10, "5s": 50},
        costs={"hyperliquid": {"round_trip_taker_bps": 0.02}},
    )


def _tick_frame(**features) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-5, size=N)))
    return pd.DataFrame({"raw_midprice": prices, **features})


@pytest.fixture(scope="module")
def basic_result():
    rng = np.random.default_rng(7)
    t = np.arange(N) / FS
    sinusoid = np.sin(2 * np.pi * 0.5 * t) + 0.3 * rng.normal(size=N)  # 2 s period
    ou = make_ou_series(n=N, theta=0.05, seed=7)  # half-life ln2/0.05 = 13.9 ticks = 1.39 s
    df = _tick_frame(feat_sine=sinusoid, feat_ou=ou)
    proc = get_process("spectral")
    return proc.evaluate(df, _ctx())


def test_sinusoid_dominant_period_recovered(basic_result):
    f = next(f for f in basic_result.findings if f.feature == "feat_sine")
    period = f.extras["dominant_period_s"]
    assert period is not None and 1.5 <= period <= 2.5, f.extras


def test_ou_halflife_within_2x_of_analytic(basic_result):
    f = next(f for f in basic_result.findings if f.feature == "feat_ou")
    analytic_s = (np.log(2) / 0.05) / FS  # 1.386 s
    assert analytic_s / 2 <= f.extras["ou_halflife_s"] <= analytic_s * 2, f.extras


def test_extras_complete(basic_result):
    for f in basic_result.findings:
        for key in ("hurst", "noise_color", "ou_halflife_s", "spectral_entropy",
                    "band_ics", "persistent_at_horizon"):
            assert key in f.extras, f"{f.feature} missing {key}"


def test_band_limited_planted_predictor_flagged():
    # Slow (ultra-low band) persistent driver: eps[t] = c*s[t] + noise, so the
    # forward return over 10-50 ticks integrates s and correlates strongly.
    rng = np.random.default_rng(21)
    s = make_ou_series(n=N, theta=0.005, seed=11)       # half-life ~13.9 s
    s = s / np.std(s)
    eps = 4e-5 * s + 1e-5 * rng.normal(size=N)
    prices = 100.0 * np.exp(np.cumsum(eps))
    df = pd.DataFrame({
        "raw_midprice": prices,
        "feat_driver": s,
        "feat_noise": rng.normal(size=N),
    })

    result = get_process("spectral").evaluate(df, _ctx())
    driver = next(f for f in result.findings if f.feature == "feat_driver")
    assert abs(driver.value) >= 0.05, driver.extras
    assert driver.extras["persistent_at_horizon"]
    assert driver.informative

    noise = next(f for f in result.findings if f.feature == "feat_noise")
    assert not noise.informative


def test_nan_column_skipped_not_crashed():
    df = _tick_frame(feat_ok=np.random.default_rng(1).normal(size=N))
    df["feat_dead"] = np.nan
    result = get_process("spectral").evaluate(df, _ctx())
    reasons = {s["feature"]: s["reason"] for s in result.features_skipped}
    assert reasons["feat_dead"] == "all_nan"
    assert [f.feature for f in result.findings] == ["feat_ok"]
