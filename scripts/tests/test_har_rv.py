"""Tests for HAR-RV components and forecast (F4)."""

import numpy as np
import pandas as pd
import pytest

from features.har_rv import (
    HAR_RV_FEATURES,
    HarRvEstimator,
    compute_har_rv,
    compute_rv_components,
    fit_per_symbol,
)


def _gbm(
    days=10.0,
    *,
    ann_vol=0.5,
    dt_s=10.0,
    seed=5,
    symbol="BTC",
    mid0=50_000.0,
    vol_schedule=None,
):
    """Geometric Brownian mid path sampled every dt_s seconds.

    vol_schedule: optional array of per-step annualized vols (overrides
    ann_vol) to create regime changes.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 86_400 / dt_s)
    vols = (
        np.full(n, ann_vol)
        if vol_schedule is None
        else np.asarray(vol_schedule, dtype=float)
    )
    step_vol = vols * np.sqrt(dt_s / (365.0 * 86_400))
    log_ret = rng.normal(0, 1, n) * step_vol
    mid = mid0 * np.exp(np.cumsum(log_ret))
    ts = (
        pd.Timestamp("2026-05-01", tz="UTC").value
        + (np.arange(n, dtype=np.int64) * int(dt_s * 1e9))
    )
    return pd.DataFrame(
        {"timestamp_ns": ts, "symbol": symbol, "raw_midprice": mid}
    )


class TestComponents:
    def test_recovers_known_volatility(self):
        df = _gbm(days=3.0, ann_vol=0.5)
        out = compute_rv_components(df)
        tail = out["rv_vol_1d"].dropna().tail(1000)
        assert len(tail) > 0
        assert tail.mean() == pytest.approx(0.5, rel=0.15)

    def test_ratio_detects_vol_expansion(self):
        # 6 quiet days at 30%, then 1 loud day at 120%
        n_per_day = int(86_400 / 10)
        schedule = np.concatenate(
            [np.full(6 * n_per_day, 0.3), np.full(n_per_day, 1.2)]
        )
        df = _gbm(days=7.0, vol_schedule=schedule, seed=9)
        out = compute_rv_components(df)
        assert out["rv_ratio_1d_1w"].dropna().iloc[-1] > 1.5

    def test_causality_future_spike_invisible(self):
        df = _gbm(days=2.0, ann_vol=0.4, seed=13)
        spiked = df.copy()
        spiked.iloc[-500:, spiked.columns.get_loc("raw_midprice")] *= np.exp(
            np.linspace(0, 0.05, 500)
        )
        a = compute_rv_components(df)
        b = compute_rv_components(spiked)
        # features 10k ticks before the spike must be identical
        idx = len(df) - 12_000
        assert a["rv_vol_1d"].iloc[idx] == b["rv_vol_1d"].iloc[idx]

    def test_insufficient_coverage_is_nan(self):
        df = _gbm(days=0.1, ann_vol=0.5)  # 2.4h cannot quarter-fill 24h
        out = compute_rv_components(df)
        assert out["rv_vol_1d"].isna().all()
        assert out["rv_vol_1m"].isna().all()

    def test_no_returns_across_gaps(self):
        # a huge price jump across a 6h gap must not contaminate RV
        df = _gbm(days=2.0, ann_vol=0.3, seed=21)
        half = len(df) // 2
        df.loc[half:, "timestamp_ns"] += int(6 * 3600 * 1e9)
        df.loc[half:, "raw_midprice"] *= 1.5  # 50% jump inside the gap
        out = compute_rv_components(df)
        post = out["rv_vol_1d"].dropna().iloc[-1]
        # one 5-min cell with a 40% return would push annualized vol > 10;
        # with the gap excluded it stays near 0.3
        assert post < 1.0

    def test_multi_symbol_isolation(self):
        btc = _gbm(days=3.0, ann_vol=0.3, symbol="BTC", seed=1)
        sol = _gbm(days=3.0, ann_vol=1.0, symbol="SOL", seed=2)
        out = compute_rv_components(pd.concat([btc, sol], ignore_index=True))
        btc_vol = out.loc[out.symbol == "BTC", "rv_vol_1d"].dropna().tail(500).mean()
        sol_vol = out.loc[out.symbol == "SOL", "rv_vol_1d"].dropna().tail(500).mean()
        assert btc_vol == pytest.approx(0.3, rel=0.2)
        assert sol_vol == pytest.approx(1.0, rel=0.2)


class TestHarEstimator:
    def test_fit_and_forecast_on_persistent_vol(self):
        # slowly varying vol regime -> HAR forecast should track realized
        n_per_day = int(86_400 / 10)
        days = 12
        base = np.repeat(
            0.3 + 0.4 * (1 + np.sin(np.linspace(0, 3 * np.pi, days))) / 2,
            n_per_day,
        )
        df = _gbm(days=float(days), vol_schedule=base, seed=3)
        split = int(len(df) * 0.7)
        est = HarRvEstimator().fit(df.iloc[:split])
        assert "1d" in est.terms_
        out = compute_har_rv(df.iloc[split:].reset_index(drop=True), {"BTC": est})
        fcst = out["rv_har_fcst_1d"].dropna()
        assert len(fcst) > 0
        assert ((fcst > 0.05) & (fcst < 3.0)).all()

    def test_short_history_drops_monthly_term(self):
        df = _gbm(days=4.0, ann_vol=0.5, seed=4)
        est = HarRvEstimator().fit(df)
        assert "1m" not in est.terms_

    def test_refuses_multi_symbol(self):
        df = pd.concat(
            [_gbm(days=1.0, symbol="BTC"), _gbm(days=1.0, symbol="ETH")]
        )
        with pytest.raises(ValueError, match="single symbol"):
            HarRvEstimator().fit(df)

    def test_refuses_insufficient_data(self):
        with pytest.raises(ValueError, match="insufficient"):
            HarRvEstimator().fit(_gbm(days=0.5))

    def test_roundtrip_dict(self):
        est = HarRvEstimator().fit(_gbm(days=4.0, seed=6))
        clone = HarRvEstimator.from_dict(est.to_dict())
        df = compute_rv_components(_gbm(days=3.0, seed=7))
        sample = df.dropna(subset=["rv_vol_1d", "rv_vol_1w"]).tail(100)
        np.testing.assert_allclose(
            clone.forecast(sample), est.forecast(sample)
        )


class TestComputeHarRv:
    def test_all_columns_present(self):
        df = _gbm(days=4.0, seed=8)
        out = compute_har_rv(df)
        for col in HAR_RV_FEATURES:
            assert col in out.columns
        assert out["rv_har_fcst_1d"].notna().any()

    def test_fit_per_symbol(self):
        df = pd.concat(
            [
                _gbm(days=4.0, ann_vol=0.3, symbol="BTC", seed=1),
                _gbm(days=4.0, ann_vol=0.9, symbol="ETH", seed=2),
            ],
            ignore_index=True,
        )
        ests = fit_per_symbol(df)
        assert set(ests) == {"BTC", "ETH"}
        out = compute_har_rv(df, ests)
        btc = out.loc[out.symbol == "BTC", "rv_har_fcst_1d"].dropna().mean()
        eth = out.loc[out.symbol == "ETH", "rv_har_fcst_1d"].dropna().mean()
        assert eth > 2 * btc
