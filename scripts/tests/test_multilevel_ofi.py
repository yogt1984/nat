"""Tests for multi-band OFI features (F3)."""

import numpy as np
import pandas as pd
import pytest

from features.multilevel_ofi import (
    BANDS,
    MULTILEVEL_OFI_FEATURES,
    OFIEstimator,
    _band_window_sums,
    compute_multilevel_ofi,
    fit_per_symbol,
)


def _frame(mid, spread, imb, d5b, d5a, d10b=None, d10a=None, symbol="BTC"):
    n = len(mid)
    ts = (
        pd.Timestamp("2026-06-11", tz="UTC").value
        + np.arange(n, dtype=np.int64) * 100_000_000
    )
    return pd.DataFrame(
        {
            "timestamp_ns": ts,
            "symbol": symbol,
            "raw_midprice": np.asarray(mid, dtype=float),
            "raw_spread": np.asarray(spread, dtype=float),
            "imbalance_qty_l1": np.asarray(imb, dtype=float),
            "raw_bid_depth_5": np.asarray(d5b, dtype=float),
            "raw_ask_depth_5": np.asarray(d5a, dtype=float),
            "raw_bid_depth_10": np.asarray(d10b if d10b is not None else d5b),
            "raw_ask_depth_10": np.asarray(d10a if d10a is not None else d5a),
        }
    )


def _synthetic_market(n=60_000, *, informative=True, seed=3, symbol="BTC"):
    """Depth flows drive the next mid move (if informative)."""
    rng = np.random.default_rng(seed)
    d5b_chg = rng.normal(0, 5, n)
    d5a_chg = rng.normal(0, 5, n)
    d5b = 1000 + np.cumsum(d5b_chg).clip(-800, 800)
    d5a = 1000 + np.cumsum(d5a_chg).clip(-800, 800)
    d10b = d5b + 500 + np.cumsum(rng.normal(0, 2, n)).clip(-300, 300)
    d10a = d5a + 500 + np.cumsum(rng.normal(0, 2, n)).clip(-300, 300)
    flow = d5b_chg - d5a_chg
    imb = np.tanh(flow / 8.0 + rng.normal(0, 0.3, n)).clip(-0.999, 0.999)

    # the mid responds to *trailing 10s cumulative* flow, matching the
    # windowed OFI features under test
    kernel = np.ones(100)
    trailing_flow = np.convolve(flow, kernel)[: n]

    mid = np.empty(n)
    mid[0] = 50_000.0
    for t in range(1, n):
        if rng.random() < 0.2:
            p_up = (
                min(max(0.5 + 0.4 * np.tanh(trailing_flow[t - 1] / 60.0), 0.05), 0.95)
                if informative
                else 0.5
            )
            mid[t] = mid[t - 1] + (0.5 if rng.random() < p_up else -0.5)
        else:
            mid[t] = mid[t - 1]
    return _frame(mid, np.full(n, 1.0), imb, d5b, d5a, d10b, d10a, symbol=symbol)


class TestStepOfi:
    def test_l1_constant_prices_equals_delta_imbalance(self):
        # constant quotes: OFI_l1 = (qb_t - qb_{t-1}) - (qa_t - qa_{t-1}) = d(imb)
        df = _frame([100, 100], [0.02, 0.02], [0.0, 0.5], [10, 10], [10, 10])
        sums = _band_window_sums(df, max_step_s=1.0)
        assert np.isnan(sums["ofi_l1_10s"].iloc[0])
        assert sums["ofi_l1_10s"].iloc[1] == pytest.approx(0.5)

    def test_l1_price_uptick(self):
        # both quotes up one tick: e_b = qb_t, e_a = -qa_{t-1}
        # imb: 0.5 -> 0.0 gives qb_t = 0.5, qa_prev = 0.25 -> OFI = 0.75
        df = _frame([100, 100.5], [0.02, 0.02], [0.5, 0.0], [10, 10], [10, 10])
        sums = _band_window_sums(df, max_step_s=1.0)
        assert sums["ofi_l1_10s"].iloc[1] == pytest.approx(0.75)

    def test_l1_price_downtick(self):
        # mirror case: e_b = -qb_{t-1}, e_a = qa_t
        # imb: 0.5 -> 0.0 gives qb_prev = 0.75, qa_t = 0.5 -> OFI = -1.25
        df = _frame([100, 99.5], [0.02, 0.02], [0.5, 0.0], [10, 10], [10, 10])
        sums = _band_window_sums(df, max_step_s=1.0)
        assert sums["ofi_l1_10s"].iloc[1] == pytest.approx(-1.25)

    def test_depth_band_constant_prices(self):
        # 200 warm rows establish the depth normalization (~mean half-depth
        # 100), then bid band grows 20 and ask shrinks 10: raw OFI = 30
        n = 202
        d5b = np.full(n, 100.0)
        d5a = np.full(n, 100.0)
        d5b[-1], d5a[-1] = 120.0, 90.0
        df = _frame(
            np.full(n, 100.0), np.full(n, 0.02), np.zeros(n), d5b, d5a
        )
        sums = _band_window_sums(df, max_step_s=1.0)
        # normalized by trailing mean half-depth ~100
        assert sums["ofi_d5_10s"].iloc[-1] - sums["ofi_d5_10s"].iloc[-2] == (
            pytest.approx(0.30, rel=0.02)
        )

    def test_gap_step_is_nan(self):
        df = _frame([100, 100, 100], [0.02] * 3, [0.0, 0.2, 0.4], [10] * 3, [10] * 3)
        df.loc[2, "timestamp_ns"] += int(10e9)  # 10s gap before row 2
        sums = _band_window_sums(df, max_step_s=1.0)
        assert sums["ofi_l1_10s"].iloc[1] == pytest.approx(0.2)
        # row 2's step is excluded and its 10s window holds no valid steps
        assert np.isnan(sums["ofi_l1_10s"].iloc[2])


class TestEstimator:
    @pytest.fixture(scope="class")
    def fitted(self):
        df = _synthetic_market(informative=True)
        return df, OFIEstimator().fit(df)

    def test_weights_positive_and_l1_normalized(self, fitted):
        _, est = fitted
        assert np.abs(est.weights_).sum() == pytest.approx(1.0)
        assert (est.weights_ > 0).all()  # bands share a common flow driver

    def test_refuses_multi_symbol(self):
        df = pd.concat(
            [
                _synthetic_market(n=5000, symbol="BTC"),
                _synthetic_market(n=5000, symbol="ETH"),
            ]
        )
        with pytest.raises(ValueError, match="single symbol"):
            OFIEstimator().fit(df)

    def test_roundtrip_dict(self, fitted):
        _, est = fitted
        clone = OFIEstimator.from_dict(est.to_dict())
        np.testing.assert_allclose(clone.weights_, est.weights_)
        np.testing.assert_allclose(clone.scales_, est.scales_)


class TestComputeMultilevelOfi:
    def test_output_columns_present(self):
        df = _synthetic_market(n=20_000)
        out = compute_multilevel_ofi(df)
        for col in MULTILEVEL_OFI_FEATURES:
            assert col in out.columns
            assert out[col].notna().mean() > 0.9

    def test_oos_ic_positive_and_integration_helps(self):
        df = _synthetic_market(n=80_000, informative=True, seed=17)
        split = 50_000
        est = OFIEstimator().fit(df.iloc[:split])
        out = compute_multilevel_ofi(
            df.iloc[split:].reset_index(drop=True), {"BTC": est}
        )
        fwd = np.log(out["raw_midprice"].shift(-1) / out["raw_midprice"])

        def ic(col):
            m = out[col].notna() & fwd.notna()
            return out[col][m].corr(fwd[m], method="spearman")

        assert ic("ofi_int_10s") > 0.05
        # integrated should not be dominated by every single band
        band_ics = [ic(f"ofi_{band}_10s") for band in BANDS]
        assert ic("ofi_int_10s") >= 0.9 * max(band_ics)

    def test_multi_symbol_isolation(self):
        btc = _synthetic_market(n=30_000, informative=True, symbol="BTC", seed=1)
        eth = _synthetic_market(n=30_000, informative=False, symbol="ETH", seed=2)
        df = pd.concat([btc, eth], ignore_index=True)
        ests = fit_per_symbol(df)
        assert set(ests) == {"BTC", "ETH"}
        out = compute_multilevel_ofi(df, ests)
        for sym in ("BTC", "ETH"):
            cov = out.loc[out.symbol == sym, "ofi_int_1m"].notna().mean()
            assert cov > 0.9
