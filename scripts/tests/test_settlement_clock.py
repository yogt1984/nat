"""Tests for settlement-clock features (F1)."""

import numpy as np
import pandas as pd
import pytest

from features.settlement_clock import (
    CYCLE_S,
    SETTLEMENT_CLOCK_FEATURES,
    compute_settlement_clock,
)


def _frame_at(utc_strings, funding=None, symbol="BTC"):
    ts = pd.to_datetime(utc_strings, utc=True, format="mixed").as_unit("ns")
    df = pd.DataFrame(
        {
            "timestamp_ns": ts.view("int64"),
            "symbol": symbol,
        }
    )
    if funding is not None:
        df["ctx_funding_rate"] = funding
    return df


class TestClockFeatures:
    def test_all_columns_present(self):
        df = _frame_at(["2026-06-11 07:59:00"], funding=[1e-5])
        out = compute_settlement_clock(df)
        for col in SETTLEMENT_CLOCK_FEATURES:
            assert col in out.columns

    def test_time_to_settlement_one_minute_before_mark(self):
        out = compute_settlement_clock(_frame_at(["2026-06-11 07:59:00"]))
        assert out["sc_tts_8h"].iloc[0] == pytest.approx(60.0)
        assert out["sc_tss_8h"].iloc[0] == pytest.approx(CYCLE_S - 60.0)

    def test_at_settlement_mark_both_zero(self):
        for mark in ["2026-06-11 00:00:00", "2026-06-11 08:00:00", "2026-06-11 16:00:00"]:
            out = compute_settlement_clock(_frame_at([mark]))
            assert out["sc_tts_8h"].iloc[0] == pytest.approx(0.0)
            assert out["sc_tss_8h"].iloc[0] == pytest.approx(0.0)

    def test_mid_cycle(self):
        out = compute_settlement_clock(_frame_at(["2026-06-11 12:00:00"]))
        assert out["sc_tss_8h"].iloc[0] == pytest.approx(4 * 3600.0)
        assert out["sc_cycle_frac"].iloc[0] == pytest.approx(0.5)

    def test_cycle_frac_bounds(self):
        times = pd.date_range("2026-06-09", "2026-06-12", freq="17min", tz="UTC")
        out = compute_settlement_clock(
            _frame_at([t.isoformat() for t in times])
        )
        assert (out["sc_cycle_frac"] >= 0).all()
        assert (out["sc_cycle_frac"] < 1).all()

    def test_harmonic_encodings_unit_norm(self):
        times = pd.date_range("2026-06-08", "2026-06-12", freq="73min", tz="UTC")
        out = compute_settlement_clock(_frame_at([t.isoformat() for t in times]))
        np.testing.assert_allclose(
            out["sc_hod_sin"] ** 2 + out["sc_hod_cos"] ** 2, 1.0, atol=1e-12
        )
        np.testing.assert_allclose(
            out["sc_dow_sin"] ** 2 + out["sc_dow_cos"] ** 2, 1.0, atol=1e-12
        )

    def test_weekend_flag(self):
        # 2026-06-13 is a Saturday, 2026-06-14 a Sunday, 2026-06-10 a Wednesday
        out = compute_settlement_clock(
            _frame_at(
                ["2026-06-13 10:00:00", "2026-06-14 23:00:00", "2026-06-10 10:00:00"]
            )
        )
        assert list(out["sc_weekend"]) == [1.0, 1.0, 0.0]

    def test_no_nan_in_clock_features(self):
        times = pd.date_range("2026-06-11", periods=500, freq="100ms", tz="UTC")
        out = compute_settlement_clock(_frame_at([t.isoformat() for t in times]))
        clock_cols = [c for c in SETTLEMENT_CLOCK_FEATURES if "funding" not in c]
        assert not out[clock_cols].isna().any().any()

    def test_missing_timestamp_raises(self):
        with pytest.raises(KeyError):
            compute_settlement_clock(pd.DataFrame({"x": [1]}))


class TestFundingTwa:
    def test_constant_rate_recovered(self):
        times = pd.date_range("2026-06-11", periods=2000, freq="1min", tz="UTC")
        df = _frame_at([t.isoformat() for t in times], funding=2.5e-5)
        out = compute_settlement_clock(df)
        valid = out["sc_funding_twa_8h"].dropna()
        assert len(valid) > 0
        np.testing.assert_allclose(valid, 2.5e-5, rtol=1e-9)

    def test_warmup_is_nan(self):
        times = pd.date_range("2026-06-11", periods=100, freq="1min", tz="UTC")
        df = _frame_at([t.isoformat() for t in times], funding=1e-5)
        out = compute_settlement_clock(df)
        # 100 minutes of data cannot half-fill an 8h window
        assert out["sc_funding_twa_8h"].isna().all()
        assert out["sc_funding_twa_24h"].isna().all()

    def test_multi_symbol_isolation(self):
        times = pd.date_range("2026-06-11", periods=1500, freq="1min", tz="UTC")
        iso = [t.isoformat() for t in times]
        btc = _frame_at(iso, funding=1e-5, symbol="BTC")
        eth = _frame_at(iso, funding=9e-5, symbol="ETH")
        df = (
            pd.concat([btc, eth])
            .sample(frac=1.0, random_state=7)  # shuffle rows + non-trivial index
        )
        out = compute_settlement_clock(df)
        btc_twa = out.loc[out["symbol"] == "BTC", "sc_funding_twa_8h"].dropna()
        eth_twa = out.loc[out["symbol"] == "ETH", "sc_funding_twa_8h"].dropna()
        np.testing.assert_allclose(btc_twa, 1e-5, rtol=1e-9)
        np.testing.assert_allclose(eth_twa, 9e-5, rtol=1e-9)

    def test_missing_funding_column_skips_gracefully(self):
        out = compute_settlement_clock(_frame_at(["2026-06-11 03:00:00"]))
        assert "sc_funding_twa_8h" not in out.columns
        assert "sc_tts_8h" in out.columns
