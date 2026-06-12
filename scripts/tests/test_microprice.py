"""Tests for Stoikov microprice deviation features (F2)."""

import numpy as np
import pandas as pd
import pytest

from features.microprice import (
    MICROPRICE_FEATURES,
    MicropriceEstimator,
    compute_microprice,
    fit_per_symbol,
)


def _synthetic_book(
    n=60_000,
    *,
    informative=True,
    seed=42,
    symbol="BTC",
    mid0=50_000.0,
    tick=0.5,
    p_change=0.2,
):
    """100ms book snapshots where (if informative) high bid imbalance makes
    the next mid change more likely to be an uptick."""
    rng = np.random.default_rng(seed)
    imb = rng.uniform(-1, 1, n)  # symmetric imbalance, like imbalance_qty_l1
    mid = np.empty(n)
    mid[0] = mid0
    for t in range(1, n):
        if rng.random() < p_change:
            if informative:
                p_up = min(max(0.5 + 0.45 * imb[t - 1], 0.02), 0.98)
            else:
                p_up = 0.5
            mid[t] = mid[t - 1] + (tick if rng.random() < p_up else -tick)
        else:
            mid[t] = mid[t - 1]
    ts = (
        pd.Timestamp("2026-06-11", tz="UTC").value
        + np.arange(n, dtype=np.int64) * 100_000_000
    )
    return pd.DataFrame(
        {
            "timestamp_ns": ts,
            "symbol": symbol,
            "raw_midprice": mid,
            "raw_spread_bps": np.full(n, 0.16),
            "imbalance_qty_l1": imb,
        }
    )


@pytest.fixture(scope="module")
def informative_fit():
    df = _synthetic_book(informative=True)
    return df, MicropriceEstimator().fit(df)


class TestEstimator:
    def test_informative_chain_recovers_sign_and_monotonicity(self, informative_fit):
        _, est = informative_fit
        g = est.g_[:, 0]  # single spread bin (constant spread)
        n = len(g)
        # high-imbalance bins -> positive adjustment, low -> negative
        assert g[-1] > 0 and g[0] < 0
        # monotone non-decreasing in imbalance (allow tiny numeric wiggle)
        assert np.all(np.diff(g) > -1e-3)

    def test_antisymmetry_by_construction(self, informative_fit):
        _, est = informative_fit
        g = est.g_
        np.testing.assert_allclose(g, -g[::-1, :], atol=1e-9)

    def test_uninformative_chain_yields_near_zero_adjustment(self):
        flat = MicropriceEstimator().fit(_synthetic_book(informative=False, seed=7))
        info = MicropriceEstimator().fit(_synthetic_book(informative=True, seed=7))
        assert np.abs(flat.g_).max() < 0.2 * np.abs(info.g_).max()

    def test_adjustment_magnitude_is_sane(self, informative_fit):
        # a single future tick move on a 50k mid is 0.1 bps; the summed
        # multi-step expectation should stay within a few ticks
        _, est = informative_fit
        assert np.abs(est.g_).max() < 1.0

    def test_gap_steps_excluded(self):
        df = _synthetic_book(n=5000)
        # introduce a 1-hour gap mid-way; fit must not crash and stays finite
        df.loc[2500:, "timestamp_ns"] += int(3600 * 1e9)
        est = MicropriceEstimator().fit(df)
        assert np.isfinite(est.g_).all()

    def test_refuses_multi_symbol_frame(self):
        df = pd.concat(
            [_synthetic_book(n=2000), _synthetic_book(n=2000, symbol="ETH")]
        )
        with pytest.raises(ValueError, match="single symbol"):
            MicropriceEstimator().fit(df)

    def test_refuses_tiny_input(self):
        with pytest.raises(ValueError, match="1000"):
            MicropriceEstimator().fit(_synthetic_book(n=500))

    def test_roundtrip_dict(self, informative_fit):
        _, est = informative_fit
        clone = MicropriceEstimator.from_dict(est.to_dict())
        imb = np.array([-0.9, -0.3, 0.0, 0.4, 0.95])
        spread = np.full(5, 0.16)
        np.testing.assert_allclose(
            clone.adjustment_bps(imb, spread), est.adjustment_bps(imb, spread)
        )


class TestComputeMicroprice:
    def test_output_columns_and_wmid_formula(self, informative_fit):
        df, est = informative_fit
        out = compute_microprice(df.head(1000), {"BTC": est})
        for col in MICROPRICE_FEATURES:
            assert col in out.columns
        # weighted-mid deviation: (I - 0.5) * spread, I = (imb+1)/2
        expected = ((out["imbalance_qty_l1"] + 1) / 2 - 0.5) * out["raw_spread_bps"]
        np.testing.assert_allclose(out["mp_wmid_dev_bps"], expected)

    def test_nan_inputs_propagate(self, informative_fit):
        df, est = informative_fit
        df = df.head(100).copy()
        df.loc[10, "imbalance_qty_l1"] = np.nan
        df.loc[20, "raw_spread_bps"] = np.nan
        out = compute_microprice(df, {"BTC": est})
        assert np.isnan(out["mp_micro_adj_bps"].iloc[10])
        assert np.isnan(out["mp_micro_adj_bps"].iloc[20])
        assert out["mp_micro_adj_bps"].iloc[30] == pytest.approx(
            est.adjustment_bps(
                np.array([df["imbalance_qty_l1"].iloc[30]]), np.array([0.16])
            )[0]
        )

    def test_in_sample_fit_when_no_estimators_given(self):
        out = compute_microprice(_synthetic_book(n=20_000))
        assert out["mp_micro_adj_bps"].notna().mean() > 0.99

    def test_multi_symbol_isolation(self):
        btc = _synthetic_book(n=30_000, informative=True, symbol="BTC", seed=1)
        eth = _synthetic_book(n=30_000, informative=False, symbol="ETH", seed=2)
        df = pd.concat([btc, eth], ignore_index=True)
        ests = fit_per_symbol(df)
        out = compute_microprice(df, ests)
        btc_adj = out.loc[out.symbol == "BTC", "mp_micro_adj_bps"].abs().max()
        eth_adj = out.loc[out.symbol == "ETH", "mp_micro_adj_bps"].abs().max()
        assert btc_adj > 5 * eth_adj  # informative symbol gets real adjustments

    def test_oos_ic_positive_on_informative_chain(self):
        df = _synthetic_book(n=80_000, informative=True, seed=11)
        split = 50_000
        est = MicropriceEstimator().fit(df.iloc[:split])
        out = compute_microprice(df.iloc[split:].reset_index(drop=True), {"BTC": est})
        # the synthetic chain draws imbalance i.i.d. per step, so its
        # predictability spans exactly one step — assert there
        fwd = np.log(out["raw_midprice"].shift(-1) / out["raw_midprice"])
        mask = out["mp_micro_adj_bps"].notna() & fwd.notna()
        ic = out["mp_micro_adj_bps"][mask].corr(fwd[mask], method="spearman")
        assert ic > 0.15
