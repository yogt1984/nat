"""Tests for relative-value pairs strategy (A1)."""

import numpy as np
import pandas as pd
import pytest

from strategies.relative_value_pairs import (
    CELLS_PER_DAY,
    GRID_S,
    PairSpreadEstimator,
    backtest,
    fit_pairs,
    generate_positions,
    load_cost_scenarios,
)


def _stack(symbol_prices: dict[str, np.ndarray], t0="2026-05-01") -> pd.DataFrame:
    """Build a multi-symbol frame at 5-min cadence from price arrays."""
    frames = []
    for sym, px in symbol_prices.items():
        n = len(px)
        ts = (
            pd.Timestamp(t0, tz="UTC").value
            + np.arange(n, dtype=np.int64) * int(GRID_S * 1e9)
        )
        frames.append(
            pd.DataFrame(
                {"timestamp_ns": ts, "symbol": sym, "raw_midprice": px}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cointegrated_pair(
    days=20, *, beta=1.05, alpha=0.5, halflife_h=12.0, spread_vol=0.01, seed=11
):
    """B is a GBM; A = exp(alpha + beta*log B + s) with s an OU process."""
    rng = np.random.default_rng(seed)
    n = days * CELLS_PER_DAY
    log_b = np.log(50_000.0) + np.cumsum(rng.normal(0, 0.002, n))
    hl_cells = halflife_h * 3600.0 / GRID_S
    rho = float(np.exp(-np.log(2) / hl_cells))
    eta = spread_vol * np.sqrt(1 - rho**2)
    s = np.empty(n)
    s[0] = 0.0
    for t in range(1, n):
        s[t] = rho * s[t - 1] + rng.normal(0, eta)
    log_a = alpha + beta * log_b + s
    return _stack({"ETH": np.exp(log_a), "BTC": np.exp(log_b)}), rho


class TestEstimator:
    @pytest.fixture(scope="class")
    def fitted(self):
        df, _ = _cointegrated_pair()
        return df, PairSpreadEstimator("ETH", "BTC").fit(df)

    def test_recovers_hedge_ratio(self, fitted):
        _, est = fitted
        assert est.beta_ == pytest.approx(1.05, rel=0.05)

    def test_cointegration_detected(self, fitted):
        _, est = fitted
        assert est.df_stat_ < -3.37
        assert est.is_cointegrated

    def test_halflife_recovered(self, fitted):
        _, est = fitted
        # AR-on-residuals biases half-life down (documented) — wide band
        assert est.halflife_h_ == pytest.approx(12.0, rel=0.5)
        assert est.is_tradeable

    def test_independent_walks_rejected(self):
        rng = np.random.default_rng(0)
        n = 15 * CELLS_PER_DAY
        a = 3000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        b = 50_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        est = PairSpreadEstimator("ETH", "BTC").fit(_stack({"ETH": a, "BTC": b}))
        assert not est.is_cointegrated

    def test_refuses_short_data(self):
        df, _ = _cointegrated_pair(days=1)
        with pytest.raises(ValueError, match="2 days"):
            PairSpreadEstimator("ETH", "BTC").fit(df)

    def test_missing_symbol_raises(self):
        df, _ = _cointegrated_pair()
        with pytest.raises(ValueError, match="no rows"):
            PairSpreadEstimator("ETH", "XRP").fit(df)

    def test_roundtrip_dict(self, fitted):
        df, est = fitted
        clone = PairSpreadEstimator.from_dict(est.to_dict())
        a = est.transform(df)
        b = clone.transform(df)
        pd.testing.assert_frame_equal(a, b)


class TestPositions:
    def _signal(self, z_values, gaps=None):
        idx = pd.date_range("2026-05-01", periods=len(z_values), freq="5min")
        return pd.DataFrame(
            {
                "spread": np.zeros(len(z_values)),
                "z": z_values,
                "gap_boundary": gaps or [False] * len(z_values),
            },
            index=idx,
        )

    def test_entry_hold_exit_stop(self):
        z = [0.0, -2.5, -1.0, -0.4, 0.0, 2.5, 5.0, 0.3]
        pos = generate_positions(self._signal(z))
        assert list(pos) == [0, 1, 1, 0, 0, -1, 0, 0]

    def test_nan_z_flattens(self):
        z = [-2.5, np.nan, -3.0]
        pos = generate_positions(self._signal(z))
        assert list(pos) == [1, 0, 1]

    def test_no_entry_on_gap_boundary(self):
        z = [0.0, -3.0, -3.0]
        gaps = [False, True, False]
        pos = generate_positions(self._signal(z, gaps))
        assert list(pos) == [0, 0, 1]


class TestBacktest:
    def _signal_with_spread(self, spread):
        idx = pd.date_range("2026-05-01", periods=len(spread), freq="5min")
        return pd.DataFrame(
            {
                "spread": spread,
                "z": np.zeros(len(spread)),
                "gap_boundary": [False] * len(spread),
            },
            index=idx,
        )

    def test_pnl_is_lagged_no_lookahead(self):
        # spread jumps +100bps at t=2; a position entered at t=2 must NOT
        # capture that same-cell move
        signal = self._signal_with_spread([0.0, 0.0, 0.01, 0.01])
        positions = pd.Series([0, 0, 1, 1], index=signal.index, dtype=float)
        r = backtest(signal, positions, beta=1.0, cost_bps_oneway=0.0)
        assert r["total_gross_bps"] == pytest.approx(0.0)

    def test_captures_next_cell_move(self):
        signal = self._signal_with_spread([0.0, 0.0, 0.01, 0.01])
        positions = pd.Series([0, 1, 1, 0], index=signal.index, dtype=float)
        r = backtest(signal, positions, beta=1.0, cost_bps_oneway=0.0)
        assert r["total_gross_bps"] == pytest.approx(100.0)

    def test_costs_scale_with_beta_and_turnover(self):
        signal = self._signal_with_spread([0.0] * 4)
        positions = pd.Series([0, 1, 1, 0], index=signal.index, dtype=float)
        r = backtest(signal, positions, beta=1.5, cost_bps_oneway=2.0)
        # entry + exit = 2 changes x (1 + 1.5) legs x 2 bps = 10 bps
        assert r["total_cost_bps"] == pytest.approx(10.0)
        assert r["n_round_trips"] == 1

    def test_end_to_end_harvests_ou_reversion(self):
        df, _ = _cointegrated_pair(days=30, seed=23)
        ts = df["timestamp_ns"].to_numpy()
        split = int(ts.min() + 0.6 * (ts.max() - ts.min()))
        est = PairSpreadEstimator("ETH", "BTC").fit(df[df.timestamp_ns < split])
        signal = est.transform(df[df.timestamp_ns >= split])
        positions = generate_positions(signal)
        assert positions.abs().sum() > 0, "no trades on a strongly reverting spread"
        r = backtest(signal, positions, beta=est.beta_, cost_bps_oneway=0.0)
        assert r["total_net_bps"] > 0
        assert r["sharpe"] > 0


class TestHelpers:
    def test_fit_pairs_skips_failures(self):
        df, _ = _cointegrated_pair()
        ests = fit_pairs(df, [("ETH", "BTC"), ("SOL", "BTC")])
        assert list(ests) == ["ETH/BTC"]

    def test_cost_scenarios_loaded_from_config(self):
        costs = load_cost_scenarios()
        assert set(costs) == {"maker", "taker", "taker+slip"}
        assert costs["taker"] >= costs["maker"]
