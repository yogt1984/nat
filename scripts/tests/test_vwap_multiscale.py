"""VW-1 — multi-scale VWAP from bucketed accumulators.

Two tests carry this unit.

**Arithmetic must be exact.** Every column downstream is a ratio of these sums, so a
planted trade stream with hand-computable VWAPs has to come back to floating-point
tolerance, not approximately.

**Bucket equivalence is the claim Phase B rests on.** The whole design — 1-minute
accumulators instead of raw scans — exists because six `trades_in_window` scans per symbol
at 10 Hz would blow the ingestor's 80 ms/tick p99 budget. If the bucketed value ever drifts
from a brute-force scan over the same trades, the Rust port inherits the drift and no test
downstream would catch it. So it is asserted directly, on random streams.

The rest guards the ways a VWAP silently lies: a partial window reported as if it were
full, a feed gap papered over with whatever trades survived, a minute with no trades
forward-filling the last price, and the sign convention — the shipped
`flow_vwap_deviation` is INVERTED, and a test pins both conventions so that lives in code
rather than in prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from features.vwap_multiscale import (  # noqa: E402
    WINDOWS, VwapRing, compute_multiscale_vwap,
)

MIN_NS = 60_000_000_000


def _trades(rows) -> pd.DataFrame:
    """rows = [(minute_offset, price, size), ...] -> a trade frame."""
    return pd.DataFrame({
        "timestamp_ns": [int(m * MIN_NS) for m, _, _ in rows],
        "symbol": "BTC",
        "price": [float(p) for _, p, _ in rows],
        "size": [float(s) for _, _, s in rows],
        "is_buy": True,
    })


def _brute_vwap(df: pd.DataFrame, end_ns: int, minutes: int) -> float:
    """Reference: scan every trade in the window. Deliberately naive.

    Minute-ALIGNED, matching the ring: the last `minutes` whole minute buckets, not the
    last `minutes*60` seconds ending at the newest trade. Bucketing cannot produce a
    trailing-exact window — that would need the raw scan the design exists to avoid — so
    the reference asserts the semantics the implementation actually claims.
    """
    end_min = end_ns // MIN_NS
    first_min = end_min - minutes + 1
    mins = df["timestamp_ns"] // MIN_NS
    sub = df[(mins >= first_min) & (mins <= end_min)]
    vol = sub["size"].sum()
    return float((sub["price"] * sub["size"]).sum() / vol) if vol > 0 else float("nan")


# ── 1. arithmetic ────────────────────────────────────────────────────────────────
class TestArithmetic:
    def test_hand_computed_vwap(self):
        """(100*1 + 200*3) / (1+3) = 175."""
        ring = VwapRing(max_minutes=10)
        ring.add(0 * MIN_NS, 100.0, 1.0)
        ring.add(1 * MIN_NS, 200.0, 3.0)
        # asks for 2 minutes because only 2 minutes have been observed — a 5-minute
        # window here is deliberately NaN (see TestWarmupBoundary)
        assert ring.vwap(2) == pytest.approx(175.0)

    def test_size_weighting_not_trade_counting(self):
        """One big trade must outweigh many small ones — this is VWAP, not TWAP."""
        ring = VwapRing(max_minutes=10)
        for _ in range(9):
            ring.add(0, 100.0, 0.1)
        ring.add(0, 200.0, 9.1)
        assert ring.vwap(1) > 180.0

    def test_window_excludes_older_trades(self):
        ring = VwapRing(max_minutes=60)
        ring.add(0, 100.0, 1.0)
        ring.add(9 * MIN_NS, 200.0, 1.0)
        assert ring.vwap(5) == pytest.approx(200.0)     # only the recent one
        assert ring.vwap(10) == pytest.approx(150.0)    # both


# ── 2. the claim Phase B rests on ────────────────────────────────────────────────
class TestBucketEquivalence:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("minutes", [5, 15, 60])
    def test_bucketed_equals_brute_force_scan(self, seed, minutes):
        """If these ever diverge, the Rust port inherits the drift silently."""
        rng = np.random.default_rng(seed)
        n = 4000
        ts = np.sort(rng.integers(0, 180 * MIN_NS, size=n))
        df = pd.DataFrame({"timestamp_ns": ts, "symbol": "BTC",
                           "price": 100 + rng.normal(scale=5, size=n),
                           "size": rng.uniform(0.01, 5, size=n), "is_buy": True})
        ring = VwapRing(max_minutes=720)
        for t, pr, sz in zip(df.timestamp_ns, df.price, df["size"]):
            ring.add(int(t), float(pr), float(sz))
        end = int(df.timestamp_ns.iloc[-1])
        got, want = ring.vwap(minutes), _brute_vwap(df, end, minutes)
        assert got == pytest.approx(want, rel=1e-9), f"{got} vs {want}"

    def test_boundary_trade_counted_once(self):
        """A trade exactly on a bucket edge belongs to exactly one bucket."""
        ring = VwapRing(max_minutes=10)
        ring.add(5 * MIN_NS, 100.0, 2.0)
        assert ring.total_volume(10) == pytest.approx(2.0)


# ── 3. the ways a VWAP silently lies ─────────────────────────────────────────────
class TestHonestNaN:
    def test_a_partial_window_is_nan_not_a_short_vwap(self):
        """10 minutes of data must not produce a '1h VWAP' — that is the number that
        looks fine in a column and is wrong in a backtest."""
        ring = VwapRing(max_minutes=720)
        for m in range(10):
            ring.add(m * MIN_NS, 100.0, 1.0)
        assert np.isfinite(ring.vwap(5))
        assert np.isnan(ring.vwap(60)), "partial window reported as full"

    def test_a_feed_gap_yields_nan_for_spanning_windows(self):
        """§7 guarantees gaps. A 6h hole must not produce a 6h VWAP from the survivors."""
        ring = VwapRing(max_minutes=720, max_gap_minutes=30)
        for m in range(60):
            ring.add(m * MIN_NS, 100.0, 1.0)
        ring.add((60 + 360) * MIN_NS, 200.0, 1.0)       # 6-hour hole
        assert np.isnan(ring.vwap(360)), "VWAP computed across a 6h gap"

    def test_an_empty_minute_is_not_forward_filled(self):
        """No trades in a minute contributes 0 notional AND 0 volume — never the last
        price carried forward, which would invent volume that did not trade."""
        ring = VwapRing(max_minutes=10)
        ring.add(0, 100.0, 1.0)
        ring.add(3 * MIN_NS, 100.0, 1.0)
        assert ring.total_volume(5) == pytest.approx(2.0)

    def test_no_trades_at_all_is_nan_not_zero(self):
        assert np.isnan(VwapRing(max_minutes=10).vwap(5))

    def test_zero_volume_does_not_divide_by_zero(self):
        ring = VwapRing(max_minutes=10)
        ring.add(0, 100.0, 0.0)
        assert np.isnan(ring.vwap(5))


class TestWarmupBoundary:
    """The warm-up rule is load-bearing, so pin it exactly rather than incidentally.

    Five of this file's tests failed on first write by asking for a 5-minute VWAP after
    two minutes of trades. The implementation was right and the tests were wrong — which
    is the correct direction for that disagreement, and worth an explicit boundary test
    so the rule cannot be quietly relaxed later to make a test pass."""

    def test_nan_at_one_minute_short_finite_exactly_at_the_span(self):
        ring = VwapRing(max_minutes=60)
        for m in range(4):
            ring.add(m * MIN_NS, 100.0, 1.0)
        assert np.isnan(ring.vwap(5)), "4 minutes observed must not yield a 5m VWAP"
        ring.add(4 * MIN_NS, 100.0, 1.0)
        assert np.isfinite(ring.vwap(5)), "5 minutes observed must yield a 5m VWAP"


# ── 4. the sign convention, pinned in a test ─────────────────────────────────────
class TestSignConvention:
    def test_price_above_vwap_gives_positive_deviation(self):
        df = _trades([(0, 100.0, 1.0), (1, 100.0, 1.0)])
        out = compute_multiscale_vwap(df, windows={"2m": 2}, mark_price=110.0)
        assert out["vwap_dev_2m"].iloc[-1] > 0

    def test_it_is_the_opposite_of_the_shipped_column(self):
        """`flow_vwap_deviation` is (vwap - price)/price — INVERTED. §1's -0.29 reads
        'price below VWAP predicts further decline' only because of that. The shipped
        column is deliberately left alone; this test is where the discrepancy lives."""
        df = _trades([(0, 100.0, 1.0)])
        out = compute_multiscale_vwap(df, windows={"1m": 1}, mark_price=110.0)
        ours = out["vwap_dev_1m"].iloc[-1]
        legacy = (100.0 - 110.0) / 110.0
        assert ours > 0 and legacy < 0
        assert np.sign(ours) == -np.sign(legacy)

    def test_deviation_is_normalised_by_vwap_not_price(self):
        df = _trades([(0, 100.0, 1.0)])
        out = compute_multiscale_vwap(df, windows={"1m": 1}, mark_price=110.0)
        assert out["vwap_dev_1m"].iloc[-1] == pytest.approx((110.0 - 100.0) / 100.0)


# ── 5. the frame-level transform ─────────────────────────────────────────────────
class TestTransform:
    @staticmethod
    def _stream(minutes=180, per_min=5, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for m in range(minutes):
            for _ in range(per_min):
                rows.append((m + rng.uniform(0, 0.99), 100 + rng.normal(), rng.uniform(.1, 2)))
        return _trades(rows)

    def test_emits_every_declared_column(self):
        out = compute_multiscale_vwap(self._stream())
        for w in WINDOWS:
            for pre in ("vwap_", "vwap_dev_", "vwap_dev_z_"):
                assert f"{pre}{w}" in out.columns

    def test_row_count_and_index_preserved(self):
        df = self._stream()
        out = compute_multiscale_vwap(df)
        assert len(out) == len(df)
        pd.testing.assert_index_equal(out.index, df.index)

    def test_input_is_not_mutated(self):
        df = self._stream()
        before = df.copy()
        compute_multiscale_vwap(df)
        pd.testing.assert_frame_equal(df, before)

    def test_causal_a_later_trade_cannot_change_an_earlier_row(self):
        """The decisive causality check: appending future trades must leave every
        earlier row byte-identical."""
        df = self._stream(minutes=120)
        base = compute_multiscale_vwap(df)
        extended = compute_multiscale_vwap(pd.concat([df, self._stream(60, seed=9)],
                                                     ignore_index=True))
        cols = [c for c in base.columns if c.startswith("vwap_")]
        np.testing.assert_allclose(base[cols].to_numpy()[:len(df)],
                                   extended[cols].to_numpy()[:len(df)],
                                   rtol=0, atol=0, equal_nan=True)

    def test_z_score_is_finite_where_the_deviation_varies(self):
        out = compute_multiscale_vwap(self._stream(minutes=240))
        z = out["vwap_dev_z_5m"].to_numpy()
        assert np.isfinite(z).sum() > 100

    def test_slow_windows_are_nan_until_warm(self):
        out = compute_multiscale_vwap(self._stream(minutes=30))
        assert out["vwap_12h"].isna().all(), "12h VWAP from 30 minutes of trades"

    def test_multi_symbol_frames_do_not_mix(self):
        a = self._stream(minutes=60, seed=1)
        b = self._stream(minutes=60, seed=2); b["symbol"] = "ETH"
        b["price"] = b["price"] + 1000.0
        out = compute_multiscale_vwap(pd.concat([a, b], ignore_index=True))
        eth = out.loc[out["symbol"] == "ETH", "vwap_5m"].dropna()
        assert (eth > 500).all(), "BTC trades leaked into ETH's VWAP"

    def test_empty_frame_returns_empty(self):
        out = compute_multiscale_vwap(_trades([]).iloc[:0])
        assert len(out) == 0

    def test_determinism(self):
        df = self._stream()
        a, b = compute_multiscale_vwap(df), compute_multiscale_vwap(df)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
