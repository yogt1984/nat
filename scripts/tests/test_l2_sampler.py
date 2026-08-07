"""XS-8 — the L2 sampler: measuring the one number B-5a turns on.

`B-5a` asks whether any perp's half-spread clears the +0.144 bps breakeven maker rate
measured in FINDINGS §4.11. Nothing in hand can answer it: candles carry no spread and
no depth, and the ingestor's book feed covers only the three symbols in `symbols.toml` —
the three *tightest* on the venue, which is the sampling bias §4.11 named as the reason
the maker line is unresolved rather than dead.

This unit samples `l2Book` over REST for the whole universe. Three properties carry it:

  1. **The half-spread arithmetic is exactly right.** It is compared against a fixed
     threshold to decide whether a whole research line lives, so a factor-of-two slip
     (half-spread vs full spread, bps vs fraction) would silently move the verdict.
     Pinned against hand-computed values, not golden output.

  2. **A degenerate book is never reported as a measurement.** Crossed and locked books
     occur in real feeds; an empty side occurs on illiquid pairs — the ones this unit
     exists to look at. Each must be flagged, never emitted as a spread.

  3. **One symbol's failure cannot end the sweep** — the XS-1 lesson, restated here
     because a sampler that dies at pair 12 of 177 silently biases the sample toward
     whatever sorts early.

Sampling is a *distribution* exercise, not a snapshot: a single book is an n=1 estimate
of a quantity that moves all day, which is the error PROC-20 corrected in LF7's priors
(n=4-31/cell). The tests therefore also pin that repeated sweeps accumulate rather than
overwrite.

Hermetic: no network. Every fetch is injected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data.fetch_l2 import (  # noqa: E402
    BookStatus,
    parse_l2_book,
    sample_universe,
    write_snapshot,
)


def _book(bids, asks, coin="TEST", t=1_786_000_000_000) -> dict:
    """Build an l2Book payload in the venue's shape: levels=[bids, asks], px/sz strings."""
    def side(levels):
        return [{"px": str(p), "sz": str(s), "n": int(n)} for p, s, n in levels]
    return {"coin": coin, "time": t, "levels": [side(bids), side(asks)]}


# ── 1. the arithmetic B-5a depends on ─────────────────────────────────────

def test_half_spread_bps_is_hand_checkable():
    """bid 100.0 / ask 100.2 -> mid 100.1, spread 0.2, half-spread 9.99001 bps."""
    row = parse_l2_book(_book([(100.0, 5, 3)], [(100.2, 4, 2)]))

    assert row["status"] is BookStatus.OK
    assert row["best_bid"] == 100.0
    assert row["best_ask"] == 100.2
    assert row["mid"] == pytest.approx(100.1)
    assert row["spread"] == pytest.approx(0.2)
    assert row["spread_bps"] == pytest.approx(0.2 / 100.1 * 1e4)
    assert row["half_spread_bps"] == pytest.approx(0.2 / 2 / 100.1 * 1e4)
    # The relation the whole study rests on — half is half, in bps of mid.
    assert row["half_spread_bps"] == pytest.approx(row["spread_bps"] / 2)


def test_a_tight_book_and_a_wide_book_differ_by_the_expected_factor():
    """Guards against a bps/fraction slip: a 30x wider book must read 30x wider."""
    tight = parse_l2_book(_book([(100.0, 1, 1)], [(100.01, 1, 1)]))
    wide = parse_l2_book(_book([(100.0, 1, 1)], [(100.30, 1, 1)]))
    assert wide["half_spread_bps"] / tight["half_spread_bps"] == pytest.approx(30, rel=0.01)


def test_touch_depth_and_notional_are_recorded():
    """XS-5's capacity floors need size at the touch, not just price."""
    row = parse_l2_book(_book([(100.0, 5, 3)], [(100.2, 4, 2)]))
    assert row["bid_sz_l1"] == 5.0
    assert row["ask_sz_l1"] == 4.0
    assert row["bid_n_l1"] == 3
    assert row["bid_notional_l1"] == pytest.approx(500.0)
    assert row["ask_notional_l1"] == pytest.approx(400.8)


def test_depth5_sums_five_levels_only():
    bids = [(100 - i, 1, 1) for i in range(8)]
    asks = [(101 + i, 2, 1) for i in range(8)]
    row = parse_l2_book(_book(bids, asks))
    assert row["bid_notional_5"] == pytest.approx(sum((100 - i) * 1 for i in range(5)))
    assert row["ask_notional_5"] == pytest.approx(sum((101 + i) * 2 for i in range(5)))
    assert row["n_bid_levels"] == 8


# ── 2. degenerate books are flagged, never measured ───────────────────────

def test_crossed_book_is_flagged_not_measured():
    """bid > ask is not a negative spread, it is a book you must not quote against."""
    row = parse_l2_book(_book([(100.5, 1, 1)], [(100.0, 1, 1)]))
    assert row["status"] is BookStatus.CROSSED
    assert row["half_spread_bps"] is None


def test_locked_book_is_flagged():
    """bid == ask: zero spread is real in the data and would read as free money."""
    row = parse_l2_book(_book([(100.0, 1, 1)], [(100.0, 1, 1)]))
    assert row["status"] is BookStatus.LOCKED
    assert row["half_spread_bps"] is None


def test_empty_side_is_flagged():
    """One-sided books happen on exactly the illiquid pairs this unit exists to measure."""
    assert parse_l2_book(_book([], [(100.0, 1, 1)]))["status"] is BookStatus.EMPTY
    assert parse_l2_book(_book([(100.0, 1, 1)], []))["status"] is BookStatus.EMPTY


def test_malformed_payload_raises_rather_than_returning_zeros():
    with pytest.raises((ValueError, KeyError, TypeError)):
        parse_l2_book({"coin": "X", "time": 1})
    with pytest.raises((ValueError, KeyError, TypeError)):
        parse_l2_book({"coin": "X", "time": 1, "levels": [[]]})     # one side only


def test_nonpositive_prices_are_rejected():
    row = parse_l2_book(_book([(0.0, 1, 1)], [(100.0, 1, 1)]))
    assert row["status"] is not BookStatus.OK


# ── 3. the sweep survives its own universe ────────────────────────────────

def test_one_failure_does_not_end_the_sweep():
    seen = []

    def fetch(symbol):
        seen.append(symbol)
        if symbol == "BAD":
            raise RuntimeError("HTTP 500")
        return _book([(100.0, 1, 1)], [(100.1, 1, 1)], coin=symbol)

    rows, report = sample_universe(["A", "BAD", "B", "C"], fetch_fn=fetch, delay=0)

    assert seen == ["A", "BAD", "B", "C"], "the sweep stopped early"
    assert {r["symbol"] for r in rows} == {"A", "B", "C"}
    assert report["failed"] == [{"symbol": "BAD", "error": "RuntimeError: HTTP 500"}]
    assert report["n_requested"] == 4


def test_degenerate_books_are_counted_separately_from_failures():
    def fetch(symbol):
        if symbol == "CROSSED":
            return _book([(100.5, 1, 1)], [(100.0, 1, 1)], coin=symbol)
        return _book([(100.0, 1, 1)], [(100.1, 1, 1)], coin=symbol)

    rows, report = sample_universe(["OK1", "CROSSED"], fetch_fn=fetch, delay=0)
    assert report["ok"] == 1
    assert report["degenerate"] == ["CROSSED"]
    assert report["failed"] == []
    assert len(rows) == 2, "a degenerate book is still recorded, just not as a measurement"


def test_rejects_symbols_that_are_not_plain_tickers():
    """Symbols reach the filesystem via the output path — same guard as XS-1."""
    rows, report = sample_universe(["../etc/passwd", "OK"],
                                   fetch_fn=lambda s: _book([(1, 1, 1)], [(2, 1, 1)], coin=s),
                                   delay=0)
    assert [r["symbol"] for r in rows] == ["OK"]
    assert report["rejected"] == ["../etc/passwd"]


# ── 4. samples accumulate — a snapshot is n=1 ─────────────────────────────

def test_repeated_sweeps_accumulate_rather_than_overwrite(tmp_path):
    """The unit's whole purpose is a distribution; overwriting would leave n=1 forever."""
    pd = pytest.importorskip("pandas")

    rows = [{"symbol": "BTC", "ts_ms": 1, "half_spread_bps": 0.08}]
    p1 = write_snapshot(rows, tmp_path, ts_ms=1_786_000_000_000)
    p2 = write_snapshot(rows, tmp_path, ts_ms=1_786_000_300_000)

    assert p1 != p2, "a second sweep overwrote the first"
    assert p1.parent == p2.parent, "same UTC day should share a directory"
    files = sorted(tmp_path.rglob("*.parquet"))
    assert len(files) == 2
    assert len(pd.concat([pd.read_parquet(f) for f in files])) == 2


def test_snapshots_are_partitioned_by_utc_day(tmp_path):
    p1 = write_snapshot([{"symbol": "A", "ts_ms": 1}], tmp_path, ts_ms=1_786_000_000_000)
    p2 = write_snapshot([{"symbol": "A", "ts_ms": 1}], tmp_path,
                        ts_ms=1_786_000_000_000 + 86_400_000)
    assert p1.parent != p2.parent
    assert p1.parent.name != p2.parent.name


def test_empty_rows_write_nothing(tmp_path):
    assert write_snapshot([], tmp_path, ts_ms=1_786_000_000_000) is None
    assert list(tmp_path.rglob("*.parquet")) == []
