"""PROC-19 — the `candles` data level and multi-symbol loading.

The three `xs_*` processes are cross-sectional: they rank the whole universe at each
rebalance, so they need many symbols in one frame. The runner today loads ONE symbol of
tick parquet from `data/features/`, which is the wrong shape and the wrong source.
`specs/maker_system.md` §7 flags this as "a small framework extension, flagged as its own
task before implementation" — this is that task.

The property that decides the unit is **alignment across unequal histories**. The XS-1
archive has 177 pairs where most reach 90 days and some are recent listings (CASHCAT 27 d,
GRAM 36 d). Two obvious implementations are both wrong:

  * an **inner join** on timestamp truncates all 177 pairs to the newest listing — 90 days
    of history for 175 pairs destroyed by one recent coin, and silently;
  * an **outer join with forward/backward fill** invents prices for pairs that did not
    trade yet, which is a lookahead that would make any rank-IC study meaningless.

The right answer is a long frame where absence is absence: each timestamp carries whatever
pairs existed then, and a cross-sectional process ranks over that. Both wrong shapes are
asserted against here.

Everything else — price-column resolution (ticks use `raw_midprice`, candles use `close`),
missing files reported rather than silently dropped, interval validation — exists so the
loader cannot return something that merely looks like a panel.
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

from processes.candles import (  # noqa: E402
    CANDLE_PRICE_COL,
    available_candle_symbols,
    load_candles,
)


def _write(tmp_path: Path, symbol: str, interval: str, n: int,
           start="2026-05-01", freq="1h") -> Path:
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n),
        "volume": np.arange(n, dtype=float),
    })
    p = tmp_path / f"{symbol}_{interval}.parquet"
    df.to_parquet(p, index=False)
    return p


# ── alignment: unequal histories must neither truncate nor be invented ────

def test_a_late_listing_does_not_truncate_the_panel(tmp_path):
    """The inner-join failure: one 27-day coin must not cost 175 pairs their history."""
    _write(tmp_path, "BTC", "1h", 500)
    _write(tmp_path, "ETH", "1h", 500)
    _write(tmp_path, "NEW", "1h", 50, start="2026-05-15")   # listed late

    df = load_candles(["BTC", "ETH", "NEW"], "1h", tmp_path)

    assert len(df[df.symbol == "BTC"]) == 500, "BTC was truncated to the late listing"
    assert len(df[df.symbol == "ETH"]) == 500
    assert len(df[df.symbol == "NEW"]) == 50


def test_a_late_listing_is_not_backfilled(tmp_path):
    """The outer-join-with-fill failure: a pair must not have prices before it existed."""
    _write(tmp_path, "BTC", "1h", 500)
    _write(tmp_path, "NEW", "1h", 50, start="2026-05-15")

    df = load_candles(["BTC", "NEW"], "1h", tmp_path)
    new_first = df[df.symbol == "NEW"].timestamp.min()
    btc_first = df[df.symbol == "BTC"].timestamp.min()

    assert new_first > btc_first
    early = df[(df.symbol == "NEW") & (df.timestamp < new_first)]
    assert early.empty, "NEW has rows before it was listed"


def test_frame_is_long_with_one_row_per_symbol_timestamp(tmp_path):
    _write(tmp_path, "BTC", "1h", 10)
    _write(tmp_path, "ETH", "1h", 10)

    df = load_candles(["BTC", "ETH"], "1h", tmp_path)
    assert set(df.columns) >= {"timestamp", "symbol", "open", "high", "low",
                               "close", "volume"}
    assert len(df) == 20
    assert not df.duplicated(subset=["symbol", "timestamp"]).any()


def test_cross_section_at_a_timestamp_contains_only_live_pairs(tmp_path):
    """What a cross-sectional process actually consumes: the universe as it was, then."""
    _write(tmp_path, "BTC", "1h", 500)
    _write(tmp_path, "NEW", "1h", 50, start="2026-05-15")

    df = load_candles(["BTC", "NEW"], "1h", tmp_path)
    early_ts = df.timestamp.min()
    late_ts = df[df.symbol == "NEW"].timestamp.max()

    assert set(df[df.timestamp == early_ts].symbol) == {"BTC"}
    assert set(df[df.timestamp == late_ts].symbol) == {"BTC", "NEW"}


def test_rows_are_sorted_by_timestamp_then_symbol(tmp_path):
    _write(tmp_path, "ETH", "1h", 5)
    _write(tmp_path, "BTC", "1h", 5)
    df = load_candles(["ETH", "BTC"], "1h", tmp_path)
    assert df.equals(df.sort_values(["timestamp", "symbol"]).reset_index(drop=True))


# ── the loader cannot silently return less than asked ─────────────────────

def test_missing_symbol_is_reported_not_silently_dropped(tmp_path):
    _write(tmp_path, "BTC", "1h", 10)
    df, report = load_candles(["BTC", "GHOST"], "1h", tmp_path, return_report=True)

    assert set(df.symbol) == {"BTC"}
    assert report["missing"] == ["GHOST"], (
        "a symbol with no file must be named — silently returning fewer pairs biases "
        "every cross-sectional rank computed afterwards"
    )
    assert report["loaded"] == ["BTC"]


def test_all_symbols_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candles(["NOPE"], "1h", tmp_path)


def test_unknown_interval_raises(tmp_path):
    _write(tmp_path, "BTC", "1h", 10)
    with pytest.raises(ValueError):
        load_candles(["BTC"], "3s", tmp_path)


def test_date_range_filters_without_dropping_symbols(tmp_path):
    _write(tmp_path, "BTC", "1h", 500)
    _write(tmp_path, "ETH", "1h", 500)
    df = load_candles(["BTC", "ETH"], "1h", tmp_path,
                      start_date="2026-05-10", end_date="2026-05-12")
    assert set(df.symbol) == {"BTC", "ETH"}
    assert df.timestamp.min() >= pd.Timestamp("2026-05-10", tz="UTC")
    assert df.timestamp.max() < pd.Timestamp("2026-05-13", tz="UTC")


# ── discovery + price column ──────────────────────────────────────────────

def test_available_symbols_enumerates_the_archive(tmp_path):
    _write(tmp_path, "BTC", "1h", 5)
    _write(tmp_path, "ETH", "1h", 5)
    _write(tmp_path, "SOL", "15m", 5)

    assert available_candle_symbols(tmp_path, "1h") == ["BTC", "ETH"]
    assert available_candle_symbols(tmp_path, "15m") == ["SOL"]
    assert available_candle_symbols(tmp_path, "4h") == []


def test_price_column_for_candles_is_close_not_the_tick_midprice():
    """Ticks use `raw_midprice`; candles have no such column.

    If the runner carried the tick default into a candle run, every price lookup would
    be NaN and the process would produce empty findings rather than an error.
    """
    assert CANDLE_PRICE_COL == "close"


def test_loaded_frame_exposes_the_declared_price_column(tmp_path):
    _write(tmp_path, "BTC", "1h", 10)
    df = load_candles(["BTC"], "1h", tmp_path)
    assert CANDLE_PRICE_COL in df.columns
    assert df[CANDLE_PRICE_COL].notna().all()


def test_columns_subset_still_includes_keys(tmp_path):
    _write(tmp_path, "BTC", "1h", 10)
    df = load_candles(["BTC"], "1h", tmp_path, columns=["close"])
    assert {"timestamp", "symbol", "close"} <= set(df.columns)
    assert "volume" not in df.columns


# ── runner wiring: a candles process gets the universe, not one symbol ────

def test_runner_routes_a_candles_process_to_the_archive(tmp_path, monkeypatch):
    """A process declaring data_level='candles' must be fed a multi-symbol long frame
    with the candle price column — not one symbol of tick parquet."""
    from processes.base import EvaluationProcess, ProcessResult
    from processes import registry
    import processes.runner as runner

    _write(tmp_path, "BTC", "1h", 60)
    _write(tmp_path, "ETH", "1h", 60)
    seen = {}

    class _Probe(EvaluationProcess):
        name = "xs_probe"
        data_level = "candles"

        def required_columns(self):
            return ["close"]

        def evaluate(self, frame, ctx):
            seen["symbols"] = sorted(frame["symbol"].unique())
            seen["price_col"] = ctx.price_col
            seen["ctx_symbols"] = list(ctx.symbols)
            seen["rows"] = len(frame)
            return ProcessResult(run_id="probe", process="xs_probe", kind="evaluation",
                                 symbol=ctx.symbol, timeframe=ctx.timeframe, params={})

    monkeypatch.setitem(registry._REGISTRY, "xs_probe", _Probe)

    runner.run_process("xs_probe", symbols=["BTC", "ETH"], interval="1h",
                       data_dir=tmp_path, save=False, db_path=None)

    assert seen["symbols"] == ["BTC", "ETH"], "the process saw only one symbol"
    assert seen["price_col"] == CANDLE_PRICE_COL
    assert seen["ctx_symbols"] == ["BTC", "ETH"]
    assert seen["rows"] == 120
