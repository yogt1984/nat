"""XS-1 — universe candle backfill. The tests are about what happens when it goes wrong.

Fetching one symbol is solved (`fetch_candles`). Fetching ~150 is a different unit, and
everything that makes it different is a failure mode:

  * **the list must come from the venue**, never a constant in the file — a hardcoded roster
    silently rots the moment a pair is listed or delisted, and the whole point of Class 3 is
    breadth that tracks the actual universe;
  * **a symbol name becomes a file path**, so anything that is not a plain ticker is rejected
    before it can touch the filesystem;
  * **one bad symbol must not lose the other 149.** A run that aborts on the first HTTP error
    after 40 minutes is worse than useless, so failures are caught, recorded, and reported;
  * **delisted pairs are excluded but recorded**, because "why is FOO missing" is the first
    question anyone asks of a coverage report;
  * **re-running must be safe** — the existing per-symbol incremental logic already dedups,
    and this unit must not defeat it by clobbering files.

Everything here runs offline: the info endpoint and the per-symbol fetcher are both injected,
so the suite never touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _meta(names_with_flags) -> dict:
    """Venue-shaped meta payload: {"universe": [{"name": ..., "isDelisted": ...}, ...]}."""
    return {"universe": [
        ({"name": n, "szDecimals": 2} if not delisted
         else {"name": n, "szDecimals": 2, "isDelisted": True})
        for n, delisted in names_with_flags]}


def _writing_fetch(n=5):
    """A stand-in for `fetch_candles` that honours its contract: it WRITES.

    `backfill_universe` deliberately does not write — the per-symbol incremental merge
    and dedup live in `fetch_candles`, and duplicating them in the sweep is how the two
    paths drift apart. So a double that only returns a frame cannot satisfy a file
    assertion, and shouldn't.
    """
    def fetch(symbol, interval="1m", start=None, days=90, output_dir=None, **kw):
        df = _candles(n)
        out = Path(output_dir) / f"{symbol}_{interval}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():                                  # mirror the incremental merge
            df = (pd.concat([pd.read_parquet(out), df], ignore_index=True)
                    .drop_duplicates(subset="timestamp")
                    .sort_values("timestamp").reset_index(drop=True))
        df.to_parquet(out, index=False)
        return df
    return fetch


def _candles(n=5) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })


# ── the universe comes from the venue ────────────────────────────────────────────
class TestUniverseEnumeration:
    def test_names_are_read_from_meta(self):
        from data.fetch_candles import fetch_universe
        got = fetch_universe(info_fn=lambda _req: _meta(
            [("BTC", False), ("ETH", False), ("kPEPE", False)]))
        assert got == ["BTC", "ETH", "kPEPE"]

    def test_delisted_are_excluded_but_recorded(self):
        from data.fetch_candles import fetch_universe
        names, excluded = fetch_universe(
            info_fn=lambda _req: _meta([("BTC", False), ("DEAD", True), ("ETH", False)]),
            return_excluded=True)
        assert names == ["BTC", "ETH"]
        assert excluded == ["DEAD"], "a dropped pair must be reported, not vanish"

    def test_delisted_can_be_opted_back_in(self):
        from data.fetch_candles import fetch_universe
        got = fetch_universe(info_fn=lambda _req: _meta([("BTC", False), ("DEAD", True)]),
                             include_delisted=True)
        assert got == ["BTC", "DEAD"]

    def test_the_roster_is_not_hardcoded_anywhere(self):
        """A constant list of coin names would rot on the next listing."""
        import re
        from data import fetch_candles
        src = Path(fetch_candles.__file__).read_text()
        # a literal list holding several tickers is the shape we forbid
        assert not re.search(r"=\s*\[\s*\"(BTC|ETH|SOL)\"\s*,\s*\"", src), \
            "universe appears hardcoded — it must be enumerated from meta"

    @pytest.mark.parametrize("payload", [{}, {"universe": None}, {"universe": [{}]},
                                         {"universe": [{"name": ""}]}, []])
    def test_malformed_meta_raises_rather_than_returning_junk(self, payload):
        from data.fetch_candles import fetch_universe
        with pytest.raises((ValueError, TypeError, KeyError)):
            fetch_universe(info_fn=lambda _req: payload)

    def test_the_request_actually_asks_for_meta(self):
        from data.fetch_candles import fetch_universe
        seen = {}

        def info(req):
            seen.update(req)
            return _meta([("BTC", False)])

        fetch_universe(info_fn=info)
        assert seen == {"type": "meta"}


# ── a symbol name becomes a file path ────────────────────────────────────────────
class TestUnsafeSymbolsNeverReachTheFilesystem:
    @pytest.mark.parametrize("bad", ["../../etc/passwd", "BTC/USD", "BTC USD", "",
                                     "A" * 64, "BTC;rm -rf /", ".", ".."])
    def test_rejected_before_any_write(self, bad, tmp_path):
        from data.fetch_candles import backfill_universe
        called = []

        def fetch(symbol, **kw):
            called.append(symbol)
            return _candles()

        report = backfill_universe(["BTC", bad], interval="1m", days=1,
                                   output_dir=tmp_path, fetch_fn=fetch, delay=0.0)
        assert called == ["BTC"], f"unsafe symbol {bad!r} was passed to the fetcher"
        assert any(bad == r["symbol"] for r in report["rejected"])
        assert not any(p.name.startswith(("..", ".")) for p in tmp_path.iterdir())


# ── one failure must not lose the rest ───────────────────────────────────────────
class TestPartialFailure:
    def test_a_failing_symbol_does_not_abort_the_run(self, tmp_path):
        from data.fetch_candles import backfill_universe

        def fetch(symbol, **kw):
            if symbol == "BOOM":
                raise RuntimeError("HTTP 500")
            return _candles()

        report = backfill_universe(["BTC", "BOOM", "ETH"], interval="1m", days=1,
                                   output_dir=tmp_path, fetch_fn=fetch, delay=0.0)
        assert report["ok"] == ["BTC", "ETH"]
        assert report["failed"][0]["symbol"] == "BOOM"
        assert "HTTP 500" in report["failed"][0]["error"]

    def test_an_empty_result_is_recorded_not_written(self, tmp_path):
        from data.fetch_candles import backfill_universe
        report = backfill_universe(["EMPTY"], interval="1m", days=1, output_dir=tmp_path,
                                   fetch_fn=lambda symbol, **kw: pd.DataFrame(), delay=0.0)
        assert report["empty"] == ["EMPTY"] and report["ok"] == []
        assert list(tmp_path.iterdir()) == [], "an empty result must not create a file"

    def test_every_symbol_is_accounted_for_exactly_once(self, tmp_path):
        from data.fetch_candles import backfill_universe

        def fetch(symbol, **kw):
            if symbol == "BOOM":
                raise RuntimeError("nope")
            if symbol == "EMPTY":
                return pd.DataFrame()
            return _candles()

        syms = ["BTC", "BOOM", "EMPTY", "ETH", "BTC/USD"]
        r = backfill_universe(syms, interval="1m", days=1, output_dir=tmp_path,
                              fetch_fn=fetch, delay=0.0)
        counted = (len(r["ok"]) + len(r["failed"]) + len(r["empty"]) + len(r["rejected"]))
        assert counted == len(syms), r
        assert r["n_requested"] == len(syms)


# ── rate limiting and re-runs ────────────────────────────────────────────────────
class TestPolitenessAndIdempotency:
    def test_a_delay_is_applied_between_symbols(self, tmp_path, monkeypatch):
        """~150 symbols back-to-back is how you get rate-limited off the venue."""
        from data import fetch_candles
        slept = []
        monkeypatch.setattr(fetch_candles.time, "sleep", lambda s: slept.append(s))
        fetch_candles.backfill_universe(
            ["BTC", "ETH", "SOL"], interval="1m", days=1, output_dir=tmp_path,
            fetch_fn=lambda symbol, **kw: _candles(), delay=0.3)
        assert sum(slept) >= 0.6, f"expected a pause between symbols, got {slept}"

    def test_rerunning_does_not_duplicate_or_clobber(self, tmp_path):
        from data.fetch_candles import backfill_universe
        calls = []
        writer = _writing_fetch()

        def fetch(symbol, **kw):
            calls.append(symbol)
            return writer(symbol, **kw)

        for _ in range(2):
            backfill_universe(["BTC"], interval="1m", days=1, output_dir=tmp_path,
                              fetch_fn=fetch, delay=0.0)
        out = tmp_path / "BTC_1m.parquet"
        df = pd.read_parquet(out)
        assert len(df) == 5 and df["timestamp"].is_unique
        assert calls == ["BTC", "BTC"], "the per-symbol incremental path must still run"

    def test_max_symbols_truncation_is_reported(self, tmp_path):
        """Silent truncation reads as 'we covered the universe' when we did not."""
        from data.fetch_candles import backfill_universe
        r = backfill_universe(["A", "B", "C", "D"], interval="1m", days=1,
                              output_dir=tmp_path, fetch_fn=lambda symbol, **kw: _candles(),
                              delay=0.0, max_symbols=2)
        assert len(r["ok"]) == 2
        assert r["truncated"] == 2 and r["n_requested"] == 4


# ── the written artifact ─────────────────────────────────────────────────────────
class TestOutput:
    def test_files_are_named_symbol_interval(self, tmp_path):
        from data.fetch_candles import backfill_universe
        backfill_universe(["BTC", "kPEPE"], interval="15m", days=1, output_dir=tmp_path,
                          fetch_fn=_writing_fetch(), delay=0.0)
        assert {p.name for p in tmp_path.iterdir()} == {"BTC_15m.parquet",
                                                        "kPEPE_15m.parquet"}

    def test_report_carries_row_counts_and_coverage(self, tmp_path):
        from data.fetch_candles import backfill_universe
        r = backfill_universe(["BTC", "ETH"], interval="1m", days=1, output_dir=tmp_path,
                              fetch_fn=lambda symbol, **kw: _candles(7), delay=0.0)
        assert r["rows"]["BTC"] == 7 and r["rows"]["ETH"] == 7
        assert r["interval"] == "1m"

    def test_cli_exposes_universe(self):
        from data.fetch_candles import build_parser
        args = build_parser().parse_args(["--universe", "--interval", "5m"])
        assert args.universe is True and args.interval == "5m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── the enumeration is a single point of failure ─────────────────────────────────
class TestEnumerationRetriesTransientErrors:
    """A 429 on the meta call killed the L2 sampler at startup on 2026-08-08.

    Per-symbol failures were already survivable, but `fetch_universe` is called ONCE
    before any work begins, so a single transient error takes down the whole run — the
    exact failure mode the rest of this unit was built to prevent, missed in the one
    place it is fatal.
    """

    def test_a_transient_error_is_retried(self):
        import urllib.error
        from data.fetch_candles import fetch_universe
        calls = []

        def flaky(_req):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
            return _meta([("BTC", False)])

        assert fetch_universe(info_fn=flaky, retries=3, backoff=0.0) == ["BTC"]
        assert len(calls) == 3

    def test_it_gives_up_loudly_rather_than_returning_an_empty_universe(self):
        import urllib.error
        from data.fetch_candles import fetch_universe

        def always_429(_req):
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

        with pytest.raises(urllib.error.HTTPError):
            fetch_universe(info_fn=always_429, retries=2, backoff=0.0)

    def test_a_malformed_payload_is_not_retried(self):
        """Retrying a schema error just wastes a minute — only transport faults recur."""
        from data.fetch_candles import fetch_universe
        calls = []

        def bad(_req):
            calls.append(1)
            return {"universe": []}

        with pytest.raises(ValueError):
            fetch_universe(info_fn=bad, retries=5, backoff=0.0)
        assert len(calls) == 1
