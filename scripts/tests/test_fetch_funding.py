"""Planted tests for the funding-history fetcher (LF8 step 0).

All venue responses are injected pages — no network. The arithmetic each test
pins is hand-computable: page sizes, timestamps, and rates are planted.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data.fetch_funding import (
    PAGE_LIMIT,
    backfill_universe,
    fetch_funding,
    update_symbol,
)

HOUR_MS = 3_600_000


def _entry(t_ms: int, rate: float) -> dict:
    return {"coin": "TST", "fundingRate": str(rate), "premium": "0.0001",
            "time": t_ms}


def _pages_fn(pages: list[list[dict]]):
    """info_fn that serves planted pages in order, then empties."""
    calls = []

    def fn(payload):
        calls.append(payload)
        return pages.pop(0) if pages else []

    fn.calls = calls
    return fn


class TestPagination:
    def test_single_short_page_is_one_request(self):
        fn = _pages_fn([[_entry(i * HOUR_MS, 1e-5) for i in range(24)]])
        df = fetch_funding("TST", 0, info_fn=fn, sleep_s=0)
        assert len(df) == 24
        assert len(fn.calls) == 1

    def test_full_page_advances_cursor_past_newest(self):
        page1 = [_entry(i * HOUR_MS, 1e-5) for i in range(PAGE_LIMIT)]
        page2 = [_entry((PAGE_LIMIT + i) * HOUR_MS, 2e-5) for i in range(10)]
        fn = _pages_fn([page1, page2])
        df = fetch_funding("TST", 0, info_fn=fn, sleep_s=0)
        assert len(df) == PAGE_LIMIT + 10
        # second request must start strictly after page1's newest entry
        assert fn.calls[1]["startTime"] == (PAGE_LIMIT - 1) * HOUR_MS + 1

    def test_page_overlap_deduplicated(self):
        page1 = [_entry(i * HOUR_MS, 1e-5) for i in range(PAGE_LIMIT)]
        # page 2 re-serves the boundary hour, as venue pagination can
        page2 = [_entry((PAGE_LIMIT - 1 + i) * HOUR_MS, 1e-5) for i in range(5)]
        fn = _pages_fn([page1, page2])
        df = fetch_funding("TST", 0, info_fn=fn, sleep_s=0)
        assert len(df) == PAGE_LIMIT + 4
        assert df["time"].is_unique

    def test_empty_history_is_empty_frame_not_error(self):
        df = fetch_funding("TST", 0, info_fn=_pages_fn([]), sleep_s=0)
        assert len(df) == 0
        assert list(df.columns) == ["time", "funding_rate", "premium"]

    def test_rates_parse_as_floats(self):
        fn = _pages_fn([[_entry(0, 1.25e-5)]])
        df = fetch_funding("TST", 0, info_fn=fn, sleep_s=0)
        assert df["funding_rate"].iloc[0] == pytest.approx(1.25e-5)


class TestIncrementalUpdate:
    def test_first_fetch_writes_parquet(self, tmp_path):
        fn = _pages_fn([[_entry(i * HOUR_MS, 1e-5) for i in range(48)]])
        path = update_symbol("TST", days=90, data_dir=tmp_path, info_fn=fn,
                             now_ms=100 * HOUR_MS, sleep_s=0)
        assert path == tmp_path / "TST.parquet"
        assert len(pd.read_parquet(path)) == 48

    def test_second_fetch_resumes_after_stored_max(self, tmp_path):
        fn1 = _pages_fn([[_entry(i * HOUR_MS, 1e-5) for i in range(48)]])
        update_symbol("TST", days=90, data_dir=tmp_path, info_fn=fn1,
                      now_ms=100 * HOUR_MS, sleep_s=0)
        fn2 = _pages_fn([[_entry((48 + i) * HOUR_MS, 2e-5) for i in range(24)]])
        path = update_symbol("TST", days=90, data_dir=tmp_path, info_fn=fn2,
                             now_ms=100 * HOUR_MS, sleep_s=0)
        # resumed strictly after hour 47, and merged without duplicates
        assert fn2.calls[0]["startTime"] == 47 * HOUR_MS + 1
        df = pd.read_parquet(path)
        assert len(df) == 72
        assert df["time"].is_unique

    def test_hostile_symbol_never_reaches_filesystem(self, tmp_path):
        with pytest.raises(ValueError, match="filename"):
            update_symbol("../evil", data_dir=tmp_path, info_fn=_pages_fn([]))
        assert list(tmp_path.iterdir()) == []


class TestUniverseSweep:
    def test_one_failure_does_not_abort_and_arithmetic_closes(self, tmp_path):
        def fn(payload):
            if payload["coin"] == "BAD":
                raise RuntimeError("HTTP 500")
            return [_entry(0, 1e-5)]

        result = backfill_universe(["AAA", "BAD", "CCC"], days=1,
                                   data_dir=tmp_path, info_fn=fn, sleep_s=0)
        assert result["ok"] == ["AAA", "CCC"]
        assert "BAD" in result["failed"]
        assert len(result["ok"]) + len(result["failed"]) == 3
        assert (tmp_path / "AAA.parquet").exists()
        assert not (tmp_path / "BAD.parquet").exists()
