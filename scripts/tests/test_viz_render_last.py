"""Planted tests for `nat viz render --last <duration>` window math.

Contract (docs/contracts/viz.md + the freshest-readable-tail plan):
  - tail_bounds(ts_min, ts_max, minutes) returns [ts_max - minutes, ts_max+1),
    clamping t0 to ts_min when less than `minutes` of data is available (partial).
  - parse_duration_minutes accepts "15m" / "1h" / "90m" / bare "15" (minutes),
    and raises ValueError on garbage.

Pure helpers (no IO) — red-first before they exist in viz/pager.py.
"""

from __future__ import annotations

import pytest

from viz.pager import tail_bounds, parse_duration_minutes

NS = 1_000_000_000
MIN = 60 * NS
T0 = 1_781_532_000_000_000_000  # arbitrary t0


def test_tail_bounds_normal_window():
    # 60 min of data, ask for the last 15 → [ts_max-15m, ts_max+1)
    ts_min, ts_max = T0, T0 + 60 * MIN
    t0, t1 = tail_bounds(ts_min, ts_max, 15)
    assert t0 == ts_max - 15 * MIN
    assert t1 == ts_max + 1


def test_tail_bounds_partial_when_less_data():
    # only 5 min of data, ask for 15 → t0 clamps to ts_min (partial), t1 exclusive
    ts_min, ts_max = T0, T0 + 5 * MIN
    t0, t1 = tail_bounds(ts_min, ts_max, 15)
    assert t0 == ts_min
    assert t1 == ts_max + 1


def test_tail_bounds_t1_is_exclusive_upper():
    ts_min, ts_max = T0, T0 + 30 * MIN
    _t0, t1 = tail_bounds(ts_min, ts_max, 10)
    assert t1 == ts_max + 1  # so a `< t1` filter includes the final tick


@pytest.mark.parametrize("s,expected", [
    ("15m", 15.0), ("30m", 30.0), ("90m", 90.0),
    ("1h", 60.0), ("2h", 120.0),
    ("15", 15.0), ("5", 5.0),
])
def test_parse_duration_minutes_valid(s, expected):
    assert parse_duration_minutes(s) == expected


@pytest.mark.parametrize("s", ["", "abc", "15x", "m", "h", "1.5.2"])
def test_parse_duration_minutes_garbage_raises(s):
    with pytest.raises(ValueError):
        parse_duration_minutes(s)
