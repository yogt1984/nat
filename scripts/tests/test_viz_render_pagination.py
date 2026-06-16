"""Planted tests for the paginated-viewer model (Capability A, AC-4).

Contract (docs/requirements/parquet_viz_validation.md §3b):
  - INDEX is 1-based and data-relative: page 1 starts at the first available
    tick t0; page N spans [t0 + (N-1)*tf, t0 + N*tf).
  - The final page may be partial (data ends mid-window) → partial=True.
  - Out-of-range / non-positive INDEX raises IndexError.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# 15m_visualize.py has a numeric prefix → spec-based import.
_spec = importlib.util.spec_from_file_location(
    "viz15m", str(Path(__file__).parent.parent / "15m_visualize.py")
)
viz15m = importlib.util.module_from_spec(_spec)
sys.modules["viz15m"] = viz15m
_spec.loader.exec_module(viz15m)

NS = 1_000_000_000
MIN = 60 * NS
T0 = 1_781_532_130_000_000_000  # a deliberately non-midnight first tick


def test_first_page_is_first_window():
    t0, t1, n_pages, partial = viz15m.window_bounds(T0, T0 + 12 * MIN, 5, 1)
    assert t0 == T0
    assert t1 == T0 + 5 * MIN
    assert n_pages == 3
    assert partial is False


def test_second_page_is_5_to_10_min():
    t0, t1, n_pages, partial = viz15m.window_bounds(T0, T0 + 12 * MIN, 5, 2)
    assert t0 == T0 + 5 * MIN
    assert t1 == T0 + 10 * MIN
    assert partial is False


def test_final_page_is_partial():
    # 12 min of data, 5-min windows → page 3 covers 10–12 min (partial).
    t0, t1, n_pages, partial = viz15m.window_bounds(T0, T0 + 12 * MIN, 5, 3)
    assert t0 == T0 + 10 * MIN
    assert partial is True


def test_out_of_range_index_raises():
    with pytest.raises(IndexError):
        viz15m.window_bounds(T0, T0 + 12 * MIN, 5, 4)


def test_non_positive_index_raises():
    with pytest.raises(IndexError):
        viz15m.window_bounds(T0, T0 + 12 * MIN, 5, 0)


def test_span_shorter_than_window_is_single_partial_page():
    # 3 min of data, 5-min window → exactly one (partial) page.
    t0, t1, n_pages, partial = viz15m.window_bounds(T0, T0 + 3 * MIN, 5, 1)
    assert n_pages == 1
    assert t0 == T0
    assert partial is True
    with pytest.raises(IndexError):
        viz15m.window_bounds(T0, T0 + 3 * MIN, 5, 2)


def test_one_minute_pages():
    # 1-min granularity over 3 min → 3 full pages.
    _, _, n_pages, _ = viz15m.window_bounds(T0, T0 + 3 * MIN, 1, 1)
    assert n_pages == 3
    t0, t1, _, partial = viz15m.window_bounds(T0, T0 + 3 * MIN, 1, 2)
    assert t0 == T0 + 1 * MIN and t1 == T0 + 2 * MIN
    assert partial is False
