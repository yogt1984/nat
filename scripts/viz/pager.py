"""Data-relative pagination for the parquet viewers (§3b of the requirements).

`--tf` is the time granularity / page width; an optional 1-based INDEX pages
through a day. Page 1 begins at the first available tick (data-relative anchor),
so "the first 5 min" is literally page 1 — no empty leading pages. The final
page may be partial (data ends mid-window).

Shared by `nat viz render` (15m_visualize.py) and `nat viz3d` (viz_mesh.py).
"""

from __future__ import annotations


def window_edges(ts_min: int, ts_max: int, window_minutes: float) -> list[int]:
    """Consecutive window edges (ns), data-relative: the first edge is the first
    available tick. Mirrors the slicing used to paginate a day into tf-windows."""
    window_ns = int(window_minutes * 60 * 1e9)
    edges = list(range(int(ts_min), int(ts_max) + 1, window_ns))
    if not edges:
        edges = [int(ts_min)]
    if edges[-1] < ts_max:
        edges.append(int(ts_max) + 1)
    if len(edges) == 1:  # span shorter than one window → a single partial page
        edges.append(int(ts_max) + 1)
    return edges


def tail_bounds(ts_min: int, ts_max: int, minutes: float):
    """Bounds for the LAST `minutes` of available data: ``[t0, t1)`` where
    ``t1 = ts_max + 1`` (exclusive upper, so a ``< t1`` filter keeps the final
    tick) and ``t0 = max(ts_min, ts_max - minutes)``. If the data spans less
    than `minutes`, ``t0`` clamps to ``ts_min`` (a partial tail)."""
    window_ns = int(minutes * 60 * 1e9)
    t0 = max(int(ts_min), int(ts_max) - window_ns)
    t1 = int(ts_max) + 1
    return t0, t1


def parse_duration_minutes(s) -> float:
    """Parse a duration into minutes: ``"15m"``→15, ``"1h"``→60, ``"90m"``→90,
    bare ``"15"``→15. Raises ValueError on anything else."""
    s = str(s).strip().lower()
    if not s:
        raise ValueError("empty duration")
    if s.endswith("m"):
        return float(s[:-1])          # float() raises ValueError on garbage
    if s.endswith("h"):
        return float(s[:-1]) * 60.0
    return float(s)                   # bare value = minutes


def window_bounds(ts_min: int, ts_max: int, window_minutes: float, index: int):
    """1-based page bounds for the data-relative pagination model.

    Returns ``(t0, t1, n_pages, partial)``. ``index`` counts windows forward
    from the first available tick. Raises IndexError if out of range. ``partial``
    is True when the page is shorter than a full window (final page).
    """
    edges = window_edges(ts_min, ts_max, window_minutes)
    n_pages = len(edges) - 1
    if index < 1 or index > n_pages:
        raise IndexError(
            f"page {index} out of range; {n_pages} {window_minutes:g}min "
            f"page(s) available for this data"
        )
    t0, t1 = edges[index - 1], edges[index]
    window_ns = int(window_minutes * 60 * 1e9)
    partial = (t1 - t0) < window_ns
    return t0, t1, n_pages, partial
