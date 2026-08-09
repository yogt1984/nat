"""VW-1: multi-scale trade VWAP from bucketed accumulators (offline, no schema change).

The ingestor publishes one VWAP — a 5-second trade window — while LF7's channel needs an
hour-scale anchor and PROC-20 found band capture concentrating at k ≈ 2.0–2.5 on hour-scale
windows using a *bar-derived* midline, because the tick-level slow VWAP does not exist as a
feature. This computes all six candidate windows offline from `data/trades/`, so
`specs/multiscale_vwap.md`'s question — *which windows earn a column?* — can be answered
before a 12-column feature-vector migration is committed to.

**Why buckets rather than scans.** VWAP is a ratio of two sums and sums decompose:

    vwap(N min) = Σ notional over the last N buckets / Σ volume over the last N buckets

A ring of 1-minute accumulators covers 12 h in **~12 KB per symbol** against ~17 MB of raw
trades, and reading a window is O(N/60) rather than an O(n) scan with allocation.

**The window is minute-ALIGNED, not trailing-exact.** `vwap(5)` covers the last five whole
minute buckets, not the last 300 seconds ending at the newest trade. This is inherent to
bucketing and is the price of the O(1) update — a trailing-exact window requires the raw
scan the design exists to avoid. It matters downstream: at 1-minute resolution the two
definitions differ by up to one bucket of trades, which is immaterial for the 1 h/6 h/12 h
windows this unit exists for and would be material for a 5-second one (which is why the
5-second VWAP stays on the raw `TradeBuffer` and is not reimplemented here). That is not
an optimisation detail — it is what makes the Rust port viable at all, since six
`trades_in_window` scans per symbol at 10 Hz would blow the ingestor's 80 ms/tick p99 emission
budget. `VwapRing` is deliberately the same structure Phase B would implement, so
`tests/test_vwap_multiscale.py` doubles as that implementation's specification.

**Three ways a VWAP lies, all refused here:**

- a **partial window** reported as if it were full — 10 minutes of trades must not produce a
  "1 h VWAP", which is the number that looks fine in a column and is wrong in a backtest.
  Windows return NaN until they are warm;
- a **feed gap** papered over with the surviving trades. §7 guarantees gaps, so a window
  spanning one longer than `max_gap_minutes` returns NaN rather than averaging across it;
- an **empty minute** forward-filled. No trades contributes 0 notional *and* 0 volume, never
  the last price carried forward, which would invent volume that never traded.

**Sign convention.** The shipped `flow_vwap_deviation` is `(vwap − price)/price` — **inverted**
relative to how the quantity is normally read, which is why §1's IC of −0.29 means *price below
VWAP predicts further decline*. Columns here use the intuitive `(price − vwap)/vwap`. The
shipped column is deliberately left untouched: re-signing it would invalidate §1's IC record.
A test pins both conventions so the discrepancy lives in code rather than in prose.

Spec: `docs/specs/multiscale_vwap.md` §A1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

NS_PER_MIN = 60_000_000_000

#: Candidate windows, in minutes. Each must earn its column in VW-2 before Phase B.
WINDOWS: dict[str, int] = {
    "5m": 5, "10m": 10, "15m": 15, "1h": 60, "6h": 360, "12h": 720,
}

#: Rolling window (in emitted rows) for the deviation's z-score — the `k` of LF7's k·sigma.
Z_WINDOW = 500

#: A hole longer than this makes any window spanning it unrepresentative.
DEFAULT_MAX_GAP_MINUTES = 30


class VwapRing:
    """Per-minute (notional, volume) accumulators over a fixed lookback.

    Trades must arrive in non-decreasing timestamp order (they do, from the writer).
    Memory is O(max_minutes), independent of trade rate.
    """

    def __init__(self, max_minutes: int = 720,
                 max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES):
        self.max_minutes = int(max_minutes)
        self.max_gap_minutes = int(max_gap_minutes)
        self._notional = np.zeros(self.max_minutes, dtype=np.float64)
        self._volume = np.zeros(self.max_minutes, dtype=np.float64)
        self._minute = np.full(self.max_minutes, -1, dtype=np.int64)   # absolute index
        self._current: Optional[int] = None      # newest absolute minute seen
        self._first: Optional[int] = None        # oldest minute ever seen (warm-up)
        self._last_active: Optional[int] = None  # newest minute that actually had a trade

    # ── ingest ───────────────────────────────────────────────────────────────────
    def add(self, timestamp_ns: int, price: float, size: float) -> None:
        minute = int(timestamp_ns) // NS_PER_MIN
        if self._first is None:
            self._first = minute
        if self._current is None or minute > self._current:
            self._current = minute
        slot = minute % self.max_minutes
        if self._minute[slot] != minute:                  # recycle a stale slot
            self._minute[slot] = minute
            self._notional[slot] = 0.0
            self._volume[slot] = 0.0
        if size > 0:
            self._notional[slot] += float(price) * float(size)
            self._volume[slot] += float(size)
            self._last_active = minute if self._last_active is None \
                else max(self._last_active, minute)

    # ── read ─────────────────────────────────────────────────────────────────────
    def _window_slots(self, minutes: int) -> np.ndarray:
        """Absolute minutes in `(current - minutes, current]`, newest first."""
        return np.arange(self._current, self._current - minutes, -1, dtype=np.int64)

    def _sums(self, minutes: int) -> tuple[float, float]:
        wanted = self._window_slots(minutes)
        slots = wanted % self.max_minutes
        live = self._minute[slots] == wanted                  # slot still holds that minute
        return (float(self._notional[slots][live].sum()),
                float(self._volume[slots][live].sum()))

    def is_warm(self, minutes: int) -> bool:
        """True once the window is fully covered by observed time.

        A partial window is NOT a short VWAP — it is a different statistic wearing the
        label of the one asked for, so it is refused rather than approximated.
        """
        if self._current is None or self._first is None:
            return False
        return (self._current - self._first + 1) >= minutes

    def has_gap(self, minutes: int) -> bool:
        """True if the window spans a hole longer than `max_gap_minutes`."""
        if self._current is None or self._last_active is None:
            return True
        if self._current - self._last_active > self.max_gap_minutes:
            return True
        wanted = self._window_slots(minutes)          # newest .. oldest
        slots = wanted % self.max_minutes
        live = (self._minute[slots] == wanted) & (self._volume[slots] > 0)
        active = wanted[live]
        if active.size == 0:
            return True
        # Holes BETWEEN active minutes, and the hole from the window's oldest edge to its
        # first active minute. Omitting the second made a single active minute inside a
        # 360-minute window read as "no gap" — 359 empty minutes reported as a clean VWAP.
        gaps = [int(active[-1] - wanted[-1])]
        if active.size > 1:
            gaps.append(int(np.max(-np.diff(active))) - 1)
        return bool(max(gaps) > self.max_gap_minutes)

    def vwap(self, minutes: int) -> float:
        if self._current is None or minutes > self.max_minutes:
            return float("nan")
        if not self.is_warm(minutes) or self.has_gap(minutes):
            return float("nan")
        notional, volume = self._sums(minutes)
        return notional / volume if volume > 0 else float("nan")

    def total_volume(self, minutes: int) -> float:
        if self._current is None:
            return 0.0
        return self._sums(minutes)[1]


# ── frame-level transform ────────────────────────────────────────────────────────
def compute_multiscale_vwap(trades: pd.DataFrame,
                            windows: Optional[dict[str, int]] = None,
                            mark_price: Optional[float] = None,
                            z_window: int = Z_WINDOW,
                            max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES
                            ) -> pd.DataFrame:
    """Per-trade multi-scale VWAP, deviation and deviation z-score.

    One row per input trade, causal by construction: row *i* sees only trades 0..i, so
    appending future trades cannot change an earlier row (asserted in the suite). Symbols
    are kept in separate rings — a shared ring would blend BTC into ETH's VWAP.

    `mark_price` overrides the trade price when computing the deviation (test hook; in
    production the mid would be passed instead of the last trade).
    """
    windows = windows or WINDOWS
    cols = ["timestamp_ns", "symbol", "price", "size"]
    missing = [c for c in cols if c not in trades.columns]
    if missing:
        raise ValueError(f"trades frame is missing {missing}")
    if len(trades) == 0:
        out = trades.copy()
        for w in windows:
            for pre in ("vwap_", "vwap_dev_", "vwap_dev_z_"):
                out[f"{pre}{w}"] = pd.Series(dtype="float64")
        return out

    max_minutes = max(windows.values())
    rings: dict[str, VwapRing] = {}
    n = len(trades)
    vw = {w: np.full(n, np.nan) for w in windows}
    dev = {w: np.full(n, np.nan) for w in windows}

    ts = trades["timestamp_ns"].to_numpy(dtype=np.int64)
    sym = trades["symbol"].to_numpy()
    px = trades["price"].to_numpy(dtype=np.float64)
    sz = trades["size"].to_numpy(dtype=np.float64)

    for i in range(n):
        s = sym[i]
        ring = rings.get(s)
        if ring is None:
            ring = rings[s] = VwapRing(max_minutes=max_minutes,
                                       max_gap_minutes=max_gap_minutes)
        ring.add(int(ts[i]), float(px[i]), float(sz[i]))
        ref = float(mark_price) if mark_price is not None else float(px[i])
        for name, minutes in windows.items():
            v = ring.vwap(minutes)
            vw[name][i] = v
            # INTUITIVE sign: price above VWAP -> positive. The shipped
            # flow_vwap_deviation is the inverse; see the module docstring.
            dev[name][i] = (ref - v) / v if (np.isfinite(v) and v > 0) else np.nan

    out = trades.copy()
    for name in windows:
        out[f"vwap_{name}"] = vw[name]
        out[f"vwap_dev_{name}"] = dev[name]
        d = pd.Series(dev[name], index=out.index)
        sd = d.rolling(z_window, min_periods=max(20, z_window // 10)).std()
        mu = d.rolling(z_window, min_periods=max(20, z_window // 10)).mean()
        out[f"vwap_dev_z_{name}"] = ((d - mu) / sd.replace(0.0, np.nan)).to_numpy()
    return out
