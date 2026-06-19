"""Planted tests for ops.gap_alert — the T0b data-gap alert daemon.

Test-first (red). The daemon pages (<5 min) when feature ingestion stalls — it
reads data freshness as ``now - max(mtime over *.parquet + *.parquet.tmp)``, so
the live ``.parquet.tmp`` the writer flushes into IS the liveness signal (the
newest *closed* ``.parquet`` can be ~an hour old mid-rotation). It alerts once on
gap-open and once on recovery (state file), never per-cycle.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ops.gap_alert import (
    GapAlerter,
    GapState,
    latest_data_age_s,
    is_gap,
    is_paused,
    read_gap_state,
    write_gap_state,
    healthy,
)


def _touch(path: Path, age_s: float, now: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    mt = now - age_s
    os.utime(path, (mt, mt))


# ---------------------------------------------------------------------------
# Freshness estimator
# ---------------------------------------------------------------------------


class TestLatestDataAge:
    def test_none_when_empty(self, tmp_path):
        assert latest_data_age_s([tmp_path]) is None

    def test_uses_newest_mtime(self, tmp_path):
        now = 1_000_000.0
        _touch(tmp_path / "2026-06-15" / "a.parquet", age_s=3600, now=now)
        _touch(tmp_path / "2026-06-15" / "b.parquet", age_s=120, now=now)
        assert latest_data_age_s([tmp_path], now=now) == pytest.approx(120, abs=1)

    def test_includes_active_tmp_file(self, tmp_path):
        # the live .parquet.tmp is fresher than any closed .parquet
        now = 1_000_000.0
        _touch(tmp_path / "2026-06-15" / "old.parquet", age_s=3600, now=now)
        _touch(tmp_path / "2026-06-15" / "live.parquet.tmp", age_s=20, now=now)
        assert latest_data_age_s([tmp_path], now=now) == pytest.approx(20, abs=1)

    def test_multiple_dirs(self, tmp_path):
        now = 1_000_000.0
        d1, d2 = tmp_path / "features", tmp_path / "trades"
        _touch(d1 / "a.parquet", age_s=500, now=now)
        _touch(d2 / "b.parquet.tmp", age_s=40, now=now)
        assert latest_data_age_s([d1, d2], now=now) == pytest.approx(40, abs=1)


class TestIsGap:
    def test_fresh_is_not_gap(self):
        assert is_gap(30.0, threshold_s=300) is False

    def test_stale_is_gap(self):
        assert is_gap(600.0, threshold_s=300) is True

    def test_none_is_not_gap(self):
        # no data at all => startup/unknown, do not page
        assert is_gap(None, threshold_s=300) is False

    def test_boundary(self):
        assert is_gap(300.0, threshold_s=300) is False
        assert is_gap(301.0, threshold_s=300) is True


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestGapStatePersistence:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "gap_state.json"
        st = GapState(gapping=True, age_s=612.0,
                      gap_started_at="2026-06-15T14:00:00+00:00",
                      last_alert_at="2026-06-15T14:00:00+00:00",
                      last_check_at="2026-06-15T14:05:00+00:00")
        write_gap_state(p, st)
        loaded = read_gap_state(p)
        assert loaded.gapping
        assert loaded.age_s == pytest.approx(612.0)

    def test_absent_is_not_gapping(self, tmp_path):
        assert read_gap_state(tmp_path / "missing.json").gapping is False


# ---------------------------------------------------------------------------
# Alerter transitions (alert-once on open + recovery)
# ---------------------------------------------------------------------------


def _alerter(tmp_path, data_dir, clock, notify=False):
    return GapAlerter(
        config={"gap_threshold_s": 300, "row_threshold_s": 600, "poll_interval_s": 30,
                "data_dirs": [str(data_dir)],
                "alert_log": str(tmp_path / "alerts.log"),
                "pause_file": str(tmp_path / "ingestion_paused")},
        state_path=tmp_path / "gap_state.json",
        heartbeat_path=tmp_path / "hb",
        pid_file=tmp_path / "pid",
        clock=clock, notify=notify,
    )


class TestAlerterTransitions:
    def test_fresh_data_no_gap(self, tmp_path):
        now = 2_000_000.0
        data = tmp_path / "features"
        _touch(data / "2026-06-15" / "live.parquet.tmp", age_s=25, now=now)
        st = _alerter(tmp_path, data, clock=lambda: now).check()
        assert not st.gapping

    def test_stale_data_opens_gap(self, tmp_path):
        now = 2_000_000.0
        data = tmp_path / "features"
        _touch(data / "2026-06-15" / "stale.parquet.tmp", age_s=900, now=now)
        a = _alerter(tmp_path, data, clock=lambda: now)
        st = a.check()
        assert st.gapping
        assert st.gap_started_at is not None
        # persisted
        assert read_gap_state(tmp_path / "gap_state.json").gapping

    def test_alert_fires_once_then_recovers(self, tmp_path):
        clk = {"now": 2_000_000.0}
        data = tmp_path / "features"
        f = data / "2026-06-15" / "f.parquet.tmp"
        alerts = []
        a = _alerter(tmp_path, data, clock=lambda: clk["now"])
        a._send = lambda msg: alerts.append(msg)  # capture pages

        # 1) gap opens -> one alert
        _touch(f, age_s=900, now=clk["now"])
        a.check()
        # 2) still gapping next cycle -> NO new alert
        clk["now"] += 30
        _touch(f, age_s=930, now=clk["now"])
        a.check()
        assert sum("GAP" in m.upper() for m in alerts) == 1
        # 3) data recovers -> recovery alert
        clk["now"] += 30
        _touch(f, age_s=5, now=clk["now"])
        st = a.check()
        assert not st.gapping
        assert any("RECOVER" in m.upper() for m in alerts)


# ---------------------------------------------------------------------------
# Pause marker (T2): intentional `nat stop` suppresses pages but keeps monitoring
# ---------------------------------------------------------------------------


class TestPause:
    def test_is_paused_helper(self, tmp_path):
        marker = tmp_path / "ingestion_paused"
        assert is_paused(marker) is False
        marker.write_text("2026-06-19T00:00:00+00:00")
        assert is_paused(marker) is True

    def test_paused_suppresses_gap_alert(self, tmp_path):
        now = 2_000_000.0
        data = tmp_path / "features"
        _touch(data / "2026-06-15" / "stale.parquet.tmp", age_s=900, now=now)  # would gap
        a = _alerter(tmp_path, data, clock=lambda: now)
        (tmp_path / "ingestion_paused").write_text("paused")        # but we paused
        alerts = []
        a._send = lambda msg: alerts.append(msg)
        st = a.check()
        assert st.paused is True
        assert st.gapping is False
        assert alerts == []                                         # no page while paused

    def test_unpaused_after_marker_removed_pages(self, tmp_path):
        now = 2_000_000.0
        data = tmp_path / "features"
        _touch(data / "2026-06-15" / "stale.parquet.tmp", age_s=900, now=now)
        a = _alerter(tmp_path, data, clock=lambda: now)
        alerts = []
        a._send = lambda msg: alerts.append(msg)
        a.check()                                                   # not paused → gap opens
        assert sum("GAP" in m.upper() for m in alerts) == 1


# ---------------------------------------------------------------------------
# Local fallback (T1): a gap is logged to alert_log even with no Telegram
# ---------------------------------------------------------------------------


class TestLocalFallback:
    def test_gap_written_to_alert_log(self, tmp_path):
        now = 2_000_000.0
        data = tmp_path / "features"
        _touch(data / "2026-06-15" / "stale.parquet.tmp", age_s=900, now=now)
        a = _alerter(tmp_path, data, clock=lambda: now, notify=True)  # real _send path
        a.check()
        alert_log = tmp_path / "alerts.log"
        assert alert_log.exists()
        assert "DATA GAP" in alert_log.read_text()


# ---------------------------------------------------------------------------
# Row-timestamp freshness (T4): stalled buffer (fresh mtime, old rows) = gap
# ---------------------------------------------------------------------------


class TestRowAge:
    def test_stalled_rows_with_fresh_mtime_is_gap(self, tmp_path):
        pq = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pqw
        import pyarrow as pa
        now = 2_000_000.0
        data = tmp_path / "features" / "2026-06-15"
        data.mkdir(parents=True)
        f = data / "rows.parquet"
        # newest row is 5000s old, but the file's mtime is fresh (10s) — mtime
        # alone would say "OK"; row-age (threshold 600) must flag the gap.
        old_ns = int((now - 5000) * 1e9)
        pqw.write_table(pa.table({"timestamp_ns": [old_ns], "symbol": ["BTC"]}), f)
        os.utime(f, (now - 10, now - 10))
        a = _alerter(tmp_path, tmp_path / "features", clock=lambda: now)
        alerts = []
        a._send = lambda msg: alerts.append(msg)
        st = a.check()
        assert st.gapping is True
        assert st.row_age_s == pytest.approx(5000, abs=2)


# ---------------------------------------------------------------------------
# Healthcheck heartbeat
# ---------------------------------------------------------------------------


class TestHealthcheck:
    def test_missing_heartbeat_unhealthy(self, tmp_path):
        assert healthy({"heartbeat_path": str(tmp_path / "hb"), "poll_interval_s": 30}) is False

    def test_fresh_heartbeat_healthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("now")
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 30}) is True

    def test_stale_heartbeat_unhealthy(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("old")
        os.utime(hb, (1_700_000_000.0, 1_700_000_000.0))
        assert healthy({"heartbeat_path": str(hb), "poll_interval_s": 30}) is False
