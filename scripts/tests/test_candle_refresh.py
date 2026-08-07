"""XS-7 — the daily candle refresh: the run that cannot be re-run.

The venue keeps ~5000 bars per interval, so 1 m history reaches 3.5 days and **cannot be
backfilled** (FINDINGS §7.1). Every day this job does not run is a day of 1 m universe
breadth that no future fetch can recover. That asymmetry is what these tests encode —
an ordinary cron may skip a day and catch up tomorrow; this one may not.

Three properties carry the unit:

  1. **A transient failure must not become a permanent hole.** The 2026-08-07 sweep
     produced two `empty` verdicts (ORDI 15 m, REZ 5 m) that both succeeded on immediate
     retry. Without a retry pass those are two lost pair-days per occurrence. The retry
     must NOT, however, launder a genuine failure into success — so the tests pin both
     directions.

  2. **"Rows came back" is not "the window was satisfied."** The 1 m sweep returned 4 % of
     the requested span and reported `ok=177 failed=0 empty=0`. A cron that reports clean
     while collecting 4 % is worse than no cron, because it removes the reason to look.
     Coverage must compare *requested* against *received*.

  3. **A missed run must catch up.** If the box is asleep at the scheduled time, systemd
     must fire the job on the next boot (`Persistent=true`) — for data that expires at the
     source, "we'll get it next time" is false.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data.fetch_candles import backfill_universe, span_days  # noqa: E402

MIN_MS = 60_000


def _frame(n: int, interval_ms: int = MIN_MS, start_ms: int = 1_750_000_000_000) -> pd.DataFrame:
    ts = [start_ms + i * interval_ms for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="ms", utc=True),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })


# ── 1. retry rescues transients without laundering real failures ──────────

def test_transient_failure_is_retried_and_recorded_ok():
    """ORDI/REZ, reproduced: one failure then success must end as ok, not empty."""
    calls = {"n": 0}

    def fetch(symbol, interval, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient HTTP 502")
        return _frame(100)

    rep = backfill_universe(["ORDI"], interval="1m", fetch_fn=fetch, retries=1, delay=0)

    assert rep["ok"] == ["ORDI"]
    assert rep["failed"] == []
    assert calls["n"] == 2, "the failure was not retried"


def test_empty_result_is_retried_too():
    """`empty` was the actual 2026-08-07 symptom — an empty frame must also get a retry."""
    calls = {"n": 0}

    def fetch(symbol, interval, **kw):
        calls["n"] += 1
        return pd.DataFrame() if calls["n"] == 1 else _frame(50)

    rep = backfill_universe(["REZ"], interval="5m", fetch_fn=fetch, retries=1, delay=0)

    assert rep["ok"] == ["REZ"]
    assert rep["empty"] == []
    assert calls["n"] == 2


def test_persistent_failure_still_fails_after_retries():
    """Retry must not launder a genuine outage into a green run."""
    calls = {"n": 0}

    def fetch(symbol, interval, **kw):
        calls["n"] += 1
        raise RuntimeError("venue down")

    rep = backfill_universe(["BTC"], interval="1m", fetch_fn=fetch, retries=2, delay=0)

    assert [f["symbol"] for f in rep["failed"]] == ["BTC"]
    assert rep["ok"] == []
    assert calls["n"] == 3, "expected initial attempt + 2 retries"


def test_retries_are_bounded():
    """A dead venue must not spin forever — attempts are exactly retries+1."""
    calls = {"n": 0}

    def fetch(symbol, interval, **kw):
        calls["n"] += 1
        raise RuntimeError("nope")

    backfill_universe(["A", "B"], interval="1m", fetch_fn=fetch, retries=1, delay=0)
    assert calls["n"] == 4          # 2 symbols x (1 + 1)


def test_retries_zero_preserves_old_behaviour():
    calls = {"n": 0}

    def fetch(symbol, interval, **kw):
        calls["n"] += 1
        raise RuntimeError("x")

    backfill_universe(["A"], interval="1m", fetch_fn=fetch, retries=0, delay=0)
    assert calls["n"] == 1


# ── 2. requested vs received span — the check the 1m run needed ───────────

def test_span_days_measures_the_frame():
    # Span is first-open → last-close, so N one-minute bars cover exactly N minutes
    # and a full day is 1440 bars (not 1441 — the last bar's own duration counts).
    assert span_days(_frame(1440), MIN_MS) == pytest.approx(1.0, abs=1e-6)
    assert span_days(_frame(720), MIN_MS) == pytest.approx(0.5, abs=1e-6)
    assert span_days(pd.DataFrame(), MIN_MS) == 0.0


def test_short_span_is_flagged_even_though_rows_came_back():
    """THE regression test for 2026-08-07: 3.5 d returned against 90 d requested."""
    def fetch(symbol, interval, **kw):
        return _frame(5000)                       # 5000 1m bars ~= 3.47 d

    rep = backfill_universe(["BTC"], interval="1m", days=90,
                            fetch_fn=fetch, delay=0)

    assert rep["ok"] == ["BTC"], "it did return rows"
    assert "BTC" in rep["short"], "but 3.5 d of a 90 d request must be flagged short"
    got, want = rep["short"]["BTC"]
    assert want == 90
    assert got == pytest.approx(3.47, abs=0.05)


def test_full_span_is_not_flagged():
    def fetch(symbol, interval, **kw):
        return _frame(90 * 1440 + 1)              # exactly 90 d of 1m bars

    rep = backfill_universe(["BTC"], interval="1m", days=90, fetch_fn=fetch, delay=0)
    assert rep["short"] == {}


def test_short_tolerance_admits_a_recent_listing():
    """A pair listed 89 d ago is not a defect; the tolerance must not cry wolf."""
    def fetch(symbol, interval, **kw):
        return _frame(int(89.5 * 1440))

    rep = backfill_universe(["NEWCOIN"], interval="1m", days=90,
                            fetch_fn=fetch, delay=0, short_tolerance=0.98)
    assert rep["short"] == {}


def test_report_still_accounts_for_every_symbol():
    def fetch(symbol, interval, **kw):
        if symbol == "F":
            raise RuntimeError("dead")
        if symbol == "S":
            return _frame(100)                     # short vs 90 d
        return _frame(90 * 1440 + 1)

    rep = backfill_universe(["A", "S", "F"], interval="1m", days=90,
                            fetch_fn=fetch, retries=0, delay=0)

    assert len(rep["ok"]) + len(rep["failed"]) + len(rep["empty"]) + len(rep["rejected"]) == 3
    assert "S" in rep["short"] and "S" in rep["ok"], (
        "short is an ANNOTATION on a successful fetch, not a separate bucket — "
        "the totals must still reconcile"
    )


# ── 3. the schedule: a missed run must catch up ───────────────────────────

def test_refresh_timer_is_persistent_and_daily(monkeypatch, tmp_path):
    systemd_units = pytest.importorskip("ops.systemd_units")
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(tmp_path / "install"))
    monkeypatch.setenv("NAT_HOME", str(tmp_path / "home"))

    units = systemd_units.render_units(python="/usr/bin/python3")
    timer = units["nat-candle-refresh.timer"]

    assert "OnCalendar=" in timer
    assert "Persistent=true" in timer, (
        "1m candles expire at the venue — a run missed while the box was off "
        "must fire on next boot, or that day is lost forever"
    )
    assert "Unit=nat-candle-refresh.service" in timer
    assert "WantedBy=timers.target" in timer


def test_refresh_service_is_oneshot_and_sweeps_the_universe(monkeypatch, tmp_path):
    systemd_units = pytest.importorskip("ops.systemd_units")
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(tmp_path / "install"))
    monkeypatch.setenv("NAT_HOME", str(tmp_path / "home"))

    svc = systemd_units.render_units(python="/usr/bin/python3")["nat-candle-refresh.service"]

    assert "Type=oneshot" in svc
    assert "--universe" in svc
    assert "fetch_candles.py" in svc
    # Line-anchored: a comment *explaining* the absence of Restart= is fine, a
    # directive is not. A oneshot sweep must wait for the next timer window rather
    # than hammer the venue in a restart loop.
    assert not any(ln.startswith("Restart=") for ln in svc.splitlines())


# ── XS-8: the L2 sampler runs as a supervised loop, not a timer ───────────

def test_l2_sampler_unit_is_a_restarting_daemon(monkeypatch, tmp_path):
    """Unlike the candle refresh (a oneshot timer), the sampler is a long-lived loop.

    Its product is a DISTRIBUTION of half-spreads, so a process that dies at 03:00 and
    waits for tomorrow leaves a hole in exactly the intraday variation being measured.
    Restart=always is the point of difference.
    """
    systemd_units = pytest.importorskip("ops.systemd_units")
    monkeypatch.setenv("NAT_INSTALL_ROOT", str(tmp_path / "install"))
    monkeypatch.setenv("NAT_HOME", str(tmp_path / "home"))

    svc = systemd_units.render_units(python="/usr/bin/python3")["nat-l2-sampler.service"]

    assert "Type=simple" in svc
    assert "Restart=always" in svc
    assert "fetch_l2.py" in svc
    assert "--loop" in svc
    assert "WantedBy=default.target" in svc
