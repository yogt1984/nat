"""T0b — data-gap alert daemon.

Pages via Telegram within minutes when feature ingestion stalls — the real-time
complement to the next-day nightly report (the plan's explicit requirement:
"<5 min page on any data gap, not next-day discovery"). Data continuity is the
project's binding constraint; a silent gap is the most expensive failure to miss.

Freshness signal: ``now - max(mtime over *.parquet AND *.parquet.tmp)`` across the
watched data dirs. The live ``.parquet.tmp`` the ingestor flushes into IS the
liveness signal — the newest *closed* ``.parquet`` can be ~an hour old mid
hourly-rotation, so a closed-file-only check (e.g. `nat status`) is too coarse
for gap alerting.

It alerts once when a gap opens and once on recovery (state in
``data/ops/gap_state.json``), never per-cycle. A heartbeat file backs the
docker-compose healthcheck (``gap_alert.py health``).

This monitor is READ-ONLY with respect to the ingestor — it only stats data
files — so it is safe to run against su-35's output mid-streak.

CLI: ``nat gap status|check|start|stop``.
"""

from __future__ import annotations

import json
import os
import signal as _signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# `nat gap start` runs this file directly; put scripts/ on the path so the
# best-effort Telegram import resolves regardless of launch method (else a gap
# would be detected but no page would fire).
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

ROOT = _SCRIPTS_ROOT.parent

try:
    from logging_config import setup_logging  # noqa: E402

    log = setup_logging("nat.ops.gap")
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("nat.ops.gap")


# ───────────────────────────────────────────────────────────────────────────
# Freshness estimator (planted-test pinned)
# ───────────────────────────────────────────────────────────────────────────

_DATA_GLOBS = ("*.parquet", "*.parquet.tmp")


def latest_data_age_s(
    data_dirs: Sequence[Path | str], now: float | None = None
) -> float | None:
    """Seconds since the most recent write across the watched dirs.

    Considers both closed ``*.parquet`` and the active ``*.parquet.tmp`` (the
    real liveness signal). Returns None when no data files exist at all.
    """
    newest: float | None = None
    for d in data_dirs:
        base = Path(d)
        if not base.exists():
            continue
        for pat in _DATA_GLOBS:
            for f in base.rglob(pat):
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if newest is None or mt > newest:
                    newest = mt
    if newest is None:
        return None
    now = now if now is not None else time.time()
    return now - newest


def is_gap(age_s: float | None, threshold_s: float) -> bool:
    """A gap = data exists but the newest write is older than threshold.

    age None (no data at all) is treated as startup/unknown — not a gap — so the
    daemon does not page before the ingestor's first write.
    """
    return age_s is not None and age_s > threshold_s


# ───────────────────────────────────────────────────────────────────────────
# State
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class GapState:
    gapping: bool = False
    age_s: float | None = None
    gap_started_at: str | None = None
    last_alert_at: str | None = None
    last_check_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def read_gap_state(path: Path | str) -> GapState:
    p = Path(path)
    if not p.exists():
        return GapState()
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return GapState()
    fields = GapState().to_dict().keys()
    return GapState(**{k: data.get(k) for k in fields})


def write_gap_state(path: Path | str, state: GapState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    os.replace(tmp, p)


def healthy(config: dict | None = None, now: float | None = None) -> bool:
    """True if the daemon's heartbeat is fresh (< 3 poll intervals, min 180s)."""
    cfg = config or load_config()
    hb = Path(cfg["heartbeat_path"])
    if not hb.exists():
        return False
    now = now if now is not None else time.time()
    max_age = max(180, 3 * int(cfg.get("poll_interval_s", 30)))
    return (now - hb.stat().st_mtime) < max_age


# ───────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────


def load_config(path: Path | str | None = None) -> dict:
    defaults = {
        # default 300s ("<5 min page"); comfortably above the writer's flush
        # cadence, so normal buffering never false-alarms. Tune per deploy.
        "gap_threshold_s": 300,
        "poll_interval_s": 30,
        "data_dirs": [str(ROOT / "data" / "features")],
        "state_path": str(ROOT / "data" / "ops" / "gap_state.json"),
        "heartbeat_path": str(ROOT / "data" / "ops" / "gap_alert.heartbeat"),
        "pid_file": str(ROOT / "data" / "ops" / "gap_alert.pid"),
    }
    cfg_path = Path(path) if path else ROOT / "config" / "ops.toml"
    if cfg_path.exists():
        try:
            import tomllib

            with open(cfg_path, "rb") as f:
                doc = tomllib.load(f)
            defaults.update(doc.get("gap_alert", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read %s: %s — using defaults", cfg_path, e)
    return defaults


# ───────────────────────────────────────────────────────────────────────────
# Alerter / daemon
# ───────────────────────────────────────────────────────────────────────────


class GapAlerter:
    def __init__(
        self,
        config: dict | None = None,
        *,
        state_path: Path | str | None = None,
        heartbeat_path: Path | str | None = None,
        pid_file: Path | str | None = None,
        clock: Callable[[], float] | None = None,
        notify: bool = True,
    ):
        self.config = config or load_config()
        self.state_path = Path(state_path or self.config["state_path"])
        self.heartbeat_path = Path(heartbeat_path or self.config["heartbeat_path"])
        self.pid_file = Path(pid_file or self.config["pid_file"])
        # Resolve relative data dirs against the repo root so freshness checks
        # work regardless of the daemon's CWD.
        self.data_dirs = [
            Path(d) if Path(d).is_absolute() else ROOT / d
            for d in self.config["data_dirs"]
        ]
        self.threshold = float(self.config["gap_threshold_s"])
        # clock returns epoch seconds (matches file mtimes)
        self._clock = clock or time.time
        self.notify = notify
        self._shutdown = False

    def _now(self) -> float:
        return self._clock()

    @staticmethod
    def _iso(epoch: float) -> str:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    def _send(self, message: str) -> None:
        """Page (best-effort). Always logged; Telegram only when notify=True."""
        log.warning(message)
        if not self.notify:
            return
        try:
            from tournament.report import send_telegram

            send_telegram(message)
        except Exception as e:  # pragma: no cover
            log.debug("Telegram send skipped: %s", e)

    def check(self, now: float | None = None) -> GapState:
        """One evaluation cycle: detect gap, alert on transitions, persist state."""
        now = now if now is not None else self._now()
        age = latest_data_age_s(self.data_dirs, now)
        gap = is_gap(age, self.threshold)
        prev = read_gap_state(self.state_path)
        now_iso = self._iso(now)

        if gap and not prev.gapping:
            # gap opens — page once
            msg = (
                f"⚠️ DATA GAP — no new feature data for {age:.0f}s "
                f"(threshold {self.threshold:.0f}s) as of {now_iso}"
            )
            self._send(msg)
            st = GapState(
                gapping=True, age_s=age, gap_started_at=now_iso,
                last_alert_at=now_iso, last_check_at=now_iso,
            )
        elif gap and prev.gapping:
            # still gapping — no new page
            st = GapState(
                gapping=True, age_s=age, gap_started_at=prev.gap_started_at,
                last_alert_at=prev.last_alert_at, last_check_at=now_iso,
            )
        elif (not gap) and prev.gapping:
            # recovery — page once
            dur = ""
            if prev.gap_started_at:
                try:
                    started = datetime.fromisoformat(prev.gap_started_at).timestamp()
                    dur = f" after {now - started:.0f}s"
                except ValueError:
                    pass
            self._send(f"✅ DATA RECOVERED — feature ingestion flowing again{dur} ({now_iso})")
            st = GapState(gapping=False, age_s=age, last_check_at=now_iso)
        else:
            # steady-state OK
            st = GapState(gapping=False, age_s=age, last_check_at=now_iso)

        write_gap_state(self.state_path, st)
        return st

    def status(self) -> dict:
        st = read_gap_state(self.state_path)
        d = st.to_dict()
        d["threshold_s"] = self.threshold
        d["data_dirs"] = [str(p) for p in self.data_dirs]
        return d

    def run(self) -> None:
        poll = int(self.config["poll_interval_s"])
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))
        _signal.signal(_signal.SIGTERM, self._handle_signal)
        _signal.signal(_signal.SIGINT, self._handle_signal)
        log.info(
            "Gap-alert daemon started (PID %d, poll %ds, threshold %.0fs, dirs=%s)",
            os.getpid(), poll, self.threshold, [str(p) for p in self.data_dirs],
        )
        try:
            while not self._shutdown:
                try:
                    st = self.check()
                    if st.gapping:
                        log.warning("DATA GAP active: %.0fs since last write", st.age_s or 0)
                except Exception as e:
                    log.error("Gap-alert cycle error: %s", e, exc_info=True)
                self.heartbeat_path.write_text(self._iso(self._now()))
                for _ in range(poll):
                    if self._shutdown:
                        break
                    time.sleep(1)
        finally:
            self.pid_file.unlink(missing_ok=True)
            log.info("Gap-alert daemon stopped")

    def _handle_signal(self, signum, frame):  # pragma: no cover
        log.info("Received signal %d, shutting down after cycle", signum)
        self._shutdown = True


# ───────────────────────────────────────────────────────────────────────────
# CLI entrypoint (invoked by `nat gap start`)
# ───────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "start"
    a = GapAlerter()
    if cmd == "start":
        a.run()
        return 0
    if cmd == "status":
        print(json.dumps(a.status(), indent=2))
        return 0
    if cmd == "check":
        st = a.check()
        print(json.dumps(st.to_dict(), indent=2))
        return 1 if st.gapping else 0
    if cmd == "health":
        return 0 if healthy(a.config) else 1
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
