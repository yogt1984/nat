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


def latest_row_age_s(
    data_dirs: Sequence[Path | str], now: float | None = None
) -> float | None:
    """T4: seconds since the newest *readable row* (max timestamp_ns) — a signal
    mtime can't fake. A stalled/zero-row buffer keeps a file's mtime fresh while
    no new rows land; this reads the newest closed parquet's column statistics
    (footer only, no full scan). Best-effort: None if unreadable / no pyarrow.
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    newest_file: Path | None = None
    newest_mt = -1.0
    for d in data_dirs:
        base = Path(d)
        if not base.exists():
            continue
        for f in base.rglob("*.parquet"):          # closed files only (stats present)
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if mt > newest_mt:
                newest_mt, newest_file = mt, f
    if newest_file is None:
        return None
    try:
        md = pq.read_metadata(newest_file)
        max_ns = None
        for rg in range(md.num_row_groups):
            col = None
            schema = md.schema.to_arrow_schema()
            if "timestamp_ns" not in schema.names:
                return None
            idx = schema.names.index("timestamp_ns")
            col = md.row_group(rg).column(idx)
            if col.statistics is not None and col.statistics.has_min_max:
                m = col.statistics.max
                if max_ns is None or m > max_ns:
                    max_ns = m
        if max_ns is None:
            return None
        now = now if now is not None else time.time()
        return now - (max_ns / 1e9)
    except Exception:
        return None


def newest_tmp(data_dirs: Sequence[Path | str]) -> tuple[Path, int, float] | None:
    """The freshest active ``*.parquet.tmp`` and its (path, size_bytes, mtime).

    The writer keeps one active ``.tmp`` whose size grows as row groups flush;
    a fresh mtime with a *non-growing* size over time is the zombie-ingestor
    (stalled-buffer) signal. None when no ``.tmp`` exists (between rotations)."""
    best: tuple[Path, int, float] | None = None
    for d in data_dirs:
        base = Path(d)
        if not base.exists():
            continue
        for f in base.rglob("*.parquet.tmp"):
            try:
                st = f.stat()
            except OSError:
                continue
            if best is None or st.st_mtime > best[2]:
                best = (f, st.st_size, st.st_mtime)
    return best


def _parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def is_paused(pause_file: Path | str) -> bool:
    """Intentional-pause marker present → suppress gap pages (but keep monitoring)."""
    return Path(pause_file).exists()


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


# ───────────────────────────────────────────────────────────────────────────
# State
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class GapState:
    gapping: bool = False
    paused: bool = False
    stalled: bool = False
    age_s: float | None = None
    row_age_s: float | None = None
    gap_started_at: str | None = None
    last_alert_at: str | None = None
    last_check_at: str | None = None
    # .parquet.tmp growth tracker (stalled-buffer detection)
    tmp_path: str | None = None
    tmp_size: int | None = None
    tmp_grew_at: str | None = None
    # auto-restart bookkeeping
    last_restart_at: str | None = None
    restart_count: int = 0

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
        "row_threshold_s": 600,
        "poll_interval_s": 30,
        "data_dirs": [str(ROOT / "data" / "features")],
        "state_path": str(ROOT / "data" / "ops" / "gap_state.json"),
        "heartbeat_path": str(ROOT / "data" / "ops" / "gap_alert.heartbeat"),
        "pid_file": str(ROOT / "data" / "ops" / "gap_alert.pid"),
        "alert_log": str(ROOT / "data" / "ops" / "alerts.log"),
        "pause_file": str(ROOT / "data" / "ops" / "ingestion_paused"),
        # Stalled-buffer detection (.tmp size flat while mtime fresh) + remediation.
        "stall_threshold_s": 900,
        "auto_restart": True,
        "restart_unit": "nat-ingestor.service",
        "restart_cooldown_s": 600,
        "max_consecutive_restarts": 3,
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


def _unit_is_managed(unit: str) -> bool:
    """True if a systemd --user unit file exists (so restarting it is meaningful)."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return (Path(xdg) / "systemd" / "user" / unit).exists()


def _systemctl_restart(unit: str) -> None:
    import subprocess
    subprocess.run(["systemctl", "--user", "restart", unit], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        restart_fn: Callable[[str], None] | None = None,
        unit_managed_fn: Callable[[str], bool] | None = None,
    ):
        self.config = config or load_config()
        self.state_path = Path(state_path or self.config["state_path"])
        self.heartbeat_path = Path(heartbeat_path or self.config["heartbeat_path"])
        self.pid_file = Path(pid_file or self.config["pid_file"])

        def _resolve(p: str | Path) -> Path:
            p = Path(p)
            return p if p.is_absolute() else ROOT / p

        # Resolve relative data dirs against the repo root so freshness checks
        # work regardless of the daemon's CWD.
        self.data_dirs = [_resolve(d) for d in self.config["data_dirs"]]
        self.alert_log = _resolve(self.config.get("alert_log", "data/ops/alerts.log"))
        self.pause_file = _resolve(self.config.get("pause_file", "data/ops/ingestion_paused"))
        self.threshold = float(self.config["gap_threshold_s"])
        self.row_threshold = float(self.config.get("row_threshold_s", 600))
        self.stall_threshold = float(self.config.get("stall_threshold_s", 900))
        self.auto_restart = bool(self.config.get("auto_restart", True))
        self.restart_unit = self.config.get("restart_unit", "nat-ingestor.service")
        self.restart_cooldown = float(self.config.get("restart_cooldown_s", 600))
        self.max_restarts = int(self.config.get("max_consecutive_restarts", 3))
        # Injectable for tests; real impls touch systemd only.
        self._restart_fn = restart_fn or _systemctl_restart
        self._unit_managed_fn = unit_managed_fn or _unit_is_managed
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
        """Page on every available channel (best-effort). Local channels always
        fire so a gap is never invisible; Telegram is push-on-top when configured."""
        log.warning(message)
        if not self.notify:
            return
        # Local fallback 1: durable, timestamped alert log (always works).
        try:
            self.alert_log.parent.mkdir(parents=True, exist_ok=True)
            with self.alert_log.open("a") as fh:
                fh.write(f"{self._iso(self._now())}  {message}\n")
        except OSError as e:  # pragma: no cover
            log.debug("alert_log write failed: %s", e)
        # Local fallback 2: desktop notification, if a display is available.
        try:
            import shutil as _sh
            if _sh.which("notify-send"):
                import subprocess as _sp
                _sp.run(["notify-send", "-u", "critical", "nat: data gap", message],
                        check=False, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        except Exception:  # pragma: no cover
            pass
        # Push: Telegram (no-op without creds — caller already warned loudly).
        try:
            from tournament.report import send_telegram

            send_telegram(message)
        except Exception as e:  # pragma: no cover
            log.debug("Telegram send skipped: %s", e)

    def _track_tmp(self, prev: GapState, now_iso: str) -> tuple[str | None, int | None, str | None, float | None]:
        """Returns (tmp_path, tmp_size, tmp_grew_at, stall_age) for this cycle."""
        cur = newest_tmp(self.data_dirs)
        if cur is None:
            # no active .tmp (between rotations) — can't judge a stall
            return prev.tmp_path, prev.tmp_size, prev.tmp_grew_at, None
        path, size, _mtime = cur
        path = str(path)
        grew = (prev.tmp_path != path or prev.tmp_size is None or size > prev.tmp_size)
        if grew:
            return path, size, now_iso, 0.0
        grew_at = prev.tmp_grew_at or now_iso
        started = _parse_iso(grew_at)
        stall_age = (self._now() - started) if started is not None else 0.0
        return path, size, grew_at, stall_age

    def _maybe_remediate(self, now: float, now_iso: str, prev: GapState) -> tuple[str | None, int]:
        """Auto-restart the ingestor on a confirmed stall. systemd-only, guarded by
        a cooldown and a max-consecutive cap. Returns (last_restart_at, restart_count)."""
        last, count = prev.last_restart_at, (prev.restart_count or 0)
        if not self.auto_restart or not self._unit_managed_fn(self.restart_unit):
            return last, count
        last_ts = _parse_iso(last)
        if last_ts is not None and (now - last_ts) < self.restart_cooldown:
            return last, count                      # cooldown
        if count >= self.max_restarts:
            return last, count                      # give up auto-restarting
        self._restart_fn(self.restart_unit)
        self._send(f"🔧 STALL REMEDIATION — restarted {self.restart_unit} "
                   f"(attempt {count + 1}/{self.max_restarts}) at {now_iso}")
        return now_iso, count + 1

    def check(self, now: float | None = None) -> GapState:
        """One evaluation cycle: detect gap/stall, alert on transitions, remediate, persist."""
        now = now if now is not None else self._now()
        age = latest_data_age_s(self.data_dirs, now)        # incl live .parquet.tmp
        row_age = latest_row_age_s(self.data_dirs, now)     # closed files only (diagnostic)
        prev = read_gap_state(self.state_path)
        now_iso = self._iso(now)

        tmp_path, tmp_size, tmp_grew_at, stall_age = self._track_tmp(prev, now_iso)

        # mtime gap = no file writes at all (dead process → systemd's job).
        # stall = mtime fresh but the .tmp isn't growing (zombie: alive, not writing).
        mtime_gap = is_gap(age, self.threshold)
        stalled = (not mtime_gap) and stall_age is not None and stall_age > self.stall_threshold
        gap = mtime_gap or stalled

        common = dict(age_s=age, row_age_s=row_age, last_check_at=now_iso, stalled=stalled,
                      tmp_path=tmp_path, tmp_size=tmp_size, tmp_grew_at=tmp_grew_at)

        # Intentional pause (nat stop): keep running + recording, but never page/restart.
        if is_paused(self.pause_file):
            st = GapState(gapping=False, paused=True,
                          last_restart_at=prev.last_restart_at, restart_count=0, **common)
            st.stalled = False
            write_gap_state(self.state_path, st)
            return st

        # Guarded auto-remediation runs every stalled cycle (subject to cooldown/cap).
        if stalled:
            restart_last, restart_count = self._maybe_remediate(now, now_iso, prev)
        else:
            restart_last, restart_count = prev.last_restart_at, 0

        if gap and not prev.gapping:
            if stalled:
                msg = (f"⚠️ STALLED — ingestor alive but .tmp not growing for "
                       f"{stall_age:.0f}s (threshold {self.stall_threshold:.0f}s) as of {now_iso}")
            else:
                msg = (f"⚠️ DATA GAP — no new file write for {age:.0f}s "
                       f"(threshold {self.threshold:.0f}s) as of {now_iso}")
            self._send(msg)
            st = GapState(gapping=True, gap_started_at=now_iso, last_alert_at=now_iso,
                          last_restart_at=restart_last, restart_count=restart_count, **common)
        elif gap and prev.gapping:
            st = GapState(gapping=True, gap_started_at=prev.gap_started_at,
                          last_alert_at=prev.last_alert_at,
                          last_restart_at=restart_last, restart_count=restart_count, **common)
        elif (not gap) and prev.gapping:
            dur = ""
            started = _parse_iso(prev.gap_started_at)
            if started is not None:
                dur = f" after {now - started:.0f}s"
            self._send(f"✅ DATA RECOVERED — feature ingestion flowing again{dur} ({now_iso})")
            st = GapState(gapping=False, last_restart_at=restart_last, restart_count=0, **common)
        else:
            st = GapState(gapping=False, last_restart_at=restart_last, restart_count=0, **common)

        write_gap_state(self.state_path, st)
        return st

    def status(self) -> dict:
        st = read_gap_state(self.state_path)
        d = st.to_dict()
        d["threshold_s"] = self.threshold
        d["row_threshold_s"] = self.row_threshold
        d["paused"] = is_paused(self.pause_file)
        d["daemon_healthy"] = healthy(self.config)
        d["stall_threshold_s"] = self.stall_threshold
        d["auto_restart"] = self.auto_restart
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
        if not telegram_configured():
            log.warning(
                "Telegram NOT configured — gap alerts are LOCAL-ONLY (%s + notify-send + "
                "`nat status`). Add TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID to .env for push alerts.",
                self.alert_log,
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
    # Pull TELEGRAM_* (and any secrets) from .env — cron/tmux children don't
    # inherit an interactive shell's environment.
    try:
        from config_utils import load_dotenv
        load_dotenv()
    except Exception:  # pragma: no cover
        pass
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
