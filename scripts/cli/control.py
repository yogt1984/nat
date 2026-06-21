"""`nat start|stop|status|doctor|build|run|15m|monitor|log|ing` — ingestor/process lifecycle.

The process-control handlers (start/stop/status/doctor/dashboard/build/run/15m/
monitor/log + the `ing` noun-group aliases) plus their private helpers. These call
into the gap/systemd/watchdog supervision helpers in `cli.ops`, so we re-export
everything via `__all__` and the `nat` entry shim does `from cli.control import *`.

The register/add_parser blocks for these groups stay inline in `nat`'s build_parser
(they're scattered), referencing these handlers through that star-import.
"""

from __future__ import annotations

import getpass
import json as _json
import os
import signal as sig
import subprocess
import sys
import time
from datetime import datetime, timezone

from cli.common import *  # noqa: F401,F403  (ROOT, RUST, BIN_ING, PY, colors, _sh/_exec/_cargo/_py/_pid/_p/_banner/_output/_ensure_release/_unwritable_data_dirs/_set_ingestion_paused/_sym/_json_mode/_TF_MAP/nat_paths/…)
from cli.ops import *  # noqa: F401,F403  (gap/systemd/watchdog helpers: _install_watchdog/_remove_watchdog/_ensure_gap_alert/_gap_monitor_state/_cron_daemon_running/_service_installed/_svc_active/_systemctl/INGESTOR_UNIT/GAP_UNIT/…)


def cmd_doctor(args):
    """Preflight for ingestion startup: data-dir ownership, binary, disk."""
    _banner("nat doctor — ingestion preflight")
    me = getpass.getuser()
    ok = True

    issues = _unwritable_data_dirs()
    if issues:
        ok = False
        _p("x", R, f"{len(issues)} data path(s) NOT writable by '{me}' — the ingestor will STALL silently:")
        for path, owner in issues:
            _p(" ", Y, f"{path}  (owned by {owner})")
        _p(" ", W, f"Fix:  sudo chown -R {me}:{me} {ROOT / 'data'}")
    else:
        _p("+", G, f"Data dirs writable by '{me}'")

    if BIN_ING.exists():
        _p("+", G, "Ingestor binary present")
    else:
        ok = False
        _p("x", R, "Ingestor binary missing — run: nat build")

    import shutil as _shutil
    du = _shutil.disk_usage(str(ROOT))
    free_pct = du.free / du.total * 100
    if free_pct >= 5:
        _p("+", G, f"Disk: {du.free / 1e9:.0f} GB free ({free_pct:.0f}%)")
    else:
        ok = False
        _p("x", R, f"Disk LOW: only {free_pct:.0f}% free")

    print()
    if ok:
        _p("*", G, "Preflight OK — safe to nat start")
    else:
        _p("x", R, "Preflight FAILED — fix the above before nat start")
    return 0 if ok else 1


# ── Top-level commands ───────────────────────────────────────────────────────

def _start_via_systemd(args):
    """`nat start` when systemd units are installed (idempotent)."""
    _set_ingestion_paused(False)
    _systemctl("start", INGESTOR_UNIT, GAP_UNIT)
    time.sleep(2)
    cmd_dashboard(args)
    if _svc_active(INGESTOR_UNIT):
        _p("*", G, f"Ingestor: active under systemd (PID {_pid()})")
        _p("~", G, "systemd supervises restart + boot-start (Restart=always, linger)")
        _p(" ", W, "Logs: journalctl --user -u nat-ingestor -f")
    else:
        _p("x", R, f"Ingestor unit did not start — journalctl --user -u {INGESTOR_UNIT}")
    return 0


def cmd_start(args):
    """Start ingestor with auto-restart watchdog + dashboard."""
    if _service_installed():
        return _start_via_systemd(args)
    pid = _pid()
    if pid:
        _p("*", G, f"Already running (PID {pid})")
        _set_ingestion_paused(False)
        _ensure_gap_alert()  # still ensure the data-gap monitor is up
        return

    if not BIN_ING.exists():
        _p("...", Y, "Binary not found, building...")
        cmd_build(args)

    # Preflight: a root-owned data dir (Docker-era) makes the native ingestor
    # stall silently — warn loudly with the fix before launching.
    issues = _unwritable_data_dirs()
    if issues:
        me = getpass.getuser()
        _p("x", R, f"PREFLIGHT: {len(issues)} data path(s) NOT writable by '{me}' — the ingestor will stall:")
        for path, owner in issues[:6]:
            _p(" ", Y, f"{path}  (owned by {owner})")
        _p(" ", W, f"Fix first:  sudo chown -R {me}:{me} {ROOT / 'data'}   (then re-run; see `nat doctor`)")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"ingestor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    subprocess.run(
        f"tmux new-session -d -s ingestor "
        f"'cd {RUST} && ./target/release/ing {ING_CFG} 2>&1 | tee {logfile}'",
        shell=True,
    )
    time.sleep(3)
    pid = _pid()
    if pid:
        _p("*", G, f"Ingestor started (PID {pid})")
        _p(" ", W, f"Log: {logfile}")
    else:
        _p("x", R, f"Failed to start. Check: cat {logfile}")
        return

    cmd_dashboard(args)
    _install_watchdog()
    _set_ingestion_paused(False)  # ingestion resumed → re-arm gap alerting
    _ensure_gap_alert()

    r = _sh("hostname -I")
    ip = r.stdout.strip().split()[0] if r.stdout.strip() else "localhost"
    print()
    print(f"  {BOLD}{'=' * 40}{W}")
    _p("*", G, "Ingestor collecting data")
    _p("o", G, "Dashboard serving")
    _p("~", G, "Watchdog auto-restarts on crash")
    print()
    _p(">", B, f"Dashboard: {BOLD}http://{ip}:8050{W}")
    print(f"  {BOLD}{'=' * 40}{W}")
    print()
    _p(" ", W, "Check anytime with: nat status")


def cmd_stop(args):
    """Stop ingestor and remove watchdog."""
    if _service_installed():
        # Intentional stop: mark paused (no false page) + stop the ingestor unit,
        # but LEAVE the gap-alert unit running so an unexpected death still alerts.
        _set_ingestion_paused(True)
        _systemctl("stop", INGESTOR_UNIT)
        subprocess.run("tmux kill-session -t nat-dashboard 2>/dev/null", shell=True)
        _p("-", Y, "Ingestor unit stopped (gap monitor still running, alerts paused)")
        _p(" ", W, "Re-arm: nat start   ·   Fully remove systemd: nat service uninstall")
        return 0
    pid = _pid()
    if pid:
        # The ingestor may be Docker-managed (a container process owned by root):
        # the host user can't signal it, and `restart: unless-stopped` would just
        # bring it back. Detect that and guide to the right tool instead of
        # crashing with a PermissionError.
        try:
            os.kill(pid, 0)  # permission probe — sends no signal
        except PermissionError:
            _p("x", Y, f"Ingestor (PID {pid}) is Docker-managed — `nat stop` can't signal it.")
            _p(" ", W, "To stop it:  nat docker down   (or: docker stop nat-ingestor)")
            _p(" ", W, "Ingestor and nat daemons left running (nothing changed).")
            return 0
        except ProcessLookupError:
            pid = None  # vanished between pgrep and the probe
    if pid:
        try:
            os.kill(pid, sig.SIGTERM)
            time.sleep(8)  # allow graceful shutdown (flush Parquet, close WS)
            if _pid():
                os.kill(pid, sig.SIGKILL)
                time.sleep(1)
            _p("-", Y, "Ingestor stopped")
        except PermissionError:
            _p("x", Y, f"Cannot signal ingestor PID {pid} (Docker-managed?) — try: nat docker down")
            return 0
        except ProcessLookupError:
            _p("-", W, "Ingestor already stopped")
    else:
        _p("-", W, "Ingestor was not running")

    # Mark the stop as INTENTIONAL so the gap daemon suppresses a page — but keep
    # the daemon (and its watchdog) RUNNING, so an *unexpected* ingestor death
    # later is still detected. (Previously `nat stop` tore down all monitoring,
    # which is exactly how a 3.4h outage went unnoticed.) `nat gap stop` fully
    # stops the monitor when that's truly intended.
    _set_ingestion_paused(True)

    for s in ["ingestor", "nat-monitor", "nat-dashboard"]:
        subprocess.run(f"tmux kill-session -t {s} 2>/dev/null", shell=True)
    _ensure_gap_alert()  # keep the data-gap monitor up across the stop

    _remove_watchdog()
    _p("-", W, "All cleaned up")


def cmd_status(args):
    """One-line health check."""
    pid = _pid()

    # Gather data stats
    data_info = {"files": 0, "size_mb": 0, "days": 0, "date_range": None, "last_write_s": None}
    if DATA_DEFAULT.exists():
        parquets = list(DATA_DEFAULT.rglob("*.parquet"))
        stats = [(f, f.stat()) for f in parquets]
        valid_stats = [(f, s) for f, s in stats if s.st_size > 0]
        total_size = sum(s.st_size for _, s in valid_stats)
        days = sorted(set(p.parent.name for p in parquets))
        data_info["files"] = len(valid_stats)
        data_info["size_mb"] = round(total_size / 1e6, 1)
        data_info["days"] = len(days)
        if days:
            data_info["date_range"] = [days[0], days[-1]]
        if stats:
            data_info["last_write_s"] = round(time.time() - max(s.st_mtime for _, s in stats), 1)

    svc = _service_installed()
    watchdog_active = "pgrep -x ing" in (_sh("crontab -l 2>/dev/null").stdout or "")
    cron_ok = _cron_daemon_running()
    dashboard_running = _sh("tmux has-session -t nat-dashboard 2>/dev/null").returncode == 0
    monitor = _gap_monitor_state()
    supervisor = "systemd" if svc else "tmux+cron"

    if _json_mode(args):
        _output({
            "ingestor": {"running": pid is not None, "pid": pid},
            "data": data_info,
            "supervisor": supervisor,
            "watchdog": watchdog_active,
            "cron_daemon": cron_ok,
            "gap_monitor": monitor,
            "dashboard": dashboard_running,
        }, args)
        return

    # Human output
    if pid:
        _p("*", G, f"Ingestor: RUNNING (PID {pid})")
    else:
        _p("*", R, "Ingestor: STOPPED")

    if data_info["files"] > 0:
        _p("+", B, f"Data: {data_info['files']} files, {data_info['size_mb']:.0f} MB, {data_info['days']} days")
        if data_info["date_range"]:
            _p(" ", W, f"Range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
        age = data_info["last_write_s"]
        if age is not None:
            if age < 300:
                _p("+", G, f"Last write: {age:.0f}s ago")
            elif age < 3600:
                _p("+", Y, f"Last write: {age / 60:.0f} min ago")
            else:
                _p("+", R, f"Last write: {age / 3600:.1f} hours ago")
    else:
        _p("+", R, "Data: no data directory")

    if svc:
        ing_active = _svc_active(INGESTOR_UNIT)
        _p("~", G if ing_active else R,
           f"Supervisor: systemd ({INGESTOR_UNIT} "
           f"{'active' if ing_active else 'INACTIVE'}, Restart=always, boot-start)")
    elif watchdog_active and cron_ok:
        _p("~", G, "Watchdog: active (tmux+cron)")
    elif watchdog_active and not cron_ok:
        _p("~", R, "Watchdog: crontab set but CRON DAEMON NOT RUNNING — restarts won't fire")
    else:
        _p("~", Y, "Watchdog: not installed (run: nat start, or nat service install)")

    # Data-gap monitor (the thing that should have caught the outage).
    if monitor["paused"]:
        _p("@", Y, "Data monitor: PAUSED (intentional stop) — alerts suppressed")
    elif not monitor["daemon_running"]:
        _p("@", R, "Data monitor: DAEMON DOWN — gaps undetected (run: nat gap start)")
    elif monitor.get("stalled"):
        rc = monitor.get("restart_count", 0)
        _p("@", R, f"Data monitor: STALLED — ingestor alive but not writing"
                   f"{f' (auto-restart x{rc})' if rc else ''}")
    elif monitor["gapping"]:
        age = monitor.get("age_s")
        det = f" ({age:.0f}s since last write)" if isinstance(age, (int, float)) else ""
        _p("@", R, f"Data monitor: GAPPING{det} — check ingestor")
    else:
        _p("@", G, "Data monitor: OK (gap daemon live)")

    if dashboard_running:
        _p("o", G, "Dashboard: running")
    else:
        _p("o", W, "Dashboard: not running (run: nat dashboard)")


def cmd_log(args):
    """Tail the latest ingestor log (journald when systemd-managed, else logfile)."""
    if _service_installed() and _svc_active(INGESTOR_UNIT):
        _p(">", B, "Following journalctl --user -u nat-ingestor (Ctrl+C to stop)")
        print()
        try:
            os.execvp("journalctl",
                      ["journalctl", "--user", "-u", INGESTOR_UNIT, "-f", "-n", "50"])
        except (KeyboardInterrupt, FileNotFoundError):
            return
    if not LOG_DIR.exists():
        _p("x", R, "No logs directory")
        return
    logs = sorted(LOG_DIR.glob("ingestor_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        _p("x", R, "No ingestor logs found")
        return
    _p(">", B, f"Tailing {logs[0].name} (Ctrl+C to stop)")
    print()
    try:
        os.execvp("tail", ["tail", "-f", "-n", "30", str(logs[0])])
    except KeyboardInterrupt:
        pass


# ── ing — ingestor noun-group (delegates to the top-level verbs) ───────────────

def cmd_ing_start(args):
    """nat ing start — start the local ingestor (alias of `nat start`)."""
    return cmd_start(args)


def cmd_ing_stop(args):
    """nat ing stop — stop the ingestor and daemons (alias of `nat stop`)."""
    return cmd_stop(args)


def cmd_ing_status(args):
    """nat ing status — ingestor + data health (alias of `nat status`)."""
    return cmd_status(args)


def cmd_ing_log(args):
    """nat ing log — tail the latest ingestor log (alias of `nat log`)."""
    return cmd_log(args)


def cmd_dashboard(args):
    """Start or show dashboard."""
    r = _sh("tmux has-session -t nat-dashboard 2>/dev/null")
    if r.returncode == 0:
        _p("o", G, "Dashboard already running")
    else:
        subprocess.run(
            f"tmux new-session -d -s nat-dashboard "
            f"'{PY} -m scripts.experiment.server --port 8050'",
            shell=True, cwd=str(ROOT),
        )
        time.sleep(2)
        _p("o", G, "Dashboard started")

    r = _sh("hostname -I")
    ip = r.stdout.strip().split()[0] if r.stdout.strip() else "localhost"
    _p(">", B, f"Open: http://{ip}:8050")


def cmd_monitor(args):
    """Live feature probe — stream computed features to the terminal (no ingestion).

    Runs the `show_features` binary: its own WebSocket, computes features, prints
    their instantaneous values at ~N Hz, persists nothing. The quickest "is the
    system functional?" check — works whether or not the persistent ingestor runs."""
    sf = RUST / "target" / "release" / "show_features"
    if not sf.exists():
        _p("...", Y, "show_features binary missing — building release binaries...")
        _ensure_release()
    sym = _sym(args)
    hz = getattr(args, "hz", 10)
    return _exec(f"./target/release/show_features {sym} {hz}", cwd=RUST).returncode


def cmd_monitor_tui(args):
    """Legacy rich dashboard (Redis-backed health/agent/features tabs)."""
    cmd = [PY, str(ROOT / "scripts" / "monitor.py")]
    if hasattr(args, "tab") and args.tab:
        cmd += ["--tab", str(args.tab)]
    subprocess.run(cmd, cwd=str(ROOT))


# ── Build commands ───────────────────────────────────────────────────────────

def cmd_build(args):
    """Build all release binaries."""
    _banner("Building release binaries")
    _cargo(
        "build --locked --release --bin ing --bin validate_api --bin validate_positions "
        "--bin validate_whales --bin validate_entropy --bin show_features --bin test_hypotheses"
    )


def cmd_build_debug(args):
    _banner("Building debug binary")
    _cargo("build --locked --bin ing")


def cmd_build_api(args):
    _banner("Building API server")
    _cargo("build --locked --release --bin nat-api")


def cmd_build_clean(args):
    _p("...", Y, "Cleaning build artifacts...")
    _cargo("clean")


def cmd_build_fmt(args):
    _cargo("fmt")


def cmd_build_lint(args):
    _cargo("clippy --locked -- -D warnings")


def cmd_build_check(args):
    _cargo("check --locked")


# ── Run commands (foreground, for development) ───────────────────────────────

def cmd_run(args):
    """Run ingestor in foreground."""
    _ensure_release()
    _banner("Running ingestor (foreground)")
    _sh("pkill -f 'target/.*ing.*config/ing.toml' 2>/dev/null")
    time.sleep(1)
    _exec(f"./target/release/ing {ING_CFG}", cwd=RUST)


def cmd_run_serve(args):
    """Run ingestor + dashboard in foreground."""
    _ensure_release()
    _banner("Running ingestor + dashboard")
    print(f"  Dashboard: http://localhost:8080\n")
    _exec(f"./target/release/ing {ING_CFG}", cwd=RUST,
          env={"ING_DASHBOARD_ENABLED": "true"})


def cmd_run_show(args):
    """Show real-time features in terminal."""
    _ensure_release()
    symbol = _sym(args)
    freq = getattr(args, 'freq', 1) or 1
    print(f"  Real-time features: {symbol} @ {freq} Hz  (Ctrl+C to stop)\n")
    _exec(f"./target/release/show_features {symbol} {freq}", cwd=RUST)


def cmd_run_tunnel(args):
    _p(">", B, "Starting Cloudflare tunnel to localhost:8080...")
    _exec("cloudflared tunnel --url http://localhost:8080")


# ── 15-Minute Experiment ─────────────────────────────────────────────────────

FIFTEEN_MIN = 15 * 60


def _today_data(args):
    """Resolve data dir: explicit --data, or today's date directory."""
    d = getattr(args, 'data', None)
    if d:
        return d
    today = datetime.now().strftime("%Y-%m-%d")
    return str(DATA_DEFAULT / today)


def _start_ingestor_bg():
    """Start ingestor as background process. Returns (Popen, logfile) or (None, None)."""
    if not BIN_ING.exists():
        _p("…", Y, "Binary not found, building release…")
        subprocess.run("cargo build --release --bin ing", shell=True, cwd=str(RUST))
        if not BIN_ING.exists():
            _p("x", R, "Build failed")
            return None, None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"15m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    lf = open(logfile, "w")
    proc = subprocess.Popen(
        [str(BIN_ING), str(ING_CFG)],
        cwd=str(RUST),
        stdout=lf, stderr=subprocess.STDOUT,
    )
    time.sleep(3)
    if proc.poll() is not None:
        _p("x", R, f"Ingestor died. Check: cat {logfile}")
        lf.close()
        return None, None
    return proc, logfile


def _stop_ingestor_bg(proc):
    """Gracefully stop a background ingestor process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _print_15m_window(label="Analyzed window"):
    """Print the time window of the latest 15m experiment (from its data_ref.json).

    Makes it obvious which 15-min slice was analyzed — two runs inside the same
    data-flush window analyze the *same* slice (the ingestor buffers a row group
    and rotates hourly, so freshly-ingested ticks aren't readable yet), which is
    why repeated runs can produce identical graphs."""
    ref = REPORTS_DIR / "smoke_test" / "latest" / "data_ref.json"
    if not ref.exists():
        return
    try:
        d = _json.loads(ref.read_text())
        tmin = datetime.fromtimestamp(d["ts_min"] / 1e9, tz=timezone.utc)
        tmax = datetime.fromtimestamp(d["ts_max"] / 1e9, tz=timezone.utc)
        dur = (d["ts_max"] - d["ts_min"]) / 1e9 / 60
        _p("→", B, f"{label}: {tmin:%Y-%m-%d %H:%M:%S}–{tmax:%H:%M:%S} UTC "
                   f"({dur:.1f} min, {d.get('rows', '?')} rows)")
    except Exception:
        pass


def cmd_15m(args):
    """Full 15-minute experiment: ingest → collect → stop → validate → profile → cluster → report."""
    duration = getattr(args, 'duration', FIFTEEN_MIN)
    _banner("15-MINUTE EXPERIMENT")

    # Capture start date before collection begins (midnight rollover safety)
    start_date = datetime.now().strftime("%Y-%m-%d")

    # Check if ingestor already running
    existing_pid = _pid()
    own_ingestor = False
    proc = None

    if existing_pid:
        _p("*", G, f"Ingestor already running (PID {existing_pid})")
    else:
        _p("→", B, "Starting ingestor…")
        proc, logfile = _start_ingestor_bg()
        if proc is None:
            return
        own_ingestor = True
        _p("*", G, f"Ingestor started (PID {proc.pid}), log: {logfile}")

    # Countdown
    _p("⏳", B, f"Collecting for {duration // 60} minutes…")
    print()
    t0 = time.time()
    try:
        while time.time() - t0 < duration:
            elapsed = int(time.time() - t0)
            remaining = duration - elapsed
            m, s = divmod(remaining, 60)
            print(f"\r    [{elapsed // 60:02d}:{elapsed % 60:02d} / "
                  f"{duration // 60:02d}:00]  {m}m {s}s left   ", end="", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n")
        _p("!", Y, "Interrupted early — analyzing what we have")

    print("\n")

    # Stop ingestor if we started it
    if own_ingestor:
        _p("→", B, "Stopping ingestor…")
        _stop_ingestor_bg(proc)
        _p("-", Y, "Ingestor stopped")
        print()

    # Analyze data — use start_date to avoid midnight rollover.
    # The ingestor writes to the date dir it started on; auto-trim picks the tail.
    data_dir = str(DATA_DEFAULT / start_date)
    _p("→", B, f"Analyzing: {data_dir}")
    print()
    extra = f" --duration {duration}"
    if getattr(args, 'skip_cluster', False):
        extra += " --skip-cluster"
    _py(f"scripts/15m_test.py run --data-dir {data_dir} --output reports/smoke_test -v{extra}")
    _print_15m_window()
    _p(" ", B, "Visualize: nat 15m viz")


def cmd_15m_offline(args):
    """Run analysis on existing data (no ingestion)."""
    data = _today_data(args)
    _banner("15-MINUTE ANALYSIS (OFFLINE)")
    _p("→", B, f"Data: {data}")
    print()
    extra = ""
    if getattr(args, 'skip_cluster', False):
        extra += " --skip-cluster"
    _py(f"scripts/15m_test.py run --data-dir {data} --output reports/smoke_test -v{extra}")
    _print_15m_window()
    _p(" ", B, "Visualize: nat 15m viz")


def cmd_15m_viz(args):
    """Visualize latest 15-minute experiment (quiet: prints only the saved PNG paths)."""
    sym = _sym(args)
    extra = ""
    data = getattr(args, 'data', None)
    extra += f" --data-dir {data}" if data else " --latest"
    window = getattr(args, 'window', None)
    if window:
        extra += f" --window {window}"
    page = getattr(args, 'page', 'all')
    extra += f" --page {page}"
    if getattr(args, 'open_after', True):
        extra += " --open"
    return _py(f"scripts/15m_visualize.py{extra} --symbol {sym} --quiet").returncode


def cmd_15m_test(args):
    """Run 15m unit tests."""
    _exec(f"cd {ROOT / 'scripts'} && {PY} -m pytest tests/test_15m_test.py -v")


# ── Log group ────────────────────────────────────────────────────────────────

def cmd_log_agent(args):
    """Tail agent daemon log."""
    # Check for agent logs in logs/ dir or stdout
    log_patterns = [
        LOG_DIR / "agent_*.log",
        ROOT / "data" / "agent" / "*.log",
    ]
    for pattern in log_patterns:
        logs = sorted(pattern.parent.glob(pattern.name), key=os.path.getmtime, reverse=True)
        if logs:
            _p(">", B, f"Tailing {logs[0].name}")
            os.execvp("tail", ["tail", "-f", "-n", "30", str(logs[0])])
            return
    _p("x", Y, "No agent logs found. Agent daemon logs to stdout by default.")
    _p(" ", W, "Start with: nat agent start 2>&1 | tee logs/agent.log")


def cmd_log_list(args):
    """List all log files with sizes and dates."""
    log_dirs = [LOG_DIR, ROOT / "data" / "agent"]
    entries = []
    for d in log_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.log"):
            s = f.stat()
            entries.append({
                "path": str(f.relative_to(ROOT)),
                "size_kb": round(s.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(s.st_mtime).isoformat(),
            })
    entries.sort(key=lambda e: e["modified"], reverse=True)

    if _json_mode(args):
        _output({"logs": entries, "count": len(entries)}, args)
        return

    print(f"\n  {BOLD}Log Files ({len(entries)}){W}\n")
    for e in entries:
        print(f"  {e['modified'][:19]}  {e['size_kb']:>8.1f} KB  {e['path']}")
    print()


__all__ = [
    # constants
    "FIFTEEN_MIN",
    # helpers
    "_start_via_systemd", "_start_ingestor_bg", "_stop_ingestor_bg",
    "_today_data", "_print_15m_window",
    # handlers
    "cmd_doctor", "cmd_start", "cmd_stop", "cmd_status", "cmd_log",
    "cmd_ing_start", "cmd_ing_stop", "cmd_ing_status", "cmd_ing_log",
    "cmd_dashboard", "cmd_monitor", "cmd_monitor_tui",
    "cmd_build", "cmd_build_debug", "cmd_build_api", "cmd_build_clean",
    "cmd_build_fmt", "cmd_build_lint", "cmd_build_check",
    "cmd_run", "cmd_run_serve", "cmd_run_show", "cmd_run_tunnel",
    "cmd_15m", "cmd_15m_offline", "cmd_15m_viz", "cmd_15m_test",
    "cmd_log_agent", "cmd_log_list",
]
