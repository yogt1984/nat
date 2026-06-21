"""`nat risk|gap|service|promotion|bridge` — ops daemons + systemd/cron supervision.

These five families share the gap/systemd supervision helpers. nat's own
start/stop/status/doctor handlers still call several of them, so `__all__`
re-exports every helper + constant they reference (nat does `from cli.ops import *`).
"""

from __future__ import annotations

import getpass
import json as _json
import os
import signal as sig
import subprocess
import sys
import time

from cli.common import (
    ROOT, ING_CFG, LOG_DIR, PY,
    G, R, Y, W, BOLD,
    GAP_TMUX, GAP_PIDFILE, GAP_STATE_FILE, PAUSE_FILE, GAP_WATCHDOG_MARKER,
    _set_ingestion_paused,
    _sh, _py, _p, _banner, _json_mode, _output,
)


# ── risk / kill-switch (T16) ───────────────────────────────────────────────
RISK_SCRIPT = "scripts/risk/kill_switch.py"


def _risk_ks(args=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from risk.kill_switch import KillSwitch
    return KillSwitch()


def cmd_risk_help(args=None):
    """Risk / kill-switch — group help."""
    print(f"""
  {BOLD}nat risk{W} — kill-switch daemon (T16); thresholds imported from ROADMAP Step 9

    status              show current halt state (level/reason/resume_at)
    resume [--confirm]  clear a halt (refuses kill_strategy; halt_review/halt need --confirm)
    start               run the kill-switch daemon (foreground; polls PnL/IC every 60s)
    stop                graceful shutdown (SIGTERM)

  daily >1% halt_24h · weekly DD >2% halt_review · monthly DD >5% kill_strategy · IC<0 5d halt
""")


def cmd_risk_status(args):
    """Show kill-switch halt state."""
    ks = _risk_ks(args)
    status = ks.get_status()
    if _json_mode(args):
        _output(status, args)
        return
    if status.get("halted"):
        _p("HALT", R, f"{status['level']} — {status.get('reason', '')}")
        if status.get("resume_at"):
            print(f"        resume_at: {status['resume_at']}")
        if status.get("triggered"):
            print(f"        triggered: {', '.join(status['triggered'])}")
    else:
        _p("✓", G, "Trading ACTIVE — no halt")
    print(f"        pnl source: {status.get('pnl_source')}")


def cmd_risk_resume(args):
    """Clear an active halt, honouring resume rules."""
    ks = _risk_ks(args)
    ok, msg = ks.resume(confirm=getattr(args, 'confirm', False))
    _p("✓" if ok else "✗", G if ok else R, msg)
    return 0 if ok else 1


def cmd_risk_start(args):
    _banner("Starting kill-switch daemon")
    _py(f"{RISK_SCRIPT} start")


def cmd_risk_stop(args):
    """Send SIGTERM to running kill-switch daemon."""
    import signal as sig
    r = _sh("pgrep -f 'kill_switch.py start'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to kill-switch (pid {pid})")
    else:
        _p("–", Y, "No running kill-switch daemon found")


# ── gap-alert / ingestion freshness (T0b) ─────────────────────────────────
GAP_SCRIPT = "scripts/ops/gap_alert.py"


def _gap_alerter(args=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from ops.gap_alert import GapAlerter
    return GapAlerter()


def cmd_gap_help(args=None):
    """Gap-alert — group help."""
    print(f"""
  {BOLD}nat gap{W} — data-gap alert daemon (T0b); Telegram page when ingestion stalls

    status     show current gap state (gapping / seconds since last write)
    check      one-shot freshness check (exit 1 if gapping) — for cron/CI
    start      run the gap-alert daemon (foreground; polls data freshness)
    stop       graceful shutdown (SIGTERM)

  Pages once on gap-open and once on recovery. Read-only re su-35 (stats files only).
""")


def cmd_gap_status(args):
    """Show data-gap state."""
    a = _gap_alerter(args)
    st = a.status()
    if _json_mode(args):
        _output(st, args)
        return
    if st.get("gapping"):
        _p("GAP", R, f"no data for {st.get('age_s') or 0:.0f}s (threshold {st.get('threshold_s'):.0f}s)")
        if st.get("gap_started_at"):
            print(f"        since: {st['gap_started_at']}")
    else:
        age = st.get("age_s")
        _p("✓", G, f"Ingestion fresh ({age:.0f}s ago)" if age is not None else "No data yet")
    print(f"        watching: {', '.join(st.get('data_dirs', []))}")


def cmd_gap_check(args):
    """One-shot freshness check; exit 1 if gapping."""
    a = _gap_alerter(args)
    st = a.check()
    if _json_mode(args):
        _output(st.to_dict(), args)
    else:
        _p("GAP" if st.gapping else "✓", R if st.gapping else G,
           f"age {st.age_s:.0f}s" if st.age_s is not None else "no data")
    return 1 if st.gapping else 0


def cmd_gap_start(args):
    _banner("Starting gap-alert daemon")
    _py(f"{GAP_SCRIPT} start")


def cmd_gap_stop(args):
    """Send SIGTERM to running gap-alert daemon."""
    import signal as sig
    r = _sh("pgrep -f 'gap_alert.py start'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to gap-alert (pid {pid})")
    else:
        _p("–", Y, "No running gap-alert daemon found")


def cmd_gap_watchdog(args):
    """Install (or --remove) the gap-alert cron watchdog (auto-restart every 5 min)."""
    if getattr(args, 'remove', False):
        _remove_gap_watchdog()
        _p("~", W, "Gap-alert watchdog removed")
    else:
        _ensure_gap_alert()  # start daemon if down + install the watchdog cron


# ── promotion daemon (T14) ─────────────────────────────────────────────────
PROMOTION_SCRIPT = "scripts/promotion_daemon.py"


def _promotion(args=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from promotion_daemon import PromotionDaemon
    return PromotionDaemon(dry_run=getattr(args, 'dry_run', False))


def cmd_promotion_help(args=None):
    """Promotion daemon — group help."""
    print(f"""
  {BOLD}nat promotion{W} — signal promotion daemon (T14)
  Drives DISCOVERED→VALIDATED→PAPER_TRADING→APPROVAL_PENDING (stops at the human gate)

    status            signals by state + clean-day guard
    once [--dry-run]  run a single cycle (--dry-run logs intended transitions, no DB writes)
    start             run the daemon (foreground; polls every 300s)
    stop              graceful shutdown (SIGTERM)

  Gates imported from alpha.toml (g4_*/g8_*); never promotes APPROVAL_PENDING→LIVE.
""")


def cmd_promotion_status(args):
    """Signals by state + the clean-day guard."""
    st = _promotion(args).status()
    if _json_mode(args):
        _output(st, args)
        return
    print(f"\n  {BOLD}Promotion{W} — clean days: {st['clean_days_now']} (need {st['min_clean_days']})\n")
    if not st.get("by_state"):
        print("    (no signals — try: nat lifecycle seed)\n")
        return
    for state, n in st["by_state"].items():
        print(f"    {state:<18} {n}")
    print()


def cmd_promotion_once(args):
    """Run a single promotion cycle."""
    summary = _promotion(args).run_cycle()
    if _json_mode(args):
        _output(summary, args)
        return
    mode = " (dry-run)" if getattr(args, 'dry_run', False) else ""
    _p("✓", G, f"Cycle complete{mode}: {summary['transitions']} transitions, {summary['errors']} errors")


def cmd_promotion_start(args):
    _banner("Starting promotion daemon")
    _py(f"{PROMOTION_SCRIPT} start")


def cmd_promotion_stop(args):
    """Send SIGTERM to running promotion daemon."""
    import signal as sig
    r = _sh("pgrep -f 'promotion_daemon.py start'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to promotion daemon (pid {pid})")
    else:
        _p("–", Y, "No running promotion daemon found")


# ── signal bridge daemon (T17) ──────────────────────────────────────────────
BRIDGE_SCRIPT = "scripts/execution/signal_bridge.py"


def _bridge(args=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from execution.signal_bridge import SignalBridgeDaemon, load_config
    cfg = load_config()
    dry = getattr(args, 'dry_run', False) or cfg.get("mode", "dry-run") != "live"
    return SignalBridgeDaemon(config=cfg, dry_run=dry)


def cmd_bridge_help(args=None):
    """Signal bridge daemon — group help."""
    print(f"""
  {BOLD}nat bridge{W} — signal bridge daemon (T17); executes LIVE signals under risk gating

    status            mode / halt state / LIVE signals
    once [--dry-run]  run a single execution cycle (dry-run = no orders, the default)
    start             run the bridge daemon (foreground; polls every 5 min)
    stop              graceful shutdown (SIGTERM)

  Checks halt_state.json every cycle (cannot be skipped); risk-parity sizing.
  Dry-run by default — no live orders without mode=live in config/execution.toml.
""")


def cmd_bridge_status(args):
    """Bridge mode / halt state / LIVE signals."""
    st = _bridge(args).status()
    if _json_mode(args):
        _output(st, args)
        return
    _p("HALT" if st["halted"] else "✓", R if st["halted"] else G,
       f"mode={st['mode']}  halt={st['halt_level'] or 'none'}  LIVE signals={len(st['live_signals'])}")
    if st["live_signals"]:
        print(f"        {', '.join(st['live_signals'])}")


def cmd_bridge_once(args):
    """Run a single bridge execution cycle."""
    summary = _bridge(args).run_cycle()
    if _json_mode(args):
        _output(summary, args)
        return
    if summary["halted"]:
        _p("HALT", R, "cycle skipped (kill-switch halt active)")
    else:
        _p("✓", G, f"cycle: {len(summary['picked_up'])} LIVE signals, {summary['fills']} fills")


def cmd_bridge_start(args):
    _banner("Starting signal-bridge daemon")
    _py(f"{BRIDGE_SCRIPT} start")


def cmd_bridge_stop(args):
    """Send SIGTERM to running signal-bridge daemon."""
    import signal as sig
    r = _sh("pgrep -f 'signal_bridge.py start'")
    if r.returncode == 0 and r.stdout.strip():
        pid = int(r.stdout.strip().split("\n")[0])
        os.kill(pid, sig.SIGTERM)
        _p("✓", G, f"Sent SIGTERM to signal-bridge (pid {pid})")
    else:
        _p("–", Y, "No running signal-bridge daemon found")


# ── Watchdog ─────────────────────────────────────────────────────────────────

def _install_watchdog():
    r = _sh("crontab -l 2>/dev/null")
    existing = r.stdout or ""
    if "pgrep -x ing" in existing:
        _p("~", G, "Watchdog already installed")
        return
    cron_line = (
        f"*/5 * * * * pgrep -x ing || "
        f"(cd {ROOT} && cd rust && ./target/release/ing {ING_CFG} "
        f">> {LOG_DIR}/cron_restart.log 2>&1 &)"
    )
    new_cron = existing.rstrip() + "\n" + cron_line + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(input=new_cron)
    if proc.returncode == 0:
        _p("~", G, "Watchdog installed (checks every 5 min)")
    else:
        _p("~", Y, "Could not install watchdog cron")


def _remove_watchdog():
    r = _sh("crontab -l 2>/dev/null")
    existing = r.stdout or ""
    if "pgrep -x ing" not in existing:
        return
    lines = [l for l in existing.split("\n") if "pgrep -x ing" not in l]
    new_cron = "\n".join(lines).strip() + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(input=new_cron)
    _p("~", W, "Watchdog removed")


# ── Gap-alert daemon (T0b) ─────────────────────────────────────────────────
# GAP_* path constants + _set_ingestion_paused now live in cli/common.py.

def _cron_daemon_running():
    """True if a cron daemon is actually running (the crontab text means nothing
    if cron isn't up — a silent way watchdogs become no-ops)."""
    return _sh("pgrep -x cron >/dev/null 2>&1 || pgrep -x crond >/dev/null 2>&1").returncode == 0


def _gap_monitor_state():
    """Gap-alert daemon + state summary for `nat status` (read-only)."""
    state = {}
    try:
        if GAP_STATE_FILE.exists():
            state = _json.loads(GAP_STATE_FILE.read_text())
    except Exception:
        state = {}
    return {
        "daemon_running": _gap_running(),
        "paused": PAUSE_FILE.exists(),
        "gapping": bool(state.get("gapping")),
        "stalled": bool(state.get("stalled")),
        "restart_count": state.get("restart_count", 0),
        "age_s": state.get("age_s"),
        "row_age_s": state.get("row_age_s"),
        "last_check_at": state.get("last_check_at"),
    }


def _gap_running():
    """True if the gap-alert pidfile points at a live process (no self-match)."""
    try:
        os.kill(int(GAP_PIDFILE.read_text().strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def _start_gap_alert():
    if _gap_running():
        _p("~", G, "Gap-alert: already running")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"tmux new-session -d -s {GAP_TMUX} "
        f"'cd {ROOT} && {PY} {GAP_SCRIPT} start >> {LOG_DIR}/gap_alert.log 2>&1'",
        shell=True,
    )
    time.sleep(1)
    if _gap_running():
        _p("~", G, "Gap-alert: started (pages on ingestion stalls)")
    else:
        _p("~", Y, "Gap-alert: failed to start (see logs/gap_alert.log)")


def _install_gap_watchdog():
    r = _sh("crontab -l 2>/dev/null")
    existing = r.stdout or ""
    if GAP_WATCHDOG_MARKER in existing:
        return
    # pidfile-based liveness (not pgrep) so the cron line never self-matches.
    cron_line = (
        f"*/5 * * * * cd {ROOT} && "
        f"(kill -0 $(cat {GAP_PIDFILE} 2>/dev/null) 2>/dev/null || "
        f"tmux new-session -d -s {GAP_TMUX} "
        f"'{PY} {GAP_SCRIPT} start >> {LOG_DIR}/gap_alert.log 2>&1') {GAP_WATCHDOG_MARKER}"
    )
    new_cron = existing.rstrip() + "\n" + cron_line + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(input=new_cron)
    if proc.returncode == 0:
        _p("~", G, "Gap-alert watchdog installed (checks every 5 min)")
    else:
        _p("~", Y, "Could not install gap-alert watchdog cron")


def _remove_gap_watchdog():
    r = _sh("crontab -l 2>/dev/null")
    existing = r.stdout or ""
    if GAP_WATCHDOG_MARKER not in existing:
        return
    lines = [l for l in existing.split("\n") if GAP_WATCHDOG_MARKER not in l]
    new_cron = "\n".join(lines).strip() + "\n"
    proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(input=new_cron)


def _ensure_gap_alert():
    """Start the gap-alert daemon (if not running) and install its watchdog."""
    _start_gap_alert()
    _install_gap_watchdog()


# ── systemd --user service supervision (T3) ────────────────────────────────
INGESTOR_UNIT = "nat-ingestor.service"
GAP_UNIT = "nat-gap-alert.service"


def _user_unit_dir():
    sys.path.insert(0, str(ROOT / "scripts"))
    from ops.systemd_units import unit_dir
    return unit_dir()


def _service_installed():
    """True once the ingestor unit file exists (systemd is the supervisor)."""
    return (_user_unit_dir() / INGESTOR_UNIT).exists()


def _systemctl(*args, check=False):
    return _sh("systemctl --user " + " ".join(args))


def _svc_active(unit):
    return _systemctl("is-active", unit).stdout.strip() == "active"


def _svc_enabled(unit):
    return _systemctl("is-enabled", unit).stdout.strip() == "enabled"


def cmd_service_install(args):
    """Install + enable systemd --user units for the ingestor and gap daemon.

    Replaces the tmux+cron self-healing with systemd (Restart=always + boot
    persistence via linger). Hands the running ingestor over with a brief restart;
    the window is marked paused so the gap daemon doesn't false-page."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from ops.systemd_units import render_units, unit_dir

    if _sh("which systemctl").returncode != 0:
        _p("x", R, "systemctl not found — this host has no systemd")
        return 1

    _banner("Installing nat systemd --user services")
    # Boot persistence without an active login.
    _sh(f"loginctl enable-linger {getpass.getuser()}")

    udir = unit_dir()
    udir.mkdir(parents=True, exist_ok=True)
    for name, text in render_units(python=PY).items():
        (udir / name).write_text(text)
        _p("+", G, f"Wrote {udir / name}")
    _systemctl("daemon-reload")

    # Handover: pause (no false page), tear down the tmux+cron supervisor so it
    # can't double-run the ingestor, then let systemd take over.
    _set_ingestion_paused(True)
    _remove_watchdog()
    _remove_gap_watchdog()
    for s in ["ingestor", GAP_TMUX]:
        subprocess.run(f"tmux kill-session -t {s} 2>/dev/null", shell=True)
    if _gap_running():
        try:
            os.kill(int(GAP_PIDFILE.read_text().strip()), sig.SIGTERM)
        except (ValueError, ProcessLookupError, FileNotFoundError):
            pass
    time.sleep(2)

    _systemctl("enable", "--now", INGESTOR_UNIT, GAP_UNIT)
    time.sleep(3)
    _set_ingestion_paused(False)

    ok = _svc_active(INGESTOR_UNIT)
    _p("*" if ok else "x", G if ok else R,
       f"{INGESTOR_UNIT}: {'active' if ok else 'NOT active — journalctl --user -u ' + INGESTOR_UNIT}")
    _p("~", G if _svc_active(GAP_UNIT) else Y, f"{GAP_UNIT}: {_systemctl('is-active', GAP_UNIT).stdout.strip()}")
    _p(" ", W, "systemd now supervises both (Restart=always, starts on boot via linger).")
    _p(" ", W, "Logs: journalctl --user -u nat-ingestor -f   ·   Undo: nat service uninstall")
    return 0 if ok else 1


def cmd_service_uninstall(args):
    """Disable + remove the systemd units and restore the tmux+cron path."""
    _banner("Removing nat systemd --user services")
    _systemctl("disable", "--now", INGESTOR_UNIT, GAP_UNIT)
    udir = _user_unit_dir()
    for name in (INGESTOR_UNIT, GAP_UNIT):
        p = udir / name
        if p.exists():
            p.unlink()
            _p("-", Y, f"Removed {p}")
    _systemctl("daemon-reload")
    _p(" ", W, "systemd units removed. `nat start` again uses the tmux+cron path.")
    return 0


def cmd_service_status(args):
    """Show systemd unit state (active/enabled) + linger."""
    installed = _service_installed()
    linger = "yes" in _sh(f"loginctl show-user {getpass.getuser()} -p Linger 2>/dev/null").stdout.lower()
    data = {
        "installed": installed,
        "linger": linger,
        "units": {u: {"active": _svc_active(u), "enabled": _svc_enabled(u)}
                  for u in (INGESTOR_UNIT, GAP_UNIT)} if installed else {},
    }

    def _human(d):
        if not d["installed"]:
            _p("o", W, "systemd services: not installed (using tmux+cron). Install: nat service install")
            return
        _p("*", G, f"systemd services installed (boot persistence/linger: {'on' if d['linger'] else 'OFF'})")
        for u, s in d["units"].items():
            col = G if s["active"] else R
            _p("~", col, f"{u}: {'active' if s['active'] else 'inactive'}, "
                         f"{'enabled' if s['enabled'] else 'disabled'}")

    _output(data, args, _human)
    return 0


def cmd_service_restart(args):
    """Restart a unit (ingestor|gap|all)."""
    if not _service_installed():
        _p("x", R, "systemd services not installed (nat service install)")
        return 1
    which = getattr(args, "target", "all")
    units = {"ingestor": [INGESTOR_UNIT], "gap": [GAP_UNIT],
             "all": [INGESTOR_UNIT, GAP_UNIT]}.get(which, [INGESTOR_UNIT, GAP_UNIT])
    _systemctl("restart", *units)
    for u in units:
        _p("~", G if _svc_active(u) else R, f"{u}: {_systemctl('is-active', u).stdout.strip()}")
    return 0


def register(sub):
    # ── risk / kill-switch daemon (T16) ──
    risk_p = sub.add_parser('risk', help='Kill-switch daemon + halt control (T16)')
    risk_p.set_defaults(func=cmd_risk_help)
    risksub = risk_p.add_subparsers(dest='subcmd')
    rkst = risksub.add_parser('status', help='Show current halt state')
    rkst.add_argument('--json', action='store_true', help='JSON output')
    rkst.set_defaults(func=cmd_risk_status)
    rkr = risksub.add_parser('resume',
        help='Clear a halt (kill_strategy refused; halt_review/halt need --confirm)')
    rkr.add_argument('--confirm', action='store_true', help='Actually clear the halt')
    rkr.set_defaults(func=cmd_risk_resume)
    rksta = risksub.add_parser('start', help='Run the kill-switch daemon (foreground)')
    rksta.set_defaults(func=cmd_risk_start)
    rksto = risksub.add_parser('stop', help='Graceful shutdown (SIGTERM)')
    rksto.set_defaults(func=cmd_risk_stop)

    # ── gap-alert / ingestion freshness daemon (T0b) ──
    gap_p = sub.add_parser('gap', help='Data-gap alert daemon (Telegram page on ingestion stall, T0b)')
    gap_p.set_defaults(func=cmd_gap_help)
    gapsub = gap_p.add_subparsers(dest='subcmd')
    gpst = gapsub.add_parser('status', help='Show current gap state')
    gpst.add_argument('--json', action='store_true', help='JSON output')
    gpst.set_defaults(func=cmd_gap_status)
    gpck = gapsub.add_parser('check', help='One-shot freshness check (exit 1 if gapping)')
    gpck.add_argument('--json', action='store_true', help='JSON output')
    gpck.set_defaults(func=cmd_gap_check)
    gpsta = gapsub.add_parser('start', help='Run the gap-alert daemon (foreground)')
    gpsta.set_defaults(func=cmd_gap_start)
    gpsto = gapsub.add_parser('stop', help='Graceful shutdown (SIGTERM)')
    gpsto.set_defaults(func=cmd_gap_stop)
    gpwd = gapsub.add_parser('watchdog', help='Install cron watchdog (auto-restart every 5 min)')
    gpwd.add_argument('--remove', action='store_true', help='Remove the watchdog cron')
    gpwd.set_defaults(func=cmd_gap_watchdog)

    # ── service (systemd --user supervision, T3) ──
    svc_p = sub.add_parser('service',
                           help='systemd --user supervision: reboot-proof ingestor + gap daemon')
    svc_p.set_defaults(func=lambda a: svc_p.print_help())
    svcsub = svc_p.add_subparsers(dest='subcmd')
    svcsub.add_parser('install',
                      help='Install + enable units (replaces tmux+cron; brief ingestor restart)'
                      ).set_defaults(func=cmd_service_install)
    svcsub.add_parser('uninstall', help='Remove units, restore the tmux+cron path').set_defaults(func=cmd_service_uninstall)
    svcst = svcsub.add_parser('status', help='Unit active/enabled + linger state')
    svcst.add_argument('--json', action='store_true', help='JSON output')
    svcst.set_defaults(func=cmd_service_status)
    svcr = svcsub.add_parser('restart', help='Restart a unit')
    svcr.add_argument('target', nargs='?', default='all', choices=['ingestor', 'gap', 'all'])
    svcr.set_defaults(func=cmd_service_restart)

    # ── promotion daemon (T14) ──
    prom_p = sub.add_parser('promotion', help='Signal promotion daemon (lifecycle automation, T14)')
    prom_p.set_defaults(func=cmd_promotion_help)
    promsub = prom_p.add_subparsers(dest='subcmd')
    pmst = promsub.add_parser('status', help='Signals by state + clean-day guard')
    pmst.add_argument('--json', action='store_true', help='JSON output')
    pmst.set_defaults(func=cmd_promotion_status)
    pmon = promsub.add_parser('once', help='Run a single promotion cycle')
    pmon.add_argument('--dry-run', action='store_true', help='Log intended transitions without applying')
    pmon.add_argument('--json', action='store_true', help='JSON output')
    pmon.set_defaults(func=cmd_promotion_once)
    pmsta = promsub.add_parser('start', help='Run the promotion daemon (foreground)')
    pmsta.set_defaults(func=cmd_promotion_start)
    pmsto = promsub.add_parser('stop', help='Graceful shutdown (SIGTERM)')
    pmsto.set_defaults(func=cmd_promotion_stop)

    # ── signal bridge daemon (T17) ──
    br_p = sub.add_parser('bridge', help='Signal bridge daemon — executes LIVE signals under risk gating (T17)')
    br_p.set_defaults(func=cmd_bridge_help)
    brsub = br_p.add_subparsers(dest='subcmd')
    brst = brsub.add_parser('status', help='Mode / halt state / LIVE signals')
    brst.add_argument('--json', action='store_true', help='JSON output')
    brst.set_defaults(func=cmd_bridge_status)
    bron = brsub.add_parser('once', help='Run a single execution cycle')
    bron.add_argument('--dry-run', action='store_true', help='No orders (the default)')
    bron.add_argument('--json', action='store_true', help='JSON output')
    bron.set_defaults(func=cmd_bridge_once)
    brsta = brsub.add_parser('start', help='Run the bridge daemon (foreground)')
    brsta.set_defaults(func=cmd_bridge_start)
    brsto = brsub.add_parser('stop', help='Graceful shutdown (SIGTERM)')
    brsto.set_defaults(func=cmd_bridge_stop)


__all__ = [
    # constants
    "RISK_SCRIPT", "GAP_SCRIPT", "PROMOTION_SCRIPT", "BRIDGE_SCRIPT",
    "INGESTOR_UNIT", "GAP_UNIT",
    # helpers (several referenced by nat's start/stop/status/doctor)
    "_risk_ks", "_gap_alerter", "_promotion", "_bridge", "_user_unit_dir",
    "_install_watchdog", "_remove_watchdog", "_cron_daemon_running",
    "_gap_monitor_state", "_gap_running", "_start_gap_alert",
    "_install_gap_watchdog", "_remove_gap_watchdog", "_ensure_gap_alert",
    "_service_installed", "_systemctl", "_svc_active", "_svc_enabled",
    # handlers
    "cmd_risk_help", "cmd_risk_status", "cmd_risk_resume", "cmd_risk_start", "cmd_risk_stop",
    "cmd_gap_help", "cmd_gap_status", "cmd_gap_check", "cmd_gap_start", "cmd_gap_stop", "cmd_gap_watchdog",
    "cmd_promotion_help", "cmd_promotion_status", "cmd_promotion_once",
    "cmd_promotion_start", "cmd_promotion_stop",
    "cmd_bridge_help", "cmd_bridge_status", "cmd_bridge_once", "cmd_bridge_start", "cmd_bridge_stop",
    "cmd_service_install", "cmd_service_uninstall", "cmd_service_status", "cmd_service_restart",
    "register",
]
