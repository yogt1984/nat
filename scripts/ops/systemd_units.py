"""Render systemd `--user` unit files for the NAT ingestor + gap-alert daemon.

Pure (no host writes) so it is unit-testable. `nat service install` writes the
returned texts to ``~/.config/systemd/user/`` and enables them. Paths and env
come from ``nat_paths`` so the units are correct whether run from a dev checkout
or an installed prefix; the Rust ``ing`` reads ``NAT_DATA_DIR``/``NAT_TRADE_DIR``
from this env, so systemd-managed ingestion writes to the resolved data dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import nat_paths
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import nat_paths

INGESTOR_UNIT = "nat-ingestor.service"
GAP_UNIT = "nat-gap-alert.service"
CANDLE_UNIT = "nat-candle-refresh.service"
CANDLE_TIMER = "nat-candle-refresh.timer"
L2_UNIT = "nat-l2-sampler.service"
POSITION_UNIT = "nat-position-sampler.service"

#: Intervals the daily refresh sweeps. Ordered cheap→expensive so a truncated run still
#: captures the perishable one first: the venue keeps ~5000 bars per interval, so 1m
#: history expires in ~3.5 days and is the only one that can be lost permanently
#: (FINDINGS §7.1). 1h/15m can always be re-fetched later; 1m cannot.
CANDLE_INTERVALS = ("1m", "5m", "15m", "1h")


def _env_lines(extra: dict[str, str] | None = None) -> str:
    env = dict(nat_paths.as_env())
    if extra:
        env.update(extra)
    return "\n".join(f'Environment="{k}={v}"' for k, v in env.items())


def render_units(python: str | None = None) -> dict[str, str]:
    """Return {unit_filename: file_text} for the ingestor and gap-alert daemon."""
    py = python or sys.executable
    root = nat_paths.install_root()
    rust = root / "rust"
    bin_ing = rust / "target" / "release" / "ing"
    ing_cfg = nat_paths.config_dir() / "ing.toml"
    gap_script = root / "scripts" / "ops" / "gap_alert.py"

    ingestor = f"""\
[Unit]
Description=NAT Hyperliquid ingestor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={rust}
ExecStart={bin_ing} {ing_cfg}
{_env_lines({"RUST_LOG": "info"})}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""

    gap = f"""\
[Unit]
Description=NAT data-gap alert daemon
After=network-online.target

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={py} {gap_script} start
{_env_lines()}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

    # XS-7 — daily candle refresh. A oneshot sweep, not a daemon: it runs, reports, exits.
    fetch_script = root / "scripts" / "data" / "fetch_candles.py"
    # 1m first (see CANDLE_INTERVALS); `;` not `&&` so one interval's failure does not
    # cancel the rest — a 5m outage must not cost the perishable 1m pull.
    sweeps = " ; ".join(
        f"{py} {fetch_script} --universe --interval {iv} --days {days}"
        for iv, days in zip(CANDLE_INTERVALS, (3, 17, 52, 90))
    )
    # XS-10: after the candles land, re-measure the rotation and append a trajectory point.
    # §7.8's conclusion is "~325 rebalances needed, 83 held"; without this the wait is a note
    # someone has to remember rather than a measurement that takes itself.
    nat_bin = root / "nat"
    sweeps = sweeps + f" ; {py} {nat_bin} xs trajectory --record"

    candle = f"""\
[Unit]
Description=NAT daily candle refresh + rotation trajectory (XS-1/XS-7/XS-10)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart=/bin/sh -c "{sweeps}"
{_env_lines()}
# No Restart= : a oneshot sweep that failed should wait for the next timer window
# rather than hammer the venue in a loop.
TimeoutStartSec=7200
"""

    candle_timer = f"""\
[Unit]
Description=NAT daily candle refresh timer

[Timer]
OnCalendar=*-*-* 03:17:00
# The venue keeps only ~5000 bars per interval, so 1m candles expire in ~3.5 days and
# a missed window is data no backfill can recover (FINDINGS §7.1). Persistent=true runs
# the sweep on next boot if the box was off when it was due.
Persistent=true
RandomizedDelaySec=300
Unit={CANDLE_UNIT}

[Install]
WantedBy=timers.target
"""

    # XS-8 — L2 sampler. A long-lived loop, NOT a timer like the candle refresh: its
    # product is the intraday *distribution* of half-spreads (a single book is n=1, the
    # error PROC-20 corrected in LF7's priors), so a process that dies at 03:00 and waits
    # for tomorrow leaves a hole in the very variation being measured. Hence Restart=always.
    l2_script = root / "scripts" / "data" / "fetch_l2.py"
    l2 = f"""\
[Unit]
# systemd only honours StartLimitIntervalSec in [Unit]; it was in [Service]
# and silently ignored, so the restart limiter was never actually disabled.
StartLimitIntervalSec=0
Description=NAT L2 order-book sampler (XS-8 — universe half-spread distribution)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={py} -u {l2_script} --loop --every 300
{_env_lines()}
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
"""

    # WP-2 — position sampler. The most schedule-critical unit here: WP-5 needs ≥90 days of
    # accrual, so the verdict date (2026-11-08) slips one-for-one with every day this is not
    # running, and no later effort buys the days back (FINAL_PLAN §2, same shape as the XS-7
    # retention cap). A loop rather than a timer because a wallet's *transitions* are the
    # signal — a daily snapshot cannot see a position opened and closed between windows.
    positions_script = root / "scripts" / "data" / "fetch_positions.py"
    positions = f"""\
[Unit]
# systemd only honours StartLimitIntervalSec in [Unit]; it was in [Service]
# and silently ignored, so the restart limiter was never actually disabled.
StartLimitIntervalSec=0
Description=NAT position sampler (WP-2 — wallet positioning panel, ≥90d accrual)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={py} -u {positions_script} --loop --every 900
{_env_lines()}
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
"""

    return {INGESTOR_UNIT: ingestor, GAP_UNIT: gap,
            CANDLE_UNIT: candle, CANDLE_TIMER: candle_timer,
            L2_UNIT: l2, POSITION_UNIT: positions}


def unit_dir() -> Path:
    """Where `nat service install` writes the units (~/.config/systemd/user)."""
    xdg = nat_paths._xdg("XDG_CONFIG_HOME", Path.home() / ".config")
    return xdg / "systemd" / "user"
