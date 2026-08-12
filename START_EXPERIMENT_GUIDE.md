# START_EXPERIMENT_GUIDE — dedicated NAT node in one command

How to turn a fresh machine (e.g. a dedicated computer at home) into an
unattended NAT data node. The whole point is **data continuity** — the binding
constraint of the project: candle history is capped at ~5000 bars/interval and
L2 is never served historically, so every hour a node is down is unrecoverable.

## Setup — fresh machine

```bash
git clone git@github.com:yogt1984/nat.git && cd nat
python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt   # python deps
nat build                # release build of the ingestor binaries (needs Rust toolchain)
nat service install      # ← the one command
```

## What `nat service install` does

Installs and enables **systemd --user units with linger**: everything starts on
boot without a login and restarts on crash (`Restart=always`).

| Unit | What it runs | Why it matters |
|---|---|---|
| `nat-ingestor.service` | the 236-feature parquet stream | redundancy for the frozen su-35 box, without touching it |
| `nat-gap-alert.service` | the freshness watchdog | threshold is 900 s (> parquet rotation cadence) since the 2026-08-11 fix, so it won't restart-loop a healthy ingestor |
| `nat-candle-refresh.timer` | periodic candle fetch | the only defence against the ~5000-bar retention cap — an hour not fetched is unrecoverable |
| `nat-l2-sampler.service` | XS-8 book-snapshot sweeps | the data behind every spread/depth measurement (and the frozen `reproduce/slice/`) |

## Verify

```bash
nat service status
nat status
journalctl --user -u nat-ingestor -f    # live log
```

## The deliberate omission — the WP-2 position sampler

`nat service install` **writes** `nat-position-sampler.service` but does
**not enable it**, on purpose. The WP-2 position clock (90-day accrual,
started 2026-08-10, WP-5 answerable 2026-11-08) must run on **exactly one
machine** — currently the T0b cloud box. Two sweepers would double the API
load and fork the panel into two divergent histories, which is worse than
either alone.

If a new box is to *become* the WP-2 machine, that is a migration, not an
install: copy `data/positions/` over intact, enable the unit there, disable it
on the old box — in that order, with no window where both run.

## Undo

```bash
nat service uninstall    # removes the units, restores the tmux+cron path
```
