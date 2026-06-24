# HETZNER_DEPLOYMENT_PLAN.md — the redundant cloud ingestor (T0b / Q1)

**Goal:** stand up a **second, independent 24/7 ingestor** on a Hetzner cloud box so data
continuity (NAT's binding constraint) no longer depends on the single su-35 feed. The box runs the
`ing` binary under systemd (auto-restart + boot-persistence), pages on data gaps within minutes
(Telegram + local fallback), and accrues an independent clean-data streak we can later reconcile
against su-35.

This is the runbook the roadmap's **Q1** task references. The deployment *vehicle* already exists:
relocatable paths (`scripts/nat_paths.py`), the Debian package (`packaging/build_deb.sh`), the
systemd units (`scripts/cli/systemd_units.py` → `nat service install`), and the gap-alert daemon
(`scripts/ops/gap_alert.py`). `nat deploy cloud` chains them.

> **Guardrail:** this is the *redundant* feed — it does **not** replace su-35 and must not touch it.
> Until the box has proven a clean streak, treat su-35 as the source of record.

---

## 0. Prerequisites (local)
- The `.deb` builds: `nat package deb` → `dist/nat_<version>_amd64.deb` (bundles the CLI, Python
  stack, release Rust binaries incl. `ing`, and `/etc/nat` config).
- SSH access to the box as a sudo-capable user; an SSH key loaded.
- (Optional but recommended) a Telegram bot token + chat id for push alerts.

## 1. Provision the box (Hetzner Cloud)
1. Create a **CX22/CPX21**-class instance (2 vCPU / 4 GB is ample for 3-symbol ingestion), Ubuntu
   22.04/24.04 LTS, in a low-latency region to the exchange.
2. Add your SSH key at creation. Note the public IP.
3. Harden: `ufw allow OpenSSH`, disable password SSH, create a non-root service user:
   ```bash
   sudo adduser --disabled-password --gecos "" nat
   sudo usermod -aG sudo nat            # for the one-time apt install; can drop later
   sudo loginctl enable-linger nat      # so --user systemd units run without an active login
   ```
4. Data dir: `sudo mkdir -p /var/lib/nat && sudo chown -R nat:nat /var/lib/nat` (this is `NAT_HOME`).
5. No inbound app ports are required (the ingestor is an *outbound* WebSocket client). Leave the
   dashboard/API ports closed unless you tunnel them.

## 2. Deploy
From the repo, one command (build-if-needed → scp → apt install → systemd units → verify):
```bash
nat deploy cloud <BOX_IP> --user nat --nat-home /var/lib/nat
# inspect first without touching the box:
nat deploy cloud <BOX_IP> --dry-run
```
What it does (also runnable by hand from the dry-run output):
1. `scp dist/nat_*.deb nat@BOX:/tmp/`
2. `ssh … 'sudo apt-get install -y /tmp/nat_*.deb'` — installs `nat` to `/usr/lib/nat`, config to
   `/etc/nat`, binaries under `/usr/lib/nat/rust/target/release/`, symlinks `/usr/bin/nat`. In
   installed mode `nat_paths` resolves config from `/etc/nat` and data from `$NAT_HOME`.
3. `ssh … 'NAT_HOME=/var/lib/nat nat service install'` — writes the per-user systemd units
   (`nat-ingestor.service`, `nat-gap-alert.service`), enables linger, and starts them
   (`Restart=always`, `WantedBy=default.target`). The unit `Environment` bakes in
   `NAT_DATA_DIR=/var/lib/nat/data/features` etc., so the Rust `ing` writes to the resolved dir.
4. `ssh … 'nat status && nat gap status'` — health check.

## 3. Configure alerting (on the box)
```bash
sudo install -m 600 -o nat -g nat /dev/stdin /etc/nat/.env <<EOF
TELEGRAM_BOT_TOKEN=...      # from @BotFather
TELEGRAM_CHAT_ID=...
EOF
nat service restart gap     # pick up the creds (gap_alert sources .env on start)
```
Without creds the gap daemon still alerts **locally** (`/var/lib/nat/ops/alerts.log` + `nat status`
"Data monitor"), and logs a loud "Telegram NOT configured" warning — so a gap is never invisible.

## 4. Verify (the acceptance gate)
- `ssh nat@BOX nat status` → Ingestor: RUNNING; **Supervisor: systemd … active**; Data monitor: OK.
- Self-heal: `ssh nat@BOX 'kill -9 $(pgrep -x ing)'` → systemd restarts it within ~5s (pid changes).
- Reboot survival: `sudo reboot`; after it's back, `nat status` shows the ingestor up with **no
  manual start** (linger + `WantedBy=default.target`).
- Gap detection: it pages on a >5-min write gap and on a stalled buffer (`.tmp` not growing 15 min).
- **Streak:** run `/streak` (or `nat gap status`) against the box's `data/features` over the next
  days; the acceptance bar is **7 consecutive clean days** (>~200 MB/day) — that lifts the su-35
  freeze and unblocks the Q-branch revalidation.

## 5. Reconcile with su-35 (after the streak)
Once the cloud box has a clean streak, compare gap profiles between the two feeds (rsync/mount the
box's `data/features` locally and diff coverage) before deciding which becomes the source of record.
Do **not** contact su-35 for the comparison until its own streak gate is resolved.

## Rollback / teardown
- Stop monitoring + ingestor: `ssh nat@BOX nat service uninstall` (removes units, restores nothing
  else) then `sudo apt-get remove nat`.
- The `/var/lib/nat` data is preserved on remove (purge manually if intended).

## Open items
- The `.deb` currently ships from a local build; for fleet/repeatable installs, publish to the
  private apt repo (see `packaging/README.md` §2) and `apt install nat` instead of scp+dpkg.
- Pre-existing, unrelated: `nat agent status` daemon raises `ModuleNotFoundError: logging_config`
  (not needed for ingestion; fix separately before running agents on the box).
