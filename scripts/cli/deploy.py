"""`nat deploy` — deploy ingestor to a remote host (build, sync, restart)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from cli.common import (ROOT, RUST, DEPLOY_HOST, DEPLOY_DIR, B, R, G, Y, W,
                        _banner, _p, _cargo, _sh, _exec)


def cmd_deploy(args):
    """Deploy ingestor to remote host (build, sync, restart)."""
    host = getattr(args, 'host', None) or DEPLOY_HOST
    rdir = getattr(args, 'dir', None) or DEPLOY_DIR

    _banner(f"Deploying ingestor to {host}")

    # Step 1: build
    _p("1", B, "Building release binary...")
    r = _cargo("build --locked --release --bin ing")
    if r.returncode != 0:
        _p("x", R, "Build failed")
        sys.exit(1)

    # Step 2: backup
    _p("2", B, f"Backing up current binary on {host}...")
    _sh(f"ssh {host} 'cp {rdir}/rust/target/release/ing {rdir}/rust/target/release/ing.bak 2>/dev/null || true'")

    # Step 3: sync
    _p("3", B, "Syncing binary + config...")
    r1 = _exec(f"rsync -az {RUST}/target/release/ing {host}:{rdir}/rust/target/release/ing")
    r2 = _exec(f"rsync -az {ROOT}/config/ {host}:{rdir}/config/")
    if r1.returncode != 0 or r2.returncode != 0:
        _p("x", R, "rsync failed")
        sys.exit(1)

    # Step 4: restart
    _p("4", B, "Restarting ingestor...")
    _exec(f"ssh {host} 'cd {rdir} && ./nat stop && sleep 3 && ./nat start'")

    # Step 5: health check
    _p("5", B, "Health check (waiting 10s)...")
    time.sleep(10)
    r = _sh(f"ssh {host} 'cd {rdir} && ./nat status'")
    print(r.stdout)
    if r.returncode == 0:
        _p("*", G, "Deploy complete")
    else:
        _p("!", Y, "Deploy finished but health check returned non-zero — verify manually")


def cmd_deploy_status(args):
    """Check remote ingestor status."""
    host = getattr(args, 'host', None) or DEPLOY_HOST
    rdir = getattr(args, 'dir', None) or DEPLOY_DIR
    _exec(f"ssh {host} 'cd {rdir} && ./nat status'")


def cmd_deploy_rollback(args):
    """Rollback remote ingestor to previous binary."""
    host = getattr(args, 'host', None) or DEPLOY_HOST
    rdir = getattr(args, 'dir', None) or DEPLOY_DIR

    _banner(f"Rolling back ingestor on {host}")

    # Check backup exists
    r = _sh(f"ssh {host} 'test -f {rdir}/rust/target/release/ing.bak'")
    if r.returncode != 0:
        _p("x", R, "No backup binary found on remote — cannot rollback")
        sys.exit(1)

    _p("1", B, "Stopping ingestor...")
    _exec(f"ssh {host} 'cd {rdir} && ./nat stop'")

    _p("2", B, "Restoring backup binary...")
    _exec(f"ssh {host} 'cp {rdir}/rust/target/release/ing.bak {rdir}/rust/target/release/ing'")

    _p("3", B, "Starting ingestor...")
    _exec(f"ssh {host} 'cd {rdir} && sleep 2 && ./nat start'")

    _p("4", B, "Health check (waiting 10s)...")
    time.sleep(10)
    r = _sh(f"ssh {host} 'cd {rdir} && ./nat status'")
    print(r.stdout)
    _p("*", G, "Rollback complete")


def cmd_deploy_cloud(args):
    """Deploy a redundant ingestor to a fresh cloud box (T0b) via the .deb + systemd.

    Unlike `nat deploy` (rsync into an existing source checkout on su-35), this ships
    the Debian package, installs it (pulling deps), and registers the systemd --user
    units so the box self-heals and survives reboot. `--dry-run` prints the exact
    ssh/scp steps without touching the remote (safe to inspect before provisioning)."""
    host = args.host
    user = getattr(args, 'user', None) or 'nat'
    target = f"{user}@{host}"
    nat_home = getattr(args, 'nat_home', None) or '/var/lib/nat'
    dry = getattr(args, 'dry_run', False)

    # Resolve the .deb (build it if absent).
    deb = getattr(args, 'deb', None)
    if not deb:
        debs = sorted((ROOT / "dist").glob("nat_*.deb"), key=lambda p: p.stat().st_mtime,
                      reverse=True) if (ROOT / "dist").exists() else []
        if not debs and not dry:
            _p("...", Y, "No .deb in dist/ — building via packaging/build_deb.sh")
            if _exec("bash packaging/build_deb.sh", cwd=ROOT).returncode != 0:
                _p("x", R, "build_deb.sh failed")
                return 1
            debs = sorted((ROOT / "dist").glob("nat_*.deb"), key=lambda p: p.stat().st_mtime, reverse=True)
        deb = str(debs[0]) if debs else "dist/nat_<version>_amd64.deb"
    deb_name = Path(deb).name

    steps = [
        ("copy .deb", f"scp {deb} {target}:/tmp/{deb_name}"),
        ("install (+deps)", f"ssh {target} 'sudo apt-get install -y /tmp/{deb_name}'"),
        ("systemd units (per-user, linger)",
         f"ssh {target} 'NAT_HOME={nat_home} nat service install'"),
        ("verify", f"ssh {target} 'nat status && nat gap status'"),
    ]
    _banner(f"Deploy redundant ingestor → {target}  (.deb + systemd, NAT_HOME={nat_home})")
    _p(" ", W, f"package: {deb}")
    for label, cmd in steps:
        _p(">", B, f"{label}:  {cmd}")
        if dry:
            continue
        rc = _exec(cmd).returncode
        if rc != 0:
            _p("x", R, f"{label} failed (rc={rc}) — see HETZNER_DEPLOYMENT_PLAN.md")
            return rc
    if dry:
        _p("i", W, "dry-run — no remote changes. Add Telegram creds to /etc/nat/.env on the box for push alerts.")
    else:
        _p("*", G, "Redundant ingestor deployed. Confirm gap alerts fire and a clean streak accrues.")
    return 0


def register(sub):
    dep_p = sub.add_parser('deploy', help='Deploy ingestor to remote host')
    dep_p.add_argument('--host', type=str, default=None, help=f'Remote host (default: {DEPLOY_HOST}, env: NAT_DEPLOY_HOST)')
    dep_p.add_argument('--dir', type=str, default=None, help=f'Remote dir (default: {DEPLOY_DIR}, env: NAT_DEPLOY_DIR)')
    dep_p.set_defaults(func=cmd_deploy)
    depsub = dep_p.add_subparsers(dest='subcmd')
    dep_s = depsub.add_parser('status', help='Check remote ingestor status')
    dep_s.add_argument('--host', type=str, default=None)
    dep_s.add_argument('--dir', type=str, default=None)
    dep_s.set_defaults(func=cmd_deploy_status)
    dep_r = depsub.add_parser('rollback', help='Rollback to previous binary')
    dep_r.add_argument('--host', type=str, default=None)
    dep_r.add_argument('--dir', type=str, default=None)
    dep_r.set_defaults(func=cmd_deploy_rollback)
    dep_c = depsub.add_parser('cloud',
        help='Deploy a redundant ingestor to a cloud box via the .deb + systemd (T0b)')
    dep_c.add_argument('host', help='Cloud host or IP')
    dep_c.add_argument('--user', default='nat', help='SSH user (default: nat)')
    dep_c.add_argument('--deb', default=None, help='Path to the .deb (default: newest in dist/, built if absent)')
    dep_c.add_argument('--nat-home', default='/var/lib/nat', help='Data root on the box (default: /var/lib/nat)')
    dep_c.add_argument('--dry-run', action='store_true', help='Print the ssh/scp steps without running them')
    dep_c.set_defaults(func=cmd_deploy_cloud)


__all__ = ["cmd_deploy", "cmd_deploy_status", "cmd_deploy_rollback", "cmd_deploy_cloud", "register"]
