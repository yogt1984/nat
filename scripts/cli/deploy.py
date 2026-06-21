"""`nat deploy` — deploy ingestor to a remote host (build, sync, restart)."""

from __future__ import annotations

import sys
import time

from cli.common import (ROOT, RUST, DEPLOY_HOST, DEPLOY_DIR, B, R, G, Y,
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


__all__ = ["cmd_deploy", "cmd_deploy_status", "cmd_deploy_rollback", "register"]
