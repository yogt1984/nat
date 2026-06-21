"""`nat docker` — Docker stack operations (build/up/down/logs/ps/stack/smoke)."""

from __future__ import annotations

import subprocess
import sys
import time

from cli.common import BOLD, W, B, R, G, Y, _banner, _p, _exec


def cmd_docker_build(args):
    services = " ".join(getattr(args, 'services', None) or [])
    verbose = "--progress=plain" if getattr(args, 'verbose', False) else ""
    _exec(f"docker compose build {verbose} {services}".strip())


def cmd_docker_up(args):
    services = " ".join(getattr(args, 'services', None) or [])
    _exec(f"docker compose up -d {services}".strip())
    print("\n  Ingestor:   http://localhost:8080")
    print("  API:        http://localhost:3010")
    print("  Grafana:    http://localhost:3002")
    print("  Prometheus: http://localhost:9090")
    print("  Redis:      localhost:6379\n")


def cmd_docker_down(args):
    _exec("docker compose down")


def cmd_docker_logs(args):
    services = " ".join(getattr(args, 'services', None) or [])
    _exec(f"docker compose logs -f {services}".strip())


def cmd_docker_ps(args):
    _exec("docker compose ps")


def cmd_docker_stack(args):
    """Build, start, and verify the full Docker stack."""
    services = " ".join(getattr(args, 'services', None) or [])
    verbose = "--progress=plain" if getattr(args, 'verbose', False) else ""
    skip_build = getattr(args, 'no_build', False)
    only_build = getattr(args, 'build_only', False)

    _banner("NAT Docker Stack")

    # Step 1: Build
    if not skip_build:
        targets = services or "ingestor api alerts"
        _p("1", B, f"Building images: {targets}")
        r = _exec(f"docker compose build {verbose} {targets}".strip())
        if r.returncode != 0:
            _p("x", R, "Build failed")
            sys.exit(1)
        _p("*", G, "Build complete")
    else:
        _p("-", Y, "Skipping build (--no-build)")

    if only_build:
        _p("*", G, "Build-only mode — done")
        return

    # Step 2: Start (core services only if no specific services requested)
    core = services or "redis ingestor api alerts prometheus grafana"
    _p("2", B, f"Starting services: {core}")
    r = _exec(f"docker compose up -d {core}".strip())
    if r.returncode != 0:
        _p("!", Y, "Some services failed to start — checking endpoints...")
    else:
        _p("*", G, "Services started")

    # Step 3: Wait + verify
    _p("3", B, "Waiting for startup (15s)...")
    time.sleep(15)

    endpoints = [
        ("Ingestor",   "http://localhost:8080",            8080),
        ("API",        "http://localhost:3010/health",      3010),
        ("Prometheus", "http://localhost:9090/api/v1/targets", 9090),
        ("Grafana",    "http://localhost:3002/api/health",  3002),
    ]
    all_ok = True
    for name, url, port in endpoints:
        try:
            r = subprocess.run(["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", url],
                               capture_output=True, text=True, timeout=5)
            code = r.stdout.strip()
            if code == "200":
                _p("*", G, f"{name:12s} :{port}  OK")
            else:
                _p("x", R, f"{name:12s} :{port}  HTTP {code}")
                all_ok = False
        except Exception:
            _p("x", R, f"{name:12s} :{port}  unreachable")
            all_ok = False

    # Step 4: Metrics check
    try:
        r = subprocess.run(
            ["curl", "-sf", "http://localhost:9090/api/v1/query",
             "--data-urlencode", "query=ing_features_emitted_total"],
            capture_output=True, text=True, timeout=5)
        import json
        data = json.loads(r.stdout)
        n = len(data.get("data", {}).get("result", []))
        if n > 0:
            _p("*", G, f"Metrics       Prometheus scraping {n} series")
        else:
            _p("!", Y, "Metrics       no data yet (wait a few minutes)")
    except Exception:
        _p("!", Y, "Metrics       could not query Prometheus")

    print()
    if all_ok:
        _p("*", G, "Stack is healthy")
        print(f"\n  {BOLD}Endpoints:{W}")
        print(f"    Ingestor:   http://localhost:8080")
        print(f"    API:        http://localhost:3010")
        print(f"    Grafana:    http://localhost:3002")
        print(f"    Prometheus: http://localhost:9090\n")
    else:
        _p("!", Y, "Some services failed — check docker compose logs")


def cmd_docker_smoke(args):
    """Quick smoke test of running Docker stack."""
    _banner("Docker Smoke Test")
    endpoints = {
        "Ingestor":   "http://localhost:8080",
        "API":        "http://localhost:3010/health",
        "Prometheus": "http://localhost:9090/api/v1/targets",
        "Grafana":    "http://localhost:3002/api/health",
    }
    ok = 0
    total = len(endpoints) + 2  # +2 for postgres and caddy
    for name, url in endpoints.items():
        try:
            r = subprocess.run(["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", url],
                               capture_output=True, text=True, timeout=5)
            if r.stdout.strip() == "200":
                _p("*", G, f"{name:12s} OK")
                ok += 1
            else:
                _p("x", R, f"{name:12s} HTTP {r.stdout.strip()}")
        except Exception:
            _p("x", R, f"{name:12s} unreachable")
    # PostgreSQL check
    try:
        r = subprocess.run(["docker", "compose", "exec", "-T", "postgres", "pg_isready", "-U", "nat"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _p("*", G, f"{'PostgreSQL':12s} OK")
            ok += 1
        else:
            _p("x", R, f"{'PostgreSQL':12s} not ready")
    except Exception:
        _p("x", R, f"{'PostgreSQL':12s} unreachable")
    # Caddy check
    try:
        r = subprocess.run(["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:80"],
                           capture_output=True, text=True, timeout=5)
        if r.stdout.strip() in ("200", "301", "308"):
            _p("*", G, f"{'Caddy':12s} OK")
            ok += 1
        else:
            _p("x", R, f"{'Caddy':12s} HTTP {r.stdout.strip()}")
    except Exception:
        _p("x", R, f"{'Caddy':12s} unreachable")
    print()
    _p("*" if ok == total else "!", G if ok == total else Y,
       f"{ok}/{total} services healthy")


def register(sub):
    dk_p = sub.add_parser('docker', help='Docker operations')
    dk_p.set_defaults(func=lambda a: dk_p.print_help())
    dksub = dk_p.add_subparsers(dest='subcmd')
    dk_build = dksub.add_parser('build', help='Build images (nat docker build [--verbose] [services...])')
    dk_build.add_argument('--verbose', '-v', action='store_true', help='Show full build output')
    dk_build.add_argument('services', nargs='*', default=None, help='Services to build (default: all)')
    dk_build.set_defaults(func=cmd_docker_build)
    dk_up = dksub.add_parser('up', help='Start services (nat docker up [services...])')
    dk_up.add_argument('services', nargs='*', default=None, help='Services to start (default: all)')
    dk_up.set_defaults(func=cmd_docker_up)
    dksub.add_parser('down', help='Stop services').set_defaults(func=cmd_docker_down)
    dksub.add_parser('ps', help='Show running services').set_defaults(func=cmd_docker_ps)
    dk_logs = dksub.add_parser('logs', help='View logs (nat docker logs [services...])')
    dk_logs.add_argument('services', nargs='*', default=None, help='Services to tail (default: all)')
    dk_logs.set_defaults(func=cmd_docker_logs)
    dk_stack = dksub.add_parser('stack', help='Build + start + verify full stack')
    dk_stack.add_argument('--verbose', '-v', action='store_true', help='Verbose build output')
    dk_stack.add_argument('--no-build', action='store_true', help='Skip build, start only')
    dk_stack.add_argument('--build-only', action='store_true', help='Build only, do not start')
    dk_stack.add_argument('services', nargs='*', default=None, help='Services (default: all)')
    dk_stack.set_defaults(func=cmd_docker_stack)
    dksub.add_parser('smoke', help='Quick health check of running stack').set_defaults(func=cmd_docker_smoke)


__all__ = ["cmd_docker_build", "cmd_docker_up", "cmd_docker_down",
           "cmd_docker_logs", "cmd_docker_ps", "cmd_docker_stack",
           "cmd_docker_smoke", "register"]
