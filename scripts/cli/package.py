"""`nat package` — build distributables (the .deb)."""

from __future__ import annotations

import sys

from cli.common import ROOT, R, _p, _banner, _exec


def cmd_package_deb(args):
    """Build the .deb package via packaging/build_deb.sh."""
    script = ROOT / "packaging" / "build_deb.sh"
    if not script.exists():
        _p("x", R, f"Missing {script}")
        return 1
    extra = " --no-build" if getattr(args, 'no_build', False) else ""
    _banner("Building nat .deb")
    sys.stdout.flush()
    return _exec(f"bash {script}{extra}").returncode


def register(sub):
    pkg_p = sub.add_parser('package', help='Build distributables (.deb)')
    pkg_p.set_defaults(func=lambda a: pkg_p.print_help())
    pkgsub = pkg_p.add_subparsers(dest='subcmd')
    pkgdeb = pkgsub.add_parser('deb', help='Build the nat .deb (packaging/build_deb.sh)')
    pkgdeb.add_argument('--no-build', action='store_true',
                        help='Skip the release Rust build (use existing binaries)')
    pkgdeb.set_defaults(func=cmd_package_deb)


__all__ = ["cmd_package_deb", "register"]
