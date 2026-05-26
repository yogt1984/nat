"""CLI for data utilities.

Usage:
    python -m data check-provenance [--db PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_check_provenance(args: argparse.Namespace) -> None:
    from .features import compute_data_version
    from .state import StateStore

    current = compute_data_version()
    print(f"Current data version: {current}")

    db = Path(args.db)
    if not db.exists():
        print(f"Database not found: {db}")
        sys.exit(1)

    store = StateStore(db)
    stale = store.check_provenance(current)
    store.close()

    if not stale:
        print("All hypotheses match current data version.")
        return

    print(f"\n{len(stale)} hypotheses tested against different data:")
    for h in stale:
        print(f"  {h['id']}  agent={h['agent']}  version={h['data_version']}"
              f"  status={h['status']}")
        print(f"    {h['claim'][:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m data")
    sub = parser.add_subparsers(dest="command")

    p_prov = sub.add_parser("check-provenance",
                            help="Flag hypotheses tested against stale data")
    p_prov.add_argument("--db", default="data/nat.db",
                        help="Path to SQLite database")

    args = parser.parse_args()
    if args.command == "check-provenance":
        cmd_check_provenance(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
