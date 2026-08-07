"""DOCS-3 — generate the `nat` CLI reference from the live parser tree.

`docs/commands.md` was hand-maintained and had drifted badly: **26 command groups absent**
and the headline count stale by 80 (`~260` against an actual 340). Hand-transcribing 26
groups would fix today and rot by next week, which is the failure mode `docs/README.md`
names explicitly — *"once headers are stamped, it can be generated from them (so it never
drifts — the failure mode the restructure exists to prevent)"*.

So the reference is generated from `nat --json commands`, i.e. from the argparse tree that
actually dispatches. The doc cannot disagree with the CLI because it is derived from it,
and `tests/test_commands_doc.py` fails if the committed file falls out of date.

Usage:
    python scripts/ops/gen_commands_doc.py            # write docs/commands.md
    python scripts/ops/gen_commands_doc.py --check    # exit 1 if stale (CI)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "commands.md"

HEADER = """# `nat` — CLI Reference

> **Generated** by `scripts/ops/gen_commands_doc.py` from the live argparse tree
> (`nat --json commands`). **Do not edit by hand** — regenerate instead. This file was
> hand-maintained until 2026-08-07, by which point 26 command groups were missing and the
> headline count was stale by 80. A reference that disagrees with the CLI is worse than no
> reference, because it is trusted.
>
> Maturity tags (`[PRELIM]` etc.) appear where a group declares one; surfacing them for
> *every* command lands with NAT9. Absence of a tag is not a claim of maturity.

"""


def live_tree() -> dict:
    out = subprocess.run([str(ROOT / "nat"), "--json", "commands"],
                         cwd=str(ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"`nat --json commands` failed: {out.stderr[-400:]}")
    return json.loads(out.stdout)


def render(tree: dict) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in tree.get("commands", []):
        name = c.get("name", "")
        groups[name.split()[0] if name else "?"].append(c)

    lines = [HEADER, f"**{tree.get('count', 0)} commands** across "
                     f"**{len(groups)} groups**.\n"]
    lines.append("| Group | Commands |\n|---|---|")
    for g in sorted(groups):
        lines.append(f"| [`{g}`](#{g}) | {len(groups[g])} |")
    lines.append("")

    for g in sorted(groups):
        lines.append(f"\n## {g}\n")
        lines.append("| Command | Description |")
        lines.append("|---|---|")
        for c in sorted(groups[g], key=lambda x: x.get("name", "")):
            # `|` inside help text would break the table row
            help_txt = (c.get("help") or "").replace("|", "\\|").strip()
            lines.append(f"| `nat {c.get('name','')}` | {help_txt} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed doc is stale (does not write)")
    args = ap.parse_args()

    text = render(live_tree())
    if args.check:
        current = DOC.read_text() if DOC.exists() else ""
        if current != text:
            print(f"{DOC} is STALE — regenerate with "
                  f"`python scripts/ops/gen_commands_doc.py`")
            return 1
        print(f"{DOC} is up to date")
        return 0

    DOC.write_text(text)
    print(f"wrote {DOC} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
