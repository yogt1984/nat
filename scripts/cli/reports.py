"""`nat reports` — list / latest-per-category / show report artifacts."""

from __future__ import annotations

import json as _json
from datetime import datetime
from pathlib import Path

from cli.common import ROOT, REPORTS_DIR, BOLD, W, R, _json_mode, _output, _p


def cmd_reports(args):
    """List all reports."""
    reports_dir = REPORTS_DIR
    if not reports_dir.exists():
        if _json_mode(args):
            _output({"reports": []}, args)
        else:
            _p("x", R, "No reports directory")
        return

    entries = []
    for f in sorted(reports_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file():
            s = f.stat()
            entries.append({
                "path": str(f.relative_to(ROOT)),
                "type": f.suffix.lstrip(".") or "unknown",
                "size_kb": round(s.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(s.st_mtime).isoformat(),
            })

    if _json_mode(args):
        _output({"reports": entries, "count": len(entries)}, args)
        return

    print(f"\n  {BOLD}Reports ({len(entries)}){W}\n")
    for e in entries[:50]:
        print(f"  {e['modified'][:10]}  {e['size_kb']:>8.1f} KB  {e['path']}")
    if len(entries) > 50:
        print(f"\n  ... and {len(entries) - 50} more")
    print()


def cmd_reports_latest(args):
    """Show most recent report per category."""
    reports_dir = REPORTS_DIR
    if not reports_dir.exists():
        if _json_mode(args):
            _output({"latest": {}}, args)
        else:
            _p("x", R, "No reports directory")
        return

    categories = {}
    for f in reports_dir.rglob("*"):
        if not f.is_file():
            continue
        cat = f.parent.name if f.parent != reports_dir else "root"
        if cat not in categories or f.stat().st_mtime > categories[cat].stat().st_mtime:
            categories[cat] = f

    if _json_mode(args):
        result = {cat: {"path": str(f.relative_to(ROOT)),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
                  for cat, f in categories.items()}
        _output({"latest": result}, args)
        return

    print(f"\n  {BOLD}Latest Reports by Category{W}\n")
    for cat, f in sorted(categories.items()):
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {cat:20s}  {mtime}  {f.relative_to(ROOT)}")
    print()


def cmd_reports_show(args):
    """Print report content."""
    p = Path(args.path)
    if not p.exists():
        p = ROOT / args.path
    if not p.exists():
        _p("x", R, f"Not found: {args.path}")
        return 1
    content = p.read_text()
    if p.suffix == ".json":
        try:
            data = _json.loads(content)
            if _json_mode(args):
                print(content)
            else:
                print(_json.dumps(data, indent=2, default=str))
        except _json.JSONDecodeError:
            print(content)
    else:
        print(content)


def register(sub):
    rpt_p = sub.add_parser('reports', help='Report management')
    rpt_p.set_defaults(func=cmd_reports)
    rptsub = rpt_p.add_subparsers(dest='subcmd')
    rptsub.add_parser('latest', help='Most recent report per category').set_defaults(func=cmd_reports_latest)
    rpt_show = rptsub.add_parser('show', help='Print report content')
    rpt_show.add_argument('path', help='Report file path')
    rpt_show.set_defaults(func=cmd_reports_show)


__all__ = ["cmd_reports", "cmd_reports_latest", "cmd_reports_show", "register"]
