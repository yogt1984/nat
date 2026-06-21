"""`nat report` — experiment reports (generate / ls)."""

from __future__ import annotations

from datetime import datetime

from cli.common import (
    ROOT, REPORTS_DIR, B, Y, BOLD, W, _p, _py, _output,
)


def cmd_report(args):
    """Generate full experiment report."""
    _p("...", B, "Generating report...")
    _py(f"{ROOT / 'scripts' / 'generate_report.py'}")


def cmd_report_ls(args):
    """List experiment-outcome artifacts under reports/ and docs/reports/."""
    roots = [REPORTS_DIR, ROOT / "docs" / "reports"]
    want_ext = getattr(args, 'ext', None)
    limit = getattr(args, 'limit', None)
    items = []
    for base in roots:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".json", ".md", ".html", ".csv"):
                continue
            if want_ext and f.suffix.lower() != f".{want_ext.lstrip('.')}":
                continue
            st = f.stat()
            items.append({
                "name": str(f.relative_to(ROOT)),
                "type": f.suffix.lstrip("."),
                "size_kb": round(st.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    if limit:
        items = items[:limit]

    def _human(_d):
        rows = _d["reports"]
        if not rows:
            _p("x", Y, "No report artifacts found")
            return
        print(f"\n  {BOLD}Experiment reports{W}  ({len(rows)} shown)\n")
        print(f"  {'TYPE':<5} {'SIZE':>9}  {'MTIME':<19}  NAME")
        for r in rows:
            print(f"  {r['type']:<5} {r['size_kb']:>7.1f}KB  {r['mtime']:<19}  {r['name']}")
        print()

    _output({"reports": items, "count": len(items)}, args, _human)
    return 0


def register(sub):
    report_p = sub.add_parser('report', help='Experiment reports (generate / ls)')
    report_p.set_defaults(func=cmd_report)  # bare `nat report` keeps generating
    repsub = report_p.add_subparsers(dest='subcmd')
    repsub.add_parser('generate', help='Generate the full experiment report').set_defaults(func=cmd_report)
    repls = repsub.add_parser('ls', aliases=['list'], help='List experiment-outcome artifacts')
    repls.add_argument('--ext', default=None, help='Filter by extension (json|md|html|csv)')
    repls.add_argument('--limit', type=int, default=None, help='Show only the newest N')
    repls.add_argument('--json', action='store_true', help='JSON output')
    repls.set_defaults(func=cmd_report_ls)


__all__ = ["cmd_report", "cmd_report_ls", "register"]
