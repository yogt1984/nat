"""`nat data` — feature-store stats, per-file listing, validation, schema."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from cli.common import (
    ROOT, PY, DATA_DEFAULT, BOLD, W, R, Y,
    _data, _json_mode, _output, _p, _banner, _py, _exec,
)


def cmd_data(args):
    """Show data statistics."""
    dd = Path(_data(args))
    if not dd.exists():
        if _json_mode(args):
            _output({"error": "No data directory", "path": str(dd)}, args)
        else:
            _p("x", R, "No data directory")
        return
    parquets = sorted(dd.rglob("*.parquet"))
    stats = [(f, f.stat()) for f in parquets]
    valid = [(f, s) for f, s in stats if s.st_size > 0]
    total_size = sum(s.st_size for _, s in valid)
    days = sorted(set(f.parent.name for f, _ in valid))

    if _json_mode(args):
        per_day = []
        for day in days:
            day_files = [(f, s) for f, s in valid if f.parent.name == day]
            day_size = sum(s.st_size for _, s in day_files)
            per_day.append({"date": day, "files": len(day_files), "size_mb": round(day_size / 1e6, 1)})
        _output({
            "path": str(dd),
            "files": len(valid), "empty_files": len(parquets) - len(valid),
            "size_mb": round(total_size / 1e6, 1),
            "days": len(days),
            "date_range": [days[0], days[-1]] if days else None,
            "estimated_rows": int(total_size // 300),
            "per_day": per_day,
        }, args)
        return

    print(f"\n  {BOLD}Data Summary{W}\n")
    print(f"  Files:     {len(valid)} ({len(parquets) - len(valid)} empty)")
    print(f"  Disk:      {total_size / 1e6:.1f} MB")
    print(f"  Days:      {len(days)}")
    if days:
        print(f"  Range:     {days[0]} to {days[-1]}")
    print(f"\n  {BOLD}Per Day{W}\n")
    for day in days:
        day_files = [(f, s) for f, s in valid if f.parent.name == day]
        day_size = sum(s.st_size for _, s in day_files)
        print(f"  {day}: {len(day_files)} files, {day_size / 1e6:.1f} MB")
    print(f"\n  ~{total_size // 300:.0f} rows (estimated)\n")


def cmd_data_ls(args):
    """List individual parquet files (path, date, size, rows, mtime)."""
    dd = Path(_data(args))
    if not dd.exists():
        if _json_mode(args):
            _output({"error": "No data directory", "path": str(dd)}, args)
        else:
            _p("x", R, "No data directory")
        return 1

    import pyarrow.parquet as _pq
    want_date = getattr(args, 'date', None)
    want_sym = getattr(args, 'symbol', None)
    limit = getattr(args, 'limit', None)

    files = sorted(dd.rglob("*.parquet"), key=os.path.getmtime, reverse=True)
    rows = []
    for f in files:
        if want_date and f.parent.name != want_date:
            continue
        st = f.stat()
        if st.st_size == 0:
            continue
        nrows = None
        try:
            nrows = _pq.read_metadata(f).num_rows  # cheap: footer only, no full read
        except Exception:
            pass
        rows.append({
            "path": str(f),
            "date": f.parent.name,
            "size_mb": round(st.st_size / 1e6, 2),
            "rows": nrows,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })

    # --symbol filter requires peeking the file; keep it best-effort and last so
    # the cheap path stays fast when no symbol filter is requested.
    if want_sym:
        keep = []
        for r in rows:
            try:
                tbl = _pq.read_table(r["path"], columns=["symbol"])
                if want_sym in set(tbl.column("symbol").to_pylist()):
                    keep.append(r)
            except Exception:
                keep.append(r)
        rows = keep

    if limit:
        rows = rows[:limit]

    def _human(_data):
        files_ = _data["files"]
        if not files_:
            _p("x", Y, "No parquet files match")
            return
        print(f"\n  {BOLD}Parquet files{W}  ({len(files_)} shown)\n")
        print(f"  {'DATE':<12} {'SIZE':>9} {'ROWS':>10}  {'MTIME':<19}  FILE")
        for r in files_:
            rws = f"{r['rows']:,}" if r['rows'] is not None else "?"
            print(f"  {r['date']:<12} {r['size_mb']:>7.2f}MB {rws:>10}  "
                  f"{r['mtime']:<19}  {Path(r['path']).name}")
        print()

    _output({"path": str(dd), "files": rows, "count": len(rows)}, args, _human)
    return 0


def cmd_data_validate(args):
    path = getattr(args, 'path', None)
    target = path or _data(args)
    as_json = _json_mode(args)
    if not as_json:
        _banner(f"Validating {'file' if path else 'collected data'}: {target}")
        sys.stdout.flush()
    hours = getattr(args, 'hours', None)
    cmd = f"scripts/validate_data.py {target}"
    if not as_json:
        cmd += " --verbose"
    if hours:
        cmd += f" --hours {hours}"
    if as_json:
        cmd += " --json"
    return _py(cmd).returncode


def cmd_data_explore(args):
    _exec("jupyter notebook notebooks/explore_features.ipynb")


def cmd_data_schema(args):
    _banner("Scanning parquet schema & vector coverage")
    _exec(
        f'{PY} -c "from cluster_pipeline.loader import print_schema_summary; '
        f"print_schema_summary('../{_data(args)}')\"",
        cwd=ROOT / "scripts",
    )


def register(sub):
    data_p = sub.add_parser('data', help='Data stats & validation')
    data_p.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    data_p.set_defaults(func=cmd_data)
    dsub = data_p.add_subparsers(dest='subcmd')
    dv = dsub.add_parser('validate', help='Validate parquet data (a directory or a single file)')
    dv.add_argument('path', nargs='?', help='A single .parquet file to validate (default: scan the data dir)')
    dv.add_argument('--hours', type=int, help='Validate only last N hours (ignored for a single file)')
    dv.add_argument('--json', action='store_true', help='Emit the report as JSON')
    dv.set_defaults(func=cmd_data_validate)
    dsub.add_parser('explore', help='Launch Jupyter').set_defaults(func=cmd_data_explore)
    dsub.add_parser('schema', help='Scan parquet schema').set_defaults(func=cmd_data_schema)
    dls = dsub.add_parser('ls', help='List individual parquet files (path/size/rows/mtime)')
    dls.add_argument('--symbol', '-s', default=None, help='Filter to files containing this symbol')
    dls.add_argument('--date', default=None, help='Filter to one day YYYY-MM-DD')
    dls.add_argument('--limit', type=int, default=None, help='Show only the newest N files')
    dls.add_argument('--json', action='store_true', help='JSON output')
    dls.set_defaults(func=cmd_data_ls)


__all__ = ["cmd_data", "cmd_data_ls", "cmd_data_validate", "cmd_data_explore",
           "cmd_data_schema", "register"]
