"""`nat viz` / `nat viz3d` — terminal-first + 3D feature visualization."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cli.common import (
    ROOT, RUST, REPORTS_DIR, DATA_DEFAULT, PY, _TF_MAP,
    G, R, Y, B, W, BOLD,
    _sym, _data, _json_mode, _output, _banner, _p, _py, _sh, _cargo,
)


def cmd_viz_features(args):
    """nat viz features — per-feature overview (value / z-score / NaN% / IC / sparkline)."""
    import math as _math
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.common import load_features, feature_table
    from viz.terminal import sparkline, ic_color, RESET
    symbol = _sym(args)
    df = load_features(symbol, hours=getattr(args, 'hours', None))
    rows = feature_table(df, alive_only=getattr(args, 'alive_only', False))
    top = getattr(args, 'top', 40)
    if _json_mode(args):
        _output({"symbol": symbol, "n_features": len(rows),
                 "features": [{k: r[k] for k in ("feature", "last", "zscore", "nan_pct", "ic")}
                              for r in rows[:top]]}, args)
        return
    if df.empty:
        print(f"  {Y}No data for {symbol}{W} — latest day may be thin (see `nat data validate`).")
        return
    shown = min(top, len(rows))
    print(f"\n  {BOLD}nat viz features{W} — {symbol}  ({len(rows)} features, top {shown} by |IC|)\n")
    print(f"    {'feature':<32} {'last':>11} {'z':>6} {'NaN%':>5} {'IC':>7}  spark")
    print(f"    {'-'*32} {'-'*11} {'-'*6} {'-'*5} {'-'*7}  {'-'*24}")
    for r in rows[:top]:
        ic, z = r['ic'], r['zscore']
        ic_s = "   -   " if _math.isnan(ic) else f"{ic:+.3f}"
        z_s = "   -" if _math.isnan(z) else f"{z:+.1f}"
        col = ic_color(None if _math.isnan(ic) else ic)
        print(f"    {r['feature']:<32} {r['last']:>11.4g} {z_s:>6} {r['nan_pct']:>4.0f}% "
              f"{col}{ic_s:>7}{RESET}  {sparkline(r['spark'])}")
    print()


def cmd_viz_algorithm(args):
    """nat viz algorithm <name> — signal timeline, IC, entry/exit count, P&L proxy."""
    import math as _math
    import numpy as _np
    import contextlib as _ctx, io as _io
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.common import load_features, forward_return, _spearman_ic, _downsample
    from viz.terminal import sparkline, ic_color, RESET, DIM, GREEN, RED
    name, symbol = args.name, _sym(args)
    feats = load_features(symbol, hours=getattr(args, 'hours', None))
    with _ctx.redirect_stdout(_io.StringIO()):       # ML algos print on import/load — mute it
        from algorithms.autodiscover import discover_all
        from algorithms.registry import get_algorithm, list_algorithms
        discover_all()
        known = name in list_algorithms()
        sig = get_algorithm(name).run_batch(feats) if (known and not feats.empty) else None
    if not known:
        print(f"  {R}Unknown algorithm:{W} {name}  (try `nat algorithm list`)")
        return 1
    if feats.empty:
        if _json_mode(args):
            _output({"algorithm": name, "symbol": symbol, "outputs": []}, args)
            return
        print(f"  {Y}No data for {symbol}{W} — latest day may be thin (see `nat data validate`).")
        return
    target = forward_return(feats, horizon=getattr(args, 'horizon', 50))
    thr = getattr(args, 'threshold', 0.3)
    outs = []
    for col in sig.columns:
        s = sig[col].astype(float)
        active = s.abs() > thr
        entries = int((active & ~active.shift(1, fill_value=False)).sum())
        equity = (_np.sign(s) * target).dropna().cumsum()
        outs.append({
            "output": col,
            "last": float(s.dropna().iloc[-1]) if s.notna().any() else float("nan"),
            "ic": _spearman_ic(s, target),
            "entries": entries,
            "pnl_proxy": float(equity.iloc[-1]) if len(equity) else float("nan"),
            "spark": _downsample(s.dropna(), 30),
            "equity_spark": _downsample(equity, 30) if len(equity) else [],
        })
    if _json_mode(args):
        _output({"algorithm": name, "symbol": symbol,
                 "outputs": [{k: o[k] for k in ("output", "last", "ic", "entries", "pnl_proxy")}
                             for o in outs]}, args)
        return
    print(f"\n  {BOLD}nat viz algorithm{W} — {name} on {symbol}  ({len(feats)} ticks)\n")
    for o in outs:
        ic, pnl = o["ic"], o["pnl_proxy"]
        ic_s = "  -  " if _math.isnan(ic) else f"{ic:+.3f}"
        pnl_s = "  -  " if _math.isnan(pnl) else f"{pnl:+.4f}"
        pnl_c = GREEN if (pnl == pnl and pnl > 0) else (RED if (pnl == pnl and pnl < 0) else "")
        print(f"    {BOLD}{o['output']}{W}")
        print(f"      last={o['last']:.4g}  IC={ic_color(None if _math.isnan(ic) else ic)}{ic_s}{RESET}"
              f"  entries={o['entries']}  P&L(proxy)={pnl_c}{pnl_s}{RESET}")
        print(f"      signal {sparkline(o['spark'])}")
        if o["equity_spark"]:
            print(f"      equity {sparkline(o['equity_spark'])}")
    print(f"\n  {DIM}P&L proxy = Σ sign(signal)·fwd-return — not a cost-aware backtest; use `nat oos30`.{W}\n")


def cmd_viz_paper(args):
    """nat viz paper — approval evidence: cumulative P&L, IC decay, G8 scorecard (NAT6)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.approval import (load_oos_state, resolve_name, paper_pnl_series,
                              ic_decay_series, per_signal_risk, g8_scorecard)
    from viz.terminal import sparkline, DIM
    from promotion_daemon import load_gates

    name, symbol = args.name, _sym(args)
    state = load_oos_state()
    gates = load_gates()
    risk = per_signal_risk(state, name, symbol)
    sc = g8_scorecard(state, name, gates, symbol)

    if _json_mode(args):
        _output({"signal": name, "symbol": symbol, "risk": risk, "g8": sc,
                 "n_pnl_points": len(paper_pnl_series(state, name, symbol))}, args)
        return
    if resolve_name(state, name) is None or not risk:
        print(f"  {Y}No OOS/paper data for {name}{W} — pending the paper window (run `nat oos30`, or after promotion).")
        return

    cum = [p["cum_bps"] for p in paper_pnl_series(state, name, symbol)]
    sh = [d["sharpe"] for d in ic_decay_series(state, name, symbol)]
    print(f"\n  {BOLD}nat viz paper{W} — {name} on {symbol}  ({risk['n_days']} days)\n")
    print(f"    cum P&L (bps)  last={(cum[-1] if cum else float('nan')):.0f}  {sparkline(cum)}")
    print(f"    IC decay (7d sharpe)         {sparkline(sh)}")
    print(f"    sharpe={risk['sharpe']:.2f}  maxDD={risk['max_dd_bps']:.0f}bps  "
          f"PF={risk['profit_factor']:.2f}  cumP&L={risk['cum_pnl_bps']:.0f}bps")
    print(f"\n    {BOLD}G8 scorecard{W} (provisional from OOS):")
    checks = [
        ("sharpe within 2x", sc.get("gate_sharpe_within_2x")),
        ("no big daily loss", sc.get("gate_no_big_daily_loss")),
        ("IC stable", sc.get("gate_ic_stable")),
        ("infra stable", sc.get("gate_infra_stable")),
        (f"duration ≥ {gates['g8_min_days']}d (n={sc.get('n_days', 0)})",
         sc.get("n_days", 0) >= gates["g8_min_days"]),
    ]
    for label, ok in checks:
        print(f"      {G}✓{W} {label}" if ok else f"      {R}✗{W} {label}")
    overall = all(ok for _, ok in checks)
    print(f"    → {G}G8 PASS{W}" if overall else f"    → {R}G8 FAIL{W}")
    print(f"\n  {DIM}Provisional: OOS proxies until a live 14-day paper window accrues.{W}\n")


def cmd_viz_portfolio(args):
    """nat viz portfolio — P&L / exposure / correlation / risk across signals (NAT7)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from viz.approval import load_oos_state, per_signal_risk, signal_correlation_matrix
    from viz.terminal import DIM

    tab, symbol = getattr(args, 'tab', 1), _sym(args)
    state = load_oos_state()
    names = sorted(state.get("algos", {}).keys())

    if _json_mode(args):
        out = {"tab": tab, "symbol": symbol, "signals": names}
        if tab in (1, 4):
            out["risk"] = {n: per_signal_risk(state, n, symbol) for n in names}
        elif tab == 2:
            out["exposure"] = None  # pending live positions
        elif tab == 3:
            out["correlation"] = signal_correlation_matrix(state, names, symbol)
        _output(out, args)
        return
    if not names:
        print(f"  {Y}No portfolio data{W} — pending OOS/paper data (run `nat oos30`).")
        return

    if tab == 1:
        print(f"\n  {BOLD}Portfolio — P&L{W} ({symbol})\n")
        print(f"    {'signal':<22} {'sharpe':>7} {'cumP&L':>9} {'maxDD':>8} {'PF':>6}")
        for n in names:
            r = per_signal_risk(state, n, symbol)
            print(f"    {n:<22} {r['sharpe']:>7.2f} {r['cum_pnl_bps']:>8.0f}b "
                  f"{r['max_dd_bps']:>7.0f}b {r['profit_factor']:>6.2f}")
    elif tab == 2:
        print(f"\n  {BOLD}Portfolio — Exposure{W}\n")
        print(f"  {Y}No live positions yet{W} — gross/net exposure lights up at live deployment (post-approval).")
    elif tab == 3:
        m = signal_correlation_matrix(state, names, symbol)
        sigs, mat = m["signals"], m["matrix"]
        print(f"\n  {BOLD}Portfolio — Cross-signal correlation{W} ({symbol}, target < 0.35)\n")
        print("    " + " ".join(f"{s[:8]:>8}" for s in sigs))
        worst = 0.0
        for i, s in enumerate(sigs):
            cells = []
            for j in range(len(sigs)):
                cells.append(f"{mat[i][j]:>8.2f}")
                if i != j and mat[i][j] == mat[i][j] and abs(mat[i][j]) > worst:
                    worst = abs(mat[i][j])
            print(f"    {s[:8]:<8}" + " ".join(cells))
        col = G if worst < 0.35 else R
        print(f"\n    max |off-diagonal| = {col}{worst:.2f}{W} (target < 0.35)")
    elif tab == 4:
        print(f"\n  {BOLD}Portfolio — Risk{W} ({symbol})\n")
        print(f"    {'signal':<22} {'sharpe':>7} {'maxDD':>8} {'n_days':>7}")
        for n in names:
            r = per_signal_risk(state, n, symbol)
            print(f"    {n:<22} {r['sharpe']:>7.2f} {r['max_dd_bps']:>7.0f}b {r['n_days']:>7}")
        print(f"\n  {DIM}Portfolio-level VaR/leverage/effective-N pending live positions.{W}")
    print()


# Time-granularity → (bar timeframe for whole-day overview, window minutes,
# fine bar timeframe inside a single page) for `nat viz render`.
def _latest_feature_day(args):
    """Resolve the most recent YYYY-MM-DD dir under the feature data dir."""
    dd = Path(_data(args))
    days = sorted(d.name for d in dd.glob('*') if d.is_dir() and d.name[:4].isdigit())
    return days[-1] if days else None


def cmd_viz_render(args):
    """nat viz render — paged PNG viewer over the feature parquet at 1m/5m/15m.

    No INDEX → whole-day overview (all features as --tf bars); INDEX N → zoom
    into the Nth --tf-width window at fine resolution (data-relative anchor)."""
    tf = getattr(args, 'tf', '15m')
    overview_tf, win_min, page_tf = _TF_MAP[tf]
    index = getattr(args, 'index', None)
    sym = _sym(args)

    # --last: freshest-readable tail view (data/features root; bypass day/INDEX/--tf)
    last = getattr(args, 'last', None)
    if last:
        out = getattr(args, 'output', None) or str(REPORTS_DIR / "figures" / "snapshots")
        _banner(f"viz render — {sym} last {last} (freshest readable)")
        sys.stdout.flush()
        cmd = (f"scripts/15m_visualize.py --data-dir {_data(args)} --last {last} "
               f"--symbol {sym} --output {out} --page all")
        feats = getattr(args, 'features', None)
        if feats:
            cmd += f" --features {feats} --max-features {getattr(args, 'max_features', 16)}"
        if getattr(args, 'open_after', False):
            cmd += " --open"
        cmd += " -v"
        return _py(cmd).returncode

    date = getattr(args, 'date', None) or _latest_feature_day(args)
    if not date:
        _p("x", R, f"No feature data days found under {_data(args)}")
        return 1
    data_dir = f"{_data(args)}/{date}"
    if not Path(data_dir).exists():
        _p("x", R, f"No data for {date}: {data_dir}")
        return 1

    out = getattr(args, 'output', None) or str(REPORTS_DIR / "figures" / "snapshots")
    mode = "overview" if index is None else f"page {index}"
    _banner(f"viz render — {sym} {tf} {mode} ({date})")
    sys.stdout.flush()

    cmd = (f"scripts/15m_visualize.py --data-dir {data_dir} --symbol {sym} "
           f"--output {out} --page all")
    if index is None:
        cmd += f" --timeframe {overview_tf}"
    else:
        cmd += f" --window {win_min} --window-index {index} --timeframe {page_tf}"
    feats = getattr(args, 'features', None)
    if feats:
        cmd += f" --features {feats} --max-features {getattr(args, 'max_features', 16)}"
    if getattr(args, 'open_after', False):
        cmd += " --open"
    cmd += " -v"
    return _py(cmd).returncode


def cmd_viz_file(args):
    """nat viz <file> — curated microstructure snapshot of one parquet → show → delete.

    Default renders the SAME curated panels as `nat viz render` / `nat 15m` via
    15m_visualize.py (resampled bars, per-symbol), into a temp dir, opens them, then
    deletes. `--raw` falls back to the raw-tick top-variance quick-look."""
    f = Path(getattr(args, 'file'))
    if not f.exists():
        # Allow a bare filename resolved against the data dir.
        cand = sorted(Path(_data(args)).rglob(f.name))
        if cand:
            f = cand[0]
        else:
            _p("x", R, f"File not found: {args.file}")
            return 1

    sys.path.insert(0, str(ROOT / "scripts"))
    from viz import slice_viewer
    delete = not getattr(args, 'no_delete', False)
    as_json = _json_mode(args)

    # ── raw-tick path (the old slice_viewer quick-look) ──
    if getattr(args, 'raw', False):
        import contextlib
        import io
        cols = getattr(args, 'cols', None)
        cols = cols.split(",") if cols else None
        # In JSON mode, mute slice_viewer's own prints so stdout stays parseable.
        stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if as_json else contextlib.nullcontext()
        try:
            with stdout_ctx:
                manifest = slice_viewer.view(
                    f,
                    slice_expr=getattr(args, 'slice', None),
                    cols=cols,
                    symbol=getattr(args, 'symbol', None),
                    max_features=getattr(args, 'max_features', 16),
                    delete=delete and not as_json,
                    prompt=not as_json,
                )
        except Exception as e:
            if as_json:
                _output({"error": str(e), "file": str(f)}, args)
            else:
                _p("x", R, str(e))
            return 1
        if as_json:
            _output(manifest, args)
        return 0

    # ── default: curated panels via 15m_visualize.py (single-file mode) ──
    import shutil
    import tempfile
    tf = getattr(args, 'tf', '1m')
    overview_tf = _TF_MAP[tf][0]
    sym = getattr(args, 'symbol', None) or 'BTC'
    out_dir = Path(tempfile.mkdtemp(prefix="nat_viz_"))
    cmd = (f"scripts/15m_visualize.py --data-dir {f} --symbol {sym} "
           f"--timeframe {overview_tf} --page all --output {out_dir} "
           f"--max-features {getattr(args, 'max_features', 16)} --quiet")
    r = _sh(f"{PY} {cmd}", cwd=str(ROOT))
    pages = [Path(ln.split("Saved:", 1)[1].strip())
             for ln in (r.stdout or "").splitlines() if "Saved:" in ln]
    if r.returncode != 0 or not pages:
        msg = (r.stderr or r.stdout or "render produced no pages").strip().splitlines()
        if as_json:
            _output({"error": (msg[-1] if msg else "render failed"), "file": str(f)}, args)
        else:
            _p("x", R, "Render failed:")
            for line in msg[-8:]:
                print(f"    {line}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return r.returncode or 1

    manifest = {"file": str(f), "symbol": sym, "timeframe": overview_tf,
                "pages": [str(p) for p in pages], "out_dir": str(out_dir)}
    if as_json:
        _output(manifest, args)
        return 0
    if delete:
        slice_viewer.show_and_delete(pages, wait=True)
    else:
        _p("+", G, f"Rendered {len(pages)} page(s) → {out_dir} (kept, --no-delete)")
        for p in pages:
            print(f"      {p.name}")
    return 0


def cmd_viz3d(args):
    """nat viz3d / nat mesh — 3D feature-surface-over-time.

    Default: interactive Plotly HTML. With --png: static PNG via the native
    `natviz3d` Rust binary (no Plotly/kaleido needed) over the freshest parquet."""
    tf = getattr(args, 'tf', '15m')
    index = getattr(args, 'index', None)
    sym = _sym(args)

    date = getattr(args, 'date', None) or _latest_feature_day(args)
    if not date:
        _p("x", R, f"No feature data days found under {_data(args)}")
        return 1
    data_dir = f"{_data(args)}/{date}"
    if not Path(data_dir).exists():
        _p("x", R, f"No data for {date}: {data_dir}")
        return 1

    # ── Static-PNG path: native natviz3d binary over a single parquet file ──
    if getattr(args, 'png', False):
        bin_v3 = RUST / "target" / "release" / "natviz3d"
        if not bin_v3.exists():
            _p("...", Y, "natviz3d binary missing — building...")
            if _cargo("build --release -p natviz3d").returncode != 0:
                _p("x", R, "Failed to build natviz3d (need `nat build`?)")
                return 1
        files = sorted(Path(data_dir).glob("*.parquet"), key=os.path.getmtime, reverse=True)
        files = [f for f in files if f.stat().st_size > 0]
        if not files:
            _p("x", R, f"No non-empty parquet in {data_dir}")
            return 1
        target = files[0]
        out_dir = Path(getattr(args, 'output', None) or (REPORTS_DIR / "figures" / "mesh"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{sym}_{date}_mesh.png"
        cmd = [str(bin_v3), str(target), "--out", str(out_png),
               "--max-features", str(getattr(args, 'max_features', 40))]
        _banner(f"viz3d (PNG) — {sym} {date}  [{target.name}]")
        sys.stdout.flush()
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            _p("+", G, f"Wrote {out_png}")
            if getattr(args, 'open_after', False):
                sys.path.insert(0, str(ROOT / "scripts"))
                from viz.open_helper import open_path
                open_path(out_png)
        else:
            _p("x", R, f"natviz3d failed (exit {rc})")
        return rc

    out = getattr(args, 'output', None) or str(REPORTS_DIR / "figures" / "mesh")
    mode = "overview" if index is None else f"page {index}"
    _banner(f"viz3d — {sym} {tf} {mode} ({date})")
    sys.stdout.flush()

    cmd = (f"scripts/viz_mesh.py --tf {tf} --symbol {sym} --data-dir {data_dir} "
           f"--date {date} --output {out} --z {getattr(args, 'zmode', 'zscore')} "
           f"--max-features {getattr(args, 'max_features', 40)}")
    if index is not None:
        cmd += f" {index}"
    feats = getattr(args, 'features', None)
    if feats:
        cmd += f" --features {feats}"
    if getattr(args, 'open_after', False):
        cmd += " --open"
    cmd += " -v"
    return _py(cmd).returncode


def register(sub):
    # ── viz (terminal-first per-unit visualization, T7/NAT3) ──
    viz_p = sub.add_parser('viz', help='Terminal-first visualization (features/algorithm/paper)')
    vizsub = viz_p.add_subparsers(dest='subcmd')
    vzfile = vizsub.add_parser('file',
                               help='Curated snapshot of one parquet file → show → delete')
    vzfile.add_argument('file', help='Parquet file (full path or bare name resolved in data dir)')
    vzfile.add_argument('--tf', default='1m', choices=['1m', '5m', '15m'],
                        help='Bar timeframe for the curated panels (default: 1m)')
    vzfile.add_argument('--symbol', '-s', default='BTC', help="Symbol (default: BTC)")
    vzfile.add_argument('--max-features', type=int, default=16, help='Max features per page')
    vzfile.add_argument('--raw', action='store_true',
                        help='Raw-tick top-variance quick-look (old slice_viewer) instead of curated panels')
    vzfile.add_argument('--slice', default=None,
                        help="(--raw only) Rows: 0:500, a count like 200, or 'all'")
    vzfile.add_argument('--cols', default=None,
                        help='(--raw only) Comma-separated columns (default: top-variance)')
    vzfile.add_argument('--data', default=str(DATA_DEFAULT), help='Data dir for bare-name lookup')
    vzfile.add_argument('--no-delete', action='store_true', help='Keep the rendered PNGs')
    vzfile.add_argument('--json', action='store_true', help='Emit manifest instead of opening')
    vzfile.set_defaults(func=cmd_viz_file)
    vzf = vizsub.add_parser('features', help='Per-feature overview (value/z/NaN%/IC/sparkline)')
    vzf.add_argument('--symbol', '-s', default='BTC')
    vzf.add_argument('--hours', type=float, default=None, help='Last N hours only')
    vzf.add_argument('--alive-only', action='store_true', help='Skip all-NaN features')
    vzf.add_argument('--top', type=int, default=40, help='Rows to show')
    vzf.add_argument('--json', action='store_true', help='JSON output')
    vzf.set_defaults(func=cmd_viz_features)
    vza = vizsub.add_parser('algorithm', help='Algorithm signal timeline, IC, entry/exit, P&L proxy')
    vza.add_argument('name', help='Algorithm name (see nat algorithm list)')
    vza.add_argument('--symbol', '-s', default='BTC')
    vza.add_argument('--hours', type=float, default=None, help='Last N hours only')
    vza.add_argument('--horizon', type=int, default=50, help='Forward-return horizon (ticks)')
    vza.add_argument('--threshold', type=float, default=0.3, help='Entry threshold on |signal|')
    vza.add_argument('--json', action='store_true', help='JSON output')
    vza.set_defaults(func=cmd_viz_algorithm)
    vzp = vizsub.add_parser('paper', help='Approval evidence: P&L, IC decay, G8 scorecard (NAT6)')
    vzp.add_argument('name', help='Signal/algorithm name (e.g. 3f_liquidity, jump_detector)')
    vzp.add_argument('--symbol', '-s', default='BTC')
    vzp.add_argument('--json', action='store_true', help='JSON output')
    vzp.set_defaults(func=cmd_viz_paper)
    vzpo = vizsub.add_parser('portfolio', help='Portfolio P&L / exposure / correlation / risk (NAT7)')
    vzpo.add_argument('--tab', type=int, default=1, choices=[1, 2, 3, 4],
                      help='1 P&L · 2 exposure · 3 correlation · 4 risk')
    vzpo.add_argument('--symbol', '-s', default='BTC')
    vzpo.add_argument('--json', action='store_true', help='JSON output')
    vzpo.set_defaults(func=cmd_viz_portfolio)
    vzr = vizsub.add_parser('render',
                            help='Paged PNG viewer at 1m/5m/15m (overview, or page INDEX)')
    vzr.add_argument('--tf', default='15m', choices=['1m', '5m', '15m'],
                     help='Time granularity / page width (default: 15m)')
    vzr.add_argument('index', nargs='?', type=int,
                     help='1-based page: omit for whole-day overview, N to zoom into the Nth window')
    vzr.add_argument('--symbol', '-s', default='BTC', help="Symbol or 'all' (default: BTC)")
    vzr.add_argument('--date', default=None, help='Day YYYY-MM-DD (default: latest available)')
    vzr.add_argument('--last', default=None, metavar='DURATION',
                     help='Render the last N of the freshest readable data (e.g. 15m, 1h); bypasses --date/INDEX')
    vzr.add_argument('--features', default=None,
                     help="Scope to a category / named vector / comma-list / 'all' "
                          "(renders a per-feature panel grid instead of the curated panels)")
    vzr.add_argument('--max-features', type=int, default=16,
                     help='With --features, cap the grid to top-N by variance (default: 16)')
    vzr.add_argument('--open', dest='open_after', action='store_true',
                     help='Open the produced PNG(s) with the default viewer')
    vzr.add_argument('--output', default=None, help='Output directory (default: reports/figures/snapshots)')
    vzr.set_defaults(func=cmd_viz_render)

    # ── viz3d / mesh (interactive 3D feature-surface-over-time) ──
    v3 = sub.add_parser('viz3d', aliases=['mesh'],
                        help='Interactive 3D feature-surface-over-time (Plotly HTML)')
    v3.add_argument('--tf', default='15m', choices=['1m', '5m', '15m'],
                    help='Time granularity / page width (default: 15m)')
    v3.add_argument('index', nargs='?', type=int,
                    help='1-based page: omit for whole-day overview, N for the Nth window')
    v3.add_argument('--symbol', '-s', default='BTC', help='Symbol (default: BTC)')
    v3.add_argument('--date', default=None, help='Day YYYY-MM-DD (default: latest available)')
    v3.add_argument('--features', default=None,
                    help="Category, named vector, comma-list, or 'all' (default: all, capped)")
    v3.add_argument('--z', dest='zmode', default='zscore', choices=['zscore', 'value'],
                    help='Per-feature normalization (default: zscore)')
    v3.add_argument('--max-features', type=int, default=40,
                    help='Cap y-axis to top-N features by variance (default: 40)')
    v3.add_argument('--png', action='store_true',
                    help='Static PNG via the native natviz3d binary (no Plotly/kaleido)')
    v3.add_argument('--open', dest='open_after', action='store_true',
                    help='Open the produced HTML/PNG with the default viewer')
    v3.add_argument('--output', default=None, help='Output directory (default: reports/figures/mesh)')
    v3.set_defaults(func=cmd_viz3d)


__all__ = [
    "cmd_viz_features", "cmd_viz_algorithm", "cmd_viz_paper", "cmd_viz_portfolio",
    "cmd_viz_render", "cmd_viz_file", "cmd_viz3d", "_latest_feature_day", "register",
]
