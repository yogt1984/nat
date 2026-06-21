"""`nat test` — Rust unit tests, capture-and-visualize, validations, pytest suites."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cli.common import (
    ROOT, RUST, ING_CFG, LOG_DIR, DATA_DEFAULT, REPORTS_DIR, PY,
    G, R, B, _TF_MAP,
    _sh, _exec, _cargo, _py, _pid, _p, _banner, _data, _json_mode, _ensure_release,
)


def cmd_test_scan(args):
    """Run scalp edge scanner tests."""
    cov = getattr(args, 'coverage', False)
    cmd = "scripts/tests/test_scalp_scanner.py -v"
    if cov:
        cmd += " --cov=scripts.scalp_edge_scanner --cov-report=term-missing"
    _py(f"-m pytest {cmd}")


def cmd_test_viz(args):
    """Run scanner visualization tests."""
    cov = getattr(args, 'coverage', False)
    cmd = "scripts/tests/test_visualize_scanner.py -v"
    if cov:
        cmd += " --cov=scripts.visualize_scanner --cov-report=term-missing"
    _py(f"-m pytest {cmd}")


def cmd_test(args):
    _banner("Running all Rust unit tests")
    _cargo("test --package ing")


def cmd_test_capture(args):
    """nat test {1m,5m,15m} — capture N minutes locally, then visualize.

    Requires the persistent ingestor to be stopped (it owns the WebSocket and the
    same data dir). Runs a time-boxed `ing`, then renders via the same
    `15m_visualize.py` pipeline as `nat viz render --tf` (resampled bars, curated
    panels, per-symbol) so the output matches the established viz path exactly."""
    minutes = getattr(args, 'minutes')
    secs = int(minutes * 60)

    if _pid():
        _p("x", R, "Ingestor is RUNNING — stop it first: nat ing stop")
        return 1
    _ensure_release()

    dd = Path(_data(args))
    before = {f: f.stat().st_mtime for f in dd.rglob("*.parquet")} if dd.exists() else {}

    verbose = getattr(args, 'verbose', False)
    _banner(f"nat test {minutes:g}m — capture & visualize")
    _p("...", B, f"Capturing {secs}s of live data...")
    sys.stdout.flush()

    # timeout sends SIGTERM at the deadline so the ingestor flushes Parquet on
    # shutdown; --kill-after is a hard backstop if the flush hangs. Quiet the
    # ingestor's tracing to a logfile (RUST_LOG=warn) unless --verbose.
    ing_cmd = (f"timeout --signal=TERM --kill-after=20 {secs} "
               f"./target/release/ing {ING_CFG}")
    if verbose:
        rc = _exec(ing_cmd, cwd=RUST).returncode
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logfile = LOG_DIR / f"test_capture_{datetime.now():%Y%m%d_%H%M%S}.log"
        with open(logfile, "w") as lf:
            rc = subprocess.run(
                ing_cmd, shell=True, cwd=str(RUST),
                env={**os.environ, "RUST_LOG": "warn"},
                stdout=lf, stderr=subprocess.STDOUT,
            ).returncode
    # timeout exits 124 when it has to terminate the child — that's the expected
    # path here (we *want* it to stop at the deadline), so don't treat it as error.
    if rc not in (0, 124, 143):
        hint = "see output above" if verbose else f"see {logfile}"
        _p("x", R, f"Ingestor exited with code {rc} — {hint}")
        return 1

    time.sleep(1)
    after = sorted(
        (f for f in dd.rglob("*.parquet") if f.stat().st_size > 0),
        key=os.path.getmtime, reverse=True,
    )
    fresh = [f for f in after if before.get(f) != f.stat().st_mtime]
    target = (fresh or after)
    if not target:
        _p("x", R, "No parquet data was written during the capture window")
        return 1
    newest = target[0]
    try:
        import pyarrow.parquet as _pq
        nrows = _pq.read_metadata(newest).num_rows
        _p("+", G, f"Captured {nrows:,} rows → {newest.name}")
    except Exception:
        _p("+", G, f"Captured → {newest.name}")

    # Render through the SAME path as `nat viz render --tf` so the image matches
    # the established viz output: resampled bars at the matching timeframe, curated
    # panels, scoped to the just-captured window via --last, per-symbol.
    tf = {1: '1m', 5: '5m', 15: '15m'}.get(int(minutes), '5m')
    overview_tf = _TF_MAP[tf][0]
    sym = getattr(args, 'symbol', None) or 'BTC'
    out = str(REPORTS_DIR / "figures" / "snapshots")
    cmd = (f"scripts/15m_visualize.py --data-dir {_data(args)} --last {int(minutes)}m "
           f"--symbol {sym} --output {out} --page all --timeframe {overview_tf} "
           f"--max-features {getattr(args, 'max_features', 16)}")
    if not _json_mode(args):
        cmd += " --open"

    if verbose:
        return _py(cmd + " -v").returncode

    # Quiet render: capture output, surface only the window + saved pages.
    r = _sh(f"{PY} {cmd}", cwd=str(ROOT))
    if r.returncode != 0:
        _p("x", R, "Render failed:")
        for line in (r.stderr or r.stdout or "").strip().splitlines()[-8:]:
            print(f"    {line}")
        return r.returncode
    saved = [ln.split("Saved:", 1)[1].strip()
             for ln in r.stdout.splitlines() if "Saved:" in ln]
    window = next((ln.split("→", 1)[1].strip()
                   for ln in r.stdout.splitlines() if "Last " in ln and "→" in ln), None)
    if window:
        _p("+", G, f"Window: {window}")
    _p("+", G, f"Rendered {len(saved)} page(s) → {Path(out).relative_to(ROOT)}/")
    for p in saved:
        print(f"      {Path(p).name}")
    return 0


def cmd_test_verbose(args):
    _cargo("test --package ing -- --nocapture")


def cmd_test_hypotheses(args):
    _ensure_release()
    _banner("Running hypothesis tests")
    _exec(f"./target/release/test_hypotheses ../{_data(args)}", cwd=RUST)


def cmd_test_validate(args):
    _ensure_release()
    target = getattr(args, 'target', None)
    if target:
        _banner(f"Validating: {target}")
        _exec(f"./target/release/validate_{target}", cwd=RUST)
    else:
        _banner("Running all API validations")
        for i, name in enumerate(["api", "positions", "whales", "entropy"], 1):
            print(f"  [{i}/4] {name}...")
            _exec(f"./target/release/validate_{name}", cwd=RUST)
            print()


def cmd_test_api(args):
    _exec("bash scripts/test_api.sh", cwd=ROOT)


def cmd_test_redis(args):
    _exec("redis-cli ping && echo 'Redis is running' || echo 'Redis not running'")
    _exec("redis-cli KEYS 'nat:latest:*'")


def cmd_test_integration(args):
    _exec("bash scripts/test_integration.sh", cwd=ROOT)


def cmd_test_backtest(args):
    cov = getattr(args, 'coverage', False)
    cmd = "backtest/tests/ -v"
    if cov:
        cmd += " --cov=backtest --cov-report=term-missing"
    _exec(f"{PY} -m pytest {cmd}", cwd=ROOT / "scripts")


def cmd_test_cluster(args):
    cov = getattr(args, 'coverage', False)
    cmd = "scripts/cluster_quality/tests/ -v"
    if cov:
        cmd += " --cov=cluster_quality --cov-report=term-missing"
    _py(f"-m pytest {cmd}")


def cmd_test_pipeline(args):
    cov = getattr(args, 'coverage', False)
    tests = (
        "tests/test_cluster_loader.py tests/test_cluster_preprocess.py "
        "tests/test_cluster_engine.py tests/test_cluster_reduce.py tests/test_cluster_viz.py"
    )
    cmd = f"{tests} -v"
    if cov:
        cmd += " --cov=cluster_pipeline --cov-report=term-missing"
    _exec(f"{PY} -m pytest {cmd}", cwd=ROOT / "scripts")


def cmd_test_pipeline_runner(args):
    _exec(f"{PY} -m pytest tests/test_pipeline_runner.py -v", cwd=ROOT / "scripts")


def cmd_test_dashboard(args):
    _exec(f"{PY} -m pytest tests/test_dashboard.py -v", cwd=ROOT / "scripts")


def cmd_test_serving(args):
    _py("-m pytest scripts/tests/test_model_serving.py -v")


def cmd_test_eamm(args):
    integration = getattr(args, 'integration', False)
    if integration:
        _exec(f"{PY} -m pytest eamm/tests/test_integration.py -v", cwd=ROOT / "scripts")
    else:
        _exec(f"{PY} -m pytest eamm/tests/ -v", cwd=ROOT / "scripts")


def cmd_test_snapshot(args):
    data_dir = getattr(args, 'data_dir', str(ROOT / "data" / "features"))
    symbol = getattr(args, 'symbol', 'BTC')
    hours = getattr(args, 'hours', 1)
    _py(f"scripts/test_regression.py snapshot --data-dir {data_dir} --symbol {symbol} --hours {hours}")


def cmd_test_regression(args):
    _py("scripts/test_regression.py check")


def cmd_test_process(args):
    """Run process framework tests."""
    _exec(f"{PY} -m pytest tests/test_process_base.py tests/test_process_ic.py "
          f"tests/test_process_persistence.py tests/test_process_info.py "
          f"tests/test_process_spectral.py tests/test_process_ml.py "
          f"tests/test_process_transforms.py tests/test_process_real_data.py -v",
          cwd=ROOT / "scripts")


def register(sub):
    # ── test ──
    test_p = sub.add_parser('test', help='Testing (default: all Rust unit tests)')
    test_p.set_defaults(func=cmd_test)
    tsub = test_p.add_subparsers(dest='subcmd')
    tsub.add_parser('unit', help='Rust unit tests').set_defaults(func=cmd_test)
    tsub.add_parser('verbose', help='Tests with --nocapture').set_defaults(func=cmd_test_verbose)
    # Capture-and-visualize: ingest N minutes locally (ingestor must be stopped),
    # then render via the same 15m_visualize.py path as `nat viz render --tf`.
    for _label, _mins in (('1m', 1), ('5m', 5), ('15m', 15)):
        _tcap = tsub.add_parser(_label, help=f'Capture {_label} of live data then visualize')
        _tcap.add_argument('--symbol', '-s', default='BTC',
                           help="Symbol or 'all' (default: BTC)")
        _tcap.add_argument('--max-features', type=int, default=16, help='Max features per page')
        _tcap.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
        _tcap.add_argument('--verbose', '-v', action='store_true',
                           help='Stream full ingestor + render logs (default: clean summary)')
        _tcap.set_defaults(func=cmd_test_capture, minutes=_mins)
    tsub.add_parser('hypotheses', help='Hypothesis tests').set_defaults(func=cmd_test_hypotheses)
    tv = tsub.add_parser('validate', help='Live API validations')
    tv.add_argument('target', nargs='?', choices=['api', 'positions', 'whales', 'entropy'],
                    help='Specific validation (omit for all)')
    tv.set_defaults(func=cmd_test_validate)
    tsub.add_parser('api', help='Test API endpoints').set_defaults(func=cmd_test_api)
    tsub.add_parser('redis', help='Test Redis').set_defaults(func=cmd_test_redis)
    tsub.add_parser('integration', help='Integration tests').set_defaults(func=cmd_test_integration)
    tb = tsub.add_parser('backtest', help='Backtest tests')
    tb.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    tb.set_defaults(func=cmd_test_backtest)
    tc = tsub.add_parser('cluster', help='Cluster quality tests')
    tc.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    tc.set_defaults(func=cmd_test_cluster)
    tp = tsub.add_parser('pipeline', help='Pipeline tests')
    tp.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    tp.set_defaults(func=cmd_test_pipeline)
    tsub.add_parser('pipeline-runner', help='Pipeline runner tests').set_defaults(func=cmd_test_pipeline_runner)
    tsub.add_parser('dashboard', help='Dashboard tests').set_defaults(func=cmd_test_dashboard)
    tsub.add_parser('serving', help='Model serving tests').set_defaults(func=cmd_test_serving)
    te = tsub.add_parser('eamm', help='EAMM tests')
    te.add_argument('--integration', action='store_true', help='Include integration tests')
    te.set_defaults(func=cmd_test_eamm)
    tsc = tsub.add_parser('scan', help='Scalp edge scanner tests')
    tsc.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    tsc.set_defaults(func=cmd_test_scan)
    tvz = tsub.add_parser('viz', help='Scanner visualization tests')
    tvz.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    tvz.set_defaults(func=cmd_test_viz)
    tsnap = tsub.add_parser('snapshot', help='Capture data snapshot + save algorithm baseline')
    tsnap.add_argument('--data-dir', default=str(ROOT / "data" / "features"))
    tsnap.add_argument('--symbol', default='BTC')
    tsnap.add_argument('--hours', type=int, default=1)
    tsnap.set_defaults(func=cmd_test_snapshot)
    tsub.add_parser('regression', help='Run algorithms and compare against baseline').set_defaults(func=cmd_test_regression)
    tsub.add_parser('process', help='Process framework tests (synthetic + real-data smoke)').set_defaults(func=cmd_test_process)


__all__ = [
    "cmd_test", "cmd_test_scan", "cmd_test_viz", "cmd_test_capture",
    "cmd_test_verbose", "cmd_test_hypotheses", "cmd_test_validate",
    "cmd_test_api", "cmd_test_redis", "cmd_test_integration", "cmd_test_backtest",
    "cmd_test_cluster", "cmd_test_pipeline", "cmd_test_pipeline_runner",
    "cmd_test_dashboard", "cmd_test_serving", "cmd_test_eamm", "cmd_test_snapshot",
    "cmd_test_regression", "cmd_test_process", "register",
]
