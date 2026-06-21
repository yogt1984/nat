"""`nat algorithm` (alias `alg`) — microstructure algorithm evaluation."""

from __future__ import annotations

import argparse
import json as _json
import sys
import textwrap

from cli.common import (
    ROOT, CONFIG_DIR, REPORTS_DIR, DATA_DEFAULT, BOLD, W, R,
    _data, _sym, _py, _json_mode, _output,
)


def cmd_algorithm_evaluate(args):
    """Evaluate algorithm IC/drift on data."""
    cmd = f"scripts/algorithms/evaluate.py --data-dir {_data(args)} --symbol {_sym(args)}"
    alg = getattr(args, 'algorithm', None)
    if alg:
        cmd += f" --algorithm {alg}"
    elif getattr(args, 'all', False):
        cmd += " --all"
    else:
        cmd += " --all"
    report = getattr(args, 'json_report', None)
    if report:
        cmd += f" --json-report {report}"
    _py(cmd)


def _algorithm_perf():
    """Best-effort per-algorithm metrics: lifecycle state + IC from eval reports.

    Returns {alg_name: {"state": str|None, "ic": {symbol: float}}}. Both sources
    are optional — missing files/DB degrade gracefully to empty entries."""
    perf: dict = {}

    # Lifecycle state (matched by signal name).
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from signal_lifecycle import SignalLifecycle
        for row in SignalLifecycle().list_signals():
            nm = row.get("name") or row.get("signal_id")
            if nm:
                perf.setdefault(nm, {}).setdefault("state", row.get("state"))
    except Exception:
        pass

    # IC from reports/algorithms/<SYM>_eval.json (list of per-algo result dicts).
    eval_dir = REPORTS_DIR / "algorithms"
    if eval_dir.exists():
        for ef in sorted(eval_dir.glob("*_eval.json")):
            sym = ef.name.split("_", 1)[0]
            try:
                results = _json.loads(ef.read_text()).get("results", [])
            except Exception:
                continue
            for r in results:
                nm = r.get("algorithm")
                ic = r.get("ic")
                if nm is not None and ic is not None:
                    perf.setdefault(nm, {}).setdefault("ic", {})[sym] = ic
    return perf


def cmd_algorithm_list(args):
    """List registered algorithms with lifecycle state + IC metrics."""
    import contextlib
    import io
    sys.path.insert(0, str(ROOT / "scripts"))
    # Algorithm/model imports print banners on load; mute them so --json stays
    # machine-parseable (and human output stays clean).
    with contextlib.redirect_stdout(io.StringIO()):
        from algorithms.autodiscover import discover_all
        discover_all()
        from algorithms.registry import list_algorithms
        from algorithms import get_algorithm
        perf = _algorithm_perf()
        result = []
        for name in list_algorithms():
            alg = get_algorithm(name)  # may print model banners — suppressed here
            feats = alg.alg_features()
            p = perf.get(name, {})
            result.append({
                "name": name,
                "state": p.get("state"),
                "ic": p.get("ic", {}),
                "features": [{"name": f.name, "warmup": f.warmup, "description": f.description} for f in feats],
                "required_columns": alg.required_columns(),
            })

    if _json_mode(args):
        _output({"algorithms": result}, args)
        return

    print(f"\n  Registered algorithms ({len(result)}):\n")
    for a in result:
        state, ic = a["state"], a["ic"]
        tag = f"  [{state}]" if state else ""
        ic_str = ("   IC: " + " ".join(f"{s}={v:+.3f}" for s, v in sorted(ic.items()))) if ic else ""
        print(f"    {BOLD}{a['name']}{W}  ({len(a['features'])} features){tag}{ic_str}")
        for f in a["features"]:
            print(f"      - {f['name']}  (warmup={f['warmup']})  {f['description']}")
    print()


def cmd_algorithm_config(args):
    """Show algorithm configuration from TOML."""
    config_path = CONFIG_DIR / "algorithms.toml"
    if not config_path.exists():
        print(f"  {R}Not found:{W} {config_path}")
        return 1

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    if _json_mode(args):
        _output(cfg, args)
        return

    section = getattr(args, 'section', None)
    if section:
        if section in cfg:
            print(f"\n  {BOLD}[{section}]{W}\n")
            for k, v in cfg[section].items():
                print(f"    {k} = {v}")
            print()
        else:
            print(f"  {R}Section not found:{W} {section}")
            print(f"  Available: {', '.join(cfg.keys())}")
        return

    for sect_name, sect_data in cfg.items():
        if isinstance(sect_data, dict):
            print(f"\n  {BOLD}[{sect_name}]{W}")
            for k, v in sect_data.items():
                if isinstance(v, dict):
                    print(f"    [{sect_name}.{k}]")
                    for kk, vv in v.items():
                        print(f"      {kk} = {vv}")
                else:
                    print(f"    {k} = {v}")
    print()


def register(sub):
    alg_p = sub.add_parser('algorithm', aliases=['alg'], help='Microstructure algorithm evaluation')
    alg_p.set_defaults(func=lambda a: alg_p.print_help())
    algsub = alg_p.add_subparsers(dest='subcmd')
    alge = algsub.add_parser('evaluate', help='Run IC/drift evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Evaluates microstructure algorithms by computing Information Coefficient (IC)
        at multiple forward horizons, with optional regime stratification.

        Mathematics:
          IC at horizon h:  IC_h = Spearman ρ(alg_feature_t, r_t(h))
          Horizons:         h ∈ {1, 5, 10, 50, 100} ticks (100ms each)
          Regime-gated IC:  IC computed only where ent_book_shape < P30
                            (low-entropy = structured book = cleaner signal)
          Significance:     z = IC · √n,  reject H0 if |z| > 1.96

        Algorithms: 18 registered (use 'nat algorithm list' to see all)

        Example:
          nat algorithm evaluate --all --symbol BTC
          nat algorithm evaluate --algorithm kalman_imbalance --json-report report.json
        """))
    alge.add_argument('--algorithm', default=None, help='Algorithm name (default: all)')
    alge.add_argument('--all', action='store_true', help='Evaluate all registered algorithms')
    alge.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    alge.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    alge.add_argument('--json-report', default=None, help='Output JSON report path')
    alge.set_defaults(func=cmd_algorithm_evaluate)
    alglist = algsub.add_parser('list', aliases=['ls'],
                                help='List registered algorithms + lifecycle state + IC')
    alglist.add_argument('--json', action='store_true', help='JSON output')
    alglist.set_defaults(func=cmd_algorithm_list)
    algc = algsub.add_parser('config', help='Show algorithm configuration (from algorithms.toml)')
    algc.add_argument('section', nargs='?', default=None, help='Specific algorithm section')
    algc.set_defaults(func=cmd_algorithm_config)


__all__ = ["cmd_algorithm_evaluate", "cmd_algorithm_list", "cmd_algorithm_config", "register"]
