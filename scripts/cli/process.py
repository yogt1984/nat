"""`nat process` — analytical processes (IC sweep, MI/TE, spectral, ML importance)."""

from __future__ import annotations

import argparse
import sys
import textwrap

from cli.common import ROOT, PY, DATA_DEFAULT, BOLD, W, R, _json_mode, _output, _exec


def cmd_process_help(args=None):
    """Curated group help for analytical processes."""
    print(f"""
  {BOLD}Analytical Processes{W} — third first-class citizen of NAT

  feature = what is computed; algorithm = how it trades; process = whether/
  where information about price action exists. A process is a statistical,
  signal-processing, or ML description scoring features (or derivatives of
  features) against forward returns. Records: data/research/processes/ +
  process_results index in nat.db. Config: config/processes.toml.

    nat process list           Registered processes (--kind evaluation|transform)
    nat process run NAME       Run on feature data (--symbol, --start-date,
                               --features pfx,..., --param k=v, --score-with X)
    nat process results        Past runs from the index (--process, --symbol)
    nat process show RUN_ID    Full findings of one run

  Evaluation processes (score existing features):
    ic_horizon        Expanding Spearman IC x horizon sweep, decay half-life, BH-FDR
    mi_ksg            KSG mutual information vs forward returns, cost-viability gate
    transfer_entropy  Directed information flow with reverse-direction control
    spectral          PSD / Hurst / OU half-life / frequency-band IC (tick-level)
    ml_importance     LightGBM walk-forward importance + confidence-filtered PnL

  Transform processes (derive new evaluable series; chain with --score-with):
    triple_barrier    Lopez de Prado 3-barrier labels (tb_label/tb_ret/tb_hit_bars)
    pca_combo         Orthogonal PCA composites pc_1..pc_k, holdout-IC scored

  Start with:  nat process run ic_horizon --symbol BTC --start-date 2026-06-10
  Chain:       nat process run pca_combo --symbol BTC --score-with ic_horizon
""")


def cmd_process_list(args):
    """List registered analytical processes."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import processes as procs
    kind = getattr(args, 'kind', None)
    names = procs.list_processes_by_kind(kind) if kind else procs.list_processes()
    if _json_mode(args):
        _output({"processes": [procs.get_process(n).describe() for n in names]}, args)
        return
    print(f"\n  Registered processes ({len(names)}):\n")
    for name in names:
        spec = procs.get_process(name).describe()
        doc = spec['doc'].splitlines()[0] if spec['doc'] else ''
        print(f"    {BOLD}{name}{W}  [{spec['kind']}, {spec['data_level']}-level]")
        print(f"      {doc}")
        print(f"      params: {', '.join(spec['params'])}")
    print()


def cmd_process_run(args):
    """Run an analytical process via the runner subprocess."""
    cmd = f"-m processes.runner {args.name} --symbol {args.symbol}"
    if getattr(args, 'data', None):
        cmd += f" --data-dir {args.data}"
    for flag in ('timeframe', 'start_date', 'end_date', 'features', 'score_with'):
        val = getattr(args, flag, None)
        if val:
            cmd += f" --{flag.replace('_', '-')} {val}"
    for kv in getattr(args, 'param', None) or []:
        cmd += f" --param {kv}"
    if getattr(args, 'no_save', False):
        cmd += " --no-save"
    if getattr(args, 'json', False):
        cmd += " --json"
    return _exec(f"{PY} {cmd}", cwd=ROOT / "scripts").returncode


def cmd_process_results(args):
    """List past process runs from the nat.db index."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from processes.persistence import list_results
    rows = list_results(
        process=getattr(args, 'process', None),
        symbol=getattr(args, 'symbol', None),
        limit=getattr(args, 'limit', 20),
    )
    if _json_mode(args):
        _output({"results": rows}, args)
        return
    if not rows:
        print("  No process runs recorded yet. Try: nat process run ic_horizon --symbol BTC")
        return
    print(f"\n  {'run_id':<44} {'tested':>6} {'info':>5} {'top finding':<40}")
    print(f"  {'-'*44} {'-'*6} {'-'*5} {'-'*40}")
    for r in rows:
        top = f"{r['top_feature']} {r['top_metric']}={r['top_value']:.4f}" if r['top_feature'] else "-"
        print(f"  {r['run_id']:<44} {r['n_tested']:>6} {r['n_informative']:>5} {top:<40}")
    print()


def cmd_process_show(args):
    """Show one process run record."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from processes.persistence import load_result
    try:
        record = load_result(args.run_id)
    except FileNotFoundError:
        print(f"  {R}Not found:{W} {args.run_id} (see `nat process results`)")
        return 1
    if _json_mode(args):
        _output(record, args)
        return
    s = record['summary']
    prov = record.get('provenance', {})
    print(f"\n  {BOLD}{record['run_id']}{W}")
    print(f"  process={record['process']} kind={record['kind']} "
          f"symbol={record['symbol']} timeframe={record['timeframe']}")
    print(f"  data: {record['data'].get('start_date')}..{record['data'].get('end_date')} "
          f"({record['data'].get('n_bars') or record['data'].get('n_rows')} rows/bars)")
    print(f"  provenance: git_sha={prov.get('git_sha')} dirty={prov.get('dirty')}")
    print(f"  tested={s['n_tested']} informative={s['n_informative']} "
          f"skipped={len(record['features_skipped'])} runtime={s['runtime_s']}s")
    if s.get('error'):
        print(f"  {R}ERROR:{W} {s['error']}")
    findings = sorted(record['findings'], key=lambda f: abs(f['value']), reverse=True)
    if findings:
        print(f"\n  {'feature':<40} {'horizon':>8} {'metric':>14} {'value':>10} {'p_adj':>8} {'info':>5}")
        for f in findings[: getattr(args, 'top', 20)]:
            p_adj = f"{f['p_adjusted']:.4f}" if f.get('p_adjusted') is not None else "-"
            mark = "*" if f['informative'] else ""
            print(f"  {f['feature']:<40} {str(f['horizon']):>8} {f['metric']:>14} "
                  f"{f['value']:>10.5f} {p_adj:>8} {mark:>5}")
    if record.get('derived'):
        print(f"\n  derived: {record['derived']['columns']} -> {record['derived']['parquet']}")
    print()


def register(sub):
    proc_p = sub.add_parser('process',
        help='Analytical processes (IC sweep, MI/TE, spectral, ML importance)')
    proc_p.set_defaults(func=cmd_process_help)
    procsub = proc_p.add_subparsers(dest='subcmd')
    pl = procsub.add_parser('list', help='List registered processes')
    pl.add_argument('--kind', choices=['evaluation', 'transform'], default=None)
    pl.add_argument('--json', action='store_true', help='JSON output')
    pl.set_defaults(func=cmd_process_list)
    pr = procsub.add_parser('run', help='Run a process on feature data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Loads only the columns the process needs (schema peek + columns= pruning),
        aggregates to bars (or feeds raw ticks for tick-level processes), runs the
        process, and persists JSON + index row with provenance.

        Examples:
          nat process run ic_horizon --symbol BTC --start-date 2026-06-10
          nat process run mi_ksg --symbol ETH --features imbalance_,ent_
          nat process run transfer_entropy --symbol BTC --param te_method=linear
        """))
    pr.add_argument('name', help='Process name (see `nat process list`)')
    pr.add_argument('--symbol', default='BTC')
    pr.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    pr.add_argument('--timeframe', default=None, help='Bar timeframe (default from processes.toml)')
    pr.add_argument('--start-date', default=None, help='e.g. 2026-06-05')
    pr.add_argument('--end-date', default=None)
    pr.add_argument('--features', default=None, help='Comma-separated name prefixes to score')
    pr.add_argument('--param', action='append', default=[], metavar='K=V',
                    help='Process param override (repeatable)')
    pr.add_argument('--score-with', default=None,
                    help='Evaluation process chained onto transform output')
    pr.add_argument('--no-save', action='store_true', help='Skip persistence')
    pr.add_argument('--json', action='store_true', help='Print full result JSON')
    pr.set_defaults(func=cmd_process_run)
    prr = procsub.add_parser('results', help='List past runs from the nat.db index')
    prr.add_argument('--process', default=None, help='Filter by process name')
    prr.add_argument('--symbol', default=None)
    prr.add_argument('--limit', type=int, default=20)
    prr.add_argument('--json', action='store_true', help='JSON output')
    prr.set_defaults(func=cmd_process_results)
    psh = procsub.add_parser('show', help='Show one run record')
    psh.add_argument('run_id')
    psh.add_argument('--top', type=int, default=20, help='Findings rows to print')
    psh.add_argument('--json', action='store_true', help='JSON output')
    psh.set_defaults(func=cmd_process_show)


__all__ = ["cmd_process_help", "cmd_process_list", "cmd_process_run",
           "cmd_process_results", "cmd_process_show", "register"]
