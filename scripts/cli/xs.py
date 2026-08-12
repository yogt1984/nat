"""`nat xs` — the Class-3 cross-sectional layer (XS-1..XS-10).

Closes METHODOLOGY step 4 for the XS units, which shipped `@register`ed and tested but
unreachable through `nat` — the contract requires register **+** command **+** maturity tag
in the same change, and only the first was done.

**Maturity tags** are carried in the help text here and in each unit's docstring, per
`contracts/README.md`: surfacing them in `nat commands --json` lands with NAT9, which is not
built, so until then the tag lives where a human reads it.

  [PRELIM]  planted + smoke pass, merged — every XS unit
  NOT BETA: BETA requires passing the discovery-IC + cost gate on real data, and
            FINDINGS §7.7/§7.8 record 4 of 6 pre-registered criteria passing. Nothing in
            this layer is promoted, and nothing here should be read as an edge.
"""

from __future__ import annotations

import sys

from cli.common import ROOT, BOLD, W, G, Y, _output


def _xs():
    sys.path.insert(0, str(ROOT / "scripts"))


def cmd_xs_help(args=None):
    print(f"""
  {BOLD}nat xs{W} — Class-3 cross-sectional layer   {Y}[PRELIM — nothing promoted]{W}

    nat xs universe            Candle archive coverage (XS-1) + L2 sampler state (XS-8)
    nat xs capacity            Tradability curve: admitted pairs vs spread/size (XS-5)
    nat xs rank                Rank-IC of scores vs relative forward returns (XS-3)
    nat xs persistence         Rank autocorrelation half-life per score (XS-4)
    nat xs trajectory          Rotation t-stat trajectory toward significance (XS-10)
    nat xs ledger              Program-level multiple-testing ledger (PROC-13)

  Findings: docs/research/FINDINGS.md §7.1-7.8.
  Status: the vol score ranks (§7.4, z -8.37) and persists (§7.5), but the rotation
  clears only 4 of 6 pre-registered criteria (§7.8) and needs ~325 rebalances vs 83.
""")
    return 0


def cmd_xs_universe(args):
    _xs()
    from processes.candles import available_candle_symbols
    from xs.capacity import DEFAULT_L2_DIR
    from pathlib import Path
    data = {}
    for iv in ("1m", "5m", "15m", "1h"):
        data[iv] = len(available_candle_symbols(interval=iv))
    snaps = sorted(Path(DEFAULT_L2_DIR).glob("**/*.parquet"))
    data["l2_snapshots"] = len(snaps)
    data["l2_days"] = len({p.parent.name for p in snaps})

    def _human(d):
        print(f"\n  {BOLD}Candle archive (XS-1){W}")
        for iv in ("1m", "5m", "15m", "1h"):
            print(f"    {iv:<4} {d[iv]:>4} symbols")
        print(f"\n  {BOLD}L2 sampler (XS-8){W}")
        print(f"    {d['l2_snapshots']} sweeps over {d['l2_days']} day(s)")
        print(f"\n  Venue retention caps 1m at ~3.5d (FINDINGS §7.1) — 1m breadth"
              f" accrues, it cannot be backfilled.\n")
    return _output(data, args, _human)


def cmd_xs_capacity(args):
    _xs()
    from xs.capacity import aggregate_l2, load_l2_snapshots, tradability_curve
    agg = aggregate_l2(load_l2_snapshots(), min_snapshots=getattr(args, "min_snapshots", 10))
    curve = tradability_curve(agg, [0.2, 0.5, 1.0, 2.0, 5.0, 50.0],
                              min_touch_notional=getattr(args, "min_notional", None))
    data = {"n_pairs": len(agg), "curve": [{k: v for k, v in c.items() if k != "admitted"}
                                           for c in curve]}

    def _human(d):
        print(f"\n  {BOLD}Tradability curve{W} ({d['n_pairs']} pairs)")
        for c in d["curve"]:
            print(f"    half-spread <= {c['max_half_spread_bps']:>5} bps -> "
                  f"{c['n_admitted']:>3} admitted")
        print(f"\n  L1 touch is the wrong capacity measure for a daily rotation"
              f" (FINDINGS §7.6): use ADV participation.\n")
    return _output(data, args, _human)


def _record_trajectory_point(args) -> dict:
    """Re-run the rotation on the CURRENT archive and append one trajectory point.

    This is what makes §7.8's "~325 rebalances needed, 83 held" self-measuring instead of a
    note someone has to remember. Fires daily from the candle-refresh timer, after new
    candles land.
    """
    _xs()
    from processes.candles import available_candle_symbols, load_candles
    from utils.costs import load_costs, taker_bps
    from xs.capacity import admit, aggregate_l2, load_l2_snapshots
    from xs.rotation import DEFAULTS, run_rotation
    from xs.trajectory import (append_trajectory, default_trajectory_path,
                               evaluate_criteria, power_status)

    costs = load_costs()
    agg = aggregate_l2(load_l2_snapshots(), min_snapshots=10)
    admitted, _ = admit(agg, max_half_spread_bps=DEFAULTS["spread_ceiling_bps"])
    frame = load_candles(available_candle_symbols(interval="1h"), "1h")
    wide = frame.pivot_table(index="timestamp", columns="symbol",
                             values="close", aggfunc="last").sort_index()
    wide = wide[[c for c in wide.columns if c in admitted]]
    # per-pair round trip: its own measured half-spread + the SSOT taker and slippage
    cost_bps = agg.loc[list(wide.columns), "half_spread_bps"] + taker_bps() \
        + float(costs["hyperliquid"]["slippage_bps"])

    # COST-9: funding on held inventory. Previously unpriced — XS-9 passes 4 of 6
    # pre-registered criteria (§7.8), and it held overnight without ever paying funding.
    from data.fetch_funding import load_funding_panel
    funding_wide = load_funding_panel(wide.index, symbols=list(wide.columns))
    if not len(funding_wide.columns):
        funding_wide = None          # reported, never a silent zero-funding price

    m = run_rotation(wide, cost_bps, funding_wide=funding_wide)
    if not m.get("n_periods"):
        return {"recorded": False, "reason": m.get("reason", "no periods")}

    # criterion (f): sign stability under a 2x cost stress
    stressed = run_rotation(wide, cost_bps, cost_stress=2.0, funding_wide=funding_wide)
    m["sign_stable_2x"] = bool(
        stressed.get("n_periods") and
        (stressed["net_total_pct"] >= 0) == (m["net_total_pct"] >= 0))
    m["dsr_p"] = None            # DSR needs the program trial count; ledger owns that

    passed, failed = evaluate_criteria(m)
    p = power_status(m["sharpe_net"], m["n_periods"])
    row = append_trajectory(default_trajectory_path(), {
        "construction": "beta_neutral_score_proportional",
        **m, "passed": passed, "failed": failed,
        "t_stat": round(p["t_stat"], 3),
        "n_required_t2": (None if p["n_required_t2"] == float("inf")
                          else round(p["n_required_t2"])),
        "n_universe": len(wide.columns),
    })
    return {"recorded": True, **row}


def cmd_xs_trajectory(args):
    _xs()
    from xs.trajectory import CRITERIA, default_trajectory_path, power_status, read_trajectory
    if getattr(args, "record", False):
        rec = _record_trajectory_point(args)
        if not rec.get("recorded"):
            print(f"  not recorded: {rec.get('reason')}")
            return 1
        print(f"  recorded: n={rec['n_periods']} SR={rec['sharpe_net']} "
              f"t={rec['t_stat']} passed={rec['passed']} failed={rec['failed']}")
    rows = read_trajectory(default_trajectory_path())
    latest = rows[-1] if rows else None
    data = {"n_runs": len(rows), "latest": latest, "criteria": CRITERIA}
    if latest and latest.get("sharpe_net") is not None:
        data["power"] = power_status(latest["sharpe_net"], latest.get("n_periods", 0))

    def _human(d):
        if not d["latest"]:
            print(f"\n  no trajectory yet — run {BOLD}nat xs trajectory --record{W}"
                  f" after the archive grows\n")
            return
        p, l = d.get("power", {}), d["latest"]
        print(f"\n  {BOLD}Rotation trajectory{W}  ({d['n_runs']} run(s))")
        print(f"    n periods      {l.get('n_periods')}")
        print(f"    net Sharpe     {l.get('sharpe_net')}")
        print(f"    t-statistic    {p.get('t_stat', float('nan')):.2f}")
        print(f"    need for t=2   {p.get('n_required_t2', float('nan')):.0f} periods"
              f"  ({p.get('n_remaining', float('nan')):.0f} remaining)")
        print(f"    resolved       {p.get('resolved')}")
        print(f"    criteria       passed {l.get('passed')} / failed {l.get('failed')}\n")
    return _output(data, args, _human)


def _run_xs_process(name: str, args):
    """Run an XS process through `run_process`.

    Deliberately via the runner rather than by calling the process directly: the runner is
    what applies PROC-13's FDR and appends the sweep to the program-level ledger. Every XS
    result so far was produced by direct calls, which is why the ledger is empty despite 13
    trials having been spent on one 83-day window.
    """
    _xs()
    from processes.runner import run_process
    res = run_process(name, interval=getattr(args, "interval", "1h"),
                      save=not getattr(args, "no_save", False))
    data = {"process": name, "summary": res.summary,
            "findings": [{"feature": f.feature, "metric": f.metric, "value": f.value,
                          "q": f.p_adjusted, "informative": f.informative,
                          "polarity": (f.extras or {}).get("polarity")}
                         for f in res.findings]}

    def _human(d):
        print(f"\n  {BOLD}{d['process']}{W}  {Y}[PRELIM]{W}")
        for f in d["findings"]:
            mark = G + "informative" + W if f["informative"] else "—"
            print(f"    {f['feature']:<32} {f['metric']}={f['value']:+.4f} "
                  f"q={f['q']} {f['polarity'] or ''} {mark}")
        if not d["findings"]:
            print(f"    no findings: {d['summary'].get('skipped_reason', '')}")
        print()
    return _output(data, args, _human)


def cmd_xs_rank(args):
    return _run_xs_process("xs_rank_predictability", args)


def cmd_xs_persistence(args):
    return _run_xs_process("xs_persistence", args)


def cmd_xs_ledger(args):
    _xs()
    from processes.fdr import default_ledger_path, read_ledger
    rows = read_ledger(default_ledger_path())
    by_proc = {}
    for r in rows:
        by_proc[r.get("process", "?")] = by_proc.get(r.get("process", "?"), 0) + int(
            r.get("n_tested", 0) or 0)
    data = {"path": str(default_ledger_path()), "n_sweeps": len(rows),
            "tests_by_process": by_proc, "total_tests": sum(by_proc.values())}

    def _human(d):
        print(f"\n  {BOLD}Program multiple-testing ledger{W} (PROC-13)")
        print(f"    {d['n_sweeps']} sweep(s), {d['total_tests']} test(s) total")
        for k, v in sorted(d["tests_by_process"].items(), key=lambda x: -x[1]):
            print(f"      {k:<28} {v:>5}")
        if not d["n_sweeps"]:
            print(f"    {Y}empty — sweeps run outside `run_process` never reach it{W}")
        print()
    return _output(data, args, _human)


def register(sub):
    xs_p = sub.add_parser('xs', help='Class-3 cross-sectional layer [PRELIM]')
    xs_p.set_defaults(func=cmd_xs_help)
    s = xs_p.add_subparsers(dest='subcmd')

    u = s.add_parser('universe', help='Candle archive + L2 sampler coverage [PRELIM]')
    u.add_argument('--json', action='store_true')
    u.set_defaults(func=cmd_xs_universe)

    c = s.add_parser('capacity', help='Tradability curve (XS-5) [PRELIM]')
    c.add_argument('--min-snapshots', type=int, default=10)
    c.add_argument('--min-notional', type=float, default=None)
    c.add_argument('--json', action='store_true')
    c.set_defaults(func=cmd_xs_capacity)

    t = s.add_parser('trajectory', help='Rotation t-stat trajectory (XS-10) [PRELIM]')
    t.add_argument('--record', action='store_true',
                   help='Re-run the rotation on the current archive and append a point')
    t.add_argument('--json', action='store_true')
    t.set_defaults(func=cmd_xs_trajectory)

    rk = s.add_parser('rank', help='Rank-IC vs relative forward returns (XS-3) [PRELIM]')
    rk.add_argument('--interval', default='1h')
    rk.add_argument('--no-save', action='store_true')
    rk.add_argument('--json', action='store_true')
    rk.set_defaults(func=cmd_xs_rank)

    ps = s.add_parser('persistence', help='Rank autocorrelation half-life (XS-4) [PRELIM]')
    ps.add_argument('--interval', default='1h')
    ps.add_argument('--no-save', action='store_true')
    ps.add_argument('--json', action='store_true')
    ps.set_defaults(func=cmd_xs_persistence)

    lg = s.add_parser('ledger', help='Program multiple-testing ledger (PROC-13)')
    lg.add_argument('--json', action='store_true')
    lg.set_defaults(func=cmd_xs_ledger)
