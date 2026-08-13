"""`nat wallets` — the on-chain wallet layer (WP-1..5).

**Maturity: [PRELIM]** — planted + smoke pass, merged. Nothing here is a signal; this is the
substrate the deterministic-liquidation family (`research/MECHANISM_FAMILIES.md`, family 5)
needs before it can be tested at all.
"""

from __future__ import annotations

import sys

from cli.common import ROOT, BOLD, W, Y, _output


def cmd_wallets_help(args=None):
    print(f"""
  {BOLD}nat wallets{W} — on-chain wallet layer   {Y}[PRELIM — substrate, not a signal]{W}

    nat wallets roster        Derive the wallet roster from the venue leaderboard (WP-1)
    nat wallets positions     One position sweep across the roster (WP-2)
    nat wallets panel         Accrual status of the collected panel (WP-2 clock)
    nat wallets cohorts       Rank wallets on realised P&L, causally (WP-3)

  Spec: docs/specs/wallet_positioning.md · Families: research/MECHANISM_FAMILIES.md (5, 7)
""")
    return 0


def cmd_wallets_roster(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from data.wallet_roster import fetch_roster
    refs = fetch_roster(
        limit=getattr(args, "limit", 200),
        window=getattr(args, "window", "month"),
        min_window_pnl=(0.0 if getattr(args, "profitable", False) else None),
        min_notional_seen=getattr(args, "min_notional", None),
        rank_by=getattr(args, "rank_by", "notional_seen"),
    )
    data = {"n": len(refs), "rank_by": getattr(args, "rank_by", "notional_seen"),
            "window": getattr(args, "window", "month"),
            "wallets": [{"address": r.address, "account_value": r.account_value,
                         "notional_seen": r.notional_seen, "window_pnl": r.window_pnl,
                         "source": r.source} for r in refs]}

    def _human(d):
        print(f"\n  {BOLD}Wallet roster{W} — {d['n']} wallets, ranked by {d['rank_by']} "
              f"({d['window']})   {Y}[PRELIM]{W}")
        print(f"    {'address':<44}{'acct value':>15}{'pnl':>13}{'volume':>17}")
        for w in d["wallets"][:15]:
            print(f"    {w['address']:<44}${w['account_value']:>14,.0f}"
                  f"${w['window_pnl']:>12,.0f}${w['notional_seen']:>16,.0f}")
        if d["n"] > 15:
            print(f"    … {d['n'] - 15} more")
        print(f"\n  Derived from the venue leaderboard, never hardcoded — a pinned list rots\n"
              f"  the moment the cohort turns over, and turnover is itself under test.\n")
    return _output(data, args, _human)


def cmd_wallets_positions(args):
    """WP-2 — one sweep now. The daemon (`nat-position-sampler.service`) is the real clock."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from data.fetch_positions import DATA_DIR, run_sweep
    from data.wallet_roster import fetch_roster

    refs = fetch_roster(limit=args.max_wallets, min_notional_seen=args.min_notional,
                        rank_by="notional_seen")
    rep = run_sweep([r.address for r in refs], DATA_DIR,
                    max_wallets=args.max_wallets)

    def _human(d):
        print(f"\n  {BOLD}Position sweep{W} — {d['ok']} with positions, {d['empty']} flat, "
              f"{d['failed']} unreachable   {Y}[PRELIM]{W}")
        print(f"    positions written : {d['n_positions']}")
        print(f"    file              : {d['path']}")
        if d["failed"]:
            print(f"    {Y}unreachable wallets are written with status=failed, never dropped{W}")
        print(f"\n  Every uncollected day is permanently lost — WP-5 needs 90 days.\n")
    return _output(rep, args, _human)


def cmd_wallets_panel(args):
    """How much accrual is actually banked. The WP-5 clock, read off disk."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from data.fetch_positions import DATA_DIR

    days = sorted(p.name for p in DATA_DIR.glob("*-*-*") if p.is_dir()) if DATA_DIR.exists() else []
    sweeps = sum(1 for p in DATA_DIR.glob("*-*-*/positions_*.parquet")) if DATA_DIR.exists() else 0
    data = {"days_collected": len(days), "sweeps": sweeps,
            "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
            "days_required_wp5": 90, "days_remaining": max(0, 90 - len(days))}

    def _human(d):
        print(f"\n  {BOLD}Position panel{W} — WP-2 accrual   {Y}[PRELIM]{W}")
        print(f"    days collected : {d['days_collected']} / 90   "
              f"({d['first_day'] or '—'} → {d['last_day'] or '—'})")
        print(f"    sweeps         : {d['sweeps']}")
        print(f"    remaining      : {d['days_remaining']} days until WP-5 is answerable")
        if d["days_collected"] == 0:
            print(f"    {Y}nothing collected — the clock has not started{W}")
        print()
    return _output(data, args, _human)


def cmd_wallets_cohorts(args):
    """WP-3 — rank wallets on realised P&L over a window ending strictly before `as_of`."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import pandas as pd
    from wallets.cohorts import load_ledger, load_position_panel, rank_cohorts

    panel = load_position_panel()
    if panel.empty:
        print("  no position panel — is nat-position-sampler.service running?")
        return 1
    ledger = load_ledger()

    as_of = (int(pd.Timestamp(args.as_of, tz="UTC").timestamp() * 1000)
             if args.as_of else int(panel.ts_ms.max()))
    lookback_ms = int(args.lookback * 86_400_000)
    try:
        out = rank_cohorts(panel, as_of=as_of, lookback_ms=lookback_ms,
                           k=args.k, ledger=ledger)
    except ValueError as e:
        print(f"  refused: {e}")
        return 1

    w0, w1 = out["window"]
    data = {"as_of": out["as_of"], "window_utc": [str(pd.Timestamp(w0, unit='ms', tz='UTC')),
                                                  str(pd.Timestamp(w1, unit='ms', tz='UTC'))],
            "n_ranked": out["n_ranked"], "n_contaminated": out["n_contaminated"],
            "top": out["top"], "bottom": out["bottom"],
            "top_scores": {w: out["scores"][w] for w in out["top"]}}

    def _human(d):
        print(f"\n  {BOLD}Cohorts{W} — ranked on realised P&L, window ends BEFORE as_of   "
              f"{Y}[PRELIM]{W}")
        print(f"    window   : {d['window_utc'][0]}  ->  {d['window_utc'][1]}")
        print(f"    ranked   : {d['n_ranked']} wallets")
        if d["n_contaminated"]:
            print(f"    {Y}contaminated: {d['n_contaminated']} wallets have an unknown "
                  f"ledger delta type in this window — their P&L is not trustworthy{W}")
        print(f"\n    {'top cohort':<44}{'realised P&L':>16}")
        for w in d["top"][:10]:
            print(f"    {w:<44}{d['top_scores'][w]:>16,.2f}")
        print(f"\n  Ranking is causal by construction — a wallet that only profits after\n"
              f"  as_of cannot enter the top cohort. Rank stability (the early kill) needs\n"
              f"  >=30 days of WP-2 accrual; this is inspection, not a verdict.\n")
    return _output(data, args, _human)


def register(sub):
    p = sub.add_parser('wallets', help='On-chain wallet layer (WP-1..5) [PRELIM]')
    p.set_defaults(func=cmd_wallets_help)
    s = p.add_subparsers(dest='subcmd')
    r = s.add_parser('roster', help='Derive the wallet roster from the leaderboard (WP-1) [PRELIM]')
    r.add_argument('--limit', type=int, default=200)
    r.add_argument('--window', default='month', choices=['day', 'week', 'month', 'allTime'])
    r.add_argument('--rank-by', default='notional_seen',
                   choices=['account_value', 'notional_seen', 'window_pnl'])
    r.add_argument('--min-notional', type=float, default=1e6,
                   help='Volume floor — excludes vaults/bridge accounts (default 1e6)')
    r.add_argument('--profitable', action='store_true', help='Keep only positive window P&L')
    r.add_argument('--json', action='store_true')
    r.set_defaults(func=cmd_wallets_roster)

    p2 = s.add_parser('positions', help='One position sweep across the roster (WP-2) [PRELIM]')
    p2.add_argument('--max-wallets', type=int, default=200)
    p2.add_argument('--min-notional', type=float, default=1e6)
    p2.add_argument('--json', action='store_true')
    p2.set_defaults(func=cmd_wallets_positions)

    p3 = s.add_parser('panel', help='Accrual status of the position panel (WP-2 clock) [PRELIM]')
    p3.add_argument('--json', action='store_true')
    p3.set_defaults(func=cmd_wallets_panel)

    p4 = s.add_parser('cohorts', help='Rank wallets on realised P&L, causally (WP-3) [PRELIM]')
    p4.add_argument('--as-of', default=None,
                    help='UTC timestamp; the ranking window ends strictly BEFORE it')
    p4.add_argument('--lookback', type=float, default=2.0, help='Ranking window, in days')
    p4.add_argument('--k', type=int, default=20, help='Cohort size')
    p4.add_argument('--json', action='store_true')
    p4.set_defaults(func=cmd_wallets_cohorts)
