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
