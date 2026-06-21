"""`nat lifecycle` — signal promotion state machine (DISCOVERED→LIVE→RETIRED)."""

from __future__ import annotations

import sys

from cli.common import ROOT, BOLD, W, R, Y, G, _json_mode, _output

# Restored with the handlers during extraction — these were dropped from the
# monolith earlier in the D2 refactor, which broke `nat lifecycle status|seed`
# (the parser-tree snapshot oracle can't catch handler-body NameErrors).
_LIFECYCLE_STATES = ["DISCOVERED", "VALIDATED", "PAPER_TRADING", "APPROVAL_PENDING",
                     "LIVE", "MONITORING", "RETIRED", "REJECTED"]
_LIFECYCLE_SEED = {
    "VALIDATED": ["jump_detector", "optimal_entry", "funding_reversion", "3f_liquidity"],
    "DISCOVERED": ["hierarchical_combiner", "mean_reversion_detector"],
}


def _lifecycle(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    from signal_lifecycle import SignalLifecycle
    return SignalLifecycle(db_path=getattr(args, 'db', None))


def cmd_lifecycle_help(args=None):
    """Signal promotion lifecycle — group help."""
    print(f"""
  {BOLD}nat lifecycle{W} — signal promotion state machine
  DISCOVERED → VALIDATED → PAPER_TRADING → APPROVAL_PENDING → LIVE → MONITORING → RETIRED (+ REJECTED)

    status              count signals by state
    list [--state S]    list signals (optionally filtered)
    history <id>        transition history (with git_sha) for one signal
    approve <id>        APPROVAL_PENDING → LIVE  (sole human gate; needs --confirm)
    reject <id>         reject a pre-LIVE signal (--reason)
    seed                seed the deployable winners (idempotent)
""")


def cmd_lifecycle_status(args):
    """Count signals by lifecycle state."""
    lc = _lifecycle(args)
    try:
        rows = lc.list_signals()
    finally:
        lc.close()
    counts = {}
    for r in rows:
        counts[r['state']] = counts.get(r['state'], 0) + 1
    if _json_mode(args):
        _output({"total": len(rows), "by_state": counts}, args)
        return
    print(f"\n  {BOLD}Signal lifecycle{W} — {len(rows)} signals\n")
    if not rows:
        print("    (none — try: nat lifecycle seed)\n")
        return
    for state in _LIFECYCLE_STATES:
        if counts.get(state):
            print(f"    {state:<18} {counts[state]}")
    print()


def cmd_lifecycle_list(args):
    """List lifecycle signals, optionally filtered by --state."""
    lc = _lifecycle(args)
    try:
        rows = lc.list_signals(state=getattr(args, 'state', None))
    finally:
        lc.close()
    if _json_mode(args):
        _output({"signals": rows}, args)
        return
    if not rows:
        print("  (no signals; try: nat lifecycle seed)")
        return
    print(f"\n  {'signal_id':<28} {'state':<16} {'name':<22} updated")
    print(f"  {'-'*28} {'-'*16} {'-'*22} {'-'*20}")
    for r in rows:
        print(f"  {r['signal_id']:<28} {r['state']:<16} {(r.get('name') or ''):<22} {r.get('updated_at', '')}")
    print()


def cmd_lifecycle_history(args):
    """Show transition history (with git_sha) for one signal."""
    lc = _lifecycle(args)
    try:
        sig = lc.get_signal(args.signal_id)
        hist = lc.history(args.signal_id)
    finally:
        lc.close()
    if sig is None:
        print(f"  {R}Not found:{W} {args.signal_id} (see `nat lifecycle list`)")
        return 1
    if _json_mode(args):
        _output({"signal": sig, "history": hist}, args)
        return
    print(f"\n  {BOLD}{args.signal_id}{W}  state={sig['state']}  name={sig.get('name')}")
    for h in hist:
        frm = h['from_state'] or '·'
        print(f"    {h['at']:<28} {frm:>16} → {h['to_state']:<16} {(h.get('git_sha') or ''):<10} {h.get('msg', '')}")
    print()


def cmd_lifecycle_approve(args):
    """The sole human gate: APPROVAL_PENDING → LIVE (requires --confirm)."""
    lc = _lifecycle(args)
    try:
        sig = lc.get_signal(args.signal_id)
        if sig is None:
            print(f"  {R}Not found:{W} {args.signal_id}")
            return 1
        print(f"\n  {BOLD}APPROVE → LIVE{W}: {args.signal_id}")
        print(f"    state    = {sig['state']}")
        print(f"    name     = {sig.get('name')}")
        print(f"    metadata = {sig.get('metadata')}")
        print(f"    git_sha  = {sig.get('git_sha')}")
        # (G8 paper scorecard surfaces here once T15/T19 land)
        if not getattr(args, 'confirm', False):
            print(f"\n  {Y}Dry-run.{W} Re-run with {BOLD}--confirm{W} to promote to LIVE.\n")
            return 0
        try:
            lc.approve(args.signal_id, msg="human approval via nat lifecycle approve")
        except Exception as e:
            print(f"  {R}Cannot approve:{W} {e}")
            return 1
        print(f"  {G}✓ promoted to LIVE{W}\n")
    finally:
        lc.close()


def cmd_lifecycle_reject(args):
    """Reject a pre-LIVE signal."""
    lc = _lifecycle(args)
    try:
        if lc.get_signal(args.signal_id) is None:
            print(f"  {R}Not found:{W} {args.signal_id}")
            return 1
        try:
            lc.reject(args.signal_id, reason=getattr(args, 'reason', None) or "manual")
        except Exception as e:
            print(f"  {R}Cannot reject:{W} {e}")
            return 1
        print(f"  {G}✓ REJECTED{W} {args.signal_id}")
    finally:
        lc.close()


def cmd_lifecycle_seed(args):
    """Idempotently seed the deployable winners into the lifecycle."""
    lc = _lifecycle(args)
    seeded = 0
    try:
        existing = {s['signal_id'] for s in lc.list_signals()}
        for name in _LIFECYCLE_SEED["VALIDATED"]:
            if name in existing:
                continue
            lc.discover(name, name=name, agent="seed", msg="seed: deployable winner")
            lc.validate(name, msg="seed: oos30-validated winner")
            seeded += 1
        for name in _LIFECYCLE_SEED["DISCOVERED"]:
            if name in existing:
                continue
            lc.discover(name, name=name, agent="seed", msg="seed: candidate")
            seeded += 1
    finally:
        lc.close()
    print(f"  Seeded {seeded} new signal(s) (idempotent).")


def register(sub):
    # ── lifecycle (signal promotion state machine, T3/T4) ──
    lc_p = sub.add_parser('lifecycle',
        help='Signal promotion lifecycle (DISCOVERED→LIVE→RETIRED)')
    lc_p.add_argument('--db', default=None, help='Path to nat.db (default: data/nat.db)')
    lc_p.set_defaults(func=cmd_lifecycle_help)
    lcsub = lc_p.add_subparsers(dest='subcmd')
    lcst = lcsub.add_parser('status', help='Count signals by state')
    lcst.add_argument('--json', action='store_true', help='JSON output')
    lcst.set_defaults(func=cmd_lifecycle_status)
    lcl = lcsub.add_parser('list', help='List signals')
    lcl.add_argument('--state', default=None, help='Filter by state')
    lcl.add_argument('--json', action='store_true', help='JSON output')
    lcl.set_defaults(func=cmd_lifecycle_list)
    lch = lcsub.add_parser('history', help='Transition history for one signal')
    lch.add_argument('signal_id')
    lch.add_argument('--json', action='store_true', help='JSON output')
    lch.set_defaults(func=cmd_lifecycle_history)
    lca = lcsub.add_parser('approve', help='APPROVAL_PENDING → LIVE (human gate)')
    lca.add_argument('signal_id')
    lca.add_argument('--confirm', action='store_true', help='Actually promote (else dry-run)')
    lca.set_defaults(func=cmd_lifecycle_approve)
    lcr = lcsub.add_parser('reject', help='Reject a pre-LIVE signal')
    lcr.add_argument('signal_id')
    lcr.add_argument('--reason', default=None, help='Rejection reason')
    lcr.set_defaults(func=cmd_lifecycle_reject)
    lcs = lcsub.add_parser('seed', help='Seed deployable winners (idempotent)')
    lcs.set_defaults(func=cmd_lifecycle_seed)


__all__ = [
    "cmd_lifecycle_help", "cmd_lifecycle_status", "cmd_lifecycle_list",
    "cmd_lifecycle_history", "cmd_lifecycle_approve", "cmd_lifecycle_reject",
    "cmd_lifecycle_seed", "register",
]
