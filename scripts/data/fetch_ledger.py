"""WP-3 Part B — non-funding ledger backfill: deposits and transfers, read rather than inferred.

WP-3 needs realised P&L per wallet, and from snapshots alone that is

    Δaccount_value − Δ uPnL  =  realised P&L + flows + fees

with no way to split the terms. Measured on 20,065 real WP-2 intervals the residual has a 99th
percentile of **0.43 of account value**, and **6.6 % of intervals move account value by >2 % net
of uPnL**, across 84 of 200 wallets. The spec's original plan was to flag those as
unattributable; that files 6.6 % of the panel as unknown.

`userNonFundingLedgerUpdates` returns the flows **explicitly**, so realised P&L becomes

    realised = Δaccount_value − Δ uPnL − net_perp_flow      (net_perp_flow from here, exact)

**Classification is the risk, not fetching.** Only some delta types move the *perp* account, and
the payloads express it differently. Shapes verified live 2026-08-13 (see `_HANDLERS`). The
subtle one is `send`: it carries `sourceDex` / `destinationDex` (`'spot'` or `''` for perp), so a
**self**-addressed send from spot to perp is a genuine perp inflow — one was observed at
1,968,806 USDC. A flat per-type sign map gets that backwards.

**An unrecognised type raises rather than contributing zero.** A new venue type silently
counted as 0 would re-contaminate realised P&L — precisely the contamination this module exists
to remove — and it would do so invisibly. `net_perp_flow` catches the raise, keeps going, and
*reports* the unknown types; nothing downstream may treat "unknown" as "no flow".

Unlike WP-2's sampler this is a **backfill, not a clock**: the history stays retrievable, so a
missed day costs nothing and it needs no systemd unit.

Usage:
    python scripts/data/fetch_ledger.py --days 90                 # the rostered wallets
    python scripts/data/fetch_ledger.py --days 90 --wallet 0x...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ledger"

#: Venue response cap, measured: a 400-day ask returned the OLDEST 2000 entries.
PAGE_LIMIT = 2000
REQUEST_DELAY = 0.25
#: 90 days needs 1-2 pages for an active wallet; 40 is years of headroom.
MAX_PAGES = 40
RETRIES = 4
BACKOFF = 2.0

try:
    from data.wallet_roster import fetch_roster, is_valid_address
except ImportError:                       # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.wallet_roster import fetch_roster, is_valid_address


class UnknownDeltaType(ValueError):
    """A ledger delta type with no handler. Never silently zero — see the module docstring."""


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _same(a, b) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a.lower() == b.lower()


def _is_perp_side(dex) -> bool:
    """`sourceDex`/`destinationDex`: `''` (or absent) is the perp account; `'spot'` is not."""
    return not dex


# ── per-type handlers: signed USDC entering the PERP account ────────────────────
# Shapes verified live 2026-08-13. A handler returning 0.0 means "known to have no perp
# effect" — which is NOT the same as an unknown type, and the two must never be conflated.

def _h_deposit(d, wallet):        # {usdc}
    return _f(d.get("usdc"))


def _h_withdraw(d, wallet):       # {usdc, fee} — fee is charged separately by the venue
    return -_f(d.get("usdc"))


def _h_subaccount(d, wallet):     # {usdc, user, destination}
    amt = _f(d.get("usdc"))
    if _same(d.get("destination"), wallet):
        return +amt
    if _same(d.get("user"), wallet):
        return -amt
    return 0.0


def _h_send(d, wallet):
    """{user, destination, sourceDex, destinationDex, token, usdcValue}

    Direction is by *account side*, not by address: a self-addressed send moving USDC from the
    spot wallet to the perp wallet is a real perp inflow (observed at 1,968,806 USDC).
    """
    if (d.get("token") or "").upper() != "USDC":
        return 0.0                                    # non-USDC does not move perp collateral
    amt = _f(d.get("usdcValue"))
    src_is_me = _same(d.get("user"), wallet)
    dst_is_me = _same(d.get("destination"), wallet)
    out = src_is_me and _is_perp_side(d.get("sourceDex"))
    inn = dst_is_me and _is_perp_side(d.get("destinationDex"))
    if out and inn:                                   # perp -> perp, same account: no net change
        return 0.0
    if inn:
        return +amt
    if out:
        return -amt
    return 0.0                                        # neither leg touches our perp account


def _h_account_class(d, wallet):
    """{toPerp, usdc} — the spot<->perp collateral transfer.

    **The most common flow type by an order of magnitude** (1,574 of the first universe
    backfill's entries, against 108 for the next). It was unknown on the first pass; a design
    that zeroed unknowns silently would have dropped the dominant flow and corrupted realised
    P&L for most wallets, invisibly. That is the case the flag-never-zero rule exists for.
    """
    amt = _f(d.get("usdc"))
    return +amt if d.get("toPerp") else -amt


def _h_internal(d, wallet):       # {user, destination, usdc, fee}
    amt = _f(d.get("usdc"))
    if _same(d.get("destination"), wallet):
        return +amt
    if _same(d.get("user"), wallet):
        return -amt
    return 0.0


def _h_liquidation(d, wallet):
    """{accountValue, leverageType, liquidatedNtlPos, liquidatedPositions}

    Zero is **correct here, not a placeholder**: a liquidation is a trading event, not a
    transfer. Its effect on account value is realised loss, which realised P&L must capture —
    subtracting it as a "flow" would erase exactly the loss the cohort ranking should see.
    """
    return 0.0


def _h_zero(d, wallet):
    """Known types with no perp-USDC effect: spot-side moves and HYPE-denominated entries."""
    return 0.0


_HANDLERS = {
    "deposit": _h_deposit,
    "withdraw": _h_withdraw,
    "subAccountTransfer": _h_subaccount,
    "send": _h_send,
    "accountClassTransfer": _h_account_class,
    "internalTransfer": _h_internal,
    "liquidation": _h_liquidation,          # zero is correct: a trade, not a transfer
    # known-zero (verified live): spot-side transfers and non-USDC token events
    "spotTransfer": _h_zero,
    "spotGenesis": _h_zero,
    "cStakingTransfer": _h_zero,
    "gossipPriorityGasAuction": _h_zero,
    "vaultDeposit": _h_zero,
    "vaultWithdraw": _h_zero,
}

#: Types seen live whose PERP effect is **not established**, deliberately left without a handler
#: so `net_perp_flow` flags them. Each needs the same treatment the handled types got: reconcile
#: the entry against the matching `account_value` step in the WP-2 panel and derive the sign.
#: That reconciliation is not yet possible — the ledger reaches back 90 days and the position
#: panel is 4 days old — so guessing now would put an unverified sign into realised P&L, which is
#: the contamination this module exists to remove.
UNRESOLVED_TYPES = ("rewardsClaim", "borrowLend", "accountActivationGas")


def is_known_type(delta_type: str) -> bool:
    """True if a handler exists. Known-zero is known; absence is not."""
    return delta_type in _HANDLERS


def perp_flow_usdc(entry: dict, wallet: str) -> float:
    """Signed USDC entering `wallet`'s PERP account from one ledger entry.

    Raises `UnknownDeltaType` for a type with no handler rather than returning 0.0.
    """
    delta = (entry or {}).get("delta") or {}
    dtype = delta.get("type")
    handler = _HANDLERS.get(dtype)
    if handler is None:
        raise UnknownDeltaType(f"no perp-flow handler for delta type {dtype!r}")
    return float(handler(delta, wallet))


def net_perp_flow(entries, wallet: str, t0_ms: int, t1_ms: int) -> tuple[float, dict]:
    """Net signed USDC into the perp account over `(t0_ms, t1_ms]`, plus a report.

    Unknown types are counted and named in the report rather than silently dropped: a caller
    that sees `unknown_count > 0` must treat the realised-P&L figure for that window as
    contaminated, not merely approximate.
    """
    net = 0.0
    unknown: dict[str, int] = {}
    n_used = 0
    for e in entries or []:
        t = int(e.get("time", 0))
        if not (t0_ms < t <= t1_ms):
            continue
        try:
            net += perp_flow_usdc(e, wallet)
            n_used += 1
        except UnknownDeltaType:
            dt = ((e.get("delta") or {}).get("type"))
            unknown[dt] = unknown.get(dt, 0) + 1
    return net, {"n_entries": n_used,
                 "unknown_types": unknown,
                 "unknown_count": sum(unknown.values())}


# ── fetch ────────────────────────────────────────────────────────────────────────

def _fetch_page(wallet: str, start_ms: int, end_ms: int) -> list[dict]:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"type": "userNonFundingLedgerUpdates", "user": wallet,
                         "startTime": int(start_ms), "endTime": int(end_ms)}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _with_backoff(fetch, wallet, start, end, retries=RETRIES, backoff=BACKOFF):
    """Transport faults retry; schema faults do not — the `fetch_universe` rule."""
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return fetch(wallet, start, end)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 >= max(1, retries):
                raise
            time.sleep(backoff * (2 ** attempt))
    if last is not None:                  # pragma: no cover - defensive
        raise last


def fetch_ledger_history(wallet: str, start_ms: int, end_ms: int, fetch_fn=None,
                         delay: float = REQUEST_DELAY, max_pages: int = MAX_PAGES,
                         retries: int = RETRIES) -> list[dict]:
    """Page the ledger until the window is covered. Time-ordered, deduped on (time, hash)."""
    fetch = fetch_fn or _fetch_page
    seen: dict[tuple, dict] = {}
    cursor = int(start_ms)

    for _ in range(max_pages):
        if cursor > end_ms:
            break
        rows = _with_backoff(fetch, wallet, cursor, end_ms, retries=retries)
        if not rows:
            break
        newest, fresh = cursor, 0
        for r in rows:
            t = int(r["time"])
            newest = max(newest, t)
            key = (t, r.get("hash"))
            if key not in seen:
                seen[key] = r
                fresh += 1
        if fresh == 0 or newest < cursor:      # a stuck venue must not spin
            break
        cursor = newest + 1
        if len(rows) < PAGE_LIMIT:
            break
        if delay:
            time.sleep(delay)

    return [seen[k] for k in sorted(seen, key=lambda k: (k[0], str(k[1])))]


def write_wallet(entries, out_dir: Path, wallet: str) -> Path | None:
    """One parquet per wallet, append-and-dedupe on (time, hash).

    The address is rendered into a path, so it is validated first (the XS-1 lesson).
    """
    if not is_valid_address(wallet):
        raise ValueError(f"refusing to use {wallet!r} as a filename")
    if not entries:
        return None
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{wallet}.parquet"

    rows = []
    for e in entries:
        d = e.get("delta") or {}
        dtype = d.get("type")
        try:
            flow = perp_flow_usdc(e, wallet)
            known = True
        except UnknownDeltaType:
            flow, known = float("nan"), False      # NaN, never 0 — see the module docstring
        rows.append({"time": int(e["time"]), "hash": e.get("hash"), "type": dtype,
                     "perp_flow_usdc": flow, "known_type": known,
                     "raw": json.dumps(d, sort_keys=True)})

    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = (df.drop_duplicates(subset=["time", "hash"], keep="last")
            .sort_values("time").reset_index(drop=True))
    df.to_parquet(path, index=False)
    return path


def backfill(wallets, start_ms: int, end_ms: int, out_dir: Path | None = None,
             fetch_fn=None, delay: float = REQUEST_DELAY,
             retries: int = RETRIES) -> tuple[list[dict], dict]:
    """Backfill every wallet. One failure is recorded, never fatal.

    `ok + failed + rejected == n_requested` must close — the WP-2 rule: a wallet that could not
    be fetched is not the same as a wallet with no flows.
    """
    requested = list(wallets)
    all_rows: list[dict] = []
    report = {"n_requested": len(requested), "ok": 0, "failed": 0, "rejected": 0,
              "per_wallet": {}, "unknown_types": {}, "errors": []}

    for w in requested:
        if not is_valid_address(w):
            report["rejected"] += 1
            continue
        try:
            entries = fetch_ledger_history(w, start_ms, end_ms, fetch_fn=fetch_fn,
                                           delay=delay, retries=retries)
        except Exception as exc:  # noqa: BLE001 — recorded, not fatal
            log.warning("%s: %s", w, exc)
            report["failed"] += 1
            report["errors"].append({"wallet": w, "error": f"{type(exc).__name__}: {exc}"})
            continue

        for e in entries:
            dt = (e.get("delta") or {}).get("type")
            if not is_known_type(dt):
                report["unknown_types"][dt] = report["unknown_types"].get(dt, 0) + 1
        all_rows.extend(entries)
        report["ok"] += 1
        report["per_wallet"][w] = len(entries)
        if out_dir is not None:
            write_wallet(entries, out_dir, w)

    return all_rows, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill non-funding ledger flows (WP-3)")
    ap.add_argument("--wallet", nargs="+", default=None,
                    help="Explicit wallets (default: the derived WP-1 roster)")
    ap.add_argument("--days", type=float, default=90.0)
    ap.add_argument("--max-wallets", type=int, default=200)
    ap.add_argument("--min-notional", type=float, default=1e6)
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    wallets = args.wallet or [r.address for r in fetch_roster(
        limit=args.max_wallets, min_notional_seen=args.min_notional,
        rank_by="notional_seen")]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(args.days * 86_400_000)
    print(f"backfilling {len(wallets)} wallets x {args.days:.0f}d", flush=True)

    started = time.time()
    _, rep = backfill(wallets, start_ms, end_ms, out_dir=out_dir, delay=args.delay)
    counts = sorted(rep["per_wallet"].values())
    median = counts[len(counts) // 2] if counts else 0
    print(f"ok={rep['ok']} failed={rep['failed']} rejected={rep['rejected']} "
          f"median_entries={median} ({time.time() - started:.0f}s) -> {out_dir}", flush=True)
    if rep["unknown_types"]:
        # Loud on purpose: an unhandled type means realised P&L is contaminated for any window
        # containing it, and the fix is a handler, not a shrug.
        print(f"  ⚠ UNKNOWN delta types (need handlers): {rep['unknown_types']}", flush=True)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
