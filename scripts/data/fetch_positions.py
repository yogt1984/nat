"""WP-2 — position snapshot collector: the clock that nothing else can start.

Step 2 of `docs/specs/wallet_positioning.md`, and tier-0 item #1 of `docs/FINAL_PLAN.md`.
One sweep = one `clearinghouseState` call per rostered wallet → one parquet under
`data/positions/YYYY-MM-DD/`.

**Why this runs before anything else.** WP-5 needs **≥90 days** of position history, so the
earliest possible verdict is 2026-11-08 and it slips one-for-one with every day of delay. This
is the same shape as the XS-7 retention cap (FINDINGS §7.1): when the binding constraint is
*accrual*, delay is irreversible and no amount of later effort buys the days back. Every day not
collected is permanently lost.

**Why the family is worth the wait** (`research/MECHANISM_FAMILIES.md`, family 5): on a fully
on-chain perp DEX, positions, entry prices, leverage and therefore **liquidation prices** are
computable in advance, where a CEX makes them invisible. H3 (liquidation cascade) was *confirmed*
in the hypothesis suite and has never been tested on live data, because the features are K2 dead
columns — `rust/ing/src/state/mod.rs:127` initialises `position_state` to `None`.

**The design problem is partial failure, not fetching.** A sweep over hundreds of wallets will
always have some unreachable, and the naive handling silently corrupts the panel:

> **A wallet that fails is written, not dropped.** XS-8 appends failures to a report and
> `continue`s, so `aggregate_l2` cannot tell "frequently unreachable" from "less history". Here
> the confusion would be worse: a wallet that stops responding would be indistinguishable from a
> wallet that **closed its positions** — which is the exact event the family exists to detect.
> So an unreachable wallet gets a row with `status=failed`, and the arithmetic
> `ok + empty + failed + rejected == n_requested` is asserted in the suite.

**`liquidation_price` is nullable and must stay that way.** The first position sampled live
(2026-08-10) carried `liquidationPx: null` — a cross-margin position with no isolated liquidation
level. Coerced to 0.0 it reads as "liquidates at zero", and every downstream cascade-distance
calculation silently becomes distance-to-zero. Null is preserved end to end, through the parquet.

**Columns beyond the spec's minimum are deliberate.** Storage is bytes; a column not collected
today cannot be backfilled at any price. `cumFunding` in particular is free here and is what
tier-0 item #2 (`COST-9` → `LF8`, funding carry as a *held* position) will need.

Stdlib + pandas + pyarrow only, like `fetch_l2.py` and `fetch_candles.py`.

Usage:
    python scripts/data/fetch_positions.py --once --max-wallets 5
    python scripts/data/fetch_positions.py --loop --every 900     # 15-min sweeps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "positions"

#: Positions move on hours. 15 min is ample; 5 min is wasteful against a rate-limited endpoint.
DEFAULT_EVERY_S = 900.0
#: Between wallets in a sweep — the `--symbol-delay` equivalent from XS-8.
WALLET_DELAY = 0.15
#: Roster size per sweep. Ranked by traded notional, so this is the top of the venue.
DEFAULT_MAX_WALLETS = 200

try:
    from data.wallet_roster import fetch_roster, is_valid_address
except ImportError:                          # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.wallet_roster import fetch_roster, is_valid_address


class PositionStatus(str, Enum):
    OK = "ok"           # wallet reachable, at least one open position
    EMPTY = "empty"     # wallet reachable, flat — NOT the same as unreachable
    FAILED = "failed"   # wallet unreachable; written anyway, with the error


#: The row contract. Asserted per row in the suite so schema drift cannot pass silently.
POSITION_COLUMNS = [
    "ts_ms", "wallet", "coin", "size", "entry_price", "position_value", "unrealized_pnl",
    "liquidation_price", "leverage", "leverage_type", "max_leverage", "margin_used",
    "return_on_equity", "cum_funding_since_open", "cum_funding_all_time",
    # `total_raw_usd` — the venue's `marginSummary.totalRawUsd`, added 2026-08-13 for WP-3.
    # Snapshot-only and unbackfillable, so days 1-4 lack it and every reader must tolerate its
    # absence (spec §B3).
    #
    # **Its semantics are NOT established, and the obvious identity is false.** It was added on
    # the assumption that `account_value - total_raw_usd ~= sum(unrealized_pnl)`, giving an
    # independent consistency check. Measured live 2026-08-13 across four rostered accounts,
    # that is wrong: a flat account has `av - raw == 0`, but an 18-position account gave
    # `av - raw = -437,517` against `sum uPnL = -377`. Whatever the field decomposes into, it is
    # not cash-plus-uPnL. It is stored because it is one float, free, and unrecoverable later --
    # NOT because anything downstream may assume a relation to the other columns. Establish the
    # semantics before using it for anything.
    "account_value", "total_raw_usd", "total_ntl_pos", "total_margin_used", "withdrawable",
    "cross_maintenance_margin_used", "venue_time_ms", "status", "error",
]


def _f(v):
    """Float, or None. Never 0.0-as-a-stand-in — see the liquidation_price note above."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _blank_row(wallet: str, ts_ms: int, status: PositionStatus, error=None) -> dict:
    row = {c: None for c in POSITION_COLUMNS}
    row.update(ts_ms=ts_ms, wallet=wallet, status=status, error=error)
    return row


def parse_clearinghouse_state(payload, wallet: str, ts_ms: int | None = None) -> list[dict]:
    """One `clearinghouseState` response → one row per open position.

    Venue shape, verified live 2026-08-10::

        {"marginSummary": {"accountValue", "totalNtlPos", "totalRawUsd", "totalMarginUsed"},
         "crossMaintenanceMarginUsed", "withdrawable", "time",
         "assetPositions": [{"type": "oneWay",
                             "position": {"coin", "szi", "leverage": {"type", "value"},
                                          "entryPx", "positionValue", "unrealizedPnl",
                                          "returnOnEquity", "liquidationPx", "marginUsed",
                                          "maxLeverage", "cumFunding": {...}}}]}

    A flat account returns exactly one row with `status=empty`, so presence in the panel never
    depends on holding a position. Raises on a malformed payload rather than returning a short
    result — a silently truncated sweep narrows every downstream cohort claim invisibly.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"clearinghouseState must be an object, got {type(payload).__name__}")
    positions = payload.get("assetPositions")
    if positions is None:
        raise ValueError("clearinghouseState payload has no 'assetPositions'")
    if not isinstance(positions, list):
        raise TypeError(f"'assetPositions' must be a list, got {type(positions).__name__}")

    ts_ms = int(ts_ms if ts_ms is not None else time.time() * 1000)
    margin = payload.get("marginSummary") or {}
    account = {
        "account_value": _f(margin.get("accountValue")),
        "total_raw_usd": _f(margin.get("totalRawUsd")),
        "total_ntl_pos": _f(margin.get("totalNtlPos")),
        "total_margin_used": _f(margin.get("totalMarginUsed")),
        "withdrawable": _f(payload.get("withdrawable")),
        "cross_maintenance_margin_used": _f(payload.get("crossMaintenanceMarginUsed")),
        "venue_time_ms": int(payload["time"]) if payload.get("time") is not None else None,
    }

    if not positions:
        row = _blank_row(wallet, ts_ms, PositionStatus.EMPTY)
        row.update(account)
        return [row]

    rows: list[dict] = []
    for entry in positions:
        pos = (entry or {}).get("position") if isinstance(entry, dict) else None
        if not isinstance(pos, dict):
            raise TypeError(f"assetPositions entry has no 'position' object: {str(entry)[:60]!r}")
        lev = pos.get("leverage") or {}
        funding = pos.get("cumFunding") or {}
        row = _blank_row(wallet, ts_ms, PositionStatus.OK)
        row.update(account)
        row.update(
            coin=pos.get("coin"),
            # Signed: positive long, negative short. The sign IS the hypothesis.
            size=_f(pos.get("szi")),
            entry_price=_f(pos.get("entryPx")),
            position_value=_f(pos.get("positionValue")),
            unrealized_pnl=_f(pos.get("unrealizedPnl")),
            # Nullable by design — null means "no isolated liquidation level", not zero.
            liquidation_price=_f(pos.get("liquidationPx")),
            leverage=_f(lev.get("value")),
            leverage_type=lev.get("type"),
            max_leverage=_f(pos.get("maxLeverage")),
            margin_used=_f(pos.get("marginUsed")),
            return_on_equity=_f(pos.get("returnOnEquity")),
            cum_funding_since_open=_f(funding.get("sinceOpen")),
            cum_funding_all_time=_f(funding.get("allTime")),
        )
        rows.append(row)
    return rows


def _fetch_state(wallet: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"type": "clearinghouseState", "user": wallet}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def sweep_positions(wallets, fetch_fn=None, delay: float = WALLET_DELAY,
                    max_wallets: int | None = None) -> tuple[list[dict], dict]:
    """Sample every wallet once. Returns (rows, report).

    Failure is data: an unreachable wallet is written with `status=failed` and its error, and
    the sweep continues. The report's counts must satisfy
    `ok + empty + failed + rejected == n_requested` — asserted in the suite, because a sweep
    that loses a wallet silently is the XS-1 failure mode.
    """
    fetch = fetch_fn or _fetch_state
    supplied = list(wallets)
    requested = supplied[:max_wallets] if max_wallets else supplied

    rows: list[dict] = []
    report = {"n_supplied": len(supplied), "n_requested": len(requested),
              "truncated": len(supplied) - len(requested),
              "ok": 0, "empty": 0, "failed": 0, "rejected": 0,
              "n_positions": 0, "errors": []}

    fetched = 0
    for wallet in requested:
        if not is_valid_address(wallet):
            # Rejected before it can reach a URL or a filesystem path (the XS-1 lesson).
            report["rejected"] += 1
            continue
        if fetched and delay:
            time.sleep(delay)
        fetched += 1
        ts_ms = int(time.time() * 1000)
        try:
            wallet_rows = parse_clearinghouse_state(fetch(wallet), wallet, ts_ms=ts_ms)
        except Exception as exc:  # noqa: BLE001 — a failure is a row, not a crash
            log.warning("%s: %s", wallet, exc)
            err = f"{type(exc).__name__}: {exc}"
            rows.append(_blank_row(wallet, ts_ms, PositionStatus.FAILED, error=err))
            report["failed"] += 1
            report["errors"].append({"wallet": wallet, "error": err})
            continue

        rows.extend(wallet_rows)
        if wallet_rows[0]["status"] is PositionStatus.EMPTY:
            report["empty"] += 1
        else:
            report["ok"] += 1
            report["n_positions"] += len(wallet_rows)

    return rows, report


def write_snapshot(rows: list[dict], out_dir: Path, ts_ms: int | None = None) -> Path | None:
    """Append-only: one parquet per sweep under a UTC day directory.

    Never rewrites an existing file — the product is the accumulated panel, so an overwrite
    would leave it permanently at n=1, and per-sweep files mean an interrupted run loses one
    sweep rather than the archive. Mirrors `data/features/` and `data/l2/`.
    """
    if not rows:
        return None
    import pandas as pd

    ts_ms = ts_ms if ts_ms is not None else int(time.time() * 1000)
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    day_dir = Path(out_dir) / dt.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"positions_{dt.strftime('%H%M%S')}.parquet"

    df = pd.DataFrame(rows, columns=POSITION_COLUMNS)
    df["status"] = df["status"].map(lambda s: s.value if isinstance(s, Enum) else s)
    df.to_parquet(path, index=False)
    return path


def run_sweep(wallets, out_dir: Path, delay: float = WALLET_DELAY,
              max_wallets: int | None = None) -> dict:
    started = time.time()
    rows, report = sweep_positions(wallets, delay=delay, max_wallets=max_wallets)
    path = write_snapshot(rows, out_dir)
    report["path"] = str(path) if path else None
    report["elapsed_s"] = round(time.time() - started, 1)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sample clearinghouseState across the wallet roster (WP-2)")
    ap.add_argument("--wallet", nargs="+", default=None,
                    help="Explicit wallets (default: the derived WP-1 roster)")
    ap.add_argument("--once", action="store_true", help="One sweep, then exit")
    ap.add_argument("--loop", action="store_true", help="Sample repeatedly")
    ap.add_argument("--every", type=float, default=DEFAULT_EVERY_S,
                    help=f"Seconds between sweeps in --loop (default: {DEFAULT_EVERY_S:.0f})")
    ap.add_argument("--wallet-delay", type=float, default=WALLET_DELAY)
    ap.add_argument("--max-wallets", type=int, default=DEFAULT_MAX_WALLETS)
    ap.add_argument("--min-notional", type=float, default=1e6,
                    help="Volume floor for the roster — excludes vaults/bridge (WP-1)")
    ap.add_argument("--refresh-roster-every", type=int, default=96,
                    help="Re-derive the roster every N sweeps (~24 h at 15 min)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    def _roster():
        refs = fetch_roster(limit=args.max_wallets, min_notional_seen=args.min_notional,
                            rank_by="notional_seen")
        return [r.address for r in refs]

    wallets = args.wallet or _roster()
    if not args.wallet:
        print(f"roster: {len(wallets)} wallets", flush=True)

    n = 0
    while True:
        n += 1
        # The roster is derived, and cohort turnover is itself under test (spec §3 failure
        # mode 1) — so it is re-derived periodically rather than pinned at startup. A refresh
        # failure keeps the previous roster: losing a sweep to a transient 429 costs a day of
        # accrual that cannot be recovered.
        if not args.wallet and args.refresh_roster_every and n > 1 \
                and n % args.refresh_roster_every == 1:
            try:
                wallets = _roster()
                log.info("roster refreshed: %d wallets", len(wallets))
            except Exception as exc:
                log.warning("roster refresh failed, keeping previous: %s", exc)

        rep = run_sweep(wallets, out_dir, delay=args.wallet_delay,
                        max_wallets=args.max_wallets)
        # flush=True: this runs for months under systemd with stdout redirected, where block
        # buffering would hold sweep lines for hours.
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
              f"ok={rep['ok']} empty={rep['empty']} failed={rep['failed']} "
              f"rejected={rep['rejected']} positions={rep['n_positions']} "
              f"({rep['elapsed_s']}s) -> {rep['path']}", flush=True)

        if args.once or not args.loop:
            return 1 if rep["failed"] and rep["ok"] == 0 else 0
        time.sleep(max(0.0, args.every - rep["elapsed_s"]))


if __name__ == "__main__":
    sys.exit(main())
