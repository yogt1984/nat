"""XS-8 — L2 order-book sampler for the whole Hyperliquid perp universe.

**Why this exists.** `FINDINGS.md` §4.11 measured the breakeven maker rate at BTC's touch:
`E[adverse|fill] - half_spread` = **+0.144 bps (bid) / +0.159 (ask)**. Below that rate a
resting quote loses money no matter how it is gated, and the best *attainable* volume tier
is zero fees — still ~0.08 bps under water. The conclusion was not "the maker line is
dead" but "every maker experiment ever run used the three tightest symbols on the venue",
i.e. the question is unresolved because the sample was biased.

Resolving it needs half-spreads for the other ~174 perps, and nothing in hand has them:
candles carry no spread and no depth, and the ingestor's book feed follows
`config/symbols.toml` (BTC/ETH/SOL). This module samples `l2Book` over REST across the
universe so `B-5a` can rank pairs by measured half-spread and apply §4.11's relation
directly — an arithmetic screen that can kill the hypothesis without a single simulation,
the way §4.10 killed the fee-tier hypothesis.

**A snapshot is not a measurement.** Half-spread moves all day; one book is an n=1
estimate. PROC-20 has just finished demonstrating what n=4-31 per cell does to a
conclusion (FINDINGS §5). So this samples on a schedule and *accumulates* — snapshots are
written append-only, one file per sweep under a UTC day directory, and the object of study
is the resulting distribution, never a single row.

Degenerate books (crossed, locked, one-sided) are recorded with a status and **no
spread**: on illiquid pairs they are common, and a zero or negative spread silently reads
as free money. One symbol's failure never ends the sweep (the XS-1 lesson).

Stdlib + pandas + pyarrow only, like `fetch_candles.py`, so it runs anywhere.

Usage:
    python scripts/data/fetch_l2.py --once                    # one sweep, then exit
    python scripts/data/fetch_l2.py --loop --every 300        # sample every 5 min
    python scripts/data/fetch_l2.py --once --symbol BTC ETH   # explicit symbols
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
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "l2"
SYMBOL_DELAY = 0.15          # seconds between symbols in a sweep
DEPTH_LEVELS = 5             # levels summed for the depth columns

try:                          # reuse the universe enumeration + ticker guard from XS-1
    from data.fetch_candles import SYMBOL_RE, fetch_universe
except ImportError:           # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.fetch_candles import SYMBOL_RE, fetch_universe


class BookStatus(str, Enum):
    OK = "ok"                # two-sided, ask > bid
    CROSSED = "crossed"      # bid > ask
    LOCKED = "locked"        # bid == ask
    EMPTY = "empty"          # a side has no levels
    INVALID = "invalid"      # non-positive prices


def _fetch_book(symbol: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"type": "l2Book", "coin": symbol}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def parse_l2_book(payload: dict) -> dict:
    """Reduce one `l2Book` response to the row B-5a and XS-5 consume.

    Venue shape: `{"coin", "time", "levels": [bids, asks]}`, each level
    `{"px": str, "sz": str, "n": int}`, bids descending and asks ascending.

    Raises on a malformed payload rather than returning zeros — a spread of 0.0 is
    indistinguishable from a very tight book and would poison the ranking B-5a builds.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"l2Book payload must be an object, got {type(payload).__name__}")
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) != 2:
        raise ValueError("l2Book payload has no two-sided 'levels'")

    bids, asks = levels[0], levels[1]
    symbol = payload.get("coin")
    ts_ms = int(payload.get("time", 0))

    row: dict = {
        "symbol": symbol,
        "ts_ms": ts_ms,
        "n_bid_levels": len(bids or []),
        "n_ask_levels": len(asks or []),
        "best_bid": None, "best_ask": None, "mid": None,
        "spread": None, "spread_bps": None, "half_spread_bps": None,
        "bid_sz_l1": None, "ask_sz_l1": None,
        "bid_n_l1": None, "ask_n_l1": None,
        "bid_notional_l1": None, "ask_notional_l1": None,
        "bid_notional_5": None, "ask_notional_5": None,
        "status": BookStatus.EMPTY,
    }

    if not bids or not asks:
        return row

    bb, ba = float(bids[0]["px"]), float(asks[0]["px"])
    if bb <= 0 or ba <= 0:
        row["status"] = BookStatus.INVALID
        return row

    row["best_bid"], row["best_ask"] = bb, ba
    row["bid_sz_l1"] = float(bids[0]["sz"])
    row["ask_sz_l1"] = float(asks[0]["sz"])
    row["bid_n_l1"] = int(bids[0].get("n", 0))
    row["ask_n_l1"] = int(asks[0].get("n", 0))
    row["bid_notional_l1"] = bb * row["bid_sz_l1"]
    row["ask_notional_l1"] = ba * row["ask_sz_l1"]
    row["bid_notional_5"] = sum(float(l["px"]) * float(l["sz"]) for l in bids[:DEPTH_LEVELS])
    row["ask_notional_5"] = sum(float(l["px"]) * float(l["sz"]) for l in asks[:DEPTH_LEVELS])

    # Degenerate books get a status and NO spread. A crossed book is not a negative
    # spread and a locked book is not a free half-spread; both would corrupt a ranking
    # built on the median of these values.
    if bb > ba:
        row["status"] = BookStatus.CROSSED
        return row
    if bb == ba:
        row["status"] = BookStatus.LOCKED
        return row

    mid = (bb + ba) / 2.0
    spread = ba - bb
    row["mid"] = mid
    row["spread"] = spread
    row["spread_bps"] = spread / mid * 1e4
    row["half_spread_bps"] = spread / 2.0 / mid * 1e4
    row["status"] = BookStatus.OK
    return row


def sample_universe(symbols, fetch_fn=None, delay: float = SYMBOL_DELAY) -> tuple[list[dict], dict]:
    """Sample every symbol once. Returns (rows, report).

    A failure is recorded and the sweep continues: dying at pair 12 of 177 would bias
    the sample toward whatever sorts early, which is precisely the bias this unit exists
    to remove.
    """
    fetch = fetch_fn or _fetch_book
    requested = list(symbols)
    rows: list[dict] = []
    report = {"n_requested": len(requested), "ok": 0, "degenerate": [],
              "failed": [], "rejected": []}

    for i, symbol in enumerate(requested):
        if not (isinstance(symbol, str) and SYMBOL_RE.match(symbol)):
            report["rejected"].append(symbol)
            continue
        if i and delay:
            time.sleep(delay)
        try:
            row = parse_l2_book(fetch(symbol))
        except Exception as exc:  # noqa: BLE001 - a failure is data, not a crash
            log.warning("%s: %s", symbol, exc)
            report["failed"].append({"symbol": symbol,
                                     "error": f"{type(exc).__name__}: {exc}"})
            continue
        row["symbol"] = symbol          # trust the request, not the echo
        rows.append(row)
        if row["status"] is BookStatus.OK:
            report["ok"] += 1
        else:
            report["degenerate"].append(symbol)

    return rows, report


def write_snapshot(rows: list[dict], out_dir: Path, ts_ms: int | None = None) -> Path | None:
    """Append-only: one parquet per sweep under a UTC day directory.

    Never rewrites an existing file. The unit's value is the accumulated distribution,
    so an overwrite would leave it permanently at n=1; and per-sweep files mean an
    interrupted run loses one sweep, not the archive. Mirrors the `data/features/`
    date-directory convention.
    """
    if not rows:
        return None
    import pandas as pd

    ts_ms = ts_ms if ts_ms is not None else int(time.time() * 1000)
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    day_dir = Path(out_dir) / dt.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"l2_{dt.strftime('%H%M%S')}.parquet"

    df = pd.DataFrame(rows)
    if "status" in df.columns:
        df["status"] = df["status"].map(lambda s: s.value if isinstance(s, Enum) else s)
    df.to_parquet(path, index=False)
    return path


def run_sweep(symbols, out_dir: Path, delay: float = SYMBOL_DELAY) -> dict:
    started = time.time()
    rows, report = sample_universe(symbols, delay=delay)
    path = write_snapshot(rows, out_dir)
    report["path"] = str(path) if path else None
    report["elapsed_s"] = round(time.time() - started, 1)

    ok_rows = [r for r in rows if r["status"] is BookStatus.OK]
    if ok_rows:
        hs = sorted(r["half_spread_bps"] for r in ok_rows)
        report["median_half_spread_bps"] = round(hs[len(hs) // 2], 4)
        report["min_half_spread_bps"] = round(hs[0], 4)
        report["max_half_spread_bps"] = round(hs[-1], 4)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample l2Book across the perp universe (XS-8)")
    ap.add_argument("--symbol", nargs="+", default=None,
                    help="Explicit symbols (default: the full venue universe)")
    ap.add_argument("--once", action="store_true", help="One sweep, then exit")
    ap.add_argument("--loop", action="store_true", help="Sample repeatedly")
    ap.add_argument("--every", type=float, default=300.0,
                    help="Seconds between sweeps in --loop (default: 300)")
    ap.add_argument("--symbol-delay", type=float, default=SYMBOL_DELAY)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--refresh-universe-every", type=int, default=48,
                    help="Re-enumerate the universe every N sweeps (listings change)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR

    symbols = args.symbol or fetch_universe()
    if not args.symbol:
        print(f"universe: {len(symbols)} listed perps", flush=True)

    n = 0
    while True:
        n += 1
        if not args.symbol and args.refresh_universe_every and n > 1 \
                and n % args.refresh_universe_every == 1:
            try:
                symbols = fetch_universe()
                log.info("universe refreshed: %d perps", len(symbols))
            except Exception as exc:      # keep sampling on the old roster
                log.warning("universe refresh failed, keeping previous roster: %s", exc)

        rep = run_sweep(symbols, out_dir, delay=args.symbol_delay)
        # flush=True: this runs for days under systemd with stdout redirected, where
        # Python's block buffering would hold sweep lines for hours. A daemon whose log
        # only appears after it dies cannot be monitored.
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
              f"ok={rep['ok']}/{rep['n_requested']} "
              f"degenerate={len(rep['degenerate'])} failed={len(rep['failed'])} "
              f"median_half_spread={rep.get('median_half_spread_bps')}bps "
              f"({rep['elapsed_s']}s) -> {rep['path']}", flush=True)

        if args.once or not args.loop:
            return 1 if rep["failed"] and rep["ok"] == 0 else 0
        sleep_s = max(0.0, args.every - rep["elapsed_s"])
        time.sleep(sleep_s)


if __name__ == "__main__":
    sys.exit(main())
