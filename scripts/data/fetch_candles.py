"""
Historical OHLCV candle fetcher — Hyperliquid candleSnapshot API.

Paginates through months of 1-minute candles and stores as parquet.
Supports incremental updates: only fetches candles newer than the last
stored timestamp.

No external dependencies beyond stdlib + pandas + pyarrow.

Usage:
    python scripts/data/fetch_candles.py --symbol BTC --interval 1m --days 90
    python scripts/data/fetch_candles.py --symbol BTC ETH SOL --days 180
    python scripts/data/fetch_candles.py --symbol BTC --start 2026-01-01
    python scripts/data/fetch_candles.py --universe --interval 1m --days 90   # XS-1

XS-1 (universe backfill) adds `fetch_universe` + `backfill_universe`: enumerate every
perp from the venue's `meta` endpoint and fetch them all. The roster is never hardcoded
— a constant list rots on the next listing, and breadth that does not track the actual
universe is not breadth. Symbol names are rendered into file paths, so anything that is
not a plain ticker is rejected before it can reach the filesystem; one symbol's failure
is recorded and the run continues (aborting 40 minutes in on an HTTP 500 loses the other
149); and truncation via `--max-symbols` is reported rather than silent, because a
partial sweep that reads as complete is how a coverage claim becomes false.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "candles"
MAX_CANDLES_PER_REQUEST = 5000
RATE_LIMIT_SLEEP = 0.25  # seconds between requests
SYMBOL_DELAY = 0.5       # seconds between SYMBOLS in a universe sweep

#: A ticker we are willing to turn into a filename. Hyperliquid uses forms like BTC,
#: kPEPE, @107 — letters, digits, @ and _ only, never a separator or a dot component.
SYMBOL_RE = re.compile(r"^[A-Za-z0-9@_]{1,32}$")

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _fetch_batch(
    symbol: str, interval: str, start_ms: int, end_ms: int,
) -> list[dict]:
    """Fetch one batch of candles from Hyperliquid."""
    payload = json.dumps({
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    if not data or not isinstance(data, list):
        return []
    return data


def _parse_candles(raw: list[dict]) -> pd.DataFrame:
    """Convert raw API response to DataFrame."""
    rows = []
    for c in raw:
        rows.append({
            "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })
    return pd.DataFrame(rows)


def fetch_candles(
    symbol: str,
    interval: str = "1m",
    start: str | None = None,
    days: int = 90,
    output_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles with pagination and incremental updates.

    Args:
        symbol: e.g. "BTC", "ETH", "SOL"
        interval: candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
        start: ISO date to start from (overrides days)
        days: how many days back (ignored if start is set)
        output_dir: directory for parquet output

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}. Use: {list(INTERVAL_MS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_{interval}.parquet"
    interval_ms = INTERVAL_MS[interval]

    # Determine time range
    end_ms = int(time.time() * 1000)

    if start is not None:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
    else:
        start_ms = end_ms - days * 86_400_000

    # Incremental: if file exists and covers the requested start, resume from end
    existing_df = None
    if out_path.exists():
        existing_df = pd.read_parquet(out_path)
        if len(existing_df) > 0:
            first_ts = existing_df["timestamp"].min()
            last_ts = existing_df["timestamp"].max()
            first_ms = int(first_ts.timestamp() * 1000)
            last_ms = int(last_ts.timestamp() * 1000)
            if first_ms <= start_ms and last_ms > start_ms:
                # Existing data covers requested start; only fetch new candles
                start_ms = last_ms + interval_ms
                log.info("Incremental update from %s", last_ts)
            else:
                # Existing data doesn't go back far enough; refetch full range
                log.info("Existing data starts at %s, need %s — refetching",
                         first_ts, datetime.fromtimestamp(start_ms/1000, tz=timezone.utc))

    if start_ms >= end_ms:
        log.info("Already up to date for %s %s", symbol, interval)
        return existing_df if existing_df is not None else pd.DataFrame()

    # Paginate
    all_candles: list[dict] = []
    current_ms = start_ms
    batch_num = 0
    total_expected = (end_ms - start_ms) // interval_ms

    print(f"Fetching {symbol} {interval} candles: "
          f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} "
          f"to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} "
          f"(~{total_expected:,} candles)")

    while current_ms < end_ms:
        try:
            batch = _fetch_batch(symbol, interval, current_ms, end_ms)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            log.warning("Batch %d failed: %s. Retrying in 2s...", batch_num, e)
            time.sleep(2)
            try:
                batch = _fetch_batch(symbol, interval, current_ms, end_ms)
            except Exception:
                log.error("Retry failed. Stopping at batch %d.", batch_num)
                break

        if not batch:
            break

        all_candles.extend(batch)
        batch_num += 1

        # Advance past the last candle in this batch
        last_t = max(c["t"] for c in batch)
        current_ms = last_t + interval_ms

        fetched = len(all_candles)
        pct = min(100, fetched / max(total_expected, 1) * 100)
        print(f"  batch {batch_num}: {fetched:,} candles ({pct:.0f}%)", end="\r")

        if len(batch) < MAX_CANDLES_PER_REQUEST:
            break  # last page

        time.sleep(RATE_LIMIT_SLEEP)

    print()

    if not all_candles:
        log.warning("No candles fetched for %s %s", symbol, interval)
        return existing_df if existing_df is not None else pd.DataFrame()

    new_df = _parse_candles(all_candles)

    # Merge with existing data
    if existing_df is not None and len(existing_df) > 0:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    # Deduplicate and sort
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    # Save
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} candles to {out_path}")
    print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return df


# ── XS-1: universe enumeration + sweep ───────────────────────────────────────────
def _info_request(payload: dict) -> object:
    """POST one request to the info endpoint."""
    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_universe(info_fn=None, include_delisted: bool = False,
                   return_excluded: bool = False, retries: int = 4,
                   backoff: float = 2.0):
    """Every perp name the venue currently lists, in meta order.

    `info_fn` is injected so this is testable offline. Raises on a malformed payload
    rather than returning a short list: a truncated universe silently narrows every
    downstream breadth claim, which is worse than a loud failure.
    """
    # The enumeration runs ONCE before any work, so a transient 429 here takes down the
    # whole sweep — it killed the L2 sampler at startup on 2026-08-08. Transport faults
    # are retried with backoff; a malformed payload is NOT (retrying a schema error just
    # burns a minute).
    fn = info_fn or _info_request
    last: Exception | None = None
    data = None
    for attempt in range(max(1, retries)):
        try:
            data = fn({"type": "meta"})
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 >= max(1, retries):
                raise
            wait = backoff * (2 ** attempt)
            log.warning("meta request failed (%s); retrying in %.1fs", exc, wait)
            time.sleep(wait)
    if data is None and last is not None:      # pragma: no cover - defensive
        raise last

    if not isinstance(data, dict):
        raise TypeError(f"meta payload must be an object, got {type(data).__name__}")
    universe = data.get("universe")
    if not isinstance(universe, list) or not universe:
        raise ValueError("meta payload has no non-empty 'universe' list")

    names, excluded = [], []
    for entry in universe:
        if not isinstance(entry, dict):
            raise TypeError(f"universe entry is not an object: {entry!r}")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"universe entry has no usable 'name': {entry!r}")
        if entry.get("isDelisted") and not include_delisted:
            excluded.append(name)
            continue
        names.append(name)

    if not names:
        raise ValueError("universe contained no listed symbols")
    return (names, excluded) if return_excluded else names


def span_days(df, interval_ms: int) -> float:
    """Calendar days actually covered by a candle frame (first→last inclusive)."""
    if df is None or len(df) == 0 or "timestamp" not in getattr(df, "columns", []):
        return 0.0
    ts = pd.to_datetime(df["timestamp"], utc=True)
    span_ms = (ts.max() - ts.min()).total_seconds() * 1000.0 + interval_ms
    return span_ms / 86_400_000.0


def backfill_universe(symbols, interval: str = "1m", days: int = 90,
                      start: str | None = None, output_dir: Path = DATA_DIR,
                      fetch_fn=None, delay: float = SYMBOL_DELAY,
                      max_symbols: int | None = None, retries: int = 2,
                      short_tolerance: float = 0.9) -> dict:
    """Fetch `symbols` one at a time, surviving individual failures.

    Returns a coverage report — `ok` / `failed` / `empty` / `rejected` / `truncated` /
    `short` — which together account for every requested symbol exactly once. A sweep
    that cannot say what it missed cannot support a breadth claim.

    Two guards added by XS-7, both paid for by the 2026-08-07 sweep:

    `retries` — that run reported two `empty` symbols (ORDI 15 m, REZ 5 m) which both
    succeeded on immediate retry. `empty` therefore conflated "the venue has none" with
    "one request hiccupped", and for 1 m candles — which expire at the source within
    ~3.5 days (FINDINGS §7.1) — a transient miss is a *permanent* hole. Both exceptions
    and empty frames are retried; a genuine outage still lands in `failed`.

    `short` — the same run reported `ok=177 failed=0 empty=0` for a 1 m sweep that
    returned **4 % of the requested span**, because `ok` only ever meant "rows came
    back". `short` compares received span against requested and flags anything under
    `short_tolerance`. It is an *annotation on a successful fetch*, not a bucket, so the
    totals still reconcile. Note that a short result is often not a defect — a pair
    listed 3 weeks ago cannot return 90 days, and the venue's per-interval retention cap
    means 1 m *never* will — so this flags for inspection, it does not fail the run.
    """
    fetch = fetch_fn or fetch_candles
    output_dir = Path(output_dir)
    requested = list(symbols)
    interval_ms = INTERVAL_MS[interval]

    report = {"interval": interval, "n_requested": len(requested), "ok": [],
              "failed": [], "empty": [], "rejected": [], "truncated": 0, "rows": {},
              "short": {}, "days_requested": days}

    safe = []
    for s in requested:
        if isinstance(s, str) and SYMBOL_RE.match(s):
            safe.append(s)
        else:
            report["rejected"].append({"symbol": s, "reason": "not a plain ticker"})

    if max_symbols is not None and len(safe) > max_symbols:
        report["truncated"] = len(safe) - max_symbols
        safe = safe[:max_symbols]

    for i, symbol in enumerate(safe):
        if i and delay:
            time.sleep(delay)                    # be polite to the venue

        df, err = None, None
        for attempt in range(retries + 1):
            if attempt and delay:
                time.sleep(delay)                # back off before a retry
            try:
                df = fetch(symbol, interval=interval, start=start, days=days,
                           output_dir=output_dir)
                err = None
            except Exception as exc:             # one symbol must not lose the rest
                df, err = None, exc
            if df is not None and len(df) > 0:
                break                            # got data — stop retrying
            if attempt < retries:
                log.info("%s %s attempt %d/%d yielded %s — retrying", symbol, interval,
                         attempt + 1, retries + 1, "an error" if err else "no rows")

        if err is not None:
            log.warning("%s failed after %d attempt(s): %s", symbol, retries + 1, err)
            report["failed"].append({"symbol": symbol, "error": str(err)[:200]})
            continue
        if df is None or len(df) == 0:
            report["empty"].append(symbol)
            continue

        report["ok"].append(symbol)
        report["rows"][symbol] = int(len(df))

        got = span_days(df, interval_ms)
        if days and got < days * short_tolerance:
            report["short"][symbol] = (round(got, 2), days)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV candles from Hyperliquid",
    )
    parser.add_argument("--universe", action="store_true",
                        help="fetch EVERY listed perp (XS-1), enumerated from meta")
    parser.add_argument("--include-delisted", action="store_true",
                        help="also fetch delisted pairs (default: excluded, reported)")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="cap the sweep; the number dropped is reported, never silent")
    parser.add_argument("--symbol-delay", type=float, default=SYMBOL_DELAY,
                        help=f"seconds between symbols (default {SYMBOL_DELAY})")
    parser.add_argument("--symbol", nargs="+", default=["BTC"],
                        help="Symbols to fetch (default: BTC)")
    parser.add_argument("--interval", default="1m",
                        help="Candle interval (default: 1m)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history to fetch (default: 90)")
    parser.add_argument("--start", default=None,
                        help="Start date (ISO format, overrides --days)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/candles/)")
    return parser


def main():
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.output_dir) if args.output_dir else DATA_DIR

    if args.universe:
        names, excluded = fetch_universe(include_delisted=args.include_delisted,
                                         return_excluded=True)
        print(f"universe: {len(names)} listed perps"
              + (f" ({len(excluded)} delisted excluded: {', '.join(excluded[:8])}"
                 f"{'...' if len(excluded) > 8 else ''})" if excluded else ""))
        report = backfill_universe(
            names, interval=args.interval, days=args.days, start=args.start,
            output_dir=out_dir, delay=args.symbol_delay, max_symbols=args.max_symbols)
        print(f"\ncoverage: ok={len(report['ok'])} failed={len(report['failed'])} "
              f"empty={len(report['empty'])} rejected={len(report['rejected'])} "
              f"truncated={report['truncated']} of {report['n_requested']} requested")
        for f in report["failed"][:10]:
            print(f"  FAILED {f['symbol']}: {f['error'][:80]}")
        if report["empty"]:
            print(f"  empty: {', '.join(report['empty'][:12])}")
        return

    for symbol in args.symbol:
        fetch_candles(
            symbol=symbol,
            interval=args.interval,
            start=args.start,
            days=args.days,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
