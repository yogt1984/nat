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
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

API_URL = "https://api.hyperliquid.xyz/info"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "candles"
MAX_CANDLES_PER_REQUEST = 5000
RATE_LIMIT_SLEEP = 0.25  # seconds between requests

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


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV candles from Hyperliquid",
    )
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.output_dir) if args.output_dir else DATA_DIR

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
