"""
Historical funding-rate fetcher — Hyperliquid fundingHistory API (LF8).

Paginates hourly funding settlements per coin and stores parquet under
``data/funding/<SYMBOL>.parquet``. Supports incremental updates (only entries
newer than the last stored timestamp are fetched).

The venue settles funding EVERY HOUR (verified live 2026-08-12 — COST-9), and
history is retrievable at ≥95-day depth (144 hourly entries observed in the
89–95-day window), so the LF8 study window matches the candle archive. Like the
candle archive, retention is the venue's choice, not ours: a day not fetched is
not guaranteed fetchable later, which is why the fetcher exists as a unit
instead of an inline cell in the study driver.

Usage:
    python scripts/data/fetch_funding.py --symbol BTC ETH SOL --days 90
    python scripts/data/fetch_funding.py --universe --days 90        # LF8 sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data.fetch_candles import (  # noqa: E402
    SYMBOL_RE,
    _info_request,
    fetch_universe,
)

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "funding"
RATE_LIMIT_SLEEP = 0.25  # seconds between requests
SYMBOL_DELAY = 0.5       # seconds between symbols in a universe sweep
#: The venue returns at most ~500 entries per fundingHistory request; a page
#: SMALLER than this is the last page. Pagination advances startTime past the
#: newest entry received, so the bound only has to be an upper bound.
PAGE_LIMIT = 500
#: Transport-fault retries per page. A universe sweep is ~5 pages x 177 coins; a run at a
#: tighter delay lost 75/177 coins to HTTP 429 with no retry in the page path.
RETRIES = 4
BACKOFF = 2.0


def _parse_history(raw: list[dict]) -> pd.DataFrame:
    """Venue entries -> typed frame. Strings are the venue's wire format."""
    if not raw:
        return pd.DataFrame(columns=["time", "funding_rate", "premium"])
    df = pd.DataFrame({
        "time": [int(e["time"]) for e in raw],
        "funding_rate": [float(e["fundingRate"]) for e in raw],
        "premium": [float(e.get("premium", "nan")) for e in raw],
    })
    return df.astype({"time": "int64"})


def _request_with_backoff(info_fn, payload: dict, retries: int = RETRIES,
                          backoff: float = BACKOFF):
    """Retry TRANSPORT faults, never schema faults — the `fetch_universe` rule.

    `_info_request` has no retry of its own; only the one-shot `meta` call in
    `fetch_universe` wraps itself. A universe sweep issues ~5 pages per coin, so it puts
    far more requests per second through the endpoint than any other collector here — a
    run at a 0.15 s delay lost **75 of 177 coins to HTTP 429**. A rate limit is exactly the
    transient that deserves a wait; a payload whose shape changed deserves to fail once,
    loudly, rather than be retried into the same wall.
    """
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return info_fn(payload)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 >= max(1, retries):
                raise
            wait = backoff * (2 ** attempt)
            log.warning("funding page failed (%s); retrying in %.1fs", exc, wait)
            time.sleep(wait)
    if last is not None:                      # pragma: no cover - defensive
        raise last


def fetch_funding(
    symbol: str,
    start_ms: int,
    end_ms: int | None = None,
    info_fn=None,
    sleep_s: float = RATE_LIMIT_SLEEP,
) -> pd.DataFrame:
    """Fetch [start_ms, end_ms] funding history for one coin, paginated.

    ``info_fn`` is injectable for tests (planted pages, no network).
    """
    info_fn = info_fn or _info_request
    pages: list[pd.DataFrame] = []
    cursor = int(start_ms)
    while True:
        payload = {"type": "fundingHistory", "coin": symbol, "startTime": cursor}
        if end_ms is not None:
            payload["endTime"] = int(end_ms)
        raw = _request_with_backoff(info_fn, payload)
        if not raw:
            break
        page = _parse_history(raw)
        pages.append(page)
        newest = int(page["time"].max())
        if len(raw) < PAGE_LIMIT or (end_ms is not None and newest >= end_ms):
            break
        if newest < cursor:  # defensive: a non-advancing cursor must not loop forever
            break
        cursor = newest + 1
        if sleep_s:
            time.sleep(sleep_s)
    if not pages:
        return _parse_history([])
    df = pd.concat(pages, ignore_index=True)
    # Settlements are hourly and unique; duplicates are page-boundary overlap.
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


def update_symbol(
    symbol: str,
    days: int = 90,
    data_dir: Path | str = DATA_DIR,
    info_fn=None,
    now_ms: int | None = None,
    sleep_s: float = RATE_LIMIT_SLEEP,
) -> Path:
    """Fetch/extend one coin's funding parquet. Returns the file path.

    Incremental: resumes after the newest stored settlement. Symbol names are
    rendered into file paths, so anything that is not a plain ticker is
    rejected before it can reach the filesystem (fetch_candles rule).
    """
    if not SYMBOL_RE.match(symbol):
        raise ValueError(f"refusing to use {symbol!r} as a filename")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{symbol}.parquet"

    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    start_ms = now_ms - days * 86_400_000
    existing = None
    if path.exists():
        existing = pd.read_parquet(path)
        if len(existing):
            start_ms = max(start_ms, int(existing["time"].max()) + 1)

    fresh = fetch_funding(symbol, start_ms, info_fn=info_fn, sleep_s=sleep_s)
    if existing is not None and len(existing):
        df = pd.concat([existing, fresh], ignore_index=True)
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    else:
        df = fresh
    df.to_parquet(path, index=False)
    return path


def backfill_universe(
    symbols,
    days: int = 90,
    data_dir: Path | str = DATA_DIR,
    info_fn=None,
    sleep_s: float = RATE_LIMIT_SLEEP,
) -> dict:
    """Fetch funding for every symbol; one failure is recorded, not fatal.

    Returns {"ok": [...], "failed": {symbol: reason}} — the arithmetic
    len(ok) + len(failed) == len(symbols) must close (WP-2 rule: a symbol that
    could not be fetched is not the same as one with no funding).
    """
    ok, failed = [], {}
    for i, sym in enumerate(symbols):
        try:
            update_symbol(sym, days=days, data_dir=data_dir,
                          info_fn=info_fn, sleep_s=sleep_s)
            ok.append(sym)
        except Exception as e:  # noqa: BLE001 — record and continue, never abort a sweep
            failed[sym] = f"{type(e).__name__}: {e}"
            log.warning("funding fetch failed for %s: %s", sym, failed[sym])
        if sleep_s and i + 1 < len(symbols):
            time.sleep(SYMBOL_DELAY if sleep_s == RATE_LIMIT_SLEEP else sleep_s)
    return {"ok": ok, "failed": failed}


def load_funding_panel(index, symbols=None, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Read the archive into a (timestamp x symbol) RATE panel aligned to `index`.

    `index` is a price panel's index, so the result drops straight into
    `xs.rotation.run_rotation(..., funding_wide=…)`. Rates are fractions per hourly
    settlement, the same units as a forward return — no bps scaling.

    **Settlements are stamped a few MILLISECONDS past the hour** (`19:00:00.037`) while
    candle bars sit exactly on it, so the timestamps are rounded to the hour before
    alignment. Without that, an exact reindex matched **32 of 2198 rows** on the real
    archive: the study would have reported funding as charged and then charged almost
    nothing — a silent near-zero, which is worse than an obvious zero because it survives
    review.

    Missing cells stay **NaN** rather than 0 so a caller can still tell "no funding" from
    "no data"; the rotation treats NaN as zero-for-that-cell. No forward-fill — an hour
    with no settlement is not the previous hour's rate. Returns an empty frame when
    nothing is archived, so a caller can detect that instead of silently pricing at zero.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return pd.DataFrame(index=index)

    want = set(symbols) if symbols is not None else None
    series = {}
    for p in sorted(data_dir.glob("*.parquet")):
        symbol = p.stem
        if want is not None and symbol not in want:
            continue
        df = pd.read_parquet(p, columns=["time", "funding_rate"])
        if df.empty:
            continue
        ts = pd.to_datetime(df["time"], unit="ms", utc=True).dt.round("h")
        s = pd.Series(df["funding_rate"].to_numpy(), index=ts).sort_index()
        series[symbol] = s[~s.index.duplicated(keep="last")]

    if not series:
        return pd.DataFrame(index=index)

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return pd.DataFrame(series).reindex(idx)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch Hyperliquid funding history")
    p.add_argument("--symbol", nargs="+", help="coin symbols (e.g. BTC ETH)")
    p.add_argument("--universe", action="store_true",
                   help="fetch every perp from the venue meta endpoint")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if not args.universe and not args.symbol:
        raise SystemExit("need --symbol or --universe")
    symbols = fetch_universe() if args.universe else args.symbol
    result = backfill_universe(symbols, days=args.days, data_dir=args.data_dir)
    n_ok, n_fail = len(result["ok"]), len(result["failed"])
    print(f"ok={n_ok} failed={n_fail} of {len(symbols)}")
    if result["failed"]:
        for sym, why in result["failed"].items():
            print(f"  FAILED {sym}: {why}")


if __name__ == "__main__":
    main()
