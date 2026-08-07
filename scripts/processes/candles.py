"""PROC-19 — the `candles` data level: multi-symbol bar loading for Class-3 processes.

The runner's existing path loads ONE symbol of 100 ms tick parquet from `data/features/`.
The three `xs_*` processes are cross-sectional — they rank the whole universe at each
rebalance — so they need many symbols in one frame, and their source is the XS-1 candle
archive (`data/candles/{SYMBOL}_{interval}.parquet`), not the tick stack. Flagged in
`specs/maker_system.md` §7 as needing its own task; this is it.

**The shape is the design decision.** The archive holds 177 pairs whose histories differ
by construction: most reach 90 days, recent listings do not (CASHCAT 27 d, GRAM 36 d).
Two natural implementations are both wrong:

  * an **inner join** on timestamp truncates the entire panel to the newest listing —
    one recent coin silently costs 175 pairs their history;
  * an **outer join with fill** invents prices for pairs that had not listed yet, a
    lookahead that would quietly inflate any rank-IC study built on it.

So this returns a **long frame** — one row per (symbol, timestamp) — where absence is
absence. A cross-sectional process reads the universe as it was at each timestamp, which
is also the only honest input for `xs_rank_predictability`.

A symbol that has no file is **named in the report**, never silently omitted: a rank
computed over 140 pairs while believing it covered 177 is biased by whatever the missing
37 had in common (usually: recently listed, thin, or delisted).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Bars carry no mid-price; `close` is the price column for this data level. The tick
#: level's default (`raw_midprice`) does not exist here, and carrying it across would
#: make every price lookup NaN — findings would come back empty rather than erroring.
CANDLE_PRICE_COL = "close"

#: Mirrors `data/fetch_candles.py::INTERVAL_MS`. Kept as a set here so the loader can
#: reject a typo'd interval instead of globbing for files that can never exist.
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

DEFAULT_CANDLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "candles"

_KEYS = ["timestamp", "symbol"]


def available_candle_symbols(data_dir: Path | str = DEFAULT_CANDLE_DIR,
                             interval: str = "1h") -> list[str]:
    """Every symbol with a parquet at this interval, sorted."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    suffix = f"_{interval}.parquet"
    return sorted(p.name[: -len(suffix)] for p in data_dir.glob(f"*{suffix}"))


def load_candles(
    symbols,
    interval: str = "1h",
    data_dir: Path | str = DEFAULT_CANDLE_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: list[str] | None = None,
    return_report: bool = False,
):
    """Load bars for `symbols` into one long frame, sorted by (timestamp, symbol).

    Args:
        symbols: tickers to load. Missing ones are reported, not silently dropped.
        interval: bar size; must be one of `VALID_INTERVALS`.
        start_date / end_date: inclusive ISO dates (UTC), applied per symbol so a
            filter never removes a symbol from the panel outright.
        columns: subset of OHLCV to keep; keys are always included.
        return_report: also return `{"loaded": [...], "missing": [...], "empty": [...]}`.

    Raises:
        ValueError: unknown interval.
        FileNotFoundError: not one requested symbol had a file — an empty panel is never
            a valid result, it is a misconfigured path.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; use one of {sorted(VALID_INTERVALS)}")

    data_dir = Path(data_dir)
    want = list(dict.fromkeys(symbols))          # de-duplicate, preserve order
    report = {"loaded": [], "missing": [], "empty": []}
    frames = []

    for sym in want:
        path = data_dir / f"{sym}_{interval}.parquet"
        if not path.exists():
            report["missing"].append(sym)
            continue
        df = pd.read_parquet(path)
        if df.empty:
            report["empty"].append(sym)
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if start_date is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date is not None:
            # end_date is inclusive of the whole day
            df = df[df["timestamp"] < pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
        if df.empty:
            report["empty"].append(sym)
            continue

        df["symbol"] = sym
        if columns is not None:
            keep = _KEYS + [c for c in columns if c in df.columns and c not in _KEYS]
            df = df[keep]
        frames.append(df)
        report["loaded"].append(sym)

    if not frames:
        raise FileNotFoundError(
            f"no candle data for {want[:5]}{'...' if len(want) > 5 else ''} "
            f"at {interval} under {data_dir}"
        )

    # concat, never join: unequal histories stay unequal. Each timestamp carries the
    # pairs that actually existed then — no truncation, no invented pre-listing rows.
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=_KEYS)
             .sort_values(_KEYS)
             .reset_index(drop=True))
    return (out, report) if return_report else out
