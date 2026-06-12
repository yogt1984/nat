"""Settlement-clock features (F1 in feature_algorithm_gaps.md).

Derived purely from `timestamp_ns` and `ctx_funding_rate` — no ingestor or
schema changes. Unblocks LF1 (funding-settlement windows) and LF5 (weekend
conditioning) from algorithm_candidates_literature.md.

Features (prefix ``sc_``):

    sc_tts_8h           Seconds until the next cross-venue 8h funding
                        settlement mark (00:00 / 08:00 / 16:00 UTC). 0 at the
                        mark itself.
    sc_tss_8h           Seconds since the previous 8h settlement mark.
    sc_cycle_frac       Position within the 8h settlement cycle, [0, 1).
    sc_hod_sin          sin(2*pi * seconds_of_day / 86400) — hour-of-day.
    sc_hod_cos          cos(...) — together a continuous 24h clock encoding.
    sc_dow_sin          sin(2*pi * continuous_day_of_week / 7).
    sc_dow_cos          cos(...) — continuous 7d clock encoding.
    sc_weekend          1.0 on Saturday/Sunday UTC, else 0.0.
    sc_funding_twa_8h   Trailing 8h time-windowed mean of ctx_funding_rate.
    sc_funding_twa_24h  Trailing 24h time-windowed mean of ctx_funding_rate.

Why the 8h marks matter on Hyperliquid: HL itself pays funding hourly (1/8 of
the 8h-equivalent rate), so it has no native 8h settlement. But Binance, OKX
and Bybit settle at the 00/08/16 UTC marks, forcing cross-venue arbitrage flow
through those times — the LF1 hypothesis is that HL prices and funding premia
show systematic behavior around the *other venues'* clock.

The funding TWA is computed on a 1-minute resampled grid (funding moves
hourly, so minute resolution loses nothing): gap minutes hold NaN, and the
rolling ``min_periods`` therefore counts *minutes of actual time coverage*,
not raw tick count — robust to the documented WebSocket gaps (K4). Windows
with less than ``min_coverage`` (default 0.25 — funding moves hourly, so a
quarter-covered window still estimates it well; current streak days hold only
~9h of ticks) of their span covered emit NaN. The grid is
shifted one step before mapping back to ticks, so the value at tick *t* uses
only data through the end of the previous minute (strictly causal).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAY_S = 86_400.0
CYCLE_S = 8 * 3600.0  # cross-venue settlement cycle (00/08/16 UTC)

CLOCK_FEATURES = [
    "sc_tts_8h",
    "sc_tss_8h",
    "sc_cycle_frac",
    "sc_hod_sin",
    "sc_hod_cos",
    "sc_dow_sin",
    "sc_dow_cos",
    "sc_weekend",
]
FUNDING_FEATURES = ["sc_funding_twa_8h", "sc_funding_twa_24h"]
SETTLEMENT_CLOCK_FEATURES = CLOCK_FEATURES + FUNDING_FEATURES


def compute_settlement_clock(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp_ns",
    funding_col: str = "ctx_funding_rate",
    symbol_col: str = "symbol",
    min_coverage: float = 0.25,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``sc_*`` feature columns appended.

    Args:
        df: Frame with ``ts_col`` (int64 epoch nanoseconds, UTC). If
            ``funding_col`` is absent the funding TWA features are skipped.
            Multi-symbol frames are handled (funding windows group by
            ``symbol_col``).
        min_coverage: Minimum fraction of a trailing window's time span that
            must hold data (measured in 1-minute grid cells) before the
            funding TWA is emitted; below this the value is NaN (warmup /
            data-gap guard).
    """
    if ts_col not in df.columns:
        raise KeyError(f"missing timestamp column {ts_col!r}")

    out = df.copy()
    ts = pd.to_datetime(out[ts_col], unit="ns", utc=True)

    sec_of_day = (
        (ts - ts.dt.normalize()).dt.total_seconds().to_numpy()
    )
    tss = sec_of_day % CYCLE_S
    out["sc_tss_8h"] = tss
    out["sc_tts_8h"] = (CYCLE_S - tss) % CYCLE_S
    out["sc_cycle_frac"] = tss / CYCLE_S

    hod_phase = 2.0 * np.pi * sec_of_day / DAY_S
    out["sc_hod_sin"] = np.sin(hod_phase)
    out["sc_hod_cos"] = np.cos(hod_phase)

    # Continuous day-of-week (Mon=0.0 ... Sun -> 7.0) so the encoding has no
    # midnight discontinuity.
    dow = ts.dt.dayofweek.to_numpy() + sec_of_day / DAY_S
    dow_phase = 2.0 * np.pi * dow / 7.0
    out["sc_dow_sin"] = np.sin(dow_phase)
    out["sc_dow_cos"] = np.cos(dow_phase)
    out["sc_weekend"] = (ts.dt.dayofweek >= 5).to_numpy().astype(np.float64)

    if funding_col in out.columns:
        for hours, name in ((8, "sc_funding_twa_8h"), (24, "sc_funding_twa_24h")):
            out[name] = _trailing_twa(
                out, ts, funding_col, symbol_col,
                window_s=hours * 3600.0, min_coverage=min_coverage,
            )
    else:
        logger.warning(
            "column %r not present — skipping funding TWA features", funding_col
        )

    return out


def _trailing_twa(
    df: pd.DataFrame,
    ts: pd.Series,
    value_col: str,
    symbol_col: str,
    *,
    window_s: float,
    min_coverage: float,
) -> pd.Series:
    """Trailing time-window mean of ``value_col``, grouped by symbol.

    Computed on a 1-minute resampled grid so that ``min_periods`` measures
    minutes of genuine time coverage (gap minutes are NaN and don't count),
    then shifted one grid step and mapped back to tick timestamps — the value
    at tick *t* only uses data through the end of the previous minute.
    """
    grid_s = 60.0
    n = len(df)
    result = np.full(n, np.nan, dtype=np.float64)
    ts_np = ts.dt.tz_localize(None).to_numpy()  # datetime64[ns], UTC
    vals = df[value_col].to_numpy(dtype=np.float64)

    if symbol_col in df.columns:
        group_positions = df.groupby(symbol_col, sort=False).indices.values()
    else:
        group_positions = [np.arange(n)]
    window = pd.Timedelta(seconds=window_s)
    min_periods = max(1, int(min_coverage * window_s / grid_s))

    for pos in group_positions:
        pos = np.asarray(pos)
        p = pos[np.argsort(ts_np[pos], kind="stable")]
        tick_index = pd.DatetimeIndex(ts_np[p])

        grid = pd.Series(vals[p], index=tick_index).resample("1min").mean()
        rolled = grid.rolling(window, min_periods=min_periods).mean()
        causal = rolled.shift(1)
        result[p] = causal.reindex(tick_index, method="ffill").to_numpy()

    return pd.Series(result, index=df.index)


# ── Validation CLI ────────────────────────────────────────────────────────


def _spearman_ic(
    feature: pd.Series, fwd_return: pd.Series
) -> float:
    mask = feature.notna() & fwd_return.notna()
    if mask.sum() < 100 or feature[mask].nunique() < 2:
        return float("nan")
    return float(feature[mask].corr(fwd_return[mask], method="spearman"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute settlement-clock features on real parquet data "
        "and report NaN rates + Spearman IC vs forward returns."
    )
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument(
        "--subsample", type=int, default=10,
        help="Keep every Nth row for the IC computation (default 10 = 1s grid)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster_pipeline.loader import load_parquet

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_parquet(
        args.data_dir,
        symbols=[args.symbol],
        start_date=args.date,
        end_date=args.date,
        columns=["timestamp_ns", "symbol", "raw_midprice", "ctx_funding_rate"],
    )
    if df.empty:
        logger.error("no data for %s on %s", args.symbol, args.date)
        return 1
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    logger.info("loaded %d rows for %s %s", len(df), args.symbol, args.date)

    out = compute_settlement_clock(df)

    print("\n— Feature summary —")
    summary = out[SETTLEMENT_CLOCK_FEATURES].describe().T[
        ["mean", "std", "min", "max"]
    ]
    summary["nan_rate"] = out[SETTLEMENT_CLOCK_FEATURES].isna().mean()
    print(summary.round(6).to_string())

    # IC vs forward returns on a subsampled grid (10 rows ~ 1s at 100ms cadence)
    sub = out.iloc[:: args.subsample].reset_index(drop=True)
    mid = sub["raw_midprice"]
    horizons = {"1m": 60, "5m": 300, "30m": 1800, "1h": 3600, "4h": 14400}
    rows_per_s = 1.0  # after default subsample of 100ms data

    print("\n— Spearman IC vs forward log-returns —")
    header = f"{'feature':<22}" + "".join(f"{h:>9}" for h in horizons)
    print(header)
    for feat in SETTLEMENT_CLOCK_FEATURES:
        cells = []
        for h_s in horizons.values():
            shift = int(h_s * rows_per_s * 10 / args.subsample)
            fwd = np.log(mid.shift(-shift) / mid)
            cells.append(_spearman_ic(sub[feat], fwd))
        print(
            f"{feat:<22}"
            + "".join(f"{c:>9.4f}" if np.isfinite(c) else f"{'—':>9}" for c in cells)
        )

    print(
        "\nNote: clock features are deterministic conditioners, not standalone "
        "alpha — expect small unconditional ICs. The LF1/LF5 hypotheses test "
        "them as gates/interactions, not raw signals."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
