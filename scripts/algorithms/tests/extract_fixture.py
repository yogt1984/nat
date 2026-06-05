#!/usr/bin/env python3
"""Extract a 1-hour real-data fixture for algorithm integration tests.

Usage:
    python scripts/algorithms/tests/extract_fixture.py \
        --data-dir data/features --date 2026-06-04 --hour 12 --symbol BTC

Output:
    scripts/algorithms/tests/fixtures/btc_1h_real.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from cluster_pipeline.loader import load_parquet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Extract 1h test fixture")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
    parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    parser.add_argument("--hour", type=int, default=12, help="Hour (0-23)")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--output", default=str(
        ROOT / "scripts" / "algorithms" / "tests" / "fixtures" / "btc_1h_real.parquet"
    ))
    args = parser.parse_args()

    print(f"Loading {args.symbol} data from {args.date}...")
    df = load_parquet(
        args.data_dir,
        symbols=[args.symbol],
        start_date=args.date,
        end_date=args.date,
    )
    print(f"  Loaded {len(df)} ticks, {len(df.columns)} columns")

    # Filter to requested hour
    ts = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
    mask = ts.dt.hour == args.hour
    df_hour = df[mask].copy().reset_index(drop=True)
    print(f"  Hour {args.hour}: {len(df_hour)} ticks")

    if len(df_hour) < 100:
        print(f"  WARNING: Only {len(df_hour)} ticks — try a different hour/date")
        sys.exit(1)

    # Quick stats
    mid = df_hour["raw_midprice"]
    returns = mid.pct_change().dropna()
    max_abs_ret = returns.abs().max()
    std_ret = returns.std()
    print(f"  Midprice range: {mid.min():.1f} - {mid.max():.1f}")
    print(f"  Max |return|: {max_abs_ret:.6f} ({max_abs_ret/std_ret:.1f} sigma)")

    if "ctx_funding_rate" in df_hour.columns:
        fr = df_hour["ctx_funding_rate"]
        print(f"  Funding rate range: {fr.min():.8f} - {fr.max():.8f}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_hour.to_parquet(out, index=False)
    print(f"  Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
