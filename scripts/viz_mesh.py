"""nat viz3d / nat mesh — interactive 3D feature-surface-over-time.

Capability C of the viz/validation tooling (docs/requirements/
parquet_viz_validation.md). Builds a Plotly surface where:
  x = time   (whole-day bars in overview; ticks/fine-bars within page N)
  y = features grouped by category
  z = per-feature normalized value (z-score by default, or raw)

Output is a standalone self-contained HTML (Plotly JS embedded) that opens in a
browser with no running server. Uses the shared §3b pagination model so an
optional 1-based INDEX zooms into the Nth --tf-width window.

CLI:
    python scripts/viz_mesh.py --tf 5m --symbol BTC --date 2026-06-15
    python scripts/viz_mesh.py --tf 5m 2 --symbol BTC          # zoom into page 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cluster_pipeline.loader import load_parquet, get_symbols
from viz.pager import window_bounds
from viz.feature_select import META_COLS, select_features  # noqa: F401 (re-exported)

log = logging.getLogger("viz_mesh")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "mesh"

# Time granularity → (overview bar freq, page window minutes, fine page freq).
TF_MAP = {
    "1m": ("1min", 1, "2s"),
    "5m": ("5min", 5, "10s"),
    "15m": ("15min", 15, "30s"),
}


# --------------------------------------------------------------------------- #
# Surface construction (testable, no file IO)
# --------------------------------------------------------------------------- #

def _resample(df_sym: pd.DataFrame, freq: str, cols: list) -> pd.DataFrame:
    """Time-index and resample (mean) the chosen feature columns."""
    idx = pd.to_datetime(df_sym["timestamp_ns"], unit="ns")
    sub = df_sym[cols].copy()
    sub.index = idx
    return sub.resample(freq).mean()


def build_surface(
    df: pd.DataFrame,
    symbol: str,
    tf: str = "15m",
    index: Optional[int] = None,
    features: Optional[str] = None,
    zmode: str = "zscore",
    max_features: int = 40,
):
    """Build a Plotly feature-surface figure. Returns ``(fig, meta)``.

    meta = {n_features, n_time, features, tf, index, partial, capped}.
    """
    import plotly.graph_objects as go  # lazy: only needed for the 3D surface

    overview_freq, win_min, page_freq = TF_MAP[tf]

    df_sym = df[df["symbol"] == symbol].sort_values("timestamp_ns").reset_index(drop=True)
    if df_sym.empty:
        raise ValueError(f"No data for symbol {symbol}")

    partial = False
    if index is None:
        freq = overview_freq
    else:
        ts_min, ts_max = df_sym["timestamp_ns"].min(), df_sym["timestamp_ns"].max()
        t0, t1, n_pages, partial = window_bounds(ts_min, ts_max, win_min, index)
        df_sym = df_sym[(df_sym["timestamp_ns"] >= t0) & (df_sym["timestamp_ns"] < t1)]
        if df_sym.empty:
            raise ValueError(f"page {index} has no rows")
        freq = page_freq

    cols = select_features(df_sym, features)
    if not cols:
        raise ValueError("no numeric feature columns to plot")

    bars = _resample(df_sym, freq, cols)

    # Drop all-NaN feature columns (unavailable, e.g. dead optional categories).
    bars = bars.dropna(axis=1, how="all")
    if bars.shape[1] == 0:
        raise ValueError("all selected features are empty (NaN) for this selection")

    # Cap to the most informative features by variance for legibility.
    capped = False
    if bars.shape[1] > max_features:
        keep = bars.var(numeric_only=True).sort_values(ascending=False).head(max_features).index
        # Preserve category-grouped order from `cols`.
        keep_set = set(keep)
        bars = bars[[c for c in bars.columns if c in keep_set]]
        capped = True

    feat_names = list(bars.columns)
    times = bars.index

    z = bars.to_numpy(dtype=float).T  # shape (n_features, n_time)
    if zmode == "zscore":
        mu = np.nanmean(z, axis=1, keepdims=True)
        sd = np.nanstd(z, axis=1, keepdims=True)
        z = (z - mu) / (sd + 1e-10)

    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    fig = go.Figure(data=[go.Surface(
        z=z, x=x, y=y, colorscale="Viridis",
        colorbar=dict(title=("z-score" if zmode == "zscore" else "value")),
    )])
    time_labels = [pd.Timestamp(t).strftime("%H:%M:%S") for t in times]
    step = max(1, len(time_labels) // 8)
    mode = "overview" if index is None else f"page {index}{' (partial)' if partial else ''}"
    fig.update_layout(
        title=f"{symbol} — feature surface · {tf} · {mode}",
        scene=dict(
            xaxis=dict(title="time",
                       tickmode="array",
                       tickvals=list(x[::step]),
                       ticktext=time_labels[::step]),
            yaxis=dict(title="feature",
                       tickmode="array",
                       tickvals=list(y),
                       ticktext=feat_names),
            zaxis=dict(title=("z-score" if zmode == "zscore" else "value")),
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    meta = {
        "n_features": int(z.shape[0]),
        "n_time": int(z.shape[1]),
        "features": feat_names,
        "tf": tf,
        "index": index,
        "partial": bool(partial),
        "capped": capped,
    }
    return fig, meta


def render(
    data_dir: str,
    symbol: str,
    tf: str = "15m",
    index: Optional[int] = None,
    features: Optional[str] = None,
    zmode: str = "zscore",
    max_features: int = 40,
    output: Optional[Path] = None,
    date: Optional[str] = None,
) -> Path:
    """Load, build the surface, and write a standalone HTML. Returns the path."""
    df = load_parquet(data_dir)
    if symbol not in get_symbols(df):
        raise ValueError(f"symbol {symbol} not in data (available: {get_symbols(df)})")

    fig, meta = build_surface(df, symbol, tf, index, features, zmode, max_features)

    out_dir = Path(output) if output else DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_p{index}" if index is not None else ""
    day = date or "latest"
    out = out_dir / f"{symbol}_{tf}_{day}{tag}.html"
    # include_plotlyjs=True embeds Plotly → fully self-contained / offline.
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    log.info("surface: %d features × %d time points%s",
             meta["n_features"], meta["n_time"], " (capped)" if meta["capped"] else "")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        prog="viz_mesh",
        description="Interactive 3D feature-surface-over-time (Plotly HTML)",
    )
    parser.add_argument("--tf", default="15m", choices=["1m", "5m", "15m"],
                        help="Time granularity / page width (default: 15m)")
    parser.add_argument("index", nargs="?", type=int,
                        help="1-based page: omit for whole-day overview")
    parser.add_argument("--symbol", default="BTC", help="Symbol (default: BTC)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Day directory under data/features (e.g. data/features/2026-06-15)")
    parser.add_argument("--date", default=None, help="Date label for the output filename")
    parser.add_argument("--features", default=None,
                        help="Category, named vector, comma-list, or 'all' (default: all, "
                             "capped by variance)")
    parser.add_argument("--z", dest="zmode", default="zscore", choices=["zscore", "value"],
                        help="Per-feature normalization (default: zscore)")
    parser.add_argument("--max-features", type=int, default=40,
                        help="Cap the y-axis to the top-N features by variance (default: 40)")
    parser.add_argument("--open", dest="open_after", action="store_true",
                        help="Open the produced HTML in the browser")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("  Error: plotly is required for nat viz3d/mesh — `pip install plotly`")
        sys.exit(1)

    try:
        out = render(
            str(args.data_dir), args.symbol, args.tf, args.index, args.features,
            args.zmode, args.max_features, args.output, args.date,
        )
    except (IndexError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)

    print(f"  Saved: {out}")
    if args.open_after:
        from viz.open_helper import open_path
        if not open_path(out):
            print("  (no opener available — open the path above manually)")


if __name__ == "__main__":
    main()
