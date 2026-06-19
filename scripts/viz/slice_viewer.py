"""slice_viewer — render a slice of a single parquet file to PNGs, show, then delete.

Powers `nat viz <file>` and the `nat test {1m,5m,15m}` capture-and-visualize loop.
Reuses the dark-theme matplotlib panels from ``viz.features`` and the cross-platform
opener from ``viz.open_helper``. PNGs are written to a temp dir, opened one at a time
(waiting for the user between images), and deleted once all have been shown.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Headless-safe backend before pyplot is imported anywhere downstream.
import matplotlib
matplotlib.use("Agg")

from viz.features import plot_feature_panel  # noqa: E402
from viz.open_helper import open_path  # noqa: E402

# Columns that are never interesting to plot as a signal.
_SKIP_COLS = {"symbol", "timestamp", "timestamp_ns", "datetime", "date"}
_PANEL_SIZE = 6  # features per PNG page


def _read_file(path: Path, symbol: Optional[str]) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    return df.reset_index(drop=True)


def parse_slice(expr: Optional[str], n: int) -> tuple[int, int]:
    """Parse a slice expression against ``n`` rows.

    Accepts: ``None``/``''``/``all`` → whole file; ``start:end`` → row range;
    a bare integer ``k`` → first k rows. Out-of-range values are clamped.
    """
    if not expr or expr.strip().lower() == "all":
        return 0, n
    expr = expr.strip()
    if ":" in expr:
        lo_s, hi_s = expr.split(":", 1)
        lo = int(lo_s) if lo_s.strip() else 0
        hi = int(hi_s) if hi_s.strip() else n
    else:
        lo, hi = 0, int(expr)
    lo = max(0, min(lo, n))
    hi = max(lo, min(hi, n))
    return lo, hi


def _pick_columns(df: pd.DataFrame, cols: Optional[list[str]], cap: int) -> list[str]:
    if cols:
        return [c for c in cols if c in df.columns]
    numeric = [
        c for c in df.columns
        if c not in _SKIP_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Rank by variance over non-NaN values; drop all-NaN / constant columns.
    scored = []
    for c in numeric:
        v = df[c].to_numpy(dtype="float64", na_value=np.nan)
        if np.isfinite(v).sum() < 2:
            continue
        var = float(np.nanvar(v))
        if var > 0:
            scored.append((var, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:cap]]


def render_slice(
    path,
    out_dir: Path,
    slice_expr: Optional[str] = None,
    cols: Optional[list[str]] = None,
    symbol: Optional[str] = None,
    max_features: int = 12,
) -> list[Path]:
    """Render the chosen slice/columns to PNG pages in ``out_dir``. Returns paths."""
    import matplotlib.pyplot as plt

    path = Path(path)
    df = _read_file(path, symbol)
    n = len(df)
    if n == 0:
        raise ValueError(f"No rows in {path}" + (f" for symbol {symbol}" if symbol else ""))

    lo, hi = parse_slice(slice_expr, n)
    sl = df.iloc[lo:hi].copy()

    chosen = _pick_columns(sl, cols, max_features)
    if not chosen:
        raise ValueError("No plottable numeric columns in the selected slice")

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    for pi in range(0, len(chosen), _PANEL_SIZE):
        group = chosen[pi:pi + _PANEL_SIZE]
        fig = plot_feature_panel(sl, group, symbol=symbol)
        fig.suptitle(
            f"{path.name}  rows {lo}:{hi} of {n}"
            + (f"  [{symbol}]" if symbol else ""),
            fontsize=11, fontweight="bold", y=1.005,
        )
        png = out_dir / f"slice_{pi // _PANEL_SIZE + 1:02d}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        pages.append(png)
    return pages


def show_and_delete(pages: list[Path], wait: bool = True) -> None:
    """Open each PNG in the system viewer, pausing between, then delete the temp dir."""
    if not pages:
        return
    parent = pages[0].parent
    try:
        for i, png in enumerate(pages, 1):
            opened = open_path(png)
            label = f"[{i}/{len(pages)}] {png.name}"
            if opened:
                print(f"  shown {label}")
            else:
                print(f"  (no viewer) {label} -> {png}")
            if wait and i < len(pages):
                try:
                    input("  press Enter for next image... ")
                except (EOFError, KeyboardInterrupt):
                    break
        if wait:
            try:
                input("  press Enter to close and delete temp images... ")
            except (EOFError, KeyboardInterrupt):
                pass
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def view(
    path,
    slice_expr: Optional[str] = None,
    cols: Optional[list[str]] = None,
    symbol: Optional[str] = None,
    max_features: int = 12,
    delete: bool = True,
    prompt: bool = True,
) -> dict:
    """End-to-end: optionally prompt for the slice, render, show, delete.

    Returns a manifest dict (useful for --json / --no-delete debugging).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Peek row count so the prompt can show it.
    n_total = pq.read_metadata(path).num_rows
    if prompt and slice_expr is None:
        print(f"  {path.name}: {n_total} rows total")
        try:
            slice_expr = input("  slice (e.g. 0:500, a count like 200, or 'all'): ").strip()
        except (EOFError, KeyboardInterrupt):
            slice_expr = "all"

    out_dir = Path(tempfile.mkdtemp(prefix="nat_viz_"))
    pages = render_slice(
        path, out_dir, slice_expr=slice_expr, cols=cols,
        symbol=symbol, max_features=max_features,
    )
    manifest = {
        "file": str(path),
        "rows_total": int(n_total),
        "slice": slice_expr or "all",
        "pages": [str(p) for p in pages],
        "out_dir": str(out_dir),
    }
    if delete:
        show_and_delete(pages, wait=prompt)
        manifest["deleted"] = True
    else:
        print(f"  rendered {len(pages)} page(s) -> {out_dir} (kept, --no-delete)")
        manifest["deleted"] = False
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render a parquet slice to PNGs, show, delete")
    ap.add_argument("file", help="Parquet file to visualize")
    ap.add_argument("--slice", dest="slice_expr", default=None, help="0:500, a count, or 'all'")
    ap.add_argument("--cols", default=None, help="Comma-separated columns (default: top-variance)")
    ap.add_argument("--symbol", "-s", default=None, help="Filter to one symbol")
    ap.add_argument("--max-features", type=int, default=12)
    ap.add_argument("--no-delete", dest="delete", action="store_false")
    ap.add_argument("--no-prompt", dest="prompt", action="store_false")
    a = ap.parse_args()
    cols = a.cols.split(",") if a.cols else None
    view(a.file, slice_expr=a.slice_expr, cols=cols, symbol=a.symbol,
         max_features=a.max_features, delete=a.delete, prompt=a.prompt)
