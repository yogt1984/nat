"""Data + theme foundation for the `nat viz` group (plan T7 / NAT3).

Thin loaders over the existing parquet / algorithm-registry / paper-trade stores,
plus the shared dark theme, consumed by `nat viz features|algorithm|paper`
(NAT4–7). Keep heavy logic in the underlying stores — this is glue + small stats.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Shared theme — reuse the matplotlib palette so terminal + PNG agree.
try:
    from viz.features import STYLE, COLORS  # noqa: F401
except Exception:  # pragma: no cover - theme is cosmetic
    STYLE, COLORS = {}, ["#58a6ff", "#3fb950", "#f85149", "#d29922"]

DATA_DIR = _ROOT / "data" / "features"
_META_COLS = {"timestamp_ns", "symbol", "datetime", "date", "bar_start"}


def _latest_date(data_dir: Path) -> Optional[str]:
    if not data_dir.exists():
        return None
    days = sorted(p.name for p in data_dir.iterdir()
                  if p.is_dir() and len(p.name) == 10 and p.name[4] == "-")
    return days[-1] if days else None


def load_features(symbol: str = "BTC", hours: Optional[float] = None,
                  date: Optional[str] = None,
                  data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Tick features for one symbol from the latest (or given) day.

    ``hours`` keeps only the last N hours of the window (by ``timestamp_ns``).
    Returns a possibly-empty DataFrame sorted by timestamp.
    """
    from data.features import load_features as _load
    dd = Path(data_dir) if data_dir else DATA_DIR
    day = date or _latest_date(dd)
    if day is None:
        return pd.DataFrame()
    df = _load(symbols=[symbol], date_range=(day, day), data_dir=dd, validate=False)
    if df.empty or hours is None or "timestamp_ns" not in df:
        return df
    cutoff = df["timestamp_ns"].max() - int(hours * 3600 * 1e9)
    return df[df["timestamp_ns"] >= cutoff].reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns (excludes metadata)."""
    return [c for c in df.columns
            if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def forward_return(df: pd.DataFrame, price_col: str = "raw_midprice",
                   horizon: int = 50) -> pd.Series:
    """h-step-ahead simple return of *price_col* (NaN tail). horizon in ticks."""
    if price_col not in df:
        return pd.Series(np.nan, index=df.index)
    p = df[price_col].astype(float)
    return p.shift(-horizon) / p - 1.0


def _downsample(series: pd.Series, n: int) -> list:
    """~n evenly-spaced points across the whole series (shows the arc, not just the tail)."""
    vals = series.tolist()
    if len(vals) <= n:
        return vals
    step = len(vals) / n
    return [vals[int(i * step)] for i in range(n)]


def _spearman_ic(feat: pd.Series, target: pd.Series) -> float:
    pair = pd.concat([feat, target], axis=1).dropna()
    if len(pair) < 30 or pair.iloc[:, 0].nunique() < 3:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def feature_table(df: pd.DataFrame, horizon: int = 50, spark_n: int = 24,
                  alive_only: bool = False) -> list[dict]:
    """Per-feature overview rows: {feature, last, zscore, nan_pct, ic, spark}.

    `ic` is the Spearman IC of the feature vs the h-step forward midprice return
    (NaN when the window is too thin to estimate). Sorted by |IC| descending.
    """
    if df.empty:
        return []
    target = forward_return(df, horizon=horizon)
    rows = []
    for col in feature_columns(df):
        s = df[col].astype(float)
        nan_pct = float(s.isna().mean() * 100.0)
        if alive_only and nan_pct >= 100.0:
            continue
        valid = s.dropna()
        last = float(valid.iloc[-1]) if len(valid) else float("nan")
        mean = float(valid.mean()) if len(valid) else float("nan")
        std = float(valid.std()) if len(valid) else float("nan")
        z = (last - mean) / std if std and std > 0 and not np.isnan(std) else float("nan")
        ic = _spearman_ic(s, target)
        rows.append({"feature": col, "last": last, "zscore": z,
                     "nan_pct": nan_pct, "ic": ic,
                     "spark": _downsample(valid, spark_n)})
    rows.sort(key=lambda r: abs(r["ic"]) if not np.isnan(r["ic"]) else -1.0,
              reverse=True)
    return rows


def load_algorithm_signals(algorithm: str, symbol: str = "BTC",
                           hours: Optional[float] = None) -> pd.DataFrame:
    """Run a registered algorithm over loaded features → its ``alg_*`` columns."""
    from algorithms.autodiscover import discover_all
    from algorithms.registry import get_algorithm
    discover_all()
    df = load_features(symbol, hours=hours)
    if df.empty:
        return pd.DataFrame()
    return get_algorithm(algorithm).run_batch(df)


def load_paper_trades(symbol: Optional[str] = None) -> pd.DataFrame:
    """Load paper-trade records from data/paper/ (.json|.jsonl). Empty if none."""
    import json
    paper_dir = _ROOT / "data" / "paper"
    if not paper_dir.exists():
        return pd.DataFrame()
    recs: list = []
    for f in sorted(paper_dir.glob("**/*.json*")):
        try:
            txt = f.read_text()
            data = ([json.loads(ln) for ln in txt.splitlines() if ln.strip()]
                    if f.suffix == ".jsonl" else json.loads(txt))
            recs.extend(data if isinstance(data, list) else [data])
        except Exception:
            continue
    df = pd.DataFrame(recs)
    if symbol and "symbol" in df:
        df = df[df["symbol"] == symbol]
    return df
