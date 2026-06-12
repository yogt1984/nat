"""HAR-RV components and forecast (F4 in feature_algorithm_gaps.md).

Heterogeneous Autoregressive Realized Volatility (Corsi 2009): tomorrow's
volatility is explained by an additive cascade of daily, weekly and monthly
realized-vol components. For crypto, HAR beats GARCH and most ML for
1-day-ahead RV forecasting. These features are **not alpha** — they feed
position sizing (LF6), `meta_portfolio.py` risk parity, the hierarchical
combiner's L3 layer, and kill-switch context.

Derived purely from ``raw_midprice`` + ``timestamp_ns`` — no ingestor or
schema changes.

Features (prefix ``rv_``), annualized volatility as a fraction (0.45 = 45%):

    rv_vol_1d        Realized vol over the trailing 24h
    rv_vol_1w        Realized vol over the trailing 7d (daily-average scale)
    rv_vol_1m        Realized vol over the trailing 30d
    rv_ratio_1d_1w   Short/long vol regime ratio (>1 = vol expanding); the
                     conditioning input for the vol-squeeze candidate (A3)
    rv_har_fcst_1d   Log-HAR forecast of the *next* 24h realized vol
                     (requires a fitted ``HarRvEstimator``)

Construction (per symbol, 24/7 crypto conventions):

    Returns are taken on a 5-minute resampled grid — the classic
    noise-robust choice; naive 100ms RV is dominated by microstructure
    noise. Gap cells are NaN, so no return ever spans a data gap, and
    rolling windows use a *time-coverage* guard exactly like the
    settlement-clock features: realized variance is the windowed **mean** of
    squared returns scaled to a full day (288 cells), which stays unbiased
    under gaps *if* missing periods have the same vol as observed ones —
    documented assumption, acceptable for sizing, revisit before any
    vol-targeted execution. Annualization uses 365 days. Components are
    shifted one grid step before mapping back to ticks (strictly causal).

    The forecaster fits log-HAR by OLS on a 20-min-subsampled grid:
    log RV_{t+1d} = b0 + bd log RV_d + bw log RV_w + bm log RV_m. With short
    histories the monthly term is dropped automatically when too few rows
    carry it (terms used are recorded and persisted). Overlapping-window OLS
    inflates t-stats but not point estimates; we only use the point forecast.

**First validation (BTC 2026-05-18..29, time-split 70/30):** components are
sane (annualized vol 0.28-0.41, monthly NaN until a true 30d history exists
~Jul 1). The forecaster is *data-limited* at this history length: an 8-day
fit learned vol mean-reversion (negative coefficients), which OOS rank-beat
the persistence baseline (Spearman +0.40 vs -0.13) but carried a low level
bias (log-RMSE 0.32 vs 0.20). HAR is designed for months of daily history;
treat ``rv_har_fcst_1d`` as experimental until ~30 days accumulate and
prefer ``rv_vol_1d`` (or a blend) for sizing until then. The components
themselves are production-usable now.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_S = 300.0  # 5-minute return grid
CELLS_PER_DAY = int(86_400 / GRID_S)
ANNUALIZE_DAYS = 365.0
_FIT_SUBSAMPLE = 4  # 20-minute samples from the 5-min grid

COMPONENT_WINDOWS = {"1d": 1, "1w": 7, "1m": 30}  # days
HAR_RV_FEATURES = [
    "rv_vol_1d",
    "rv_vol_1w",
    "rv_vol_1m",
    "rv_ratio_1d_1w",
    "rv_har_fcst_1d",
]


def _grid_components(
    sub: pd.DataFrame,
    *,
    mid_col: str,
    ts_col: str,
    min_coverage: float,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Annualized vol components on the 5-min grid for one sorted symbol.

    Returns (grid frame with one column per component, tick DatetimeIndex).
    """
    ts = pd.to_datetime(sub[ts_col], unit="ns", utc=True).dt.tz_localize(None)
    tick_index = pd.DatetimeIndex(ts)
    px = pd.Series(sub[mid_col].to_numpy(dtype=np.float64), index=tick_index)
    grid_px = px.resample(f"{int(GRID_S)}s").last()
    sq = np.log(grid_px).diff() ** 2

    out = pd.DataFrame(index=sq.index)
    for name, days in COMPONENT_WINDOWS.items():
        window = pd.Timedelta(days=days)
        min_periods = max(2, int(min_coverage * days * CELLS_PER_DAY))
        mean_sq = sq.rolling(window, min_periods=min_periods).mean()
        out[f"rv_vol_{name}"] = np.sqrt(mean_sq * CELLS_PER_DAY * ANNUALIZE_DAYS)
    return out, tick_index


def compute_rv_components(
    df: pd.DataFrame,
    *,
    mid_col: str = "raw_midprice",
    ts_col: str = "timestamp_ns",
    symbol_col: str = "symbol",
    min_coverage: float = 0.25,
) -> pd.DataFrame:
    """Return a copy of ``df`` with rv_vol_* and rv_ratio_1d_1w appended."""
    if ts_col not in df.columns:
        raise KeyError(f"missing timestamp column {ts_col!r}")

    out = df.copy()
    for name in COMPONENT_WINDOWS:
        out[f"rv_vol_{name}"] = np.nan

    groups = (
        out.groupby(symbol_col, sort=False).indices.items()
        if symbol_col in out.columns
        else [(None, np.arange(len(out)))]
    )
    for _, pos in groups:
        pos = np.asarray(pos)
        sub = out.iloc[pos].sort_values(ts_col)
        grid, tick_index = _grid_components(
            sub, mid_col=mid_col, ts_col=ts_col, min_coverage=min_coverage
        )
        sorted_pos = pos[np.argsort(out[ts_col].iloc[pos].to_numpy(), kind="stable")]
        for name in COMPONENT_WINDOWS:
            col = f"rv_vol_{name}"
            causal = grid[col].shift(1).reindex(tick_index, method="ffill")
            out.iloc[sorted_pos, out.columns.get_loc(col)] = causal.to_numpy()

    out["rv_ratio_1d_1w"] = out["rv_vol_1d"] / out["rv_vol_1w"]
    return out


class HarRvEstimator:
    """Per-symbol log-HAR forecaster of next-24h realized vol."""

    def __init__(
        self,
        *,
        min_coverage: float = 0.25,
        min_fit_rows: int = 50,
        min_term_std: float = 0.02,
    ):
        self.min_coverage = min_coverage
        self.min_fit_rows = min_fit_rows
        # a term whose log-vol barely varies in the fit window (e.g. a weekly
        # component seen through a short, gappy frame) produces wild OLS
        # coefficients — drop to the next smaller term set instead
        self.min_term_std = min_term_std
        self.coef_: np.ndarray | None = None  # intercept + one per term
        self.terms_: list[str] = []
        self.symbol_: str = ""

    def fit(
        self,
        df: pd.DataFrame,
        *,
        mid_col: str = "raw_midprice",
        ts_col: str = "timestamp_ns",
    ) -> "HarRvEstimator":
        if "symbol" in df.columns:
            symbols = df["symbol"].unique()
            if len(symbols) > 1:
                raise ValueError(
                    f"fit() expects a single symbol, got {list(symbols)} — "
                    "use fit_per_symbol() for multi-symbol frames"
                )
            self.symbol_ = str(symbols[0])

        sub = df.sort_values(ts_col)
        grid, _ = _grid_components(
            sub, mid_col=mid_col, ts_col=ts_col, min_coverage=self.min_coverage
        )
        target = grid["rv_vol_1d"].shift(-CELLS_PER_DAY)  # vol over (t, t+1d]
        hourly = grid.iloc[::_FIT_SUBSAMPLE]
        y = np.log(target.iloc[::_FIT_SUBSAMPLE])

        for terms in (["1d", "1w", "1m"], ["1d", "1w"], ["1d"]):
            x = np.log(hourly[[f"rv_vol_{t}" for t in terms]])
            mask = (
                np.isfinite(y.to_numpy())
                & np.isfinite(x.to_numpy()).all(axis=1)
            )
            if mask.sum() >= self.min_fit_rows:
                x_arr = x.to_numpy()[mask]
                if np.any(x_arr.std(axis=0) < self.min_term_std):
                    continue  # degenerate term — try a smaller term set
                design = np.column_stack([np.ones(mask.sum()), x_arr])
                self.coef_, *_ = np.linalg.lstsq(
                    design, y.to_numpy()[mask], rcond=None
                )
                self.terms_ = terms
                logger.info(
                    "fitted %s log-HAR on %d rows, terms %s, coef %s",
                    self.symbol_ or "<unnamed>", int(mask.sum()), terms,
                    np.round(self.coef_, 3),
                )
                return self
        raise ValueError(
            f"insufficient data to fit HAR (need {self.min_fit_rows} rows "
            "with finite components and target)"
        )

    def forecast(self, components: pd.DataFrame) -> np.ndarray:
        """Next-24h annualized vol from rv_vol_* columns."""
        if self.coef_ is None:
            raise RuntimeError("estimator not fitted")
        x = np.log(
            components[[f"rv_vol_{t}" for t in self.terms_]].to_numpy(
                dtype=np.float64
            )
        )
        design = np.column_stack([np.ones(len(x)), x])
        with np.errstate(invalid="ignore"):
            return np.exp(design @ self.coef_)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol_,
            "min_coverage": self.min_coverage,
            "terms": self.terms_,
            "coef": self.coef_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HarRvEstimator":
        est = cls(min_coverage=d.get("min_coverage", 0.25))
        est.symbol_ = d.get("symbol", "")
        est.terms_ = list(d["terms"])
        est.coef_ = np.asarray(d["coef"], dtype=np.float64)
        return est


def fit_per_symbol(df: pd.DataFrame, **kwargs) -> dict[str, HarRvEstimator]:
    """Fit one estimator per symbol from a multi-symbol frame."""
    out: dict[str, HarRvEstimator] = {}
    for symbol, group in df.groupby("symbol", sort=False):
        try:
            out[str(symbol)] = HarRvEstimator(**kwargs).fit(group)
        except ValueError as exc:
            logger.warning("skipping %s: %s", symbol, exc)
    return out


def compute_har_rv(
    df: pd.DataFrame,
    estimators: dict[str, HarRvEstimator] | None = None,
    **component_kwargs,
) -> pd.DataFrame:
    """Return a copy of ``df`` with all ``rv_*`` features appended.

    If ``estimators`` is None they are fitted on ``df`` itself (in-sample —
    fine for exploration; use a train/eval split for honest forecast
    numbers, as the CLI does).
    """
    if estimators is None:
        estimators = fit_per_symbol(df)

    out = compute_rv_components(df, **component_kwargs)
    out["rv_har_fcst_1d"] = np.nan
    if "symbol" in out.columns:
        for symbol, est in estimators.items():
            mask = (out["symbol"] == symbol).to_numpy()
            if mask.any():
                out.iloc[
                    np.flatnonzero(mask), out.columns.get_loc("rv_har_fcst_1d")
                ] = est.forecast(out.loc[mask])
    elif estimators:
        est = next(iter(estimators.values()))
        out["rv_har_fcst_1d"] = est.forecast(out)
    return out


# ── Validation CLI ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit log-HAR on the first part of a date range, evaluate "
        "the 1-day-ahead vol forecast OOS against a persistence baseline."
    )
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--fit-frac", type=float, default=0.7)
    parser.add_argument("--save-table", help="Write fitted coefficients JSON here")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster_pipeline.loader import load_parquet

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_parquet(
        args.data_dir,
        symbols=[args.symbol],
        start_date=args.start,
        end_date=args.end,
        columns=["timestamp_ns", "symbol", "raw_midprice"],
    )
    if df.empty:
        logger.error("no data for %s in %s..%s", args.symbol, args.start, args.end)
        return 1
    df = df.sort_values("timestamp_ns").reset_index(drop=True)

    # Time split: coefficients are fitted strictly on the past, but the
    # components themselves are causal, so the eval segment legitimately
    # uses full-history components (a window that began before the split
    # only contains pre-t data).
    ts_arr = df["timestamp_ns"].to_numpy()
    split_ns = int(ts_arr[0] + args.fit_frac * (ts_arr[-1] - ts_arr[0]))
    fit_df = df[df["timestamp_ns"] < split_ns]
    logger.info(
        "fit on %d rows (< %s), evaluating after",
        len(fit_df), pd.Timestamp(split_ns, unit="ns"),
    )

    est = HarRvEstimator().fit(fit_df)
    print("\n— Log-HAR fit —")
    print(f"terms: {est.terms_}")
    print(f"coef (intercept first): {np.round(est.coef_, 4).tolist()}")

    if args.save_table:
        Path(args.save_table).write_text(json.dumps(est.to_dict()))
        logger.info("coefficients saved to %s", args.save_table)

    grid, _ = _grid_components(
        df, mid_col="raw_midprice", ts_col="timestamp_ns", min_coverage=0.25
    )
    eval_grid = grid[grid.index >= pd.Timestamp(split_ns, unit="ns")]
    eval_grid = eval_grid.assign(
        rv_ratio_1d_1w=eval_grid["rv_vol_1d"] / eval_grid["rv_vol_1w"],
        rv_har_fcst_1d=est.forecast(eval_grid),
    )

    print("\n— OOS feature summary (full-history components, 5-min grid) —")
    cols = [c for c in HAR_RV_FEATURES if c in eval_grid.columns]
    summary = eval_grid[cols].describe().T[["mean", "std", "min", "max"]]
    summary["nan_rate"] = eval_grid[cols].isna().mean()
    print(summary.round(4).to_string())

    # OOS forecast quality vs persistence baseline
    sub = eval_grid.iloc[::_FIT_SUBSAMPLE]
    realized_next = grid["rv_vol_1d"].shift(-CELLS_PER_DAY).reindex(sub.index)
    frame = pd.DataFrame(
        {
            "realized": realized_next,
            "har": sub["rv_har_fcst_1d"],
            "persistence": sub["rv_vol_1d"],
        }
    ).dropna()
    if len(frame) > 10:
        print(f"\n— OOS 1-day-ahead forecast quality ({len(frame)} 20-min obs) —")
        for name in ("har", "persistence"):
            corr = frame["realized"].corr(frame[name], method="spearman")
            rmse = float(
                np.sqrt(np.mean((np.log(frame[name]) - np.log(frame["realized"])) ** 2))
            )
            print(f"{name:>12}: spearman {corr:+.3f}, log-RMSE {rmse:.4f}")
    else:
        print("\n(too little OOS data for forecast evaluation — need >1 day)")

    print(
        "\nNote: rv_* features are sizing/conditioning inputs, not alpha. "
        "HAR should beat persistence on log-RMSE; both should correlate "
        "strongly with realized vol."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
