"""Multi-band order flow imbalance features (F3 in feature_algorithm_gaps.md).

Order Flow Imbalance (Cont, Kukanov & Stoikov 2014) measures the *flow* of
book changes between snapshots — distinct from the existing ``imbalance_*``
features, which are level *snapshots*. The integrated multi-level variant
(Cont, Cucuringu & Zhang 2023, arXiv 2112.13213) combines OFIs across book
depths via the first principal component and is the configuration with
documented predictive power; the failed ``weighted_ofi`` algorithm never
tested it.

**Data limitation (read this):** the exact CCZ estimator needs per-level
prices and sizes for L1-L10. The ingestor emits only L1 imbalance (a size
*ratio*), best prices via ``raw_midprice``/``raw_spread``, and *cumulative*
depths over the top 5/10 levels. This module therefore computes a 3-band
best-effort variant:

    band l1   exact OFI flow rules on normalized L1 sizes
              (qb, qa) = ((1+imb)/2, (1-imb)/2) — direction and structure
              preserved, absolute volume lost
    band d5   OFI flow rules applied to 5-level cumulative depths with the
              best quotes as the price reference
    band d10  same on 10-level cumulative depths

If this variant shows promise, the full per-level OFI is the post-Jun-17
ingestor item (F6-adjacent) in feature_algorithm_gaps.md.

Features (prefix ``ofi_``), rolling time-window sums of step OFIs:

    ofi_l1_10s,  ofi_l1_1m,  ofi_l1_5m     L1 band
    ofi_d5_10s,  ofi_d5_1m,  ofi_d5_5m     5-level band (depth-normalized)
    ofi_d10_10s, ofi_d10_1m, ofi_d10_5m    10-level band (depth-normalized)
    ofi_int_10s, ofi_int_1m, ofi_int_5m    PCA-integrated across bands

OFI flow rules per step (bid side; ask side mirrored):

    Pb up        ->  +qb_t        (new liquidity at a better price)
    Pb unchanged ->  qb_t - qb_{t-1}
    Pb down      ->  -qb_{t-1}    (liquidity consumed or pulled)

Depth bands are normalized by a trailing 10-minute average band depth (per
CCZ) before summing. Steps spanning data gaps (> ``max_step_s``) emit NaN
and drop out of the rolling sums. Integration weights are the first
eigenvector of the correlation matrix of the 1-minute band OFIs (fitted per
symbol, sign-fixed positive, L1-normalized), applied to standardized bands.

**First validation verdict (BTC 2026-06-11, 70/30 split — preliminary, one
partial day):** the 3-band OFI carries real short-horizon signal (OOS IC
0.21-0.24 @ 1s for 10s windows) but is dominated by the *snapshot*
``imbalance_qty_l1`` baseline (0.56 @ 1s) at every window x horizon tested,
and adds **zero incremental IC** once residualized on the snapshot (-0.003);
rank-combining the two *hurts* (0.47 vs 0.56 alone). At the 100ms snapshot
grid with only 3 coarse bands, OFI appears to be a noisier copy of the
imbalance level. Consequences: (1) do NOT promote these features to the Rust
ingestor on current evidence; (2) HF2's candidate priority drops; (3) the
untested configuration remains true per-level L1-L10 OFI from raw book
events — the post-Jun-17 ingestor item. Re-validate across more days before
a final verdict.
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

BANDS = ["l1", "d5", "d10"]
WINDOWS = {"10s": 10.0, "1m": 60.0, "5m": 300.0}
MULTILEVEL_OFI_FEATURES = [
    f"ofi_{band}_{win}" for band in BANDS for win in WINDOWS
] + [f"ofi_int_{win}" for win in WINDOWS]

_FIT_WINDOW = "1m"  # PCA weights are estimated on this window's sums
_DEPTH_NORM_WINDOW_S = 600.0
_PRICE_REL_TOL = 1e-9


def _step_ofi(
    price_ref_bid: np.ndarray,
    price_ref_ask: np.ndarray,
    qty_bid: np.ndarray,
    qty_ask: np.ndarray,
    gap_ok: np.ndarray,
) -> np.ndarray:
    """Vectorized Cont-Kukanov-Stoikov OFI per step; index t holds the flow
    over (t-1, t]. First element and gap steps are NaN."""
    n = len(price_ref_bid)
    tol_b = np.abs(price_ref_bid[:-1]) * _PRICE_REL_TOL
    tol_a = np.abs(price_ref_ask[:-1]) * _PRICE_REL_TOL
    d_pb = price_ref_bid[1:] - price_ref_bid[:-1]
    d_pa = price_ref_ask[1:] - price_ref_ask[:-1]

    e_bid = np.where(d_pb >= -tol_b, qty_bid[1:], 0.0) - np.where(
        d_pb <= tol_b, qty_bid[:-1], 0.0
    )
    e_ask = np.where(d_pa <= tol_a, qty_ask[1:], 0.0) - np.where(
        d_pa >= -tol_a, qty_ask[:-1], 0.0
    )

    out = np.full(n, np.nan)
    out[1:] = np.where(gap_ok, e_bid - e_ask, np.nan)
    return out


class OFIEstimator:
    """Per-symbol PCA integration weights for the 3 OFI bands."""

    def __init__(self, *, max_step_s: float = 1.0) -> None:
        self.max_step_s = max_step_s
        self.weights_: np.ndarray | None = None  # (len(BANDS),), ||w||_1 = 1
        self.scales_: np.ndarray | None = None  # per-band std used for PCA
        self.symbol_: str = ""

    def fit(self, df: pd.DataFrame) -> "OFIEstimator":
        """Fit integration weights from a single-symbol frame."""
        if "symbol" in df.columns:
            symbols = df["symbol"].unique()
            if len(symbols) > 1:
                raise ValueError(
                    f"fit() expects a single symbol, got {list(symbols)} — "
                    "use fit_per_symbol() for multi-symbol frames"
                )
            self.symbol_ = str(symbols[0])

        sums = _band_window_sums(df, max_step_s=self.max_step_s)
        x = sums[[f"ofi_{band}_{_FIT_WINDOW}" for band in BANDS]].dropna()
        if len(x) < 1000:
            raise ValueError(f"need >= 1000 valid windowed rows, got {len(x)}")

        scales = x.std().to_numpy().copy()
        scales[scales == 0] = 1.0
        corr = np.corrcoef((x.to_numpy() / scales).T)
        eigvals, eigvecs = np.linalg.eigh(corr)
        w = eigvecs[:, -1]
        if w.sum() < 0:
            w = -w
        self.weights_ = w / np.abs(w).sum()
        self.scales_ = scales
        logger.info(
            "fitted %s: weights %s (1st PC explains %.0f%%)",
            self.symbol_ or "<unnamed>",
            np.round(self.weights_, 3),
            100 * eigvals[-1] / eigvals.sum(),
        )
        return self

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol_,
            "max_step_s": self.max_step_s,
            "weights": self.weights_.tolist(),
            "scales": self.scales_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OFIEstimator":
        est = cls(max_step_s=d.get("max_step_s", 1.0))
        est.symbol_ = d.get("symbol", "")
        est.weights_ = np.asarray(d["weights"], dtype=np.float64)
        est.scales_ = np.asarray(d["scales"], dtype=np.float64)
        return est


def _band_window_sums(
    df: pd.DataFrame, *, max_step_s: float
) -> pd.DataFrame:
    """Step OFIs per band -> trailing time-window sums. Single symbol,
    returns a frame aligned to ``df``'s index with the 9 band columns."""
    sub = df.sort_values("timestamp_ns")
    ts = pd.to_datetime(sub["timestamp_ns"], unit="ns", utc=True)
    ts_index = pd.DatetimeIndex(ts.dt.tz_localize(None))

    mid = sub["raw_midprice"].to_numpy(dtype=np.float64)
    spread = sub["raw_spread"].to_numpy(dtype=np.float64)
    pb, pa = mid - spread / 2.0, mid + spread / 2.0
    gap_ok = np.diff(sub["timestamp_ns"].to_numpy()) <= max_step_s * 1e9

    imb = sub["imbalance_qty_l1"].to_numpy(dtype=np.float64)
    qb1, qa1 = (1.0 + imb) / 2.0, (1.0 - imb) / 2.0

    steps: dict[str, np.ndarray] = {
        "l1": _step_ofi(pb, pa, qb1, qa1, gap_ok)
    }
    for band, n_levels in (("d5", 5), ("d10", 10)):
        db = sub[f"raw_bid_depth_{n_levels}"].to_numpy(dtype=np.float64)
        da = sub[f"raw_ask_depth_{n_levels}"].to_numpy(dtype=np.float64)
        raw = _step_ofi(pb, pa, db, da, gap_ok)
        # CCZ normalization: divide by trailing average band depth
        half_depth = pd.Series((db + da) / 2.0, index=ts_index)
        norm = (
            half_depth.rolling(
                pd.Timedelta(seconds=_DEPTH_NORM_WINDOW_S), min_periods=50
            )
            .mean()
            .to_numpy()
        )
        steps[band] = raw / np.where(norm > 0, norm, np.nan)

    out = pd.DataFrame(index=sub.index)
    for band, step in steps.items():
        series = pd.Series(step, index=ts_index)
        for win_name, win_s in WINDOWS.items():
            rolled = series.rolling(
                pd.Timedelta(seconds=win_s), min_periods=1
            ).sum()
            # a window that holds no valid steps must be NaN, not 0
            has_data = (
                series.notna()
                .astype(float)
                .rolling(pd.Timedelta(seconds=win_s), min_periods=1)
                .sum()
            )
            out[f"ofi_{band}_{win_name}"] = np.where(
                has_data.to_numpy() > 0, rolled.to_numpy(), np.nan
            )
    return out.reindex(df.index)


def fit_per_symbol(df: pd.DataFrame, **kwargs) -> dict[str, OFIEstimator]:
    """Fit one estimator per symbol from a multi-symbol frame."""
    out: dict[str, OFIEstimator] = {}
    for symbol, group in df.groupby("symbol", sort=False):
        try:
            out[str(symbol)] = OFIEstimator(**kwargs).fit(group)
        except ValueError as exc:
            logger.warning("skipping %s: %s", symbol, exc)
    return out


def compute_multilevel_ofi(
    df: pd.DataFrame,
    estimators: dict[str, OFIEstimator] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``ofi_*`` feature columns appended.

    If ``estimators`` is None they are fitted on ``df`` itself (in-sample —
    fine for exploration; use a train/eval split for honest IC numbers, as
    the CLI does).
    """
    if estimators is None:
        estimators = fit_per_symbol(df)

    out = df.copy()
    for col in MULTILEVEL_OFI_FEATURES:
        out[col] = np.nan

    groups = (
        out.groupby("symbol", sort=False).indices.items()
        if "symbol" in out.columns
        else [(None, np.arange(len(out)))]
    )
    for symbol, pos in groups:
        est = estimators.get(str(symbol)) if symbol is not None else (
            next(iter(estimators.values()), None)
        )
        sub = out.iloc[pos]
        max_step = est.max_step_s if est else 1.0
        sums = _band_window_sums(sub, max_step_s=max_step)
        for col in sums.columns:
            out.iloc[pos, out.columns.get_loc(col)] = sums[col].to_numpy()
        if est is None:
            continue
        for win_name in WINDOWS:
            bands = np.column_stack(
                [sums[f"ofi_{band}_{win_name}"].to_numpy() for band in BANDS]
            )
            integrated = (bands / est.scales_) @ est.weights_
            out.iloc[pos, out.columns.get_loc(f"ofi_int_{win_name}")] = integrated

    return out


# ── Validation CLI ────────────────────────────────────────────────────────


def _spearman_ic(feature: pd.Series, fwd: pd.Series) -> float:
    mask = feature.notna() & fwd.notna()
    if mask.sum() < 100 or feature[mask].nunique() < 2:
        return float("nan")
    return float(feature[mask].corr(fwd[mask], method="spearman"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit OFI integration weights on the first part of a day, "
        "evaluate IC out-of-sample on the rest."
    )
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--fit-frac", type=float, default=0.7)
    parser.add_argument("--save-table", help="Write fitted weights JSON here")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster_pipeline.loader import load_parquet

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_parquet(
        args.data_dir,
        symbols=[args.symbol],
        start_date=args.date,
        end_date=args.date,
        columns=[
            "timestamp_ns", "symbol", "raw_midprice", "raw_spread",
            "imbalance_qty_l1", "raw_bid_depth_5", "raw_ask_depth_5",
            "raw_bid_depth_10", "raw_ask_depth_10",
        ],
    )
    if df.empty:
        logger.error("no data for %s on %s", args.symbol, args.date)
        return 1
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    split = int(len(df) * args.fit_frac)
    fit_df, eval_df = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
    logger.info("fit on %d rows, evaluate on %d rows", len(fit_df), len(eval_df))

    est = OFIEstimator().fit(fit_df)
    print("\n— Integration weights (bands l1/d5/d10) —")
    print(dict(zip(BANDS, np.round(est.weights_, 4))))

    if args.save_table:
        Path(args.save_table).write_text(json.dumps(est.to_dict()))
        logger.info("weights saved to %s", args.save_table)

    out = compute_multilevel_ofi(eval_df, {args.symbol: est})
    print("\n— OOS NaN rates —")
    print(out[MULTILEVEL_OFI_FEATURES].isna().mean().round(4).to_string())

    mid = out["raw_midprice"]
    horizons = {"1s": 10, "5s": 50, "30s": 300, "1m": 600, "5m": 3000}
    print("\n— OOS Spearman IC vs forward log-returns (100ms grid) —")
    print(f"{'feature':<16}" + "".join(f"{h:>9}" for h in horizons))
    for feat in MULTILEVEL_OFI_FEATURES + ["imbalance_qty_l1"]:
        cells = [
            _spearman_ic(out[feat], np.log(mid.shift(-shift) / mid))
            for shift in horizons.values()
        ]
        print(
            f"{feat:<16}"
            + "".join(f"{c:>9.4f}" if np.isfinite(c) else f"{'—':>9}" for c in cells)
        )

    print(
        "\nThe CCZ claim under test: integrated OFI (ofi_int_*) should beat "
        "every single band at matched windows. imbalance_qty_l1 is the "
        "snapshot baseline — OFI is flow, so they measure different things; "
        "low correlation between them is good news for the ensemble."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
