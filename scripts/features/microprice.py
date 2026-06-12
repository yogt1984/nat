"""Microprice deviation features (F2 in feature_algorithm_gaps.md).

Implements the Stoikov micro-price (Stoikov 2018, "The Micro-Price: A High
Frequency Estimator of Future Prices", SSRN 2970694): the limit of expected
future mid-prices conditional on the order-book state (imbalance, spread),
estimated as a Markov chain over discretized states. The deviation
``microprice − mid`` is the fair-value anchor for maker-side designs (HF1 in
algorithm_candidates_literature.md).

Derived purely from existing parquet columns (``imbalance_qty_l1``,
``raw_spread_bps``, ``raw_midprice``, ``timestamp_ns``) — no ingestor or
schema changes.

Features (prefix ``mp_``):

    mp_micro_adj_bps     Stoikov adjustment g*(imbalance, spread): the
                         micro-price minus the mid, in bps. Primary feature.
    mp_wmid_dev_bps      Size-weighted mid minus mid, in bps — equals
                         (I − 0.5) * spread_bps. The naive baseline
                         (equivalent to the existing ``raw_microprice``).
    mp_micro_excess_bps  mp_micro_adj_bps − mp_wmid_dev_bps: what the Markov
                         structure adds beyond naive size weighting.

Estimation (per symbol, on the 100ms snapshot grid):

    State x = (imbalance bin, spread bin). One-step transitions split into
    B(x, x') — mid unchanged — and Q(x, x') — mid changed; K1(x) is the
    one-step expected mid change in bps. Then

        G1 = (I − B)^{-1} K1            (expected change up to 1st mid move)
        B* = (I − B)^{-1} Q             (embedded chain at mid-change events)
        g* = G1 + B* G1 + B*^2 G1 + ... (summed until convergence)

    Data are symmetrized (I → 1−I, ΔM → −ΔM) so g* is antisymmetric by
    construction. Steps spanning data gaps (> ``max_step_s``) are excluded.
    States with fewer than ``min_state_samples`` observations get a zero
    adjustment (conservative) rather than a noisy estimate.

    Bin-count sensitivity (BTC 2026-06-11, 70/30 split, OOS Spearman IC @ 1s):
    10 bins 0.515, 20 bins 0.532, 40 bins 0.542, vs continuous raw imbalance
    0.558 — finer bins recover the rank resolution that discretization costs.
    The feature's value over raw imbalance is not rank IC but *calibration*:
    g* is the expected mid move in bps conditional on state, which is exactly
    what a maker quote engine prices against. Default 20 bins balances
    resolution vs per-state sample counts on partial days.
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

MICROPRICE_FEATURES = [
    "mp_micro_adj_bps",
    "mp_wmid_dev_bps",
    "mp_micro_excess_bps",
]

_CHANGE_EPS_BPS = 1e-6


class MicropriceEstimator:
    """Per-symbol Stoikov micro-price adjustment table."""

    def __init__(
        self,
        *,
        n_imbalance_bins: int = 20,
        n_spread_bins: int = 3,
        max_levels: int = 6,
        tol_bps: float = 1e-4,
        min_state_samples: int = 50,
        max_step_s: float = 1.0,
    ) -> None:
        self.n_imbalance_bins = n_imbalance_bins
        self.n_spread_bins = n_spread_bins
        self.max_levels = max_levels
        self.tol_bps = tol_bps
        self.min_state_samples = min_state_samples
        self.max_step_s = max_step_s

        self.spread_edges_: np.ndarray | None = None  # internal bin edges
        self.g_: np.ndarray | None = None  # (n_imbalance_bins, n_spread_bins)
        self.state_counts_: np.ndarray | None = None
        self.symbol_: str = ""

    # ── fitting ───────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        *,
        imbalance_col: str = "imbalance_qty_l1",
        spread_col: str = "raw_spread_bps",
        mid_col: str = "raw_midprice",
        ts_col: str = "timestamp_ns",
    ) -> "MicropriceEstimator":
        """Estimate the adjustment table from a single-symbol frame."""
        if "symbol" in df.columns:
            symbols = df["symbol"].unique()
            if len(symbols) > 1:
                raise ValueError(
                    f"fit() expects a single symbol, got {list(symbols)} — "
                    "use fit_per_symbol() for multi-symbol frames"
                )
            self.symbol_ = str(symbols[0])

        sub = df[[ts_col, imbalance_col, spread_col, mid_col]].dropna()
        sub = sub.sort_values(ts_col)
        if len(sub) < 1000:
            raise ValueError(f"need >= 1000 valid rows to fit, got {len(sub)}")

        imb = ((sub[imbalance_col].to_numpy() + 1.0) / 2.0).clip(0.0, 1.0)
        spread = sub[spread_col].to_numpy(dtype=np.float64)
        mid = sub[mid_col].to_numpy(dtype=np.float64)
        ts_s = sub[ts_col].to_numpy(dtype=np.int64) / 1e9

        self.spread_edges_ = self._spread_edges(spread)
        n_s = len(self.spread_edges_) + 1
        n_i = self.n_imbalance_bins
        n_states = n_i * n_s

        i_bin = np.minimum((imb * n_i).astype(np.int64), n_i - 1)
        s_bin = np.searchsorted(self.spread_edges_, spread)
        state = i_bin * n_s + s_bin

        # One-step transitions, excluding steps that span data gaps
        dm_bps = (mid[1:] - mid[:-1]) / mid[:-1] * 1e4
        ok = (ts_s[1:] - ts_s[:-1]) <= self.max_step_s
        x, x_next, dm = state[:-1][ok], state[1:][ok], dm_bps[ok]

        # Symmetrize: mirror imbalance, negate mid changes, same spread bin
        def mirror(s: np.ndarray) -> np.ndarray:
            return (n_i - 1 - s // n_s) * n_s + s % n_s

        x = np.concatenate([x, mirror(x)])
        x_next = np.concatenate([x_next, mirror(x_next)])
        dm = np.concatenate([dm, -dm])

        changed = np.abs(dm) > _CHANGE_EPS_BPS
        B_cnt = np.zeros((n_states, n_states))
        Q_cnt = np.zeros((n_states, n_states))
        np.add.at(B_cnt, (x[~changed], x_next[~changed]), 1.0)
        np.add.at(Q_cnt, (x[changed], x_next[changed]), 1.0)
        K_sum = np.zeros(n_states)
        np.add.at(K_sum, x, dm)

        n_x = B_cnt.sum(axis=1) + Q_cnt.sum(axis=1)
        self.state_counts_ = n_x.copy()
        reliable = n_x >= self.min_state_samples
        safe_n = np.where(n_x > 0, n_x, 1.0)

        B = np.where(reliable[:, None], B_cnt / safe_n[:, None], 0.0)
        Q = np.where(reliable[:, None], Q_cnt / safe_n[:, None], 0.0)
        K1 = np.where(reliable, K_sum / safe_n, 0.0)

        ident = np.eye(n_states)
        G1 = np.linalg.solve(ident - B, K1)
        B_star = np.linalg.solve(ident - B, Q)

        g = G1.copy()
        term = G1
        for _ in range(self.max_levels):
            term = B_star @ term
            g += term
            if np.max(np.abs(term)) < self.tol_bps:
                break

        self.g_ = g.reshape(n_i, n_s)
        logger.info(
            "fitted %s: %d states (%d reliable), |g| max %.4f bps",
            self.symbol_ or "<unnamed>", n_states, int(reliable.sum()),
            float(np.abs(g).max()),
        )
        return self

    def _spread_edges(self, spread: np.ndarray) -> np.ndarray:
        """Internal spread-bin edges by quantile; degenerates gracefully when
        the spread is pinned (BTC sits at ~0.16 bps most of the day)."""
        qs = np.linspace(0, 1, self.n_spread_bins + 1)[1:-1]
        edges = np.unique(np.quantile(spread, qs))
        # Drop edges that don't actually split the data
        return edges[(edges > spread.min()) & (edges < spread.max())]

    # ── application ───────────────────────────────────────────────────

    def adjustment_bps(
        self, imbalance: np.ndarray, spread_bps: np.ndarray
    ) -> np.ndarray:
        """Look up g*(I, S) for symmetric imbalance in [-1, 1] and spread."""
        if self.g_ is None:
            raise RuntimeError("estimator not fitted")
        n_i, n_s = self.g_.shape
        out = np.full(len(imbalance), np.nan)
        valid = np.isfinite(imbalance) & np.isfinite(spread_bps)
        imb01 = ((imbalance[valid] + 1.0) / 2.0).clip(0.0, 1.0)
        i_bin = np.minimum((imb01 * n_i).astype(np.int64), n_i - 1)
        s_bin = np.searchsorted(self.spread_edges_, spread_bps[valid])
        s_bin = np.minimum(s_bin, n_s - 1)
        out[valid] = self.g_[i_bin, s_bin]
        return out

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol_,
            "n_imbalance_bins": self.n_imbalance_bins,
            "spread_edges": self.spread_edges_.tolist(),
            "g_bps": self.g_.tolist(),
            "state_counts": self.state_counts_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MicropriceEstimator":
        est = cls(n_imbalance_bins=d["n_imbalance_bins"])
        est.symbol_ = d.get("symbol", "")
        est.spread_edges_ = np.asarray(d["spread_edges"], dtype=np.float64)
        est.g_ = np.asarray(d["g_bps"], dtype=np.float64)
        est.state_counts_ = np.asarray(d["state_counts"], dtype=np.float64)
        return est


def fit_per_symbol(df: pd.DataFrame, **kwargs) -> dict[str, MicropriceEstimator]:
    """Fit one estimator per symbol from a multi-symbol frame."""
    out: dict[str, MicropriceEstimator] = {}
    for symbol, group in df.groupby("symbol", sort=False):
        try:
            out[str(symbol)] = MicropriceEstimator(**kwargs).fit(group)
        except ValueError as exc:
            logger.warning("skipping %s: %s", symbol, exc)
    return out


def compute_microprice(
    df: pd.DataFrame,
    estimators: dict[str, MicropriceEstimator] | None = None,
    *,
    imbalance_col: str = "imbalance_qty_l1",
    spread_col: str = "raw_spread_bps",
) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``mp_*`` feature columns appended.

    If ``estimators`` is None they are fitted on ``df`` itself (in-sample —
    fine for exploration; use a train/eval split for honest IC numbers, as
    the CLI does).
    """
    if estimators is None:
        estimators = fit_per_symbol(df)

    out = df.copy()
    imb = out[imbalance_col].to_numpy(dtype=np.float64)
    spread = out[spread_col].to_numpy(dtype=np.float64)

    out["mp_wmid_dev_bps"] = ((imb + 1.0) / 2.0 - 0.5) * spread
    adj = np.full(len(out), np.nan)
    if "symbol" in out.columns:
        for symbol, est in estimators.items():
            mask = (out["symbol"] == symbol).to_numpy()
            if mask.any():
                adj[mask] = est.adjustment_bps(imb[mask], spread[mask])
    elif estimators:
        est = next(iter(estimators.values()))
        adj = est.adjustment_bps(imb, spread)
    out["mp_micro_adj_bps"] = adj
    out["mp_micro_excess_bps"] = out["mp_micro_adj_bps"] - out["mp_wmid_dev_bps"]
    return out


# ── Validation CLI ────────────────────────────────────────────────────────


def _spearman_ic(feature: pd.Series, fwd: pd.Series) -> float:
    mask = feature.notna() & fwd.notna()
    if mask.sum() < 100 or feature[mask].nunique() < 2:
        return float("nan")
    return float(feature[mask].corr(fwd[mask], method="spearman"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit the Stoikov microprice on the first part of a day, "
        "evaluate IC out-of-sample on the rest."
    )
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--fit-frac", type=float, default=0.7)
    parser.add_argument("--save-table", help="Write fitted table JSON here")
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
            "timestamp_ns", "symbol", "raw_midprice",
            "raw_spread_bps", "imbalance_qty_l1",
        ],
    )
    if df.empty:
        logger.error("no data for %s on %s", args.symbol, args.date)
        return 1
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    split = int(len(df) * args.fit_frac)
    fit_df, eval_df = df.iloc[:split], df.iloc[split:].reset_index(drop=True)
    logger.info("fit on %d rows, evaluate on %d rows", len(fit_df), len(eval_df))

    est = MicropriceEstimator().fit(fit_df)

    print("\n— Adjustment table g* (bps), rows = imbalance bins low->high —")
    g_frame = pd.DataFrame(
        est.g_,
        index=[f"I~{(i + 0.5) / est.g_.shape[0]:.2f}" for i in range(est.g_.shape[0])],
        columns=[f"S{j}" for j in range(est.g_.shape[1])],
    )
    print(g_frame.round(4).to_string())

    if args.save_table:
        Path(args.save_table).write_text(json.dumps(est.to_dict()))
        logger.info("table saved to %s", args.save_table)

    out = compute_microprice(eval_df, {args.symbol: est})
    print("\n— OOS NaN rates —")
    print(out[MICROPRICE_FEATURES].isna().mean().round(4).to_string())

    mid = out["raw_midprice"]
    horizons = {"1s": 10, "5s": 50, "30s": 300, "1m": 600, "5m": 3000}
    print("\n— OOS Spearman IC vs forward log-returns (100ms grid) —")
    print(f"{'feature':<22}" + "".join(f"{h:>9}" for h in horizons))
    for feat in MICROPRICE_FEATURES + ["imbalance_qty_l1"]:
        cells = [
            _spearman_ic(out[feat], np.log(mid.shift(-shift) / mid))
            for shift in horizons.values()
        ]
        print(
            f"{feat:<22}"
            + "".join(f"{c:>9.4f}" if np.isfinite(c) else f"{'—':>9}" for c in cells)
        )

    print(
        "\nBaselines: mp_wmid_dev_bps is the naive size-weighted mid; "
        "imbalance_qty_l1 is the raw Spannung signal (IC~0.45 @ 1-5s). "
        "mp_micro_adj_bps should match or beat both at 1-30s, and "
        "mp_micro_excess_bps shows what the Markov structure adds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
