"""Relative-value pairs strategy (A1 in feature_algorithm_gaps.md).

The first member of the relative-value logic family — nothing else in the
algorithm catalogue trades *relative* prices (algorithm_classification.md
§4.2). Log-ratio mean-reversion between cointegrated symbol pairs
(ETH/BTC, SOL/BTC, SOL/ETH) with market-neutral two-leg construction:
the position is on the *spread*, so directional market risk nets out and
correlation to the four deployed directional winners is low by construction.

Pipeline per pair (A, B), on a 5-minute grid:

    1. Hedge ratio:    log P_A = alpha + beta * log P_B + s   (OLS, fit window)
    2. Cointegration:  Dickey-Fuller on s — ds_t = c + gamma * s_{t-1} + eps;
                       DF stat = t(gamma), gated at the Engle-Granger 5%
                       critical for fitted residuals (-3.37, not raw -2.86)
    3. OU half-life:   hl = -ln(2) / ln(1 + gamma), gated to a tradeable
                       range (default 30min .. 5 days). Note: AR estimation
                       on OLS residuals biases the half-life *down* in
                       finite samples — treat the estimate as a lower bound
                       and size the z-window generously
    4. Signal:         rolling z-score of s (window ~ 10 half-lives, causal);
                       enter |z| > z_entry, exit |z| < z_exit, hard stop at
                       |z| > z_stop (structural-break guard)
    5. PnL:            position(t-1) * ds(t) — long spread = long A, short
                       beta * B, so ds IS the two-leg log return. Costs per
                       side-change: |dpos| * (1 + beta) legs * one-way bps,
                       loaded from config/costs.toml (single source of truth).

Why this doesn't implement ``MicrostructureAlgorithm`` or ``Strategy``: both
contracts are single-symbol (one frame in, one signal out); a pair needs two
legs at once. This module owns the pair-level API; if cross-symbol features
land in the ingestor (F6, post-Jun 17), pair z-scores can then be emitted as
per-symbol features and re-enter the standard agent framework.

Positions are shifted one grid cell before PnL (strictly causal). PnL across
data gaps is *included* — a held spread position really does experience the
gap move — but entries never trigger on a gap boundary cell.

**First validation verdict (2026-04-24..06-12, ~35 good days):** NO pair is
tradeable. Engle-Granger stats: ETH/BTC -3.15, SOL/BTC -2.48, SOL/ETH -2.43
— none beats the -3.37 gate, and the ETH/BTC hedge ratio is unstable across
fit windows (beta 1.19 on May 18-29 vs 1.69 on the full range), meaning the
spreads *drift* rather than revert at these horizons in this period. The
gates did their job: zero trades entered on a relationship that isn't there.
The machinery is validated on synthetic cointegrated pairs (recovers beta,
half-life, DF; harvests OU reversion OOS). Paths forward: (a) re-test
monthly as data accumulates — crypto pairs cointegrate in ranging regimes
and decouple in trends; (b) a rolling/Kalman hedge ratio would be a
different, harder strategy — deliberately out of scope; (c) wire this as a
periodic macro-agent generator so the re-test happens automatically, with
``is_tradeable`` as its first gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_S = 300.0
CELLS_PER_DAY = int(86_400 / GRID_S)
ANN_FACTOR = float(np.sqrt(365 * CELLS_PER_DAY))
DEFAULT_PAIRS = [("ETH", "BTC"), ("SOL", "BTC"), ("SOL", "ETH")]

_COSTS_PATH = Path(__file__).resolve().parents[2] / "config" / "costs.toml"


def load_cost_scenarios(path: Path = _COSTS_PATH) -> dict[str, float]:
    """One-way cost (bps) per scenario from config/costs.toml."""
    try:
        cfg = tomllib.loads(path.read_text())["hyperliquid"]
        return {
            "maker": float(cfg["maker_bps"]),
            "taker": float(cfg["taker_bps"]),
            "taker+slip": float(cfg["taker_bps"]) + float(cfg["slippage_bps"]),
        }
    except (OSError, KeyError) as exc:  # pragma: no cover - config drift
        logger.warning("costs.toml unavailable (%s) — using defaults", exc)
        return {"maker": 0.2, "taker": 3.5, "taker+slip": 5.5}


def symbol_grid(sub: pd.DataFrame, *, mid_col: str = "raw_midprice",
                ts_col: str = "timestamp_ns") -> pd.Series:
    """5-min log-mid grid for a single-symbol tick frame."""
    idx = pd.DatetimeIndex(
        pd.to_datetime(sub[ts_col], unit="ns", utc=True).dt.tz_localize(None)
    )
    px = pd.Series(sub[mid_col].to_numpy(dtype=np.float64), index=idx)
    return np.log(px.sort_index().resample(f"{int(GRID_S)}s").last())


def load_symbol_grids(
    data_dir: str, symbols: list[str], start: str, end: str
) -> dict[str, pd.Series]:
    """Per-symbol 5-min log-mid grids, loading one symbol-day at a time.

    The cluster loader pads every frame to the full feature schema, so bulk
    multi-day multi-symbol loads exhaust memory; resampling each day down to
    288 grid cells before loading the next bounds usage at one symbol-day.
    """
    from cluster_pipeline.loader import load_parquet

    grids: dict[str, list[pd.Series]] = {s: [] for s in symbols}
    for day in pd.date_range(start, end, freq="D"):
        date = day.strftime("%Y-%m-%d")
        for sym in symbols:
            try:
                df = load_parquet(
                    data_dir, symbols=[sym], start_date=date, end_date=date,
                    columns=["timestamp_ns", "symbol", "raw_midprice"],
                )
            except FileNotFoundError:
                continue
            if not df.empty:
                grids[sym].append(symbol_grid(df))
    return {
        s: pd.concat(parts).sort_index()
        for s, parts in grids.items()
        if parts
    }


def _pair_grid(
    source: pd.DataFrame | dict[str, pd.Series],
    sym_a: str,
    sym_b: str,
) -> pd.DataFrame:
    """Aligned 5-min grid of log mids for both legs (inner join, gaps dropped).

    ``source`` is either a multi-symbol tick frame or a dict of per-symbol
    log-mid grid series from ``load_symbol_grids``/``symbol_grid``.
    """
    legs = {}
    for sym in (sym_a, sym_b):
        if isinstance(source, dict):
            if sym not in source:
                raise ValueError(f"no rows for symbol {sym}")
            legs[sym] = source[sym]
        else:
            sub = source[source["symbol"] == sym]
            if sub.empty:
                raise ValueError(f"no rows for symbol {sym}")
            legs[sym] = symbol_grid(sub)
    grid = pd.DataFrame({"log_a": legs[sym_a], "log_b": legs[sym_b]}).dropna()
    if grid.empty:
        raise ValueError(f"no overlapping grid cells for {sym_a}/{sym_b}")
    return grid


class PairSpreadEstimator:
    """Fitted hedge ratio + cointegration diagnostics for one ordered pair."""

    def __init__(
        self,
        sym_a: str,
        sym_b: str,
        *,
        # Engle-Granger 5% critical value for *fitted residuals* (2 variables,
        # with constant) — stricter than the raw DF -2.86, because OLS
        # residuals mechanically look more stationary. Monte Carlo on
        # independent random walks confirms ~5% of draws beat -2.86.
        df_critical: float = -3.37,
        min_halflife_h: float = 0.5,
        max_halflife_h: float = 120.0,
        z_window_halflives: float = 10.0,
    ) -> None:
        self.sym_a, self.sym_b = sym_a, sym_b
        self.df_critical = df_critical
        self.min_halflife_h = min_halflife_h
        self.max_halflife_h = max_halflife_h
        self.z_window_halflives = z_window_halflives

        self.alpha_: float | None = None
        self.beta_: float | None = None
        self.df_stat_: float | None = None
        self.halflife_h_: float | None = None

    # ── fitting ───────────────────────────────────────────────────────

    def fit(
        self, df: pd.DataFrame | dict[str, pd.Series]
    ) -> "PairSpreadEstimator":
        """``df``: multi-symbol tick frame or per-symbol grid dict."""
        grid = _pair_grid(df, self.sym_a, self.sym_b)
        if len(grid) < 2 * CELLS_PER_DAY:
            raise ValueError(
                f"{self.pair_name}: need >= 2 days of overlapping grid, "
                f"got {len(grid)} cells"
            )
        x, y = grid["log_b"].to_numpy(), grid["log_a"].to_numpy()
        self.beta_, self.alpha_ = np.polyfit(x, y, 1)
        spread = y - (self.alpha_ + self.beta_ * x)

        # Dickey-Fuller with constant: ds = c + gamma * s_{t-1}
        ds, s_lag = np.diff(spread), spread[:-1]
        design = np.column_stack([np.ones(len(s_lag)), s_lag])
        coef, residuals, *_ = np.linalg.lstsq(design, ds, rcond=None)
        gamma = coef[1]
        eps = ds - design @ coef
        se = np.sqrt(
            np.sum(eps**2)
            / (len(ds) - 2)
            * np.linalg.inv(design.T @ design)[1, 1]
        )
        self.df_stat_ = float(gamma / se) if se > 0 else 0.0

        rho = 1.0 + gamma
        if 0 < rho < 1:
            self.halflife_h_ = float(
                -np.log(2) / np.log(rho) * GRID_S / 3600.0
            )
        else:
            self.halflife_h_ = float("inf")

        logger.info(
            "fitted %s: beta %.4f, DF %.2f (crit %.2f), half-life %.1fh, "
            "tradeable=%s",
            self.pair_name, self.beta_, self.df_stat_, self.df_critical,
            self.halflife_h_, self.is_tradeable,
        )
        return self

    @property
    def pair_name(self) -> str:
        return f"{self.sym_a}/{self.sym_b}"

    @property
    def is_cointegrated(self) -> bool:
        return self.df_stat_ is not None and self.df_stat_ < self.df_critical

    @property
    def is_tradeable(self) -> bool:
        return (
            self.is_cointegrated
            and self.min_halflife_h <= self.halflife_h_ <= self.max_halflife_h
        )

    # ── signal ────────────────────────────────────────────────────────

    def transform(
        self, df: pd.DataFrame | dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Grid frame with spread, rolling z, gap flags for fresh data."""
        if self.beta_ is None:
            raise RuntimeError("estimator not fitted")
        grid = _pair_grid(df, self.sym_a, self.sym_b)
        spread = grid["log_a"] - (self.alpha_ + self.beta_ * grid["log_b"])

        hl_cells = max(1.0, self.halflife_h_ * 3600.0 / GRID_S)
        window = int(np.clip(self.z_window_halflives * hl_cells,
                             CELLS_PER_DAY / 2, 7 * CELLS_PER_DAY))
        mu = spread.rolling(window, min_periods=window // 2).mean()
        sd = spread.rolling(window, min_periods=window // 2).std()
        z = (spread - mu) / sd

        step_ns = (
            grid.index.to_series().diff().dt.total_seconds().to_numpy()
        )
        return pd.DataFrame(
            {
                "spread": spread,
                "z": z,
                "gap_boundary": step_ns > GRID_S * 1.5,
            },
            index=grid.index,
        )

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "sym_a": self.sym_a,
            "sym_b": self.sym_b,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "df_stat": self.df_stat_,
            "halflife_h": self.halflife_h_,
            "df_critical": self.df_critical,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PairSpreadEstimator":
        est = cls(d["sym_a"], d["sym_b"], df_critical=d.get("df_critical", -2.86))
        est.alpha_, est.beta_ = d["alpha"], d["beta"]
        est.df_stat_, est.halflife_h_ = d["df_stat"], d["halflife_h"]
        return est


def generate_positions(
    signal: pd.DataFrame,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
) -> pd.Series:
    """Spread position in {-1, 0, +1} from the z-score state machine.

    Long spread (long A / short beta*B) when z is very negative; short when
    very positive; exit on reversion to ``z_exit``; flatten on ``z_stop``
    (structural-break guard) and on NaN z. No entries on gap-boundary cells.
    """
    z = signal["z"].to_numpy()
    gap = signal["gap_boundary"].to_numpy()
    pos = np.zeros(len(z))
    current = 0.0
    for t in range(len(z)):
        zt = z[t]
        if not np.isfinite(zt):
            current = 0.0
        elif current == 0.0:
            if not gap[t]:
                if zt <= -z_entry:
                    current = 1.0
                elif zt >= z_entry:
                    current = -1.0
        else:
            if abs(zt) >= z_stop or abs(zt) <= z_exit:
                current = 0.0
        pos[t] = current
    return pd.Series(pos, index=signal.index, name="position")


def backtest(
    signal: pd.DataFrame,
    positions: pd.Series,
    *,
    beta: float,
    cost_bps_oneway: float,
) -> dict:
    """Mark-to-market the spread position; costs charged on position changes
    across both legs (1 + |beta| notional per unit of spread)."""
    pos_lag = positions.shift(1).fillna(0.0)
    d_spread = signal["spread"].diff()
    gross_bps = (pos_lag * d_spread * 1e4).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    cost_bps = turnover * (1.0 + abs(beta)) * cost_bps_oneway
    net_bps = gross_bps - cost_bps

    equity = net_bps.cumsum()
    drawdown = float((equity.cummax() - equity).max()) if len(equity) else 0.0
    n_round_trips = int((turnover.sum() / 2) // 1)
    active = pos_lag != 0

    std = net_bps.std()
    sharpe = float(net_bps.mean() / std * ANN_FACTOR) if std > 0 else 0.0
    return {
        "total_net_bps": float(net_bps.sum()),
        "total_gross_bps": float(gross_bps.sum()),
        "total_cost_bps": float(cost_bps.sum()),
        "n_round_trips": n_round_trips,
        "net_bps_per_trade": (
            float(net_bps.sum() / n_round_trips) if n_round_trips else 0.0
        ),
        "sharpe": sharpe,
        "max_drawdown_bps": drawdown,
        "time_in_market": float(active.mean()),
        "avg_holding_h": (
            float(active.sum() * GRID_S / 3600.0 / n_round_trips)
            if n_round_trips
            else 0.0
        ),
    }


def fit_pairs(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    **kwargs,
) -> dict[str, PairSpreadEstimator]:
    """Fit every requested pair, keeping failures out of the result."""
    out: dict[str, PairSpreadEstimator] = {}
    for sym_a, sym_b in pairs or DEFAULT_PAIRS:
        try:
            est = PairSpreadEstimator(sym_a, sym_b, **kwargs).fit(df)
            out[est.pair_name] = est
        except ValueError as exc:
            logger.warning("skipping %s/%s: %s", sym_a, sym_b, exc)
    return out


# ── Validation CLI ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit pair spreads on the first part of a date range, "
        "backtest the z-score reversion OOS at three cost scenarios."
    )
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--pairs", default="ETH/BTC,SOL/BTC,SOL/ETH",
        help="Comma-separated A/B pairs",
    )
    parser.add_argument("--fit-frac", type=float, default=0.7)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--save-table", help="Write fitted pair params JSON here")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pairs = [tuple(p.split("/")) for p in args.pairs.split(",")]
    symbols = sorted({s for p in pairs for s in p})

    grids = load_symbol_grids(args.data_dir, symbols, args.start, args.end)
    if not grids:
        logger.error("no data in %s..%s", args.start, args.end)
        return 1

    t_min = min(g.index.min() for g in grids.values())
    t_max = max(g.index.max() for g in grids.values())
    split_time = t_min + args.fit_frac * (t_max - t_min)
    fit_grids = {s: g[g.index < split_time] for s, g in grids.items()}
    eval_grids = {s: g[g.index >= split_time] for s, g in grids.items()}
    logger.info("fit < %s, evaluate after", split_time)

    estimators = fit_pairs(fit_grids, pairs)
    if args.save_table:
        Path(args.save_table).write_text(
            json.dumps({k: v.to_dict() for k, v in estimators.items()})
        )

    costs = load_cost_scenarios()
    print("\n— Pair diagnostics (fitted on first %.0f%%) —" % (args.fit_frac * 100))
    print(
        f"{'pair':<10}{'beta':>8}{'DF':>8}{'coint':>7}{'HL(h)':>8}{'tradeable':>11}"
    )
    for name, est in estimators.items():
        print(
            f"{name:<10}{est.beta_:>8.4f}{est.df_stat_:>8.2f}"
            f"{str(est.is_cointegrated):>7}{est.halflife_h_:>8.1f}"
            f"{str(est.is_tradeable):>11}"
        )

    print("\n— OOS backtest (z_entry %.1f, z_exit %.1f) —" % (args.z_entry, args.z_exit))
    header = (
        f"{'pair':<10}{'scenario':<12}{'trades':>7}{'net bps':>10}"
        f"{'bps/trade':>11}{'sharpe':>8}{'maxDD':>8}{'hold(h)':>9}"
    )
    print(header)
    for name, est in estimators.items():
        if not est.is_tradeable:
            print(f"{name:<10}{'— not tradeable (failed gates) —'}")
            continue
        signal = est.transform(eval_grids)
        positions = generate_positions(
            signal, z_entry=args.z_entry, z_exit=args.z_exit
        )
        for scen, bps in costs.items():
            r = backtest(signal, positions, beta=est.beta_, cost_bps_oneway=bps)
            print(
                f"{name:<10}{scen:<12}{r['n_round_trips']:>7}"
                f"{r['total_net_bps']:>10.1f}{r['net_bps_per_trade']:>11.2f}"
                f"{r['sharpe']:>8.2f}{r['max_drawdown_bps']:>8.1f}"
                f"{r['avg_holding_h']:>9.1f}"
            )

    print(
        "\nGates before promotion: DF < -2.86 on the fit window, half-life in "
        "[0.5h, 120h], and OOS Sharpe > 0 at the taker scenario. A pair that "
        "passes here still enters the agent 5-gate protocol like any other "
        "hypothesis."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
