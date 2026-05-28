"""
Portfolio Assembly (Alpha Roadmap Step 7).

Runs Steps 1-6 per symbol (BTC, ETH, SOL), then combines using
risk parity weights with correlation adjustment and drawdown control.

Quality Gate G7:
  - Portfolio Sharpe > max(individual symbol Sharpes)
  - Portfolio max DD < 80% of worst individual DD

Usage:
    python -m alpha.portfolio --data data/features
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbolAllocation:
    """Per-symbol allocation detail."""
    symbol: str
    raw_weight: float  # from inverse volatility
    adjusted_weight: float  # after correlation adjustment
    volatility: float  # annualized vol
    sharpe: float
    max_dd: float


@dataclass
class DrawdownControl:
    """Drawdown control state."""
    current_dd: float
    dd_threshold_reduce: float  # reduce at this DD level
    dd_threshold_recover: float  # recover at this DD level
    is_reduced: bool
    scale_factor: float  # 1.0 normal, 0.5 reduced


@dataclass
class PortfolioResult:
    """Output of portfolio assembly."""
    symbols: List[str]
    allocations: List[SymbolAllocation]
    correlation_matrix: Dict[str, Dict[str, float]]
    portfolio_sharpe: float
    portfolio_max_dd: float
    max_individual_sharpe: float
    worst_individual_dd: float
    drawdown_control: DrawdownControl
    gate_sharpe_improves: bool
    gate_dd_improves: bool
    gate_pass: bool


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


def compute_risk_parity_weights(
    returns: Dict[str, np.ndarray],
    lookback: int = 2880,  # ~30 days at 15min
) -> Dict[str, float]:
    """
    Inverse-volatility risk parity weights.

    w_i = (1/vol_i) / sum(1/vol_j)
    """
    vols = {}
    for sym, ret in returns.items():
        if len(ret) < 10:
            vols[sym] = 1.0
        else:
            # Use recent lookback
            recent = ret[-lookback:] if len(ret) > lookback else ret
            vol = np.std(recent)
            vols[sym] = max(vol, 1e-10)

    inv_vols = {s: 1.0 / v for s, v in vols.items()}
    total = sum(inv_vols.values())

    return {s: iv / total for s, iv in inv_vols.items()}


def adjust_for_correlation(
    weights: Dict[str, float],
    returns: Dict[str, np.ndarray],
    corr_threshold: float = 0.8,
    reduction: float = 0.20,
) -> Dict[str, float]:
    """
    Reduce combined weight by 20% if any pair has corr > threshold.
    """
    symbols = list(weights.keys())
    n = len(symbols)

    # Compute pairwise correlations
    min_len = min(len(returns[s]) for s in symbols)
    if min_len < 30:
        return weights

    ret_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])
    corr_matrix = np.corrcoef(ret_matrix.T)

    # Check for high correlations
    high_corr = False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > corr_threshold:
                high_corr = True
                break

    if high_corr:
        adjusted = {s: w * (1 - reduction) for s, w in weights.items()}
        # Redistribute to maintain sum = 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {s: w / total for s, w in adjusted.items()}
        return adjusted

    return weights


def compute_correlation_matrix(
    returns: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise correlation matrix."""
    symbols = list(returns.keys())
    min_len = min(len(returns[s]) for s in symbols)
    if min_len < 10:
        return {s: {t: 0.0 for t in symbols} for s in symbols}

    ret_matrix = np.column_stack([returns[s][-min_len:] for s in symbols])
    corr = np.corrcoef(ret_matrix.T)

    result = {}
    for i, s in enumerate(symbols):
        result[s] = {}
        for j, t in enumerate(symbols):
            result[s][t] = float(corr[i, j])

    return result


# ---------------------------------------------------------------------------
# Drawdown control
# ---------------------------------------------------------------------------


def apply_drawdown_control(
    portfolio_pnl: np.ndarray,
    dd_reduce: float = 0.02,
    dd_recover: float = 0.01,
) -> DrawdownControl:
    """
    Monitor portfolio drawdown and compute scaling.

    Rules:
      - When DD > dd_reduce: scale positions to 50%
      - Recover to 100% when DD < dd_recover
    """
    cum = np.cumsum(portfolio_pnl)
    if len(cum) == 0:
        return DrawdownControl(
            current_dd=0.0, dd_threshold_reduce=dd_reduce,
            dd_threshold_recover=dd_recover, is_reduced=False, scale_factor=1.0,
        )

    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    current_dd = float(dd[-1])

    is_reduced = current_dd > dd_reduce
    scale_factor = 0.5 if is_reduced else 1.0

    return DrawdownControl(
        current_dd=current_dd,
        dd_threshold_reduce=dd_reduce,
        dd_threshold_recover=dd_recover,
        is_reduced=is_reduced,
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------------
# Portfolio PnL simulation
# ---------------------------------------------------------------------------


def simulate_portfolio(
    signals: Dict[str, np.ndarray],
    prices: Dict[str, np.ndarray],
    weights: Dict[str, float],
    dd_reduce: float = 0.02,
    dd_recover: float = 0.01,
) -> tuple:
    """
    Simulate weighted portfolio PnL with drawdown control.

    Returns:
        (portfolio_pnl, per_symbol_pnl, dd_control)
    """
    symbols = list(signals.keys())
    min_len = min(len(signals[s]) for s in symbols)

    per_symbol_pnl = {}
    for s in symbols:
        p = prices[s][:min_len]
        sig = signals[s][:min_len]
        ret = np.diff(p) / p[:-1]
        pnl = sig[:-1] * ret
        per_symbol_pnl[s] = pnl

    # Weight and combine
    port_pnl = np.zeros(min_len - 1)
    for s in symbols:
        port_pnl += weights[s] * per_symbol_pnl[s]

    # Apply drawdown control
    dd_control = apply_drawdown_control(port_pnl, dd_reduce, dd_recover)

    return port_pnl, per_symbol_pnl, dd_control


def _sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2 or np.std(pnl) < 1e-15:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 96))


def _max_dd(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_portfolio(
    data_dir: str = "data/features",
    symbols: Optional[List[str]] = None,
    timeframe: str = "15min",
    output: str = "reports/alpha_portfolio.json",
) -> PortfolioResult:
    """
    Full portfolio assembly pipeline.

    1. Load data per symbol
    2. Generate proxy signals per symbol
    3. Compute risk parity weights
    4. Adjust for correlation
    5. Simulate portfolio with drawdown control
    6. Evaluate G7 quality gate
    """
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars

    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]

    df_all = load_parquet(data_dir)

    signals = {}
    all_prices = {}
    returns = {}

    for sym in symbols:
        if "symbol" in df_all.columns:
            df_sym = df_all[df_all["symbol"] == sym].reset_index(drop=True)
        else:
            df_sym = df_all

        if len(df_sym) == 0:
            log.warning(f"No data for {sym}, skipping")
            continue

        bars = aggregate_bars(df_sym, timeframe=timeframe)
        bars_pd = bars.to_pandas() if hasattr(bars, "to_pandas") else bars

        # Price
        price_col = None
        for c in ["midprice_mean", "close", "mid_price"]:
            if c in bars_pd.columns:
                price_col = c
                break
        if price_col is None:
            continue

        p = bars_pd[price_col].values
        all_prices[sym] = p

        # Returns
        ret = np.zeros(len(p))
        ret[1:] = (p[1:] - p[:-1]) / p[:-1]
        returns[sym] = ret

        # Proxy signal: momentum
        sig = pd.Series(ret).rolling(20, min_periods=1).mean().values
        std = np.std(sig)
        if std > 1e-10:
            sig = np.clip(sig / (3 * std), -1, 1)
        signals[sym] = sig

    valid_symbols = list(signals.keys())
    if len(valid_symbols) < 2:
        log.warning(f"Only {len(valid_symbols)} symbols available, need >=2")
        return PortfolioResult(
            symbols=valid_symbols,
            allocations=[],
            correlation_matrix={},
            portfolio_sharpe=0.0, portfolio_max_dd=0.0,
            max_individual_sharpe=0.0, worst_individual_dd=0.0,
            drawdown_control=DrawdownControl(0, 0.02, 0.01, False, 1.0),
            gate_sharpe_improves=False, gate_dd_improves=False, gate_pass=False,
        )

    # Risk parity weights
    raw_weights = compute_risk_parity_weights(returns)
    adj_weights = adjust_for_correlation(raw_weights, returns)

    # Correlation matrix
    corr_matrix = compute_correlation_matrix(returns)

    # Per-symbol metrics
    allocations = []
    for sym in valid_symbols:
        p = all_prices[sym]
        sig = signals[sym]
        ret = np.diff(p) / p[:-1]
        pnl = sig[:-1] * ret

        alloc = SymbolAllocation(
            symbol=sym,
            raw_weight=raw_weights.get(sym, 0.0),
            adjusted_weight=adj_weights.get(sym, 0.0),
            volatility=float(np.std(returns[sym]) * np.sqrt(252 * 96)),
            sharpe=_sharpe(pnl),
            max_dd=_max_dd(pnl),
        )
        allocations.append(alloc)

    # Portfolio simulation
    port_pnl, per_sym_pnl, dd_control = simulate_portfolio(
        signals, all_prices, adj_weights,
    )

    port_sharpe = _sharpe(port_pnl)
    port_dd = _max_dd(port_pnl)
    max_ind_sharpe = max(a.sharpe for a in allocations)
    worst_ind_dd = max(a.max_dd for a in allocations)

    # G7 quality gate
    gate_sharpe = port_sharpe > max_ind_sharpe
    gate_dd = port_dd < 0.8 * worst_ind_dd if worst_ind_dd > 0 else False

    result = PortfolioResult(
        symbols=valid_symbols,
        allocations=allocations,
        correlation_matrix=corr_matrix,
        portfolio_sharpe=port_sharpe,
        portfolio_max_dd=port_dd,
        max_individual_sharpe=max_ind_sharpe,
        worst_individual_dd=worst_ind_dd,
        drawdown_control=dd_control,
        gate_sharpe_improves=gate_sharpe,
        gate_dd_improves=gate_dd,
        gate_pass=gate_sharpe and gate_dd,
    )

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    log.info(f"Portfolio: {valid_symbols}, S={port_sharpe:.2f}, DD={port_dd:.4f}, "
             f"G7={'PASS' if result.gate_pass else 'FAIL'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Portfolio assembly (Step 7)")
    parser.add_argument("--data", default="data/features")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--output", default="reports/alpha_portfolio.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_portfolio(
        data_dir=args.data, symbols=args.symbols,
        timeframe=args.timeframe, output=args.output,
    )
    gate = "PASS" if result.gate_pass else "FAIL"
    print(f"\nG7 Quality Gate: {gate}")
    print(f"  Portfolio Sharpe: {result.portfolio_sharpe:.3f}")
    print(f"  Best Individual:  {result.max_individual_sharpe:.3f}")
    print(f"  Portfolio MaxDD:  {result.portfolio_max_dd:.4f}")
    print(f"  Worst IndivDD:    {result.worst_individual_dd:.4f}")
    for a in result.allocations:
        print(f"  {a.symbol}: w={a.adjusted_weight:.2f}, S={a.sharpe:.3f}, vol={a.volatility:.2f}")


if __name__ == "__main__":
    main()
