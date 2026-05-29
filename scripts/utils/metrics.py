"""Standardized performance metrics (single source of truth for Sharpe, etc.)."""

import numpy as np

# Crypto markets trade 24/7 (1440 minutes/day)
CRYPTO_BARS_PER_DAY = {
    "1min": 1440,
    "5min": 288,
    "15min": 96,
    "30min": 48,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}

# 10Hz emission = 10 bars/sec * 86400 sec/day
BARS_PER_DAY_10HZ = 864_000


def bars_per_day_for_timeframe(timeframe: str) -> int:
    """Return bars per 24h day for a timeframe string (crypto, 24/7)."""
    if timeframe not in CRYPTO_BARS_PER_DAY:
        raise ValueError(
            f"Unknown timeframe '{timeframe}', "
            f"expected one of {list(CRYPTO_BARS_PER_DAY)}"
        )
    return CRYPTO_BARS_PER_DAY[timeframe]


def annualized_sharpe(
    pnl: np.ndarray,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio from a PnL series.

    Uses sample standard deviation (ddof=1) for unbiased estimation.

    Args:
        pnl: Array of PnL values (daily, intraday bars, etc.)
        periods_per_year: Number of periods per year matching the PnL frequency.
            Common values:
            - 252          for daily PnL
            - 252 * 96     for 15-min bars (96 bars/day)
            - 252 * 24     for 1-hour bars
        risk_free_rate: Annual risk-free rate (e.g. 0.05 for 5%).  Subtracted
            per-period before computing the ratio.

    Returns:
        Annualized Sharpe ratio, or 0.0 if std is zero or len < 2.
    """
    arr = np.asarray(pnl, dtype=float)
    if len(arr) < 2:
        return 0.0
    if risk_free_rate != 0.0:
        arr = arr - risk_free_rate / periods_per_year
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(periods_per_year))


def sharpe_daily(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from daily PnL (252 trading days/year)."""
    return annualized_sharpe(daily_pnl, periods_per_year=252.0)


def sharpe_intraday(pnl: np.ndarray, bars_per_day: float) -> float:
    """Annualized Sharpe from intraday bar PnL.

    Args:
        pnl: Per-bar PnL array
        bars_per_day: Number of bars in one trading day (e.g. 96 for 15-min)
    """
    return annualized_sharpe(pnl, periods_per_year=252.0 * bars_per_day)
