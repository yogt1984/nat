"""
Strategy Definitions for NAT Backtester

Strategies are defined as structured objects with:
- Entry conditions (when to open position)
- Exit conditions (when to close position)
- Risk parameters (stop loss, take profit, timeout)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional
import polars as pl


@dataclass
class Strategy:
    """
    Trading strategy definition.

    Attributes
    ----------
    name : str
        Unique identifier for the strategy
    entry_condition : Callable
        Function that takes DataFrame and returns boolean Series for entry signals
    exit_condition : Callable
        Function that takes DataFrame and returns boolean Series for exit signals
    stop_loss_pct : float
        Stop loss percentage (e.g., 2.0 means -2%)
    take_profit_pct : float
        Take profit percentage (e.g., 4.0 means +4%)
    max_holding_bars : int
        Maximum bars to hold position before forced exit
    direction : str
        "long" or "short"
    required_features : List[str]
        Features that must exist in data for this strategy
    description : str
        Human-readable description of the strategy logic
    """

    name: str
    entry_condition: Callable[[pl.DataFrame], pl.Series]
    exit_condition: Callable[[pl.DataFrame], pl.Series]
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_holding_bars: int = 600  # ~1 hour at 100ms
    direction: str = "long"
    required_features: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction}")
        if self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be positive, got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            raise ValueError(f"take_profit_pct must be positive, got {self.take_profit_pct}")
        if self.max_holding_bars <= 0:
            raise ValueError(f"max_holding_bars must be positive, got {self.max_holding_bars}")


def _safe_condition(df: pl.DataFrame, condition_expr) -> pl.Series:
    """Safely evaluate a condition, returning False for any errors."""
    try:
        result = condition_expr(df)
        # Handle nulls - treat as False
        if isinstance(result, pl.Series):
            return result.fill_null(False)
        return result
    except Exception:
        return pl.Series([False] * len(df))


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================


def accumulation_long() -> Strategy:
    """
    Accumulation Long Strategy

    Enter long when:
    - Regime shows accumulation (accumulation_score > 0.6)
    - Whales are buying (whale_flow_zscore > 1.5)
    - Price is in lower part of range (range_position < 0.3)

    Exit when:
    - Accumulation weakens (score < 0.3)
    - Price reaches upper range (position > 0.7)
    """

    def entry(df: pl.DataFrame) -> pl.Series:
        return (
            (df["accumulation_score"] > 0.6)
            & (df["whale_flow_zscore_1h"] > 1.5)
            & (df["range_position_24h"] < 0.3)
        ).fill_null(False)

    def exit(df: pl.DataFrame) -> pl.Series:
        return (
            (df["accumulation_score"] < 0.3)
            | (df["range_position_24h"] > 0.7)
        ).fill_null(False)

    return Strategy(
        name="accumulation_long",
        entry_condition=entry,
        exit_condition=exit,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_bars=600,
        direction="long",
        required_features=[
            "accumulation_score",
            "whale_flow_zscore_1h",
            "range_position_24h",
        ],
        description="Long when accumulation detected with whale buying at range lows",
    )


def distribution_short() -> Strategy:
    """
    Distribution Short Strategy

    Enter short when:
    - Regime shows distribution (distribution_score > 0.6)
    - Whales are selling (whale_flow_zscore < -1.5)
    - Price is in upper part of range (range_position > 0.7)

    Exit when:
    - Distribution weakens (score < 0.3)
    - Price reaches lower range (position < 0.3)
    """

    def entry(df: pl.DataFrame) -> pl.Series:
        return (
            (df["distribution_score"] > 0.6)
            & (df["whale_flow_zscore_1h"] < -1.5)
            & (df["range_position_24h"] > 0.7)
        ).fill_null(False)

    def exit(df: pl.DataFrame) -> pl.Series:
        return (
            (df["distribution_score"] < 0.3)
            | (df["range_position_24h"] < 0.3)
        ).fill_null(False)

    return Strategy(
        name="distribution_short",
        entry_condition=entry,
        exit_condition=exit,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_bars=600,
        direction="short",
        required_features=[
            "distribution_score",
            "whale_flow_zscore_1h",
            "range_position_24h",
        ],
        description="Short when distribution detected with whale selling at range highs",
    )


def entropy_breakout() -> Strategy:
    """
    Entropy Breakout Strategy

    Enter long when:
    - Entropy drops significantly (predictable market)
    - High absorption (volume absorbed without price move)
    - Positive whale flow

    This targets breakouts after consolidation periods.
    """

    def entry(df: pl.DataFrame) -> pl.Series:
        return (
            (df["tick_entropy_1m"] < 0.5)
            & (df["absorption_zscore"] > 2.0)
            & (df["whale_net_flow_1h"] > 0)
        ).fill_null(False)

    def exit(df: pl.DataFrame) -> pl.Series:
        return (
            (df["tick_entropy_1m"] > 0.8)
            | (df["absorption_zscore"] < 0.5)
        ).fill_null(False)

    return Strategy(
        name="entropy_breakout",
        entry_condition=entry,
        exit_condition=exit,
        stop_loss_pct=1.5,
        take_profit_pct=3.0,
        max_holding_bars=300,  # Shorter hold for breakouts
        direction="long",
        required_features=[
            "tick_entropy_1m",
            "absorption_zscore",
            "whale_net_flow_1h",
        ],
        description="Long on entropy drop with high absorption and whale buying",
    )


def regime_momentum() -> Strategy:
    """
    Regime Momentum Strategy

    Enter long when:
    - Clear regime detected (clarity > 0.7)
    - Positive momentum indicated by regime
    - Hurst exponent shows trending behavior

    Conservative strategy that waits for regime confirmation.
    """

    def entry(df: pl.DataFrame) -> pl.Series:
        return (
            (df["regime_clarity"] > 0.7)
            & (df["accumulation_score"] > df["distribution_score"])
            & (df["trend_hurst_300"] > 0.55)
        ).fill_null(False)

    def exit(df: pl.DataFrame) -> pl.Series:
        return (
            (df["regime_clarity"] < 0.4)
            | (df["distribution_score"] > df["accumulation_score"])
            | (df["trend_hurst_300"] < 0.45)
        ).fill_null(False)

    return Strategy(
        name="regime_momentum",
        entry_condition=entry,
        exit_condition=exit,
        stop_loss_pct=2.5,
        take_profit_pct=5.0,
        max_holding_bars=900,
        direction="long",
        required_features=[
            "regime_clarity",
            "accumulation_score",
            "distribution_score",
            "trend_hurst_300",
        ],
        description="Long on clear bullish regime with trending behavior",
    )


def whale_flow_simple() -> Strategy:
    """
    Simple Whale Flow Strategy

    The simplest possible strategy based on whale flow.
    Enter long when whale flow z-score is strongly positive.
    Exit when it reverses.

    This is a baseline to compare more complex strategies against.
    """

    def entry(df: pl.DataFrame) -> pl.Series:
        return (df["whale_flow_zscore_1h"] > 2.0).fill_null(False)

    def exit(df: pl.DataFrame) -> pl.Series:
        return (df["whale_flow_zscore_1h"] < 0.0).fill_null(False)

    return Strategy(
        name="whale_flow_simple",
        entry_condition=entry,
        exit_condition=exit,
        stop_loss_pct=3.0,
        take_profit_pct=6.0,
        max_holding_bars=1200,
        direction="long",
        required_features=["whale_flow_zscore_1h"],
        description="Simple long on strong whale buying, exit on reversal",
    )


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================


def get_all_strategies() -> dict:
    """Return dictionary of all available strategies."""
    return {
        "accumulation_long": accumulation_long(),
        "distribution_short": distribution_short(),
        "entropy_breakout": entropy_breakout(),
        "regime_momentum": regime_momentum(),
        "whale_flow_simple": whale_flow_simple(),
    }


def get_strategy(name: str) -> Strategy:
    """Get strategy by name."""
    strategies = get_all_strategies()
    if name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return strategies[name]
