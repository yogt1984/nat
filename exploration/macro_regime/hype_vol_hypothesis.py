"""
HYPE Token Volatility-Buyback Hypothesis Testing

Thesis: Crypto volatility → Hyperliquid fees → Buybacks → HYPE price appreciation

This module provides tools to:
1. Estimate/forecast crypto volatility
2. Test the correlation chain
3. Generate trading signals
4. Backtest the strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# VOLATILITY ESTIMATION
# =============================================================================

class VolatilityEstimator:
    """
    Multiple methods for estimating and forecasting volatility.
    """

    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 24) -> float:
        """
        Simple realized volatility (annualized std of returns).

        Args:
            returns: Array of log returns
            window: Lookback window in periods

        Returns:
            Annualized volatility
        """
        if len(returns) < window:
            return np.nan

        recent = returns[-window:]
        # Assuming hourly data, annualize: sqrt(24 * 365)
        return np.std(recent) * np.sqrt(24 * 365)

    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int = 24) -> float:
        """
        Parkinson volatility estimator - more efficient than close-to-close.

        Uses high-low range which captures intraday volatility better.
        """
        if len(high) < window or len(low) < window:
            return np.nan

        h = high[-window:]
        l = low[-window:]

        # Parkinson formula
        hl_ratio = np.log(h / l) ** 2
        factor = 1 / (4 * np.log(2))
        daily_var = factor * np.mean(hl_ratio)

        # Annualize
        return np.sqrt(daily_var * 365)

    @staticmethod
    def garman_klass_volatility(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 24
    ) -> float:
        """
        Garman-Klass volatility - uses OHLC for better efficiency.
        """
        if len(close) < window:
            return np.nan

        o, h, l, c = open_[-window:], high[-window:], low[-window:], close[-window:]

        # GK formula
        term1 = 0.5 * (np.log(h / l) ** 2)
        term2 = (2 * np.log(2) - 1) * (np.log(c / o) ** 2)
        daily_var = np.mean(term1 - term2)

        return np.sqrt(daily_var * 365)

    @staticmethod
    def ewma_volatility(returns: np.ndarray, decay: float = 0.94) -> float:
        """
        Exponentially weighted moving average volatility.
        More responsive to recent changes.
        """
        if len(returns) < 2:
            return np.nan

        # Initialize with sample variance
        var = np.var(returns[:20]) if len(returns) >= 20 else np.var(returns)

        for r in returns:
            var = decay * var + (1 - decay) * r ** 2

        return np.sqrt(var * 24 * 365)

    @staticmethod
    def volatility_of_volatility(vol_series: np.ndarray, window: int = 30) -> float:
        """
        Vol of vol - useful for regime detection.
        High vol-of-vol suggests regime uncertainty.
        """
        if len(vol_series) < window:
            return np.nan

        return np.std(vol_series[-window:]) / np.mean(vol_series[-window:])


# =============================================================================
# CORRELATION CHAIN TESTING
# =============================================================================

@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    correlation: float
    p_value: float
    lag_days: int
    n_observations: int
    interpretation: str


class HypothesisTester:
    """
    Test each link in the hypothesis chain:
    Volatility → Volume → Fees → Buybacks → HYPE Price
    """

    def __init__(self):
        self.results: Dict[str, CorrelationResult] = {}

    def test_vol_to_volume(
        self,
        volatility: pd.Series,
        volume: pd.Series,
        max_lag: int = 7
    ) -> CorrelationResult:
        """
        Test: Does volatility predict trading volume?

        Expected: Positive correlation, possibly contemporaneous or vol leading.
        """
        from scipy import stats

        best_corr = 0
        best_lag = 0
        best_pval = 1

        for lag in range(max_lag + 1):
            if lag > 0:
                vol_shifted = volatility.shift(lag)
            else:
                vol_shifted = volatility

            # Align and drop NaN
            aligned = pd.concat([vol_shifted, volume], axis=1).dropna()
            if len(aligned) < 30:
                continue

            corr, pval = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                best_pval = pval

        interpretation = self._interpret_correlation(best_corr, best_pval, "vol→volume")

        result = CorrelationResult(
            correlation=best_corr,
            p_value=best_pval,
            lag_days=best_lag,
            n_observations=len(aligned),
            interpretation=interpretation,
        )

        self.results["vol_to_volume"] = result
        return result

    def test_volume_to_fees(
        self,
        volume: pd.Series,
        fees: pd.Series,
    ) -> CorrelationResult:
        """
        Test: Does volume correlate with fees?

        Expected: Very high positive correlation (almost definitional).
        """
        from scipy import stats

        aligned = pd.concat([volume, fees], axis=1).dropna()
        if len(aligned) < 10:
            return CorrelationResult(0, 1, 0, 0, "Insufficient data")

        corr, pval = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

        interpretation = self._interpret_correlation(corr, pval, "volume→fees")

        result = CorrelationResult(
            correlation=corr,
            p_value=pval,
            lag_days=0,
            n_observations=len(aligned),
            interpretation=interpretation,
        )

        self.results["volume_to_fees"] = result
        return result

    def test_fees_to_buybacks(
        self,
        fees: pd.Series,
        buybacks: pd.Series,
        max_lag: int = 14
    ) -> CorrelationResult:
        """
        Test: Do fees predict buybacks?

        Expected: Positive correlation with some lag (buyback timing).
        """
        from scipy import stats

        best_corr = 0
        best_lag = 0
        best_pval = 1

        for lag in range(max_lag + 1):
            fees_shifted = fees.shift(lag)
            aligned = pd.concat([fees_shifted, buybacks], axis=1).dropna()

            if len(aligned) < 20:
                continue

            corr, pval = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                best_pval = pval

        interpretation = self._interpret_correlation(best_corr, best_pval, "fees→buybacks")

        result = CorrelationResult(
            correlation=best_corr,
            p_value=best_pval,
            lag_days=best_lag,
            n_observations=len(aligned) if 'aligned' in dir() else 0,
            interpretation=interpretation,
        )

        self.results["fees_to_buybacks"] = result
        return result

    def test_buybacks_to_price(
        self,
        buybacks: pd.Series,
        hype_returns: pd.Series,
        max_lag: int = 7
    ) -> CorrelationResult:
        """
        Test: Do buybacks predict HYPE returns?

        Expected: Positive correlation, but market may front-run.
        """
        from scipy import stats

        best_corr = 0
        best_lag = 0
        best_pval = 1

        for lag in range(-3, max_lag + 1):  # Also check if market front-runs (negative lag)
            if lag > 0:
                buybacks_shifted = buybacks.shift(lag)
            elif lag < 0:
                buybacks_shifted = buybacks.shift(lag)  # Future buybacks
            else:
                buybacks_shifted = buybacks

            aligned = pd.concat([buybacks_shifted, hype_returns], axis=1).dropna()

            if len(aligned) < 20:
                continue

            corr, pval = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                best_pval = pval

        if best_lag < 0:
            interpretation = f"Market front-runs buybacks by {abs(best_lag)} days"
        else:
            interpretation = self._interpret_correlation(best_corr, best_pval, "buybacks→price")

        result = CorrelationResult(
            correlation=best_corr,
            p_value=best_pval,
            lag_days=best_lag,
            n_observations=len(aligned) if 'aligned' in dir() else 0,
            interpretation=interpretation,
        )

        self.results["buybacks_to_price"] = result
        return result

    def test_end_to_end(
        self,
        volatility: pd.Series,
        hype_returns: pd.Series,
        max_lag: int = 14
    ) -> CorrelationResult:
        """
        Test: Does volatility predict HYPE returns directly?

        This is the ultimate test of the thesis.
        """
        from scipy import stats

        best_corr = 0
        best_lag = 0
        best_pval = 1

        for lag in range(max_lag + 1):
            vol_shifted = volatility.shift(lag)
            aligned = pd.concat([vol_shifted, hype_returns], axis=1).dropna()

            if len(aligned) < 30:
                continue

            corr, pval = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                best_pval = pval

        interpretation = self._interpret_correlation(best_corr, best_pval, "vol→HYPE returns")

        result = CorrelationResult(
            correlation=best_corr,
            p_value=best_pval,
            lag_days=best_lag,
            n_observations=len(aligned) if 'aligned' in dir() else 0,
            interpretation=interpretation,
        )

        self.results["end_to_end"] = result
        return result

    def _interpret_correlation(self, corr: float, pval: float, link: str) -> str:
        """Generate interpretation text"""
        if pval > 0.05:
            return f"{link}: No statistically significant relationship (p={pval:.3f})"

        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        direction = "positive" if corr > 0 else "negative"

        return f"{link}: {strength.title()} {direction} correlation (r={corr:.3f}, p={pval:.3f})"

    def summary_report(self) -> str:
        """Generate summary of all tested links"""
        lines = [
            "=" * 60,
            "HYPOTHESIS CHAIN TEST RESULTS",
            "=" * 60,
            "",
            "Thesis: Volatility → Volume → Fees → Buybacks → HYPE Price",
            "",
        ]

        chain_links = [
            ("vol_to_volume", "Link 1: Volatility → Volume"),
            ("volume_to_fees", "Link 2: Volume → Fees"),
            ("fees_to_buybacks", "Link 3: Fees → Buybacks"),
            ("buybacks_to_price", "Link 4: Buybacks → HYPE Price"),
            ("end_to_end", "End-to-End: Volatility → HYPE Returns"),
        ]

        for key, title in chain_links:
            if key in self.results:
                r = self.results[key]
                lines.append(f"{title}")
                lines.append(f"  Correlation: {r.correlation:+.3f}")
                lines.append(f"  P-value:     {r.p_value:.4f}")
                lines.append(f"  Optimal lag: {r.lag_days} days")
                lines.append(f"  N:           {r.n_observations}")
                lines.append(f"  → {r.interpretation}")
                lines.append("")
            else:
                lines.append(f"{title}: NOT TESTED")
                lines.append("")

        # Overall verdict
        lines.append("-" * 60)
        lines.append("VERDICT")
        lines.append("-" * 60)

        if "end_to_end" in self.results:
            e2e = self.results["end_to_end"]
            if e2e.p_value < 0.05 and e2e.correlation > 0.2:
                lines.append("✓ Thesis SUPPORTED: Vol predicts HYPE returns")
                lines.append(f"  Optimal entry: {e2e.lag_days} days after vol spike")
            elif e2e.p_value < 0.05 and e2e.correlation < -0.2:
                lines.append("✗ Thesis REJECTED: Vol negatively correlates with HYPE")
                lines.append("  High vol periods may be bad for HYPE")
            else:
                lines.append("? Thesis INCONCLUSIVE: Weak or no relationship")
                lines.append("  Need more data or refined hypothesis")

        return "\n".join(lines)


# =============================================================================
# TRADING SIGNAL GENERATION
# =============================================================================

class VolRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class TradingSignal:
    """Trading signal based on volatility regime"""
    timestamp: datetime
    vol_regime: VolRegime
    current_vol: float
    vol_percentile: float
    signal: str  # "long", "hold", "close"
    confidence: float
    suggested_leverage: float
    reasoning: str


class SignalGenerator:
    """
    Generate trading signals based on volatility regime.
    """

    def __init__(
        self,
        vol_threshold_high: float = 0.7,      # Percentile
        vol_threshold_extreme: float = 0.9,
        vol_threshold_low: float = 0.3,
        min_hold_periods: int = 24,           # Minimum holding period (hours)
    ):
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_extreme = vol_threshold_extreme
        self.vol_threshold_low = vol_threshold_low
        self.min_hold_periods = min_hold_periods

        self.vol_history: List[float] = []
        self.position_open_time: Optional[datetime] = None
        self.current_position: str = "flat"

    def update_volatility(self, vol: float, timestamp: datetime = None) -> TradingSignal:
        """
        Update volatility reading and generate signal.
        """
        timestamp = timestamp or datetime.now()
        self.vol_history.append(vol)

        # Compute percentile
        if len(self.vol_history) < 30:
            percentile = 0.5  # Not enough data
        else:
            percentile = np.sum(np.array(self.vol_history) <= vol) / len(self.vol_history)

        # Determine regime
        if percentile >= self.vol_threshold_extreme:
            regime = VolRegime.EXTREME
        elif percentile >= self.vol_threshold_high:
            regime = VolRegime.HIGH
        elif percentile <= self.vol_threshold_low:
            regime = VolRegime.LOW
        else:
            regime = VolRegime.NORMAL

        # Generate signal
        signal, confidence, leverage, reasoning = self._generate_signal(
            regime, percentile, timestamp
        )

        return TradingSignal(
            timestamp=timestamp,
            vol_regime=regime,
            current_vol=vol,
            vol_percentile=percentile,
            signal=signal,
            confidence=confidence,
            suggested_leverage=leverage,
            reasoning=reasoning,
        )

    def _generate_signal(
        self,
        regime: VolRegime,
        percentile: float,
        timestamp: datetime
    ) -> Tuple[str, float, float, str]:
        """
        Generate signal based on regime.

        Strategy logic:
        - HIGH vol: Enter long (vol → fees → buybacks)
        - EXTREME vol: Reduce leverage (liquidation risk)
        - LOW vol: Close or don't enter (thesis not active)
        - NORMAL: Hold if already in, don't enter new
        """
        # Check minimum hold period
        if self.position_open_time:
            hours_held = (timestamp - self.position_open_time).total_seconds() / 3600
            can_close = hours_held >= self.min_hold_periods
        else:
            can_close = True

        if regime == VolRegime.HIGH:
            if self.current_position == "flat":
                self.current_position = "long"
                self.position_open_time = timestamp
                return (
                    "long",
                    0.7,
                    1.5,
                    "HIGH vol regime: Enter long. Elevated vol → more fees → buybacks expected."
                )
            else:
                return (
                    "hold",
                    0.7,
                    1.5,
                    "HIGH vol regime: Maintain long position."
                )

        elif regime == VolRegime.EXTREME:
            if self.current_position == "long":
                return (
                    "hold",
                    0.5,
                    1.0,  # Reduce leverage!
                    "EXTREME vol: Reduce leverage to avoid liquidation. Thesis still valid but risk elevated."
                )
            else:
                return (
                    "hold",
                    0.4,
                    1.0,
                    "EXTREME vol: Don't enter new position. Liquidation risk too high."
                )

        elif regime == VolRegime.LOW:
            if self.current_position == "long" and can_close:
                self.current_position = "flat"
                self.position_open_time = None
                return (
                    "close",
                    0.6,
                    0.0,
                    "LOW vol regime: Close position. Thesis inactive - low fees expected."
                )
            elif self.current_position == "long":
                return (
                    "hold",
                    0.4,
                    1.0,
                    f"LOW vol but minimum hold period not met. Wait {self.min_hold_periods - hours_held:.0f}h."
                )
            else:
                return (
                    "hold",
                    0.5,
                    0.0,
                    "LOW vol regime: Stay flat. Wait for vol expansion."
                )

        else:  # NORMAL
            if self.current_position == "long":
                return (
                    "hold",
                    0.5,
                    1.2,
                    "NORMAL vol: Maintain position with moderate leverage."
                )
            else:
                return (
                    "hold",
                    0.5,
                    0.0,
                    "NORMAL vol: Stay flat. Wait for HIGH vol entry signal."
                )


# =============================================================================
# BACKTEST FRAMEWORK
# =============================================================================

@dataclass
class BacktestResult:
    """Backtest performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_trade_duration_hours: float
    vol_regime_accuracy: Dict[str, float]  # Return by regime


class Backtester:
    """
    Backtest the volatility-based HYPE trading strategy.
    """

    def __init__(
        self,
        vol_data: pd.Series,
        hype_returns: pd.Series,
        signal_generator: SignalGenerator = None,
    ):
        self.vol_data = vol_data
        self.hype_returns = hype_returns
        self.signal_generator = signal_generator or SignalGenerator()

        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def run(self, leverage: float = 1.5) -> BacktestResult:
        """
        Run backtest.

        Args:
            leverage: Base leverage multiplier

        Returns:
            BacktestResult with performance metrics
        """
        # Align data
        aligned = pd.concat([self.vol_data, self.hype_returns], axis=1).dropna()
        aligned.columns = ["vol", "returns"]

        position = 0.0
        equity = 1.0
        self.equity_curve = [1.0]

        current_trade = None
        regime_returns = {r.value: [] for r in VolRegime}

        for idx, row in aligned.iterrows():
            vol = row["vol"]
            ret = row["returns"]

            # Generate signal
            signal = self.signal_generator.update_volatility(vol, idx)

            # Track regime returns
            if position > 0:
                regime_returns[signal.vol_regime.value].append(ret * position * leverage)

            # Execute signal
            if signal.signal == "long" and position == 0:
                position = 1.0
                current_trade = {
                    "entry_time": idx,
                    "entry_equity": equity,
                    "regime": signal.vol_regime.value,
                }

            elif signal.signal == "close" and position > 0:
                position = 0.0
                if current_trade:
                    current_trade["exit_time"] = idx
                    current_trade["exit_equity"] = equity
                    current_trade["return"] = equity / current_trade["entry_equity"] - 1
                    current_trade["duration_hours"] = (
                        idx - current_trade["entry_time"]
                    ).total_seconds() / 3600
                    self.trades.append(current_trade)
                    current_trade = None

            # Update equity
            pnl = ret * position * min(leverage, signal.suggested_leverage)
            equity *= (1 + pnl)
            self.equity_curve.append(equity)

        # Compute metrics
        total_return = equity - 1
        returns_array = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(24 * 365) if len(returns_array) > 1 else 0

        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (np.array(self.equity_curve) - peak) / peak
        max_dd = np.min(drawdown)

        # Win rate
        if self.trades:
            wins = sum(1 for t in self.trades if t["return"] > 0)
            win_rate = wins / len(self.trades)
            avg_duration = np.mean([t["duration_hours"] for t in self.trades])
        else:
            win_rate = 0
            avg_duration = 0

        # Regime accuracy
        regime_accuracy = {}
        for regime, rets in regime_returns.items():
            if rets:
                regime_accuracy[regime] = np.mean(rets)
            else:
                regime_accuracy[regime] = 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=len(self.trades),
            avg_trade_duration_hours=avg_duration,
            vol_regime_accuracy=regime_accuracy,
        )

    def print_report(self, result: BacktestResult) -> None:
        """Print backtest report"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS: Volatility-Based HYPE Strategy")
        print("=" * 60)
        print()
        print(f"Total Return:     {result.total_return:+.1%}")
        print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:     {result.max_drawdown:.1%}")
        print(f"Win Rate:         {result.win_rate:.1%}")
        print(f"Number of Trades: {result.n_trades}")
        print(f"Avg Trade Length: {result.avg_trade_duration_hours:.1f} hours")
        print()
        print("Returns by Volatility Regime:")
        for regime, ret in result.vol_regime_accuracy.items():
            print(f"  {regime.upper():10}: {ret:+.2%}")
        print("=" * 60)


# =============================================================================
# DATA REQUIREMENTS
# =============================================================================

def print_data_requirements():
    """Print what data is needed to test this hypothesis"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DATA REQUIREMENTS FOR HYPOTHESIS TESTING               ║
╚══════════════════════════════════════════════════════════════════╝

To fully test the thesis, you need:

1. VOLATILITY DATA (you have this from your ingestor)
   - BTC/ETH price data (OHLC) from Hyperliquid
   - Compute: Realized vol, Parkinson, EWMA
   - Frequency: Hourly minimum, ideally sub-hourly

2. VOLUME DATA
   - Hyperliquid total trading volume
   - Source: Hyperliquid API or your existing WS connection
   - Frequency: Daily or hourly aggregates

3. FEE DATA
   - Hyperliquid protocol fees (may need to estimate from volume)
   - Source: On-chain or Hyperliquid dashboard
   - Alternative: Estimate as volume * avg_fee_rate

4. BUYBACK DATA (this is the hardest)
   - HYPE token buyback events
   - Source: On-chain transactions, Hyperliquid announcements
   - Note: May not be public/transparent

5. HYPE PRICE DATA
   - HYPE token price (OHLC)
   - Source: Hyperliquid, DEX aggregators
   - Frequency: Hourly

MINIMUM VIABLE TEST:
If you can't get fee/buyback data, test END-TO-END:
- Just correlate: Volatility (t) → HYPE Returns (t + lag)
- If positive correlation exists, thesis has merit even without
  observing the intermediate steps.

YOUR EXISTING INFRASTRUCTURE:
- rust/ing/ already collects Hyperliquid order book + trades
- exploration/validation/ has feature computation
- You can extend to track HYPE price and compute vol features
""")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print_data_requirements()

    # Synthetic example to demonstrate the framework
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA EXAMPLE")
    print("=" * 60 + "\n")

    np.random.seed(42)
    n = 1000  # ~41 days of hourly data

    # Generate synthetic volatility (regime-switching)
    vol = np.zeros(n)
    regime = 0  # 0 = low, 1 = high
    for i in range(n):
        if np.random.random() < 0.02:  # Regime switch probability
            regime = 1 - regime
        base_vol = 0.3 if regime == 0 else 0.8
        vol[i] = base_vol + np.random.normal(0, 0.1)

    # Generate synthetic returns (correlated with vol, with lag)
    returns = np.zeros(n)
    for i in range(5, n):
        # Returns partially driven by past volatility (the thesis!)
        vol_component = 0.001 * (vol[i-3] - 0.5)  # 3-period lag
        noise = np.random.normal(0, 0.02)
        returns[i] = vol_component + noise

    # Create series
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    vol_series = pd.Series(vol, index=dates)
    ret_series = pd.Series(returns, index=dates)

    # Test hypothesis
    tester = HypothesisTester()
    result = tester.test_end_to_end(vol_series, ret_series)
    print(tester.summary_report())

    # Run backtest
    print("\nRunning backtest...")
    bt = Backtester(vol_series, ret_series)
    bt_result = bt.run(leverage=1.5)
    bt.print_report(bt_result)
