"""
Macro Regime Detector

This module implements the regime detection logic using the indicator framework.
It computes composite scores, determines regimes, and provides actionable signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json

from config import (
    MacroRegimeConfig,
    MacroRegime,
    CryptoRegime,
    IndicatorConfig,
    DEFAULT_CONFIG
)


@dataclass
class IndicatorReading:
    """A single indicator reading with metadata"""
    name: str
    value: float
    timestamp: datetime
    source: str

    # Computed fields
    score: float = 0.0              # Normalized score [-1, 1]
    signal: str = "neutral"          # expansion, contraction, neutral
    zscore: Optional[float] = None   # If z-score normalization used
    percentile: Optional[float] = None  # Historical percentile


@dataclass
class CategoryScore:
    """Score for an indicator category"""
    category: str
    raw_score: float                 # Weighted sum of indicator scores
    normalized_score: float          # Normalized to [-1, 1]
    indicator_scores: Dict[str, float]
    signal: str                      # expansion, contraction, neutral


@dataclass
class RegimeOutput:
    """Complete regime detection output"""
    timestamp: datetime
    regime: MacroRegime
    confidence: float                # 0-1, how strong is the signal

    # Scores
    composite_score: float           # Overall score [-1, 1]
    category_scores: Dict[str, CategoryScore]
    indicator_readings: Dict[str, IndicatorReading]

    # Actions
    position_size_multiplier: float  # 0-1
    kill_switches_triggered: List[str]
    warnings: List[str]

    # Crypto-specific
    crypto_regime: Optional[CryptoRegime] = None
    crypto_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "composite_score": round(self.composite_score, 3),
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "category_scores": {
                k: {"score": round(v.normalized_score, 3), "signal": v.signal}
                for k, v in self.category_scores.items()
            },
            "kill_switches_triggered": self.kill_switches_triggered,
            "warnings": self.warnings,
            "crypto_regime": self.crypto_regime.value if self.crypto_regime else None,
        }


class MacroRegimeDetector:
    """
    Main regime detection engine.

    Usage:
        detector = MacroRegimeDetector()
        detector.update_indicator("ism_pmi", 52.3)
        detector.update_indicator("dxy", 103.5)
        ...
        result = detector.compute_regime()
    """

    def __init__(self, config: MacroRegimeConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.readings: Dict[str, IndicatorReading] = {}
        self.history: List[RegimeOutput] = []
        self.historical_data: Dict[str, List[Tuple[datetime, float]]] = {}

    def update_indicator(
        self,
        name: str,
        value: float,
        timestamp: datetime = None,
        source: str = "manual"
    ) -> None:
        """
        Update a single indicator reading.

        Args:
            name: Indicator name (must match config)
            value: Current value
            timestamp: Reading timestamp (defaults to now)
            source: Data source
        """
        timestamp = timestamp or datetime.now()

        # Find indicator config
        indicator_config = self._get_indicator_config(name)
        if not indicator_config:
            raise ValueError(f"Unknown indicator: {name}")

        # Store historical data for z-score computation
        if name not in self.historical_data:
            self.historical_data[name] = []
        self.historical_data[name].append((timestamp, value))

        # Compute score
        score, signal = self._compute_indicator_score(name, value, indicator_config)

        # Compute z-score if applicable
        zscore = None
        if indicator_config.use_zscore:
            zscore = self._compute_zscore(name, value)
            # Override score with z-score based scoring
            score = np.clip(zscore / 2, -1, 1)  # Normalize z-score to [-1, 1]
            signal = "expansion" if zscore > 0.5 else "contraction" if zscore < -0.5 else "neutral"

        # Compute percentile
        percentile = self._compute_percentile(name, value)

        self.readings[name] = IndicatorReading(
            name=name,
            value=value,
            timestamp=timestamp,
            source=source,
            score=score,
            signal=signal,
            zscore=zscore,
            percentile=percentile,
        )

    def update_indicators_batch(self, data: Dict[str, float], timestamp: datetime = None) -> None:
        """Update multiple indicators at once"""
        for name, value in data.items():
            self.update_indicator(name, value, timestamp)

    def compute_regime(self) -> RegimeOutput:
        """
        Compute the current macro regime based on all indicator readings.

        Returns:
            RegimeOutput with full analysis
        """
        timestamp = datetime.now()

        # Compute category scores
        category_scores = self._compute_category_scores()

        # Compute composite score
        composite_score = self._compute_composite_score(category_scores)

        # Determine regime
        regime = self._determine_regime(composite_score)
        confidence = self._compute_confidence(composite_score, category_scores)

        # Check kill switches
        kill_switches = self._check_kill_switches()

        # Compute position sizing
        base_size = self.config.position_sizing.get(regime.value, 0.5)
        # Apply kill switch reductions
        for ks in kill_switches:
            ks_config = self.config.kill_switches.get(ks, {})
            action = ks_config.get("action", "")
            if "reduce_50pct" in action:
                base_size *= 0.5
            elif "reduce_30pct" in action:
                base_size *= 0.7
            elif "reduce_20pct" in action:
                base_size *= 0.8

        # Generate warnings
        warnings = self._generate_warnings(category_scores)

        # Crypto-specific regime
        crypto_regime, crypto_confidence = self._determine_crypto_regime()

        output = RegimeOutput(
            timestamp=timestamp,
            regime=regime,
            confidence=confidence,
            composite_score=composite_score,
            category_scores=category_scores,
            indicator_readings=self.readings.copy(),
            position_size_multiplier=base_size,
            kill_switches_triggered=kill_switches,
            warnings=warnings,
            crypto_regime=crypto_regime,
            crypto_confidence=crypto_confidence,
        )

        self.history.append(output)
        return output

    def _get_indicator_config(self, name: str) -> Optional[IndicatorConfig]:
        """Find indicator config by name across all categories"""
        all_indicators = {
            **self.config.business_cycle_indicators,
            **self.config.liquidity_indicators,
            **self.config.real_economy_indicators,
            **self.config.crypto_indicators,
            **self.config.onchain_indicators,
        }
        return all_indicators.get(name)

    def _compute_indicator_score(
        self,
        name: str,
        value: float,
        config: IndicatorConfig
    ) -> Tuple[float, str]:
        """
        Compute normalized score [-1, 1] for an indicator.

        Returns:
            Tuple of (score, signal)
        """
        direction = config.direction

        # Determine raw position relative to thresholds
        if direction == 1:
            # Higher is better (expansion)
            if value >= config.expansion_threshold:
                raw_score = 1.0
                signal = "expansion"
            elif value <= config.contraction_threshold:
                raw_score = -1.0
                signal = "contraction"
            else:
                # Interpolate between thresholds
                range_size = config.expansion_threshold - config.contraction_threshold
                if range_size > 0:
                    raw_score = (value - config.contraction_threshold) / range_size * 2 - 1
                else:
                    raw_score = 0.0
                signal = "neutral"
        else:
            # Lower is better (inverse indicators like DXY)
            if value <= config.expansion_threshold:
                raw_score = 1.0
                signal = "expansion"
            elif value >= config.contraction_threshold:
                raw_score = -1.0
                signal = "contraction"
            else:
                range_size = config.contraction_threshold - config.expansion_threshold
                if range_size > 0:
                    raw_score = (config.contraction_threshold - value) / range_size * 2 - 1
                else:
                    raw_score = 0.0
                signal = "neutral"

        return np.clip(raw_score, -1, 1), signal

    def _compute_zscore(self, name: str, value: float) -> float:
        """Compute z-score using historical data"""
        if name not in self.historical_data:
            return 0.0

        values = [v for _, v in self.historical_data[name]]
        if len(values) < 2:
            return 0.0

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0

        return (value - mean) / std

    def _compute_percentile(self, name: str, value: float) -> float:
        """Compute percentile rank using historical data"""
        if name not in self.historical_data:
            return 50.0

        values = [v for _, v in self.historical_data[name]]
        if len(values) < 2:
            return 50.0

        return float(np.sum(np.array(values) <= value) / len(values) * 100)

    def _compute_category_scores(self) -> Dict[str, CategoryScore]:
        """Compute scores for each indicator category"""
        categories = {
            "business_cycle": self.config.business_cycle_indicators,
            "liquidity": self.config.liquidity_indicators,
            "real_economy": self.config.real_economy_indicators,
            "crypto": self.config.crypto_indicators,
            "onchain": self.config.onchain_indicators,
        }

        category_scores = {}

        for cat_name, indicators in categories.items():
            indicator_scores = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for ind_name, ind_config in indicators.items():
                if ind_name in self.readings:
                    reading = self.readings[ind_name]
                    indicator_scores[ind_name] = reading.score
                    weighted_sum += reading.score * ind_config.weight
                    total_weight += ind_config.weight

            if total_weight > 0:
                raw_score = weighted_sum / total_weight
            else:
                raw_score = 0.0

            normalized_score = np.clip(raw_score, -1, 1)

            # Determine signal
            if normalized_score > 0.2:
                signal = "expansion"
            elif normalized_score < -0.2:
                signal = "contraction"
            else:
                signal = "neutral"

            category_scores[cat_name] = CategoryScore(
                category=cat_name,
                raw_score=raw_score,
                normalized_score=normalized_score,
                indicator_scores=indicator_scores,
                signal=signal,
            )

        return category_scores

    def _compute_composite_score(self, category_scores: Dict[str, CategoryScore]) -> float:
        """Compute overall composite score from category scores"""
        weighted_sum = 0.0
        total_weight = 0.0

        for cat_name, cat_score in category_scores.items():
            weight = self.config.category_weights.get(cat_name, 1.0)
            weighted_sum += cat_score.normalized_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _determine_regime(self, composite_score: float) -> MacroRegime:
        """Map composite score to regime"""
        for regime_name, (low, high) in self.config.regime_thresholds.items():
            if low <= composite_score < high:
                return MacroRegime(regime_name)

        # Default fallback
        if composite_score >= 0.3:
            return MacroRegime.EXPANSION
        elif composite_score <= -0.3:
            return MacroRegime.CONTRACTION
        else:
            return MacroRegime.UNCERTAIN

    def _compute_confidence(
        self,
        composite_score: float,
        category_scores: Dict[str, CategoryScore]
    ) -> float:
        """
        Compute confidence in regime classification.

        Higher confidence when:
        - Categories agree (low dispersion)
        - Score is far from regime boundaries
        - More indicators have readings
        """
        # Agreement: standard deviation of category scores
        cat_scores = [cs.normalized_score for cs in category_scores.values()]
        if len(cat_scores) > 1:
            agreement = 1 - min(np.std(cat_scores), 1)
        else:
            agreement = 0.5

        # Distance from boundaries
        boundary_distance = min(
            abs(composite_score - 0.3),
            abs(composite_score - 0.1),
            abs(composite_score + 0.1),
            abs(composite_score + 0.3),
        )
        boundary_conf = min(boundary_distance / 0.2, 1.0)

        # Coverage: what fraction of indicators have readings
        total_indicators = (
            len(self.config.business_cycle_indicators) +
            len(self.config.liquidity_indicators) +
            len(self.config.real_economy_indicators) +
            len(self.config.crypto_indicators) +
            len(self.config.onchain_indicators)
        )
        coverage = len(self.readings) / max(total_indicators, 1)

        # Weighted confidence
        confidence = 0.4 * agreement + 0.3 * boundary_conf + 0.3 * coverage
        return np.clip(confidence, 0, 1)

    def _check_kill_switches(self) -> List[str]:
        """Check if any kill switches are triggered"""
        triggered = []

        for ks_name, ks_config in self.config.kill_switches.items():
            indicator_name = ks_config["indicator"]
            if indicator_name not in self.readings:
                continue

            value = self.readings[indicator_name].value
            threshold = ks_config["threshold"]
            condition = ks_config["condition"]

            if condition == "above" and value > threshold:
                triggered.append(ks_name)
            elif condition == "below" and value < threshold:
                triggered.append(ks_name)

        return triggered

    def _generate_warnings(self, category_scores: Dict[str, CategoryScore]) -> List[str]:
        """Generate warning messages based on current state"""
        warnings = []

        # Check for category disagreement
        signals = [cs.signal for cs in category_scores.values()]
        if "expansion" in signals and "contraction" in signals:
            warnings.append("Mixed signals: some categories show expansion, others contraction")

        # Check for missing data
        missing_categories = []
        if not any(n in self.readings for n in self.config.business_cycle_indicators):
            missing_categories.append("business_cycle")
        if not any(n in self.readings for n in self.config.liquidity_indicators):
            missing_categories.append("liquidity")

        if missing_categories:
            warnings.append(f"Missing data for categories: {', '.join(missing_categories)}")

        # Specific indicator warnings
        if "mvrv" in self.readings:
            mvrv = self.readings["mvrv"].value
            if mvrv > 2.5:
                warnings.append(f"MVRV at {mvrv:.2f} - approaching historically overheated levels")
            elif mvrv < 1.0:
                warnings.append(f"MVRV at {mvrv:.2f} - historically undervalued territory")

        if "yield_curve_10y2y" in self.readings:
            yc = self.readings["yield_curve_10y2y"].value
            if yc < 0:
                warnings.append(f"Yield curve inverted ({yc:.2f}%) - recession signal")

        return warnings

    def _determine_crypto_regime(self) -> Tuple[Optional[CryptoRegime], float]:
        """Determine crypto-specific regime based on crypto and on-chain indicators"""
        crypto_readings = {
            k: v for k, v in self.readings.items()
            if k in self.config.crypto_indicators or k in self.config.onchain_indicators
        }

        if len(crypto_readings) < 3:
            return None, 0.0

        # Scoring logic for crypto regime
        eth_btc_score = self.readings.get("eth_btc", IndicatorReading("", 0, datetime.now(), "")).score
        others_btc_score = self.readings.get("others_btc", IndicatorReading("", 0, datetime.now(), "")).score
        btc_dom_score = self.readings.get("btc_dominance", IndicatorReading("", 0, datetime.now(), "")).score
        mvrv_value = self.readings.get("mvrv", IndicatorReading("", 1.5, datetime.now(), "")).value
        lth_score = self.readings.get("lth_supply", IndicatorReading("", 0, datetime.now(), "")).score

        # Simple regime logic
        if mvrv_value < 1.2 and lth_score > 0:
            regime = CryptoRegime.BEAR_LATE  # Accumulation
            confidence = 0.7
        elif mvrv_value > 2.5:
            if eth_btc_score > 0.3 or others_btc_score > 0.3:
                regime = CryptoRegime.BULL_LATE  # Euphoria
                confidence = 0.8
            else:
                regime = CryptoRegime.BULL_MID
                confidence = 0.6
        elif mvrv_value > 1.5 and mvrv_value <= 2.5:
            if btc_dom_score < -0.3:  # BTC dominance falling
                regime = CryptoRegime.BULL_MID
                confidence = 0.7
            else:
                regime = CryptoRegime.BULL_EARLY
                confidence = 0.6
        elif lth_score < -0.3:
            regime = CryptoRegime.BEAR_EARLY  # Distribution
            confidence = 0.7
        else:
            regime = CryptoRegime.BULL_EARLY
            confidence = 0.5

        return regime, confidence

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for a dashboard display.

        Returns dict with all relevant metrics formatted for display.
        """
        if not self.history:
            # Compute if not yet done
            self.compute_regime()

        latest = self.history[-1]

        return {
            "timestamp": latest.timestamp.isoformat(),
            "regime": {
                "name": latest.regime.value.upper(),
                "confidence": f"{latest.confidence:.0%}",
                "position_size": f"{latest.position_size_multiplier:.0%}",
            },
            "composite_score": {
                "value": latest.composite_score,
                "interpretation": self._interpret_score(latest.composite_score),
            },
            "categories": {
                cat: {
                    "score": f"{score.normalized_score:+.2f}",
                    "signal": score.signal.upper(),
                    "indicators": score.indicator_scores,
                }
                for cat, score in latest.category_scores.items()
            },
            "crypto_regime": {
                "name": latest.crypto_regime.value if latest.crypto_regime else "N/A",
                "confidence": f"{latest.crypto_confidence:.0%}" if latest.crypto_regime else "N/A",
            },
            "alerts": {
                "kill_switches": latest.kill_switches_triggered,
                "warnings": latest.warnings,
            },
        }

    def _interpret_score(self, score: float) -> str:
        """Human-readable interpretation of composite score"""
        if score > 0.5:
            return "Strong expansion signals - full risk-on"
        elif score > 0.2:
            return "Moderate expansion - constructive"
        elif score > 0:
            return "Mild expansion bias - cautiously optimistic"
        elif score > -0.2:
            return "Neutral to slight contraction - reduce risk"
        elif score > -0.5:
            return "Moderate contraction signals - defensive"
        else:
            return "Strong contraction - risk-off, preserve capital"

    def print_report(self) -> None:
        """Print a formatted regime report to console"""
        data = self.get_dashboard_data()

        print("\n" + "=" * 60)
        print("MACRO REGIME DETECTION REPORT")
        print("=" * 60)
        print(f"Timestamp: {data['timestamp']}")
        print()

        print("OVERALL REGIME")
        print("-" * 40)
        regime = data["regime"]
        print(f"  Regime:        {regime['name']}")
        print(f"  Confidence:    {regime['confidence']}")
        print(f"  Position Size: {regime['position_size']}")
        print(f"  Score:         {data['composite_score']['value']:+.3f}")
        print(f"  Interpretation: {data['composite_score']['interpretation']}")
        print()

        print("CATEGORY BREAKDOWN")
        print("-" * 40)
        for cat, cat_data in data["categories"].items():
            print(f"  {cat.upper()}: {cat_data['score']} ({cat_data['signal']})")
        print()

        print("CRYPTO REGIME")
        print("-" * 40)
        crypto = data["crypto_regime"]
        print(f"  Regime:     {crypto['name']}")
        print(f"  Confidence: {crypto['confidence']}")
        print()

        if data["alerts"]["kill_switches"]:
            print("⚠️  KILL SWITCHES TRIGGERED")
            print("-" * 40)
            for ks in data["alerts"]["kill_switches"]:
                print(f"  - {ks}")
            print()

        if data["alerts"]["warnings"]:
            print("WARNINGS")
            print("-" * 40)
            for w in data["alerts"]["warnings"]:
                print(f"  - {w}")
            print()

        print("=" * 60)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create detector
    detector = MacroRegimeDetector()

    # Simulate updating indicators
    # Business cycle
    detector.update_indicator("ism_pmi", 51.2)
    detector.update_indicator("ism_services", 52.8)
    detector.update_indicator("new_orders_inventories", 3.5)

    # Liquidity
    detector.update_indicator("dxy", 103.5)
    detector.update_indicator("us10y", 4.25)
    detector.update_indicator("yield_curve_10y2y", 0.15)
    detector.update_indicator("credit_spreads", 380)

    # Real economy
    detector.update_indicator("copper_gold", 0.0048)  # Will use z-score
    detector.update_indicator("jobless_claims", 235000)

    # Crypto
    detector.update_indicator("eth_btc", 0.052)
    detector.update_indicator("btc_dominance", 52.0)

    # On-chain
    detector.update_indicator("mvrv", 1.8)
    detector.update_indicator("exchange_whale_ratio", 0.38)
    detector.update_indicator("funding_rate", 0.015)

    # Compute and print
    result = detector.compute_regime()
    detector.print_report()

    # Also print JSON output
    print("\nJSON Output:")
    print(json.dumps(result.to_dict(), indent=2))
