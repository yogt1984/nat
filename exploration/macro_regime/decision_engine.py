"""
Decision Engine for Macro Regime Detection

Translates regime detection into actionable decisions with:
- Position sizing rules
- Entry/exit signals
- Risk management
- Pre-commitment framework to avoid emotional decisions
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from config import MacroRegime, CryptoRegime, MacroRegimeConfig, DEFAULT_CONFIG
from regime_detector import MacroRegimeDetector, RegimeOutput


class ActionType(Enum):
    """Types of trading actions"""
    HOLD = "hold"                    # Maintain current position
    ACCUMULATE = "accumulate"        # Slowly add to position
    REDUCE = "reduce"                # Reduce position size
    EXIT = "exit"                    # Exit position entirely
    WAIT = "wait"                    # No action, wait for clarity


@dataclass
class PositionGuidance:
    """Position sizing and allocation guidance"""
    overall_exposure: float          # 0-1, total risk budget to deploy
    crypto_allocation: float         # Of risk budget, how much in crypto
    btc_vs_alts: float              # 0-1, BTC dominance in crypto allocation
    leverage_allowed: float          # Max leverage multiplier
    rebalance_urgency: str          # "none", "low", "medium", "high"


@dataclass
class RiskGuardrails:
    """Risk management guardrails"""
    max_single_position: float       # Max % in any single asset
    stop_loss_distance: float        # Suggested stop loss %
    take_profit_targets: List[float] # Take profit levels
    trailing_stop: bool              # Use trailing stop
    scale_in_tranches: int           # Number of tranches for entry


@dataclass
class ActionRecommendation:
    """Complete action recommendation"""
    timestamp: datetime
    action_type: ActionType
    confidence: float                # 0-1
    position_guidance: PositionGuidance
    risk_guardrails: RiskGuardrails

    reasoning: List[str]             # Why this recommendation
    warnings: List[str]              # Risk warnings
    kill_switch_active: bool         # Is any kill switch active

    # Pre-commitment checks
    allowed_to_act: bool             # Based on pre-commitment rules
    blocked_reason: Optional[str]    # Why action is blocked


class PreCommitmentRules:
    """
    Pre-commitment framework to prevent emotional decisions.

    The user defines rules BEFORE the market moves, and the system
    enforces them to prevent FOMO/panic selling.
    """

    def __init__(self, config_path: str = None):
        self.rules = self._default_rules()
        if config_path:
            self._load_rules(config_path)

    def _default_rules(self) -> Dict[str, Any]:
        """Default pre-commitment rules"""
        return {
            # Minimum time between actions
            "min_action_interval_hours": 24,

            # Regime-based entry rules
            "entry_allowed_regimes": [
                MacroRegime.EXPANSION.value,
                MacroRegime.EARLY_RECOVERY.value,
            ],

            # Regime-based exit rules
            "forced_reduce_regimes": [
                MacroRegime.CONTRACTION.value,
            ],

            # Don't chase pumps
            "no_buy_after_pump_pct": 10,  # Don't buy after >10% daily move
            "pump_cooldown_hours": 48,

            # Don't panic sell dumps
            "no_sell_after_dump_pct": 15,  # Don't sell after >15% dump
            "dump_cooldown_hours": 72,

            # Minimum score thresholds for action
            "min_score_to_increase": 0.2,
            "max_score_to_decrease": -0.1,

            # Confidence thresholds
            "min_confidence_to_act": 0.5,

            # Scale-in requirements
            "always_scale_in": True,
            "min_tranches": 3,

            # Maximum position change per action
            "max_position_change_pct": 25,

            # Forced wait after regime change
            "regime_change_cooldown_hours": 24,
        }

    def _load_rules(self, config_path: str) -> None:
        """Load rules from config file"""
        with open(config_path, "r") as f:
            custom_rules = json.load(f)
            self.rules.update(custom_rules)

    def save_rules(self, config_path: str) -> None:
        """Save current rules to config file"""
        with open(config_path, "w") as f:
            json.dump(self.rules, f, indent=2)

    def check_allowed(
        self,
        action: ActionType,
        regime_output: RegimeOutput,
        last_action_time: datetime = None,
        last_regime: MacroRegime = None,
        recent_price_change: float = None,
    ) -> Tuple[bool, str]:
        """
        Check if an action is allowed by pre-commitment rules.

        Returns: (allowed, reason)
        """
        # Check minimum interval
        if last_action_time:
            hours_since = (datetime.now() - last_action_time).total_seconds() / 3600
            if hours_since < self.rules["min_action_interval_hours"]:
                return False, f"Action cooldown: {self.rules['min_action_interval_hours'] - hours_since:.1f}h remaining"

        # Check regime change cooldown
        if last_regime and last_regime != regime_output.regime:
            return False, f"Regime just changed to {regime_output.regime.value}. Cooldown in effect."

        # Check confidence threshold
        if regime_output.confidence < self.rules["min_confidence_to_act"]:
            return False, f"Confidence too low: {regime_output.confidence:.0%} < {self.rules['min_confidence_to_act']:.0%}"

        # Action-specific checks
        if action in [ActionType.ACCUMULATE]:
            # Entry checks
            if regime_output.regime.value not in self.rules["entry_allowed_regimes"]:
                return False, f"Current regime {regime_output.regime.value} not in allowed entry regimes"

            if regime_output.composite_score < self.rules["min_score_to_increase"]:
                return False, f"Score {regime_output.composite_score:.2f} below threshold {self.rules['min_score_to_increase']}"

            # Check pump chasing
            if recent_price_change and recent_price_change > self.rules["no_buy_after_pump_pct"]:
                return False, f"No buying after {recent_price_change:.1f}% pump (threshold: {self.rules['no_buy_after_pump_pct']}%)"

        elif action in [ActionType.REDUCE, ActionType.EXIT]:
            # Exit checks
            if regime_output.composite_score > self.rules["max_score_to_decrease"]:
                return False, f"Score {regime_output.composite_score:.2f} above threshold {self.rules['max_score_to_decrease']}"

            # Check panic selling
            if recent_price_change and recent_price_change < -self.rules["no_sell_after_dump_pct"]:
                return False, f"No selling after {recent_price_change:.1f}% dump (threshold: -{self.rules['no_sell_after_dump_pct']}%)"

        return True, "Action allowed"


class DecisionEngine:
    """
    Main decision engine that combines regime detection with actionable decisions.
    """

    def __init__(
        self,
        detector: MacroRegimeDetector = None,
        pre_commitment: PreCommitmentRules = None,
    ):
        self.detector = detector or MacroRegimeDetector()
        self.pre_commitment = pre_commitment or PreCommitmentRules()
        self.action_history: List[ActionRecommendation] = []
        self.last_regime: MacroRegime = None
        self.last_action_time: datetime = None

    def get_recommendation(
        self,
        recent_price_change: float = None,
    ) -> ActionRecommendation:
        """
        Generate an action recommendation based on current regime.
        """
        # Compute current regime
        regime_output = self.detector.compute_regime()

        # Determine action type
        action_type = self._determine_action(regime_output)

        # Check pre-commitment rules
        allowed, blocked_reason = self.pre_commitment.check_allowed(
            action=action_type,
            regime_output=regime_output,
            last_action_time=self.last_action_time,
            last_regime=self.last_regime,
            recent_price_change=recent_price_change,
        )

        # Generate position guidance
        position_guidance = self._generate_position_guidance(regime_output)

        # Generate risk guardrails
        risk_guardrails = self._generate_risk_guardrails(regime_output)

        # Generate reasoning
        reasoning = self._generate_reasoning(regime_output, action_type)

        recommendation = ActionRecommendation(
            timestamp=datetime.now(),
            action_type=action_type,
            confidence=regime_output.confidence,
            position_guidance=position_guidance,
            risk_guardrails=risk_guardrails,
            reasoning=reasoning,
            warnings=regime_output.warnings,
            kill_switch_active=len(regime_output.kill_switches_triggered) > 0,
            allowed_to_act=allowed,
            blocked_reason=blocked_reason if not allowed else None,
        )

        # Track history
        self.action_history.append(recommendation)
        self.last_regime = regime_output.regime

        return recommendation

    def _determine_action(self, regime_output: RegimeOutput) -> ActionType:
        """Determine action type based on regime"""
        regime = regime_output.regime
        score = regime_output.composite_score

        # Kill switch overrides
        if regime_output.kill_switches_triggered:
            return ActionType.REDUCE

        # Regime-based logic
        if regime == MacroRegime.EXPANSION:
            if score > 0.5:
                return ActionType.ACCUMULATE
            else:
                return ActionType.HOLD

        elif regime == MacroRegime.LATE_CYCLE:
            return ActionType.HOLD  # Maintain, don't add

        elif regime == MacroRegime.UNCERTAIN:
            return ActionType.WAIT

        elif regime == MacroRegime.EARLY_RECOVERY:
            if score > -0.15:  # Improving
                return ActionType.ACCUMULATE
            else:
                return ActionType.WAIT

        elif regime == MacroRegime.CONTRACTION:
            if score < -0.5:
                return ActionType.EXIT
            else:
                return ActionType.REDUCE

        return ActionType.HOLD

    def _generate_position_guidance(self, regime_output: RegimeOutput) -> PositionGuidance:
        """Generate position sizing guidance"""
        regime = regime_output.regime
        crypto_regime = regime_output.crypto_regime

        # Base exposure from regime
        base_exposure = regime_output.position_size_multiplier

        # Crypto allocation based on macro + crypto regime
        if regime in [MacroRegime.EXPANSION, MacroRegime.EARLY_RECOVERY]:
            crypto_allocation = 0.8  # Risk-on
        elif regime == MacroRegime.LATE_CYCLE:
            crypto_allocation = 0.5
        elif regime == MacroRegime.UNCERTAIN:
            crypto_allocation = 0.3
        else:
            crypto_allocation = 0.2

        # BTC vs alts based on crypto regime
        btc_vs_alts = 0.7  # Default: BTC heavy
        if crypto_regime == CryptoRegime.BULL_LATE:
            btc_vs_alts = 0.4  # More alts in late bull
        elif crypto_regime == CryptoRegime.BULL_MID:
            btc_vs_alts = 0.5
        elif crypto_regime in [CryptoRegime.BEAR_EARLY, CryptoRegime.BEAR_MID]:
            btc_vs_alts = 0.9  # BTC only in bear
        elif crypto_regime == CryptoRegime.BEAR_LATE:
            btc_vs_alts = 0.8

        # Leverage based on regime
        if regime == MacroRegime.EXPANSION and regime_output.confidence > 0.7:
            leverage = 1.5
        elif regime in [MacroRegime.CONTRACTION, MacroRegime.UNCERTAIN]:
            leverage = 1.0
        else:
            leverage = 1.2

        # Rebalance urgency
        if regime_output.kill_switches_triggered:
            urgency = "high"
        elif abs(regime_output.composite_score) > 0.5:
            urgency = "medium"
        else:
            urgency = "low"

        return PositionGuidance(
            overall_exposure=base_exposure,
            crypto_allocation=crypto_allocation,
            btc_vs_alts=btc_vs_alts,
            leverage_allowed=leverage,
            rebalance_urgency=urgency,
        )

    def _generate_risk_guardrails(self, regime_output: RegimeOutput) -> RiskGuardrails:
        """Generate risk management parameters"""
        regime = regime_output.regime

        # More conservative in contraction
        if regime == MacroRegime.CONTRACTION:
            return RiskGuardrails(
                max_single_position=0.15,
                stop_loss_distance=0.10,
                take_profit_targets=[0.05, 0.10, 0.15],
                trailing_stop=True,
                scale_in_tranches=5,
            )
        elif regime == MacroRegime.UNCERTAIN:
            return RiskGuardrails(
                max_single_position=0.20,
                stop_loss_distance=0.12,
                take_profit_targets=[0.10, 0.20, 0.30],
                trailing_stop=True,
                scale_in_tranches=4,
            )
        elif regime == MacroRegime.EXPANSION:
            return RiskGuardrails(
                max_single_position=0.25,
                stop_loss_distance=0.15,
                take_profit_targets=[0.20, 0.40, 0.60],
                trailing_stop=True,
                scale_in_tranches=3,
            )
        else:
            return RiskGuardrails(
                max_single_position=0.20,
                stop_loss_distance=0.12,
                take_profit_targets=[0.15, 0.30, 0.50],
                trailing_stop=True,
                scale_in_tranches=4,
            )

    def _generate_reasoning(
        self,
        regime_output: RegimeOutput,
        action_type: ActionType
    ) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""
        reasoning = []

        # Regime reasoning
        reasoning.append(
            f"Current macro regime: {regime_output.regime.value.upper()} "
            f"(score: {regime_output.composite_score:+.2f}, confidence: {regime_output.confidence:.0%})"
        )

        # Category breakdown
        for cat_name, cat_score in regime_output.category_scores.items():
            if abs(cat_score.normalized_score) > 0.3:
                direction = "positive" if cat_score.normalized_score > 0 else "negative"
                reasoning.append(f"{cat_name.replace('_', ' ').title()} indicators {direction}: {cat_score.normalized_score:+.2f}")

        # Crypto regime
        if regime_output.crypto_regime:
            reasoning.append(f"Crypto regime: {regime_output.crypto_regime.value}")

        # Action reasoning
        action_reasons = {
            ActionType.ACCUMULATE: "Conditions favor adding to positions",
            ActionType.HOLD: "Maintain current exposure, wait for stronger signal",
            ActionType.REDUCE: "Risk indicators elevated, reduce exposure",
            ActionType.EXIT: "Significant downside risk, exit to preserve capital",
            ActionType.WAIT: "Mixed signals, wait for clarity before acting",
        }
        reasoning.append(f"Action: {action_type.value.upper()} - {action_reasons.get(action_type, '')}")

        # Kill switches
        if regime_output.kill_switches_triggered:
            reasoning.append(f"⚠️ KILL SWITCHES ACTIVE: {', '.join(regime_output.kill_switches_triggered)}")

        return reasoning

    def print_recommendation(self, recommendation: ActionRecommendation = None) -> None:
        """Print formatted recommendation"""
        if recommendation is None:
            recommendation = self.get_recommendation()

        print("\n" + "=" * 70)
        print("DECISION ENGINE RECOMMENDATION")
        print("=" * 70)
        print(f"Timestamp: {recommendation.timestamp.isoformat()}")
        print()

        # Action
        action_emoji = {
            ActionType.ACCUMULATE: "🟢",
            ActionType.HOLD: "🟡",
            ActionType.REDUCE: "🟠",
            ActionType.EXIT: "🔴",
            ActionType.WAIT: "⚪",
        }
        print(f"ACTION: {action_emoji.get(recommendation.action_type, '')} {recommendation.action_type.value.upper()}")
        print(f"Confidence: {recommendation.confidence:.0%}")
        print()

        # Blocked status
        if not recommendation.allowed_to_act:
            print("⛔ ACTION BLOCKED BY PRE-COMMITMENT RULES")
            print(f"   Reason: {recommendation.blocked_reason}")
            print()

        # Position Guidance
        pg = recommendation.position_guidance
        print("POSITION GUIDANCE")
        print("-" * 40)
        print(f"  Overall Exposure:  {pg.overall_exposure:.0%}")
        print(f"  Crypto Allocation: {pg.crypto_allocation:.0%}")
        print(f"  BTC vs Alts:       {pg.btc_vs_alts:.0%} BTC")
        print(f"  Max Leverage:      {pg.leverage_allowed:.1f}x")
        print(f"  Rebalance Urgency: {pg.rebalance_urgency.upper()}")
        print()

        # Risk Guardrails
        rg = recommendation.risk_guardrails
        print("RISK GUARDRAILS")
        print("-" * 40)
        print(f"  Max Single Position: {rg.max_single_position:.0%}")
        print(f"  Stop Loss Distance:  {rg.stop_loss_distance:.0%}")
        print(f"  Take Profit Targets: {[f'{t:.0%}' for t in rg.take_profit_targets]}")
        print(f"  Scale-in Tranches:   {rg.scale_in_tranches}")
        print()

        # Reasoning
        print("REASONING")
        print("-" * 40)
        for r in recommendation.reasoning:
            print(f"  • {r}")
        print()

        # Warnings
        if recommendation.warnings:
            print("⚠️ WARNINGS")
            print("-" * 40)
            for w in recommendation.warnings:
                print(f"  • {w}")
            print()

        if recommendation.kill_switch_active:
            print("🚨 KILL SWITCH ACTIVE - REDUCE RISK IMMEDIATELY")
            print()

        print("=" * 70)


# =============================================================================
# Integration with main pipeline
# =============================================================================

def run_decision_pipeline(indicator_data: Dict[str, float]) -> ActionRecommendation:
    """
    Full pipeline: update indicators → detect regime → generate recommendation

    Args:
        indicator_data: Dictionary of indicator name → current value

    Returns:
        ActionRecommendation with full guidance
    """
    engine = DecisionEngine()

    # Update all indicators
    engine.detector.update_indicators_batch(indicator_data)

    # Get recommendation
    recommendation = engine.get_recommendation()

    return recommendation


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create engine
    engine = DecisionEngine()

    # Update indicators (simulated)
    engine.detector.update_indicator("ism_pmi", 51.2)
    engine.detector.update_indicator("ism_services", 52.8)
    engine.detector.update_indicator("new_orders_inventories", 3.5)
    engine.detector.update_indicator("dxy", 103.5)
    engine.detector.update_indicator("us10y", 4.25)
    engine.detector.update_indicator("yield_curve_10y2y", 0.15)
    engine.detector.update_indicator("credit_spreads", 380)
    engine.detector.update_indicator("copper_gold", 0.0048)
    engine.detector.update_indicator("jobless_claims", 235000)
    engine.detector.update_indicator("eth_btc", 0.052)
    engine.detector.update_indicator("btc_dominance", 52.0)
    engine.detector.update_indicator("mvrv", 1.8)
    engine.detector.update_indicator("exchange_whale_ratio", 0.38)
    engine.detector.update_indicator("funding_rate", 0.015)
    engine.detector.update_indicator("lth_supply", 0.5)  # z-score

    # Get and print recommendation
    engine.print_recommendation()
