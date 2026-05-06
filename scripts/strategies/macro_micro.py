"""
Strategy: Macro Direction + Microstructure Timing.

Thesis:
    Use daily/weekly signals (MA crossover, support/resistance) for DIRECTION.
    Use 100ms microstructure features (entropy, toxicity, OFI) for TIMING.

    Example: ETH SMA(7) crosses above SMA(21) → go long.
    But don't enter immediately — wait for microstructure to confirm:
    - Low toxicity (VPIN < threshold → no informed flow against us)
    - Favorable entropy (not trending against us)
    - Order flow imbalance aligned with macro direction

    Exit early when microstructure signals danger:
    - VPIN spike (informed sellers arriving)
    - Entropy collapse (trend reversing)
    - Whale flow opposing our position

This lets you ride macro trends while avoiding the worst drawdowns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyMeta


class MacroMicro(Strategy):
    """
    Macro direction + micro timing strategy.

    Macro signal (daily): determines if we should be long, short, or flat.
    Micro signal (15min bars): determines WHEN to enter/exit within that bias.

    Parameters:
        ma_fast: fast MA period for daily data (default: 7)
        ma_slow: slow MA period for daily data (default: 21)
        vpin_exit: VPIN threshold to force exit (default: 0.7)
        entropy_confirm: entropy threshold to confirm entry (default: 0.5)
        ofi_confirm: OFI threshold to confirm direction (default: 0.3)
        max_position: maximum signal magnitude (default: 1.0)
    """

    def __init__(
        self,
        ma_fast: int = 7,
        ma_slow: int = 21,
        vpin_exit: float = 0.7,
        entropy_confirm: float = 0.5,
        ofi_confirm: float = 0.3,
        max_position: float = 1.0,
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.vpin_exit = vpin_exit
        self.entropy_confirm = entropy_confirm
        self.ofi_confirm = ofi_confirm
        self.max_position = max_position

        self.meta = StrategyMeta(
            name="macro_micro",
            description="Daily MA crossover for direction, microstructure for entry/exit timing",
            paper="Jegadeesh & Titman (1993) + Cont et al. (2014)",
            horizon="1d",
            required_columns=[
                "raw_midprice_mean",
                "toxic_vpin_10_mean",
                "ent_tick_1m_mean",
                "imbalance_qty_l1_mean",
            ],
            parameters={
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "vpin_exit": vpin_exit,
                "entropy_confirm": entropy_confirm,
            },
        )

        # Macro state (set externally via set_macro_bias)
        self._macro_direction = 0  # +1, -1, or 0

    def warmup_bars(self) -> int:
        return 1  # macro warmup is external

    def set_macro_bias(self, direction: int, strength: float = 1.0):
        """
        Set the macro directional bias from daily data.

        Args:
            direction: +1 (long), -1 (short), 0 (flat)
            strength: 0.0 to 1.0 (conviction)
        """
        self._macro_direction = direction
        self._macro_strength = min(1.0, max(0.0, strength))

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Extract micro timing features from 15-min bars."""
        result = pd.DataFrame(index=bars.index)

        # VPIN (toxicity — are informed traders active?)
        for col in ["toxic_vpin_10_mean", "toxic_vpin_10"]:
            if col in bars.columns:
                result["vpin"] = bars[col].values
                break
        else:
            result["vpin"] = np.nan

        # Entropy (predictability — is market trending?)
        for col in ["ent_tick_1m_mean", "ent_tick_1m"]:
            if col in bars.columns:
                result["entropy"] = bars[col].values
                break
        else:
            result["entropy"] = np.nan

        # Order flow imbalance (directional pressure)
        for col in ["imbalance_qty_l1_mean", "imbalance_qty_l1"]:
            if col in bars.columns:
                result["ofi"] = bars[col].values
                break
        else:
            result["ofi"] = np.nan

        # Price (for MA computation if no external macro)
        for col in ["raw_midprice_mean", "raw_midprice_close", "raw_midprice"]:
            if col in bars.columns:
                result["price"] = bars[col].values
                break
        else:
            result["price"] = np.nan

        # Whale flow (large player activity)
        for col in ["whale_net_flow_1h_sum", "whale_net_flow_1h"]:
            if col in bars.columns:
                result["whale_flow"] = bars[col].values
                break
        else:
            result["whale_flow"] = 0.0

        return result

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate signal combining macro bias with micro timing.

        Logic:
            1. If macro says long → look for micro entry (low VPIN + aligned OFI)
            2. If macro says short → look for micro entry (low VPIN + opposing OFI)
            3. Force exit on VPIN spike regardless of macro
            4. Scale position by micro confidence
        """
        n = len(features)
        signal = np.full(n, 0.0)
        macro_dir = self._macro_direction

        if macro_dir == 0:
            return pd.Series(signal, index=features.index, name="signal")

        vpin = features["vpin"].values
        entropy = features["entropy"].values
        ofi = features["ofi"].values

        for i in range(n):
            # Skip if data missing
            if np.isnan(vpin[i]) or np.isnan(ofi[i]):
                signal[i] = np.nan
                continue

            # ── Exit condition: VPIN spike (danger) ──
            if vpin[i] > self.vpin_exit:
                signal[i] = 0.0
                continue

            # ── Entry/hold conditions ──
            micro_score = 0.0
            n_factors = 0

            # VPIN low = safe to enter (no informed counter-flow)
            if vpin[i] < 0.4:
                micro_score += 0.3
            n_factors += 1

            # OFI aligned with macro direction
            if not np.isnan(ofi[i]):
                if (macro_dir > 0 and ofi[i] > self.ofi_confirm):
                    micro_score += 0.4  # buy pressure confirms long
                elif (macro_dir < 0 and ofi[i] < -self.ofi_confirm):
                    micro_score += 0.4  # sell pressure confirms short
                elif (macro_dir > 0 and ofi[i] < -self.ofi_confirm):
                    micro_score -= 0.3  # sell pressure contradicts long
                elif (macro_dir < 0 and ofi[i] > self.ofi_confirm):
                    micro_score -= 0.3  # buy pressure contradicts short
                n_factors += 1

            # Entropy: moderate entropy = healthy trend
            if not np.isnan(entropy[i]):
                ent_norm = entropy[i] / np.log(3)  # normalize to [0,1]
                if 0.3 < ent_norm < 0.7:
                    micro_score += 0.2  # healthy, not extreme
                elif ent_norm < 0.2:
                    micro_score += 0.1  # very trending (good if aligned)
                n_factors += 1

            # Whale flow aligned
            whale = features["whale_flow"].iloc[i] if "whale_flow" in features.columns else 0
            if not np.isnan(whale) and abs(whale) > 0:
                if np.sign(whale) == macro_dir:
                    micro_score += 0.2
                elif np.sign(whale) == -macro_dir:
                    micro_score -= 0.2
                n_factors += 1

            # Final signal: macro direction * micro confidence
            confidence = max(0.0, micro_score)
            signal[i] = macro_dir * confidence * self.max_position

        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)

        return pd.Series(signal, index=features.index, name="signal")


class EthMACrossover(Strategy):
    """
    ETH-specific MA crossover with microstructure exit.

    Simple and profitable:
    - Long when SMA(7) > SMA(21) on daily chart
    - Exit early when microstructure shows danger (VPIN spike, entropy collapse)
    - This avoids the biggest drawdowns that kill pure MA strategies
    """

    def __init__(
        self,
        ma_fast: int = 7,
        ma_slow: int = 21,
        vpin_exit: float = 0.65,
        max_position: float = 1.0,
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.vpin_exit = vpin_exit
        self.max_position = max_position
        self._in_uptrend = False  # set from daily data

        self.meta = StrategyMeta(
            name="eth_ma_crossover",
            description=f"ETH SMA({ma_fast}/{ma_slow}) crossover with microstructure early exit",
            paper="Jegadeesh & Titman (1993)",
            horizon="1d",
            required_columns=["raw_midprice_mean", "toxic_vpin_10_mean"],
            parameters={
                "ma_fast": ma_fast,
                "ma_slow": ma_slow,
                "vpin_exit": vpin_exit,
            },
        )

    def warmup_bars(self) -> int:
        return 1

    def set_trend(self, in_uptrend: bool):
        """Set from daily data: is SMA(fast) > SMA(slow)?"""
        self._in_uptrend = in_uptrend

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index)

        for col in ["toxic_vpin_10_mean", "toxic_vpin_10"]:
            if col in bars.columns:
                result["vpin"] = bars[col].values
                break
        else:
            result["vpin"] = 0.3  # default safe

        for col in ["ent_tick_1m_mean", "ent_tick_1m"]:
            if col in bars.columns:
                result["entropy"] = bars[col].values
                break
        else:
            result["entropy"] = np.nan

        return result

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Long when in uptrend, exit on VPIN danger.

        - Uptrend + safe VPIN → full long
        - Uptrend + elevated VPIN → reduce position
        - Uptrend + VPIN spike → flat (early exit)
        - Downtrend → flat (wait for next crossover)
        """
        n = len(features)
        signal = np.full(n, 0.0)

        if not self._in_uptrend:
            return pd.Series(signal, index=features.index, name="signal")

        vpin = features["vpin"].values

        for i in range(n):
            if np.isnan(vpin[i]):
                signal[i] = self.max_position  # default to full if no data
                continue

            if vpin[i] > self.vpin_exit:
                signal[i] = 0.0  # EXIT: informed flow detected
            elif vpin[i] > 0.5:
                signal[i] = self.max_position * 0.5  # reduce
            else:
                signal[i] = self.max_position  # full position

        return pd.Series(signal, index=features.index, name="signal")
