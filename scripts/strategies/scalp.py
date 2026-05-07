"""
Strategy: Microstructure Scalping.

Algorithm based on 4 interacting mechanics:

1. SPREAD CAPTURE — only enter when spread is wide enough to profit after fees.
   Wider spread = more edge per trade. Narrow spread = sit out.

2. ORDER FLOW DIRECTION — OFI (order flow imbalance) determines direction.
   Strong bid imbalance → go long. Strong ask imbalance → go short.
   This front-runs the expected price impact of queued orders.

3. MEAN REVERSION — VWAP deviation provides secondary directional signal.
   Price below VWAP + positive OFI = strong long (mean reversion + flow agree).
   Price above VWAP + negative OFI = strong short.
   Disagreement (price below VWAP but negative OFI) = reduce conviction.

4. INVENTORY/RISK MANAGEMENT — VPIN and entropy control position sizing and exit.
   High VPIN = informed traders arriving → reduce or exit immediately.
   Low entropy = predictable trending → don't fight the trend, go flat.
   Whale flow opposing position → exit early.
   Time decay: if trade hasn't worked in N bars, close it.

The key insight: mechanics 1-3 determine WHAT to do.
Mechanic 4 determines WHETHER to do it and WHEN to stop.

Signal output is continuous [-1, +1]:
  +1.0 = maximum long conviction
  -1.0 = maximum short conviction
   0.0 = flat (no trade or danger detected)
  NaN  = insufficient data

Designed for 5min bars but works at 15min with reduced frequency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyMeta


def _safe_col(bars: pd.DataFrame, candidates: list, default=np.nan) -> np.ndarray:
    """Extract first matching column from bars, returning default array if none found."""
    for col in candidates:
        if col in bars.columns:
            return bars[col].values.astype(np.float64)
    return np.full(len(bars), default, dtype=np.float64)


class MicrostructureScalp(Strategy):
    """
    Microstructure scalping: spread capture + OFI direction + VWAP
    reversion + VPIN/entropy risk management.

    Parameters:
        min_spread_bps:     Minimum spread to consider entry (default: 1.0 bps)
        ofi_entry:          OFI threshold for directional signal (default: 0.15)
        vwap_weight:        Weight of VWAP deviation in signal (default: 0.3)
        vpin_caution:       VPIN level to reduce position (default: 0.5)
        vpin_exit:          VPIN level to force flat (default: 0.65)
        entropy_floor:      Entropy below this = trending, reduce (default: 0.3)
        whale_weight:       Weight of whale flow in signal (default: 0.15)
        max_position:       Maximum signal magnitude (default: 1.0)
    """

    def __init__(
        self,
        min_spread_bps: float = 0.3,
        ofi_entry: float = 0.15,
        vwap_weight: float = 0.3,
        vpin_caution: float = 0.5,
        vpin_exit: float = 0.65,
        entropy_floor: float = 0.3,
        whale_weight: float = 0.15,
        max_position: float = 1.0,
    ):
        self.min_spread_bps = min_spread_bps
        self.ofi_entry = ofi_entry
        self.vwap_weight = vwap_weight
        self.vpin_caution = vpin_caution
        self.vpin_exit = vpin_exit
        self.entropy_floor = entropy_floor
        self.whale_weight = whale_weight
        self.max_position = max_position

        self.meta = StrategyMeta(
            name="microstructure_scalp",
            description=(
                "Spread capture + OFI direction + VWAP reversion, "
                "gated by VPIN/entropy risk management"
            ),
            paper="Cont et al. (2014), Easley et al. (2012)",
            horizon="5min",
            required_columns=[
                "raw_spread_bps",
                "imbalance_qty_l1",
                "toxic_vpin_10",
                "flow_vwap_deviation",
                "ent_tick_1m",
            ],
            parameters={
                "min_spread_bps": min_spread_bps,
                "ofi_entry": ofi_entry,
                "vwap_weight": vwap_weight,
                "vpin_caution": vpin_caution,
                "vpin_exit": vpin_exit,
                "entropy_floor": entropy_floor,
            },
        )

    def warmup_bars(self) -> int:
        return 2

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Extract the 4 mechanic inputs from bar data."""
        result = pd.DataFrame(index=bars.index)

        # ── Mechanic 1: Spread ──
        result["spread_bps"] = _safe_col(bars, [
            "raw_spread_bps_mean", "raw_spread_bps_last", "raw_spread_bps",
        ])

        # ── Mechanic 2: Order Flow Imbalance (direction) ──
        # L1 imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty), range [-1, +1]
        result["ofi_l1"] = _safe_col(bars, [
            "imbalance_qty_l1_mean", "imbalance_qty_l1_last", "imbalance_qty_l1",
        ])
        # Multi-level for confirmation
        result["ofi_l5"] = _safe_col(bars, [
            "imbalance_qty_l5_mean", "imbalance_qty_l5_last", "imbalance_qty_l5",
        ])
        # Depth-weighted imbalance (accounts for order sizes)
        result["ofi_depth"] = _safe_col(bars, [
            "imbalance_depth_weighted_mean", "imbalance_depth_weighted_last",
            "imbalance_depth_weighted",
        ])
        # Flow imbalance from toxicity vector (trade-based, not book-based)
        result["flow_imbalance"] = _safe_col(bars, [
            "toxic_flow_imbalance_mean", "toxic_flow_imbalance_last",
            "toxic_flow_imbalance",
        ])

        # ── Mechanic 3: VWAP deviation (mean reversion) ──
        result["vwap_dev"] = _safe_col(bars, [
            "flow_vwap_deviation_mean", "flow_vwap_deviation_last",
            "flow_vwap_deviation",
        ])

        # ── Mechanic 4: Risk signals ──
        # VPIN (toxicity — informed flow detection)
        result["vpin"] = _safe_col(bars, [
            "toxic_vpin_10_mean", "toxic_vpin_10_last", "toxic_vpin_10",
        ])
        # Entropy (predictability — low = trending = danger for scalping)
        result["entropy"] = _safe_col(bars, [
            "ent_tick_1m_mean", "ent_tick_1m_last", "ent_tick_1m",
        ])
        # Whale flow (large player directional pressure)
        result["whale_flow"] = _safe_col(bars, [
            "whale_net_flow_1h_sum", "whale_net_flow_1h_mean",
            "whale_net_flow_1h",
        ], default=0.0)
        # Volatility (high vol = wider stops needed)
        result["vol"] = _safe_col(bars, [
            "vol_returns_1m_mean", "vol_returns_1m_last", "vol_returns_1m",
        ])
        # Trend monotonicity (strong trend = don't scalp against it)
        result["trend_mono"] = _safe_col(bars, [
            "trend_monotonicity_60_mean", "trend_monotonicity_60_last",
            "trend_monotonicity_60",
        ], default=0.0)

        return result

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Combine 4 mechanics into a single signal.

        Signal construction:
            1. Check spread gate (mechanic 1) → if spread too narrow, flat
            2. Compute direction score from OFI (mechanic 2) + VWAP (mechanic 3)
            3. Apply risk scaling from VPIN/entropy/whale (mechanic 4)
            4. Output = direction * risk_scale * max_position
        """
        n = len(features)
        signal = np.full(n, 0.0)

        spread = features["spread_bps"].values
        ofi_l1 = features["ofi_l1"].values
        ofi_l5 = features["ofi_l5"].values
        ofi_depth = features["ofi_depth"].values
        flow_imb = features["flow_imbalance"].values
        vwap_dev = features["vwap_dev"].values
        vpin = features["vpin"].values
        entropy = features["entropy"].values
        whale = features["whale_flow"].values
        trend_mono = features["trend_mono"].values

        for i in range(n):
            # ════════════════════════════════════════════
            # MECHANIC 1: Spread regime
            # ════════════════════════════════════════════
            # Spread determines edge magnitude, not a hard gate.
            # Wider spread = more edge per trade = larger position.
            # Very narrow spread = OFI/reversion must be stronger to justify.
            if np.isnan(spread[i]):
                spread_scale = 0.5
            elif spread[i] >= self.min_spread_bps:
                spread_scale = 1.0  # full edge from spread capture
            else:
                # Below threshold: scale down but don't block entirely
                spread_scale = 0.4 + 0.6 * (spread[i] / self.min_spread_bps)

            # ════════════════════════════════════════════
            # GATE 2: VPIN kill switch
            # ════════════════════════════════════════════
            # High VPIN = informed traders active = don't trade against them
            if not np.isnan(vpin[i]) and vpin[i] > self.vpin_exit:
                signal[i] = 0.0
                continue

            # ════════════════════════════════════════════
            # MECHANIC 2: OFI direction score
            # ════════════════════════════════════════════
            # Combine multiple OFI sources for robustness
            direction_score = 0.0
            direction_sources = 0

            # L1 imbalance (fastest, noisiest)
            if not np.isnan(ofi_l1[i]):
                if abs(ofi_l1[i]) > self.ofi_entry:
                    direction_score += np.sign(ofi_l1[i]) * 0.35
                direction_sources += 1

            # L5 imbalance (deeper book, more stable)
            if not np.isnan(ofi_l5[i]):
                if abs(ofi_l5[i]) > self.ofi_entry * 0.8:
                    direction_score += np.sign(ofi_l5[i]) * 0.25
                direction_sources += 1

            # Depth-weighted imbalance (size-adjusted)
            if not np.isnan(ofi_depth[i]):
                if abs(ofi_depth[i]) > self.ofi_entry:
                    direction_score += np.sign(ofi_depth[i]) * 0.2
                direction_sources += 1

            # Trade flow imbalance (actual executions, not book)
            if not np.isnan(flow_imb[i]):
                if abs(flow_imb[i]) > self.ofi_entry:
                    direction_score += np.sign(flow_imb[i]) * 0.2
                direction_sources += 1

            if direction_sources == 0:
                signal[i] = np.nan
                continue

            # ════════════════════════════════════════════
            # MECHANIC 3: VWAP mean reversion overlay
            # ════════════════════════════════════════════
            # Negative vwap_dev = price below VWAP → bullish reversion
            # Positive vwap_dev = price above VWAP → bearish reversion
            if not np.isnan(vwap_dev[i]) and abs(vwap_dev[i]) > 1e-8:
                reversion_signal = -np.sign(vwap_dev[i])
                # Boost if OFI agrees with reversion direction
                if np.sign(reversion_signal) == np.sign(direction_score):
                    direction_score += reversion_signal * self.vwap_weight
                else:
                    # OFI and VWAP disagree — reduce conviction
                    direction_score *= 0.7

            # ════════════════════════════════════════════
            # MECHANIC 4: Risk scaling
            # ════════════════════════════════════════════
            risk_scale = 1.0

            # VPIN: cautious zone reduces position
            if not np.isnan(vpin[i]) and vpin[i] > self.vpin_caution:
                # Linear ramp-down from caution to exit
                ramp = (vpin[i] - self.vpin_caution) / (self.vpin_exit - self.vpin_caution)
                risk_scale *= max(0.0, 1.0 - ramp * 0.8)

            # Entropy: low entropy = trending = bad for scalping
            if not np.isnan(entropy[i]):
                ent_norm = entropy[i] / max(np.log(3), 1.0)
                if ent_norm < self.entropy_floor:
                    # Strong trend — scale down heavily
                    risk_scale *= 0.3
                elif ent_norm < 0.5:
                    # Mild trend — scale down moderately
                    risk_scale *= 0.7

            # Trend monotonicity: strong trend = don't scalp against it
            if not np.isnan(trend_mono[i]) and abs(trend_mono[i]) > 0.7:
                # If our direction opposes the trend, reduce heavily
                if np.sign(direction_score) != np.sign(trend_mono[i]):
                    risk_scale *= 0.3
                # If aligned with trend, slight boost
                else:
                    risk_scale *= 1.1

            # Whale flow: opposing whale flow = danger
            if not np.isnan(whale[i]) and abs(whale[i]) > 0:
                whale_dir = np.sign(whale[i])
                if whale_dir != 0:
                    if whale_dir == np.sign(direction_score):
                        risk_scale *= (1.0 + self.whale_weight)
                    else:
                        risk_scale *= (1.0 - self.whale_weight)

            # ════════════════════════════════════════════
            # FINAL SIGNAL
            # ════════════════════════════════════════════
            raw = direction_score * risk_scale * spread_scale * self.max_position
            signal[i] = np.clip(raw, -1.0, 1.0)

        return pd.Series(signal, index=features.index, name="signal")
