//! Trade Flow Feature Extraction
//!
//! Captures trade arrival patterns, volume dynamics, and aggressor behaviour.
//! These features measure the pace and directional conviction of order flow.
//!
//! # Features (12 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Count (1s/5s/30s)** | Trade count in window | [0, +inf) | Higher = more active |
//! | **Volume (1s/5s/30s)** | Total traded volume in window | [0, +inf) | Higher = more liquidity consumed |
//! | **Aggressor ratio (5s/30s)** | Buy volume / total volume | [0, 1] | >0.5 = buy-dominated |
//! | **VWAP** | Volume-weighted average price (5s) | [0, +inf) | Where volume transacted |
//! | **VWAP deviation** | (VWAP - last_price) / last_price | (-inf, +inf) | >0 = buying above market |
//! | **Avg trade size** | Mean trade size over 30s | [0, +inf) | Larger = institutional flow |
//! | **Intensity** | Trades per second (5s EMA) | [0, +inf) | Measures urgency |
//!
//! Window sizes — 1s: microstructure/HFT timescale. 5s: short-term flow patterns.
//! 30s: medium-term directional conviction. These correspond to typical market-maker
//! quote update frequencies (1-5s) and informed trader execution horizons (30s+).

use crate::state::TradeBuffer;

/// Trade flow features (12 features)
#[derive(Debug, Clone, Default)]
pub struct FlowFeatures {
    /// Trade count in 1 second
    pub count_1s: u32,
    /// Trade count in 5 seconds
    pub count_5s: u32,
    /// Trade count in 30 seconds
    pub count_30s: u32,
    /// Volume in 1 second
    pub volume_1s: f64,
    /// Volume in 5 seconds
    pub volume_5s: f64,
    /// Volume in 30 seconds
    pub volume_30s: f64,
    /// Buy aggressor ratio in 5 seconds
    pub aggressor_ratio_5s: f64,
    /// Buy aggressor ratio in 30 seconds
    pub aggressor_ratio_30s: f64,
    /// VWAP in 5 seconds
    pub vwap_5s: f64,
    /// VWAP deviation from midprice (as ratio)
    pub vwap_deviation: f64,
    /// Average trade size in 30 seconds
    pub avg_trade_size_30s: f64,
    /// Trade intensity (trades per second, EMA)
    pub intensity: f64,
}

impl FlowFeatures {
    pub fn count() -> usize { 12 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "flow_count_1s",
            "flow_count_5s",
            "flow_count_30s",
            "flow_volume_1s",
            "flow_volume_5s",
            "flow_volume_30s",
            "flow_aggressor_ratio_5s",
            "flow_aggressor_ratio_30s",
            "flow_vwap_5s",
            "flow_vwap_deviation",
            "flow_avg_trade_size_30s",
            "flow_intensity",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.count_1s as f64,
            self.count_5s as f64,
            self.count_30s as f64,
            self.volume_1s,
            self.volume_5s,
            self.volume_30s,
            self.aggressor_ratio_5s,
            self.aggressor_ratio_30s,
            self.vwap_5s,
            self.vwap_deviation,
            self.avg_trade_size_30s,
            self.intensity,
        ]
    }
}

/// Compute flow features from trade buffer
pub fn compute(trade_buffer: &TradeBuffer) -> FlowFeatures {
    let count_1s = trade_buffer.count_in_window(1) as u32;
    let count_5s = trade_buffer.count_in_window(5) as u32;
    let count_30s = trade_buffer.count_in_window(30) as u32;

    let volume_1s = trade_buffer.volume_in_window(1);
    let volume_5s = trade_buffer.volume_in_window(5);
    let volume_30s = trade_buffer.volume_in_window(30);

    let aggressor_ratio_5s = trade_buffer.aggressor_ratio_in_window(5);
    let aggressor_ratio_30s = trade_buffer.aggressor_ratio_in_window(30);

    let vwap_5s = trade_buffer.vwap_in_window(5).unwrap_or(0.0);

    // VWAP deviation: needs current price reference
    // For now, use ratio of vwap_5s to last trade price
    let vwap_deviation = if let (Some(last), Some(vwap)) = (
        trade_buffer.last_trade().map(|t| t.price),
        trade_buffer.vwap_in_window(5)
    ) {
        if last > 0.0 {
            (vwap - last) / last
        } else {
            0.0
        }
    } else {
        0.0
    };

    let avg_trade_size_30s = trade_buffer.avg_trade_size_in_window(30);
    let intensity = trade_buffer.intensity_in_window(5);

    FlowFeatures {
        count_1s,
        count_5s,
        count_30s,
        volume_1s,
        volume_5s,
        volume_30s,
        aggressor_ratio_5s,
        aggressor_ratio_30s,
        vwap_5s,
        vwap_deviation,
        avg_trade_size_30s,
        intensity,
    }
}
