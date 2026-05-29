//! Microstructure Feature Extraction
//!
//! High-frequency structural features derived from order book dynamics and trade flow.
//!
//! # Features (5 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **OBI velocity** | d(OBI_l5)/dt via 1s finite difference | (-inf, +inf) | Positive = imbalance growing toward bids |
//! | **OBI acceleration** | d²(OBI_l5)/dt² | (-inf, +inf) | Inflection points predict direction changes |
//! | **Queue position bid** | depth_at_touch / avg_trade_size (bid) | [0, +inf) | Estimated time-to-fill for maker bid |
//! | **Queue position ask** | depth_at_touch / avg_trade_size (ask) | [0, +inf) | Estimated time-to-fill for maker ask |
//! | **Depth recovery ratio** | Current depth / depth 1s ago | [0, +inf) | >1 = book refilling after take |
//!
//! # References
//!
//! - OBI dynamics: Biais, Hillion & Spatt (1995) — inflection in imbalance velocity

use ing_types::{OrderBook, RingBuffer, TradeBuffer};

/// Microstructure features (5 features)
#[derive(Debug, Clone, Default)]
pub struct MicrostructureFeatures {
    /// d(OBI_l5)/dt — imbalance velocity (1s finite difference)
    pub obi_velocity: f64,
    /// d²(OBI_l5)/dt² — imbalance acceleration
    pub obi_acceleration: f64,
    /// Bid queue position: depth_at_touch / avg_trade_size
    pub queue_position_bid: f64,
    /// Ask queue position: depth_at_touch / avg_trade_size
    pub queue_position_ask: f64,
    /// Depth recovery ratio: current 5-level depth / depth 1s ago
    pub depth_recovery_ratio: f64,
}

impl MicrostructureFeatures {
    pub fn count() -> usize {
        5
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "micro_obi_velocity",
            "micro_obi_acceleration",
            "micro_queue_position_bid",
            "micro_queue_position_ask",
            "micro_depth_recovery_ratio",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.obi_velocity,
            self.obi_acceleration,
            self.queue_position_bid,
            self.queue_position_ask,
            self.depth_recovery_ratio,
        ]
    }
}

/// Compute microstructure features
///
/// obi_buffer: ring buffer of OBI_l5 values sampled at ~100ms (10 samples ≈ 1s)
pub fn compute(
    order_book: &OrderBook,
    trade_buffer: &TradeBuffer,
    obi_buffer: &RingBuffer<f64>,
    depth_buffer: &RingBuffer<f64>,
) -> MicrostructureFeatures {
    // --- OBI dynamics (5.2) ---
    let n = obi_buffer.len();
    let (obi_velocity, obi_acceleration) = if n >= 20 {
        // Velocity: difference over ~10 samples (~1s at 100ms emission)
        let current = obi_buffer.last().copied().unwrap_or(0.0);
        let past_1s = obi_buffer
            .get(n.saturating_sub(10))
            .copied()
            .unwrap_or(current);
        let past_2s = obi_buffer
            .get(n.saturating_sub(20))
            .copied()
            .unwrap_or(past_1s);
        let v1 = current - past_1s;
        let v0 = past_1s - past_2s;
        (v1, v1 - v0)
    } else if n >= 10 {
        let current = obi_buffer.last().copied().unwrap_or(0.0);
        let past = obi_buffer
            .get(n.saturating_sub(10))
            .copied()
            .unwrap_or(current);
        (current - past, 0.0)
    } else {
        (0.0, 0.0)
    };

    // --- Queue position proxy (5.5) ---
    let avg_trade_size = {
        let vol = trade_buffer.volume_in_window(5); // 5s window
        let count = trade_buffer.count_in_window(5) as f64;
        if count > 0.0 {
            vol / count
        } else {
            0.0
        }
    };

    let queue_position_bid = if avg_trade_size > 0.0 {
        match order_book.best_bid_level() {
            Some(level) => level.size / avg_trade_size,
            None => 0.0,
        }
    } else {
        0.0
    };

    let queue_position_ask = if avg_trade_size > 0.0 {
        match order_book.best_ask_level() {
            Some(level) => level.size / avg_trade_size,
            None => 0.0,
        }
    } else {
        0.0
    };

    // --- Depth recovery (5.1 simplified) ---
    let current_depth = order_book.bid_depth(5) + order_book.ask_depth(5);
    let depth_n = depth_buffer.len();
    let depth_recovery_ratio = if depth_n >= 10 {
        let past_depth = depth_buffer
            .get(depth_n.saturating_sub(10))
            .copied()
            .unwrap_or(current_depth);
        if past_depth > 0.0 {
            current_depth / past_depth
        } else {
            1.0
        }
    } else {
        1.0
    };

    MicrostructureFeatures {
        obi_velocity,
        obi_acceleration,
        queue_position_bid,
        queue_position_ask,
        depth_recovery_ratio,
    }
}
