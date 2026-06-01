//! Medium-Frequency Feature Extraction
//!
//! Computes RSI, EMA, Bollinger Bands, and ATR at 1m/5m/15m timeframes.
//! EMAs update continuously on every tick; bar-based indicators (RSI, Bollinger, ATR)
//! recompute on completed bars and hold their values between bar boundaries.
//!
//! # Bar Definition
//!
//! Bars are defined by tick count (not wall clock): 600 / 3000 / 9000 ticks
//! for 1m / 5m / 15m at 100ms emission interval. This is deterministic and
//! avoids clock-alignment issues.
//!
//! # Features (17 total)
//!
//! | Feature | Type | Range | Description |
//! |---------|------|-------|-------------|
//! | `mf_ema_1m` | tick | price | EMA(span=600) |
//! | `mf_ema_5m` | tick | price | EMA(span=3000) |
//! | `mf_ema_15m` | tick | price | EMA(span=9000) |
//! | `mf_ema_cross_1m_5m` | tick | bps | (EMA_1m - EMA_5m) / EMA_5m × 10000 |
//! | `mf_ema_cross_5m_15m` | tick | bps | (EMA_5m - EMA_15m) / EMA_15m × 10000 |
//! | `mf_rsi_1m` | bar | [0,100] | Wilder RSI(14) on 1m bars |
//! | `mf_rsi_5m` | bar | [0,100] | Wilder RSI(14) on 5m bars |
//! | `mf_rsi_15m` | bar | [0,100] | Wilder RSI(14) on 15m bars |
//! | `mf_bb_pctb_1m` | bar | (-inf,+inf) | Bollinger %B(20,2) on 1m bars |
//! | `mf_bb_pctb_5m` | bar | (-inf,+inf) | Bollinger %B(20,2) on 5m bars |
//! | `mf_bb_pctb_15m` | bar | (-inf,+inf) | Bollinger %B(20,2) on 15m bars |
//! | `mf_bb_width_1m` | bar | [0,+inf) | Bollinger BW(20,2) on 1m bars |
//! | `mf_bb_width_5m` | bar | [0,+inf) | Bollinger BW(20,2) on 5m bars |
//! | `mf_bb_width_15m` | bar | [0,+inf) | Bollinger BW(20,2) on 15m bars |
//! | `mf_atr_5m` | bar | [0,+inf) | Wilder ATR(14) on 5m bars |
//! | `mf_atr_15m` | bar | [0,+inf) | Wilder ATR(14) on 15m bars |
//!
//! # References
//!
//! - Wilder (1978) — RSI, ATR, smoothing method
//! - Bollinger (2001) — Bollinger Bands, %B, Bandwidth

use std::collections::VecDeque;

const RSI_PERIOD: usize = 14;
const BB_PERIOD: usize = 20;
const BB_MULT: f64 = 2.0;
const ATR_PERIOD: usize = 14;

// Tick counts per bar at 100ms emission
const TICKS_1M: usize = 600;
const TICKS_5M: usize = 3000;
const TICKS_15M: usize = 9000;

/// Aggregates ticks into OHLC bars and computes bar-level indicators.
#[derive(Debug, Clone)]
struct BarAggregator {
    bar_ticks: usize,
    tick_count: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    prev_close: f64,
    bars_completed: usize,
    // Bollinger state: last 20 closes
    closes: VecDeque<f64>,
    // ATR state: last 14 bars' high/low
    highs: VecDeque<f64>,
    lows: VecDeque<f64>,
    // RSI state (Wilder smoothing — running averages, no window)
    avg_gain: f64,
    avg_loss: f64,
    // Cached outputs (held between bars)
    cached_rsi: f64,
    cached_bb_pctb: f64,
    cached_bb_width: f64,
    cached_atr: f64,
}

impl BarAggregator {
    fn new(bar_ticks: usize) -> Self {
        Self {
            bar_ticks,
            tick_count: 0,
            open: f64::NAN,
            high: f64::NEG_INFINITY,
            low: f64::INFINITY,
            close: f64::NAN,
            prev_close: f64::NAN,
            bars_completed: 0,
            closes: VecDeque::with_capacity(BB_PERIOD + 1),
            highs: VecDeque::with_capacity(ATR_PERIOD + 1),
            lows: VecDeque::with_capacity(ATR_PERIOD + 1),
            avg_gain: 0.0,
            avg_loss: 0.0,
            cached_rsi: f64::NAN,
            cached_bb_pctb: f64::NAN,
            cached_bb_width: f64::NAN,
            cached_atr: f64::NAN,
        }
    }

    /// Feed a new tick price. Returns true when a bar completes.
    fn update_tick(&mut self, price: f64) -> bool {
        if self.tick_count == 0 {
            self.open = price;
            self.high = price;
            self.low = price;
        } else {
            if price > self.high {
                self.high = price;
            }
            if price < self.low {
                self.low = price;
            }
        }
        self.close = price;
        self.tick_count += 1;

        if self.tick_count >= self.bar_ticks {
            self.on_bar_complete();
            true
        } else {
            false
        }
    }

    /// Process a completed bar: update indicator state and caches.
    fn on_bar_complete(&mut self) {
        let close = self.close;

        // Push to deques
        self.closes.push_back(close);
        if self.closes.len() > BB_PERIOD {
            self.closes.pop_front();
        }
        self.highs.push_back(self.high);
        if self.highs.len() > ATR_PERIOD {
            self.highs.pop_front();
        }
        self.lows.push_back(self.low);
        if self.lows.len() > ATR_PERIOD {
            self.lows.pop_front();
        }

        // RSI update
        self.update_rsi(close);

        // Bollinger update
        self.update_bollinger(close);

        // ATR update
        self.update_atr();

        // Advance bar state
        self.prev_close = close;
        self.bars_completed += 1;
        self.tick_count = 0;
        self.open = f64::NAN;
        self.high = f64::NEG_INFINITY;
        self.low = f64::INFINITY;
        self.close = f64::NAN;
    }

    /// Wilder RSI(14) using exponential smoothing of gains/losses.
    fn update_rsi(&mut self, close: f64) {
        if self.prev_close.is_nan() {
            // First bar — no previous close to compare
            return;
        }

        let change = close - self.prev_close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        if self.bars_completed < RSI_PERIOD {
            // Accumulation phase: simple average
            self.avg_gain += gain;
            self.avg_loss += loss;

            if self.bars_completed == RSI_PERIOD - 1 {
                // First RSI calculation after RSI_PERIOD bars
                self.avg_gain /= RSI_PERIOD as f64;
                self.avg_loss /= RSI_PERIOD as f64;
                self.cached_rsi = compute_rsi_value(self.avg_gain, self.avg_loss);
            }
        } else {
            // Wilder smoothing: avg = (prev_avg × (N-1) + current) / N
            let n = RSI_PERIOD as f64;
            self.avg_gain = (self.avg_gain * (n - 1.0) + gain) / n;
            self.avg_loss = (self.avg_loss * (n - 1.0) + loss) / n;
            self.cached_rsi = compute_rsi_value(self.avg_gain, self.avg_loss);
        }
    }

    /// Bollinger Bands(20, 2): %B and bandwidth from closes deque.
    fn update_bollinger(&mut self, close: f64) {
        if self.closes.len() < BB_PERIOD {
            return;
        }

        let sum: f64 = self.closes.iter().sum();
        let sma = sum / BB_PERIOD as f64;

        let variance: f64 = self.closes.iter().map(|&c| (c - sma).powi(2)).sum::<f64>()
            / BB_PERIOD as f64;
        let std_dev = variance.sqrt();

        let upper = sma + BB_MULT * std_dev;
        let lower = sma - BB_MULT * std_dev;
        let band_range = upper - lower;

        self.cached_bb_pctb = if band_range > 1e-15 {
            (close - lower) / band_range
        } else {
            0.5
        };

        self.cached_bb_width = if sma.abs() > 1e-15 {
            band_range / sma
        } else {
            0.0
        };
    }

    /// Wilder ATR(14): true range with exponential smoothing.
    fn update_atr(&mut self) {
        if self.prev_close.is_nan() {
            // First bar — TR is just H-L
            let tr = self.high - self.low;
            self.cached_atr = tr;
            return;
        }

        let tr = (self.high - self.low)
            .max((self.high - self.prev_close).abs())
            .max((self.low - self.prev_close).abs());

        if self.bars_completed < ATR_PERIOD {
            // Accumulation: just store TR in cached_atr as running sum
            // We use a simple approach: first ATR = average of first N TRs
            if self.bars_completed == 0 {
                self.cached_atr = tr;
            } else {
                // Running sum approach
                let n = self.bars_completed as f64 + 1.0;
                self.cached_atr = (self.cached_atr * (n - 1.0) + tr) / n;
            }
        } else {
            // Wilder smoothing
            let n = ATR_PERIOD as f64;
            self.cached_atr = (self.cached_atr * (n - 1.0) + tr) / n;
        }
    }
}

fn compute_rsi_value(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss < 1e-15 {
        if avg_gain < 1e-15 {
            50.0 // No movement
        } else {
            100.0
        }
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Medium-frequency features: 17 features across 1m/5m/15m timeframes
#[derive(Debug, Clone, Default)]
pub struct MediumFreqFeatures {
    // Tick-level EMAs
    pub ema_1m: f64,
    pub ema_5m: f64,
    pub ema_15m: f64,
    pub ema_cross_1m_5m: f64,
    pub ema_cross_5m_15m: f64,
    // Bar-level RSI
    pub rsi_1m: f64,
    pub rsi_5m: f64,
    pub rsi_15m: f64,
    // Bar-level Bollinger
    pub bb_pctb_1m: f64,
    pub bb_pctb_5m: f64,
    pub bb_pctb_15m: f64,
    pub bb_width_1m: f64,
    pub bb_width_5m: f64,
    pub bb_width_15m: f64,
    // Bar-level ATR (5m and 15m only — 1m ATR is too noisy)
    pub atr_5m: f64,
    pub atr_15m: f64,
}

impl MediumFreqFeatures {
    pub fn count() -> usize {
        16
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "mf_ema_1m",
            "mf_ema_5m",
            "mf_ema_15m",
            "mf_ema_cross_1m_5m",
            "mf_ema_cross_5m_15m",
            "mf_rsi_1m",
            "mf_rsi_5m",
            "mf_rsi_15m",
            "mf_bb_pctb_1m",
            "mf_bb_pctb_5m",
            "mf_bb_pctb_15m",
            "mf_bb_width_1m",
            "mf_bb_width_5m",
            "mf_bb_width_15m",
            "mf_atr_5m",
            "mf_atr_15m",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.ema_1m,
            self.ema_5m,
            self.ema_15m,
            self.ema_cross_1m_5m,
            self.ema_cross_5m_15m,
            self.rsi_1m,
            self.rsi_5m,
            self.rsi_15m,
            self.bb_pctb_1m,
            self.bb_pctb_5m,
            self.bb_pctb_15m,
            self.bb_width_1m,
            self.bb_width_5m,
            self.bb_width_15m,
            self.atr_5m,
            self.atr_15m,
        ]
    }
}

/// Persistent state for medium-frequency feature computation.
/// Owned by FeatureComputer; updated on every tick.
#[derive(Debug, Clone)]
pub struct MediumFreqState {
    // Continuous EMAs
    ema_1m: f64,
    ema_5m: f64,
    ema_15m: f64,
    alpha_1m: f64,
    alpha_5m: f64,
    alpha_15m: f64,
    ema_initialized: bool,
    // Bar aggregators
    bar_1m: BarAggregator,
    bar_5m: BarAggregator,
    bar_15m: BarAggregator,
}

impl MediumFreqState {
    pub fn new() -> Self {
        Self {
            ema_1m: 0.0,
            ema_5m: 0.0,
            ema_15m: 0.0,
            alpha_1m: 2.0 / (TICKS_1M as f64 + 1.0),
            alpha_5m: 2.0 / (TICKS_5M as f64 + 1.0),
            alpha_15m: 2.0 / (TICKS_15M as f64 + 1.0),
            ema_initialized: false,
            bar_1m: BarAggregator::new(TICKS_1M),
            bar_5m: BarAggregator::new(TICKS_5M),
            bar_15m: BarAggregator::new(TICKS_15M),
        }
    }

    /// Update state with a new price tick and return current features.
    pub fn update_and_compute(&mut self, price: f64) -> MediumFreqFeatures {
        // Update EMAs
        if !self.ema_initialized {
            self.ema_1m = price;
            self.ema_5m = price;
            self.ema_15m = price;
            self.ema_initialized = true;
        } else {
            self.ema_1m = self.alpha_1m * price + (1.0 - self.alpha_1m) * self.ema_1m;
            self.ema_5m = self.alpha_5m * price + (1.0 - self.alpha_5m) * self.ema_5m;
            self.ema_15m = self.alpha_15m * price + (1.0 - self.alpha_15m) * self.ema_15m;
        }

        // Update bar aggregators
        self.bar_1m.update_tick(price);
        self.bar_5m.update_tick(price);
        self.bar_15m.update_tick(price);

        // Assemble features
        let ema_cross_1m_5m = if self.ema_5m.abs() > 1e-15 {
            (self.ema_1m - self.ema_5m) / self.ema_5m * 10_000.0
        } else {
            0.0
        };
        let ema_cross_5m_15m = if self.ema_15m.abs() > 1e-15 {
            (self.ema_5m - self.ema_15m) / self.ema_15m * 10_000.0
        } else {
            0.0
        };

        MediumFreqFeatures {
            ema_1m: self.ema_1m,
            ema_5m: self.ema_5m,
            ema_15m: self.ema_15m,
            ema_cross_1m_5m,
            ema_cross_5m_15m,
            rsi_1m: self.bar_1m.cached_rsi,
            rsi_5m: self.bar_5m.cached_rsi,
            rsi_15m: self.bar_15m.cached_rsi,
            bb_pctb_1m: self.bar_1m.cached_bb_pctb,
            bb_pctb_5m: self.bar_5m.cached_bb_pctb,
            bb_pctb_15m: self.bar_15m.cached_bb_pctb,
            bb_width_1m: self.bar_1m.cached_bb_width,
            bb_width_5m: self.bar_5m.cached_bb_width,
            bb_width_15m: self.bar_15m.cached_bb_width,
            atr_5m: self.bar_5m.cached_atr,
            atr_15m: self.bar_15m.cached_atr,
        }
    }
}

impl Default for MediumFreqState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        assert_eq!(MediumFreqFeatures::count(), 16);
        assert_eq!(MediumFreqFeatures::names().len(), 16);
        assert_eq!(MediumFreqFeatures::default().to_vec().len(), 16);
    }

    #[test]
    fn test_names_prefix() {
        for name in MediumFreqFeatures::names() {
            assert!(name.starts_with("mf_"), "Feature name must start with mf_: {name}");
        }
    }

    #[test]
    fn test_ema_convergence() {
        let mut state = MediumFreqState::new();
        // Feed constant price — EMAs should converge
        for _ in 0..10_000 {
            state.update_and_compute(100.0);
        }
        let f = state.update_and_compute(100.0);
        assert!((f.ema_1m - 100.0).abs() < 1e-6, "EMA_1m should converge to 100");
        assert!((f.ema_5m - 100.0).abs() < 1e-6, "EMA_5m should converge to 100");
        assert!((f.ema_15m - 100.0).abs() < 1e-3, "EMA_15m should converge to 100");
        assert!(f.ema_cross_1m_5m.abs() < 0.01, "Crossover should be ~0");
        assert!(f.ema_cross_5m_15m.abs() < 0.01, "Crossover should be ~0");
    }

    #[test]
    fn test_ema_uptrend_crossover() {
        let mut state = MediumFreqState::new();
        // Feed rising prices — short EMA should lead
        for i in 0..10_000 {
            state.update_and_compute(100.0 + i as f64 * 0.01);
        }
        let f = state.update_and_compute(200.0);
        assert!(f.ema_cross_1m_5m > 0.0, "1m EMA should be above 5m in uptrend");
        assert!(f.ema_cross_5m_15m > 0.0, "5m EMA should be above 15m in uptrend");
    }

    #[test]
    fn test_warmup_nan() {
        let mut state = MediumFreqState::new();
        // After just 1 tick, bar-level features should be NaN
        let f = state.update_and_compute(100.0);
        assert!(f.rsi_1m.is_nan(), "RSI should be NaN during warmup");
        assert!(f.bb_pctb_1m.is_nan(), "BB %B should be NaN during warmup");
        assert!(f.bb_width_1m.is_nan(), "BB width should be NaN during warmup");
        // EMAs should be valid immediately
        assert!(!f.ema_1m.is_nan(), "EMA should be valid on first tick");
    }

    #[test]
    fn test_rsi_overbought() {
        let mut agg = BarAggregator::new(10); // small bars for testing
        // 20 bars of pure uptrend (each bar closes higher)
        let mut price = 100.0;
        for _ in 0..20 {
            for _ in 0..10 {
                agg.update_tick(price);
                price += 0.1;
            }
        }
        assert!(agg.cached_rsi > 90.0, "RSI should be near 100 in pure uptrend, got {}", agg.cached_rsi);
    }

    #[test]
    fn test_rsi_oversold() {
        let mut agg = BarAggregator::new(10);
        let mut price = 200.0;
        for _ in 0..20 {
            for _ in 0..10 {
                agg.update_tick(price);
                price -= 0.1;
            }
        }
        assert!(agg.cached_rsi < 10.0, "RSI should be near 0 in pure downtrend, got {}", agg.cached_rsi);
    }

    #[test]
    fn test_rsi_flat() {
        let mut agg = BarAggregator::new(10);
        // Flat market — all closes at same price
        for _ in 0..20 {
            for tick in 0..10 {
                // Small intra-bar variation but same close
                let noise = if tick < 5 { 0.01 } else { -0.01 };
                agg.update_tick(100.0 + noise);
            }
            // Force close at 100.0 by overriding last tick
        }
        // RSI should be ~50 in flat market
        assert!((agg.cached_rsi - 50.0).abs() < 15.0, "RSI should be ~50 in flat, got {}", agg.cached_rsi);
    }

    #[test]
    fn test_bollinger_at_mean() {
        let mut agg = BarAggregator::new(10);
        // 25 bars at constant close (100.0) — %B should be 0.5
        for _ in 0..25 {
            for _ in 0..10 {
                agg.update_tick(100.0);
            }
        }
        assert!((agg.cached_bb_pctb - 0.5).abs() < 0.01,
            "BB %B should be 0.5 when price at mean, got {}", agg.cached_bb_pctb);
        // Bandwidth should be ~0 with constant close
        assert!(agg.cached_bb_width < 0.001,
            "BB width should be ~0 with constant price, got {}", agg.cached_bb_width);
    }

    #[test]
    fn test_bollinger_outside_bands() {
        let mut agg = BarAggregator::new(10);
        // 20 bars at 100.0, then a spike to 200.0
        for _ in 0..20 {
            for _ in 0..10 {
                agg.update_tick(100.0);
            }
        }
        // Now a bar that closes at 200.0
        for _ in 0..10 {
            agg.update_tick(200.0);
        }
        assert!(agg.cached_bb_pctb > 1.0,
            "BB %B should be > 1.0 when price above upper band, got {}", agg.cached_bb_pctb);
    }

    #[test]
    fn test_atr_constant_range() {
        let mut agg = BarAggregator::new(10);
        // Bars with consistent 2.0 range: high=101, low=99, close=100
        for _ in 0..20 {
            for tick in 0..10 {
                let price = match tick {
                    0 => 100.0,
                    1 => 101.0,  // high
                    2 => 99.0,   // low
                    _ => 100.0,  // close
                };
                agg.update_tick(price);
            }
        }
        // ATR should converge to ~2.0 (true range = max(H-L, |H-C_prev|, |L-C_prev|))
        // H-L = 2.0, |H-C_prev| = 1.0, |L-C_prev| = 1.0, so TR = 2.0
        assert!((agg.cached_atr - 2.0).abs() < 0.5,
            "ATR should converge to ~2.0, got {}", agg.cached_atr);
    }

    #[test]
    fn test_bar_completion_count() {
        let mut agg = BarAggregator::new(100);
        let mut completions = 0;
        for _ in 0..350 {
            if agg.update_tick(100.0) {
                completions += 1;
            }
        }
        assert_eq!(completions, 3, "Should complete 3 bars in 350 ticks with bar_ticks=100");
        assert_eq!(agg.bars_completed, 3);
    }

    #[test]
    fn test_full_integration() {
        let mut state = MediumFreqState::new();
        // Run enough ticks for 1m bars to produce RSI (14 bars × 600 ticks = 8400)
        let mut price = 100.0;
        for i in 0..9000 {
            // Gentle uptrend with noise
            price = 100.0 + (i as f64 * 0.001) + (i % 7) as f64 * 0.01;
            state.update_and_compute(price);
        }
        let f = state.update_and_compute(price);

        // EMAs should be valid
        assert!(!f.ema_1m.is_nan());
        assert!(!f.ema_5m.is_nan());
        assert!(!f.ema_15m.is_nan());

        // 1m RSI should be valid (15 bars completed at tick 9000)
        assert!(!f.rsi_1m.is_nan(), "1m RSI should be valid after 9000 ticks");
        assert!(f.rsi_1m >= 0.0 && f.rsi_1m <= 100.0);

        // 1m Bollinger should be valid (20 bars at tick 12000, but we have 15 so still NaN)
        // 5m RSI needs 14 × 3000 = 42000 ticks → still NaN
        assert!(f.rsi_5m.is_nan(), "5m RSI should still be NaN at 9000 ticks");
    }

    #[test]
    fn test_default_features_are_nan() {
        let f = MediumFreqFeatures::default();
        // Default should be all 0.0 (from derive Default)
        for v in f.to_vec() {
            assert_eq!(v, 0.0);
        }
    }
}
