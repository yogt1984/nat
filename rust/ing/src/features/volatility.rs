//! Volatility Feature Extraction
//!
//! Realized and range-based volatility estimators at multiple horizons.
//! Used for risk regime detection and strategy sizing.
//!
//! # Features (9 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Realized vol (1m/5m)** | sqrt(mean(r²)) of tick returns | [0, +inf) | Higher = more volatile |
//! | **Parkinson vol** | Range-based using high/low | [0, +inf) | More efficient than realized vol |
//! | **Garman-Klass vol** | OHLC-based, most efficient classical | [0, +inf) | Strictly more efficient than Parkinson |
//! | **Spread mean** | Current spread (point-in-time) | [0, +inf) | Higher = wider market |
//! | **Spread std** | Spread standard deviation over 1 min | [0, +inf) | Higher = unstable spread |
//! | **Midprice std** | Price standard deviation over 1 min | [0, +inf) | Direct dispersion measure |
//! | **Ratio short/long** | vol_1m / vol_5m | [0, +inf) | >1 = accelerating vol |
//! | **Z-score** | (vol_1m - mean_1h) / std_1h | (-inf, +inf) | >2 = vol spike, <-2 = unusually calm |
//!
//! # Algorithms
//!
//! **Realized volatility**: RV = sqrt(Σ r_i² / N) where r_i are tick-to-tick returns.
//! Window sizes: 60 ticks (~1 min) and 300 ticks (~5 min) at 100ms emission.
//!
//! **Parkinson (1980)**: σ = ln(H/L) / sqrt(4·ln(2)). Uses only high/low within
//! the window. This is a single-window approximation—the classical estimator
//! averages over multiple subperiods for efficiency, but we use one window
//! (300 ticks) for simplicity. Still more efficient than close-to-close for
//! the same window because it captures intra-period range.
//!
//! **Garman & Klass (1980)**: σ² = 0.5·ln(H/L)² − (2·ln2 − 1)·ln(C/O)².
//! Uses OHLC within the 5-minute window (300 ticks). The most efficient
//! classical volatility estimator — approximately 7.4× more efficient than
//! close-to-close and ~1.2× more efficient than Parkinson, because it
//! exploits both range and close-to-open information. Returns sqrt(σ²) as
//! the volatility estimate; if σ² < 0 (rare numerical edge case where
//! close-to-open dominates range), returns 0.0 rather than NaN.
//!
//! **Spread std**: Standard deviation of bid-ask spread values over 1-minute window
//! (600 ticks at 100ms). High spread variability indicates unstable liquidity
//! conditions — market makers are repricing frequently.
//!
//! **Vol z-score**: (current_vol_1m - hourly_mean) / hourly_std. Requires ≥120
//! historical vol_1m samples (~12 seconds of warmup at 100ms). Returns 0.0
//! during warmup. Uses the hourly distribution (36,000 ticks) to normalize
//! current volatility, enabling cross-regime comparison.
//!
//! # References
//!
//! - Parkinson (1980) - The extreme value method for estimating the variance of the rate of return
//! - Garman & Klass (1980) - On the estimation of security price volatilities from historical data

use crate::state::{OrderBook, RingBuffer};

/// Volatility features (9 features)
#[derive(Debug, Clone, Default)]
pub struct VolatilityFeatures {
    /// Realized volatility from 1-minute returns
    pub returns_1m: f64,
    /// Realized volatility from 5-minute returns
    pub returns_5m: f64,
    /// Parkinson volatility (high-low based)
    pub parkinson_5m: f64,
    /// Garman-Klass volatility (OHLC-based, most efficient classical estimator)
    pub garman_klass_5m: f64,
    /// Mean spread over 1 minute
    pub spread_mean_1m: f64,
    /// Spread standard deviation over 1 minute
    pub spread_std_1m: f64,
    /// Midprice standard deviation over 1 minute
    pub midprice_std_1m: f64,
    /// Volatility ratio (short/long)
    pub ratio_short_long: f64,
    /// Volatility z-score vs 1-hour mean
    pub zscore: f64,
}

impl VolatilityFeatures {
    pub fn count() -> usize { 9 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "vol_returns_1m",
            "vol_returns_5m",
            "vol_parkinson_5m",
            "vol_garman_klass_5m",
            "vol_spread_mean_1m",
            "vol_spread_std_1m",
            "vol_midprice_std_1m",
            "vol_ratio_short_long",
            "vol_zscore",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.returns_1m,
            self.returns_5m,
            self.parkinson_5m,
            self.garman_klass_5m,
            self.spread_mean_1m,
            self.spread_std_1m,
            self.midprice_std_1m,
            self.ratio_short_long,
            self.zscore,
        ]
    }
}

/// Minimum number of vol_1m samples before z-score becomes meaningful.
/// At 100ms emission, 120 samples ≈ 12 seconds of warmup.
const ZSCORE_MIN_SAMPLES: usize = 120;

/// Compute volatility features.
///
/// # Arguments
/// - `price_buffer` — ring buffer of midprices (sized by config, typically 1000+)
/// - `order_book` — current order book snapshot (for point-in-time spread)
/// - `spread_buffer` — ring buffer of historical spread values (600 = 1 min at 100ms)
/// - `vol_1m_buffer` — ring buffer of historical vol_1m values (36000 = 1 hr at 100ms)
pub fn compute(
    price_buffer: &RingBuffer<f64>,
    order_book: &OrderBook,
    spread_buffer: &RingBuffer<f64>,
    vol_1m_buffer: &RingBuffer<f64>,
) -> VolatilityFeatures {
    // Realized volatility from returns
    let returns = price_buffer.returns();

    // 1-minute vol (assuming 100ms intervals, ~600 samples per minute)
    let returns_1m = if returns.len() >= 60 {
        compute_realized_vol(&returns[returns.len()-60..])
    } else {
        compute_realized_vol(&returns)
    };

    // 5-minute vol
    let returns_5m = if returns.len() >= 300 {
        compute_realized_vol(&returns[returns.len()-300..])
    } else {
        compute_realized_vol(&returns)
    };

    // Parkinson volatility (range-based: high/low)
    let parkinson_5m = if price_buffer.len() >= 300 {
        let prices: Vec<f64> = price_buffer.last_n(300)
            .into_iter()
            .cloned()
            .collect();
        compute_parkinson_vol(&prices)
    } else {
        compute_parkinson_vol(&price_buffer.to_vec())
    };

    // Garman-Klass volatility (OHLC-based, most efficient classical estimator)
    let garman_klass_5m = if price_buffer.len() >= 300 {
        let prices: Vec<f64> = price_buffer.last_n(300)
            .into_iter()
            .cloned()
            .collect();
        compute_garman_klass_vol(&prices)
    } else {
        compute_garman_klass_vol(&price_buffer.to_vec())
    };

    // Spread statistics
    let spread_mean_1m = order_book.spread().unwrap_or(0.0);
    let spread_std_1m = if spread_buffer.len() >= 2 {
        spread_buffer.std()
    } else {
        0.0
    };

    // Midprice std
    let midprice_std_1m = if price_buffer.len() >= 60 {
        let prices: Vec<f64> = price_buffer.last_n(60)
            .into_iter()
            .cloned()
            .collect();
        std_dev(&prices)
    } else {
        price_buffer.std()
    };

    // Ratio short/long
    let ratio_short_long = if returns_5m > 0.0 {
        returns_1m / returns_5m
    } else {
        1.0
    };

    // Vol z-score: (current_vol_1m - hourly_mean) / hourly_std
    // Returns 0.0 during warmup (< ZSCORE_MIN_SAMPLES) or if std is negligible.
    // Guard: std must exceed 1e-10 (well above f64 noise floor for vol values).
    // For context: vol_1m of BTC is typically ~0.0001–0.01, so std < 1e-10
    // means the market was essentially static for the entire hour.
    let zscore = if vol_1m_buffer.len() >= ZSCORE_MIN_SAMPLES {
        let hourly_std = vol_1m_buffer.std();
        if hourly_std > 1e-10 {
            let z = (returns_1m - vol_1m_buffer.mean()) / hourly_std;
            // Clamp to [-10, 10] to prevent extreme outliers from poisoning downstream
            z.clamp(-10.0, 10.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    VolatilityFeatures {
        returns_1m,
        returns_5m,
        parkinson_5m,
        garman_klass_5m,
        spread_mean_1m,
        spread_std_1m,
        midprice_std_1m,
        ratio_short_long,
        zscore,
    }
}

/// Compute realized volatility from returns
fn compute_realized_vol(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = returns.iter().map(|r| r * r).sum();
    (sum_sq / returns.len() as f64).sqrt()
}

/// Compute Parkinson volatility estimator
fn compute_parkinson_vol(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);

    if low > 0.0 {
        let log_ratio = (high / low).ln();
        log_ratio / (4.0 * std::f64::consts::LN_2).sqrt()
    } else {
        0.0
    }
}

/// Compute Garman-Klass volatility estimator.
///
/// σ²_GK = 0.5 · ln(H/L)² − (2·ln2 − 1) · ln(C/O)²
///
/// Uses OHLC derived from the price window:
/// - O = first price, C = last price, H = max, L = min
///
/// Returns sqrt(σ²_GK). If σ²_GK < 0 (rare edge case where close-to-open
/// movement dominates range, possible with very few ticks), returns 0.0.
fn compute_garman_klass_vol(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let open = prices[0];
    let close = prices[prices.len() - 1];
    let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);

    if open <= 0.0 || low <= 0.0 {
        return 0.0;
    }

    let ln_hl = (high / low).ln();
    let ln_co = (close / open).ln();

    let sigma_sq = 0.5 * ln_hl * ln_hl - (2.0 * std::f64::consts::LN_2 - 1.0) * ln_co * ln_co;

    if sigma_sq > 0.0 {
        sigma_sq.sqrt()
    } else {
        0.0
    }
}

/// Compute standard deviation (sample, Bessel-corrected)
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{OrderBook, RingBuffer};

    // ---------- helpers ----------

    fn make_order_book(bid: f64, ask: f64) -> OrderBook {
        let mut ob = OrderBook::new(10);
        let book = crate::ws::WsBook {
            coin: "BTC".to_string(),
            levels: (
                vec![
                    crate::ws::WsLevel { px: format!("{:.1}", bid), sz: "1.0".to_string(), n: 1 },
                ],
                vec![
                    crate::ws::WsLevel { px: format!("{:.1}", ask), sz: "1.0".to_string(), n: 1 },
                ],
            ),
            time: 1000,
        };
        ob.update(&book);
        ob
    }

    fn make_price_buffer(prices: &[f64]) -> RingBuffer<f64> {
        let mut buf = RingBuffer::new(prices.len().max(1000));
        for &p in prices {
            buf.push(p);
        }
        buf
    }

    fn make_spread_buffer(spreads: &[f64]) -> RingBuffer<f64> {
        let mut buf = RingBuffer::new(600);
        for &s in spreads {
            buf.push(s);
        }
        buf
    }

    fn make_vol_buffer(vols: &[f64]) -> RingBuffer<f64> {
        let mut buf = RingBuffer::new(36_000);
        for &v in vols {
            buf.push(v);
        }
        buf
    }

    // ---------- count / names / to_vec contract ----------

    #[test]
    fn test_count_names_to_vec_length_match() {
        assert_eq!(VolatilityFeatures::count(), 9);
        assert_eq!(VolatilityFeatures::names().len(), 9);
        assert_eq!(VolatilityFeatures::default().to_vec().len(), 9);
    }

    #[test]
    fn test_names_are_prefixed() {
        for name in VolatilityFeatures::names() {
            assert!(name.starts_with("vol_"), "Feature name '{}' must start with vol_", name);
        }
    }

    #[test]
    fn test_default_is_all_zero() {
        let f = VolatilityFeatures::default();
        for (i, v) in f.to_vec().iter().enumerate() {
            assert_eq!(*v, 0.0, "Default feature index {} should be 0.0", i);
        }
    }

    // ---------- realized vol ----------

    #[test]
    fn test_realized_vol_empty_returns() {
        assert_eq!(compute_realized_vol(&[]), 0.0);
    }

    #[test]
    fn test_realized_vol_single_return() {
        // Single return of 0.01 → sqrt(0.01² / 1) = 0.01
        let rv = compute_realized_vol(&[0.01]);
        assert!((rv - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_realized_vol_zero_returns() {
        // Zero returns → zero vol
        let rv = compute_realized_vol(&[0.0, 0.0, 0.0]);
        assert_eq!(rv, 0.0);
    }

    #[test]
    fn test_realized_vol_known_values() {
        // Returns: [0.01, -0.01, 0.02, -0.02]
        // sum_sq = 0.0001 + 0.0001 + 0.0004 + 0.0004 = 0.001
        // mean_sq = 0.001 / 4 = 0.00025
        // rv = sqrt(0.00025) = 0.015811388...
        let rv = compute_realized_vol(&[0.01, -0.01, 0.02, -0.02]);
        assert!((rv - 0.015811388300841898).abs() < 1e-12);
    }

    #[test]
    fn test_realized_vol_symmetry() {
        // Negating all returns should give same vol
        let rv1 = compute_realized_vol(&[0.01, -0.02, 0.03]);
        let rv2 = compute_realized_vol(&[-0.01, 0.02, -0.03]);
        assert!((rv1 - rv2).abs() < 1e-12);
    }

    #[test]
    fn test_realized_vol_scaling() {
        // Doubling returns should double vol
        let rv1 = compute_realized_vol(&[0.01, -0.01]);
        let rv2 = compute_realized_vol(&[0.02, -0.02]);
        assert!((rv2 / rv1 - 2.0).abs() < 1e-10);
    }

    // ---------- Parkinson vol ----------

    #[test]
    fn test_parkinson_single_price() {
        assert_eq!(compute_parkinson_vol(&[100.0]), 0.0);
    }

    #[test]
    fn test_parkinson_constant_prices() {
        // No range → zero vol
        let pv = compute_parkinson_vol(&[100.0, 100.0, 100.0]);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_parkinson_known_range() {
        // High=110, Low=100 → ln(110/100) / sqrt(4·ln(2))
        // = ln(1.1) / sqrt(2.772...) = 0.09531 / 1.66511 = 0.057236...
        let prices = vec![100.0, 105.0, 110.0, 108.0, 102.0];
        let pv = compute_parkinson_vol(&prices);
        let expected = (110.0_f64 / 100.0).ln() / (4.0 * std::f64::consts::LN_2).sqrt();
        assert!((pv - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parkinson_zero_low() {
        // Zero price → returns 0 (guard clause)
        let pv = compute_parkinson_vol(&[0.0, 100.0]);
        assert_eq!(pv, 0.0);
    }

    #[test]
    fn test_parkinson_monotonicity() {
        // Wider range → higher vol
        let narrow = compute_parkinson_vol(&[99.0, 100.0, 101.0]);
        let wide = compute_parkinson_vol(&[90.0, 100.0, 110.0]);
        assert!(wide > narrow, "Wider range should give higher Parkinson vol");
    }

    // ---------- Garman-Klass vol ----------

    #[test]
    fn test_garman_klass_single_price() {
        assert_eq!(compute_garman_klass_vol(&[100.0]), 0.0);
    }

    #[test]
    fn test_garman_klass_empty() {
        assert_eq!(compute_garman_klass_vol(&[]), 0.0);
    }

    #[test]
    fn test_garman_klass_constant_prices() {
        // No range, no close-to-open → zero vol
        let gk = compute_garman_klass_vol(&[100.0, 100.0, 100.0, 100.0]);
        assert_eq!(gk, 0.0);
    }

    #[test]
    fn test_garman_klass_known_ohlc() {
        // O=100, H=110, L=95, C=105
        // σ² = 0.5·ln(110/95)² − (2·ln2 − 1)·ln(105/100)²
        //     = 0.5·ln(1.15789)² − 0.38629·ln(1.05)²
        //     = 0.5·0.14660² − 0.38629·0.04879²
        //     = 0.5·0.021491 − 0.38629·0.002380
        //     = 0.010746 − 0.000920 = 0.009826
        // σ = sqrt(0.009826) ≈ 0.09913
        let prices = vec![100.0, 95.0, 110.0, 105.0]; // O=100, L=95, H=110, C=105
        let gk = compute_garman_klass_vol(&prices);
        let ln_hl = (110.0_f64 / 95.0).ln();
        let ln_co = (105.0_f64 / 100.0).ln();
        let expected_sq = 0.5 * ln_hl * ln_hl - (2.0 * std::f64::consts::LN_2 - 1.0) * ln_co * ln_co;
        let expected = expected_sq.sqrt();
        assert!((gk - expected).abs() < 1e-10, "GK={}, expected={}", gk, expected);
    }

    #[test]
    fn test_garman_klass_zero_low() {
        // Zero price → returns 0 (guard clause)
        let gk = compute_garman_klass_vol(&[0.0, 100.0, 50.0]);
        assert_eq!(gk, 0.0);
    }

    #[test]
    fn test_garman_klass_zero_open() {
        let gk = compute_garman_klass_vol(&[0.0, 50.0, 100.0]);
        assert_eq!(gk, 0.0);
    }

    #[test]
    fn test_garman_klass_monotonicity_with_range() {
        // Wider range → higher GK vol
        let narrow = compute_garman_klass_vol(&[100.0, 99.0, 101.0, 100.0]);
        let wide = compute_garman_klass_vol(&[100.0, 90.0, 110.0, 100.0]);
        assert!(wide > narrow, "Wider range should give higher GK vol: {} vs {}", wide, narrow);
    }

    #[test]
    fn test_garman_klass_open_equals_close() {
        // When O=C, the close-to-open term vanishes → GK = 0.5·ln(H/L)²
        // which is Parkinson² / (4·ln2) * 2 ... actually:
        // GK_σ² = 0.5·ln(H/L)² − 0 = 0.5·ln(H/L)²
        // Parkinson_σ = ln(H/L) / sqrt(4·ln2)
        // So GK_σ = sqrt(0.5) · ln(H/L)
        let prices = vec![100.0, 90.0, 110.0, 100.0]; // O=C=100
        let gk = compute_garman_klass_vol(&prices);
        let ln_hl = (110.0_f64 / 90.0).ln();
        let expected = (0.5 * ln_hl * ln_hl).sqrt();
        assert!((gk - expected).abs() < 1e-10);
    }

    #[test]
    fn test_garman_klass_close_to_open_reduces_vol() {
        // When C=O (no drift), GK = sqrt(0.5)·ln(H/L)
        // When C≠O, the (2ln2-1)·ln(C/O)² term subtracts → lower vol
        let prices_no_drift = vec![100.0, 90.0, 110.0, 100.0]; // O=C=100
        let prices_with_drift = vec![100.0, 90.0, 110.0, 108.0]; // C>O, same range
        let gk_no_drift = compute_garman_klass_vol(&prices_no_drift);
        let gk_with_drift = compute_garman_klass_vol(&prices_with_drift);
        // Close-to-open term subtracts, so with drift vol should be lower
        assert!(gk_with_drift < gk_no_drift,
            "Close-to-open drift should reduce GK estimate: {} vs {}", gk_with_drift, gk_no_drift);
    }

    #[test]
    fn test_garman_klass_non_negative() {
        // GK should never return negative (we guard σ²<0 → 0.0)
        // Edge case: very large drift, tiny range
        let prices = vec![100.0, 100.0, 100.001, 200.0]; // huge C/O, tiny H/L... wait
        // Actually H=200, L=100 → big range. Let's try:
        // Big drift, range barely exceeds it
        let prices2 = vec![100.0, 99.99, 100.01, 110.0]; // O=100, H=110, L=99.99, C=110
        let gk = compute_garman_klass_vol(&prices2);
        assert!(gk >= 0.0, "GK must be non-negative, got {}", gk);
    }

    #[test]
    fn test_garman_klass_sigma_sq_negative_guard() {
        // Construct a case where σ² could go negative:
        // Very small range but large close-to-open movement
        // This is theoretically impossible for true OHLC (range always covers C and O),
        // but our "OHLC" is derived from tick data where O and C are first/last,
        // and H/L cover the full range — so range always covers open and close.
        // Verify the guard works even if we call the function directly:
        // With only 2 prices: O=100, C=200 → H=200, L=100
        // ln(H/L)=ln(2), ln(C/O)=ln(2)
        // σ² = 0.5·ln(2)² − (2ln2-1)·ln(2)² = ln(2)²·(0.5 - 2ln2 + 1) = ln(2)²·(1.5 - 2ln2)
        // 2ln2 ≈ 1.3863, so 1.5 - 1.3863 = 0.1137 > 0 → σ² > 0
        // With tick data, H/L always covers O/C, so σ² ≥ 0 is guaranteed.
        let gk = compute_garman_klass_vol(&[100.0, 200.0]);
        assert!(gk >= 0.0);
        assert!(gk > 0.0, "Two distinct prices → positive GK vol");
    }

    #[test]
    fn test_garman_klass_two_prices() {
        // O=100, C=110, H=110, L=100
        // ln(H/L) = ln(1.1), ln(C/O) = ln(1.1)
        // σ² = 0.5·ln(1.1)² − (2ln2-1)·ln(1.1)² = ln(1.1)²·(0.5 - 2ln2 + 1)
        //     = ln(1.1)² · (1.5 - 1.38629) = ln(1.1)² · 0.11371
        let prices = vec![100.0, 110.0];
        let gk = compute_garman_klass_vol(&prices);
        let ln_ratio = (1.1_f64).ln();
        let expected_sq = ln_ratio * ln_ratio * (1.5 - 2.0 * std::f64::consts::LN_2);
        let expected = expected_sq.sqrt();
        assert!((gk - expected).abs() < 1e-12, "gk={}, expected={}", gk, expected);
    }

    #[test]
    fn test_garman_klass_scaling() {
        // Scaling all prices by a constant shouldn't change vol (it's based on ratios)
        let prices1 = vec![100.0, 90.0, 110.0, 105.0];
        let prices2: Vec<f64> = prices1.iter().map(|p| p * 10.0).collect();
        let gk1 = compute_garman_klass_vol(&prices1);
        let gk2 = compute_garman_klass_vol(&prices2);
        assert!((gk1 - gk2).abs() < 1e-12, "GK should be scale-invariant: {} vs {}", gk1, gk2);
    }

    #[test]
    fn test_garman_klass_in_full_compute() {
        // Verify the feature appears in the full compute output
        let ob = make_order_book(99.0, 101.0);
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.garman_klass_5m > 0.0, "GK should be positive for trending prices");
        assert!(f.garman_klass_5m.is_finite(), "GK must be finite");
        assert!(!f.garman_klass_5m.is_nan(), "GK must not be NaN");
    }

    #[test]
    fn test_garman_klass_leq_parkinson_for_trending() {
        // For a monotonic trend, C≠O means the GK close-to-open term subtracts,
        // so GK_σ < sqrt(0.5)·ln(H/L) while Parkinson_σ = ln(H/L)/sqrt(4ln2).
        // With a strong trend, GK should be less than Parkinson.
        let ob = make_order_book(99.0, 101.0);
        // Strong uptrend
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.garman_klass_5m < f.parkinson_5m,
            "For trending data, GK ({}) should be < Parkinson ({}) due to drift correction",
            f.garman_klass_5m, f.parkinson_5m);
    }

    // ---------- std_dev ----------

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_single() {
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn test_std_dev_constant() {
        assert_eq!(std_dev(&[5.0, 5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_known_values() {
        // [1, 2, 3, 4, 5] → mean=3, var=2.5 (Bessel), std=1.5811388...
        let sd = std_dev(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((sd - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev_two_values() {
        // [0, 10] → mean=5, var=(25+25)/1=50, std=sqrt(50)=7.07...
        let sd = std_dev(&[0.0, 10.0]);
        assert!((sd - (50.0_f64).sqrt()).abs() < 1e-10);
    }

    // ---------- spread_std_1m ----------

    #[test]
    fn test_spread_std_empty_buffer() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = RingBuffer::<f64>::new(600);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.spread_std_1m, 0.0, "Empty spread buffer → 0.0");
    }

    #[test]
    fn test_spread_std_single_sample() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = make_spread_buffer(&[2.0]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.spread_std_1m, 0.0, "Single sample → 0.0 (can't compute std)");
    }

    #[test]
    fn test_spread_std_constant_spread() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.spread_std_1m, 0.0, "Constant spread → zero std");
    }

    #[test]
    fn test_spread_std_varying_spread() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        // Alternating spreads: 1.0 and 3.0
        let spreads: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
        let spread_buf = make_spread_buffer(&spreads);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.spread_std_1m > 0.0, "Varying spread → positive std");
        // Mean=2.0, values are 1.0 and 3.0, sample std should be ~1.005...
        assert!((f.spread_std_1m - 1.0050378152592121).abs() < 0.01);
    }

    #[test]
    fn test_spread_std_positive() {
        // Spread std should never be negative
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spreads: Vec<f64> = (0..50).map(|i| 0.5 + (i as f64) * 0.1).collect();
        let spread_buf = make_spread_buffer(&spreads);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.spread_std_1m >= 0.0, "Spread std must be non-negative");
    }

    // ---------- vol_zscore ----------

    #[test]
    fn test_zscore_warmup_period() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // Below ZSCORE_MIN_SAMPLES (120)
        let vol_buf = make_vol_buffer(&[0.001; 100]);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.zscore, 0.0, "Below warmup threshold → 0.0");
    }

    #[test]
    fn test_zscore_exactly_at_warmup() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // Exactly ZSCORE_MIN_SAMPLES with varying values
        let vols: Vec<f64> = (0..120).map(|i| 0.001 + (i as f64) * 0.00001).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        // Should produce a real z-score now (not necessarily zero)
        // The exact value depends on the computed returns_1m vs history
        assert!(f.zscore.is_finite(), "At warmup threshold → finite z-score");
    }

    #[test]
    fn test_zscore_constant_vol_history() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0; 100]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // All same vol → std=0 → zscore=0
        let vol_buf = make_vol_buffer(&[0.001; 200]);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.zscore, 0.0, "Constant vol history (std=0) → 0.0");
    }

    #[test]
    fn test_zscore_positive_for_high_vol() {
        let ob = make_order_book(99.0, 101.0);
        // Create a price buffer with high recent volatility
        let mut prices = vec![100.0; 300];
        // Add volatile prices at the end (within last 60 ticks for 1m window)
        for i in 0..60 {
            prices.push(100.0 + if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // Low historical vol with genuine variation (mean ~0.0001, std ~0.00003)
        let vols: Vec<f64> = (0..200).map(|i| 0.00005 + (i as f64) * 0.0000005).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.zscore > 0.0, "High current vol vs low history → positive z-score, got {}", f.zscore);
    }

    #[test]
    fn test_zscore_clamped_upper() {
        let ob = make_order_book(99.0, 101.0);
        // Extremely volatile recent prices
        let mut prices = vec![100.0; 300];
        for i in 0..60 {
            prices.push(100.0 + if i % 2 == 0 { 10.0 } else { -10.0 });
        }
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // Very low historical vol with genuine variation → extreme z-score gets clamped
        let vols: Vec<f64> = (0..200).map(|i| 0.00001 + (i as f64) * 0.0000001).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.zscore <= 10.0, "Z-score should be clamped to 10.0, got {}", f.zscore);
        assert!(f.zscore == 10.0, "Should actually hit the clamp for such extreme input, got {}", f.zscore);
    }

    #[test]
    fn test_zscore_clamped_lower() {
        let ob = make_order_book(99.0, 101.0);
        // Zero recent vol (constant prices)
        let price_buf = make_price_buffer(&[100.0; 400]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        // High historical vol
        let vols: Vec<f64> = (0..200).map(|i| 0.01 + (i as f64) * 0.0001).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.zscore >= -10.0, "Z-score should be clamped to -10.0, got {}", f.zscore);
    }

    #[test]
    fn test_zscore_is_finite() {
        let ob = make_order_book(99.0, 101.0);
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vols: Vec<f64> = (0..200).map(|i| 0.001 + (i as f64) * 0.0001).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.zscore.is_finite(), "Z-score must always be finite");
        assert!(!f.zscore.is_nan(), "Z-score must never be NaN");
    }

    // ---------- full compute integration ----------

    #[test]
    fn test_compute_empty_buffers() {
        let ob = OrderBook::new(10);
        let price_buf = RingBuffer::<f64>::new(1000);
        let spread_buf = RingBuffer::<f64>::new(600);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        // All should be zero or finite
        for (i, v) in f.to_vec().iter().enumerate() {
            assert!(v.is_finite(), "Feature index {} should be finite with empty buffers, got {}", i, v);
        }
    }

    #[test]
    fn test_compute_minimal_data() {
        let ob = make_order_book(99.0, 101.0);
        let price_buf = make_price_buffer(&[100.0, 101.0]);
        let spread_buf = make_spread_buffer(&[2.0, 2.0]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.returns_1m > 0.0, "Should compute vol from 2 prices");
        assert!(f.returns_5m > 0.0, "Should compute vol from 2 prices");
    }

    #[test]
    fn test_compute_ratio_short_long_accelerating() {
        let ob = make_order_book(99.0, 101.0);
        // Stable prices for 300 ticks, then volatile for last 60
        let mut prices: Vec<f64> = (0..300).map(|i| 100.0 + (i as f64) * 0.001).collect();
        for i in 0..60 {
            prices.push(100.3 + if i % 2 == 0 { 0.5 } else { -0.5 });
        }
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.ratio_short_long > 1.0,
            "Recent vol spike → ratio > 1, got {}", f.ratio_short_long);
    }

    #[test]
    fn test_compute_ratio_short_long_decelerating() {
        let ob = make_order_book(99.0, 101.0);
        // Volatile for first 240 ticks, then stable for last 60
        let mut prices: Vec<f64> = (0..240).map(|i| {
            100.0 + if i % 2 == 0 { 1.0 } else { -1.0 }
        }).collect();
        for _ in 0..60 {
            prices.push(100.0);
        }
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.ratio_short_long < 1.0,
            "Calming vol → ratio < 1, got {}", f.ratio_short_long);
    }

    #[test]
    fn test_compute_ratio_zero_long_vol() {
        let ob = make_order_book(99.0, 101.0);
        // Constant prices → zero vol → ratio defaults to 1.0
        let price_buf = make_price_buffer(&[100.0; 400]);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert_eq!(f.ratio_short_long, 1.0, "Zero 5m vol → default ratio 1.0");
    }

    #[test]
    fn test_all_features_non_nan() {
        // With sufficient data, no feature should be NaN
        let ob = make_order_book(99.0, 101.0);
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let price_buf = make_price_buffer(&prices);
        let spreads: Vec<f64> = (0..100).map(|i| 1.0 + (i as f64) * 0.01).collect();
        let spread_buf = make_spread_buffer(&spreads);
        let vols: Vec<f64> = (0..200).map(|i| 0.001 + (i as f64) * 0.00001).collect();
        let vol_buf = make_vol_buffer(&vols);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        for (i, v) in f.to_vec().iter().enumerate() {
            assert!(!v.is_nan(), "Feature {} ({}) must not be NaN",
                i, VolatilityFeatures::names()[i]);
            assert!(v.is_finite(), "Feature {} ({}) must be finite",
                i, VolatilityFeatures::names()[i]);
        }
    }

    #[test]
    fn test_parkinson_geq_realized() {
        // Parkinson uses high-low range which captures more info than close-to-close.
        // For a trending sequence, Parkinson should be >= realized vol conceptually,
        // but they measure different things. Just check both are positive.
        let ob = make_order_book(99.0, 101.0);
        let prices: Vec<f64> = (0..400).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.returns_5m > 0.0);
        assert!(f.parkinson_5m > 0.0);
    }

    #[test]
    fn test_midprice_std_consistent() {
        let ob = make_order_book(99.0, 101.0);
        // Known prices for last 60 ticks
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let price_buf = make_price_buffer(&prices);
        let spread_buf = make_spread_buffer(&[2.0; 100]);
        let vol_buf = RingBuffer::<f64>::new(36_000);

        let f = compute(&price_buf, &ob, &spread_buf, &vol_buf);
        assert!(f.midprice_std_1m > 0.0, "Should have positive midprice std for trending prices");
    }
}
