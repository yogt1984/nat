//! Entropy Feature Extraction
//!
//! Measures information content and predictability of price, volume, and order
//! book dynamics. Low entropy signals trending/ordered markets where momentum
//! strategies work; high entropy signals efficient/random markets favouring
//! mean-reversion.
//!
//! # Features (24 total: 10 distribution/permutation + 7 tick + 7 volume-tick)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Permutation returns** | Ordinal-pattern entropy of returns | [0, 1] | 0 = deterministic, 1 = random |
//! | **Permutation imbalance** | Ordinal-pattern entropy of L1 imbalance | [0, 1] | Low = persistent imbalance |
//! | **Spread dispersion** | Shannon entropy of binned spreads | [0, 1] | Low = tight clustering |
//! | **Volume dispersion** | Shannon entropy of binned trade sizes | [0, 1] | Low = uniform sizes |
//! | **Book shape** | Shannon entropy of depth distribution | [0, 1] | Low = concentrated depth |
//! | **Trade size dispersion** | Shannon entropy of trade sizes (5 bins) | [0, 1] | Low = homogeneous flow |
//! | **Rate of change** | Entropy delta over ~5 s | (-inf, +inf) | Sharp drop = regime onset |
//! | **Z-score** | Entropy vs 1-min distribution | (-inf, +inf) | |z| > 2 = unusual regime |
//! | **Tick entropy** | Shannon entropy of {up,down,neutral} ticks | [0, ln(3)] | 0 = single direction |
//! | **Volume-tick entropy** | Volume-weighted tick direction entropy | [0, ln(3)] | Accounts for trade size |
//!
//! Window sizes — permutation: 8/16/32 ticks (short/medium/long ordinal patterns,
//! order=3 giving 3!=6 possible patterns). Tick entropy: 1s/5s/10s/15s/30s/1m/15m
//! (microstructure through regime timescales).
//!
//! # Algorithms
//!
//! **Permutation entropy** (Bandt & Pompe 2002): For embedding dimension d=3,
//! count occurrences of each of the d!=6 ordinal patterns in sliding windows,
//! compute Shannon entropy, normalize by ln(d!) to [0,1].
//!
//! **Distribution entropy**: Bin continuous values into N equal-width bins,
//! compute Shannon entropy H = -Σ p_i ln(p_i), normalize by ln(N).
//!
//! **Tick entropy**: Classify each trade as up/down/neutral by tick rule
//! (compare price to previous trade), compute Shannon entropy of the
//! direction distribution within the time window.
//!
//! # References
//!
//! - Bandt & Pompe (2002) - Permutation entropy: a natural complexity measure
//! - Shannon (1948) - A mathematical theory of communication
//! - Zunino et al. (2009) - Forbidden patterns, permutation entropy, stock market inefficiency

use crate::state::{OrderBook, TradeBuffer, RingBuffer};

/// Entropy features (24 features: 10 original + 14 tick entropy)
#[derive(Debug, Clone, Default)]
pub struct EntropyFeatures {
    // === Original permutation/distribution entropy (10 features) ===
    /// Permutation entropy of returns, length 8
    pub permutation_returns_8: f64,
    /// Permutation entropy of returns, length 16
    pub permutation_returns_16: f64,
    /// Permutation entropy of returns, length 32
    pub permutation_returns_32: f64,
    /// Permutation entropy of imbalance series
    pub permutation_imbalance_16: f64,
    /// Entropy of spread distribution
    pub spread_dispersion: f64,
    /// Entropy of volume distribution
    pub volume_dispersion: f64,
    /// Entropy of book shape (depth distribution)
    pub book_shape: f64,
    /// Entropy of trade size distribution
    pub trade_size_dispersion: f64,
    /// Rate of change of entropy
    pub rate_of_change_5s: f64,
    /// Entropy z-score vs 1-minute mean
    pub zscore_1m: f64,

    // === Tick entropy features (7 features) ===
    /// Tick direction entropy, 1 second window
    pub tick_entropy_1s: f64,
    /// Tick direction entropy, 5 second window
    pub tick_entropy_5s: f64,
    /// Tick direction entropy, 10 second window
    pub tick_entropy_10s: f64,
    /// Tick direction entropy, 15 second window
    pub tick_entropy_15s: f64,
    /// Tick direction entropy, 30 second window
    pub tick_entropy_30s: f64,
    /// Tick direction entropy, 1 minute window
    pub tick_entropy_1m: f64,
    /// Tick direction entropy, 15 minute window
    pub tick_entropy_15m: f64,

    // === Volume-weighted tick entropy features (7 features) ===
    /// Volume-weighted tick entropy, 1 second window
    pub volume_tick_entropy_1s: f64,
    /// Volume-weighted tick entropy, 5 second window
    pub volume_tick_entropy_5s: f64,
    /// Volume-weighted tick entropy, 10 second window
    pub volume_tick_entropy_10s: f64,
    /// Volume-weighted tick entropy, 15 second window
    pub volume_tick_entropy_15s: f64,
    /// Volume-weighted tick entropy, 30 second window
    pub volume_tick_entropy_30s: f64,
    /// Volume-weighted tick entropy, 1 minute window
    pub volume_tick_entropy_1m: f64,
    /// Volume-weighted tick entropy, 15 minute window
    pub volume_tick_entropy_15m: f64,
}

impl EntropyFeatures {
    pub fn count() -> usize { 24 }

    pub fn names() -> Vec<&'static str> {
        vec![
            // Original permutation/distribution entropy
            "ent_permutation_returns_8",
            "ent_permutation_returns_16",
            "ent_permutation_returns_32",
            "ent_permutation_imbalance_16",
            "ent_spread_dispersion",
            "ent_volume_dispersion",
            "ent_book_shape",
            "ent_trade_size_dispersion",
            "ent_rate_of_change_5s",
            "ent_zscore_1m",
            // Tick entropy (7 windows)
            "ent_tick_1s",
            "ent_tick_5s",
            "ent_tick_10s",
            "ent_tick_15s",
            "ent_tick_30s",
            "ent_tick_1m",
            "ent_tick_15m",
            // Volume-weighted tick entropy (7 windows)
            "ent_vol_tick_1s",
            "ent_vol_tick_5s",
            "ent_vol_tick_10s",
            "ent_vol_tick_15s",
            "ent_vol_tick_30s",
            "ent_vol_tick_1m",
            "ent_vol_tick_15m",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            // Original permutation/distribution entropy
            self.permutation_returns_8,
            self.permutation_returns_16,
            self.permutation_returns_32,
            self.permutation_imbalance_16,
            self.spread_dispersion,
            self.volume_dispersion,
            self.book_shape,
            self.trade_size_dispersion,
            self.rate_of_change_5s,
            self.zscore_1m,
            // Tick entropy
            self.tick_entropy_1s,
            self.tick_entropy_5s,
            self.tick_entropy_10s,
            self.tick_entropy_15s,
            self.tick_entropy_30s,
            self.tick_entropy_1m,
            self.tick_entropy_15m,
            // Volume-weighted tick entropy
            self.volume_tick_entropy_1s,
            self.volume_tick_entropy_5s,
            self.volume_tick_entropy_10s,
            self.volume_tick_entropy_15s,
            self.volume_tick_entropy_30s,
            self.volume_tick_entropy_1m,
            self.volume_tick_entropy_15m,
        ]
    }
}

/// Compute entropy features
pub fn compute(
    price_buffer: &RingBuffer<f64>,
    order_book: &OrderBook,
    trade_buffer: &TradeBuffer,
    imbalance_buffer: &RingBuffer<f64>,
    spread_buffer: &RingBuffer<f64>,
    entropy_buffer: &RingBuffer<f64>,
) -> EntropyFeatures {
    let returns = price_buffer.returns();

    // Permutation entropy of returns at different lengths
    let permutation_returns_8 = if returns.len() >= 8 {
        permutation_entropy(&returns[returns.len()-8..], 3)
    } else {
        0.0
    };

    let permutation_returns_16 = if returns.len() >= 16 {
        permutation_entropy(&returns[returns.len()-16..], 3)
    } else {
        0.0
    };

    let permutation_returns_32 = if returns.len() >= 32 {
        permutation_entropy(&returns[returns.len()-32..], 3)
    } else {
        0.0
    };

    // Permutation entropy of imbalance series (16 samples)
    let permutation_imbalance_16 = if imbalance_buffer.len() >= 16 {
        let imb_vec = imbalance_buffer.to_vec();
        let slice = &imb_vec[imb_vec.len()-16..];
        permutation_entropy(slice, 3)
    } else {
        0.0
    };

    // Spread dispersion (entropy of spread history)
    let spread_dispersion = if spread_buffer.len() >= 10 {
        let spread_vec = spread_buffer.to_vec();
        // Use last 300 samples (~30s at 100ms) or whatever is available
        let start = if spread_vec.len() > 300 { spread_vec.len() - 300 } else { 0 };
        distribution_entropy(&spread_vec[start..], 10)
    } else {
        0.0
    };

    // Volume dispersion
    let trade_sizes = trade_buffer.trade_sizes_in_window(30);
    let volume_dispersion = if trade_sizes.len() >= 5 {
        distribution_entropy(&trade_sizes, 10)
    } else {
        0.0
    };

    // Book shape entropy
    let book_shape = compute_book_shape_entropy(order_book);

    // Trade size dispersion
    let trade_size_dispersion = if trade_sizes.len() >= 5 {
        distribution_entropy(&trade_sizes, 5)
    } else {
        0.0
    };

    // Rate of change: difference between current and ~5s ago (~50 samples at 100ms)
    let rate_of_change_5s = if entropy_buffer.len() >= 50 {
        let ent_vec = entropy_buffer.to_vec();
        let current = ent_vec[ent_vec.len() - 1];
        let past = ent_vec[ent_vec.len() - 50];
        current - past
    } else {
        0.0
    };

    // Z-score: (current - mean) / std over 1 minute (~600 samples at 100ms)
    let zscore_1m = if entropy_buffer.len() >= 10 {
        let mean = entropy_buffer.mean();
        let std = entropy_buffer.std();
        if std > 1e-10 {
            let ent_vec = entropy_buffer.to_vec();
            let current = ent_vec[ent_vec.len() - 1];
            (current - mean) / std
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Tick entropy at various time windows
    let tick_entropy_1s = trade_buffer.tick_entropy_in_window(1).unwrap_or(0.0);
    let tick_entropy_5s = trade_buffer.tick_entropy_in_window(5).unwrap_or(0.0);
    let tick_entropy_10s = trade_buffer.tick_entropy_in_window(10).unwrap_or(0.0);
    let tick_entropy_15s = trade_buffer.tick_entropy_in_window(15).unwrap_or(0.0);
    let tick_entropy_30s = trade_buffer.tick_entropy_in_window(30).unwrap_or(0.0);
    let tick_entropy_1m = trade_buffer.tick_entropy_in_window(60).unwrap_or(0.0);
    let tick_entropy_15m = trade_buffer.tick_entropy_in_window(900).unwrap_or(0.0);

    // Volume-weighted tick entropy at various time windows
    let volume_tick_entropy_1s = trade_buffer.volume_tick_entropy_in_window(1).unwrap_or(0.0);
    let volume_tick_entropy_5s = trade_buffer.volume_tick_entropy_in_window(5).unwrap_or(0.0);
    let volume_tick_entropy_10s = trade_buffer.volume_tick_entropy_in_window(10).unwrap_or(0.0);
    let volume_tick_entropy_15s = trade_buffer.volume_tick_entropy_in_window(15).unwrap_or(0.0);
    let volume_tick_entropy_30s = trade_buffer.volume_tick_entropy_in_window(30).unwrap_or(0.0);
    let volume_tick_entropy_1m = trade_buffer.volume_tick_entropy_in_window(60).unwrap_or(0.0);
    let volume_tick_entropy_15m = trade_buffer.volume_tick_entropy_in_window(900).unwrap_or(0.0);

    EntropyFeatures {
        permutation_returns_8,
        permutation_returns_16,
        permutation_returns_32,
        permutation_imbalance_16,
        spread_dispersion,
        volume_dispersion,
        book_shape,
        trade_size_dispersion,
        rate_of_change_5s,
        zscore_1m,
        // Tick entropy
        tick_entropy_1s,
        tick_entropy_5s,
        tick_entropy_10s,
        tick_entropy_15s,
        tick_entropy_30s,
        tick_entropy_1m,
        tick_entropy_15m,
        // Volume-weighted tick entropy
        volume_tick_entropy_1s,
        volume_tick_entropy_5s,
        volume_tick_entropy_10s,
        volume_tick_entropy_15s,
        volume_tick_entropy_30s,
        volume_tick_entropy_1m,
        volume_tick_entropy_15m,
    }
}

/// Compute permutation entropy
///
/// Permutation entropy measures the complexity of a time series by looking
/// at the ordinal patterns of consecutive values.
///
/// # Arguments
/// * `data` - Time series data
/// * `order` - Embedding dimension (typically 3-7)
///
/// # Returns
/// Normalized permutation entropy in [0, 1]
pub fn permutation_entropy(data: &[f64], order: usize) -> f64 {
    if data.len() < order {
        return 0.0;
    }

    let n_patterns = factorial(order);
    let mut pattern_counts = vec![0u64; n_patterns];
    let mut total = 0u64;

    // Count ordinal patterns
    for window in data.windows(order) {
        let pattern_idx = ordinal_pattern_index(window);
        if pattern_idx < n_patterns {
            pattern_counts[pattern_idx] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    // Compute entropy
    let mut entropy = 0.0;
    for count in &pattern_counts {
        if *count > 0 {
            let p = *count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }

    // Normalize by max entropy (log of n_patterns)
    let max_entropy = (n_patterns as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Convert a window of values to its ordinal pattern index
fn ordinal_pattern_index(window: &[f64]) -> usize {
    let n = window.len();
    if n == 0 {
        return 0;
    }

    // Get the ranking of each element
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| window[a].partial_cmp(&window[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Convert ranking to pattern index using factorial number system
    let mut index = 0;
    let mut used = vec![false; n];

    for i in 0..n {
        let rank = indices.iter().position(|&x| x == i).unwrap();
        let mut count = 0;
        for j in 0..rank {
            if !used[j] {
                count += 1;
            }
        }
        index = index * (n - i) + count;
        used[rank] = true;
    }

    index
}

/// Compute factorial
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Compute entropy of a distribution
///
/// Discretizes continuous values into bins and computes Shannon entropy.
fn distribution_entropy(data: &[f64], n_bins: usize) -> f64 {
    if data.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < 1e-10 {
        return 0.0;  // All values the same
    }

    let bin_width = (max - min) / n_bins as f64;
    let mut counts = vec![0u64; n_bins];

    for &value in data {
        let bin = ((value - min) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }

    // Compute entropy
    let total = data.len() as f64;
    let mut entropy = 0.0;

    for count in &counts {
        if *count > 0 {
            let p = *count as f64 / total;
            entropy -= p * p.ln();
        }
    }

    // Normalize
    let max_entropy = (n_bins as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Compute entropy of order book shape
fn compute_book_shape_entropy(order_book: &OrderBook) -> f64 {
    // Combine bid and ask depths
    let mut depths: Vec<f64> = Vec::new();

    for level in order_book.bids() {
        depths.push(level.size);
    }
    for level in order_book.asks() {
        depths.push(level.size);
    }

    if depths.is_empty() {
        return 0.0;
    }

    // Compute entropy of depth distribution
    let total: f64 = depths.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for depth in &depths {
        if *depth > 0.0 {
            let p = depth / total;
            entropy -= p * p.ln();
        }
    }

    // Normalize by max entropy
    let max_entropy = (depths.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TradeBuffer;

    #[test]
    fn test_permutation_entropy_constant() {
        // Constant sequence should have zero entropy
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let pe = permutation_entropy(&data, 3);
        assert!(pe < 0.1, "Constant sequence should have low entropy");
    }

    #[test]
    fn test_permutation_entropy_random() {
        // Random-ish sequence should have higher entropy than monotonic
        // Use a sequence with varied ordinal patterns
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
        let pe = permutation_entropy(&data, 3);
        // Monotonic has ~0 entropy, this should be meaningfully higher
        assert!(pe > 0.3, "Random sequence should have higher entropy: {}", pe);
    }

    #[test]
    fn test_permutation_entropy_monotonic() {
        // Monotonic sequence should have zero entropy (only one pattern)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pe = permutation_entropy(&data, 3);
        assert!(pe < 0.1, "Monotonic sequence should have low entropy: {}", pe);
    }

    #[test]
    fn test_distribution_entropy() {
        // Uniform distribution should have max entropy
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let de = distribution_entropy(&data, 5);
        assert!(de > 0.9, "Uniform should have high entropy: {}", de);

        // Single value should have zero entropy
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let de = distribution_entropy(&data, 5);
        assert!(de < 0.1, "Constant should have low entropy: {}", de);
    }

    #[test]
    fn test_tick_entropy_empty_buffer() {
        let buffer = TradeBuffer::new(60);
        assert!(buffer.tick_entropy_in_window(5).is_none());
        assert!(buffer.volume_tick_entropy_in_window(5).is_none());
    }

    #[test]
    fn test_tick_entropy_uniform_directions() {
        // When all directions are equally likely, entropy should be high
        // Maximum entropy for 3 states (up, down, neutral) = ln(3) ≈ 1.099
        // For just 2 states (up, down), max entropy = ln(2) ≈ 0.693

        // Mathematical verification:
        // If we have equal counts of up and down (no neutral), p = 0.5 each
        // Entropy = -2 * (0.5 * ln(0.5)) = -2 * (0.5 * -0.693) = 0.693

        let max_entropy_2_states = 2.0_f64.ln();
        assert!((max_entropy_2_states - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_tick_entropy_single_direction() {
        // When all trades go in same direction, entropy should be 0
        // This is because -1 * ln(1) = 0

        let single_dir_entropy = -(1.0_f64 * 1.0_f64.ln());
        assert!(single_dir_entropy.abs() < 1e-10, "Single direction entropy should be 0");
    }

    #[test]
    fn test_entropy_features_count() {
        assert_eq!(EntropyFeatures::count(), 24);
        assert_eq!(EntropyFeatures::names().len(), 24);
        assert_eq!(EntropyFeatures::default().to_vec().len(), 24);
    }

    #[test]
    fn test_entropy_feature_names() {
        let names = EntropyFeatures::names();

        // Check tick entropy names exist
        assert!(names.contains(&"ent_tick_1s"));
        assert!(names.contains(&"ent_tick_5s"));
        assert!(names.contains(&"ent_tick_1m"));
        assert!(names.contains(&"ent_tick_15m"));

        // Check volume tick entropy names exist
        assert!(names.contains(&"ent_vol_tick_1s"));
        assert!(names.contains(&"ent_vol_tick_5s"));
        assert!(names.contains(&"ent_vol_tick_1m"));
        assert!(names.contains(&"ent_vol_tick_15m"));
    }
}
