//! Entropy features
//!
//! Implements permutation entropy and other entropy measures for regime detection.

use crate::state::{OrderBook, TradeBuffer, RingBuffer};

/// Entropy features (10 features)
#[derive(Debug, Clone, Default)]
pub struct EntropyFeatures {
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
}

impl EntropyFeatures {
    pub fn count() -> usize { 10 }

    pub fn names() -> Vec<&'static str> {
        vec![
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
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
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
        ]
    }
}

/// Compute entropy features
pub fn compute(
    price_buffer: &RingBuffer<f64>,
    order_book: &OrderBook,
    trade_buffer: &TradeBuffer,
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

    // Permutation entropy of imbalance (would need imbalance history)
    let permutation_imbalance_16 = 0.0;  // TODO: track imbalance history

    // Spread dispersion (entropy of spread values)
    let spread_dispersion = 0.0;  // TODO: track spread history

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

    // Rate of change (would need entropy history)
    let rate_of_change_5s = 0.0;

    // Z-score (would need entropy history)
    let zscore_1m = 0.0;

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
}
