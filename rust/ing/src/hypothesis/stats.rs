//! Statistical Functions for Hypothesis Testing
//!
//! This module provides rigorous statistical functions for:
//! - Correlation (Pearson and Spearman)
//! - Mutual Information estimation
//! - P-value computation with multiple testing correction
//! - Walk-forward validation

use std::collections::HashMap;

/// Result of a correlation test with statistical significance
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Pearson correlation coefficient
    pub pearson: f64,
    /// Spearman rank correlation
    pub spearman: f64,
    /// T-statistic for pearson
    pub t_statistic: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Sample size
    pub n: usize,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
    /// Whether result is significant at given alpha
    pub significant: bool,
}

/// Result of walk-forward validation
#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    /// In-sample correlations per fold
    pub is_correlations: Vec<f64>,
    /// Out-of-sample correlations per fold
    pub oos_correlations: Vec<f64>,
    /// Mean IS correlation
    pub mean_is_corr: f64,
    /// Mean OOS correlation
    pub mean_oos_corr: f64,
    /// Std of IS correlations
    pub std_is_corr: f64,
    /// Std of OOS correlations
    pub std_oos_corr: f64,
    /// OOS/IS ratio (measures overfitting)
    pub oos_is_ratio: f64,
    /// Whether test passes (OOS > threshold * IS)
    pub passes: bool,
    /// Number of folds
    pub n_folds: usize,
}

/// Compute Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    cov / denom
}

/// Compute Spearman rank correlation
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let rank_x = compute_ranks(&x[..n]);
    let rank_y = compute_ranks(&y[..n]);

    pearson_correlation(&rank_x, &rank_y)
}

/// Compute ranks for a slice (handles ties by averaging)
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find ties
        while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }

    ranks
}

/// Perform t-test for correlation coefficient
/// Returns (t_statistic, p_value)
pub fn t_test_correlation(r: f64, n: usize) -> (f64, f64) {
    if n < 3 {
        return (0.0, 1.0);
    }

    let df = n as f64 - 2.0;

    // Handle r = ±1 case
    if (r.abs() - 1.0).abs() < 1e-15 {
        return (f64::INFINITY.copysign(r), 0.0);
    }

    let t = r * (df.sqrt()) / (1.0 - r * r).sqrt();
    let p = 2.0 * (1.0 - t_distribution_cdf(t.abs(), df));

    (t, p)
}

/// Apply Bonferroni correction to p-values
pub fn bonferroni_correct(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len();
    let adjusted_alpha = alpha / m as f64;
    p_values.iter().map(|&p| p < adjusted_alpha).collect()
}

/// Compute comprehensive correlation result
pub fn correlation_test(x: &[f64], y: &[f64], alpha: f64) -> CorrelationResult {
    let n = x.len().min(y.len());

    if n < 3 {
        return CorrelationResult {
            pearson: 0.0,
            spearman: 0.0,
            t_statistic: 0.0,
            p_value: 1.0,
            n,
            ci_lower: -1.0,
            ci_upper: 1.0,
            significant: false,
        };
    }

    let pearson = pearson_correlation(x, y);
    let spearman = spearman_correlation(x, y);
    let (t_statistic, p_value) = t_test_correlation(pearson, n);

    // Fisher z-transform for confidence interval
    let z = fisher_z_transform(pearson);
    let se = 1.0 / ((n - 3) as f64).sqrt();
    let z_crit = 1.96; // 95% CI
    let z_lower = z - z_crit * se;
    let z_upper = z + z_crit * se;
    let ci_lower = fisher_z_inverse(z_lower);
    let ci_upper = fisher_z_inverse(z_upper);

    CorrelationResult {
        pearson,
        spearman,
        t_statistic,
        p_value,
        n,
        ci_lower,
        ci_upper,
        significant: p_value < alpha,
    }
}

/// Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
fn fisher_z_transform(r: f64) -> f64 {
    let r = r.clamp(-0.9999, 0.9999);
    0.5 * ((1.0 + r) / (1.0 - r)).ln()
}

/// Inverse Fisher z-transform
fn fisher_z_inverse(z: f64) -> f64 {
    let e2z = (2.0 * z).exp();
    (e2z - 1.0) / (e2z + 1.0)
}

/// Estimate Mutual Information using binning
/// Returns MI in bits
pub fn mutual_information(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
    let n = x.len().min(y.len());
    if n < 10 || n_bins < 2 {
        return 0.0;
    }

    let x = &x[..n];
    let y = &y[..n];

    // Find ranges
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    if x_range < 1e-15 || y_range < 1e-15 {
        return 0.0;
    }

    // Count joint and marginal distributions
    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut marginal_x: HashMap<usize, usize> = HashMap::new();
    let mut marginal_y: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let bx = ((x[i] - x_min) / x_range * (n_bins - 1) as f64).floor() as usize;
        let by = ((y[i] - y_min) / y_range * (n_bins - 1) as f64).floor() as usize;
        let bx = bx.min(n_bins - 1);
        let by = by.min(n_bins - 1);

        *joint.entry((bx, by)).or_insert(0) += 1;
        *marginal_x.entry(bx).or_insert(0) += 1;
        *marginal_y.entry(by).or_insert(0) += 1;
    }

    // Compute MI: sum p(x,y) * log2(p(x,y) / (p(x) * p(y)))
    let n_f64 = n as f64;
    let mut mi = 0.0;

    for (&(bx, by), &count) in &joint {
        let p_xy = count as f64 / n_f64;
        let p_x = marginal_x[&bx] as f64 / n_f64;
        let p_y = marginal_y[&by] as f64 / n_f64;

        if p_xy > 1e-15 && p_x > 1e-15 && p_y > 1e-15 {
            mi += p_xy * (p_xy / (p_x * p_y)).log2();
        }
    }

    mi.max(0.0) // MI is always non-negative
}

/// Adaptive MI estimation using multiple bin sizes
pub fn mutual_information_adaptive(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());

    // Use Sturges' rule for bin count
    let n_bins = (1.0 + (n as f64).log2()).ceil() as usize;
    let n_bins = n_bins.max(5).min(50);

    // Try multiple bin sizes and take median (robust)
    let bin_sizes = [
        n_bins / 2,
        n_bins,
        n_bins * 2,
    ];

    let mut mis: Vec<f64> = bin_sizes.iter()
        .filter(|&&b| b >= 3)
        .map(|&b| mutual_information(x, y, b))
        .collect();

    mis.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if mis.is_empty() {
        0.0
    } else {
        mis[mis.len() / 2] // Median
    }
}

/// Walk-forward validation for correlation
/// Uses expanding window: train on [0, train_end], test on [train_end, test_end]
pub fn walk_forward_correlation(
    x: &[f64],
    y: &[f64],
    n_folds: usize,
    oos_ratio: f64, // Fraction of data for OOS in each fold
    required_oos_is_ratio: f64, // Minimum OOS/IS ratio to pass
) -> WalkForwardResult {
    let n = x.len().min(y.len());

    if n < 50 || n_folds < 2 {
        return WalkForwardResult {
            is_correlations: vec![],
            oos_correlations: vec![],
            mean_is_corr: 0.0,
            mean_oos_corr: 0.0,
            std_is_corr: 0.0,
            std_oos_corr: 0.0,
            oos_is_ratio: 0.0,
            passes: false,
            n_folds: 0,
        };
    }

    let x = &x[..n];
    let y = &y[..n];

    let fold_size = n / n_folds;
    let oos_size = (fold_size as f64 * oos_ratio).ceil() as usize;

    let mut is_correlations = Vec::with_capacity(n_folds);
    let mut oos_correlations = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        // Expanding window: train on data up to this fold
        let train_end = (fold + 1) * fold_size - oos_size;
        let test_start = train_end;
        let test_end = ((fold + 1) * fold_size).min(n);

        if train_end < 30 || test_end <= test_start + 10 {
            continue;
        }

        // In-sample correlation
        let is_corr = pearson_correlation(&x[..train_end], &y[..train_end]);

        // Out-of-sample correlation
        let oos_corr = pearson_correlation(&x[test_start..test_end], &y[test_start..test_end]);

        is_correlations.push(is_corr);
        oos_correlations.push(oos_corr);
    }

    let actual_folds = is_correlations.len();
    if actual_folds == 0 {
        return WalkForwardResult {
            is_correlations: vec![],
            oos_correlations: vec![],
            mean_is_corr: 0.0,
            mean_oos_corr: 0.0,
            std_is_corr: 0.0,
            std_oos_corr: 0.0,
            oos_is_ratio: 0.0,
            passes: false,
            n_folds: 0,
        };
    }

    let mean_is = is_correlations.iter().sum::<f64>() / actual_folds as f64;
    let mean_oos = oos_correlations.iter().sum::<f64>() / actual_folds as f64;

    let var_is: f64 = is_correlations.iter()
        .map(|c| (c - mean_is).powi(2))
        .sum::<f64>() / actual_folds as f64;
    let var_oos: f64 = oos_correlations.iter()
        .map(|c| (c - mean_oos).powi(2))
        .sum::<f64>() / actual_folds as f64;

    let ratio = if mean_is.abs() > 1e-10 {
        mean_oos.abs() / mean_is.abs()
    } else {
        0.0
    };

    // Sign consistency: OOS should have same sign as IS
    let sign_consistent = mean_is.signum() == mean_oos.signum() || mean_oos.abs() < 0.01;
    let passes = ratio >= required_oos_is_ratio && sign_consistent;

    WalkForwardResult {
        is_correlations,
        oos_correlations,
        mean_is_corr: mean_is,
        mean_oos_corr: mean_oos,
        std_is_corr: var_is.sqrt(),
        std_oos_corr: var_oos.sqrt(),
        oos_is_ratio: ratio,
        passes,
        n_folds: actual_folds,
    }
}

/// Approximate t-distribution CDF
fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    if df > 30.0 {
        return normal_cdf(t);
    }

    let x = df / (df + t * t);
    let beta_reg = incomplete_beta(df / 2.0, 0.5, x);

    if t >= 0.0 {
        1.0 - beta_reg / 2.0
    } else {
        beta_reg / 2.0
    }
}

/// Standard normal CDF (Abramowitz & Stegun approximation)
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

/// Incomplete beta function approximation (for t-distribution)
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use continued fraction approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for incomplete beta
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(1e-30);
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;

        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        h *= d * c;

        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = (1.0 + num / c).max(1e-30);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log gamma function (Lanczos approximation)
fn ln_gamma(x: f64) -> f64 {
    let coef = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let x = x - 1.0;
    let mut y = x + 5.5;
    y -= (x + 0.5) * y.ln();

    let mut ser = 1.000000000190015;
    for (i, &c) in coef.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }

    -y + (2.5066282746310005 * ser / (x + 1.0)).ln()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Perfect positive correlation, got {}", r);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-10, "Perfect negative correlation, got {}", r);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        // Orthogonal data
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let r = pearson_correlation(&x, &y);
        assert!(r.abs() < 0.01, "Should be uncorrelated, got {}", r);
    }

    #[test]
    fn test_spearman_monotonic() {
        // Monotonic but non-linear
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2
        let r = spearman_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "Monotonic should give r=1, got {}", r);
    }

    #[test]
    fn test_t_test_significant() {
        // Large correlation with many samples should be significant
        let (_, p) = t_test_correlation(0.5, 100);
        assert!(p < 0.001, "r=0.5, n=100 should be significant, p={}", p);
    }

    #[test]
    fn test_t_test_not_significant() {
        // Small correlation with few samples
        let (_, p) = t_test_correlation(0.1, 10);
        assert!(p > 0.05, "r=0.1, n=10 should not be significant, p={}", p);
    }

    #[test]
    fn test_bonferroni_correction() {
        let p_values = vec![0.01, 0.02, 0.04, 0.06];
        let significant = bonferroni_correct(&p_values, 0.05);
        // Adjusted alpha = 0.05 / 4 = 0.0125
        assert_eq!(significant, vec![true, false, false, false]);
    }

    #[test]
    fn test_mutual_information_perfect() {
        // Perfect dependency
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 2.0).collect();
        let mi = mutual_information(&x, &y, 10);
        assert!(mi > 2.0, "Perfect linear should have high MI, got {}", mi);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent (random) - use deterministic pattern that looks uncorrelated
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).cos()).collect();
        let mi = mutual_information(&x, &y, 10);
        // Random data should have low MI (though not exactly 0 due to finite sample)
        assert!(mi < 1.0, "Independent should have low MI, got {}", mi);
    }

    #[test]
    fn test_walk_forward_validation() {
        // Create data where correlation holds across time
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| {
            (i as f64 * 0.1).sin() * 0.8 + (i as f64 * 0.3).cos() * 0.2
        }).collect();

        let result = walk_forward_correlation(&x, &y, 5, 0.3, 0.5);

        assert!(result.n_folds >= 3, "Should have multiple folds");
        assert!(result.mean_is_corr > 0.5, "IS correlation should be high");
        // OOS may be lower but should be consistent
    }

    #[test]
    fn test_correlation_test_comprehensive() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 + (i as f64 * 0.1).sin() * 10.0).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 0.9 + (i as f64 * 0.2).cos() * 5.0).collect();

        let result = correlation_test(&x, &y, 0.05);

        assert!(result.n == 100);
        assert!(result.pearson > 0.9, "Should be highly correlated");
        assert!(result.p_value < 0.001, "Should be significant");
        assert!(result.significant, "Should be marked significant");
        assert!(result.ci_lower > 0.8, "CI lower should be > 0.8");
    }

    #[test]
    fn test_fisher_z_roundtrip() {
        for r in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let z = fisher_z_transform(r);
            let r_back = fisher_z_inverse(z);
            assert!((r - r_back).abs() < 1e-10, "Fisher roundtrip failed for r={}", r);
        }
    }
}
