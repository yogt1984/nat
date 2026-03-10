//! H1 Hypothesis Test: Does Whale Flow Predict Returns?
//!
//! This is the CRITICAL hypothesis test for the Hyperliquid analytics layer.
//!
//! # Hypothesis
//!
//! H1: whale_net_flow_t → return_{t+Δ} (positive correlation)
//!
//! If whales are informed traders, their aggregate flow should predict
//! future price movements.
//!
//! # Test Protocol
//!
//! 1. Compute correlations for all window/horizon combinations:
//!    - Windows: 1h, 4h, 24h flow
//!    - Horizons: 1h, 4h, 24h returns
//!    - Total: 9 tests
//!
//! 2. Apply Bonferroni correction (9 tests)
//!
//! 3. Compute Mutual Information for non-linear relationships
//!
//! 4. Walk-forward validation (5 folds, 70/30 split)
//!
//! # Success Criteria
//!
//! - correlation > 0.05 with p < 0.001 (after Bonferroni)
//! - MI > 0.02 bits
//! - Walk-forward OOS correlation > 0.5 * in-sample
//!
//! # Failure Criteria
//!
//! - correlation < 0.02 or p > 0.01
//! - MI < 0.01 bits
//! - Walk-forward degradation > 60%

use crate::hypothesis::stats::{
    pearson_correlation, spearman_correlation, mutual_information_adaptive,
    t_test_correlation, bonferroni_correct, correlation_test,
    walk_forward_correlation, CorrelationResult, WalkForwardResult,
};
use crate::hypothesis::HypothesisDecision;

/// Configuration for H1 test
#[derive(Debug, Clone)]
pub struct H1TestConfig {
    /// Minimum correlation for success
    pub min_correlation: f64,
    /// Maximum p-value for significance (before Bonferroni)
    pub max_p_value: f64,
    /// Minimum mutual information (bits)
    pub min_mi_bits: f64,
    /// Minimum OOS/IS ratio for walk-forward
    pub min_oos_is_ratio: f64,
    /// Number of walk-forward folds
    pub n_folds: usize,
    /// OOS fraction per fold
    pub oos_fraction: f64,
    /// MI failure threshold
    pub mi_failure_threshold: f64,
    /// Correlation failure threshold
    pub corr_failure_threshold: f64,
}

impl Default for H1TestConfig {
    fn default() -> Self {
        Self {
            min_correlation: 0.05,
            max_p_value: 0.001,
            min_mi_bits: 0.02,
            min_oos_is_ratio: 0.5,
            n_folds: 5,
            oos_fraction: 0.3,
            mi_failure_threshold: 0.01,
            corr_failure_threshold: 0.02,
        }
    }
}

/// Result for a single window/horizon combination
#[derive(Debug, Clone)]
pub struct WindowHorizonResult {
    /// Flow window name (e.g., "1h", "4h", "24h")
    pub flow_window: String,
    /// Return horizon name
    pub return_horizon: String,
    /// Correlation result
    pub correlation: CorrelationResult,
    /// Mutual information in bits
    pub mi_bits: f64,
    /// Walk-forward result
    pub walk_forward: WalkForwardResult,
    /// Whether this combination passes
    pub passes: bool,
    /// Reason for pass/fail
    pub reason: String,
}

/// Final decision for H1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H1Decision {
    /// GO: Whale flow has predictive power
    Go,
    /// NO-GO: No evidence of predictive power
    NoGo,
    /// INCONCLUSIVE: Mixed results
    Inconclusive,
}

impl std::fmt::Display for H1Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H1Decision::Go => write!(f, "GO"),
            H1Decision::NoGo => write!(f, "NO-GO"),
            H1Decision::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Complete H1 test result
#[derive(Debug, Clone)]
pub struct H1TestResult {
    /// Config used for the test
    pub config: H1TestConfig,
    /// Results for each window/horizon combination
    pub window_horizon_results: Vec<WindowHorizonResult>,
    /// Best performing combination
    pub best_combination: Option<WindowHorizonResult>,
    /// Number of combinations that pass
    pub n_passing: usize,
    /// Total combinations tested
    pub n_total: usize,
    /// Bonferroni-corrected significance results
    pub bonferroni_significant: Vec<bool>,
    /// Overall decision
    pub decision: H1Decision,
    /// Summary message
    pub summary: String,
    /// Detailed report
    pub report: String,
}

impl H1TestResult {
    /// Convert decision to HypothesisDecision
    pub fn to_hypothesis_decision(&self) -> HypothesisDecision {
        match self.decision {
            H1Decision::Go => HypothesisDecision::Accept,
            H1Decision::NoGo => HypothesisDecision::Reject,
            H1Decision::Inconclusive => HypothesisDecision::Inconclusive,
        }
    }
}

/// Input data for H1 test
#[derive(Debug, Clone)]
pub struct H1TestData {
    /// Whale net flow over 1h window
    pub whale_flow_1h: Vec<f64>,
    /// Whale net flow over 4h window
    pub whale_flow_4h: Vec<f64>,
    /// Whale net flow over 24h window
    pub whale_flow_24h: Vec<f64>,
    /// Forward returns over 1h horizon
    pub returns_1h: Vec<f64>,
    /// Forward returns over 4h horizon
    pub returns_4h: Vec<f64>,
    /// Forward returns over 24h horizon
    pub returns_24h: Vec<f64>,
    /// Timestamps (for alignment verification)
    pub timestamps_ms: Vec<i64>,
}

impl H1TestData {
    /// Validate data integrity
    pub fn validate(&self) -> Result<(), String> {
        let n = self.whale_flow_1h.len();

        if n < 100 {
            return Err(format!("Insufficient data: {} samples (need 100+)", n));
        }

        if self.whale_flow_4h.len() != n
            || self.whale_flow_24h.len() != n
            || self.returns_1h.len() != n
            || self.returns_4h.len() != n
            || self.returns_24h.len() != n
        {
            return Err("All data series must have the same length".to_string());
        }

        // Check for all-zero flow (no whale activity)
        let flow_variance: f64 = {
            let mean = self.whale_flow_1h.iter().sum::<f64>() / n as f64;
            self.whale_flow_1h.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
        };

        if flow_variance < 1e-10 {
            return Err("Whale flow has zero variance (no activity)".to_string());
        }

        Ok(())
    }

    /// Get flow data by window name
    pub fn get_flow(&self, window: &str) -> &[f64] {
        match window {
            "1h" => &self.whale_flow_1h,
            "4h" => &self.whale_flow_4h,
            "24h" => &self.whale_flow_24h,
            _ => &self.whale_flow_1h,
        }
    }

    /// Get return data by horizon name
    pub fn get_returns(&self, horizon: &str) -> &[f64] {
        match horizon {
            "1h" => &self.returns_1h,
            "4h" => &self.returns_4h,
            "24h" => &self.returns_24h,
            _ => &self.returns_1h,
        }
    }
}

/// Run the complete H1 hypothesis test
pub fn run_h1_whale_flow_test(data: &H1TestData, config: &H1TestConfig) -> H1TestResult {
    // Validate input
    if let Err(e) = data.validate() {
        return H1TestResult {
            config: config.clone(),
            window_horizon_results: vec![],
            best_combination: None,
            n_passing: 0,
            n_total: 0,
            bonferroni_significant: vec![],
            decision: H1Decision::NoGo,
            summary: format!("Data validation failed: {}", e),
            report: format!("# H1 Test Report\n\n**Status:** FAILED\n\n**Reason:** {}", e),
        };
    }

    let windows = ["1h", "4h", "24h"];
    let horizons = ["1h", "4h", "24h"];

    let mut results = Vec::with_capacity(9);
    let mut p_values = Vec::with_capacity(9);

    // Test all window/horizon combinations
    for window in &windows {
        for horizon in &horizons {
            let flow = data.get_flow(window);
            let returns = data.get_returns(horizon);

            let result = test_single_combination(flow, returns, window, horizon, config);
            p_values.push(result.correlation.p_value);
            results.push(result);
        }
    }

    // Apply Bonferroni correction
    let bonferroni_significant = bonferroni_correct(&p_values, config.max_p_value);

    // Update pass/fail based on Bonferroni
    for (i, result) in results.iter_mut().enumerate() {
        if !bonferroni_significant[i] && result.passes {
            result.passes = false;
            result.reason = format!(
                "Failed Bonferroni correction (p={:.4} > {:.4})",
                result.correlation.p_value,
                config.max_p_value / 9.0
            );
        }
    }

    // Count passing combinations
    let n_passing = results.iter().filter(|r| r.passes).count();
    let n_total = results.len();

    // Find best combination
    let best_combination = results.iter()
        .filter(|r| r.passes)
        .max_by(|a, b| {
            // Rank by correlation magnitude
            a.correlation.pearson.abs()
                .partial_cmp(&b.correlation.pearson.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    // Determine overall decision
    let decision = determine_decision(&results, config, n_passing);

    // Generate summary and report
    let summary = generate_summary(&results, n_passing, n_total, &decision);
    let report = generate_report(&results, &bonferroni_significant, &best_combination, config, &decision);

    H1TestResult {
        config: config.clone(),
        window_horizon_results: results,
        best_combination,
        n_passing,
        n_total,
        bonferroni_significant,
        decision,
        summary,
        report,
    }
}

/// Test a single window/horizon combination
fn test_single_combination(
    flow: &[f64],
    returns: &[f64],
    window: &str,
    horizon: &str,
    config: &H1TestConfig,
) -> WindowHorizonResult {
    // Compute correlation
    let correlation = correlation_test(flow, returns, config.max_p_value);

    // Compute mutual information
    let mi_bits = mutual_information_adaptive(flow, returns);

    // Walk-forward validation
    let walk_forward = walk_forward_correlation(
        flow,
        returns,
        config.n_folds,
        config.oos_fraction,
        config.min_oos_is_ratio,
    );

    // Determine pass/fail
    let (passes, reason) = evaluate_combination(&correlation, mi_bits, &walk_forward, config);

    WindowHorizonResult {
        flow_window: window.to_string(),
        return_horizon: horizon.to_string(),
        correlation,
        mi_bits,
        walk_forward,
        passes,
        reason,
    }
}

/// Evaluate a single combination against criteria
fn evaluate_combination(
    corr: &CorrelationResult,
    mi_bits: f64,
    wf: &WalkForwardResult,
    config: &H1TestConfig,
) -> (bool, String) {
    // Check correlation magnitude
    if corr.pearson.abs() < config.corr_failure_threshold {
        return (false, format!(
            "Correlation too weak: {:.4} < {:.4}",
            corr.pearson.abs(),
            config.corr_failure_threshold
        ));
    }

    // Check p-value (before Bonferroni - that's applied later)
    if corr.p_value > config.max_p_value * 10.0 {
        return (false, format!(
            "Not significant: p={:.4} > {:.4}",
            corr.p_value,
            config.max_p_value * 10.0
        ));
    }

    // Check MI
    if mi_bits < config.mi_failure_threshold {
        return (false, format!(
            "MI too low: {:.4} bits < {:.4} bits",
            mi_bits,
            config.mi_failure_threshold
        ));
    }

    // Check walk-forward
    if wf.n_folds > 0 && !wf.passes {
        return (false, format!(
            "Walk-forward failed: OOS/IS={:.2} < {:.2}",
            wf.oos_is_ratio,
            config.min_oos_is_ratio
        ));
    }

    // All success criteria
    let meets_corr = corr.pearson.abs() >= config.min_correlation;
    let meets_pval = corr.p_value <= config.max_p_value;
    let meets_mi = mi_bits >= config.min_mi_bits;
    let meets_wf = wf.passes || wf.n_folds == 0;

    if meets_corr && meets_pval && meets_mi && meets_wf {
        (true, format!(
            "PASS: r={:.4}, p={:.2e}, MI={:.4} bits, OOS/IS={:.2}",
            corr.pearson, corr.p_value, mi_bits, wf.oos_is_ratio
        ))
    } else {
        let mut reasons = Vec::new();
        if !meets_corr {
            reasons.push(format!("r={:.4}<{:.4}", corr.pearson.abs(), config.min_correlation));
        }
        if !meets_pval {
            reasons.push(format!("p={:.2e}>{:.2e}", corr.p_value, config.max_p_value));
        }
        if !meets_mi {
            reasons.push(format!("MI={:.4}<{:.4}", mi_bits, config.min_mi_bits));
        }
        (false, format!("Near-miss: {}", reasons.join(", ")))
    }
}

/// Determine overall decision based on results
fn determine_decision(
    results: &[WindowHorizonResult],
    config: &H1TestConfig,
    n_passing: usize,
) -> H1Decision {
    // Strong GO: Multiple combinations pass
    if n_passing >= 3 {
        return H1Decision::Go;
    }

    // GO: At least one combination passes with strong evidence
    if n_passing >= 1 {
        let best = results.iter()
            .filter(|r| r.passes)
            .max_by(|a, b| {
                a.correlation.pearson.abs()
                    .partial_cmp(&b.correlation.pearson.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(best) = best {
            if best.correlation.pearson.abs() >= config.min_correlation * 2.0
                && best.mi_bits >= config.min_mi_bits * 1.5
            {
                return H1Decision::Go;
            }
        }
        return H1Decision::Inconclusive;
    }

    // Check for clear failure
    let all_weak = results.iter().all(|r| {
        r.correlation.pearson.abs() < config.corr_failure_threshold
            || r.mi_bits < config.mi_failure_threshold
    });

    if all_weak {
        H1Decision::NoGo
    } else {
        H1Decision::Inconclusive
    }
}

/// Generate summary message
fn generate_summary(
    results: &[WindowHorizonResult],
    n_passing: usize,
    n_total: usize,
    decision: &H1Decision,
) -> String {
    let best_corr = results.iter()
        .map(|r| r.correlation.pearson.abs())
        .fold(0.0_f64, f64::max);

    let best_mi = results.iter()
        .map(|r| r.mi_bits)
        .fold(0.0_f64, f64::max);

    format!(
        "H1 {} - {}/{} combinations pass. Best: r={:.4}, MI={:.4} bits",
        decision, n_passing, n_total, best_corr, best_mi
    )
}

/// Generate detailed report
fn generate_report(
    results: &[WindowHorizonResult],
    bonferroni_sig: &[bool],
    best: &Option<WindowHorizonResult>,
    config: &H1TestConfig,
    decision: &H1Decision,
) -> String {
    let mut report = String::new();

    report.push_str("# H1 Hypothesis Test Report: Does Whale Flow Predict Returns?\n\n");

    // Decision banner
    report.push_str(&format!("## Decision: **{}**\n\n", decision));

    // Config
    report.push_str("## Test Configuration\n\n");
    report.push_str(&format!("- Minimum correlation: {}\n", config.min_correlation));
    report.push_str(&format!("- Maximum p-value: {} (Bonferroni-corrected)\n", config.max_p_value));
    report.push_str(&format!("- Minimum MI: {} bits\n", config.min_mi_bits));
    report.push_str(&format!("- Walk-forward folds: {}\n", config.n_folds));
    report.push_str(&format!("- Required OOS/IS ratio: {}\n\n", config.min_oos_is_ratio));

    // Results matrix
    report.push_str("## Correlation Matrix (Pearson r)\n\n");
    report.push_str("| Flow \\ Return | 1h | 4h | 24h |\n");
    report.push_str("|---------------|-----|-----|-----|\n");

    for window in ["1h", "4h", "24h"] {
        report.push_str(&format!("| {} |", window));
        for horizon in ["1h", "4h", "24h"] {
            let r = results.iter()
                .find(|r| r.flow_window == window && r.return_horizon == horizon)
                .map(|r| r.correlation.pearson)
                .unwrap_or(0.0);
            report.push_str(&format!(" {:.4} |", r));
        }
        report.push_str("\n");
    }

    // P-value matrix
    report.push_str("\n## P-value Matrix\n\n");
    report.push_str("| Flow \\ Return | 1h | 4h | 24h |\n");
    report.push_str("|---------------|------|------|------|\n");

    for window in ["1h", "4h", "24h"] {
        report.push_str(&format!("| {} |", window));
        for horizon in ["1h", "4h", "24h"] {
            let p = results.iter()
                .find(|r| r.flow_window == window && r.return_horizon == horizon)
                .map(|r| r.correlation.p_value)
                .unwrap_or(1.0);
            report.push_str(&format!(" {:.2e} |", p));
        }
        report.push_str("\n");
    }

    // MI matrix
    report.push_str("\n## Mutual Information Matrix (bits)\n\n");
    report.push_str("| Flow \\ Return | 1h | 4h | 24h |\n");
    report.push_str("|---------------|------|------|------|\n");

    for window in ["1h", "4h", "24h"] {
        report.push_str(&format!("| {} |", window));
        for horizon in ["1h", "4h", "24h"] {
            let mi = results.iter()
                .find(|r| r.flow_window == window && r.return_horizon == horizon)
                .map(|r| r.mi_bits)
                .unwrap_or(0.0);
            report.push_str(&format!(" {:.4} |", mi));
        }
        report.push_str("\n");
    }

    // Bonferroni results
    report.push_str("\n## Bonferroni Correction Results\n\n");
    report.push_str(&format!("Adjusted alpha: {:.6}\n\n", config.max_p_value / 9.0));

    for (i, result) in results.iter().enumerate() {
        let sig = if bonferroni_sig[i] { "✓" } else { "✗" };
        report.push_str(&format!(
            "- Flow {} → Return {}: {} (p={:.2e})\n",
            result.flow_window, result.return_horizon, sig, result.correlation.p_value
        ));
    }

    // Walk-forward results
    report.push_str("\n## Walk-Forward Validation\n\n");
    for result in results {
        let wf = &result.walk_forward;
        let status = if wf.passes { "PASS" } else { "FAIL" };
        report.push_str(&format!(
            "- Flow {} → Return {}: {} (IS={:.4}, OOS={:.4}, ratio={:.2})\n",
            result.flow_window, result.return_horizon, status,
            wf.mean_is_corr, wf.mean_oos_corr, wf.oos_is_ratio
        ));
    }

    // Best combination
    if let Some(best) = best {
        report.push_str("\n## Best Performing Combination\n\n");
        report.push_str(&format!("**Flow Window:** {}\n", best.flow_window));
        report.push_str(&format!("**Return Horizon:** {}\n", best.return_horizon));
        report.push_str(&format!("**Pearson Correlation:** {:.4}\n", best.correlation.pearson));
        report.push_str(&format!("**Spearman Correlation:** {:.4}\n", best.correlation.spearman));
        report.push_str(&format!("**P-value:** {:.2e}\n", best.correlation.p_value));
        report.push_str(&format!("**95% CI:** [{:.4}, {:.4}]\n",
            best.correlation.ci_lower, best.correlation.ci_upper));
        report.push_str(&format!("**Mutual Information:** {:.4} bits\n", best.mi_bits));
        report.push_str(&format!("**Walk-Forward OOS/IS:** {:.2}\n", best.walk_forward.oos_is_ratio));
    }

    // Conclusion
    report.push_str("\n## Conclusion\n\n");
    match decision {
        H1Decision::Go => {
            report.push_str("**ACCEPT H1:** Whale flow demonstrates statistically significant \
                predictive power for future returns. The signal passes multiple testing correction, \
                shows meaningful mutual information, and holds up in walk-forward validation.\n\n");
            report.push_str("**Recommendation:** Proceed with whale flow features in the analytics layer.\n");
        }
        H1Decision::NoGo => {
            report.push_str("**REJECT H1:** No evidence that whale flow predicts returns. \
                Correlations are weak, not significant after correction, or fail walk-forward validation.\n\n");
            report.push_str("**Recommendation:** Do not rely on whale flow for alpha generation.\n");
        }
        H1Decision::Inconclusive => {
            report.push_str("**INCONCLUSIVE:** Mixed results. Some combinations show promise but \
                don't meet all criteria. More data or alternative specifications may be needed.\n\n");
            report.push_str("**Recommendation:** Gather more data before making a final decision.\n");
        }
    }

    report
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_predictive_data(n: usize, correlation: f64, noise: f64) -> H1TestData {
        // Generate flow with trend
        let whale_flow_1h: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.05).sin() * 100.0 + (i as f64 * 0.1).cos() * 50.0)
            .collect();

        // Returns follow flow with specified correlation + noise
        let returns_1h: Vec<f64> = (0..n)
            .map(|i| {
                let signal = whale_flow_1h.get(i.saturating_sub(1)).copied().unwrap_or(0.0);
                signal * correlation * 0.001 + (i as f64 * 0.3 + noise).sin() * (1.0 - correlation) * 0.01
            })
            .collect();

        // 4h and 24h are smoothed versions
        let whale_flow_4h: Vec<f64> = whale_flow_1h.windows(4)
            .map(|w| w.iter().sum::<f64>() / 4.0)
            .chain(std::iter::repeat(0.0))
            .take(n)
            .collect();

        let whale_flow_24h: Vec<f64> = whale_flow_1h.windows(24.min(n))
            .map(|w| w.iter().sum::<f64>() / w.len() as f64)
            .chain(std::iter::repeat(0.0))
            .take(n)
            .collect();

        let returns_4h: Vec<f64> = returns_1h.windows(4)
            .map(|w| w.iter().sum::<f64>())
            .chain(std::iter::repeat(0.0))
            .take(n)
            .collect();

        let returns_24h: Vec<f64> = returns_1h.windows(24.min(n))
            .map(|w| w.iter().sum::<f64>())
            .chain(std::iter::repeat(0.0))
            .take(n)
            .collect();

        let timestamps_ms: Vec<i64> = (0..n as i64).map(|i| i * 3600000).collect();

        H1TestData {
            whale_flow_1h,
            whale_flow_4h,
            whale_flow_24h,
            returns_1h,
            returns_4h,
            returns_24h,
            timestamps_ms,
        }
    }

    #[test]
    fn test_h1_with_strong_signal() {
        // Generate data where flow strongly predicts returns
        let data = generate_predictive_data(500, 0.8, 0.1);
        let config = H1TestConfig::default();

        let result = run_h1_whale_flow_test(&data, &config);

        // With strong correlation, should at least not be NoGo
        assert_ne!(result.decision, H1Decision::NoGo,
            "Strong signal should not result in NoGo. Summary: {}", result.summary);

        // Should have some passing combinations
        assert!(result.n_passing > 0 || result.decision == H1Decision::Inconclusive,
            "Should have passing combinations or be inconclusive");

        // Report should be non-empty
        assert!(!result.report.is_empty());
        assert!(result.report.contains("H1 Hypothesis Test Report"));
    }

    #[test]
    fn test_h1_with_no_signal() {
        // Generate independent data (no relationship)
        let n = 500;
        let whale_flow_1h: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.1).sin() * 100.0)
            .collect();

        // Completely independent returns
        let returns_1h: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37 + 2.5).cos() * 0.01)
            .collect();

        let data = H1TestData {
            whale_flow_1h: whale_flow_1h.clone(),
            whale_flow_4h: whale_flow_1h.clone(),
            whale_flow_24h: whale_flow_1h.clone(),
            returns_1h: returns_1h.clone(),
            returns_4h: returns_1h.clone(),
            returns_24h: returns_1h.clone(),
            timestamps_ms: (0..n as i64).collect(),
        };

        let config = H1TestConfig::default();
        let result = run_h1_whale_flow_test(&data, &config);

        // Independent data should not show strong signal
        assert!(result.n_passing < 3,
            "Independent data should not have many passing combinations, got {}", result.n_passing);
    }

    #[test]
    fn test_h1_insufficient_data() {
        let data = H1TestData {
            whale_flow_1h: vec![1.0; 50],
            whale_flow_4h: vec![1.0; 50],
            whale_flow_24h: vec![1.0; 50],
            returns_1h: vec![0.01; 50],
            returns_4h: vec![0.01; 50],
            returns_24h: vec![0.01; 50],
            timestamps_ms: (0..50).collect(),
        };

        let config = H1TestConfig::default();
        let result = run_h1_whale_flow_test(&data, &config);

        // Should fail due to insufficient data
        assert_eq!(result.decision, H1Decision::NoGo);
        assert!(result.summary.contains("validation failed"));
    }

    #[test]
    fn test_h1_zero_variance_flow() {
        let data = H1TestData {
            whale_flow_1h: vec![100.0; 200],
            whale_flow_4h: vec![100.0; 200],
            whale_flow_24h: vec![100.0; 200],
            returns_1h: (0..200).map(|i| (i as f64 * 0.1).sin() * 0.01).collect(),
            returns_4h: (0..200).map(|i| (i as f64 * 0.1).sin() * 0.01).collect(),
            returns_24h: (0..200).map(|i| (i as f64 * 0.1).sin() * 0.01).collect(),
            timestamps_ms: (0..200).collect(),
        };

        let config = H1TestConfig::default();
        let result = run_h1_whale_flow_test(&data, &config);

        // Should fail due to zero variance
        assert_eq!(result.decision, H1Decision::NoGo);
        assert!(result.summary.contains("zero variance"));
    }

    #[test]
    fn test_window_horizon_result_structure() {
        let data = generate_predictive_data(300, 0.5, 0.2);
        let config = H1TestConfig::default();

        let result = run_h1_whale_flow_test(&data, &config);

        // Should have 9 combinations (3 windows × 3 horizons)
        assert_eq!(result.n_total, 9);
        assert_eq!(result.window_horizon_results.len(), 9);
        assert_eq!(result.bonferroni_significant.len(), 9);

        // Verify all combinations are present
        let combinations: Vec<_> = result.window_horizon_results.iter()
            .map(|r| (r.flow_window.as_str(), r.return_horizon.as_str()))
            .collect();

        for window in ["1h", "4h", "24h"] {
            for horizon in ["1h", "4h", "24h"] {
                assert!(combinations.contains(&(window, horizon)),
                    "Missing combination: {} -> {}", window, horizon);
            }
        }
    }

    #[test]
    fn test_bonferroni_applied() {
        let data = generate_predictive_data(500, 0.3, 0.5);
        let config = H1TestConfig {
            max_p_value: 0.05, // Liberal threshold to see correction effects
            ..Default::default()
        };

        let result = run_h1_whale_flow_test(&data, &config);

        // Count results where raw p < 0.05 but Bonferroni fails
        let raw_significant = result.window_horizon_results.iter()
            .filter(|r| r.correlation.p_value < 0.05)
            .count();

        let bonferroni_significant = result.bonferroni_significant.iter()
            .filter(|&&s| s)
            .count();

        // Bonferroni should be more conservative
        assert!(bonferroni_significant <= raw_significant,
            "Bonferroni should be more conservative");
    }

    #[test]
    fn test_report_contains_all_sections() {
        let data = generate_predictive_data(300, 0.4, 0.3);
        let config = H1TestConfig::default();

        let result = run_h1_whale_flow_test(&data, &config);

        // Check report sections
        assert!(result.report.contains("# H1 Hypothesis Test Report"));
        assert!(result.report.contains("## Decision:"));
        assert!(result.report.contains("## Test Configuration"));
        assert!(result.report.contains("## Correlation Matrix"));
        assert!(result.report.contains("## P-value Matrix"));
        assert!(result.report.contains("## Mutual Information Matrix"));
        assert!(result.report.contains("## Bonferroni Correction"));
        assert!(result.report.contains("## Walk-Forward Validation"));
        assert!(result.report.contains("## Conclusion"));
    }

    #[test]
    fn test_walk_forward_validation_runs() {
        let data = generate_predictive_data(500, 0.5, 0.2);
        let config = H1TestConfig {
            n_folds: 5,
            oos_fraction: 0.3,
            ..Default::default()
        };

        let result = run_h1_whale_flow_test(&data, &config);

        // All results should have walk-forward data
        for whr in &result.window_horizon_results {
            assert!(whr.walk_forward.n_folds > 0,
                "Walk-forward should run for {} -> {}", whr.flow_window, whr.return_horizon);
        }
    }

    #[test]
    fn test_decision_to_hypothesis_decision() {
        let data = generate_predictive_data(300, 0.5, 0.2);
        let config = H1TestConfig::default();
        let result = run_h1_whale_flow_test(&data, &config);

        let hyp_decision = result.to_hypothesis_decision();

        match result.decision {
            H1Decision::Go => assert_eq!(hyp_decision, HypothesisDecision::Accept),
            H1Decision::NoGo => assert_eq!(hyp_decision, HypothesisDecision::Reject),
            H1Decision::Inconclusive => assert_eq!(hyp_decision, HypothesisDecision::Inconclusive),
        }
    }

    #[test]
    fn test_mutual_information_computed() {
        let data = generate_predictive_data(500, 0.6, 0.2);
        let config = H1TestConfig::default();

        let result = run_h1_whale_flow_test(&data, &config);

        // All results should have MI computed
        for whr in &result.window_horizon_results {
            // MI should be non-negative
            assert!(whr.mi_bits >= 0.0,
                "MI should be non-negative, got {} for {} -> {}",
                whr.mi_bits, whr.flow_window, whr.return_horizon);
        }
    }

    #[test]
    fn test_correlation_confidence_intervals() {
        let data = generate_predictive_data(500, 0.5, 0.2);
        let config = H1TestConfig::default();

        let result = run_h1_whale_flow_test(&data, &config);

        for whr in &result.window_horizon_results {
            let corr = &whr.correlation;

            // CI should bracket the point estimate
            assert!(corr.ci_lower <= corr.pearson && corr.pearson <= corr.ci_upper,
                "CI [{}, {}] should bracket r={}", corr.ci_lower, corr.ci_upper, corr.pearson);

            // CI should be within [-1, 1]
            assert!(corr.ci_lower >= -1.0 && corr.ci_upper <= 1.0,
                "CI should be within [-1, 1]");
        }
    }
}
