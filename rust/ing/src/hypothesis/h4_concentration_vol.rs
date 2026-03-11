//! H4 Hypothesis Test: Does Position Concentration Predict Volatility?
//!
//! This tests whether crowded positioning predicts future volatility.
//!
//! # Hypothesis
//!
//! H4: High Gini coefficient → future volatility increase
//!
//! The theory is that:
//! - Concentrated positions create fragility
//! - When few players hold most of the risk, exits are correlated
//! - High concentration → potential for violent unwinds
//!
//! # Test Protocol
//!
//! 1. Compute position concentration metrics daily (Gini, HHI, top-10, top-20)
//! 2. Compute realized volatility over next 24h, 7d
//! 3. Test correlation(concentration_t, volatility_{t+horizon})
//! 4. Control for current volatility (partial correlation)
//! 5. Test multiple concentration measures
//!
//! # Success Criteria
//!
//! - correlation > 0.2 with p < 0.01
//! - Relationship holds after controlling for current vol
//! - Predictive power at meaningful horizon (24h+)
//!
//! # Failure Criteria
//!
//! - correlation < 0.1
//! - Relationship disappears when controlling for current vol
//! - Only predicts very short-term (not actionable)

use crate::hypothesis::stats::{
    pearson_correlation, spearman_correlation, correlation_test,
    mutual_information_adaptive, CorrelationResult,
};
use crate::hypothesis::HypothesisDecision;

/// Configuration for H4 test
#[derive(Debug, Clone)]
pub struct H4TestConfig {
    /// Minimum correlation for success
    pub min_correlation: f64,
    /// Maximum p-value for significance
    pub max_p_value: f64,
    /// Correlation failure threshold
    pub corr_failure_threshold: f64,
    /// Minimum partial correlation (after controlling for current vol)
    pub min_partial_correlation: f64,
    /// Horizons to test (in periods)
    pub horizons: Vec<usize>,
    /// Minimum samples for valid test
    pub min_samples: usize,
}

impl Default for H4TestConfig {
    fn default() -> Self {
        Self {
            min_correlation: 0.2,
            max_p_value: 0.01,
            corr_failure_threshold: 0.1,
            min_partial_correlation: 0.1,
            horizons: vec![24, 168], // 24h and 7d (assuming hourly data)
            min_samples: 100,
        }
    }
}

/// Result for a single concentration measure
#[derive(Debug, Clone)]
pub struct ConcentrationMeasureResult {
    /// Name of the measure (gini, hhi, top10, etc.)
    pub measure_name: String,
    /// Horizon tested
    pub horizon: usize,
    /// Raw correlation with future volatility
    pub correlation: CorrelationResult,
    /// Spearman (rank) correlation
    pub spearman: f64,
    /// Partial correlation controlling for current vol
    pub partial_correlation: f64,
    /// Partial correlation p-value (approximate)
    pub partial_p_value: f64,
    /// Mutual information
    pub mi_bits: f64,
    /// Whether this measure passes criteria
    pub passes: bool,
    /// Reason for pass/fail
    pub reason: String,
}

/// Volatility regime analysis
#[derive(Debug, Clone)]
pub struct RegimeAnalysis {
    /// Correlation in low volatility regime
    pub corr_low_vol: f64,
    /// Correlation in high volatility regime
    pub corr_high_vol: f64,
    /// Sample count in low vol regime
    pub n_low_vol: usize,
    /// Sample count in high vol regime
    pub n_high_vol: usize,
    /// Is relationship stronger in low vol? (more useful for prediction)
    pub stronger_in_low_vol: bool,
}

/// Granger-like causality test result
#[derive(Debug, Clone)]
pub struct CausalityAnalysis {
    /// Correlation: concentration_t → vol_{t+h}
    pub forward_correlation: f64,
    /// Correlation: vol_t → concentration_{t+h}
    pub reverse_correlation: f64,
    /// Does concentration lead volatility?
    pub concentration_leads: bool,
    /// Lead-lag ratio
    pub lead_lag_ratio: f64,
}

/// Decision for H4
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H4Decision {
    /// GO: Concentration predicts volatility
    Go,
    /// NO-GO: No predictive power
    NoGo,
    /// INCONCLUSIVE: Mixed results
    Inconclusive,
}

impl std::fmt::Display for H4Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H4Decision::Go => write!(f, "GO"),
            H4Decision::NoGo => write!(f, "NO-GO"),
            H4Decision::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Complete H4 test result
#[derive(Debug, Clone)]
pub struct H4TestResult {
    pub config: H4TestConfig,
    /// Results for each concentration measure × horizon
    pub measure_results: Vec<ConcentrationMeasureResult>,
    /// Best performing measure
    pub best_measure: Option<ConcentrationMeasureResult>,
    /// Regime analysis (using best measure)
    pub regime_analysis: Option<RegimeAnalysis>,
    /// Causality analysis
    pub causality_analysis: Option<CausalityAnalysis>,
    /// Number of measures that pass
    pub n_passing: usize,
    /// Total measures tested
    pub n_total: usize,
    /// Overall decision
    pub decision: H4Decision,
    /// Summary message
    pub summary: String,
    /// Detailed report
    pub report: String,
    /// Sample size
    pub n_samples: usize,
}

impl H4TestResult {
    pub fn to_hypothesis_decision(&self) -> HypothesisDecision {
        match self.decision {
            H4Decision::Go => HypothesisDecision::Accept,
            H4Decision::NoGo => HypothesisDecision::Reject,
            H4Decision::Inconclusive => HypothesisDecision::Inconclusive,
        }
    }
}

/// Input data for H4 test
#[derive(Debug, Clone)]
pub struct H4TestData {
    /// Gini coefficient time series
    pub gini: Vec<f64>,
    /// Herfindahl-Hirschman Index time series
    pub hhi: Vec<f64>,
    /// Top 10 concentration (share of OI)
    pub top10: Vec<f64>,
    /// Top 20 concentration
    pub top20: Vec<f64>,
    /// Theil index (entropy-based concentration)
    pub theil: Vec<f64>,
    /// Current realized volatility (for control)
    pub current_volatility: Vec<f64>,
    /// Price series (for computing forward volatility)
    pub prices: Vec<f64>,
    /// Timestamps
    pub timestamps_ms: Vec<i64>,
}

impl H4TestData {
    pub fn validate(&self) -> Result<(), String> {
        let n = self.gini.len();

        if n < 100 {
            return Err(format!("Insufficient data: {} samples (need 100+)", n));
        }

        // Check all series have same length
        if self.hhi.len() != n || self.top10.len() != n || self.top20.len() != n
            || self.theil.len() != n || self.current_volatility.len() != n
            || self.prices.len() != n
        {
            return Err("All data series must have same length".to_string());
        }

        // Check for variance in concentration measures
        let gini_var = variance(&self.gini);
        if gini_var < 1e-10 {
            return Err("Gini has zero variance".to_string());
        }

        // Check for variance in prices
        let price_var = variance(&self.prices);
        if price_var < 1e-10 {
            return Err("Prices have zero variance".to_string());
        }

        Ok(())
    }

    /// Compute realized volatility over a horizon
    pub fn compute_forward_volatility(&self, horizon: usize) -> Vec<f64> {
        let n = self.prices.len();

        (0..n)
            .map(|i| {
                if i + horizon >= n {
                    return f64::NAN;
                }

                // Compute returns in the forward window
                let returns: Vec<f64> = (i..i + horizon)
                    .filter_map(|j| {
                        if j + 1 < n && self.prices[j] > 0.0 {
                            Some((self.prices[j + 1] / self.prices[j]).ln())
                        } else {
                            None
                        }
                    })
                    .collect();

                if returns.len() < 2 {
                    return f64::NAN;
                }

                // Standard deviation of returns (annualized would multiply by sqrt(periods_per_year))
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let var = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;

                var.sqrt()
            })
            .collect()
    }

    /// Get concentration measure by name
    pub fn get_measure(&self, name: &str) -> &[f64] {
        match name {
            "gini" => &self.gini,
            "hhi" => &self.hhi,
            "top10" => &self.top10,
            "top20" => &self.top20,
            "theil" => &self.theil,
            _ => &self.gini,
        }
    }

    /// Get all measure names
    pub fn measure_names() -> Vec<&'static str> {
        vec!["gini", "hhi", "top10", "top20", "theil"]
    }
}

/// Run the H4 hypothesis test
pub fn run_h4_concentration_vol_test(data: &H4TestData, config: &H4TestConfig) -> H4TestResult {
    // Validate
    if let Err(e) = data.validate() {
        return make_error_result(config, &e);
    }

    let n = data.gini.len();

    // Test all measure × horizon combinations
    let mut results = Vec::new();

    for &horizon in &config.horizons {
        let forward_vol = data.compute_forward_volatility(horizon);

        // Filter out NaN values
        let valid_indices: Vec<usize> = (0..n)
            .filter(|&i| !forward_vol[i].is_nan() && !data.current_volatility[i].is_nan())
            .collect();

        if valid_indices.len() < config.min_samples {
            continue;
        }

        for measure_name in H4TestData::measure_names() {
            let measure = data.get_measure(measure_name);

            let result = test_single_measure(
                measure,
                &forward_vol,
                &data.current_volatility,
                &valid_indices,
                measure_name,
                horizon,
                config,
            );

            results.push(result);
        }
    }

    if results.is_empty() {
        return make_error_result(config, "No valid measure/horizon combinations");
    }

    // Count passing measures
    let n_passing = results.iter().filter(|r| r.passes).count();
    let n_total = results.len();

    // Find best measure
    let best_measure = results.iter()
        .filter(|r| r.passes)
        .max_by(|a, b| {
            a.correlation.pearson.abs()
                .partial_cmp(&b.correlation.pearson.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    // Regime analysis using best measure or gini
    let regime_analysis = if let Some(ref best) = best_measure {
        let measure = data.get_measure(&best.measure_name);
        let forward_vol = data.compute_forward_volatility(best.horizon);
        Some(analyze_regimes(measure, &forward_vol, &data.current_volatility))
    } else {
        let forward_vol = data.compute_forward_volatility(config.horizons[0]);
        Some(analyze_regimes(&data.gini, &forward_vol, &data.current_volatility))
    };

    // Causality analysis
    let causality_analysis = if let Some(ref best) = best_measure {
        let measure = data.get_measure(&best.measure_name);
        Some(analyze_causality(measure, &data.current_volatility, best.horizon))
    } else {
        Some(analyze_causality(&data.gini, &data.current_volatility, config.horizons[0]))
    };

    // Determine decision
    let decision = determine_h4_decision(&results, &best_measure, &causality_analysis, config, n_passing);

    // Generate outputs
    let summary = generate_h4_summary(&decision, &best_measure, n_passing, n_total);
    let report = generate_h4_report(
        &results, &best_measure, &regime_analysis, &causality_analysis,
        &decision, config, n
    );

    H4TestResult {
        config: config.clone(),
        measure_results: results,
        best_measure,
        regime_analysis,
        causality_analysis,
        n_passing,
        n_total,
        decision,
        summary,
        report,
        n_samples: n,
    }
}

fn make_error_result(config: &H4TestConfig, error: &str) -> H4TestResult {
    H4TestResult {
        config: config.clone(),
        measure_results: vec![],
        best_measure: None,
        regime_analysis: None,
        causality_analysis: None,
        n_passing: 0,
        n_total: 0,
        decision: H4Decision::NoGo,
        summary: format!("Data validation failed: {}", error),
        report: format!("# H4 Test Report\n\n**Status:** FAILED\n\n**Reason:** {}", error),
        n_samples: 0,
    }
}

fn test_single_measure(
    measure: &[f64],
    forward_vol: &[f64],
    current_vol: &[f64],
    valid_indices: &[usize],
    measure_name: &str,
    horizon: usize,
    config: &H4TestConfig,
) -> ConcentrationMeasureResult {
    // Extract valid data points
    let x: Vec<f64> = valid_indices.iter().map(|&i| measure[i]).collect();
    let y: Vec<f64> = valid_indices.iter().map(|&i| forward_vol[i]).collect();
    let z: Vec<f64> = valid_indices.iter().map(|&i| current_vol[i]).collect();

    // Raw correlation
    let correlation = correlation_test(&x, &y, config.max_p_value);
    let spearman = spearman_correlation(&x, &y);

    // Partial correlation (controlling for current vol)
    let (partial_corr, partial_p) = partial_correlation(&x, &y, &z);

    // Mutual information
    let mi_bits = mutual_information_adaptive(&x, &y);

    // Evaluate pass/fail
    let (passes, reason) = evaluate_measure(
        &correlation, partial_corr, mi_bits, config
    );

    ConcentrationMeasureResult {
        measure_name: measure_name.to_string(),
        horizon,
        correlation,
        spearman,
        partial_correlation: partial_corr,
        partial_p_value: partial_p,
        mi_bits,
        passes,
        reason,
    }
}

/// Compute partial correlation: corr(x, y | z)
/// Uses residualization method
fn partial_correlation(x: &[f64], y: &[f64], z: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len()).min(z.len());
    if n < 10 {
        return (0.0, 1.0);
    }

    // Residualize x on z
    let x_resid = residualize(&x[..n], &z[..n]);
    // Residualize y on z
    let y_resid = residualize(&y[..n], &z[..n]);

    // Correlation of residuals
    let partial_r = pearson_correlation(&x_resid, &y_resid);

    // Approximate p-value using t-test with df = n - 3
    let df = (n - 3) as f64;
    let t = if (1.0 - partial_r * partial_r).abs() < 1e-10 {
        f64::INFINITY.copysign(partial_r)
    } else {
        partial_r * df.sqrt() / (1.0 - partial_r * partial_r).sqrt()
    };

    let p = 2.0 * (1.0 - t_cdf(t.abs(), df));

    (partial_r, p)
}

/// Residualize y on x (simple linear regression residuals)
fn residualize(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n < 2 {
        return y.to_vec();
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
    }

    let beta = if var_x > 1e-15 { cov / var_x } else { 0.0 };
    let alpha = mean_y - beta * mean_x;

    // Residuals = y - (alpha + beta * x)
    (0..n).map(|i| y[i] - alpha - beta * x[i]).collect()
}

fn evaluate_measure(
    corr: &CorrelationResult,
    partial_corr: f64,
    mi_bits: f64,
    config: &H4TestConfig,
) -> (bool, String) {
    // Check raw correlation
    if corr.pearson.abs() < config.corr_failure_threshold {
        return (false, format!(
            "Correlation too weak: {:.3} < {:.3}",
            corr.pearson.abs(), config.corr_failure_threshold
        ));
    }

    // Check significance
    if corr.p_value > config.max_p_value * 10.0 {
        return (false, format!(
            "Not significant: p={:.3} > {:.3}",
            corr.p_value, config.max_p_value * 10.0
        ));
    }

    // Check partial correlation (must survive controlling for current vol)
    if partial_corr.abs() < config.min_partial_correlation {
        return (false, format!(
            "Partial corr too weak: {:.3} < {:.3} (disappears when controlling for current vol)",
            partial_corr.abs(), config.min_partial_correlation
        ));
    }

    // Success criteria
    let meets_corr = corr.pearson.abs() >= config.min_correlation;
    let meets_pval = corr.p_value <= config.max_p_value;
    let meets_partial = partial_corr.abs() >= config.min_partial_correlation;

    if meets_corr && meets_pval && meets_partial {
        (true, format!(
            "PASS: r={:.3}, p={:.2e}, partial_r={:.3}, MI={:.4} bits",
            corr.pearson, corr.p_value, partial_corr, mi_bits
        ))
    } else {
        let mut reasons = Vec::new();
        if !meets_corr {
            reasons.push(format!("r={:.3}<{:.3}", corr.pearson.abs(), config.min_correlation));
        }
        if !meets_pval {
            reasons.push(format!("p={:.2e}>{:.2e}", corr.p_value, config.max_p_value));
        }
        (false, format!("Near-miss: {}", reasons.join(", ")))
    }
}

fn analyze_regimes(
    concentration: &[f64],
    forward_vol: &[f64],
    current_vol: &[f64],
) -> RegimeAnalysis {
    let n = concentration.len().min(forward_vol.len()).min(current_vol.len());

    // Split by current volatility median
    let vol_median = median(&current_vol[..n]);

    let mut low_vol_conc = Vec::new();
    let mut low_vol_fwd = Vec::new();
    let mut high_vol_conc = Vec::new();
    let mut high_vol_fwd = Vec::new();

    for i in 0..n {
        if forward_vol[i].is_nan() {
            continue;
        }

        if current_vol[i] <= vol_median {
            low_vol_conc.push(concentration[i]);
            low_vol_fwd.push(forward_vol[i]);
        } else {
            high_vol_conc.push(concentration[i]);
            high_vol_fwd.push(forward_vol[i]);
        }
    }

    let corr_low = if low_vol_conc.len() >= 20 {
        pearson_correlation(&low_vol_conc, &low_vol_fwd)
    } else {
        0.0
    };

    let corr_high = if high_vol_conc.len() >= 20 {
        pearson_correlation(&high_vol_conc, &high_vol_fwd)
    } else {
        0.0
    };

    RegimeAnalysis {
        corr_low_vol: corr_low,
        corr_high_vol: corr_high,
        n_low_vol: low_vol_conc.len(),
        n_high_vol: high_vol_conc.len(),
        stronger_in_low_vol: corr_low.abs() > corr_high.abs(),
    }
}

fn analyze_causality(
    concentration: &[f64],
    volatility: &[f64],
    horizon: usize,
) -> CausalityAnalysis {
    let n = concentration.len().min(volatility.len());

    if n < horizon + 50 {
        return CausalityAnalysis {
            forward_correlation: 0.0,
            reverse_correlation: 0.0,
            concentration_leads: false,
            lead_lag_ratio: 1.0,
        };
    }

    // Forward: concentration_t → volatility_{t+h}
    let forward_corr = pearson_correlation(
        &concentration[..n - horizon],
        &volatility[horizon..n]
    );

    // Reverse: volatility_t → concentration_{t+h}
    let reverse_corr = pearson_correlation(
        &volatility[..n - horizon],
        &concentration[horizon..n]
    );

    let concentration_leads = forward_corr.abs() > reverse_corr.abs() * 1.2;
    let lead_lag_ratio = if reverse_corr.abs() > 1e-10 {
        forward_corr.abs() / reverse_corr.abs()
    } else {
        10.0
    };

    CausalityAnalysis {
        forward_correlation: forward_corr,
        reverse_correlation: reverse_corr,
        concentration_leads,
        lead_lag_ratio,
    }
}

fn determine_h4_decision(
    results: &[ConcentrationMeasureResult],
    best: &Option<ConcentrationMeasureResult>,
    causality: &Option<CausalityAnalysis>,
    config: &H4TestConfig,
    n_passing: usize,
) -> H4Decision {
    // Strong GO: Multiple measures pass
    if n_passing >= 3 {
        return H4Decision::Go;
    }

    // GO: At least one measure passes with strong evidence
    if let Some(best) = best {
        if best.correlation.pearson.abs() >= config.min_correlation * 1.2
            && best.partial_correlation.abs() >= config.min_partial_correlation * 1.2
        {
            // Check causality direction
            if let Some(c) = causality {
                if c.concentration_leads {
                    return H4Decision::Go;
                }
            } else {
                return H4Decision::Go;
            }
        }

        if n_passing >= 1 {
            return H4Decision::Inconclusive;
        }
    }

    // Check for clear failure
    let all_weak = results.iter().all(|r| {
        r.correlation.pearson.abs() < config.corr_failure_threshold
    });

    if all_weak {
        return H4Decision::NoGo;
    }

    // Check if partial correlations all collapse
    let all_partial_weak = results.iter().all(|r| {
        r.partial_correlation.abs() < config.min_partial_correlation * 0.5
    });

    if all_partial_weak {
        return H4Decision::NoGo;
    }

    H4Decision::Inconclusive
}

fn generate_h4_summary(
    decision: &H4Decision,
    best: &Option<ConcentrationMeasureResult>,
    n_passing: usize,
    n_total: usize,
) -> String {
    match best {
        Some(b) => format!(
            "H4 {} - {}/{} measures pass. Best: {} ({}h), r={:.3}, partial_r={:.3}",
            decision, n_passing, n_total,
            b.measure_name, b.horizon,
            b.correlation.pearson, b.partial_correlation
        ),
        None => format!(
            "H4 {} - {}/{} measures pass. No measure met criteria.",
            decision, n_passing, n_total
        ),
    }
}

fn generate_h4_report(
    results: &[ConcentrationMeasureResult],
    best: &Option<ConcentrationMeasureResult>,
    regime: &Option<RegimeAnalysis>,
    causality: &Option<CausalityAnalysis>,
    decision: &H4Decision,
    config: &H4TestConfig,
    n: usize,
) -> String {
    let mut report = String::new();

    report.push_str("# H4 Hypothesis Test Report: Does Concentration Predict Volatility?\n\n");

    // Decision banner
    report.push_str(&format!("## Decision: **{}**\n\n", decision));

    // Data summary
    report.push_str("## Data Summary\n\n");
    report.push_str(&format!("- Total samples: {}\n", n));
    report.push_str(&format!("- Horizons tested: {:?}\n\n", config.horizons));

    // Configuration
    report.push_str("## Test Configuration\n\n");
    report.push_str(&format!("- Minimum correlation: {}\n", config.min_correlation));
    report.push_str(&format!("- Maximum p-value: {}\n", config.max_p_value));
    report.push_str(&format!("- Minimum partial correlation: {}\n", config.min_partial_correlation));
    report.push_str(&format!("- Correlation failure threshold: {}\n\n", config.corr_failure_threshold));

    // Results by measure
    report.push_str("## Results by Concentration Measure\n\n");

    for horizon in &config.horizons {
        report.push_str(&format!("### Horizon: {} periods\n\n", horizon));
        report.push_str("| Measure | Pearson | Spearman | Partial r | p-value | MI (bits) | Pass |\n");
        report.push_str("|---------|---------|----------|-----------|---------|-----------|------|\n");

        for result in results.iter().filter(|r| r.horizon == *horizon) {
            let pass_mark = if result.passes { "✓" } else { "✗" };
            report.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.3} | {:.2e} | {:.4} | {} |\n",
                result.measure_name,
                result.correlation.pearson,
                result.spearman,
                result.partial_correlation,
                result.correlation.p_value,
                result.mi_bits,
                pass_mark
            ));
        }
        report.push_str("\n");
    }

    // Best measure details
    if let Some(best) = best {
        report.push_str("## Best Performing Measure\n\n");
        report.push_str(&format!("**Measure:** {}\n", best.measure_name));
        report.push_str(&format!("**Horizon:** {} periods\n", best.horizon));
        report.push_str(&format!("**Pearson Correlation:** {:.4}\n", best.correlation.pearson));
        report.push_str(&format!("**Spearman Correlation:** {:.4}\n", best.spearman));
        report.push_str(&format!("**95% CI:** [{:.4}, {:.4}]\n",
            best.correlation.ci_lower, best.correlation.ci_upper));
        report.push_str(&format!("**P-value:** {:.2e}\n", best.correlation.p_value));
        report.push_str(&format!("**Partial Correlation (controlling for current vol):** {:.4}\n",
            best.partial_correlation));
        report.push_str(&format!("**Mutual Information:** {:.4} bits\n\n", best.mi_bits));
    }

    // Regime analysis
    if let Some(regime) = regime {
        report.push_str("## Regime Analysis\n\n");
        report.push_str("Does the relationship hold across volatility regimes?\n\n");
        report.push_str(&format!("- **Low volatility regime:**\n"));
        report.push_str(&format!("  - Samples: {}\n", regime.n_low_vol));
        report.push_str(&format!("  - Correlation: {:.4}\n", regime.corr_low_vol));
        report.push_str(&format!("- **High volatility regime:**\n"));
        report.push_str(&format!("  - Samples: {}\n", regime.n_high_vol));
        report.push_str(&format!("  - Correlation: {:.4}\n", regime.corr_high_vol));
        report.push_str(&format!("- **Stronger in low vol:** {}\n\n",
            if regime.stronger_in_low_vol { "Yes (more useful for prediction)" } else { "No" }));
    }

    // Causality analysis
    if let Some(causality) = causality {
        report.push_str("## Causality Analysis\n\n");
        report.push_str("Does concentration LEAD volatility, or just correlate?\n\n");
        report.push_str(&format!("- **Forward (concentration → future vol):** {:.4}\n",
            causality.forward_correlation));
        report.push_str(&format!("- **Reverse (volatility → future concentration):** {:.4}\n",
            causality.reverse_correlation));
        report.push_str(&format!("- **Lead-lag ratio:** {:.2}x\n", causality.lead_lag_ratio));
        report.push_str(&format!("- **Concentration leads:** {}\n\n",
            if causality.concentration_leads { "Yes ✓" } else { "No ✗" }));
    }

    // Success criteria evaluation
    report.push_str("## Success Criteria Evaluation\n\n");

    let any_passes = results.iter().any(|r| r.passes);
    let any_strong_corr = results.iter().any(|r| r.correlation.pearson.abs() >= config.min_correlation);
    let any_survives_control = results.iter().any(|r| r.partial_correlation.abs() >= config.min_partial_correlation);

    let corr_check = if any_strong_corr { "✓" } else { "✗" };
    let partial_check = if any_survives_control { "✓" } else { "✗" };
    let pass_check = if any_passes { "✓" } else { "✗" };

    report.push_str(&format!("- {} Correlation ≥ {} for at least one measure\n",
        corr_check, config.min_correlation));
    report.push_str(&format!("- {} Partial correlation ≥ {} (survives controlling for current vol)\n",
        partial_check, config.min_partial_correlation));
    report.push_str(&format!("- {} At least one measure passes all criteria: {}\n\n",
        pass_check, results.iter().filter(|r| r.passes).count()));

    // Conclusion
    report.push_str("## Conclusion\n\n");
    match decision {
        H4Decision::Go => {
            report.push_str("**ACCEPT H4:** Position concentration demonstrates predictive power for \
                future volatility. The relationship survives controlling for current volatility, \
                indicating genuine predictive value rather than just persistence.\n\n");
            report.push_str("**Recommendation:** Use concentration metrics for volatility forecasting \
                and risk management.\n");
        }
        H4Decision::NoGo => {
            report.push_str("**REJECT H4:** No evidence that concentration predicts volatility. \
                Either correlations are too weak, or the relationship disappears when controlling \
                for current volatility (indicating it's just volatility persistence, not prediction).\n\n");
            report.push_str("**Recommendation:** Do not use concentration for volatility prediction.\n");
        }
        H4Decision::Inconclusive => {
            report.push_str("**INCONCLUSIVE:** Mixed results. Some measures show promise but don't \
                meet all criteria. The relationship may exist but is weak or inconsistent.\n\n");
            report.push_str("**Recommendation:** Gather more data or test alternative specifications.\n");
        }
    }

    report
}

// Helper functions

fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
}

fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

fn t_cdf(t: f64, df: f64) -> f64 {
    if df > 30.0 {
        return normal_cdf(t);
    }
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - x.powf(df / 2.0)).copysign(t)
}

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_predictive_data(n: usize, true_correlation: f64) -> H4TestData {
        // Generate concentration that predicts volatility
        let gini: Vec<f64> = (0..n)
            .map(|i| 0.5 + 0.3 * (i as f64 * 0.05).sin())
            .collect();

        // Future volatility correlates with concentration
        let mut prices = vec![100.0];
        for i in 1..n {
            let vol_factor = 1.0 + gini.get(i.saturating_sub(24)).copied().unwrap_or(0.5) * true_correlation;
            let return_val = (i as f64 * 0.1).sin() * 0.02 * vol_factor;
            prices.push(prices[i - 1] * (1.0 + return_val));
        }

        // Current volatility (lagged)
        let current_volatility: Vec<f64> = (0..n)
            .map(|i| {
                if i < 24 {
                    0.02
                } else {
                    let window: Vec<f64> = (i - 24..i)
                        .map(|j| ((prices[j + 1] / prices[j]).ln()).abs())
                        .collect();
                    window.iter().sum::<f64>() / window.len() as f64
                }
            })
            .collect();

        // Other concentration measures (correlated with gini)
        let hhi: Vec<f64> = gini.iter().map(|g| g * g * 1.5).collect();
        let top10: Vec<f64> = gini.iter().map(|g| 0.3 + g * 0.4).collect();
        let top20: Vec<f64> = gini.iter().map(|g| 0.5 + g * 0.3).collect();
        let theil: Vec<f64> = gini.iter().map(|g| g * 0.8).collect();

        let timestamps_ms: Vec<i64> = (0..n as i64).map(|i| i * 3600000).collect();

        H4TestData {
            gini,
            hhi,
            top10,
            top20,
            theil,
            current_volatility,
            prices,
            timestamps_ms,
        }
    }

    #[test]
    fn test_h4_with_predictive_signal() {
        let data = generate_predictive_data(500, 0.5);
        let config = H4TestConfig::default();

        let result = run_h4_concentration_vol_test(&data, &config);

        assert!(result.n_samples == 500);
        assert!(!result.report.is_empty());
        assert!(result.report.contains("H4 Hypothesis Test Report"));
    }

    #[test]
    fn test_h4_insufficient_data() {
        let data = H4TestData {
            gini: vec![0.5; 50],
            hhi: vec![0.25; 50],
            top10: vec![0.4; 50],
            top20: vec![0.6; 50],
            theil: vec![0.3; 50],
            current_volatility: vec![0.02; 50],
            prices: (0..50).map(|i| 100.0 + i as f64).collect(),
            timestamps_ms: (0..50).collect(),
        };

        let config = H4TestConfig::default();
        let result = run_h4_concentration_vol_test(&data, &config);

        assert_eq!(result.decision, H4Decision::NoGo);
        assert!(result.summary.contains("validation failed"));
    }

    #[test]
    fn test_h4_zero_variance() {
        let data = H4TestData {
            gini: vec![0.5; 200],
            hhi: vec![0.25; 200],
            top10: vec![0.4; 200],
            top20: vec![0.6; 200],
            theil: vec![0.3; 200],
            current_volatility: vec![0.02; 200],
            prices: (0..200).map(|i| 100.0 + i as f64 * 0.1).collect(),
            timestamps_ms: (0..200).collect(),
        };

        let config = H4TestConfig::default();
        let result = run_h4_concentration_vol_test(&data, &config);

        assert_eq!(result.decision, H4Decision::NoGo);
    }

    #[test]
    fn test_partial_correlation() {
        // x and y both correlate with z, but x has additional info about y
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x: Vec<f64> = (0..n).map(|i| z[i] * 0.7 + (i as f64 * 0.2).cos() * 0.3).collect();
        let y: Vec<f64> = (0..n).map(|i| z[i] * 0.5 + x[i] * 0.3 + (i as f64 * 0.15).sin() * 0.2).collect();

        let (partial_r, _p) = partial_correlation(&x, &y, &z);

        // Partial correlation should be non-zero (x has info about y beyond z)
        assert!(partial_r.abs() > 0.1,
            "Partial correlation should be non-zero, got {}", partial_r);
    }

    #[test]
    fn test_partial_correlation_collapses() {
        // x and y both caused by z, no direct relationship
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x: Vec<f64> = z.iter().map(|&zi| zi + 0.1).collect();
        let y: Vec<f64> = z.iter().map(|&zi| zi * 0.9).collect();

        let raw_r = pearson_correlation(&x, &y);
        let (partial_r, _p) = partial_correlation(&x, &y, &z);

        // Raw correlation should be high
        assert!(raw_r.abs() > 0.8, "Raw correlation should be high");
        // Partial correlation should be much smaller
        assert!(partial_r.abs() < raw_r.abs() * 0.3,
            "Partial correlation should collapse, raw={}, partial={}", raw_r, partial_r);
    }

    #[test]
    fn test_forward_volatility_computation() {
        let data = H4TestData {
            gini: vec![0.5; 100],
            hhi: vec![0.25; 100],
            top10: vec![0.4; 100],
            top20: vec![0.6; 100],
            theil: vec![0.3; 100],
            current_volatility: vec![0.02; 100],
            prices: (0..100).map(|i| 100.0 * (1.0 + 0.01 * (i as f64 * 0.1).sin())).collect(),
            timestamps_ms: (0..100).collect(),
        };

        let fwd_vol = data.compute_forward_volatility(24);

        // Should have n - horizon + some valid values
        assert_eq!(fwd_vol.len(), 100);

        // Last 24 should be NaN
        assert!(fwd_vol[99].is_nan());
        assert!(fwd_vol[76].is_nan());

        // Earlier values should be valid
        assert!(!fwd_vol[0].is_nan());
        assert!(fwd_vol[0] >= 0.0);
    }

    #[test]
    fn test_regime_analysis() {
        let data = generate_predictive_data(500, 0.3);
        let forward_vol = data.compute_forward_volatility(24);

        let regime = analyze_regimes(&data.gini, &forward_vol, &data.current_volatility);

        assert!(regime.n_low_vol > 0);
        assert!(regime.n_high_vol > 0);
        // Both regimes should have some correlation (not exactly 0)
    }

    #[test]
    fn test_causality_analysis() {
        let data = generate_predictive_data(500, 0.4);

        let causality = analyze_causality(&data.gini, &data.current_volatility, 24);

        // Forward and reverse correlations should be computed
        // In synthetic data where concentration is designed to lead, forward should be higher
    }

    #[test]
    fn test_measure_results_structure() {
        let data = generate_predictive_data(500, 0.3);
        let config = H4TestConfig::default();

        let result = run_h4_concentration_vol_test(&data, &config);

        // Should test 5 measures × 2 horizons = 10 combinations
        let expected = H4TestData::measure_names().len() * config.horizons.len();
        assert_eq!(result.n_total, expected,
            "Should test {} combinations", expected);
    }

    #[test]
    fn test_report_sections() {
        let data = generate_predictive_data(500, 0.3);
        let config = H4TestConfig::default();

        let result = run_h4_concentration_vol_test(&data, &config);

        assert!(result.report.contains("## Decision:"));
        assert!(result.report.contains("## Data Summary"));
        assert!(result.report.contains("## Results by Concentration Measure"));
        assert!(result.report.contains("## Regime Analysis"));
        assert!(result.report.contains("## Causality Analysis"));
        assert!(result.report.contains("## Conclusion"));
    }

    #[test]
    fn test_decision_to_hypothesis_decision() {
        let data = generate_predictive_data(500, 0.3);
        let config = H4TestConfig::default();
        let result = run_h4_concentration_vol_test(&data, &config);

        let hyp_decision = result.to_hypothesis_decision();

        match result.decision {
            H4Decision::Go => assert_eq!(hyp_decision, HypothesisDecision::Accept),
            H4Decision::NoGo => assert_eq!(hyp_decision, HypothesisDecision::Reject),
            H4Decision::Inconclusive => assert_eq!(hyp_decision, HypothesisDecision::Inconclusive),
        }
    }

    #[test]
    fn test_residualize() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let resid = residualize(&y, &x);

        // Perfect linear relationship, residuals should be ~0
        let resid_sum: f64 = resid.iter().map(|r| r.abs()).sum();
        assert!(resid_sum < 1e-10, "Residuals should be ~0 for perfect fit");
    }

    #[test]
    fn test_variance_helper() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&data);
        assert!((v - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_median_helper() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = median(&data);
        assert!((m - 5.0).abs() < 0.01);
    }
}
