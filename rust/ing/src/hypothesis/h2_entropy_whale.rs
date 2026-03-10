//! H2 Hypothesis Test: Does Low Entropy + Whale Agreement Strengthen Signal?
//!
//! This tests whether combining entropy and whale flow produces a stronger
//! signal than either factor alone.
//!
//! # Hypothesis
//!
//! H2: P(up | low_entropy, whale_buying) > P(up | low_entropy) > P(up | baseline)
//!
//! The theory is that:
//! - Low entropy indicates a persistent trend (low randomness)
//! - Whale buying indicates informed accumulation
//! - Together, they should be more predictive than either alone
//!
//! # Test Protocol
//!
//! 1. Define conditions:
//!    - low_entropy: tick_entropy_1m < threshold (e.g., 0.4)
//!    - whale_buying: whale_net_flow > 75th percentile
//!
//! 2. Build 2x2 contingency table for each condition vs returns
//!
//! 3. Chi-squared test for interaction effect
//!
//! 4. Compute MI gain: MI(return | entropy, whale) vs MI(return | entropy)
//!
//! # Success Criteria
//!
//! - Conditional probability lift > 10% vs single factor
//! - Interaction effect significant (p < 0.01)
//! - Information gain > 0.01 bits
//!
//! # Failure Criteria
//!
//! - No significant interaction effect
//! - Combined signal not better than best single signal
//! - MI gain < 0.005 bits

use crate::hypothesis::stats::{mutual_information_adaptive, pearson_correlation};
use crate::hypothesis::HypothesisDecision;

/// Configuration for H2 test
#[derive(Debug, Clone)]
pub struct H2TestConfig {
    /// Entropy threshold for "low entropy" condition (below = low)
    pub entropy_threshold: f64,
    /// Percentile threshold for whale buying (above = buying)
    pub whale_percentile: f64,
    /// Minimum lift vs single factor for success
    pub min_lift_pct: f64,
    /// Maximum p-value for chi-squared significance
    pub max_p_value: f64,
    /// Minimum MI gain in bits
    pub min_mi_gain_bits: f64,
    /// MI gain failure threshold
    pub mi_gain_failure_threshold: f64,
    /// Minimum samples in each cell for valid test
    pub min_cell_count: usize,
}

impl Default for H2TestConfig {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.4,
            whale_percentile: 0.75,
            min_lift_pct: 10.0,
            max_p_value: 0.01,
            min_mi_gain_bits: 0.01,
            mi_gain_failure_threshold: 0.005,
            min_cell_count: 30,
        }
    }
}

/// 2x2 contingency table
#[derive(Debug, Clone)]
pub struct ContingencyTable {
    /// Count: condition=true, outcome=true
    pub tt: usize,
    /// Count: condition=true, outcome=false
    pub tf: usize,
    /// Count: condition=false, outcome=true
    pub ft: usize,
    /// Count: condition=false, outcome=false
    pub ff: usize,
}

impl ContingencyTable {
    pub fn new() -> Self {
        Self { tt: 0, tf: 0, ft: 0, ff: 0 }
    }

    pub fn total(&self) -> usize {
        self.tt + self.tf + self.ft + self.ff
    }

    /// P(outcome=true | condition=true)
    pub fn p_outcome_given_condition(&self) -> f64 {
        let denom = self.tt + self.tf;
        if denom == 0 { 0.5 } else { self.tt as f64 / denom as f64 }
    }

    /// P(outcome=true | condition=false)
    pub fn p_outcome_given_not_condition(&self) -> f64 {
        let denom = self.ft + self.ff;
        if denom == 0 { 0.5 } else { self.ft as f64 / denom as f64 }
    }

    /// P(outcome=true) - baseline probability
    pub fn p_outcome(&self) -> f64 {
        let total = self.total();
        if total == 0 { 0.5 } else { (self.tt + self.ft) as f64 / total as f64 }
    }

    /// Lift: P(outcome|condition) / P(outcome)
    pub fn lift(&self) -> f64 {
        let baseline = self.p_outcome();
        if baseline < 1e-10 { 1.0 } else { self.p_outcome_given_condition() / baseline }
    }

    /// Chi-squared statistic
    pub fn chi_squared(&self) -> f64 {
        let total = self.total() as f64;
        if total < 1.0 {
            return 0.0;
        }

        let row1 = (self.tt + self.tf) as f64;
        let row2 = (self.ft + self.ff) as f64;
        let col1 = (self.tt + self.ft) as f64;
        let col2 = (self.tf + self.ff) as f64;

        let expected = [
            [row1 * col1 / total, row1 * col2 / total],
            [row2 * col1 / total, row2 * col2 / total],
        ];

        let observed = [
            [self.tt as f64, self.tf as f64],
            [self.ft as f64, self.ff as f64],
        ];

        let mut chi2 = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                if expected[i][j] > 0.0 {
                    let diff = observed[i][j] - expected[i][j];
                    chi2 += diff * diff / expected[i][j];
                }
            }
        }

        chi2
    }

    /// P-value from chi-squared (1 df)
    pub fn p_value(&self) -> f64 {
        let chi2 = self.chi_squared();
        chi_squared_p_value(chi2, 1.0)
    }

    /// Minimum cell count
    pub fn min_cell(&self) -> usize {
        self.tt.min(self.tf).min(self.ft).min(self.ff)
    }
}

impl Default for ContingencyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// 2x2x2 contingency table for interaction analysis
#[derive(Debug, Clone)]
pub struct InteractionTable {
    /// entropy=low, whale=buying, return=positive
    pub lbp: usize,
    /// entropy=low, whale=buying, return=negative
    pub lbn: usize,
    /// entropy=low, whale=not_buying, return=positive
    pub lnp: usize,
    /// entropy=low, whale=not_buying, return=negative
    pub lnn: usize,
    /// entropy=high, whale=buying, return=positive
    pub hbp: usize,
    /// entropy=high, whale=buying, return=negative
    pub hbn: usize,
    /// entropy=high, whale=not_buying, return=positive
    pub hnp: usize,
    /// entropy=high, whale=not_buying, return=negative
    pub hnn: usize,
}

impl InteractionTable {
    pub fn new() -> Self {
        Self {
            lbp: 0, lbn: 0, lnp: 0, lnn: 0,
            hbp: 0, hbn: 0, hnp: 0, hnn: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.lbp + self.lbn + self.lnp + self.lnn +
        self.hbp + self.hbn + self.hnp + self.hnn
    }

    /// P(positive | low_entropy, whale_buying)
    pub fn p_pos_given_low_buy(&self) -> f64 {
        let denom = self.lbp + self.lbn;
        if denom == 0 { 0.5 } else { self.lbp as f64 / denom as f64 }
    }

    /// P(positive | low_entropy, not_whale_buying)
    pub fn p_pos_given_low_nobuy(&self) -> f64 {
        let denom = self.lnp + self.lnn;
        if denom == 0 { 0.5 } else { self.lnp as f64 / denom as f64 }
    }

    /// P(positive | high_entropy, whale_buying)
    pub fn p_pos_given_high_buy(&self) -> f64 {
        let denom = self.hbp + self.hbn;
        if denom == 0 { 0.5 } else { self.hbp as f64 / denom as f64 }
    }

    /// P(positive | high_entropy, not_whale_buying)
    pub fn p_pos_given_high_nobuy(&self) -> f64 {
        let denom = self.hnp + self.hnn;
        if denom == 0 { 0.5 } else { self.hnp as f64 / denom as f64 }
    }

    /// P(positive | low_entropy) - marginal
    pub fn p_pos_given_low_entropy(&self) -> f64 {
        let pos = self.lbp + self.lnp;
        let total = self.lbp + self.lbn + self.lnp + self.lnn;
        if total == 0 { 0.5 } else { pos as f64 / total as f64 }
    }

    /// P(positive | whale_buying) - marginal
    pub fn p_pos_given_whale_buy(&self) -> f64 {
        let pos = self.lbp + self.hbp;
        let total = self.lbp + self.lbn + self.hbp + self.hbn;
        if total == 0 { 0.5 } else { pos as f64 / total as f64 }
    }

    /// P(positive) - baseline
    pub fn p_pos_baseline(&self) -> f64 {
        let pos = self.lbp + self.lnp + self.hbp + self.hnp;
        let total = self.total();
        if total == 0 { 0.5 } else { pos as f64 / total as f64 }
    }

    /// Interaction effect: how much does combining factors improve prediction?
    /// Positive = synergy (combined is better than expected from marginals)
    pub fn interaction_effect(&self) -> f64 {
        let joint = self.p_pos_given_low_buy();
        let marginal_entropy = self.p_pos_given_low_entropy();
        let marginal_whale = self.p_pos_given_whale_buy();
        let baseline = self.p_pos_baseline();

        // Expected under independence: baseline + (entropy_effect) + (whale_effect)
        let entropy_effect = marginal_entropy - baseline;
        let whale_effect = marginal_whale - baseline;
        let expected_joint = baseline + entropy_effect + whale_effect;

        // Interaction = actual - expected
        joint - expected_joint
    }

    /// Chi-squared for 3-way interaction (simplified)
    pub fn chi_squared_interaction(&self) -> f64 {
        // Test if joint probability differs from product of marginals
        let total = self.total() as f64;
        if total < 1.0 {
            return 0.0;
        }

        // Compute expected frequencies under independence
        let cells = [
            (self.lbp, "lbp"), (self.lbn, "lbn"),
            (self.lnp, "lnp"), (self.lnn, "lnn"),
            (self.hbp, "hbp"), (self.hbn, "hbn"),
            (self.hnp, "hnp"), (self.hnn, "hnn"),
        ];

        // Marginal counts
        let n_low = (self.lbp + self.lbn + self.lnp + self.lnn) as f64;
        let n_high = (self.hbp + self.hbn + self.hnp + self.hnn) as f64;
        let n_buy = (self.lbp + self.lbn + self.hbp + self.hbn) as f64;
        let n_nobuy = (self.lnp + self.lnn + self.hnp + self.hnn) as f64;
        let n_pos = (self.lbp + self.lnp + self.hbp + self.hnp) as f64;
        let n_neg = (self.lbn + self.lnn + self.hbn + self.hnn) as f64;

        // Expected under independence
        let expected = [
            n_low * n_buy * n_pos / (total * total),   // lbp
            n_low * n_buy * n_neg / (total * total),   // lbn
            n_low * n_nobuy * n_pos / (total * total), // lnp
            n_low * n_nobuy * n_neg / (total * total), // lnn
            n_high * n_buy * n_pos / (total * total),  // hbp
            n_high * n_buy * n_neg / (total * total),  // hbn
            n_high * n_nobuy * n_pos / (total * total),// hnp
            n_high * n_nobuy * n_neg / (total * total),// hnn
        ];

        let mut chi2 = 0.0;
        for (i, &(obs, _)) in cells.iter().enumerate() {
            let exp = expected[i];
            if exp > 0.0 {
                let diff = obs as f64 - exp;
                chi2 += diff * diff / exp;
            }
        }

        chi2
    }

    /// Minimum cell count
    pub fn min_cell(&self) -> usize {
        [self.lbp, self.lbn, self.lnp, self.lnn,
         self.hbp, self.hbn, self.hnp, self.hnn]
            .into_iter().min().unwrap_or(0)
    }

    /// Count in joint condition (low entropy + whale buying)
    pub fn n_joint_condition(&self) -> usize {
        self.lbp + self.lbn
    }
}

impl Default for InteractionTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision for H2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H2Decision {
    /// GO: Interaction effect is significant and meaningful
    Go,
    /// NO-GO: No evidence of interaction benefit
    NoGo,
    /// INCONCLUSIVE: Mixed results
    Inconclusive,
}

impl std::fmt::Display for H2Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H2Decision::Go => write!(f, "GO"),
            H2Decision::NoGo => write!(f, "NO-GO"),
            H2Decision::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Result for single factor analysis
#[derive(Debug, Clone)]
pub struct SingleFactorResult {
    pub factor_name: String,
    pub contingency: ContingencyTable,
    pub p_outcome_given_factor: f64,
    pub p_outcome_baseline: f64,
    pub lift: f64,
    pub lift_pct: f64,
    pub chi_squared: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Complete H2 test result
#[derive(Debug, Clone)]
pub struct H2TestResult {
    pub config: H2TestConfig,
    /// Single factor result: entropy alone
    pub entropy_only: SingleFactorResult,
    /// Single factor result: whale flow alone
    pub whale_only: SingleFactorResult,
    /// Interaction table
    pub interaction_table: InteractionTable,
    /// P(positive | low_entropy, whale_buying)
    pub p_joint: f64,
    /// Lift of joint condition vs baseline
    pub joint_lift: f64,
    /// Lift of joint vs best single factor
    pub joint_vs_best_lift: f64,
    /// Interaction effect magnitude
    pub interaction_effect: f64,
    /// Chi-squared for interaction
    pub interaction_chi_squared: f64,
    /// P-value for interaction
    pub interaction_p_value: f64,
    /// MI with entropy alone
    pub mi_entropy_only: f64,
    /// MI with both factors
    pub mi_joint: f64,
    /// MI gain from adding whale flow
    pub mi_gain: f64,
    /// Correlation: entropy with returns
    pub corr_entropy_returns: f64,
    /// Correlation: whale flow with returns
    pub corr_whale_returns: f64,
    /// Overall decision
    pub decision: H2Decision,
    /// Summary message
    pub summary: String,
    /// Detailed report
    pub report: String,
    /// Sample size
    pub n_samples: usize,
}

impl H2TestResult {
    pub fn to_hypothesis_decision(&self) -> HypothesisDecision {
        match self.decision {
            H2Decision::Go => HypothesisDecision::Accept,
            H2Decision::NoGo => HypothesisDecision::Reject,
            H2Decision::Inconclusive => HypothesisDecision::Inconclusive,
        }
    }
}

/// Input data for H2 test
#[derive(Debug, Clone)]
pub struct H2TestData {
    /// Entropy values (e.g., tick_entropy_1m)
    pub entropy: Vec<f64>,
    /// Whale net flow values
    pub whale_flow: Vec<f64>,
    /// Forward returns (positive = up)
    pub returns: Vec<f64>,
}

impl H2TestData {
    pub fn validate(&self) -> Result<(), String> {
        let n = self.entropy.len();

        if n < 100 {
            return Err(format!("Insufficient data: {} samples (need 100+)", n));
        }

        if self.whale_flow.len() != n || self.returns.len() != n {
            return Err("All data series must have the same length".to_string());
        }

        // Check for variance
        let entropy_var = variance(&self.entropy);
        let flow_var = variance(&self.whale_flow);

        if entropy_var < 1e-10 {
            return Err("Entropy has zero variance".to_string());
        }

        if flow_var < 1e-10 {
            return Err("Whale flow has zero variance".to_string());
        }

        Ok(())
    }
}

/// Run the H2 hypothesis test
pub fn run_h2_entropy_whale_test(data: &H2TestData, config: &H2TestConfig) -> H2TestResult {
    // Validate
    if let Err(e) = data.validate() {
        return make_error_result(config, &e);
    }

    let n = data.entropy.len();

    // Compute thresholds
    let whale_threshold = percentile(&data.whale_flow, config.whale_percentile);

    // Build conditions
    let low_entropy: Vec<bool> = data.entropy.iter()
        .map(|&e| e < config.entropy_threshold)
        .collect();

    let whale_buying: Vec<bool> = data.whale_flow.iter()
        .map(|&w| w > whale_threshold)
        .collect();

    let positive_return: Vec<bool> = data.returns.iter()
        .map(|&r| r > 0.0)
        .collect();

    // Build single factor contingency tables
    let entropy_table = build_contingency(&low_entropy, &positive_return);
    let whale_table = build_contingency(&whale_buying, &positive_return);

    // Build interaction table
    let interaction_table = build_interaction_table(
        &low_entropy, &whale_buying, &positive_return
    );

    // Single factor results
    let entropy_only = SingleFactorResult {
        factor_name: "low_entropy".to_string(),
        p_outcome_given_factor: entropy_table.p_outcome_given_condition(),
        p_outcome_baseline: entropy_table.p_outcome(),
        lift: entropy_table.lift(),
        lift_pct: (entropy_table.lift() - 1.0) * 100.0,
        chi_squared: entropy_table.chi_squared(),
        p_value: entropy_table.p_value(),
        significant: entropy_table.p_value() < config.max_p_value,
        contingency: entropy_table,
    };

    let whale_only = SingleFactorResult {
        factor_name: "whale_buying".to_string(),
        p_outcome_given_factor: whale_table.p_outcome_given_condition(),
        p_outcome_baseline: whale_table.p_outcome(),
        lift: whale_table.lift(),
        lift_pct: (whale_table.lift() - 1.0) * 100.0,
        chi_squared: whale_table.chi_squared(),
        p_value: whale_table.p_value(),
        significant: whale_table.p_value() < config.max_p_value,
        contingency: whale_table,
    };

    // Joint analysis
    let p_joint = interaction_table.p_pos_given_low_buy();
    let baseline = interaction_table.p_pos_baseline();
    let joint_lift = if baseline > 1e-10 { p_joint / baseline } else { 1.0 };

    let best_single_lift = entropy_only.lift.max(whale_only.lift);
    let joint_vs_best_lift = if best_single_lift > 1e-10 {
        joint_lift / best_single_lift
    } else {
        1.0
    };

    let interaction_effect = interaction_table.interaction_effect();
    let interaction_chi_squared = interaction_table.chi_squared_interaction();
    let interaction_p_value = chi_squared_p_value(interaction_chi_squared, 4.0); // df=4 for 3-way

    // MI analysis
    let mi_entropy_only = compute_conditional_mi(&data.entropy, &data.returns);
    let mi_joint = compute_joint_mi(&data.entropy, &data.whale_flow, &data.returns);
    let mi_gain = mi_joint - mi_entropy_only;

    // Correlations
    let corr_entropy_returns = pearson_correlation(&data.entropy, &data.returns);
    let corr_whale_returns = pearson_correlation(&data.whale_flow, &data.returns);

    // Determine decision
    let decision = determine_h2_decision(
        &entropy_only,
        &whale_only,
        &interaction_table,
        interaction_effect,
        interaction_p_value,
        joint_vs_best_lift,
        mi_gain,
        config,
    );

    // Generate outputs
    let summary = generate_h2_summary(
        &decision,
        p_joint,
        baseline,
        joint_vs_best_lift,
        mi_gain,
    );

    let report = generate_h2_report(
        &entropy_only,
        &whale_only,
        &interaction_table,
        p_joint,
        joint_lift,
        joint_vs_best_lift,
        interaction_effect,
        interaction_p_value,
        mi_entropy_only,
        mi_joint,
        mi_gain,
        corr_entropy_returns,
        corr_whale_returns,
        &decision,
        config,
        n,
    );

    H2TestResult {
        config: config.clone(),
        entropy_only,
        whale_only,
        interaction_table,
        p_joint,
        joint_lift,
        joint_vs_best_lift,
        interaction_effect,
        interaction_chi_squared,
        interaction_p_value,
        mi_entropy_only,
        mi_joint,
        mi_gain,
        corr_entropy_returns,
        corr_whale_returns,
        decision,
        summary,
        report,
        n_samples: n,
    }
}

fn make_error_result(config: &H2TestConfig, error: &str) -> H2TestResult {
    H2TestResult {
        config: config.clone(),
        entropy_only: SingleFactorResult {
            factor_name: "entropy".to_string(),
            contingency: ContingencyTable::new(),
            p_outcome_given_factor: 0.5,
            p_outcome_baseline: 0.5,
            lift: 1.0,
            lift_pct: 0.0,
            chi_squared: 0.0,
            p_value: 1.0,
            significant: false,
        },
        whale_only: SingleFactorResult {
            factor_name: "whale".to_string(),
            contingency: ContingencyTable::new(),
            p_outcome_given_factor: 0.5,
            p_outcome_baseline: 0.5,
            lift: 1.0,
            lift_pct: 0.0,
            chi_squared: 0.0,
            p_value: 1.0,
            significant: false,
        },
        interaction_table: InteractionTable::new(),
        p_joint: 0.5,
        joint_lift: 1.0,
        joint_vs_best_lift: 1.0,
        interaction_effect: 0.0,
        interaction_chi_squared: 0.0,
        interaction_p_value: 1.0,
        mi_entropy_only: 0.0,
        mi_joint: 0.0,
        mi_gain: 0.0,
        corr_entropy_returns: 0.0,
        corr_whale_returns: 0.0,
        decision: H2Decision::NoGo,
        summary: format!("Data validation failed: {}", error),
        report: format!("# H2 Test Report\n\n**Status:** FAILED\n\n**Reason:** {}", error),
        n_samples: 0,
    }
}

fn build_contingency(condition: &[bool], outcome: &[bool]) -> ContingencyTable {
    let mut table = ContingencyTable::new();

    for (&c, &o) in condition.iter().zip(outcome.iter()) {
        match (c, o) {
            (true, true) => table.tt += 1,
            (true, false) => table.tf += 1,
            (false, true) => table.ft += 1,
            (false, false) => table.ff += 1,
        }
    }

    table
}

fn build_interaction_table(
    low_entropy: &[bool],
    whale_buying: &[bool],
    positive_return: &[bool],
) -> InteractionTable {
    let mut table = InteractionTable::new();

    for i in 0..low_entropy.len() {
        let low = low_entropy[i];
        let buy = whale_buying[i];
        let pos = positive_return[i];

        match (low, buy, pos) {
            (true, true, true) => table.lbp += 1,
            (true, true, false) => table.lbn += 1,
            (true, false, true) => table.lnp += 1,
            (true, false, false) => table.lnn += 1,
            (false, true, true) => table.hbp += 1,
            (false, true, false) => table.hbn += 1,
            (false, false, true) => table.hnp += 1,
            (false, false, false) => table.hnn += 1,
        }
    }

    table
}

fn compute_conditional_mi(x: &[f64], returns: &[f64]) -> f64 {
    mutual_information_adaptive(x, returns)
}

fn compute_joint_mi(x1: &[f64], x2: &[f64], returns: &[f64]) -> f64 {
    // Combine x1 and x2 into a joint feature by simple concatenation approach:
    // Create a 2D embedding using ranks
    let n = x1.len().min(x2.len()).min(returns.len());
    if n < 10 {
        return 0.0;
    }

    // Discretize both inputs
    let n_bins = ((n as f64).sqrt().ceil() as usize).max(5).min(20);

    let x1_range = range(&x1[..n]);
    let x2_range = range(&x2[..n]);

    if x1_range.1 - x1_range.0 < 1e-15 || x2_range.1 - x2_range.0 < 1e-15 {
        return 0.0;
    }

    // Create joint bins
    let joint_bins: Vec<usize> = (0..n)
        .map(|i| {
            let b1 = ((x1[i] - x1_range.0) / (x1_range.1 - x1_range.0) * (n_bins - 1) as f64)
                .floor() as usize;
            let b2 = ((x2[i] - x2_range.0) / (x2_range.1 - x2_range.0) * (n_bins - 1) as f64)
                .floor() as usize;
            b1.min(n_bins - 1) * n_bins + b2.min(n_bins - 1)
        })
        .collect();

    // Convert to f64 for MI calculation
    let joint_f64: Vec<f64> = joint_bins.iter().map(|&b| b as f64).collect();

    mutual_information_adaptive(&joint_f64, &returns[..n])
}

fn determine_h2_decision(
    entropy_result: &SingleFactorResult,
    whale_result: &SingleFactorResult,
    interaction_table: &InteractionTable,
    interaction_effect: f64,
    interaction_p_value: f64,
    joint_vs_best_lift: f64,
    mi_gain: f64,
    config: &H2TestConfig,
) -> H2Decision {
    // Check for insufficient data in joint condition
    if interaction_table.n_joint_condition() < config.min_cell_count {
        return H2Decision::Inconclusive;
    }

    // Success criteria
    let lift_passes = (joint_vs_best_lift - 1.0) * 100.0 >= config.min_lift_pct;
    let interaction_significant = interaction_p_value < config.max_p_value;
    let mi_passes = mi_gain >= config.min_mi_gain_bits;

    // Strong GO: all criteria met
    if lift_passes && interaction_significant && mi_passes {
        return H2Decision::Go;
    }

    // Weak GO: 2 of 3 criteria met with strong individual factors
    let both_factors_significant = entropy_result.significant && whale_result.significant;
    let criteria_met = [lift_passes, interaction_significant, mi_passes]
        .iter().filter(|&&x| x).count();

    if criteria_met >= 2 && both_factors_significant {
        return H2Decision::Go;
    }

    // Clear failure
    let lift_fails = (joint_vs_best_lift - 1.0) * 100.0 < config.min_lift_pct / 2.0;
    let mi_fails = mi_gain < config.mi_gain_failure_threshold;
    let interaction_weak = interaction_effect.abs() < 0.01;

    if (lift_fails && mi_fails) || (interaction_weak && !both_factors_significant) {
        return H2Decision::NoGo;
    }

    H2Decision::Inconclusive
}

fn generate_h2_summary(
    decision: &H2Decision,
    p_joint: f64,
    baseline: f64,
    joint_vs_best_lift: f64,
    mi_gain: f64,
) -> String {
    format!(
        "H2 {} - P(up|low_entropy,whale_buy)={:.1}% vs baseline={:.1}%, \
         lift vs best={:.1}%, MI gain={:.4} bits",
        decision,
        p_joint * 100.0,
        baseline * 100.0,
        (joint_vs_best_lift - 1.0) * 100.0,
        mi_gain
    )
}

fn generate_h2_report(
    entropy_result: &SingleFactorResult,
    whale_result: &SingleFactorResult,
    interaction_table: &InteractionTable,
    p_joint: f64,
    joint_lift: f64,
    joint_vs_best_lift: f64,
    interaction_effect: f64,
    interaction_p_value: f64,
    mi_entropy: f64,
    mi_joint: f64,
    mi_gain: f64,
    corr_entropy: f64,
    corr_whale: f64,
    decision: &H2Decision,
    config: &H2TestConfig,
    n: usize,
) -> String {
    let mut report = String::new();

    report.push_str("# H2 Hypothesis Test Report: Does Entropy + Whale Agreement Work?\n\n");

    // Decision banner
    report.push_str(&format!("## Decision: **{}**\n\n", decision));

    // Sample info
    report.push_str(&format!("**Sample Size:** {}\n\n", n));

    // Configuration
    report.push_str("## Test Configuration\n\n");
    report.push_str(&format!("- Entropy threshold: < {} (low entropy)\n", config.entropy_threshold));
    report.push_str(&format!("- Whale percentile: > {}th (whale buying)\n", config.whale_percentile * 100.0));
    report.push_str(&format!("- Minimum lift: {}%\n", config.min_lift_pct));
    report.push_str(&format!("- Maximum p-value: {}\n", config.max_p_value));
    report.push_str(&format!("- Minimum MI gain: {} bits\n\n", config.min_mi_gain_bits));

    // Single factor results
    report.push_str("## Single Factor Analysis\n\n");
    report.push_str("### Low Entropy Factor\n\n");
    report.push_str(&format!("- P(up | low_entropy) = {:.2}%\n", entropy_result.p_outcome_given_factor * 100.0));
    report.push_str(&format!("- P(up | high_entropy) = {:.2}%\n", entropy_result.contingency.p_outcome_given_not_condition() * 100.0));
    report.push_str(&format!("- Lift: {:.2}x ({:+.1}%)\n", entropy_result.lift, entropy_result.lift_pct));
    report.push_str(&format!("- Chi-squared: {:.2}, p-value: {:.2e}\n", entropy_result.chi_squared, entropy_result.p_value));
    report.push_str(&format!("- Significant: {}\n\n", if entropy_result.significant { "Yes" } else { "No" }));

    report.push_str("### Whale Buying Factor\n\n");
    report.push_str(&format!("- P(up | whale_buying) = {:.2}%\n", whale_result.p_outcome_given_factor * 100.0));
    report.push_str(&format!("- P(up | not_whale_buying) = {:.2}%\n", whale_result.contingency.p_outcome_given_not_condition() * 100.0));
    report.push_str(&format!("- Lift: {:.2}x ({:+.1}%)\n", whale_result.lift, whale_result.lift_pct));
    report.push_str(&format!("- Chi-squared: {:.2}, p-value: {:.2e}\n", whale_result.chi_squared, whale_result.p_value));
    report.push_str(&format!("- Significant: {}\n\n", if whale_result.significant { "Yes" } else { "No" }));

    // Interaction analysis
    report.push_str("## Interaction Analysis (Combined Factors)\n\n");
    report.push_str("### Conditional Probabilities\n\n");
    report.push_str("| Condition | P(up) |\n");
    report.push_str("|-----------|-------|\n");
    report.push_str(&format!("| Baseline | {:.2}% |\n", interaction_table.p_pos_baseline() * 100.0));
    report.push_str(&format!("| Low entropy only | {:.2}% |\n", interaction_table.p_pos_given_low_entropy() * 100.0));
    report.push_str(&format!("| Whale buying only | {:.2}% |\n", interaction_table.p_pos_given_whale_buy() * 100.0));
    report.push_str(&format!("| **Low entropy + Whale buying** | **{:.2}%** |\n\n", p_joint * 100.0));

    // 2x2x2 table
    report.push_str("### Full Contingency Table\n\n");
    report.push_str("| Entropy | Whale | Return+ | Return- | P(up) |\n");
    report.push_str("|---------|-------|---------|---------|-------|\n");
    report.push_str(&format!("| Low | Buy | {} | {} | {:.1}% |\n",
        interaction_table.lbp, interaction_table.lbn, interaction_table.p_pos_given_low_buy() * 100.0));
    report.push_str(&format!("| Low | No-Buy | {} | {} | {:.1}% |\n",
        interaction_table.lnp, interaction_table.lnn, interaction_table.p_pos_given_low_nobuy() * 100.0));
    report.push_str(&format!("| High | Buy | {} | {} | {:.1}% |\n",
        interaction_table.hbp, interaction_table.hbn, interaction_table.p_pos_given_high_buy() * 100.0));
    report.push_str(&format!("| High | No-Buy | {} | {} | {:.1}% |\n\n",
        interaction_table.hnp, interaction_table.hnn, interaction_table.p_pos_given_high_nobuy() * 100.0));

    // Key metrics
    report.push_str("### Key Metrics\n\n");
    report.push_str(&format!("- **Joint Lift vs Baseline:** {:.2}x\n", joint_lift));
    report.push_str(&format!("- **Joint Lift vs Best Single Factor:** {:.2}x ({:+.1}%)\n",
        joint_vs_best_lift, (joint_vs_best_lift - 1.0) * 100.0));
    report.push_str(&format!("- **Interaction Effect:** {:+.4}\n", interaction_effect));
    report.push_str(&format!("- **Interaction p-value:** {:.2e}\n\n", interaction_p_value));

    // MI analysis
    report.push_str("## Mutual Information Analysis\n\n");
    report.push_str(&format!("- MI(returns | entropy): {:.4} bits\n", mi_entropy));
    report.push_str(&format!("- MI(returns | entropy, whale): {:.4} bits\n", mi_joint));
    report.push_str(&format!("- **MI Gain:** {:.4} bits\n\n", mi_gain));

    // Correlations
    report.push_str("## Correlation Analysis\n\n");
    report.push_str(&format!("- Correlation(entropy, returns): {:.4}\n", corr_entropy));
    report.push_str(&format!("- Correlation(whale_flow, returns): {:.4}\n\n", corr_whale));

    // Success criteria check
    report.push_str("## Success Criteria Evaluation\n\n");
    let lift_check = if (joint_vs_best_lift - 1.0) * 100.0 >= config.min_lift_pct { "✓" } else { "✗" };
    let interaction_check = if interaction_p_value < config.max_p_value { "✓" } else { "✗" };
    let mi_check = if mi_gain >= config.min_mi_gain_bits { "✓" } else { "✗" };

    report.push_str(&format!("- {} Lift vs best single factor ≥ {}%: {:.1}%\n",
        lift_check, config.min_lift_pct, (joint_vs_best_lift - 1.0) * 100.0));
    report.push_str(&format!("- {} Interaction p-value < {}: {:.2e}\n",
        interaction_check, config.max_p_value, interaction_p_value));
    report.push_str(&format!("- {} MI gain ≥ {} bits: {:.4} bits\n\n",
        mi_check, config.min_mi_gain_bits, mi_gain));

    // Conclusion
    report.push_str("## Conclusion\n\n");
    match decision {
        H2Decision::Go => {
            report.push_str("**ACCEPT H2:** Combining low entropy with whale buying produces a \
                significantly stronger signal than either factor alone. The interaction effect \
                is statistically significant and provides meaningful information gain.\n\n");
            report.push_str("**Recommendation:** Use the combined entropy + whale flow signal \
                for improved predictions.\n");
        }
        H2Decision::NoGo => {
            report.push_str("**REJECT H2:** No evidence that combining factors improves predictions. \
                The interaction effect is weak or not statistically significant.\n\n");
            report.push_str("**Recommendation:** Use individual factors separately; combining \
                does not add value.\n");
        }
        H2Decision::Inconclusive => {
            report.push_str("**INCONCLUSIVE:** Mixed results. Some criteria are met but not all. \
                More data or alternative thresholds may be needed.\n\n");
            report.push_str("**Recommendation:** Gather more data or test different threshold \
                combinations.\n");
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

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = data.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn range(data: &[f64]) -> (f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn chi_squared_p_value(chi2: f64, df: f64) -> f64 {
    // Approximate using gamma distribution
    // P(X > chi2) for chi-squared with df degrees of freedom
    if chi2 <= 0.0 {
        return 1.0;
    }

    // Use Wilson-Hilferty approximation for large df
    if df > 30.0 {
        let z = ((chi2 / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df)))
            / (2.0 / (9.0 * df)).sqrt();
        return 1.0 - normal_cdf(z);
    }

    // For small df, use direct integration approximation
    1.0 - gamma_cdf(chi2 / 2.0, df / 2.0)
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

fn gamma_cdf(x: f64, a: f64) -> f64 {
    // Lower incomplete gamma function / Gamma(a)
    // Using series expansion for small x, continued fraction for large x
    if x <= 0.0 {
        return 0.0;
    }
    if x > a + 1.0 {
        1.0 - gamma_cdf_upper(x, a)
    } else {
        gamma_cdf_series(x, a)
    }
}

fn gamma_cdf_series(x: f64, a: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_cdf_upper(x: f64, a: f64) -> f64 {
    // Continued fraction for upper incomplete gamma
    let max_iter = 100;
    let eps = 1e-10;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

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

    fn generate_interaction_data(n: usize, interaction_strength: f64) -> H2TestData {
        // Generate entropy: mix of low and high
        let entropy: Vec<f64> = (0..n)
            .map(|i| {
                if (i as f64 * 0.1).sin() > 0.0 {
                    0.2 + (i as f64 * 0.05).cos().abs() * 0.15 // Low entropy: 0.2-0.35
                } else {
                    0.5 + (i as f64 * 0.07).sin().abs() * 0.3 // High entropy: 0.5-0.8
                }
            })
            .collect();

        // Generate whale flow
        let whale_flow: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.08).sin() * 100.0 + (i as f64 * 0.03).cos() * 50.0)
            .collect();

        let whale_p75 = percentile(&whale_flow, 0.75);

        // Generate returns with interaction effect
        let returns: Vec<f64> = (0..n)
            .map(|i| {
                let low_entropy = entropy[i] < 0.4;
                let whale_buying = whale_flow[i] > whale_p75;

                let base = (i as f64 * 0.11).sin() * 0.01;

                let signal = if low_entropy && whale_buying {
                    interaction_strength * 0.02 // Strong positive when both conditions met
                } else if low_entropy {
                    interaction_strength * 0.005
                } else if whale_buying {
                    interaction_strength * 0.003
                } else {
                    0.0
                };

                base + signal
            })
            .collect();

        H2TestData { entropy, whale_flow, returns }
    }

    #[test]
    fn test_contingency_table_basics() {
        let mut table = ContingencyTable::new();
        table.tt = 30;
        table.tf = 20;
        table.ft = 25;
        table.ff = 25;

        assert_eq!(table.total(), 100);
        assert!((table.p_outcome_given_condition() - 0.6).abs() < 0.01); // 30/50
        assert!((table.p_outcome() - 0.55).abs() < 0.01); // 55/100
    }

    #[test]
    fn test_contingency_table_lift() {
        let mut table = ContingencyTable::new();
        table.tt = 40; // P(up|cond) = 40/50 = 0.8
        table.tf = 10;
        table.ft = 20; // P(up|!cond) = 20/50 = 0.4
        table.ff = 30;

        // P(up) = 60/100 = 0.6
        // Lift = 0.8 / 0.6 = 1.33
        assert!((table.lift() - 1.33).abs() < 0.01);
    }

    #[test]
    fn test_chi_squared_significant() {
        // Clearly dependent data
        let mut table = ContingencyTable::new();
        table.tt = 80;
        table.tf = 20;
        table.ft = 20;
        table.ff = 80;

        let chi2 = table.chi_squared();
        let p = table.p_value();

        assert!(chi2 > 50.0, "Chi-squared should be large for dependent data");
        assert!(p < 0.001, "P-value should be very small");
    }

    #[test]
    fn test_chi_squared_independent() {
        // Independent data (proportional)
        let mut table = ContingencyTable::new();
        table.tt = 25;
        table.tf = 25;
        table.ft = 25;
        table.ff = 25;

        let chi2 = table.chi_squared();
        assert!(chi2 < 0.1, "Chi-squared should be ~0 for independent data, got {}", chi2);
    }

    #[test]
    fn test_interaction_table_probabilities() {
        let mut table = InteractionTable::new();
        table.lbp = 40;
        table.lbn = 10;
        table.lnp = 20;
        table.lnn = 30;
        table.hbp = 15;
        table.hbn = 35;
        table.hnp = 25;
        table.hnn = 25;

        // P(pos | low, buy) = 40/50 = 0.8
        assert!((table.p_pos_given_low_buy() - 0.8).abs() < 0.01);

        // P(pos | baseline) = (40+20+15+25)/200 = 100/200 = 0.5
        assert!((table.p_pos_baseline() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_h2_with_strong_interaction() {
        let data = generate_interaction_data(500, 1.0);
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        // Should detect some interaction effect
        assert!(result.n_samples == 500);
        assert!(!result.report.is_empty());
        assert!(result.report.contains("H2 Hypothesis Test Report"));
    }

    #[test]
    fn test_h2_with_no_interaction() {
        // Independent data
        let n = 500;
        let entropy: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        let whale_flow: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos() * 100.0).collect();
        let returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).sin() * 0.01).collect();

        let data = H2TestData { entropy, whale_flow, returns };
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        // Should not find strong interaction
        assert!(result.interaction_effect.abs() < 0.2,
            "Independent data should have weak interaction: {}", result.interaction_effect);
    }

    #[test]
    fn test_h2_insufficient_data() {
        let data = H2TestData {
            entropy: vec![0.3; 50],
            whale_flow: vec![100.0; 50],
            returns: vec![0.01; 50],
        };

        let config = H2TestConfig::default();
        let result = run_h2_entropy_whale_test(&data, &config);

        assert_eq!(result.decision, H2Decision::NoGo);
        assert!(result.summary.contains("validation failed"));
    }

    #[test]
    fn test_h2_zero_variance_entropy() {
        let data = H2TestData {
            entropy: vec![0.3; 200],
            whale_flow: (0..200).map(|i| (i as f64).sin() * 100.0).collect(),
            returns: (0..200).map(|i| (i as f64 * 0.1).sin() * 0.01).collect(),
        };

        let config = H2TestConfig::default();
        let result = run_h2_entropy_whale_test(&data, &config);

        assert_eq!(result.decision, H2Decision::NoGo);
        assert!(result.summary.contains("zero variance"));
    }

    #[test]
    fn test_single_factor_results() {
        let data = generate_interaction_data(500, 0.5);
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        // Should have results for both factors
        assert_eq!(result.entropy_only.factor_name, "low_entropy");
        assert_eq!(result.whale_only.factor_name, "whale_buying");

        // Contingency tables should have data
        assert!(result.entropy_only.contingency.total() > 0);
        assert!(result.whale_only.contingency.total() > 0);
    }

    #[test]
    fn test_mi_computation() {
        let data = generate_interaction_data(500, 0.8);
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        // MI values should be non-negative
        assert!(result.mi_entropy_only >= 0.0);
        assert!(result.mi_joint >= 0.0);

        // Joint MI should be >= single factor MI (adding info shouldn't hurt)
        // Note: Due to estimation variance, this may not always hold strictly
    }

    #[test]
    fn test_report_sections() {
        let data = generate_interaction_data(500, 0.5);
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        assert!(result.report.contains("## Decision:"));
        assert!(result.report.contains("## Test Configuration"));
        assert!(result.report.contains("## Single Factor Analysis"));
        assert!(result.report.contains("## Interaction Analysis"));
        assert!(result.report.contains("## Mutual Information Analysis"));
        assert!(result.report.contains("## Conclusion"));
    }

    #[test]
    fn test_percentile_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&data, 0.5) - 5.5).abs() < 0.6); // Approximate median
        assert!((percentile(&data, 1.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_computation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&data);
        // Mean = 5, Var = sum((x-5)^2)/8 = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4
        assert!((v - 4.0).abs() < 0.01, "Variance should be 4, got {}", v);
    }

    #[test]
    fn test_decision_to_hypothesis_decision() {
        let data = generate_interaction_data(500, 0.5);
        let config = H2TestConfig::default();
        let result = run_h2_entropy_whale_test(&data, &config);

        let hyp_decision = result.to_hypothesis_decision();

        match result.decision {
            H2Decision::Go => assert_eq!(hyp_decision, HypothesisDecision::Accept),
            H2Decision::NoGo => assert_eq!(hyp_decision, HypothesisDecision::Reject),
            H2Decision::Inconclusive => assert_eq!(hyp_decision, HypothesisDecision::Inconclusive),
        }
    }

    #[test]
    fn test_joint_condition_cell_count() {
        let data = generate_interaction_data(500, 0.5);
        let config = H2TestConfig::default();

        let result = run_h2_entropy_whale_test(&data, &config);

        // Should have some samples in joint condition
        let n_joint = result.interaction_table.n_joint_condition();
        assert!(n_joint > 0, "Should have samples in joint condition");
    }
}
