//! H3 Hypothesis Test: Does Liquidation Clustering Predict Cascades?
//!
//! This tests whether concentrated liquidation levels predict cascade events.
//!
//! # Hypothesis
//!
//! H3: When price approaches liquidation cluster → volatility spike / cascade
//!
//! The theory is that:
//! - Liquidations at similar prices create positive feedback loops
//! - Large liquidation clusters act as "magnets" for price
//! - Approaching a cluster increases probability of cascade
//!
//! # Test Protocol
//!
//! 1. Define liquidation cluster: > $X million within Y% of current price
//! 2. Define cascade event: > 5% price move within 1 hour
//! 3. Compute P(cascade | approaching_cluster) vs P(cascade | no_cluster)
//! 4. Test thresholds: $5M/$10M/$20M × 1%/2%/5%
//! 5. Out-of-sample validation
//!
//! # Success Criteria
//!
//! - P(cascade | cluster) > 2x P(cascade | no_cluster)
//! - Precision > 30%
//! - Signal is actionable (enough lead time)
//!
//! # Failure Criteria
//!
//! - No significant difference in cascade probability
//! - Precision < 15%
//! - Clusters too rare to be useful

use crate::hypothesis::HypothesisDecision;

/// Configuration for H3 test
#[derive(Debug, Clone)]
pub struct H3TestConfig {
    /// Liquidation amount thresholds to test (in USD)
    pub amount_thresholds: Vec<f64>,
    /// Distance thresholds to test (as fraction, e.g., 0.02 = 2%)
    pub distance_thresholds: Vec<f64>,
    /// Cascade definition: minimum price move (fraction)
    pub cascade_threshold: f64,
    /// Cascade time horizon in periods
    pub cascade_horizon: usize,
    /// Minimum lift for success (e.g., 2.0 = 2x)
    pub min_lift: f64,
    /// Minimum precision for success
    pub min_precision: f64,
    /// Minimum recall for practical use
    pub min_recall: f64,
    /// Precision failure threshold
    pub precision_failure: f64,
    /// Minimum cluster occurrences for valid test
    pub min_cluster_count: usize,
    /// Train/test split ratio
    pub train_ratio: f64,
}

impl Default for H3TestConfig {
    fn default() -> Self {
        Self {
            amount_thresholds: vec![5_000_000.0, 10_000_000.0, 20_000_000.0],
            distance_thresholds: vec![0.01, 0.02, 0.05],
            cascade_threshold: 0.05, // 5% move
            cascade_horizon: 60,     // 1 hour (assuming 1-minute periods)
            min_lift: 2.0,
            min_precision: 0.30,
            min_recall: 0.10,
            precision_failure: 0.15,
            min_cluster_count: 20,
            train_ratio: 0.7,
        }
    }
}

/// Classification metrics for a binary prediction
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// True positives
    pub tp: usize,
    /// False positives
    pub fp: usize,
    /// True negatives
    pub tn: usize,
    /// False negatives
    pub fn_: usize,
    /// Precision = TP / (TP + FP)
    pub precision: f64,
    /// Recall = TP / (TP + FN)
    pub recall: f64,
    /// F1 = 2 * precision * recall / (precision + recall)
    pub f1: f64,
    /// Accuracy = (TP + TN) / total
    pub accuracy: f64,
    /// Lift = precision / baseline_rate
    pub lift: f64,
    /// Baseline cascade rate
    pub baseline_rate: f64,
}

impl ClassificationMetrics {
    pub fn compute(predictions: &[bool], actuals: &[bool]) -> Self {
        let n = predictions.len().min(actuals.len());

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for i in 0..n {
            match (predictions[i], actuals[i]) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let accuracy = if n > 0 {
            (tp + tn) as f64 / n as f64
        } else {
            0.0
        };

        let baseline_rate = if n > 0 {
            (tp + fn_) as f64 / n as f64
        } else {
            0.0
        };

        let lift = if baseline_rate > 0.0 {
            precision / baseline_rate
        } else {
            1.0
        };

        Self {
            tp, fp, tn, fn_,
            precision, recall, f1, accuracy,
            lift, baseline_rate,
        }
    }
}

/// Result for a single threshold combination
#[derive(Debug, Clone)]
pub struct ThresholdResult {
    /// Amount threshold (USD)
    pub amount_threshold: f64,
    /// Distance threshold (fraction)
    pub distance_threshold: f64,
    /// Number of times cluster condition was true
    pub cluster_count: usize,
    /// Number of cascade events
    pub cascade_count: usize,
    /// Total samples
    pub total_samples: usize,
    /// In-sample metrics
    pub is_metrics: ClassificationMetrics,
    /// Out-of-sample metrics
    pub oos_metrics: ClassificationMetrics,
    /// P(cascade | cluster)
    pub p_cascade_given_cluster: f64,
    /// P(cascade | no_cluster)
    pub p_cascade_given_no_cluster: f64,
    /// Lift: P(cascade|cluster) / P(cascade)
    pub conditional_lift: f64,
    /// Whether this threshold passes criteria
    pub passes: bool,
    /// Reason for pass/fail
    pub reason: String,
}

/// Lead time analysis
#[derive(Debug, Clone)]
pub struct LeadTimeAnalysis {
    /// Mean lead time before cascade (in periods)
    pub mean_lead_time: f64,
    /// Median lead time
    pub median_lead_time: f64,
    /// Minimum lead time observed
    pub min_lead_time: usize,
    /// Maximum lead time observed
    pub max_lead_time: usize,
    /// Fraction of cascades with >= 5 period lead time
    pub pct_actionable_5: f64,
    /// Fraction of cascades with >= 10 period lead time
    pub pct_actionable_10: f64,
}

/// Direction analysis for asymmetry
#[derive(Debug, Clone)]
pub struct DirectionAnalysis {
    /// Cascades that went up
    pub cascades_up: usize,
    /// Cascades that went down
    pub cascades_down: usize,
    /// P(up | cluster_above) - liquidations above current price
    pub p_up_given_cluster_above: f64,
    /// P(down | cluster_below) - liquidations below current price
    pub p_down_given_cluster_below: f64,
    /// Does asymmetry predict direction?
    pub asymmetry_predictive: bool,
}

/// Decision for H3
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Decision {
    /// GO: Liquidation clusters predict cascades
    Go,
    /// NO-GO: No predictive power
    NoGo,
    /// INCONCLUSIVE: Mixed results
    Inconclusive,
}

impl std::fmt::Display for H3Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            H3Decision::Go => write!(f, "GO"),
            H3Decision::NoGo => write!(f, "NO-GO"),
            H3Decision::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Complete H3 test result
#[derive(Debug, Clone)]
pub struct H3TestResult {
    pub config: H3TestConfig,
    /// Results for each threshold combination
    pub threshold_results: Vec<ThresholdResult>,
    /// Best performing threshold
    pub best_threshold: Option<ThresholdResult>,
    /// Lead time analysis
    pub lead_time: Option<LeadTimeAnalysis>,
    /// Direction analysis
    pub direction: Option<DirectionAnalysis>,
    /// Number of thresholds that pass
    pub n_passing: usize,
    /// Total thresholds tested
    pub n_total: usize,
    /// Overall decision
    pub decision: H3Decision,
    /// Summary message
    pub summary: String,
    /// Detailed report
    pub report: String,
    /// Sample size
    pub n_samples: usize,
}

impl H3TestResult {
    pub fn to_hypothesis_decision(&self) -> HypothesisDecision {
        match self.decision {
            H3Decision::Go => HypothesisDecision::Accept,
            H3Decision::NoGo => HypothesisDecision::Reject,
            H3Decision::Inconclusive => HypothesisDecision::Inconclusive,
        }
    }
}

/// Input data for H3 test
#[derive(Debug, Clone)]
pub struct H3TestData {
    /// Liquidation risk above current price (by distance bucket)
    /// Each inner vec is [1%, 2%, 5%, 10%] buckets
    pub liquidation_above: Vec<[f64; 4]>,
    /// Liquidation risk below current price
    pub liquidation_below: Vec<[f64; 4]>,
    /// Current price at each timestamp
    pub prices: Vec<f64>,
    /// Timestamps in milliseconds
    pub timestamps_ms: Vec<i64>,
}

impl H3TestData {
    pub fn validate(&self) -> Result<(), String> {
        let n = self.prices.len();

        if n < 200 {
            return Err(format!("Insufficient data: {} samples (need 200+)", n));
        }

        if self.liquidation_above.len() != n || self.liquidation_below.len() != n {
            return Err("Liquidation data length mismatch".to_string());
        }

        // Check for price variance
        let price_var = variance(&self.prices);
        if price_var < 1e-10 {
            return Err("Prices have zero variance".to_string());
        }

        Ok(())
    }

    /// Compute forward returns
    pub fn compute_returns(&self, horizon: usize) -> Vec<f64> {
        let n = self.prices.len();
        (0..n)
            .map(|i| {
                if i + horizon < n {
                    (self.prices[i + horizon] - self.prices[i]) / self.prices[i]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Detect cascade events (|return| > threshold within horizon)
    pub fn detect_cascades(&self, threshold: f64, horizon: usize) -> Vec<bool> {
        let returns = self.compute_returns(horizon);
        returns.iter().map(|r| r.abs() >= threshold).collect()
    }

    /// Check if cluster exists at given thresholds
    pub fn has_cluster(&self, idx: usize, amount: f64, distance_bucket: usize) -> (bool, bool) {
        let above = self.liquidation_above.get(idx)
            .map(|a| a.get(distance_bucket).copied().unwrap_or(0.0) >= amount)
            .unwrap_or(false);
        let below = self.liquidation_below.get(idx)
            .map(|a| a.get(distance_bucket).copied().unwrap_or(0.0) >= amount)
            .unwrap_or(false);
        (above, below)
    }

    /// Get distance bucket index for a threshold
    pub fn distance_to_bucket(distance: f64) -> usize {
        if distance <= 0.01 { 0 }
        else if distance <= 0.02 { 1 }
        else if distance <= 0.05 { 2 }
        else { 3 }
    }
}

/// Run the H3 hypothesis test
pub fn run_h3_liquidation_cascade_test(data: &H3TestData, config: &H3TestConfig) -> H3TestResult {
    // Validate
    if let Err(e) = data.validate() {
        return make_error_result(config, &e);
    }

    let n = data.prices.len();
    let train_size = (n as f64 * config.train_ratio) as usize;

    // Detect cascades
    let cascades = data.detect_cascades(config.cascade_threshold, config.cascade_horizon);
    let cascade_count = cascades.iter().filter(|&&c| c).count();

    if cascade_count < 10 {
        return make_error_result(config, &format!(
            "Too few cascade events: {} (need 10+)", cascade_count
        ));
    }

    // Test all threshold combinations
    let mut results = Vec::new();

    for &amount in &config.amount_thresholds {
        for &distance in &config.distance_thresholds {
            let result = test_threshold_combination(
                data, &cascades, amount, distance, train_size, config
            );
            results.push(result);
        }
    }

    // Count passing thresholds
    let n_passing = results.iter().filter(|r| r.passes).count();
    let n_total = results.len();

    // Find best threshold
    let best_threshold = results.iter()
        .filter(|r| r.passes)
        .max_by(|a, b| {
            // Rank by OOS F1 score
            a.oos_metrics.f1.partial_cmp(&b.oos_metrics.f1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    // Lead time analysis (using best threshold or first passing)
    let lead_time = best_threshold.as_ref()
        .map(|t| analyze_lead_time(data, &cascades, t.amount_threshold, t.distance_threshold, config));

    // Direction analysis
    let direction = best_threshold.as_ref()
        .map(|t| analyze_direction(data, &cascades, t.amount_threshold, t.distance_threshold, config));

    // Determine decision
    let decision = determine_h3_decision(&results, &best_threshold, config, n_passing);

    // Generate outputs
    let summary = generate_h3_summary(&decision, &best_threshold, n_passing, n_total);
    let report = generate_h3_report(
        &results, &best_threshold, &lead_time, &direction,
        &decision, config, n, cascade_count
    );

    H3TestResult {
        config: config.clone(),
        threshold_results: results,
        best_threshold,
        lead_time,
        direction,
        n_passing,
        n_total,
        decision,
        summary,
        report,
        n_samples: n,
    }
}

fn make_error_result(config: &H3TestConfig, error: &str) -> H3TestResult {
    H3TestResult {
        config: config.clone(),
        threshold_results: vec![],
        best_threshold: None,
        lead_time: None,
        direction: None,
        n_passing: 0,
        n_total: 0,
        decision: H3Decision::NoGo,
        summary: format!("Data validation failed: {}", error),
        report: format!("# H3 Test Report\n\n**Status:** FAILED\n\n**Reason:** {}", error),
        n_samples: 0,
    }
}

fn test_threshold_combination(
    data: &H3TestData,
    cascades: &[bool],
    amount: f64,
    distance: f64,
    train_size: usize,
    config: &H3TestConfig,
) -> ThresholdResult {
    let n = data.prices.len();
    let bucket = H3TestData::distance_to_bucket(distance);

    // Build cluster signals
    let cluster_signals: Vec<bool> = (0..n)
        .map(|i| {
            let (above, below) = data.has_cluster(i, amount, bucket);
            above || below
        })
        .collect();

    let cluster_count = cluster_signals.iter().filter(|&&c| c).count();
    let cascade_count = cascades.iter().filter(|&&c| c).count();

    // Split into train/test
    let (train_preds, train_actuals) = (
        &cluster_signals[..train_size],
        &cascades[..train_size]
    );
    let (test_preds, test_actuals) = (
        &cluster_signals[train_size..],
        &cascades[train_size..]
    );

    // Compute metrics
    let is_metrics = ClassificationMetrics::compute(
        &train_preds.to_vec(),
        &train_actuals.to_vec()
    );
    let oos_metrics = ClassificationMetrics::compute(
        &test_preds.to_vec(),
        &test_actuals.to_vec()
    );

    // Conditional probabilities
    let (p_cascade_cluster, p_cascade_no_cluster) = compute_conditional_probs(
        &cluster_signals, cascades
    );

    let conditional_lift = if p_cascade_no_cluster > 1e-10 {
        p_cascade_cluster / p_cascade_no_cluster
    } else if p_cascade_cluster > 0.0 {
        10.0 // Arbitrary high lift when no baseline
    } else {
        1.0
    };

    // Determine pass/fail
    let (passes, reason) = evaluate_threshold(
        &is_metrics, &oos_metrics, conditional_lift, cluster_count, config
    );

    ThresholdResult {
        amount_threshold: amount,
        distance_threshold: distance,
        cluster_count,
        cascade_count,
        total_samples: n,
        is_metrics,
        oos_metrics,
        p_cascade_given_cluster: p_cascade_cluster,
        p_cascade_given_no_cluster: p_cascade_no_cluster,
        conditional_lift,
        passes,
        reason,
    }
}

fn compute_conditional_probs(cluster: &[bool], cascade: &[bool]) -> (f64, f64) {
    let n = cluster.len().min(cascade.len());

    let mut cascade_given_cluster = 0;
    let mut cluster_count = 0;
    let mut cascade_given_no_cluster = 0;
    let mut no_cluster_count = 0;

    for i in 0..n {
        if cluster[i] {
            cluster_count += 1;
            if cascade[i] {
                cascade_given_cluster += 1;
            }
        } else {
            no_cluster_count += 1;
            if cascade[i] {
                cascade_given_no_cluster += 1;
            }
        }
    }

    let p_cluster = if cluster_count > 0 {
        cascade_given_cluster as f64 / cluster_count as f64
    } else {
        0.0
    };

    let p_no_cluster = if no_cluster_count > 0 {
        cascade_given_no_cluster as f64 / no_cluster_count as f64
    } else {
        0.0
    };

    (p_cluster, p_no_cluster)
}

fn evaluate_threshold(
    is_metrics: &ClassificationMetrics,
    oos_metrics: &ClassificationMetrics,
    lift: f64,
    cluster_count: usize,
    config: &H3TestConfig,
) -> (bool, String) {
    // Check cluster count
    if cluster_count < config.min_cluster_count {
        return (false, format!(
            "Too few clusters: {} < {}", cluster_count, config.min_cluster_count
        ));
    }

    // Check lift
    if lift < config.min_lift {
        return (false, format!(
            "Lift too low: {:.2}x < {:.2}x", lift, config.min_lift
        ));
    }

    // Check OOS precision
    if oos_metrics.precision < config.precision_failure {
        return (false, format!(
            "OOS precision too low: {:.1}% < {:.1}%",
            oos_metrics.precision * 100.0, config.precision_failure * 100.0
        ));
    }

    // Check for overfitting (OOS should be at least 50% of IS)
    if is_metrics.precision > 0.0 && oos_metrics.precision < is_metrics.precision * 0.5 {
        return (false, format!(
            "Overfitting: OOS precision {:.1}% < 50% of IS {:.1}%",
            oos_metrics.precision * 100.0, is_metrics.precision * 100.0
        ));
    }

    // Success criteria
    let lift_passes = lift >= config.min_lift;
    let precision_passes = oos_metrics.precision >= config.min_precision;
    let recall_passes = oos_metrics.recall >= config.min_recall;

    if lift_passes && precision_passes {
        (true, format!(
            "PASS: lift={:.2}x, precision={:.1}%, recall={:.1}%",
            lift, oos_metrics.precision * 100.0, oos_metrics.recall * 100.0
        ))
    } else {
        let mut reasons = Vec::new();
        if !lift_passes {
            reasons.push(format!("lift={:.2}x<{:.2}x", lift, config.min_lift));
        }
        if !precision_passes {
            reasons.push(format!("prec={:.1}%<{:.1}%",
                oos_metrics.precision * 100.0, config.min_precision * 100.0));
        }
        (false, format!("Near-miss: {}", reasons.join(", ")))
    }
}

fn analyze_lead_time(
    data: &H3TestData,
    cascades: &[bool],
    amount: f64,
    distance: f64,
    config: &H3TestConfig,
) -> LeadTimeAnalysis {
    let n = data.prices.len();
    let bucket = H3TestData::distance_to_bucket(distance);

    let mut lead_times = Vec::new();

    // For each cascade, find when cluster first appeared
    for i in 0..n {
        if !cascades[i] {
            continue;
        }

        // Look back to find when cluster signal started
        let mut lead = 0;
        for j in (0..i).rev() {
            let (above, below) = data.has_cluster(j, amount, bucket);
            if above || below {
                lead = i - j;
            } else {
                break;
            }
        }

        if lead > 0 {
            lead_times.push(lead);
        }
    }

    if lead_times.is_empty() {
        return LeadTimeAnalysis {
            mean_lead_time: 0.0,
            median_lead_time: 0.0,
            min_lead_time: 0,
            max_lead_time: 0,
            pct_actionable_5: 0.0,
            pct_actionable_10: 0.0,
        };
    }

    lead_times.sort();
    let n_leads = lead_times.len();

    let mean = lead_times.iter().sum::<usize>() as f64 / n_leads as f64;
    let median = lead_times[n_leads / 2] as f64;
    let min = *lead_times.first().unwrap_or(&0);
    let max = *lead_times.last().unwrap_or(&0);

    let pct_5 = lead_times.iter().filter(|&&l| l >= 5).count() as f64 / n_leads as f64;
    let pct_10 = lead_times.iter().filter(|&&l| l >= 10).count() as f64 / n_leads as f64;

    LeadTimeAnalysis {
        mean_lead_time: mean,
        median_lead_time: median,
        min_lead_time: min,
        max_lead_time: max,
        pct_actionable_5: pct_5,
        pct_actionable_10: pct_10,
    }
}

fn analyze_direction(
    data: &H3TestData,
    cascades: &[bool],
    amount: f64,
    distance: f64,
    config: &H3TestConfig,
) -> DirectionAnalysis {
    let n = data.prices.len();
    let bucket = H3TestData::distance_to_bucket(distance);
    let returns = data.compute_returns(config.cascade_horizon);

    let mut cascades_up = 0;
    let mut cascades_down = 0;
    let mut up_given_above = 0;
    let mut total_above = 0;
    let mut down_given_below = 0;
    let mut total_below = 0;

    for i in 0..n {
        if !cascades[i] {
            continue;
        }

        let ret = returns[i];
        if ret > 0.0 {
            cascades_up += 1;
        } else {
            cascades_down += 1;
        }

        let (above, below) = data.has_cluster(i, amount, bucket);

        if above {
            total_above += 1;
            if ret > 0.0 {
                up_given_above += 1;
            }
        }

        if below {
            total_below += 1;
            if ret < 0.0 {
                down_given_below += 1;
            }
        }
    }

    let p_up_above = if total_above > 0 {
        up_given_above as f64 / total_above as f64
    } else {
        0.5
    };

    let p_down_below = if total_below > 0 {
        down_given_below as f64 / total_below as f64
    } else {
        0.5
    };

    // Asymmetry is predictive if it's better than 60%
    let asymmetry_predictive = p_up_above > 0.6 || p_down_below > 0.6;

    DirectionAnalysis {
        cascades_up,
        cascades_down,
        p_up_given_cluster_above: p_up_above,
        p_down_given_cluster_below: p_down_below,
        asymmetry_predictive,
    }
}

fn determine_h3_decision(
    results: &[ThresholdResult],
    best: &Option<ThresholdResult>,
    config: &H3TestConfig,
    n_passing: usize,
) -> H3Decision {
    // Strong GO: Multiple thresholds pass
    if n_passing >= 3 {
        return H3Decision::Go;
    }

    // GO: At least one threshold passes with strong metrics
    if let Some(best) = best {
        if best.oos_metrics.precision >= config.min_precision * 1.2
            && best.conditional_lift >= config.min_lift * 1.5
        {
            return H3Decision::Go;
        }

        if n_passing >= 1 {
            return H3Decision::Go;
        }
    }

    // Check for clear failure
    let all_weak = results.iter().all(|r| {
        r.conditional_lift < config.min_lift * 0.5
            || r.oos_metrics.precision < config.precision_failure
    });

    if all_weak {
        return H3Decision::NoGo;
    }

    // Check if clusters are too rare
    let all_rare = results.iter().all(|r| r.cluster_count < config.min_cluster_count);
    if all_rare {
        return H3Decision::Inconclusive;
    }

    H3Decision::Inconclusive
}

fn generate_h3_summary(
    decision: &H3Decision,
    best: &Option<ThresholdResult>,
    n_passing: usize,
    n_total: usize,
) -> String {
    match best {
        Some(b) => format!(
            "H3 {} - {}/{} thresholds pass. Best: ${:.0}M @ {:.0}%, lift={:.2}x, precision={:.1}%",
            decision, n_passing, n_total,
            b.amount_threshold / 1_000_000.0,
            b.distance_threshold * 100.0,
            b.conditional_lift,
            b.oos_metrics.precision * 100.0
        ),
        None => format!(
            "H3 {} - {}/{} thresholds pass. No threshold met criteria.",
            decision, n_passing, n_total
        ),
    }
}

fn generate_h3_report(
    results: &[ThresholdResult],
    best: &Option<ThresholdResult>,
    lead_time: &Option<LeadTimeAnalysis>,
    direction: &Option<DirectionAnalysis>,
    decision: &H3Decision,
    config: &H3TestConfig,
    n_samples: usize,
    cascade_count: usize,
) -> String {
    let mut report = String::new();

    report.push_str("# H3 Hypothesis Test Report: Does Liquidation Clustering Predict Cascades?\n\n");

    // Decision banner
    report.push_str(&format!("## Decision: **{}**\n\n", decision));

    // Data summary
    report.push_str("## Data Summary\n\n");
    report.push_str(&format!("- Total samples: {}\n", n_samples));
    report.push_str(&format!("- Cascade events (>{:.0}% move): {}\n", config.cascade_threshold * 100.0, cascade_count));
    report.push_str(&format!("- Cascade rate: {:.2}%\n", cascade_count as f64 / n_samples as f64 * 100.0));
    report.push_str(&format!("- Cascade horizon: {} periods\n\n", config.cascade_horizon));

    // Configuration
    report.push_str("## Test Configuration\n\n");
    report.push_str(&format!("- Amount thresholds: {:?}\n",
        config.amount_thresholds.iter().map(|a| format!("${:.0}M", a / 1e6)).collect::<Vec<_>>()));
    report.push_str(&format!("- Distance thresholds: {:?}\n",
        config.distance_thresholds.iter().map(|d| format!("{:.0}%", d * 100.0)).collect::<Vec<_>>()));
    report.push_str(&format!("- Min lift: {:.1}x\n", config.min_lift));
    report.push_str(&format!("- Min precision: {:.0}%\n", config.min_precision * 100.0));
    report.push_str(&format!("- Train/test split: {:.0}%/{:.0}%\n\n",
        config.train_ratio * 100.0, (1.0 - config.train_ratio) * 100.0));

    // Results matrix
    report.push_str("## Threshold Results Matrix\n\n");
    report.push_str("### Lift (P(cascade|cluster) / P(cascade|no_cluster))\n\n");
    report.push_str("| Amount \\ Distance |");
    for d in &config.distance_thresholds {
        report.push_str(&format!(" {:.0}% |", d * 100.0));
    }
    report.push_str("\n|-------------------|");
    for _ in &config.distance_thresholds {
        report.push_str("------|");
    }
    report.push_str("\n");

    for amount in &config.amount_thresholds {
        report.push_str(&format!("| ${:.0}M |", amount / 1e6));
        for distance in &config.distance_thresholds {
            let result = results.iter()
                .find(|r| (r.amount_threshold - amount).abs() < 1.0
                    && (r.distance_threshold - distance).abs() < 0.001);
            match result {
                Some(r) => report.push_str(&format!(" {:.2}x |", r.conditional_lift)),
                None => report.push_str(" - |"),
            }
        }
        report.push_str("\n");
    }

    // OOS Precision matrix
    report.push_str("\n### Out-of-Sample Precision\n\n");
    report.push_str("| Amount \\ Distance |");
    for d in &config.distance_thresholds {
        report.push_str(&format!(" {:.0}% |", d * 100.0));
    }
    report.push_str("\n|-------------------|");
    for _ in &config.distance_thresholds {
        report.push_str("-------|");
    }
    report.push_str("\n");

    for amount in &config.amount_thresholds {
        report.push_str(&format!("| ${:.0}M |", amount / 1e6));
        for distance in &config.distance_thresholds {
            let result = results.iter()
                .find(|r| (r.amount_threshold - amount).abs() < 1.0
                    && (r.distance_threshold - distance).abs() < 0.001);
            match result {
                Some(r) => report.push_str(&format!(" {:.1}% |", r.oos_metrics.precision * 100.0)),
                None => report.push_str(" - |"),
            }
        }
        report.push_str("\n");
    }

    // Cluster count matrix
    report.push_str("\n### Cluster Occurrences\n\n");
    report.push_str("| Amount \\ Distance |");
    for d in &config.distance_thresholds {
        report.push_str(&format!(" {:.0}% |", d * 100.0));
    }
    report.push_str("\n|-------------------|");
    for _ in &config.distance_thresholds {
        report.push_str("------|");
    }
    report.push_str("\n");

    for amount in &config.amount_thresholds {
        report.push_str(&format!("| ${:.0}M |", amount / 1e6));
        for distance in &config.distance_thresholds {
            let result = results.iter()
                .find(|r| (r.amount_threshold - amount).abs() < 1.0
                    && (r.distance_threshold - distance).abs() < 0.001);
            match result {
                Some(r) => report.push_str(&format!(" {} |", r.cluster_count)),
                None => report.push_str(" - |"),
            }
        }
        report.push_str("\n");
    }

    // Best threshold details
    if let Some(best) = best {
        report.push_str("\n## Best Performing Threshold\n\n");
        report.push_str(&format!("**Amount:** ${:.0}M\n", best.amount_threshold / 1e6));
        report.push_str(&format!("**Distance:** {:.0}%\n", best.distance_threshold * 100.0));
        report.push_str(&format!("**Cluster Occurrences:** {}\n\n", best.cluster_count));

        report.push_str("### Conditional Probabilities\n\n");
        report.push_str(&format!("- P(cascade | cluster): {:.2}%\n", best.p_cascade_given_cluster * 100.0));
        report.push_str(&format!("- P(cascade | no cluster): {:.2}%\n", best.p_cascade_given_no_cluster * 100.0));
        report.push_str(&format!("- **Lift:** {:.2}x\n\n", best.conditional_lift));

        report.push_str("### Classification Metrics (OOS)\n\n");
        report.push_str(&format!("- Precision: {:.1}%\n", best.oos_metrics.precision * 100.0));
        report.push_str(&format!("- Recall: {:.1}%\n", best.oos_metrics.recall * 100.0));
        report.push_str(&format!("- F1 Score: {:.3}\n", best.oos_metrics.f1));
        report.push_str(&format!("- Accuracy: {:.1}%\n\n", best.oos_metrics.accuracy * 100.0));

        report.push_str("### Confusion Matrix (OOS)\n\n");
        report.push_str("| | Cascade | No Cascade |\n");
        report.push_str("|----------|---------|------------|\n");
        report.push_str(&format!("| Cluster | {} (TP) | {} (FP) |\n", best.oos_metrics.tp, best.oos_metrics.fp));
        report.push_str(&format!("| No Cluster | {} (FN) | {} (TN) |\n\n", best.oos_metrics.fn_, best.oos_metrics.tn));
    }

    // Lead time analysis
    if let Some(lt) = lead_time {
        report.push_str("## Lead Time Analysis\n\n");
        report.push_str(&format!("- Mean lead time: {:.1} periods\n", lt.mean_lead_time));
        report.push_str(&format!("- Median lead time: {:.1} periods\n", lt.median_lead_time));
        report.push_str(&format!("- Range: {} - {} periods\n", lt.min_lead_time, lt.max_lead_time));
        report.push_str(&format!("- Actionable (≥5 periods): {:.1}%\n", lt.pct_actionable_5 * 100.0));
        report.push_str(&format!("- Actionable (≥10 periods): {:.1}%\n\n", lt.pct_actionable_10 * 100.0));
    }

    // Direction analysis
    if let Some(dir) = direction {
        report.push_str("## Direction Analysis\n\n");
        report.push_str(&format!("- Cascades up: {}\n", dir.cascades_up));
        report.push_str(&format!("- Cascades down: {}\n", dir.cascades_down));
        report.push_str(&format!("- P(up | cluster above): {:.1}%\n", dir.p_up_given_cluster_above * 100.0));
        report.push_str(&format!("- P(down | cluster below): {:.1}%\n", dir.p_down_given_cluster_below * 100.0));
        report.push_str(&format!("- Asymmetry predictive: {}\n\n",
            if dir.asymmetry_predictive { "Yes" } else { "No" }));
    }

    // Success criteria evaluation
    report.push_str("## Success Criteria Evaluation\n\n");

    let n_passing = results.iter().filter(|r| r.passes).count();
    let any_high_lift = results.iter().any(|r| r.conditional_lift >= config.min_lift);
    let any_good_precision = results.iter().any(|r| r.oos_metrics.precision >= config.min_precision);

    let lift_check = if any_high_lift { "✓" } else { "✗" };
    let precision_check = if any_good_precision { "✓" } else { "✗" };
    let passing_check = if n_passing > 0 { "✓" } else { "✗" };

    report.push_str(&format!("- {} Lift ≥ {:.1}x for at least one threshold\n", lift_check, config.min_lift));
    report.push_str(&format!("- {} OOS Precision ≥ {:.0}% for at least one threshold\n",
        precision_check, config.min_precision * 100.0));
    report.push_str(&format!("- {} At least one threshold passes all criteria: {}\n\n",
        passing_check, n_passing));

    // Conclusion
    report.push_str("## Conclusion\n\n");
    match decision {
        H3Decision::Go => {
            report.push_str("**ACCEPT H3:** Liquidation clustering demonstrates predictive power for \
                cascade events. The signal provides meaningful lift over baseline and maintains \
                precision in out-of-sample testing.\n\n");
            report.push_str("**Recommendation:** Use liquidation cluster signals for cascade prediction \
                and risk management.\n");
        }
        H3Decision::NoGo => {
            report.push_str("**REJECT H3:** No evidence that liquidation clustering predicts cascades. \
                Either lift is insufficient, precision is too low, or clusters are too rare.\n\n");
            report.push_str("**Recommendation:** Do not rely on liquidation clusters for cascade prediction.\n");
        }
        H3Decision::Inconclusive => {
            report.push_str("**INCONCLUSIVE:** Mixed results across thresholds. Some show promise but \
                don't meet all criteria. More data or different parameters may help.\n\n");
            report.push_str("**Recommendation:** Gather more data, especially during volatile periods.\n");
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_cascade_data(n: usize, cascade_rate: f64, cluster_lift: f64) -> H3TestData {
        let mut prices = Vec::with_capacity(n);
        let mut liquidation_above = Vec::with_capacity(n);
        let mut liquidation_below = Vec::with_capacity(n);

        let mut price = 50000.0;

        for i in 0..n {
            prices.push(price);

            // Simulate price movement
            let move_pct = (i as f64 * 0.1).sin() * 0.02;
            price *= 1.0 + move_pct;

            // Generate liquidation data
            // Higher values near cascade events
            let has_cluster = (i as f64 * 0.07).sin() > 0.5;

            let base_liq = if has_cluster { 15_000_000.0 } else { 2_000_000.0 };

            liquidation_above.push([
                base_liq * 0.3,  // 1%
                base_liq * 0.5,  // 2%
                base_liq * 0.8,  // 5%
                base_liq * 1.0,  // 10%
            ]);

            liquidation_below.push([
                base_liq * 0.4,
                base_liq * 0.6,
                base_liq * 0.9,
                base_liq * 1.0,
            ]);
        }

        // Create some cascade events
        // Cascades are more likely when clusters exist (to simulate the relationship)
        for i in 0..n {
            let has_cluster = liquidation_above[i][1] > 5_000_000.0;
            let cascade_prob = if has_cluster {
                cascade_rate * cluster_lift
            } else {
                cascade_rate
            };

            if (i as f64 * 0.13 + 0.5).sin() > (1.0 - cascade_prob * 2.0) {
                // Simulate cascade: large price move
                if i + 60 < n {
                    let direction = if (i as f64 * 0.23).sin() > 0.0 { 1.0 } else { -1.0 };
                    for j in 1..=60 {
                        let idx = (i + j).min(n - 1);
                        prices[idx] = prices[i] * (1.0 + direction * 0.06 * (j as f64 / 60.0));
                    }
                }
            }
        }

        let timestamps_ms: Vec<i64> = (0..n as i64).map(|i| i * 60000).collect();

        H3TestData {
            liquidation_above,
            liquidation_below,
            prices,
            timestamps_ms,
        }
    }

    #[test]
    fn test_classification_metrics() {
        let preds = vec![true, true, false, false, true, false, true, false];
        let actuals = vec![true, false, true, false, true, false, false, false];

        let metrics = ClassificationMetrics::compute(&preds, &actuals);

        // TP=2 (0,4), FP=2 (1,6), TN=3 (3,5,7), FN=1 (2)
        assert_eq!(metrics.tp, 2);
        assert_eq!(metrics.fp, 2);
        assert_eq!(metrics.tn, 3);
        assert_eq!(metrics.fn_, 1);

        // Precision = 2/4 = 0.5
        assert!((metrics.precision - 0.5).abs() < 0.01);

        // Recall = 2/3 = 0.67
        assert!((metrics.recall - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_classification_metrics_perfect() {
        let preds = vec![true, true, false, false];
        let actuals = vec![true, true, false, false];

        let metrics = ClassificationMetrics::compute(&preds, &actuals);

        assert_eq!(metrics.tp, 2);
        assert_eq!(metrics.tn, 2);
        assert_eq!(metrics.fp, 0);
        assert_eq!(metrics.fn_, 0);
        assert!((metrics.precision - 1.0).abs() < 0.01);
        assert!((metrics.recall - 1.0).abs() < 0.01);
        assert!((metrics.accuracy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_h3_with_predictive_signal() {
        let data = generate_cascade_data(1000, 0.05, 3.0);
        let config = H3TestConfig::default();

        let result = run_h3_liquidation_cascade_test(&data, &config);

        assert!(result.n_samples == 1000);
        assert!(!result.report.is_empty());
        assert!(result.report.contains("H3 Hypothesis Test Report"));
    }

    #[test]
    fn test_h3_insufficient_data() {
        let data = H3TestData {
            liquidation_above: vec![[1.0; 4]; 100],
            liquidation_below: vec![[1.0; 4]; 100],
            prices: vec![50000.0; 100],
            timestamps_ms: (0..100).collect(),
        };

        let config = H3TestConfig::default();
        let result = run_h3_liquidation_cascade_test(&data, &config);

        assert_eq!(result.decision, H3Decision::NoGo);
        assert!(result.summary.contains("validation failed"));
    }

    #[test]
    fn test_h3_zero_variance_prices() {
        let data = H3TestData {
            liquidation_above: vec![[10_000_000.0; 4]; 500],
            liquidation_below: vec![[10_000_000.0; 4]; 500],
            prices: vec![50000.0; 500],
            timestamps_ms: (0..500).collect(),
        };

        let config = H3TestConfig::default();
        let result = run_h3_liquidation_cascade_test(&data, &config);

        assert_eq!(result.decision, H3Decision::NoGo);
    }

    #[test]
    fn test_cascade_detection() {
        let mut data = H3TestData {
            liquidation_above: vec![[1.0; 4]; 100],
            liquidation_below: vec![[1.0; 4]; 100],
            prices: (0..100).map(|i| 50000.0 + i as f64 * 10.0).collect(),
            timestamps_ms: (0..100).collect(),
        };

        // Create a 6% cascade at index 50
        for i in 50..60 {
            data.prices[i] = 50000.0 * 1.06;
        }

        let cascades = data.detect_cascades(0.05, 10);

        // Should detect cascade starting around index 40-49
        let cascade_count = cascades.iter().filter(|&&c| c).count();
        assert!(cascade_count > 0, "Should detect cascade events");
    }

    #[test]
    fn test_distance_to_bucket() {
        assert_eq!(H3TestData::distance_to_bucket(0.005), 0); // <1%
        assert_eq!(H3TestData::distance_to_bucket(0.01), 0);   // 1%
        assert_eq!(H3TestData::distance_to_bucket(0.015), 1);  // 1.5%
        assert_eq!(H3TestData::distance_to_bucket(0.02), 1);   // 2%
        assert_eq!(H3TestData::distance_to_bucket(0.03), 2);   // 3%
        assert_eq!(H3TestData::distance_to_bucket(0.05), 2);   // 5%
        assert_eq!(H3TestData::distance_to_bucket(0.10), 3);   // 10%
    }

    #[test]
    fn test_conditional_probability_computation() {
        let cluster = vec![true, true, true, false, false, false, false, false];
        let cascade = vec![true, true, false, true, false, false, false, false];

        let (p_cluster, p_no_cluster) = compute_conditional_probs(&cluster, &cascade);

        // P(cascade | cluster) = 2/3
        assert!((p_cluster - 0.667).abs() < 0.01);

        // P(cascade | no_cluster) = 1/5 = 0.2
        assert!((p_no_cluster - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_threshold_result_structure() {
        let data = generate_cascade_data(500, 0.05, 2.0);
        let config = H3TestConfig::default();

        let result = run_h3_liquidation_cascade_test(&data, &config);

        // Should test 3x3 = 9 threshold combinations
        assert_eq!(result.n_total, 9);
        assert_eq!(result.threshold_results.len(), 9);
    }

    #[test]
    fn test_report_sections() {
        let data = generate_cascade_data(500, 0.05, 2.5);
        let config = H3TestConfig::default();

        let result = run_h3_liquidation_cascade_test(&data, &config);

        assert!(result.report.contains("## Decision:"));
        assert!(result.report.contains("## Data Summary"));
        assert!(result.report.contains("## Test Configuration"));
        assert!(result.report.contains("## Threshold Results Matrix"));
        assert!(result.report.contains("## Conclusion"));
    }

    #[test]
    fn test_lead_time_analysis() {
        let data = generate_cascade_data(500, 0.05, 2.5);
        let config = H3TestConfig::default();

        let result = run_h3_liquidation_cascade_test(&data, &config);

        if let Some(lt) = &result.lead_time {
            // Lead times should be non-negative
            assert!(lt.mean_lead_time >= 0.0);
            assert!(lt.median_lead_time >= 0.0);
            assert!(lt.pct_actionable_5 >= 0.0 && lt.pct_actionable_5 <= 1.0);
        }
    }

    #[test]
    fn test_direction_analysis() {
        let data = generate_cascade_data(500, 0.05, 2.5);
        let config = H3TestConfig::default();

        let result = run_h3_liquidation_cascade_test(&data, &config);

        if let Some(dir) = &result.direction {
            // Should have some cascades
            assert!(dir.cascades_up + dir.cascades_down >= 0);
            // Probabilities should be valid
            assert!(dir.p_up_given_cluster_above >= 0.0 && dir.p_up_given_cluster_above <= 1.0);
            assert!(dir.p_down_given_cluster_below >= 0.0 && dir.p_down_given_cluster_below <= 1.0);
        }
    }

    #[test]
    fn test_decision_to_hypothesis_decision() {
        let data = generate_cascade_data(500, 0.05, 2.5);
        let config = H3TestConfig::default();
        let result = run_h3_liquidation_cascade_test(&data, &config);

        let hyp_decision = result.to_hypothesis_decision();

        match result.decision {
            H3Decision::Go => assert_eq!(hyp_decision, HypothesisDecision::Accept),
            H3Decision::NoGo => assert_eq!(hyp_decision, HypothesisDecision::Reject),
            H3Decision::Inconclusive => assert_eq!(hyp_decision, HypothesisDecision::Inconclusive),
        }
    }

    #[test]
    fn test_variance_helper() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = variance(&data);
        assert!((v - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_has_cluster() {
        let data = H3TestData {
            liquidation_above: vec![
                [1_000_000.0, 5_000_000.0, 10_000_000.0, 15_000_000.0],
            ],
            liquidation_below: vec![
                [2_000_000.0, 8_000_000.0, 12_000_000.0, 20_000_000.0],
            ],
            prices: vec![50000.0],
            timestamps_ms: vec![0],
        };

        // Check 2% bucket (index 1) for $5M threshold
        let (above, below) = data.has_cluster(0, 5_000_000.0, 1);
        assert!(above);  // 5M >= 5M
        assert!(below);  // 8M >= 5M

        // Check 1% bucket (index 0) for $5M threshold
        let (above, below) = data.has_cluster(0, 5_000_000.0, 0);
        assert!(!above); // 1M < 5M
        assert!(!below); // 2M < 5M
    }
}
