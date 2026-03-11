//! Final GO/PIVOT/NO-GO Decision Module
//!
//! This is the CAPSTONE module that aggregates all hypothesis test results
//! and feature analysis into a final project viability assessment.
//!
//! # Decision Matrix
//!
//! | Hypotheses Passed | Decision |
//! |-------------------|----------|
//! | 0-1 of 5          | NO-GO: Insufficient evidence of alpha |
//! | 2-3 of 5          | PIVOT: Focus only on validated features |
//! | 4-5 of 5          | GO: Full development of analytics layer |

use super::{
    H1TestResult, H2TestResult, H3TestResult, H4TestResult, H5TestResult,
    HypothesisDecision,
};
use super::feature_analysis::FeatureAnalysisResult;

/// Final project decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalDecision {
    /// Full development - 4-5 hypotheses pass
    Go,
    /// Focus on validated features only - 2-3 hypotheses pass
    Pivot,
    /// Insufficient evidence of alpha - 0-1 hypotheses pass
    NoGo,
}

impl std::fmt::Display for FinalDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinalDecision::Go => write!(f, "GO"),
            FinalDecision::Pivot => write!(f, "PIVOT"),
            FinalDecision::NoGo => write!(f, "NO-GO"),
        }
    }
}

/// Individual hypothesis result summary
#[derive(Debug, Clone)]
pub struct HypothesisSummary {
    /// Hypothesis ID (H1, H2, etc.)
    pub id: String,
    /// Hypothesis description
    pub description: String,
    /// Pass/Fail/Inconclusive
    pub decision: HypothesisDecision,
    /// Key metric value
    pub key_metric: f64,
    /// Key metric name
    pub key_metric_name: String,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Brief explanation
    pub explanation: String,
}

/// Confidence level for overall assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfidenceLevel::High => write!(f, "HIGH"),
            ConfidenceLevel::Medium => write!(f, "MEDIUM"),
            ConfidenceLevel::Low => write!(f, "LOW"),
        }
    }
}

/// Recommended next steps based on decision
#[derive(Debug, Clone)]
pub struct NextSteps {
    pub immediate: Vec<String>,
    pub short_term: Vec<String>,
    pub long_term: Vec<String>,
}

/// Strategy estimate
#[derive(Debug, Clone)]
pub struct StrategyEstimate {
    pub sharpe_conservative: f64,
    pub sharpe_optimistic: f64,
    pub alpha_decay_months: f64,
    pub capacity_usd: f64,
    pub risks: Vec<String>,
}

/// What worked and what didn't
#[derive(Debug, Clone)]
pub struct HonestAssessment {
    pub what_worked: Vec<String>,
    pub what_didnt_work: Vec<String>,
    pub wrong_assumptions: Vec<String>,
    pub enough_edge: bool,
    pub edge_explanation: String,
}

/// Full final decision result
#[derive(Debug, Clone)]
pub struct FinalDecisionResult {
    pub decision: FinalDecision,
    pub confidence: ConfidenceLevel,
    pub hypotheses_passed: usize,
    pub hypotheses_failed: usize,
    pub hypotheses_inconclusive: usize,
    pub hypothesis_summaries: Vec<HypothesisSummary>,
    pub recommended_features: Vec<String>,
    pub features_to_avoid: Vec<String>,
    pub strategy_estimate: StrategyEstimate,
    pub honest_assessment: HonestAssessment,
    pub next_steps: NextSteps,
}

/// Input for generating final decision
#[derive(Debug, Clone, Default)]
pub struct DecisionInput {
    pub h1: Option<H1TestResult>,
    pub h2: Option<H2TestResult>,
    pub h3: Option<H3TestResult>,
    pub h4: Option<H4TestResult>,
    pub h5: Option<H5TestResult>,
    pub feature_analysis: Option<FeatureAnalysisResult>,
}

impl DecisionInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_h1(mut self, result: H1TestResult) -> Self {
        self.h1 = Some(result);
        self
    }

    pub fn with_h2(mut self, result: H2TestResult) -> Self {
        self.h2 = Some(result);
        self
    }

    pub fn with_h3(mut self, result: H3TestResult) -> Self {
        self.h3 = Some(result);
        self
    }

    pub fn with_h4(mut self, result: H4TestResult) -> Self {
        self.h4 = Some(result);
        self
    }

    pub fn with_h5(mut self, result: H5TestResult) -> Self {
        self.h5 = Some(result);
        self
    }

    pub fn with_feature_analysis(mut self, result: FeatureAnalysisResult) -> Self {
        self.feature_analysis = Some(result);
        self
    }
}

/// Summarize H1 result
fn summarize_h1(result: &H1TestResult) -> HypothesisSummary {
    let decision = result.to_hypothesis_decision();

    // Get key metric from best combination
    let (key_metric, key_metric_name) = if let Some(ref best) = result.best_combination {
        let name = format!("{}/{}", best.flow_window, best.return_horizon);
        (best.correlation.pearson, format!("correlation ({})", name))
    } else {
        (0.0, "correlation".to_string())
    };

    let confidence = if result.n_passing >= 6 {
        0.9
    } else if result.n_passing >= 3 {
        0.6
    } else {
        0.3
    };

    let explanation = match decision {
        HypothesisDecision::Accept => format!(
            "{} of {} window/horizon combinations showed significant correlation",
            result.n_passing, result.n_total
        ),
        HypothesisDecision::Reject => format!(
            "Only {} of {} combinations passed; whale flow does not reliably predict returns",
            result.n_passing, result.n_total
        ),
        HypothesisDecision::Inconclusive => format!(
            "{} of {} passed - mixed evidence",
            result.n_passing, result.n_total
        ),
    };

    HypothesisSummary {
        id: "H1".to_string(),
        description: "Whale flow predicts returns".to_string(),
        decision,
        key_metric,
        key_metric_name,
        confidence,
        explanation,
    }
}

/// Summarize H2 result
fn summarize_h2(result: &H2TestResult) -> HypothesisSummary {
    let decision = result.to_hypothesis_decision();

    let key_metric = result.joint_lift;
    let key_metric_name = "joint lift".to_string();

    let confidence = if result.interaction_p_value < 0.001 && result.joint_vs_best_lift > 0.2 {
        0.9
    } else if result.interaction_p_value < 0.01 {
        0.6
    } else {
        0.3
    };

    let explanation = match decision {
        HypothesisDecision::Accept => format!(
            "Entropy + whale interaction shows {:.1}% lift (p={:.4})",
            result.joint_vs_best_lift * 100.0, result.interaction_p_value
        ),
        HypothesisDecision::Reject => format!(
            "Interaction lift of {:.1}% is below threshold; no synergy detected",
            result.joint_vs_best_lift * 100.0
        ),
        HypothesisDecision::Inconclusive => format!(
            "Lift={:.1}%, p={:.4} - evidence is mixed",
            result.joint_vs_best_lift * 100.0, result.interaction_p_value
        ),
    };

    HypothesisSummary {
        id: "H2".to_string(),
        description: "Entropy + whale interaction effect".to_string(),
        decision,
        key_metric,
        key_metric_name,
        confidence,
        explanation,
    }
}

/// Summarize H3 result
fn summarize_h3(result: &H3TestResult) -> HypothesisSummary {
    let decision = result.to_hypothesis_decision();

    let (key_metric, key_metric_name) = if let Some(ref best) = result.best_threshold {
        let name = format!("${:.0}M/{:.0}%", best.amount_threshold / 1_000_000.0, best.distance_threshold * 100.0);
        (best.oos_metrics.lift, format!("lift ({})", name))
    } else {
        (0.0, "lift".to_string())
    };

    let confidence = if result.n_passing >= 6 {
        0.9
    } else if result.n_passing >= 3 {
        0.6
    } else {
        0.3
    };

    let explanation = match decision {
        HypothesisDecision::Accept => format!(
            "{} threshold combinations show predictive power for liquidation cascades",
            result.n_passing
        ),
        HypothesisDecision::Reject => format!(
            "Liquidation cascade prediction unreliable; only {} of {} thresholds passed",
            result.n_passing, result.n_total
        ),
        HypothesisDecision::Inconclusive => "Mixed results across thresholds".to_string(),
    };

    HypothesisSummary {
        id: "H3".to_string(),
        description: "Liquidation cascades are predictable".to_string(),
        decision,
        key_metric,
        key_metric_name,
        confidence,
        explanation,
    }
}

/// Summarize H4 result
fn summarize_h4(result: &H4TestResult) -> HypothesisSummary {
    let decision = result.to_hypothesis_decision();

    let (key_metric, key_metric_name) = if let Some(ref best) = result.best_measure {
        let name = format!("{}_{}", best.measure_name, best.horizon);
        (best.correlation.pearson, format!("correlation ({})", name))
    } else {
        (0.0, "correlation".to_string())
    };

    let confidence = if result.n_passing >= 7 {
        0.9
    } else if result.n_passing >= 4 {
        0.6
    } else {
        0.3
    };

    let explanation = match decision {
        HypothesisDecision::Accept => format!(
            "Position concentration predicts volatility; {} measures show significant correlation",
            result.n_passing
        ),
        HypothesisDecision::Reject => format!(
            "Concentration-volatility relationship weak; only {} of {} measures passed",
            result.n_passing, result.n_total
        ),
        HypothesisDecision::Inconclusive => "Mixed evidence across concentration measures".to_string(),
    };

    HypothesisSummary {
        id: "H4".to_string(),
        description: "Concentration predicts volatility".to_string(),
        decision,
        key_metric,
        key_metric_name,
        confidence,
        explanation,
    }
}

/// Summarize H5 result
fn summarize_h5(result: &H5TestResult) -> HypothesisSummary {
    let decision = result.decision.to_hypothesis_decision();

    let key_metric = result.overall_wf_sharpe;
    let key_metric_name = "walk-forward Sharpe".to_string();

    let confidence = if result.horizons_passed >= 2 && result.overall_oos_is_ratio > 0.8 {
        0.9
    } else if result.horizons_passed >= 1 && result.overall_oos_is_ratio > 0.6 {
        0.6
    } else {
        0.3
    };

    let explanation = match decision {
        HypothesisDecision::Accept => format!(
            "Persistence indicator works! WF Sharpe={:.2}, OOS/IS={:.2}",
            result.overall_wf_sharpe, result.overall_oos_is_ratio
        ),
        HypothesisDecision::Reject => format!(
            "Persistence indicator fails; WF Sharpe={:.2} below threshold",
            result.overall_wf_sharpe
        ),
        HypothesisDecision::Inconclusive => format!(
            "Mixed results: {} horizons pass, {} fail",
            result.horizons_passed, result.horizon_results.len() - result.horizons_passed
        ),
    };

    HypothesisSummary {
        id: "H5".to_string(),
        description: "Persistence indicator works".to_string(),
        decision,
        key_metric,
        key_metric_name,
        confidence,
        explanation,
    }
}

/// Generate strategy estimate based on results
fn generate_strategy_estimate(
    summaries: &[HypothesisSummary],
    h5: Option<&H5TestResult>,
) -> StrategyEstimate {
    let passed_count = summaries.iter()
        .filter(|s| s.decision == HypothesisDecision::Accept)
        .count();

    let base_sharpe = h5.map(|h| h.overall_wf_sharpe).unwrap_or(0.0);
    let sharpe_conservative = (base_sharpe * 0.5).max(0.0);
    let sharpe_optimistic = (base_sharpe * 0.8).max(0.0);

    let alpha_decay_months = match passed_count {
        0..=1 => 1.0,
        2..=3 => 3.0,
        _ => 6.0,
    };

    let capacity_usd = match passed_count {
        0..=1 => 100_000.0,
        2..=3 => 500_000.0,
        _ => 2_000_000.0,
    };

    let mut risks = vec![
        "Market regime change could invalidate signals".to_string(),
        "Hyperliquid-specific dynamics may differ from other venues".to_string(),
        "Execution slippage in real trading".to_string(),
    ];

    if passed_count <= 2 {
        risks.push("Limited statistical evidence - high uncertainty".to_string());
    }
    if base_sharpe < 0.5 {
        risks.push("Low base Sharpe - may not be profitable after costs".to_string());
    }

    StrategyEstimate {
        sharpe_conservative,
        sharpe_optimistic,
        alpha_decay_months,
        capacity_usd,
        risks,
    }
}

/// Generate honest assessment
fn generate_honest_assessment(
    summaries: &[HypothesisSummary],
    feature_analysis: Option<&FeatureAnalysisResult>,
) -> HonestAssessment {
    let mut what_worked = Vec::new();
    let mut what_didnt_work = Vec::new();
    let mut wrong_assumptions = Vec::new();

    for summary in summaries {
        match summary.decision {
            HypothesisDecision::Accept => {
                what_worked.push(format!("{}: {}", summary.id, summary.description));
            }
            HypothesisDecision::Reject => {
                what_didnt_work.push(format!("{}: {}", summary.id, summary.description));
            }
            HypothesisDecision::Inconclusive => {}
        }
    }

    let h1_failed = summaries.iter().any(|s| s.id == "H1" && s.decision == HypothesisDecision::Reject);
    let h2_failed = summaries.iter().any(|s| s.id == "H2" && s.decision == HypothesisDecision::Reject);
    let h5_failed = summaries.iter().any(|s| s.id == "H5" && s.decision == HypothesisDecision::Reject);

    if h1_failed {
        wrong_assumptions.push("Whale flow predictive power may not exist for this market".to_string());
    }
    if h2_failed {
        wrong_assumptions.push("Entropy-whale interaction effect may be too weak".to_string());
    }
    if h5_failed {
        wrong_assumptions.push("Core persistence thesis may not apply to Hyperliquid".to_string());
    }

    if let Some(fa) = feature_analysis {
        if fa.summary.excluded_redundant > fa.summary.subset_size {
            wrong_assumptions.push("Many features are redundant - simpler model may be better".to_string());
        }
    }

    let passed_count = summaries.iter()
        .filter(|s| s.decision == HypothesisDecision::Accept)
        .count();

    let enough_edge = passed_count >= 2;
    let edge_explanation = if passed_count >= 4 {
        "Strong evidence of multiple alpha sources. Development should proceed.".to_string()
    } else if passed_count >= 2 {
        "Moderate evidence of alpha. Consider focusing on validated signals only.".to_string()
    } else {
        "Insufficient evidence of alpha. Project may not be viable.".to_string()
    };

    HonestAssessment {
        what_worked,
        what_didnt_work,
        wrong_assumptions,
        enough_edge,
        edge_explanation,
    }
}

/// Generate next steps based on decision
fn generate_next_steps(decision: FinalDecision, summaries: &[HypothesisSummary]) -> NextSteps {
    match decision {
        FinalDecision::Go => NextSteps {
            immediate: vec![
                "Begin backtesting infrastructure development".to_string(),
                "Set up paper trading environment".to_string(),
                "Implement feature pipeline for live data".to_string(),
            ],
            short_term: vec![
                "Run 2-week paper trading validation".to_string(),
                "Implement risk management module".to_string(),
                "Build monitoring dashboard".to_string(),
            ],
            long_term: vec![
                "Deploy with small capital allocation".to_string(),
                "Scale based on live performance".to_string(),
                "Research additional alpha sources".to_string(),
            ],
        },
        FinalDecision::Pivot => {
            let passed_ids: Vec<&str> = summaries.iter()
                .filter(|s| s.decision == HypothesisDecision::Accept)
                .map(|s| s.id.as_str())
                .collect();

            NextSteps {
                immediate: vec![
                    format!("Focus development on validated signals: {}", passed_ids.join(", ")),
                    "Remove/deprioritize failed hypothesis features".to_string(),
                    "Simplify model to reduce overfitting risk".to_string(),
                ],
                short_term: vec![
                    "Collect more data to increase statistical power".to_string(),
                    "Re-test failed hypotheses with refined methodology".to_string(),
                    "Consider alternative data sources".to_string(),
                ],
                long_term: vec![
                    "Paper trade focused strategy".to_string(),
                    "Re-evaluate full strategy when more data available".to_string(),
                    "Consider pivoting to different market/asset".to_string(),
                ],
            }
        }
        FinalDecision::NoGo => NextSteps {
            immediate: vec![
                "Document learnings from this research".to_string(),
                "Archive code for potential future use".to_string(),
                "Identify what additional data might help".to_string(),
            ],
            short_term: vec![
                "Explore alternative markets/strategies".to_string(),
                "Research why hypotheses failed".to_string(),
                "Consider if methodology was flawed".to_string(),
            ],
            long_term: vec![
                "Revisit if market conditions change".to_string(),
                "Apply learnings to other projects".to_string(),
                "Consider fundamental strategy changes".to_string(),
            ],
        },
    }
}

/// Run the final decision analysis
pub fn run_final_decision(input: &DecisionInput) -> FinalDecisionResult {
    let mut summaries = Vec::new();

    if let Some(ref h1) = input.h1 {
        summaries.push(summarize_h1(h1));
    }
    if let Some(ref h2) = input.h2 {
        summaries.push(summarize_h2(h2));
    }
    if let Some(ref h3) = input.h3 {
        summaries.push(summarize_h3(h3));
    }
    if let Some(ref h4) = input.h4 {
        summaries.push(summarize_h4(h4));
    }
    if let Some(ref h5) = input.h5 {
        summaries.push(summarize_h5(h5));
    }

    let hypotheses_passed = summaries.iter()
        .filter(|s| s.decision == HypothesisDecision::Accept)
        .count();
    let hypotheses_failed = summaries.iter()
        .filter(|s| s.decision == HypothesisDecision::Reject)
        .count();
    let hypotheses_inconclusive = summaries.iter()
        .filter(|s| s.decision == HypothesisDecision::Inconclusive)
        .count();

    let decision = match hypotheses_passed {
        0..=1 => FinalDecision::NoGo,
        2..=3 => FinalDecision::Pivot,
        _ => FinalDecision::Go,
    };

    let avg_confidence = if summaries.is_empty() {
        0.0
    } else {
        summaries.iter().map(|s| s.confidence).sum::<f64>() / summaries.len() as f64
    };

    let confidence = if avg_confidence > 0.7 && hypotheses_inconclusive == 0 {
        ConfidenceLevel::High
    } else if avg_confidence > 0.5 {
        ConfidenceLevel::Medium
    } else {
        ConfidenceLevel::Low
    };

    let (recommended_features, features_to_avoid) = if let Some(ref fa) = input.feature_analysis {
        (
            fa.recommended_subset.clone(),
            fa.excluded_redundant.iter()
                .chain(fa.excluded_low_mi.iter())
                .cloned()
                .collect(),
        )
    } else {
        (vec![], vec![])
    };

    let strategy_estimate = generate_strategy_estimate(&summaries, input.h5.as_ref());
    let honest_assessment = generate_honest_assessment(&summaries, input.feature_analysis.as_ref());
    let next_steps = generate_next_steps(decision, &summaries);

    FinalDecisionResult {
        decision,
        confidence,
        hypotheses_passed,
        hypotheses_failed,
        hypotheses_inconclusive,
        hypothesis_summaries: summaries,
        recommended_features,
        features_to_avoid,
        strategy_estimate,
        honest_assessment,
        next_steps,
    }
}

impl FinalDecisionResult {
    /// Generate comprehensive markdown report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# FINAL DECISION REPORT\n\n");
        report.push_str(&format!("## Decision: **{}**\n\n", self.decision));
        report.push_str(&format!("**Confidence Level:** {}\n\n", self.confidence));

        report.push_str("```\n");
        report.push_str("╔══════════════════════════════════════╗\n");
        report.push_str(&format!("║  HYPOTHESES PASSED: {}/{}             ║\n",
            self.hypotheses_passed,
            self.hypotheses_passed + self.hypotheses_failed + self.hypotheses_inconclusive
        ));
        report.push_str(&format!("║  DECISION: {:25} ║\n", self.decision.to_string()));
        report.push_str(&format!("║  CONFIDENCE: {:22} ║\n", self.confidence.to_string()));
        report.push_str("╚══════════════════════════════════════╝\n");
        report.push_str("```\n\n");

        report.push_str("## Hypothesis Test Results\n\n");
        report.push_str("| ID | Hypothesis | Decision | Key Metric | Confidence |\n");
        report.push_str("|----|------------|----------|------------|------------|\n");

        for summary in &self.hypothesis_summaries {
            let decision_str = match summary.decision {
                HypothesisDecision::Accept => "PASS",
                HypothesisDecision::Reject => "FAIL",
                HypothesisDecision::Inconclusive => "MIXED",
            };
            report.push_str(&format!(
                "| {} | {} | {} | {:.3} ({}) | {:.0}% |\n",
                summary.id,
                summary.description,
                decision_str,
                summary.key_metric,
                summary.key_metric_name,
                summary.confidence * 100.0,
            ));
        }

        report.push_str("\n### Detailed Results\n\n");
        for summary in &self.hypothesis_summaries {
            report.push_str(&format!("**{}**: {}\n\n", summary.id, summary.explanation));
        }

        report.push_str("## Strategy Estimate\n\n");
        report.push_str(&format!("- **Conservative Sharpe:** {:.2}\n", self.strategy_estimate.sharpe_conservative));
        report.push_str(&format!("- **Optimistic Sharpe:** {:.2}\n", self.strategy_estimate.sharpe_optimistic));
        report.push_str(&format!("- **Estimated Alpha Decay:** {:.0} months\n", self.strategy_estimate.alpha_decay_months));
        report.push_str(&format!("- **Capacity Estimate:** ${:.0}\n\n", self.strategy_estimate.capacity_usd));

        report.push_str("### Key Risks\n\n");
        for risk in &self.strategy_estimate.risks {
            report.push_str(&format!("- {}\n", risk));
        }

        if !self.recommended_features.is_empty() {
            report.push_str("\n## Recommended Feature Subset\n\n");
            for (i, feature) in self.recommended_features.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, feature));
            }
        }

        if !self.features_to_avoid.is_empty() {
            report.push_str("\n### Features to Avoid\n\n");
            for feature in &self.features_to_avoid {
                report.push_str(&format!("- {}\n", feature));
            }
        }

        report.push_str("\n## Honest Assessment\n\n");

        if !self.honest_assessment.what_worked.is_empty() {
            report.push_str("### What Worked\n\n");
            for item in &self.honest_assessment.what_worked {
                report.push_str(&format!("- {}\n", item));
            }
        }

        if !self.honest_assessment.what_didnt_work.is_empty() {
            report.push_str("\n### What Didn't Work\n\n");
            for item in &self.honest_assessment.what_didnt_work {
                report.push_str(&format!("- {}\n", item));
            }
        }

        if !self.honest_assessment.wrong_assumptions.is_empty() {
            report.push_str("\n### Wrong Assumptions\n\n");
            for item in &self.honest_assessment.wrong_assumptions {
                report.push_str(&format!("- {}\n", item));
            }
        }

        report.push_str("\n### Edge Assessment\n\n");
        report.push_str(&format!("**Enough Edge:** {}\n\n",
            if self.honest_assessment.enough_edge { "Yes" } else { "No" }
        ));
        report.push_str(&format!("{}\n", self.honest_assessment.edge_explanation));

        report.push_str("\n## Recommended Next Steps\n\n");

        report.push_str("### Immediate Actions\n\n");
        for step in &self.next_steps.immediate {
            report.push_str(&format!("- [ ] {}\n", step));
        }

        report.push_str("\n### Short-Term (1-2 weeks)\n\n");
        for step in &self.next_steps.short_term {
            report.push_str(&format!("- [ ] {}\n", step));
        }

        report.push_str("\n### Long-Term (1+ month)\n\n");
        for step in &self.next_steps.long_term {
            report.push_str(&format!("- [ ] {}\n", step));
        }

        report.push_str("\n---\n\n");
        report.push_str("*Report generated by Hyperliquid Analytics Layer*\n");
        report.push_str("*Decision framework: 0-1 pass=NO-GO, 2-3 pass=PIVOT, 4-5 pass=GO*\n");

        report
    }

    /// Get a one-line summary
    pub fn one_line_summary(&self) -> String {
        format!(
            "{} ({} confidence) - {}/{} hypotheses passed",
            self.decision,
            self.confidence,
            self.hypotheses_passed,
            self.hypotheses_passed + self.hypotheses_failed + self.hypotheses_inconclusive
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let input = DecisionInput::new();
        let result = run_final_decision(&input);

        assert_eq!(result.decision, FinalDecision::NoGo);
        assert_eq!(result.hypotheses_passed, 0);
        assert!(result.hypothesis_summaries.is_empty());
    }

    #[test]
    fn test_decision_display() {
        assert_eq!(format!("{}", FinalDecision::Go), "GO");
        assert_eq!(format!("{}", FinalDecision::Pivot), "PIVOT");
        assert_eq!(format!("{}", FinalDecision::NoGo), "NO-GO");
    }

    #[test]
    fn test_confidence_display() {
        assert_eq!(format!("{}", ConfidenceLevel::High), "HIGH");
        assert_eq!(format!("{}", ConfidenceLevel::Medium), "MEDIUM");
        assert_eq!(format!("{}", ConfidenceLevel::Low), "LOW");
    }

    #[test]
    fn test_decision_input_builder() {
        let input = DecisionInput::new();
        assert!(input.h1.is_none());
        assert!(input.h2.is_none());
        assert!(input.h3.is_none());
        assert!(input.h4.is_none());
        assert!(input.h5.is_none());
        assert!(input.feature_analysis.is_none());
    }

    #[test]
    fn test_strategy_estimate_risks() {
        let summaries: Vec<HypothesisSummary> = vec![];
        let estimate = generate_strategy_estimate(&summaries, None);

        assert!(!estimate.risks.is_empty());
        assert!(estimate.sharpe_conservative >= 0.0);
        assert!(estimate.sharpe_optimistic >= estimate.sharpe_conservative);
    }

    #[test]
    fn test_honest_assessment_empty() {
        let summaries: Vec<HypothesisSummary> = vec![];
        let assessment = generate_honest_assessment(&summaries, None);

        assert!(assessment.what_worked.is_empty());
        assert!(assessment.what_didnt_work.is_empty());
        assert!(!assessment.enough_edge);
    }

    #[test]
    fn test_next_steps_nogo() {
        let summaries: Vec<HypothesisSummary> = vec![];
        let steps = generate_next_steps(FinalDecision::NoGo, &summaries);

        assert!(!steps.immediate.is_empty());
        assert!(!steps.short_term.is_empty());
        assert!(!steps.long_term.is_empty());
    }

    #[test]
    fn test_next_steps_go() {
        let summaries: Vec<HypothesisSummary> = vec![];
        let steps = generate_next_steps(FinalDecision::Go, &summaries);

        assert!(steps.immediate.iter().any(|s| s.contains("backtesting")));
    }

    #[test]
    fn test_next_steps_pivot() {
        let summaries = vec![
            HypothesisSummary {
                id: "H1".to_string(),
                description: "Test".to_string(),
                decision: HypothesisDecision::Accept,
                key_metric: 0.5,
                key_metric_name: "metric".to_string(),
                confidence: 0.8,
                explanation: "Passed".to_string(),
            },
        ];
        let steps = generate_next_steps(FinalDecision::Pivot, &summaries);

        assert!(steps.immediate.iter().any(|s| s.contains("H1")));
    }

    #[test]
    fn test_report_generation_empty() {
        let input = DecisionInput::new();
        let result = run_final_decision(&input);
        let report = result.generate_report();

        assert!(report.contains("FINAL DECISION REPORT"));
        assert!(report.contains("NO-GO"));
    }

    #[test]
    fn test_one_line_summary() {
        let input = DecisionInput::new();
        let result = run_final_decision(&input);
        let summary = result.one_line_summary();

        assert!(summary.contains("NO-GO"));
        assert!(summary.contains("0/0"));
    }

    #[test]
    fn test_hypothesis_summary_structure() {
        let summary = HypothesisSummary {
            id: "H1".to_string(),
            description: "Test hypothesis".to_string(),
            decision: HypothesisDecision::Accept,
            key_metric: 0.15,
            key_metric_name: "correlation".to_string(),
            confidence: 0.85,
            explanation: "Strong correlation found".to_string(),
        };

        assert_eq!(summary.id, "H1");
        assert_eq!(summary.decision, HypothesisDecision::Accept);
        assert!(summary.confidence > 0.8);
    }
}
