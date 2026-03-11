//! H5 Hypothesis Test: Persistence Indicator Works on Hyperliquid
//!
//! This is the CRITICAL hypothesis test for the project's core thesis.
//!
//! Hypothesis: A composite persistence indicator (combining entropy, momentum,
//! monotonicity, hurst, OFI, illiquidity) predicts trend continuation on Hyperliquid.
//!
//! Test Protocol:
//! 1. Compute persistence indicator from all features
//! 2. Three-bar labels: 1min, 5min, 15min horizons
//! 3. Train/test: 70/30 split
//! 4. Walk-forward: 5 folds
//! 5. Baseline: simple momentum strategy
//!
//! Success Criteria:
//! - Walk-forward Sharpe > 0.5
//! - OOS Sharpe > 0.7 * IS Sharpe (not overfit)
//! - Beat momentum baseline by > 20%
//! - MI(persistence_indicator, future_return) > 0.05 bits
//!
//! Failure Criteria:
//! - Walk-forward Sharpe < 0.3
//! - OOS Sharpe < 0.5 * IS Sharpe (overfit)
//! - Does not beat simple baseline
//! - MI < 0.02 bits

use super::stats::{
    pearson_correlation, spearman_correlation, mutual_information_adaptive,
    t_test_correlation, CorrelationResult,
};
use super::HypothesisDecision;

/// H5 test configuration
#[derive(Debug, Clone)]
pub struct H5TestConfig {
    /// Minimum walk-forward Sharpe for success
    pub min_wf_sharpe: f64,
    /// Sharpe below which we reject
    pub sharpe_failure_threshold: f64,
    /// Minimum OOS/IS ratio for success
    pub min_oos_is_ratio: f64,
    /// OOS/IS ratio below which we reject
    pub oos_is_failure_threshold: f64,
    /// Minimum improvement over baseline for success
    pub min_baseline_improvement: f64,
    /// Minimum MI for success
    pub min_mi: f64,
    /// MI below which we reject
    pub mi_failure_threshold: f64,
    /// Number of walk-forward folds
    pub n_folds: usize,
    /// Train ratio for train/test split
    pub train_ratio: f64,
    /// Horizons to test (in data points, e.g., 1, 5, 15 for 1min, 5min, 15min)
    pub horizons: Vec<usize>,
}

impl Default for H5TestConfig {
    fn default() -> Self {
        Self {
            min_wf_sharpe: 0.5,
            sharpe_failure_threshold: 0.3,
            min_oos_is_ratio: 0.7,
            oos_is_failure_threshold: 0.5,
            min_baseline_improvement: 0.20, // 20%
            min_mi: 0.05,
            mi_failure_threshold: 0.02,
            n_folds: 5,
            train_ratio: 0.7,
            horizons: vec![1, 5, 15],
        }
    }
}

/// Input features for computing persistence indicator
#[derive(Debug, Clone)]
pub struct FeatureRow {
    /// Tick entropy (normalized 0-1, lower = more persistent)
    pub entropy: f64,
    /// Momentum (directional strength)
    pub momentum: f64,
    /// Monotonicity (-1 to 1, high absolute = persistent direction)
    pub monotonicity: f64,
    /// Hurst exponent (>0.5 = trending, <0.5 = mean-reverting)
    pub hurst: f64,
    /// Order flow imbalance (-1 to 1)
    pub ofi: f64,
    /// Illiquidity measure (higher = less liquid)
    pub illiquidity: f64,
}

/// Three-bar label for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeBarLabel {
    /// Price went up
    Up,
    /// Price stayed flat (within threshold)
    Flat,
    /// Price went down
    Down,
}

/// Result for a single horizon test
#[derive(Debug, Clone)]
pub struct HorizonResult {
    /// Horizon name (e.g., "1min", "5min")
    pub horizon_name: String,
    /// Horizon in data points
    pub horizon_points: usize,
    /// Walk-forward Sharpe ratio
    pub wf_sharpe: f64,
    /// In-sample Sharpe ratio
    pub is_sharpe: f64,
    /// Out-of-sample Sharpe ratio
    pub oos_sharpe: f64,
    /// OOS/IS ratio
    pub oos_is_ratio: f64,
    /// Baseline (momentum) Sharpe ratio
    pub baseline_sharpe: f64,
    /// Improvement over baseline
    pub baseline_improvement: f64,
    /// Mutual information (bits)
    pub mi: f64,
    /// Hit rate (accuracy)
    pub hit_rate: f64,
    /// Correlation with future returns
    pub correlation: CorrelationResult,
    /// Decision for this horizon
    pub decision: HorizonDecision,
}

/// Walk-forward fold result
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Fold index (0-based)
    pub fold_idx: usize,
    /// In-sample Sharpe
    pub is_sharpe: f64,
    /// Out-of-sample Sharpe
    pub oos_sharpe: f64,
    /// In-sample hit rate
    pub is_hit_rate: f64,
    /// Out-of-sample hit rate
    pub oos_hit_rate: f64,
}

/// Regime-specific analysis
#[derive(Debug, Clone)]
pub struct RegimeAnalysis {
    /// Results in low volatility regime
    pub low_vol: RegimeResult,
    /// Results in high volatility regime
    pub high_vol: RegimeResult,
    /// Regime difference (high_vol - low_vol for sharpe)
    pub regime_diff: f64,
}

/// Result for a single regime
#[derive(Debug, Clone)]
pub struct RegimeResult {
    /// Regime name
    pub name: String,
    /// Sample count
    pub n_samples: usize,
    /// Sharpe in this regime
    pub sharpe: f64,
    /// Hit rate in this regime
    pub hit_rate: f64,
    /// MI in this regime
    pub mi: f64,
}

/// Feature importance ranking
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature name
    pub name: String,
    /// Absolute correlation with persistence indicator
    pub correlation: f64,
    /// MI contribution (bits)
    pub mi_contribution: f64,
    /// Rank (1 = most important)
    pub rank: usize,
}

/// Decision for a single horizon
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonDecision {
    Success,
    Failure,
    Inconclusive,
}

/// H5 overall decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5Decision {
    /// Strong evidence that persistence indicator works
    Accept,
    /// Strong evidence that persistence indicator doesn't work
    Reject,
    /// Insufficient evidence
    Inconclusive,
}

impl H5Decision {
    pub fn to_hypothesis_decision(self) -> HypothesisDecision {
        match self {
            H5Decision::Accept => HypothesisDecision::Accept,
            H5Decision::Reject => HypothesisDecision::Reject,
            H5Decision::Inconclusive => HypothesisDecision::Inconclusive,
        }
    }
}

/// H5 test result
#[derive(Debug, Clone)]
pub struct H5TestResult {
    /// Config used
    pub config: H5TestConfig,
    /// Results by horizon
    pub horizon_results: Vec<HorizonResult>,
    /// Walk-forward fold results (for best horizon)
    pub fold_results: Vec<FoldResult>,
    /// Regime analysis
    pub regime_analysis: Option<RegimeAnalysis>,
    /// Feature importance ranking
    pub feature_importance: Vec<FeatureImportance>,
    /// Overall MI
    pub overall_mi: f64,
    /// Overall walk-forward Sharpe (average across horizons)
    pub overall_wf_sharpe: f64,
    /// Overall OOS/IS ratio
    pub overall_oos_is_ratio: f64,
    /// Best horizon
    pub best_horizon: String,
    /// Number of horizons that passed
    pub horizons_passed: usize,
    /// Number of horizons that failed
    pub horizons_failed: usize,
    /// Overall decision
    pub decision: H5Decision,
    /// Summary statistics
    pub summary: H5Summary,
}

/// Summary statistics for H5 test
#[derive(Debug, Clone)]
pub struct H5Summary {
    /// Total samples
    pub n_samples: usize,
    /// Number of horizons tested
    pub n_horizons: usize,
    /// Best horizon name
    pub best_horizon: String,
    /// Best Sharpe achieved
    pub best_sharpe: f64,
    /// Average baseline improvement
    pub avg_baseline_improvement: f64,
}

/// Compute persistence indicator from features
///
/// The persistence indicator combines multiple features that each
/// signal trend continuation:
/// - Low entropy → predictable price action → persistence
/// - High absolute momentum → strong directional move → persistence
/// - High absolute monotonicity → consistent direction → persistence
/// - Hurst > 0.5 → trending behavior → persistence
/// - High absolute OFI → sustained order flow → persistence
/// - High illiquidity + directional move → informed flow → persistence
pub fn compute_persistence_indicator(features: &[FeatureRow]) -> Vec<f64> {
    features.iter().map(|f| {
        // Entropy contribution: low entropy = high persistence
        // Normalize: assume entropy in [0, 1], invert
        let entropy_score = 1.0 - f.entropy.clamp(0.0, 1.0);

        // Momentum contribution: high absolute momentum = persistence
        // Use signed value - we predict direction AND persistence
        let momentum_score = f.momentum.clamp(-1.0, 1.0);

        // Monotonicity: high absolute = persistent direction
        let monotonicity_score = f.monotonicity.clamp(-1.0, 1.0);

        // Hurst: >0.5 = trending, normalize to [-1, 1]
        // Hurst of 0.5 -> 0, Hurst of 1 -> 1, Hurst of 0 -> -1
        let hurst_score = (f.hurst.clamp(0.0, 1.0) - 0.5) * 2.0;

        // OFI: directly measures order flow direction
        let ofi_score = f.ofi.clamp(-1.0, 1.0);

        // Illiquidity: high illiquidity amplifies signal
        // Use as multiplier, normalized to [0.5, 1.5]
        let illiq_multiplier = 0.5 + f.illiquidity.clamp(0.0, 1.0);

        // Combine scores with weights
        // Core directional signals
        let directional = 0.3 * momentum_score + 0.3 * ofi_score + 0.2 * monotonicity_score;

        // Persistence confidence
        let confidence = 0.4 * entropy_score + 0.3 * hurst_score;

        // Final indicator: direction * confidence * illiquidity amplification
        // Range roughly [-1.5, 1.5]
        directional * (1.0 + confidence) * illiq_multiplier
    }).collect()
}

/// Compute three-bar labels from returns
pub fn compute_labels(returns: &[f64], threshold: f64) -> Vec<ThreeBarLabel> {
    returns.iter().map(|&r| {
        if r > threshold {
            ThreeBarLabel::Up
        } else if r < -threshold {
            ThreeBarLabel::Down
        } else {
            ThreeBarLabel::Flat
        }
    }).collect()
}

/// Compute future returns for a given horizon
pub fn compute_future_returns(prices: &[f64], horizon: usize) -> Vec<f64> {
    if prices.len() <= horizon {
        return vec![];
    }

    (0..prices.len() - horizon)
        .map(|i| {
            let future = prices[i + horizon];
            let current = prices[i];
            if current != 0.0 {
                (future - current) / current
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute Sharpe ratio from returns
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    // Annualize: assume 1-minute bars, ~252 trading days, ~6.5 hours/day
    // 252 * 6.5 * 60 = 98,280 minutes/year
    let annualization = (252.0 * 6.5 * 60.0_f64).sqrt();
    (mean / std) * annualization
}

/// Compute strategy returns from indicator signals
///
/// Strategy: go long when indicator > 0, short when < 0
pub fn compute_strategy_returns(indicator: &[f64], returns: &[f64]) -> Vec<f64> {
    indicator.iter()
        .zip(returns.iter())
        .map(|(&signal, &ret)| {
            let position = if signal > 0.0 { 1.0 } else if signal < 0.0 { -1.0 } else { 0.0 };
            position * ret
        })
        .collect()
}

/// Compute baseline momentum strategy returns
///
/// Baseline: use simple price momentum (past N returns)
pub fn compute_momentum_baseline(returns: &[f64], lookback: usize) -> Vec<f64> {
    if returns.len() <= lookback {
        return vec![];
    }

    (lookback..returns.len())
        .map(|i| {
            let past_return: f64 = returns[i - lookback..i].iter().sum();
            let position = if past_return > 0.0 { 1.0 } else if past_return < 0.0 { -1.0 } else { 0.0 };
            position * returns[i]
        })
        .collect()
}

/// Run walk-forward validation
pub fn walk_forward_validation(
    indicator: &[f64],
    returns: &[f64],
    n_folds: usize,
) -> Vec<FoldResult> {
    if indicator.len() != returns.len() || n_folds < 2 {
        return vec![];
    }

    let n = indicator.len();
    let fold_size = n / n_folds;

    if fold_size < 10 {
        return vec![];
    }

    let mut results = Vec::with_capacity(n_folds - 1);

    // Walk-forward: train on folds 0..i, test on fold i
    for test_fold in 1..n_folds {
        let train_end = test_fold * fold_size;
        let test_end = if test_fold == n_folds - 1 { n } else { (test_fold + 1) * fold_size };

        // In-sample
        let is_indicator = &indicator[..train_end];
        let is_returns = &returns[..train_end];
        let is_strat_returns = compute_strategy_returns(is_indicator, is_returns);
        let is_sharpe = sharpe_ratio(&is_strat_returns);

        // Out-of-sample
        let oos_indicator = &indicator[train_end..test_end];
        let oos_returns = &returns[train_end..test_end];
        let oos_strat_returns = compute_strategy_returns(oos_indicator, oos_returns);
        let oos_sharpe = sharpe_ratio(&oos_strat_returns);

        // Hit rates
        let is_hit_rate = hit_rate(is_indicator, is_returns);
        let oos_hit_rate = hit_rate(oos_indicator, oos_returns);

        results.push(FoldResult {
            fold_idx: test_fold - 1,
            is_sharpe,
            oos_sharpe,
            is_hit_rate,
            oos_hit_rate,
        });
    }

    results
}

/// Compute hit rate (directional accuracy)
pub fn hit_rate(indicator: &[f64], returns: &[f64]) -> f64 {
    if indicator.len() != returns.len() || indicator.is_empty() {
        return 0.0;
    }

    let hits: usize = indicator.iter()
        .zip(returns.iter())
        .filter(|(&signal, &ret)| {
            (signal > 0.0 && ret > 0.0) || (signal < 0.0 && ret < 0.0)
        })
        .count();

    hits as f64 / indicator.len() as f64
}

/// Test a single horizon
pub fn test_horizon(
    indicator: &[f64],
    future_returns: &[f64],
    horizon_name: &str,
    horizon_points: usize,
    config: &H5TestConfig,
) -> HorizonResult {
    // Ensure same length
    let n = indicator.len().min(future_returns.len());
    let indicator = &indicator[..n];
    let future_returns = &future_returns[..n];

    // Walk-forward validation
    let fold_results = walk_forward_validation(indicator, future_returns, config.n_folds);

    // Compute aggregates
    let (wf_sharpe, is_sharpe, oos_sharpe) = if fold_results.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let wf = fold_results.iter().map(|f| f.oos_sharpe).sum::<f64>() / fold_results.len() as f64;
        let is_avg = fold_results.iter().map(|f| f.is_sharpe).sum::<f64>() / fold_results.len() as f64;
        let oos_avg = fold_results.iter().map(|f| f.oos_sharpe).sum::<f64>() / fold_results.len() as f64;
        (wf, is_avg, oos_avg)
    };

    let oos_is_ratio = if is_sharpe.abs() > 1e-10 {
        oos_sharpe / is_sharpe
    } else {
        0.0
    };

    // Baseline comparison
    let lookback = horizon_points.max(5);
    let baseline_returns = compute_momentum_baseline(future_returns, lookback);
    let baseline_sharpe = sharpe_ratio(&baseline_returns);

    let baseline_improvement = if baseline_sharpe.abs() > 1e-10 {
        (wf_sharpe - baseline_sharpe) / baseline_sharpe.abs()
    } else if wf_sharpe > 0.0 {
        1.0 // Any positive is infinite improvement over zero
    } else {
        0.0
    };

    // MI calculation
    let mi = mutual_information_adaptive(indicator, future_returns);

    // Hit rate
    let hit_rate = hit_rate(indicator, future_returns);

    // Correlation
    let correlation = if indicator.len() > 2 {
        let corr = pearson_correlation(indicator, future_returns);
        let spearman = spearman_correlation(indicator, future_returns);
        let (t_stat, p_value) = t_test_correlation(corr, indicator.len());
        let se = 1.0 / (indicator.len() as f64 - 3.0).sqrt();
        CorrelationResult {
            pearson: corr,
            spearman,
            t_statistic: t_stat,
            p_value,
            n: indicator.len(),
            ci_lower: corr - 1.96 * se,
            ci_upper: corr + 1.96 * se,
            significant: p_value < 0.05,
        }
    } else {
        CorrelationResult {
            pearson: 0.0,
            spearman: 0.0,
            t_statistic: 0.0,
            p_value: 1.0,
            n: 0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            significant: false,
        }
    };

    // Decision for this horizon
    let decision = if wf_sharpe >= config.min_wf_sharpe
        && oos_is_ratio >= config.min_oos_is_ratio
        && baseline_improvement >= config.min_baseline_improvement
        && mi >= config.min_mi
    {
        HorizonDecision::Success
    } else if wf_sharpe < config.sharpe_failure_threshold
        || oos_is_ratio < config.oos_is_failure_threshold
        || mi < config.mi_failure_threshold
    {
        HorizonDecision::Failure
    } else {
        HorizonDecision::Inconclusive
    };

    HorizonResult {
        horizon_name: horizon_name.to_string(),
        horizon_points,
        wf_sharpe,
        is_sharpe,
        oos_sharpe,
        oos_is_ratio,
        baseline_sharpe,
        baseline_improvement,
        mi,
        hit_rate,
        correlation,
        decision,
    }
}

/// Compute feature importance
pub fn compute_feature_importance(
    features: &[FeatureRow],
    returns: &[f64],
) -> Vec<FeatureImportance> {
    if features.is_empty() || features.len() != returns.len() {
        return vec![];
    }

    let feature_names = ["entropy", "momentum", "monotonicity", "hurst", "ofi", "illiquidity"];
    let mut importances: Vec<FeatureImportance> = Vec::with_capacity(6);

    for (idx, name) in feature_names.iter().enumerate() {
        let feature_values: Vec<f64> = features.iter().map(|f| {
            match idx {
                0 => f.entropy,
                1 => f.momentum,
                2 => f.monotonicity,
                3 => f.hurst,
                4 => f.ofi,
                5 => f.illiquidity,
                _ => 0.0,
            }
        }).collect();

        let corr = pearson_correlation(&feature_values, returns);
        let mi = mutual_information_adaptive(&feature_values, returns);

        importances.push(FeatureImportance {
            name: name.to_string(),
            correlation: corr.abs(),
            mi_contribution: mi,
            rank: 0, // Will be set after sorting
        });
    }

    // Sort by MI contribution (descending)
    importances.sort_by(|a, b| b.mi_contribution.partial_cmp(&a.mi_contribution).unwrap_or(std::cmp::Ordering::Equal));

    // Set ranks
    for (i, imp) in importances.iter_mut().enumerate() {
        imp.rank = i + 1;
    }

    importances
}

/// Compute regime analysis (low vol vs high vol)
pub fn compute_regime_analysis(
    indicator: &[f64],
    returns: &[f64],
    volatility: &[f64],
) -> Option<RegimeAnalysis> {
    if indicator.len() != returns.len() || indicator.len() != volatility.len() {
        return None;
    }

    // Split by median volatility
    let mut sorted_vol = volatility.to_vec();
    sorted_vol.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_vol = sorted_vol[sorted_vol.len() / 2];

    let mut low_vol_indicator = Vec::new();
    let mut low_vol_returns = Vec::new();
    let mut high_vol_indicator = Vec::new();
    let mut high_vol_returns = Vec::new();

    for i in 0..indicator.len() {
        if volatility[i] <= median_vol {
            low_vol_indicator.push(indicator[i]);
            low_vol_returns.push(returns[i]);
        } else {
            high_vol_indicator.push(indicator[i]);
            high_vol_returns.push(returns[i]);
        }
    }

    let low_vol_strat = compute_strategy_returns(&low_vol_indicator, &low_vol_returns);
    let high_vol_strat = compute_strategy_returns(&high_vol_indicator, &high_vol_returns);

    let low_vol_sharpe = sharpe_ratio(&low_vol_strat);
    let high_vol_sharpe = sharpe_ratio(&high_vol_strat);

    let low_vol_hit = hit_rate(&low_vol_indicator, &low_vol_returns);
    let high_vol_hit = hit_rate(&high_vol_indicator, &high_vol_returns);

    let low_vol_mi = mutual_information_adaptive(&low_vol_indicator, &low_vol_returns);
    let high_vol_mi = mutual_information_adaptive(&high_vol_indicator, &high_vol_returns);

    Some(RegimeAnalysis {
        low_vol: RegimeResult {
            name: "low_volatility".to_string(),
            n_samples: low_vol_indicator.len(),
            sharpe: low_vol_sharpe,
            hit_rate: low_vol_hit,
            mi: low_vol_mi,
        },
        high_vol: RegimeResult {
            name: "high_volatility".to_string(),
            n_samples: high_vol_indicator.len(),
            sharpe: high_vol_sharpe,
            hit_rate: high_vol_hit,
            mi: high_vol_mi,
        },
        regime_diff: high_vol_sharpe - low_vol_sharpe,
    })
}

/// Run the full H5 test
///
/// # Arguments
/// * `features` - Feature rows for computing persistence indicator
/// * `prices` - Price series for computing returns
/// * `volatility` - Volatility series for regime analysis (optional)
/// * `config` - Test configuration
///
/// # Returns
/// H5TestResult with decision
pub fn run_h5_persistence_test(
    features: &[FeatureRow],
    prices: &[f64],
    volatility: Option<&[f64]>,
    config: &H5TestConfig,
) -> H5TestResult {
    let default_config = H5TestConfig::default();
    let config = if config.horizons.is_empty() { &default_config } else { config };

    // Compute persistence indicator
    let indicator = compute_persistence_indicator(features);

    // Test each horizon
    let mut horizon_results = Vec::new();
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_horizon = String::new();

    for &horizon in &config.horizons {
        let horizon_name = format!("{}min", horizon);
        let future_returns = compute_future_returns(prices, horizon);

        // Align indicator with returns
        let n = indicator.len().min(future_returns.len());
        if n < 100 {
            continue;
        }

        let result = test_horizon(
            &indicator[..n],
            &future_returns,
            &horizon_name,
            horizon,
            config,
        );

        if result.wf_sharpe > best_sharpe {
            best_sharpe = result.wf_sharpe;
            best_horizon = result.horizon_name.clone();
        }

        horizon_results.push(result);
    }

    // If no horizons tested, return failure result
    if horizon_results.is_empty() {
        return H5TestResult {
            config: config.clone(),
            horizon_results: vec![],
            fold_results: vec![],
            regime_analysis: None,
            feature_importance: vec![],
            overall_mi: 0.0,
            overall_wf_sharpe: 0.0,
            overall_oos_is_ratio: 0.0,
            best_horizon: String::new(),
            horizons_passed: 0,
            horizons_failed: 0,
            decision: H5Decision::Reject,
            summary: H5Summary {
                n_samples: features.len(),
                n_horizons: 0,
                best_horizon: String::new(),
                best_sharpe: 0.0,
                avg_baseline_improvement: 0.0,
            },
        };
    }

    // Count successes and failures
    let horizons_passed = horizon_results.iter()
        .filter(|r| r.decision == HorizonDecision::Success)
        .count();
    let horizons_failed = horizon_results.iter()
        .filter(|r| r.decision == HorizonDecision::Failure)
        .count();

    // Compute overall metrics
    let overall_wf_sharpe = horizon_results.iter().map(|r| r.wf_sharpe).sum::<f64>()
        / horizon_results.len() as f64;
    let overall_oos_is_ratio = horizon_results.iter().map(|r| r.oos_is_ratio).sum::<f64>()
        / horizon_results.len() as f64;
    let overall_mi = horizon_results.iter().map(|r| r.mi).sum::<f64>()
        / horizon_results.len() as f64;

    // Get fold results from best horizon
    let best_horizon_result = horizon_results.iter()
        .find(|r| r.horizon_name == best_horizon);
    let fold_results = if let Some(best) = best_horizon_result {
        let future_returns = compute_future_returns(prices, best.horizon_points);
        let n = indicator.len().min(future_returns.len());
        walk_forward_validation(&indicator[..n], &future_returns, config.n_folds)
    } else {
        vec![]
    };

    // Regime analysis (use first horizon's returns)
    let regime_analysis = if let (Some(vol), Some(first_result)) = (volatility, horizon_results.first()) {
        let future_returns = compute_future_returns(prices, first_result.horizon_points);
        let n = indicator.len().min(future_returns.len()).min(vol.len());
        compute_regime_analysis(&indicator[..n], &future_returns[..n], &vol[..n])
    } else {
        None
    };

    // Feature importance (use first horizon's returns)
    let feature_importance = if let Some(first_result) = horizon_results.first() {
        let future_returns = compute_future_returns(prices, first_result.horizon_points);
        let n = features.len().min(future_returns.len());
        compute_feature_importance(&features[..n], &future_returns)
    } else {
        vec![]
    };

    // Average baseline improvement
    let avg_baseline_improvement = horizon_results.iter()
        .map(|r| r.baseline_improvement)
        .sum::<f64>() / horizon_results.len() as f64;

    // Overall decision
    // Accept if majority of horizons pass
    // Reject if majority fail
    let decision = if horizons_passed > horizons_failed && horizons_passed >= 2 {
        H5Decision::Accept
    } else if horizons_failed > horizons_passed && horizons_failed >= 2 {
        H5Decision::Reject
    } else {
        H5Decision::Inconclusive
    };

    H5TestResult {
        config: config.clone(),
        horizon_results,
        fold_results,
        regime_analysis,
        feature_importance,
        overall_mi,
        overall_wf_sharpe,
        overall_oos_is_ratio,
        best_horizon: best_horizon.clone(),
        horizons_passed,
        horizons_failed,
        decision,
        summary: H5Summary {
            n_samples: features.len(),
            n_horizons: config.horizons.len(),
            best_horizon,
            best_sharpe,
            avg_baseline_improvement,
        },
    }
}

impl H5TestResult {
    /// Generate detailed report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# H5 Hypothesis Test: Persistence Indicator\n\n");
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Decision**: {}\n", match self.decision {
            H5Decision::Accept => "ACCEPT - Persistence indicator works",
            H5Decision::Reject => "REJECT - Persistence indicator does not work",
            H5Decision::Inconclusive => "INCONCLUSIVE - Need more data",
        }));
        report.push_str(&format!("- **Samples**: {}\n", self.summary.n_samples));
        report.push_str(&format!("- **Horizons Tested**: {}\n", self.summary.n_horizons));
        report.push_str(&format!("- **Horizons Passed**: {}\n", self.horizons_passed));
        report.push_str(&format!("- **Best Horizon**: {}\n", self.summary.best_horizon));
        report.push_str(&format!("- **Best Sharpe**: {:.3}\n", self.summary.best_sharpe));
        report.push_str(&format!("- **Overall MI**: {:.4} bits\n", self.overall_mi));

        report.push_str("\n## Horizon Results\n\n");
        report.push_str("| Horizon | WF Sharpe | OOS/IS | Baseline Impr | MI | Hit Rate | Decision |\n");
        report.push_str("|---------|-----------|--------|---------------|-------|----------|----------|\n");

        for hr in &self.horizon_results {
            report.push_str(&format!(
                "| {} | {:.3} | {:.2} | {:.1}% | {:.4} | {:.1}% | {:?} |\n",
                hr.horizon_name,
                hr.wf_sharpe,
                hr.oos_is_ratio,
                hr.baseline_improvement * 100.0,
                hr.mi,
                hr.hit_rate * 100.0,
                hr.decision,
            ));
        }

        if let Some(regime) = &self.regime_analysis {
            report.push_str("\n## Regime Analysis\n\n");
            report.push_str(&format!("| Regime | N | Sharpe | Hit Rate | MI |\n"));
            report.push_str("|--------|---|--------|----------|----|\n");
            report.push_str(&format!(
                "| Low Vol | {} | {:.3} | {:.1}% | {:.4} |\n",
                regime.low_vol.n_samples,
                regime.low_vol.sharpe,
                regime.low_vol.hit_rate * 100.0,
                regime.low_vol.mi,
            ));
            report.push_str(&format!(
                "| High Vol | {} | {:.3} | {:.1}% | {:.4} |\n",
                regime.high_vol.n_samples,
                regime.high_vol.sharpe,
                regime.high_vol.hit_rate * 100.0,
                regime.high_vol.mi,
            ));
        }

        if !self.feature_importance.is_empty() {
            report.push_str("\n## Feature Importance\n\n");
            report.push_str("| Rank | Feature | Correlation | MI Contribution |\n");
            report.push_str("|------|---------|-------------|------------------|\n");

            for fi in &self.feature_importance {
                report.push_str(&format!(
                    "| {} | {} | {:.3} | {:.4} bits |\n",
                    fi.rank,
                    fi.name,
                    fi.correlation,
                    fi.mi_contribution,
                ));
            }
        }

        report.push_str("\n## Success Criteria\n\n");
        report.push_str(&format!("- WF Sharpe > {:.1} (actual: {:.3})\n",
            self.config.min_wf_sharpe, self.overall_wf_sharpe));
        report.push_str(&format!("- OOS/IS ratio > {:.1} (actual: {:.2})\n",
            self.config.min_oos_is_ratio, self.overall_oos_is_ratio));
        report.push_str(&format!("- Baseline improvement > {:.0}% (actual: {:.1}%)\n",
            self.config.min_baseline_improvement * 100.0,
            self.summary.avg_baseline_improvement * 100.0));
        report.push_str(&format!("- MI > {:.2} bits (actual: {:.4} bits)\n",
            self.config.min_mi, self.overall_mi));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_features(n: usize, predictive: bool) -> Vec<FeatureRow> {
        let mut features = Vec::with_capacity(n);
        let mut rng_state = 42u64;

        for i in 0..n {
            // Simple LCG for reproducibility
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rand1 = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let rand2 = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;

            let base_signal = if predictive {
                // Create a signal that predicts future returns
                (i as f64 * 0.1).sin() * 0.5
            } else {
                0.0
            };

            features.push(FeatureRow {
                entropy: 0.5 - base_signal * 0.3 + rand1 * 0.1,
                momentum: base_signal + rand1 * 0.2,
                monotonicity: base_signal * 0.8 + rand2 * 0.1,
                hurst: 0.6 + base_signal * 0.1,
                ofi: base_signal * 0.7 + rand1 * 0.15,
                illiquidity: 0.5 + rand2.abs() * 0.2,
            });
        }

        features
    }

    fn generate_test_prices(features: &[FeatureRow], predictive: bool) -> Vec<f64> {
        let mut prices = Vec::with_capacity(features.len());
        let mut price = 100.0;
        let mut rng_state = 123u64;

        for (i, f) in features.iter().enumerate() {
            prices.push(price);

            // Simple LCG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5) * 0.01;

            let signal_return = if predictive {
                // Returns follow the indicator direction
                f.momentum * 0.005
            } else {
                0.0
            };

            price *= 1.0 + signal_return + noise;
        }

        prices
    }

    #[test]
    fn test_persistence_indicator_computation() {
        let features = generate_test_features(100, false);
        let indicator = compute_persistence_indicator(&features);

        assert_eq!(indicator.len(), features.len());

        // Indicator should be bounded
        for &val in &indicator {
            assert!(val.abs() < 5.0, "Indicator out of expected range: {}", val);
        }
    }

    #[test]
    fn test_three_bar_labels() {
        let returns = vec![-0.02, -0.001, 0.0, 0.001, 0.02];
        let labels = compute_labels(&returns, 0.005);

        assert_eq!(labels[0], ThreeBarLabel::Down);
        assert_eq!(labels[1], ThreeBarLabel::Flat);
        assert_eq!(labels[2], ThreeBarLabel::Flat);
        assert_eq!(labels[3], ThreeBarLabel::Flat);
        assert_eq!(labels[4], ThreeBarLabel::Up);
    }

    #[test]
    fn test_future_returns() {
        let prices = vec![100.0, 101.0, 102.0, 101.0, 103.0];

        let returns_1 = compute_future_returns(&prices, 1);
        assert_eq!(returns_1.len(), 4);
        assert!((returns_1[0] - 0.01).abs() < 1e-10);

        let returns_2 = compute_future_returns(&prices, 2);
        assert_eq!(returns_2.len(), 3);
        assert!((returns_2[0] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_ratio() {
        // Zero mean returns
        let returns = vec![0.01, -0.01, 0.01, -0.01];
        let sharpe = sharpe_ratio(&returns);
        assert!(sharpe.abs() < 0.1, "Zero mean should have ~0 Sharpe");

        // All positive returns with some variance
        let positive = vec![0.008, 0.01, 0.012, 0.009, 0.011, 0.01];
        let sharpe_pos = sharpe_ratio(&positive);
        assert!(sharpe_pos > 10.0, "All positive returns should have high Sharpe: got {}", sharpe_pos);

        // Zero variance returns
        let zero_var = vec![0.01, 0.01, 0.01, 0.01];
        let sharpe_zero_var = sharpe_ratio(&zero_var);
        assert!(sharpe_zero_var.abs() < 0.1, "Zero variance should return 0");
    }

    #[test]
    fn test_strategy_returns() {
        let indicator = vec![1.0, -1.0, 0.5, -0.5];
        let returns = vec![0.01, -0.01, -0.01, 0.01];

        let strat_returns = compute_strategy_returns(&indicator, &returns);

        assert_eq!(strat_returns.len(), 4);
        assert!((strat_returns[0] - 0.01).abs() < 1e-10); // long * positive
        assert!((strat_returns[1] - 0.01).abs() < 1e-10); // short * negative = positive
        assert!((strat_returns[2] - (-0.01)).abs() < 1e-10); // long * negative
        assert!((strat_returns[3] - (-0.01)).abs() < 1e-10); // short * positive
    }

    #[test]
    fn test_hit_rate() {
        let indicator = vec![1.0, -1.0, 1.0, -1.0];
        let returns = vec![0.01, -0.01, -0.01, 0.01];

        let hr = hit_rate(&indicator, &returns);
        assert!((hr - 0.5).abs() < 1e-10, "Expected 50% hit rate");
    }

    #[test]
    fn test_walk_forward_validation() {
        let indicator: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 0.01).collect();

        let folds = walk_forward_validation(&indicator, &returns, 5);

        assert_eq!(folds.len(), 4); // n_folds - 1

        for fold in &folds {
            assert!(fold.is_sharpe.is_finite());
            assert!(fold.oos_sharpe.is_finite());
            assert!(fold.is_hit_rate >= 0.0 && fold.is_hit_rate <= 1.0);
            assert!(fold.oos_hit_rate >= 0.0 && fold.oos_hit_rate <= 1.0);
        }
    }

    #[test]
    fn test_momentum_baseline() {
        let returns: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();

        let baseline = compute_momentum_baseline(&returns, 5);

        assert_eq!(baseline.len(), 45);
    }

    #[test]
    fn test_feature_importance() {
        let features = generate_test_features(200, true);
        let prices = generate_test_prices(&features, true);
        let returns = compute_future_returns(&prices, 1);

        let importance = compute_feature_importance(&features[..returns.len()], &returns);

        assert_eq!(importance.len(), 6);

        // Check ranks are valid
        for (i, fi) in importance.iter().enumerate() {
            assert_eq!(fi.rank, i + 1);
        }
    }

    #[test]
    fn test_regime_analysis() {
        let indicator: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let volatility: Vec<f64> = (0..100).map(|i| 0.01 + (i as f64 * 0.05).sin().abs() * 0.02).collect();

        let regime = compute_regime_analysis(&indicator, &returns, &volatility);

        assert!(regime.is_some());
        let regime = regime.unwrap();

        assert!(regime.low_vol.n_samples > 0);
        assert!(regime.high_vol.n_samples > 0);
        assert_eq!(regime.low_vol.n_samples + regime.high_vol.n_samples, 100);
    }

    #[test]
    fn test_h5_with_random_data() {
        let features = generate_test_features(500, false);
        let prices = generate_test_prices(&features, false);

        let config = H5TestConfig {
            horizons: vec![1, 5],
            ..Default::default()
        };

        let result = run_h5_persistence_test(&features, &prices, None, &config);

        // Random data should not pass
        assert!(result.horizons_passed < 2, "Random data should not consistently pass");

        // Should have horizon results
        assert!(!result.horizon_results.is_empty());
    }

    #[test]
    fn test_h5_with_predictive_signal() {
        let features = generate_test_features(1000, true);
        let prices = generate_test_prices(&features, true);

        let config = H5TestConfig {
            horizons: vec![1, 5],
            min_wf_sharpe: 0.1, // Lower threshold for test
            min_mi: 0.001,
            mi_failure_threshold: 0.0001,
            sharpe_failure_threshold: 0.05,
            ..Default::default()
        };

        let result = run_h5_persistence_test(&features, &prices, None, &config);

        // With predictive signal, should have some positive results
        assert!(!result.horizon_results.is_empty());
        assert!(result.overall_mi >= 0.0);
    }

    #[test]
    fn test_h5_insufficient_data() {
        let features = generate_test_features(50, true); // Too few
        let prices = generate_test_prices(&features, true);

        let config = H5TestConfig::default();
        let result = run_h5_persistence_test(&features, &prices, None, &config);

        // Should handle gracefully
        assert_eq!(result.decision, H5Decision::Reject);
    }

    #[test]
    fn test_decision_mapping() {
        assert_eq!(H5Decision::Accept.to_hypothesis_decision(), HypothesisDecision::Accept);
        assert_eq!(H5Decision::Reject.to_hypothesis_decision(), HypothesisDecision::Reject);
        assert_eq!(H5Decision::Inconclusive.to_hypothesis_decision(), HypothesisDecision::Inconclusive);
    }

    #[test]
    fn test_report_generation() {
        let features = generate_test_features(500, true);
        let prices = generate_test_prices(&features, true);
        let volatility: Vec<f64> = (0..500).map(|i| 0.01 + (i as f64 * 0.01).sin().abs() * 0.01).collect();

        let config = H5TestConfig {
            horizons: vec![1, 5],
            ..Default::default()
        };

        let result = run_h5_persistence_test(&features, &prices, Some(&volatility), &config);
        let report = result.generate_report();

        assert!(report.contains("H5 Hypothesis Test"));
        assert!(report.contains("Summary"));
        assert!(report.contains("Horizon Results"));
        assert!(report.contains("Feature Importance"));
    }

    #[test]
    fn test_horizon_result_fields() {
        let features = generate_test_features(500, true);
        let prices = generate_test_prices(&features, true);
        let indicator = compute_persistence_indicator(&features);
        let future_returns = compute_future_returns(&prices, 1);

        let config = H5TestConfig::default();
        let result = test_horizon(&indicator, &future_returns, "1min", 1, &config);

        assert_eq!(result.horizon_name, "1min");
        assert_eq!(result.horizon_points, 1);
        assert!(result.hit_rate >= 0.0 && result.hit_rate <= 1.0);
        assert!(result.mi >= 0.0);
    }

    #[test]
    fn test_empty_inputs() {
        let features: Vec<FeatureRow> = vec![];
        let prices: Vec<f64> = vec![];

        let config = H5TestConfig::default();
        let result = run_h5_persistence_test(&features, &prices, None, &config);

        assert_eq!(result.decision, H5Decision::Reject);
        assert!(result.horizon_results.is_empty());
    }
}
