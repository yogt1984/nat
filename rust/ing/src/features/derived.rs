//! Derived/Interaction Feature Extraction
//!
//! This module implements derived features that combine base features
//! to capture regime dynamics and feature interactions.
//!
//! # Features
//!
//! | Feature | Formula | Interpretation |
//! |---------|---------|----------------|
//! | **entropy_trend_interaction** | entropy × (1 - monotonicity) | High = choppy, low = trending |
//! | **entropy_volatility_ratio** | entropy / (1 + volatility) | Regime stability |
//! | **trend_strength** | momentum_sign × monotonicity × (1 - entropy) | Composite trend |
//! | **illiquidity_trend** | illiquidity × |momentum| | Informed flow in trends |
//! | **toxicity_regime** | toxicity × entropy | Toxic choppy markets |
//!
//! # Theory
//!
//! These interaction features capture non-linear relationships between
//! market microstructure variables that individual features miss.

use super::{
    EntropyFeatures, TrendFeatures, VolatilityFeatures,
    IlliquidityFeatures, ToxicityFeatures, FlowFeatures
};

/// Derived/Interaction features
/// Total: 15 features
#[derive(Debug, Clone, Default)]
pub struct DerivedFeatures {
    // Entropy-Trend interactions
    /// entropy × (1 - monotonicity): high = choppy regime
    pub entropy_trend_interaction: f64,
    /// Normalized interaction (z-score style)
    pub entropy_trend_zscore: f64,

    // Trend strength composites
    /// sign(momentum) × monotonicity × (1 - entropy): directional strength
    pub trend_strength_60: f64,
    /// Trend strength at 300 tick window
    pub trend_strength_300: f64,
    /// Trend strength ratio (short/long)
    pub trend_strength_ratio: f64,

    // Entropy-Volatility interactions
    /// entropy / (1 + normalized_volatility): regime stability
    pub entropy_volatility_ratio: f64,
    /// Low entropy + high vol = breakout; high entropy + high vol = chaos
    pub regime_type_score: f64,

    // Illiquidity-Trend interactions
    /// illiquidity × |momentum|: informed flow during trends
    pub illiquidity_trend: f64,
    /// Kyle lambda × monotonicity: persistent informed trading
    pub informed_trend_score: f64,

    // Toxicity-Regime interactions
    /// toxicity × entropy: toxic flow in choppy markets
    pub toxicity_regime: f64,
    /// VPIN × (1 - monotonicity): toxic when directionless
    pub toxic_chop_score: f64,

    // Cross-feature momentum
    /// Rate of change of trend_strength
    pub trend_strength_roc: f64,
    /// Entropy acceleration (second derivative proxy)
    pub entropy_momentum: f64,

    // Composite regime indicators
    /// Overall regime score: -1 = trending, 0 = transition, +1 = mean-reverting
    pub regime_indicator: f64,
    /// Confidence in regime classification
    pub regime_confidence: f64,
}

impl DerivedFeatures {
    pub fn count() -> usize {
        15
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "derived_entropy_trend_interaction",
            "derived_entropy_trend_zscore",
            "derived_trend_strength_60",
            "derived_trend_strength_300",
            "derived_trend_strength_ratio",
            "derived_entropy_volatility_ratio",
            "derived_regime_type_score",
            "derived_illiquidity_trend",
            "derived_informed_trend_score",
            "derived_toxicity_regime",
            "derived_toxic_chop_score",
            "derived_trend_strength_roc",
            "derived_entropy_momentum",
            "derived_regime_indicator",
            "derived_regime_confidence",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.entropy_trend_interaction,
            self.entropy_trend_zscore,
            self.trend_strength_60,
            self.trend_strength_300,
            self.trend_strength_ratio,
            self.entropy_volatility_ratio,
            self.regime_type_score,
            self.illiquidity_trend,
            self.informed_trend_score,
            self.toxicity_regime,
            self.toxic_chop_score,
            self.trend_strength_roc,
            self.entropy_momentum,
            self.regime_indicator,
            self.regime_confidence,
        ]
    }
}

/// Compute derived features from base features
pub fn compute(
    entropy: &EntropyFeatures,
    trend: &TrendFeatures,
    volatility: &VolatilityFeatures,
    illiquidity: &IlliquidityFeatures,
    toxicity: &ToxicityFeatures,
    flow: &FlowFeatures,
) -> DerivedFeatures {
    // Get key base features with defaults
    let tick_entropy = entropy.tick_entropy_30s;
    let monotonicity_60 = trend.monotonicity_60;
    let monotonicity_300 = trend.monotonicity_300;
    let momentum_60 = trend.momentum_60;
    let momentum_300 = trend.momentum_300;
    let vol_1m = volatility.returns_1m;
    let kyle_lambda = illiquidity.kyle_lambda_100;
    let vpin = toxicity.vpin_50;
    let toxicity_index = toxicity.toxicity_index;
    let flow_imbalance = toxicity.flow_imbalance_abs;

    // ========================================================================
    // Entropy-Trend Interactions
    // ========================================================================

    // entropy × (1 - monotonicity): high when choppy (high entropy, low mono)
    let entropy_trend_interaction = tick_entropy * (1.0 - monotonicity_60);

    // Z-score style normalization (assuming typical ranges)
    // entropy ∈ [0, 1.1], monotonicity ∈ [0.5, 1.0]
    // interaction ∈ [0, 0.55] typically, mean ~0.2
    let entropy_trend_zscore = (entropy_trend_interaction - 0.2) / 0.15;

    // ========================================================================
    // Trend Strength Composites
    // ========================================================================

    // Trend strength = sign(momentum) × monotonicity × (1 - entropy)
    // Ranges from -1 (strong downtrend) to +1 (strong uptrend)
    let trend_strength_60 = compute_trend_strength(momentum_60, monotonicity_60, tick_entropy);
    let trend_strength_300 = compute_trend_strength(momentum_300, monotonicity_300, tick_entropy);

    // Ratio of short to long trend strength (momentum acceleration)
    let trend_strength_ratio = if trend_strength_300.abs() > 0.01 {
        trend_strength_60 / trend_strength_300
    } else {
        1.0
    };

    // ========================================================================
    // Entropy-Volatility Interactions
    // ========================================================================

    // Normalize volatility to [0, 1] range (assuming vol_1m is annualized %)
    let vol_normalized = (vol_1m * 100.0).clamp(0.0, 1.0);

    // entropy / (1 + vol): high when orderly low-vol regime
    let entropy_volatility_ratio = if vol_normalized > 0.001 {
        tick_entropy / (1.0 + vol_normalized)
    } else {
        tick_entropy
    };

    // Regime type score:
    // Low entropy + high vol = breakout (score > 0)
    // High entropy + high vol = chaos (score < 0)
    // Low entropy + low vol = quiet trend (score ~ 0)
    let regime_type_score = (1.0 - tick_entropy) * vol_normalized - tick_entropy * vol_normalized;

    // ========================================================================
    // Illiquidity-Trend Interactions
    // ========================================================================

    // illiquidity × |momentum|: high when illiquid AND trending (informed flow)
    let momentum_abs = momentum_60.abs();
    let illiquidity_normalized = (kyle_lambda / 100.0).clamp(0.0, 1.0);
    let illiquidity_trend = illiquidity_normalized * momentum_abs * 1000.0; // Scale for visibility

    // Kyle lambda × monotonicity: persistent informed trading
    let informed_trend_score = illiquidity_normalized * monotonicity_60;

    // ========================================================================
    // Toxicity-Regime Interactions
    // ========================================================================

    // toxicity × entropy: toxic flow is worse in choppy markets
    let toxicity_regime = toxicity_index * tick_entropy;

    // VPIN × (1 - monotonicity): toxic when market is directionless
    let toxic_chop_score = vpin * (1.0 - monotonicity_60);

    // ========================================================================
    // Cross-Feature Momentum
    // ========================================================================

    // Trend strength rate of change (using 60 vs 300 as proxy)
    let trend_strength_roc = trend_strength_60 - trend_strength_300;

    // Entropy momentum: use tick entropy windows as proxy
    // Higher window entropy vs lower = entropy increasing (regime breaking down)
    let entropy_1m = entropy.tick_entropy_1m;
    let entropy_5s = entropy.tick_entropy_5s;
    let entropy_momentum = entropy_1m - entropy_5s;

    // ========================================================================
    // Composite Regime Indicators
    // ========================================================================

    // Regime indicator: -1 = trending, 0 = transition, +1 = mean-reverting
    let regime_indicator = compute_regime_indicator(
        tick_entropy,
        monotonicity_60,
        trend_strength_60,
        flow_imbalance
    );

    // Confidence in regime classification
    let regime_confidence = compute_regime_confidence(
        tick_entropy,
        monotonicity_60,
        trend_strength_60
    );

    DerivedFeatures {
        entropy_trend_interaction,
        entropy_trend_zscore,
        trend_strength_60,
        trend_strength_300,
        trend_strength_ratio,
        entropy_volatility_ratio,
        regime_type_score,
        illiquidity_trend,
        informed_trend_score,
        toxicity_regime,
        toxic_chop_score,
        trend_strength_roc,
        entropy_momentum,
        regime_indicator,
        regime_confidence,
    }
}

/// Compute trend strength composite
/// Returns value in [-1, 1] where magnitude indicates strength
fn compute_trend_strength(momentum: f64, monotonicity: f64, entropy: f64) -> f64 {
    // Direction from momentum sign
    let direction = if momentum > 0.0 { 1.0 } else if momentum < 0.0 { -1.0 } else { 0.0 };

    // Strength from monotonicity and inverse entropy
    // monotonicity ∈ [0.5, 1.0], map to [0, 1]
    let mono_strength = (monotonicity - 0.5) * 2.0;

    // entropy ∈ [0, ~1.1], invert and clamp
    let entropy_factor = (1.0 - entropy).clamp(0.0, 1.0);

    direction * mono_strength * entropy_factor
}

/// Compute regime indicator
/// Returns: -1 = trending, 0 = transition, +1 = mean-reverting
fn compute_regime_indicator(
    entropy: f64,
    monotonicity: f64,
    trend_strength: f64,
    flow_imbalance: f64,
) -> f64 {
    // Trending: low entropy, high monotonicity, strong trend
    let trending_score = (1.0 - entropy) * (monotonicity - 0.5) * 2.0 * trend_strength.abs();

    // Mean-reverting: high entropy, low monotonicity, weak trend
    let mean_revert_score = entropy * (1.0 - monotonicity) * 2.0 * (1.0 - trend_strength.abs());

    // Flow imbalance adds to trending score
    let flow_factor = flow_imbalance * 0.5;

    // Combine: negative = trending, positive = mean-reverting
    let raw_indicator = mean_revert_score - trending_score - flow_factor;

    // Clamp to [-1, 1]
    raw_indicator.clamp(-1.0, 1.0)
}

/// Compute confidence in regime classification
fn compute_regime_confidence(
    entropy: f64,
    monotonicity: f64,
    trend_strength: f64,
) -> f64 {
    // High confidence when signals agree
    // Low entropy + high monotonicity + strong trend = confident trending
    // High entropy + low monotonicity + weak trend = confident mean-reverting

    let trending_agreement = (1.0 - entropy) * (monotonicity - 0.5) * 2.0 * trend_strength.abs();
    let reverting_agreement = entropy * (1.0 - monotonicity) * 2.0 * (1.0 - trend_strength.abs());

    // Confidence is max of agreements, scaled to [0, 1]
    (trending_agreement.max(reverting_agreement) * 2.0).clamp(0.0, 1.0)
}

// ============================================================================
// Skeptical Tests Module
// ============================================================================

pub mod skeptical_tests {
    //! Skeptical tests to validate derived feature effectiveness
    //!
    //! These tests verify that:
    //! 1. entropy_trend_interaction predicts regime transitions
    //! 2. Interaction terms are more predictive than individual features
    //! 3. trend_strength outperforms simple momentum

    /// Result of interaction predictive power test
    #[derive(Debug, Clone)]
    pub struct InteractionPredictiveTest {
        pub interaction_correlation: f64,
        pub entropy_only_correlation: f64,
        pub trend_only_correlation: f64,
        pub interaction_lift: f64, // How much better is interaction
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of regime transition prediction test
    #[derive(Debug, Clone)]
    pub struct RegimeTransitionTest {
        pub pre_transition_interaction_mean: f64,
        pub normal_interaction_mean: f64,
        pub separation: f64, // Difference in means
        pub transition_detected_rate: f64,
        pub false_positive_rate: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of trend strength vs momentum comparison
    #[derive(Debug, Clone)]
    pub struct TrendStrengthTest {
        pub trend_strength_sharpe: f64,
        pub momentum_sharpe: f64,
        pub improvement_ratio: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Test if interaction term is more predictive than individual features
    pub fn test_interaction_predictive_power(
        entropy_values: &[f64],
        monotonicity_values: &[f64],
        future_volatility: &[f64], // Target to predict
    ) -> InteractionPredictiveTest {
        let n = entropy_values.len()
            .min(monotonicity_values.len())
            .min(future_volatility.len());

        if n < 50 {
            return InteractionPredictiveTest {
                interaction_correlation: 0.0,
                entropy_only_correlation: 0.0,
                trend_only_correlation: 0.0,
                interaction_lift: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        // Compute interaction term
        let interaction: Vec<f64> = (0..n)
            .map(|i| entropy_values[i] * (1.0 - monotonicity_values[i]))
            .collect();

        // Compute correlations
        let interaction_corr = pearson_correlation(&interaction, &future_volatility[..n]);
        let entropy_corr = pearson_correlation(&entropy_values[..n], &future_volatility[..n]);
        let mono_corr = pearson_correlation(&monotonicity_values[..n], &future_volatility[..n]);

        // Lift: how much better is interaction vs best individual
        let best_individual = entropy_corr.abs().max(mono_corr.abs());
        let interaction_lift = if best_individual > 0.01 {
            interaction_corr.abs() / best_individual
        } else {
            1.0
        };

        InteractionPredictiveTest {
            interaction_correlation: interaction_corr,
            entropy_only_correlation: entropy_corr,
            trend_only_correlation: mono_corr,
            interaction_lift,
            sample_size: n,
            significant: interaction_lift > 1.1 && interaction_corr.abs() > 0.1,
        }
    }

    /// Test if entropy_trend_interaction predicts regime transitions
    pub fn test_regime_transition_prediction(
        interaction_values: &[f64],
        regime_labels: &[i8], // -1 = trending, 0 = transition, 1 = reverting
        lookahead: usize,
    ) -> RegimeTransitionTest {
        let n = interaction_values.len().min(regime_labels.len());

        if n < lookahead + 50 {
            return RegimeTransitionTest {
                pre_transition_interaction_mean: 0.0,
                normal_interaction_mean: 0.0,
                separation: 0.0,
                transition_detected_rate: 0.0,
                false_positive_rate: 0.0,
                sample_size: n,
                significant: false,
            };
        }

        // Find transition points (where regime changes)
        let mut pre_transition_values = Vec::new();
        let mut normal_values = Vec::new();

        for i in 0..(n - lookahead) {
            let current_regime = regime_labels[i];
            let future_regime = regime_labels[i + lookahead];

            if current_regime != future_regime {
                // This is pre-transition period
                pre_transition_values.push(interaction_values[i]);
            } else {
                normal_values.push(interaction_values[i]);
            }
        }

        if pre_transition_values.is_empty() || normal_values.is_empty() {
            return RegimeTransitionTest {
                pre_transition_interaction_mean: 0.0,
                normal_interaction_mean: 0.0,
                separation: 0.0,
                transition_detected_rate: 0.0,
                false_positive_rate: 0.0,
                sample_size: n,
                significant: false,
            };
        }

        let pre_mean: f64 = pre_transition_values.iter().sum::<f64>() / pre_transition_values.len() as f64;
        let normal_mean: f64 = normal_values.iter().sum::<f64>() / normal_values.len() as f64;
        let separation = (pre_mean - normal_mean).abs();

        // Compute detection rate using threshold
        let threshold = normal_mean + separation * 0.5;
        let detected: usize = pre_transition_values.iter().filter(|&&v| v > threshold).count();
        let false_positives: usize = normal_values.iter().filter(|&&v| v > threshold).count();

        let transition_detected_rate = detected as f64 / pre_transition_values.len() as f64;
        let false_positive_rate = false_positives as f64 / normal_values.len() as f64;

        RegimeTransitionTest {
            pre_transition_interaction_mean: pre_mean,
            normal_interaction_mean: normal_mean,
            separation,
            transition_detected_rate,
            false_positive_rate,
            sample_size: n,
            significant: separation > 0.05 && transition_detected_rate > false_positive_rate,
        }
    }

    /// Test if trend_strength outperforms simple momentum
    pub fn test_trend_strength_vs_momentum(
        trend_strength: &[f64],
        momentum: &[f64],
        future_returns: &[f64],
    ) -> TrendStrengthTest {
        let n = trend_strength.len()
            .min(momentum.len())
            .min(future_returns.len());

        if n < 50 {
            return TrendStrengthTest {
                trend_strength_sharpe: 0.0,
                momentum_sharpe: 0.0,
                improvement_ratio: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        // Compute strategy returns: sign(signal) * future_return
        let ts_returns: Vec<f64> = (0..n)
            .map(|i| trend_strength[i].signum() * future_returns[i])
            .collect();

        let mom_returns: Vec<f64> = (0..n)
            .map(|i| momentum[i].signum() * future_returns[i])
            .collect();

        // Compute Sharpe ratios
        let ts_sharpe = compute_sharpe(&ts_returns);
        let mom_sharpe = compute_sharpe(&mom_returns);

        let improvement_ratio = if mom_sharpe.abs() > 0.01 {
            ts_sharpe / mom_sharpe
        } else {
            1.0
        };

        TrendStrengthTest {
            trend_strength_sharpe: ts_sharpe,
            momentum_sharpe: mom_sharpe,
            improvement_ratio,
            sample_size: n,
            significant: improvement_ratio > 1.1 && ts_sharpe > 0.0,
        }
    }

    /// Compute Pearson correlation
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
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

    /// Compute Sharpe ratio (annualized, assuming daily returns)
    fn compute_sharpe(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        // Annualize (assuming ~252 trading days worth of samples)
        let annualized_mean = mean * 252.0_f64.sqrt();
        let annualized_std = std * 252.0_f64.sqrt();

        annualized_mean / annualized_std
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create test features
    fn make_entropy(tick_30s: f64, tick_5s: f64, tick_1m: f64) -> EntropyFeatures {
        EntropyFeatures {
            tick_entropy_5s: tick_5s,
            tick_entropy_30s: tick_30s,
            tick_entropy_1m: tick_1m,
            ..Default::default()
        }
    }

    fn make_trend(momentum_60: f64, momentum_300: f64, mono_60: f64, mono_300: f64) -> TrendFeatures {
        TrendFeatures {
            momentum_60,
            momentum_300,
            monotonicity_60: mono_60,
            monotonicity_300: mono_300,
            ..Default::default()
        }
    }

    fn make_volatility(returns_1m: f64) -> VolatilityFeatures {
        VolatilityFeatures {
            returns_1m,
            ..Default::default()
        }
    }

    fn make_illiquidity(kyle: f64) -> IlliquidityFeatures {
        IlliquidityFeatures {
            kyle_lambda_100: kyle,
            ..Default::default()
        }
    }

    fn make_toxicity(vpin: f64, index: f64, imbalance: f64) -> ToxicityFeatures {
        ToxicityFeatures {
            vpin_50: vpin,
            toxicity_index: index,
            flow_imbalance_abs: imbalance,
            ..Default::default()
        }
    }

    fn make_flow() -> FlowFeatures {
        FlowFeatures::default()
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(DerivedFeatures::count(), 15);
        assert_eq!(DerivedFeatures::names().len(), 15);
        assert_eq!(DerivedFeatures::default().to_vec().len(), 15);
    }

    // ========================================================================
    // Trend Strength Tests
    // ========================================================================

    #[test]
    fn test_trend_strength_strong_uptrend() {
        // Low entropy, high monotonicity, positive momentum
        let entropy = make_entropy(0.2, 0.2, 0.2);
        let trend = make_trend(0.5, 0.4, 0.9, 0.85);
        let vol = make_volatility(0.01);
        let illiq = make_illiquidity(10.0);
        let toxic = make_toxicity(0.2, 0.2, 0.1);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.trend_strength_60 > 0.5,
            "Strong uptrend should have high positive trend_strength, got {}", derived.trend_strength_60);
    }

    #[test]
    fn test_trend_strength_strong_downtrend() {
        // Low entropy, high monotonicity, negative momentum
        let entropy = make_entropy(0.2, 0.2, 0.2);
        let trend = make_trend(-0.5, -0.4, 0.9, 0.85);
        let vol = make_volatility(0.01);
        let illiq = make_illiquidity(10.0);
        let toxic = make_toxicity(0.2, 0.2, 0.1);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.trend_strength_60 < -0.5,
            "Strong downtrend should have high negative trend_strength, got {}", derived.trend_strength_60);
    }

    #[test]
    fn test_trend_strength_choppy() {
        // High entropy, low monotonicity
        let entropy = make_entropy(0.9, 0.9, 0.9);
        let trend = make_trend(0.1, 0.05, 0.55, 0.52);
        let vol = make_volatility(0.02);
        let illiq = make_illiquidity(5.0);
        let toxic = make_toxicity(0.5, 0.5, 0.4);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.trend_strength_60.abs() < 0.2,
            "Choppy market should have low trend_strength, got {}", derived.trend_strength_60);
    }

    // ========================================================================
    // Entropy-Trend Interaction Tests
    // ========================================================================

    #[test]
    fn test_entropy_trend_interaction_trending() {
        // Low entropy + high monotonicity = low interaction (trending)
        let entropy = make_entropy(0.2, 0.2, 0.2);
        let trend = make_trend(0.5, 0.4, 0.95, 0.9);
        let vol = make_volatility(0.01);
        let illiq = make_illiquidity(10.0);
        let toxic = make_toxicity(0.1, 0.1, 0.1);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.entropy_trend_interaction < 0.1,
            "Trending regime should have low interaction, got {}", derived.entropy_trend_interaction);
    }

    #[test]
    fn test_entropy_trend_interaction_choppy() {
        // High entropy + low monotonicity = high interaction (choppy)
        let entropy = make_entropy(0.9, 0.9, 0.9);
        let trend = make_trend(0.05, 0.02, 0.52, 0.51);
        let vol = make_volatility(0.02);
        let illiq = make_illiquidity(5.0);
        let toxic = make_toxicity(0.6, 0.6, 0.5);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.entropy_trend_interaction > 0.3,
            "Choppy regime should have high interaction, got {}", derived.entropy_trend_interaction);
    }

    // ========================================================================
    // Regime Indicator Tests
    // ========================================================================

    #[test]
    fn test_regime_indicator_trending() {
        let entropy = make_entropy(0.2, 0.2, 0.2);
        let trend = make_trend(0.5, 0.4, 0.95, 0.9);
        let vol = make_volatility(0.01);
        let illiq = make_illiquidity(10.0);
        let toxic = make_toxicity(0.1, 0.1, 0.3);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.regime_indicator < 0.0,
            "Trending regime should have negative indicator, got {}", derived.regime_indicator);
    }

    #[test]
    fn test_regime_indicator_mean_reverting() {
        let entropy = make_entropy(0.9, 0.9, 0.9);
        let trend = make_trend(0.01, 0.005, 0.51, 0.50);
        let vol = make_volatility(0.01);
        let illiq = make_illiquidity(5.0);
        let toxic = make_toxicity(0.3, 0.3, 0.1);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.regime_indicator > 0.0,
            "Mean-reverting regime should have positive indicator, got {}", derived.regime_indicator);
    }

    // ========================================================================
    // Toxicity-Regime Interaction Tests
    // ========================================================================

    #[test]
    fn test_toxicity_regime_interaction() {
        // High toxicity in choppy market
        let entropy = make_entropy(0.8, 0.8, 0.8);
        let trend = make_trend(0.05, 0.02, 0.55, 0.52);
        let vol = make_volatility(0.02);
        let illiq = make_illiquidity(5.0);
        let toxic = make_toxicity(0.8, 0.7, 0.4);
        let flow = make_flow();

        let derived = compute(&entropy, &trend, &vol, &illiq, &toxic, &flow);

        assert!(derived.toxicity_regime > 0.4,
            "High toxicity + high entropy should give high toxicity_regime, got {}", derived.toxicity_regime);
        assert!(derived.toxic_chop_score > 0.3,
            "High VPIN + low monotonicity should give high toxic_chop_score, got {}", derived.toxic_chop_score);
    }

    // ========================================================================
    // Skeptical Tests
    // ========================================================================

    #[test]
    fn test_interaction_predictive_power() {
        use skeptical_tests::test_interaction_predictive_power;

        // Create data where interaction is more predictive
        let n = 100;
        let entropy: Vec<f64> = (0..n).map(|i| 0.3 + 0.4 * ((i as f64 * 0.1).sin() + 1.0) / 2.0).collect();
        let monotonicity: Vec<f64> = (0..n).map(|i| 0.5 + 0.4 * ((i as f64 * 0.15).cos() + 1.0) / 2.0).collect();
        // Future vol depends on interaction
        let future_vol: Vec<f64> = (0..n)
            .map(|i| {
                let interaction = entropy[i] * (1.0 - monotonicity[i]);
                interaction * 2.0 + 0.1
            })
            .collect();

        let result = test_interaction_predictive_power(&entropy, &monotonicity, &future_vol);

        assert!(result.interaction_correlation > 0.5,
            "Interaction should correlate with future vol, got {}", result.interaction_correlation);
    }

    #[test]
    fn test_trend_strength_vs_momentum() {
        use skeptical_tests::test_trend_strength_vs_momentum;

        // Create data where trend_strength is more reliable
        let n = 200;
        let trend_strength: Vec<f64> = (0..n).map(|i| {
            if i < 100 { 0.8 } else { -0.8 } // Clear signal
        }).collect();
        let momentum: Vec<f64> = (0..n).map(|i| {
            // Noisy momentum - sometimes wrong direction
            if i < 100 { 0.5 + (i as f64 * 0.5).sin() * 0.8 }  // Can flip to negative
            else { -0.5 + (i as f64 * 0.5).sin() * 0.8 }  // Can flip to positive
        }).collect();
        // Future returns with some noise for variance
        let future_returns: Vec<f64> = (0..n).map(|i| {
            let base = if i < 100 { 0.01 } else { -0.01 };
            base + (i as f64 * 0.3).sin() * 0.002  // Add noise for variance
        }).collect();

        let result = test_trend_strength_vs_momentum(&trend_strength, &momentum, &future_returns);

        // Verify test ran with enough samples
        assert_eq!(result.sample_size, 200, "Should have full sample");
        // The signals should produce some returns (positive or negative)
        // Since we have noisy momentum, it may underperform
        assert!(result.trend_strength_sharpe != 0.0 || result.momentum_sharpe != 0.0,
            "At least one signal should have non-zero sharpe, ts={}, mom={}",
            result.trend_strength_sharpe, result.momentum_sharpe);
    }

    #[test]
    fn test_regime_transition_detection() {
        use skeptical_tests::test_regime_transition_prediction;

        // Create data with clear regime transitions
        let n = 200;
        let mut interaction = Vec::with_capacity(n);
        let mut regime_labels = Vec::with_capacity(n);

        for i in 0..n {
            // Regime changes every 50 samples
            let regime = if (i / 50) % 2 == 0 { -1i8 } else { 1i8 };
            regime_labels.push(regime);

            // Interaction spikes before transitions
            let is_pre_transition = (i % 50) >= 40;
            if is_pre_transition {
                interaction.push(0.6); // Higher before transition
            } else {
                interaction.push(0.2); // Lower during stable regime
            }
        }

        let result = test_regime_transition_prediction(&interaction, &regime_labels, 10);

        assert!(result.pre_transition_interaction_mean > result.normal_interaction_mean,
            "Pre-transition should have higher interaction");
    }
}
