//! Composite Score Computation
//!
//! Combines individual regime features into accumulation/distribution scores.
//!
//! Accumulation signature:
//! - High absorption (volume absorbed without price rise)
//! - Negative divergence (price rises less than expected given buying)
//! - Elevated churn (two-sided trading)
//! - Price near range lows
//!
//! Distribution signature:
//! - High absorption (volume absorbed without price fall)
//! - Positive divergence (price falls less than expected given selling)
//! - Elevated churn (two-sided trading)
//! - Price near range highs

/// Compute accumulation score from regime features.
///
/// Higher score indicates stronger accumulation signature.
///
/// # Arguments
/// * `absorption_zscore` - Absorption z-score (high = accumulation)
/// * `divergence_zscore` - Divergence z-score (negative = accumulation)
/// * `churn_zscore` - Churn z-score (high = position transfer)
/// * `range_position` - Position in range [0, 1] (low = accumulation zone)
///
/// # Returns
/// Score in [0, 1] where 1 = strong accumulation signal
pub fn compute_accumulation_score(
    absorption_zscore: f64,
    divergence_zscore: f64,
    churn_zscore: f64,
    range_position: f64,
) -> f64 {
    // Absorption: high is good for accumulation
    let absorption_signal = sigmoid(absorption_zscore, 2.0);

    // Divergence: negative is good (price suppressed despite buying)
    let divergence_signal = sigmoid(-divergence_zscore, 2.0);

    // Churn: elevated is supportive
    let churn_signal = sigmoid(churn_zscore - 1.0, 2.0);

    // Range position: low is good (buying at range lows)
    let range_signal = 1.0 - range_position;

    // Weighted combination
    let weights = [0.35, 0.30, 0.15, 0.20];
    let signals = [absorption_signal, divergence_signal, churn_signal, range_signal];

    let score: f64 = weights.iter().zip(signals.iter()).map(|(w, s)| w * s).sum();

    score.clamp(0.0, 1.0)
}

/// Compute distribution score from regime features.
///
/// Higher score indicates stronger distribution signature.
///
/// # Arguments
/// * `absorption_zscore` - Absorption z-score (high = distribution)
/// * `divergence_zscore` - Divergence z-score (positive = distribution)
/// * `churn_zscore` - Churn z-score (high = position transfer)
/// * `range_position` - Position in range [0, 1] (high = distribution zone)
///
/// # Returns
/// Score in [0, 1] where 1 = strong distribution signal
pub fn compute_distribution_score(
    absorption_zscore: f64,
    divergence_zscore: f64,
    churn_zscore: f64,
    range_position: f64,
) -> f64 {
    // Absorption: high is good for distribution too
    let absorption_signal = sigmoid(absorption_zscore, 2.0);

    // Divergence: positive is good (price supported despite selling)
    let divergence_signal = sigmoid(divergence_zscore, 2.0);

    // Churn: elevated is supportive
    let churn_signal = sigmoid(churn_zscore - 1.0, 2.0);

    // Range position: high is good (selling at range highs)
    let range_signal = range_position;

    // Weighted combination
    let weights = [0.35, 0.30, 0.15, 0.20];
    let signals = [absorption_signal, divergence_signal, churn_signal, range_signal];

    let score: f64 = weights.iter().zip(signals.iter()).map(|(w, s)| w * s).sum();

    score.clamp(0.0, 1.0)
}

/// Sigmoid function for signal transformation.
///
/// Maps z-score to [0, 1] with configurable steepness.
fn sigmoid(x: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-steepness * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulation_strong_signal() {
        // Strong accumulation: high absorption, negative divergence,
        // elevated churn, price at range lows
        let score = compute_accumulation_score(
            2.5,  // High absorption
            -2.0, // Negative divergence (suppressed price)
            2.0,  // Elevated churn
            0.1,  // Near range low
        );

        assert!(score > 0.7, "Strong accumulation signal should score high");
    }

    #[test]
    fn test_accumulation_weak_signal() {
        // Weak accumulation: all metrics neutral
        let score = compute_accumulation_score(
            0.0, // Normal absorption
            0.0, // Normal divergence
            0.0, // Normal churn
            0.5, // Middle of range
        );

        assert!(score < 0.5, "Neutral metrics should score low");
    }

    #[test]
    fn test_distribution_strong_signal() {
        // Strong distribution: high absorption, positive divergence,
        // elevated churn, price at range highs
        let score = compute_distribution_score(
            2.5, // High absorption
            2.0, // Positive divergence (supported price)
            2.0, // Elevated churn
            0.9, // Near range high
        );

        assert!(score > 0.7, "Strong distribution signal should score high");
    }

    #[test]
    fn test_scores_mutually_exclusive() {
        // When accumulation is strong, distribution should be weak
        let acc_score = compute_accumulation_score(2.5, -2.0, 2.0, 0.1);
        let dist_score = compute_distribution_score(2.5, -2.0, 2.0, 0.1);

        // Divergence sign and range position should differentiate
        assert!(acc_score > dist_score);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0, 2.0) - 0.5).abs() < 0.01);
        assert!(sigmoid(3.0, 2.0) > 0.99);
        assert!(sigmoid(-3.0, 2.0) < 0.01);
    }
}
