//! Liquidity Heatmap Feature Extraction
//!
//! Discretises the price axis into 200 bins (10 bps each, ±10% from mid) and maps
//! liquidation positions into a spatial density field. Eight scalar features are
//! extracted from the heatmap per snapshot.
//!
//! # Features
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Nearest cluster distance** | Min |δ| where H(t,k) > τ | [0, 0.10] | Smaller = cascade closer |
//! | **Cluster mass ratio** | H(k*) / mean(H) | [1, +inf) | Higher = more concentrated |
//! | **Cascade chain length** | Consecutive bins > τ_chain within ±5% | [0, 100] | Longer = more domino potential |
//! | **Asymmetric cascade potential** | Normalised up vs down mass | [-1, 1] | >0 = short squeeze, <0 = long cascade |
//! | **Absorption capacity** | Book depth / liquidation pressure | [0, +inf) | <1 = insufficient damping |
//! | **Cluster velocity** | Δd_min / Δt | (-inf, +inf) | <0 = cluster approaching |
//! | **Mass-weighted distance** | Centre of liquidation gravity | [0, 0.10] | Lower = higher immediate risk |
//! | **Heatmap entropy** | Shannon entropy of mass distribution | [0, ln(K)] | Low = concentrated clusters |
//!
//! # References
//!
//! - Cont & Wagalath (2016) — Fire sales forensics
//! - Brunnermeier & Pedersen (2009) — Market liquidity and funding liquidity
//! - Caccioli et al. (2014) — Stability analysis of financial contagion
//!
//! See `docs/liquidity_heatmap_model.md` for full mathematical specification.

use super::liquidation::LiquidationPosition;

/// Number of bins in the heatmap
const DEFAULT_NUM_BINS: usize = 200;
/// Half-range in log-return space (±10%)
const DEFAULT_HALF_RANGE: f64 = 0.10;
/// Centre bin index
const CENTRE_BIN: usize = DEFAULT_NUM_BINS / 2;
/// 5% boundary in bins (for chain length and ACP)
const BINS_5PCT: usize = 50; // 0.05 / (2*0.10/200) = 50

/// Configuration for heatmap computation
#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    /// Number of bins (default 200)
    pub num_bins: usize,
    /// Half-range in log-return space (default 0.10)
    pub half_range: f64,
    /// Cluster detection threshold in USD (default $1M)
    pub cluster_threshold_usd: f64,
    /// Chain detection threshold in USD (default $100K — 1/10 of cluster)
    pub chain_threshold_usd: f64,
    /// Minimum position value filter in USD (default $1K)
    pub min_position_value: f64,
    /// Velocity lag in ticks (default 600 = 1 min at 100ms)
    pub velocity_lag_ticks: u64,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            num_bins: DEFAULT_NUM_BINS,
            half_range: DEFAULT_HALF_RANGE,
            cluster_threshold_usd: 1_000_000.0,
            chain_threshold_usd: 100_000.0,
            min_position_value: 1_000.0,
            velocity_lag_ticks: 600,
        }
    }
}

/// Stateful buffer for heatmap temporal features (Channel 4 and F6)
#[derive(Debug, Clone)]
pub struct HeatmapBuffer {
    /// Heatmap snapshot from 1 minute ago (for temporal change)
    lag_1min: Option<[f64; DEFAULT_NUM_BINS]>,
    /// Heatmap snapshot from 5 minutes ago (for temporal change)
    lag_5min: Option<[f64; DEFAULT_NUM_BINS]>,
    /// Previous nearest cluster distance (for F6 velocity)
    prev_nearest_dist: Option<f64>,
    /// Tick counter since creation
    tick_counter: u64,
    /// Configuration
    config: HeatmapConfig,
}

impl HeatmapBuffer {
    pub fn new(config: HeatmapConfig) -> Self {
        Self {
            lag_1min: None,
            lag_5min: None,
            prev_nearest_dist: None,
            tick_counter: 0,
            config,
        }
    }

    /// Update buffer state and compute features from current positions
    pub fn update_and_compute(
        &mut self,
        positions: &[LiquidationPosition],
        current_price: f64,
        bid_depth_2pct: f64,
        ask_depth_2pct: f64,
    ) -> HeatmapFeatures {
        if positions.is_empty() || current_price <= 0.0 {
            self.tick_counter += 1;
            return HeatmapFeatures::default();
        }

        // Build current heatmap
        let current = build_heatmap(positions, current_price, &self.config);

        // Extract features
        let features = extract_features(
            &current,
            self.lag_1min.as_ref(),
            self.prev_nearest_dist,
            bid_depth_2pct,
            ask_depth_2pct,
            &self.config,
        );

        // Update lagged snapshots
        // Store at 1-min intervals (every 600 ticks)
        if self.tick_counter % self.config.velocity_lag_ticks == 0 {
            // Shift: 1min lag becomes 5min lag (every 5th update = every 3000 ticks)
            if self.tick_counter % (self.config.velocity_lag_ticks * 5) == 0 {
                self.lag_5min = self.lag_1min;
            }
            self.lag_1min = Some(current);
            self.prev_nearest_dist = Some(features.hm_nearest_cluster_dist);
        }

        self.tick_counter += 1;
        features
    }
}

/// Heatmap features: 8 scalar features per snapshot
#[derive(Debug, Clone)]
pub struct HeatmapFeatures {
    /// F1: Distance to nearest cluster exceeding threshold (log-return units)
    pub hm_nearest_cluster_dist: f64,
    /// F2: Peak bin density / mean bin density (dimensionless)
    pub hm_cluster_mass_ratio: f64,
    /// F3: Max consecutive bins above chain threshold within ±5% (bin count)
    pub hm_cascade_chain_length: f64,
    /// F4: Normalised upward vs downward liquidation mass within ±5%
    pub hm_asymmetric_cascade_pot: f64,
    /// F5: Order book depth / downward liquidation pressure (dimensionless)
    pub hm_absorption_capacity: f64,
    /// F6: Rate of change of nearest cluster distance (log-return/tick)
    pub hm_cluster_velocity: f64,
    /// F7: Mass-weighted mean distance from mid (log-return units)
    pub hm_mass_weighted_distance: f64,
    /// F8: Shannon entropy of normalised mass distribution
    pub hm_heatmap_entropy: f64,
}

impl Default for HeatmapFeatures {
    fn default() -> Self {
        Self {
            hm_nearest_cluster_dist: DEFAULT_HALF_RANGE, // max distance when no data
            hm_cluster_mass_ratio: 1.0,
            hm_cascade_chain_length: 0.0,
            hm_asymmetric_cascade_pot: 0.0,
            hm_absorption_capacity: 100.0, // clamped max when no liquidation pressure
            hm_cluster_velocity: 0.0,
            hm_mass_weighted_distance: DEFAULT_HALF_RANGE,
            hm_heatmap_entropy: 0.0,
        }
    }
}

impl HeatmapFeatures {
    pub fn count() -> usize {
        8
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "hm_nearest_cluster_dist",
            "hm_cluster_mass_ratio",
            "hm_cascade_chain_length",
            "hm_asymmetric_cascade_pot",
            "hm_absorption_capacity",
            "hm_cluster_velocity",
            "hm_mass_weighted_distance",
            "hm_heatmap_entropy",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.hm_nearest_cluster_dist,
            self.hm_cluster_mass_ratio,
            self.hm_cascade_chain_length,
            self.hm_asymmetric_cascade_pot,
            self.hm_absorption_capacity,
            self.hm_cluster_velocity,
            self.hm_mass_weighted_distance,
            self.hm_heatmap_entropy,
        ]
    }
}

// ============================================================================
// Heatmap Construction
// ============================================================================

/// Build the heatmap density array from positions.
///
/// H(k) = Σ V_i · 𝟙[ℓ_i ∈ P_k(mid)]
///
/// Uses log-return offsets: δ_k = -R + (k + 0.5) · Δ
fn build_heatmap(
    positions: &[LiquidationPosition],
    current_price: f64,
    config: &HeatmapConfig,
) -> [f64; DEFAULT_NUM_BINS] {
    let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
    let bin_width = 2.0 * config.half_range / config.num_bins as f64;

    for pos in positions {
        if pos.position_value_usd < config.min_position_value {
            continue;
        }

        // Log-return offset from mid
        if pos.liquidation_price <= 0.0 {
            continue;
        }
        let delta = (pos.liquidation_price / current_price).ln();

        // Map to bin index: k = floor((δ + R) / Δ)
        let k = ((delta + config.half_range) / bin_width) as i64;

        if k >= 0 && (k as usize) < config.num_bins {
            heatmap[k as usize] += pos.position_value_usd;
        }
    }

    heatmap
}

/// Compute the log-return offset of bin k's centre
#[inline]
fn bin_centre_delta(k: usize, half_range: f64, num_bins: usize) -> f64 {
    let bin_width = 2.0 * half_range / num_bins as f64;
    -half_range + (k as f64 + 0.5) * bin_width
}

// ============================================================================
// Feature Extraction
// ============================================================================

/// Extract 8 scalar features from the heatmap
fn extract_features(
    heatmap: &[f64; DEFAULT_NUM_BINS],
    _lag_1min: Option<&[f64; DEFAULT_NUM_BINS]>,
    prev_nearest_dist: Option<f64>,
    bid_depth_2pct: f64,
    ask_depth_2pct: f64,
    config: &HeatmapConfig,
) -> HeatmapFeatures {
    let half_range = config.half_range;
    let num_bins = config.num_bins;

    // F1: Nearest cluster distance
    let nearest_cluster_dist = compute_nearest_cluster_dist(heatmap, config);

    // F2: Cluster mass ratio
    let cluster_mass_ratio = compute_cluster_mass_ratio(heatmap);

    // F3: Cascade chain length
    let cascade_chain_length = compute_cascade_chain_length(heatmap, config);

    // F4: Asymmetric cascade potential
    let asymmetric_cascade_pot = compute_asymmetric_cascade_pot(heatmap);

    // F5: Absorption capacity (use worst-case direction)
    let absorption_capacity = compute_absorption_capacity(
        heatmap,
        bid_depth_2pct,
        ask_depth_2pct,
        half_range,
        num_bins,
    );

    // F6: Cluster velocity (finite difference)
    let cluster_velocity = match prev_nearest_dist {
        Some(prev) => (nearest_cluster_dist - prev) / config.velocity_lag_ticks as f64,
        None => 0.0,
    };

    // F7: Mass-weighted distance
    let mass_weighted_distance = compute_mass_weighted_distance(heatmap, half_range, num_bins);

    // F8: Heatmap entropy
    let heatmap_entropy = compute_heatmap_entropy(heatmap);

    // Clamp absorption to prevent infinity in downstream models
    let absorption_clamped = if absorption_capacity.is_finite() {
        absorption_capacity.min(100.0)
    } else {
        100.0
    };

    HeatmapFeatures {
        hm_nearest_cluster_dist: nearest_cluster_dist,
        hm_cluster_mass_ratio: cluster_mass_ratio,
        hm_cascade_chain_length: cascade_chain_length,
        hm_asymmetric_cascade_pot: asymmetric_cascade_pot,
        hm_absorption_capacity: absorption_clamped,
        hm_cluster_velocity: cluster_velocity,
        hm_mass_weighted_distance: mass_weighted_distance,
        hm_heatmap_entropy: heatmap_entropy,
    }
}

/// F1: min |δ_k| where H(t,k) > τ_cluster
fn compute_nearest_cluster_dist(heatmap: &[f64; DEFAULT_NUM_BINS], config: &HeatmapConfig) -> f64 {
    let mut min_dist = config.half_range;

    for k in 0..config.num_bins {
        if heatmap[k] > config.cluster_threshold_usd {
            let delta = bin_centre_delta(k, config.half_range, config.num_bins);
            let dist = delta.abs();
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }

    min_dist
}

/// F2: H(k*) / mean(H)
fn compute_cluster_mass_ratio(heatmap: &[f64; DEFAULT_NUM_BINS]) -> f64 {
    let max_val = heatmap.iter().cloned().fold(0.0f64, f64::max);
    let sum: f64 = heatmap.iter().sum();
    let mean = sum / DEFAULT_NUM_BINS as f64;

    if mean > 1.0 {
        max_val / mean
    } else {
        1.0 // no meaningful mass
    }
}

/// F3: Max consecutive bins above τ_chain within ±5% of mid
fn compute_cascade_chain_length(heatmap: &[f64; DEFAULT_NUM_BINS], config: &HeatmapConfig) -> f64 {
    let start = CENTRE_BIN.saturating_sub(BINS_5PCT);
    let end = (CENTRE_BIN + BINS_5PCT).min(DEFAULT_NUM_BINS);

    let mut max_run = 0u32;
    let mut current_run = 0u32;

    for k in start..end {
        if heatmap[k] > config.chain_threshold_usd {
            current_run += 1;
            if current_run > max_run {
                max_run = current_run;
            }
        } else {
            current_run = 0;
        }
    }

    max_run as f64
}

/// F4: (Σ H(δ>0) - Σ H(δ<0)) / (Σ H(|δ|≤5%) + ε) within ±5% of mid
fn compute_asymmetric_cascade_pot(heatmap: &[f64; DEFAULT_NUM_BINS]) -> f64 {
    let mut mass_above = 0.0f64; // bins above centre (short liquidations, upward cascade)
    let mut mass_below = 0.0f64; // bins below centre (long liquidations, downward cascade)

    let start = CENTRE_BIN.saturating_sub(BINS_5PCT);
    let end = (CENTRE_BIN + BINS_5PCT).min(DEFAULT_NUM_BINS);

    for k in start..CENTRE_BIN {
        mass_below += heatmap[k];
    }
    for k in CENTRE_BIN..end {
        mass_above += heatmap[k];
    }

    let total = mass_above + mass_below + 1.0; // ε = 1 USD
    (mass_above - mass_below) / total
}

/// F5: min(bid_depth/downward_mass, ask_depth/upward_mass)
fn compute_absorption_capacity(
    heatmap: &[f64; DEFAULT_NUM_BINS],
    bid_depth_2pct: f64,
    ask_depth_2pct: f64,
    half_range: f64,
    num_bins: usize,
) -> f64 {
    let bin_width = 2.0 * half_range / num_bins as f64;
    // 2% range in bins below mid
    let bins_2pct = (0.02 / bin_width) as usize;

    // Downward liquidation mass (longs liquidated, bins below mid within 2%)
    let down_start = CENTRE_BIN.saturating_sub(bins_2pct);
    let mut down_mass = 0.0f64;
    for k in down_start..CENTRE_BIN {
        down_mass += heatmap[k];
    }

    // Upward liquidation mass (shorts liquidated, bins above mid within 2%)
    let up_end = (CENTRE_BIN + bins_2pct).min(DEFAULT_NUM_BINS);
    let mut up_mass = 0.0f64;
    for k in CENTRE_BIN..up_end {
        up_mass += heatmap[k];
    }

    let down_absorption = if down_mass > 1.0 {
        bid_depth_2pct / down_mass
    } else {
        f64::INFINITY
    };

    let up_absorption = if up_mass > 1.0 {
        ask_depth_2pct / up_mass
    } else {
        f64::INFINITY
    };

    // Worst-case (minimum) absorption
    down_absorption.min(up_absorption)
}

/// F7: Σ |δ_k| · H(k) / Σ H(k)
fn compute_mass_weighted_distance(
    heatmap: &[f64; DEFAULT_NUM_BINS],
    half_range: f64,
    num_bins: usize,
) -> f64 {
    let mut weighted_sum = 0.0f64;
    let mut total_mass = 0.0f64;

    for k in 0..num_bins {
        let delta = bin_centre_delta(k, half_range, num_bins).abs();
        weighted_sum += delta * heatmap[k];
        total_mass += heatmap[k];
    }

    if total_mass > 1.0 {
        weighted_sum / total_mass
    } else {
        half_range // no mass → return max distance
    }
}

/// F8: -Σ p_k · ln(p_k) where p_k = H(k) / Σ H
fn compute_heatmap_entropy(heatmap: &[f64; DEFAULT_NUM_BINS]) -> f64 {
    let total: f64 = heatmap.iter().sum();
    if total <= 1.0 {
        return 0.0; // no meaningful mass
    }

    let mut entropy = 0.0f64;
    for &h in heatmap.iter() {
        if h > 0.0 {
            let p = h / total;
            entropy -= p * p.ln();
        }
    }

    entropy
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> HeatmapConfig {
        HeatmapConfig::default()
    }

    fn make_position(value: f64, liq_price: f64, is_long: bool) -> LiquidationPosition {
        LiquidationPosition {
            position_value_usd: value,
            liquidation_price: liq_price,
            is_long,
            entry_price: if is_long {
                liq_price * 1.5
            } else {
                liq_price * 0.67
            },
        }
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(HeatmapFeatures::count(), 8);
        assert_eq!(HeatmapFeatures::names().len(), 8);
        assert_eq!(HeatmapFeatures::default().to_vec().len(), 8);
    }

    #[test]
    fn test_feature_names_prefix() {
        for name in HeatmapFeatures::names() {
            assert!(
                name.starts_with("hm_"),
                "Feature name should start with hm_: {}",
                name
            );
        }
    }

    // ========================================================================
    // Heatmap Construction Tests
    // ========================================================================

    #[test]
    fn test_empty_positions() {
        let config = make_config();
        let heatmap = build_heatmap(&[], 100_000.0, &config);
        assert!(heatmap.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_single_position_binning() {
        let config = make_config();
        let mid = 100_000.0;
        // Position with liq price 1% below mid (long position)
        let liq_price = mid * 0.99; // -1% → δ ≈ -0.01
        let positions = vec![make_position(500_000.0, liq_price, true)];

        let heatmap = build_heatmap(&positions, mid, &config);

        // δ = ln(0.99) ≈ -0.01005
        // bin = floor((-0.01005 + 0.10) / 0.001) = floor(89.95) = 89
        let total: f64 = heatmap.iter().sum();
        assert!(
            (total - 500_000.0).abs() < 1e-6,
            "Total mass should be 500K"
        );

        // Verify it landed in the right region (bins 85-95, around -1%)
        let expected_bin = ((-0.01005 + 0.10) / 0.001) as usize;
        assert!(
            heatmap[expected_bin] > 0.0,
            "Position should be in bin {}",
            expected_bin
        );
    }

    #[test]
    fn test_position_below_min_value_filtered() {
        let config = make_config(); // min_position_value = 1000
        let positions = vec![make_position(500.0, 99_000.0, true)]; // Below threshold
        let heatmap = build_heatmap(&positions, 100_000.0, &config);
        assert!(
            heatmap.iter().all(|&v| v == 0.0),
            "Sub-threshold positions should be filtered"
        );
    }

    // ========================================================================
    // F1: Nearest Cluster Distance
    // ========================================================================

    #[test]
    fn test_nearest_cluster_no_cluster() {
        let config = make_config();
        let heatmap = [0.0f64; DEFAULT_NUM_BINS];
        let dist = compute_nearest_cluster_dist(&heatmap, &config);
        assert!(
            (dist - config.half_range).abs() < 1e-10,
            "No cluster → max distance"
        );
    }

    #[test]
    fn test_nearest_cluster_at_2pct() {
        let config = make_config();
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // Place $2M cluster at +2% (bin ≈ 120)
        let bin_2pct = ((0.02 + 0.10) / 0.001) as usize; // bin 120
        heatmap[bin_2pct] = 2_000_000.0;

        let dist = compute_nearest_cluster_dist(&heatmap, &config);
        // Centre of bin 120: δ = -0.10 + (120 + 0.5)*0.001 = 0.0205
        let expected = bin_centre_delta(bin_2pct, config.half_range, config.num_bins).abs();
        assert!((dist - expected).abs() < 1e-10);
        assert!((dist - 0.0205).abs() < 0.001, "Should be ~2% away");
    }

    // ========================================================================
    // F2: Cluster Mass Ratio
    // ========================================================================

    #[test]
    fn test_mass_ratio_uniform() {
        let heatmap = [100.0f64; DEFAULT_NUM_BINS];
        let ratio = compute_cluster_mass_ratio(&heatmap);
        assert!(
            (ratio - 1.0).abs() < 1e-10,
            "Uniform distribution → ratio = 1"
        );
    }

    #[test]
    fn test_mass_ratio_concentrated() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        heatmap[50] = 1_000_000.0;
        let ratio = compute_cluster_mass_ratio(&heatmap);
        // mean = 1M / 200 = 5K, ratio = 1M / 5K = 200
        assert!(
            (ratio - 200.0).abs() < 1e-6,
            "Single bin → ratio = num_bins"
        );
    }

    // ========================================================================
    // F3: Cascade Chain Length
    // ========================================================================

    #[test]
    fn test_chain_length_no_chain() {
        let config = make_config();
        let heatmap = [0.0f64; DEFAULT_NUM_BINS];
        let len = compute_cascade_chain_length(&heatmap, &config);
        assert_eq!(len, 0.0);
    }

    #[test]
    fn test_chain_length_five_bins() {
        let config = make_config();
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // 5 consecutive bins above threshold near mid
        for k in CENTRE_BIN..CENTRE_BIN + 5 {
            heatmap[k] = 200_000.0; // > chain_threshold (100K)
        }
        let len = compute_cascade_chain_length(&heatmap, &config);
        assert_eq!(len, 5.0);
    }

    #[test]
    fn test_chain_ignores_outside_5pct() {
        let config = make_config();
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // 10 consecutive bins far from mid (outside ±5%)
        for k in 0..10 {
            heatmap[k] = 200_000.0;
        }
        let len = compute_cascade_chain_length(&heatmap, &config);
        assert_eq!(len, 0.0, "Chain outside ±5% should be ignored");
    }

    // ========================================================================
    // F4: Asymmetric Cascade Potential
    // ========================================================================

    #[test]
    fn test_acp_symmetric() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // Equal mass above and below
        heatmap[CENTRE_BIN - 10] = 1_000_000.0;
        heatmap[CENTRE_BIN + 10] = 1_000_000.0;
        let acp = compute_asymmetric_cascade_pot(&heatmap);
        assert!(acp.abs() < 0.01, "Symmetric → ACP ≈ 0, got {}", acp);
    }

    #[test]
    fn test_acp_upward_bias() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // More mass above mid (short liquidations → short squeeze)
        heatmap[CENTRE_BIN + 10] = 2_000_000.0;
        heatmap[CENTRE_BIN - 10] = 500_000.0;
        let acp = compute_asymmetric_cascade_pot(&heatmap);
        assert!(acp > 0.0, "More upward mass → positive ACP, got {}", acp);
    }

    #[test]
    fn test_acp_bounded() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        heatmap[CENTRE_BIN + 5] = 10_000_000.0; // all mass above
        let acp = compute_asymmetric_cascade_pot(&heatmap);
        assert!(
            acp > 0.0 && acp <= 1.0,
            "ACP should be in (0, 1], got {}",
            acp
        );
    }

    // ========================================================================
    // F5: Absorption Capacity
    // ========================================================================

    #[test]
    fn test_absorption_no_liquidation_mass() {
        let heatmap = [0.0f64; DEFAULT_NUM_BINS];
        let absorption = compute_absorption_capacity(&heatmap, 1e6, 1e6, 0.10, 200);
        assert!(
            absorption.is_infinite(),
            "No liquidation mass → infinite absorption"
        );
    }

    #[test]
    fn test_absorption_thin_book() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        // $5M downward liquidation mass within 2%
        for k in (CENTRE_BIN - 20)..CENTRE_BIN {
            heatmap[k] = 250_000.0; // 20 bins × 250K = $5M
        }
        let absorption =
            compute_absorption_capacity(&heatmap, 1_000_000.0, 10_000_000.0, 0.10, 200);
        // bid_depth ($1M) / down_mass ($5M) = 0.2
        assert!(
            (absorption - 0.2).abs() < 0.01,
            "Thin book → low absorption, got {}",
            absorption
        );
    }

    // ========================================================================
    // F6: Cluster Velocity
    // ========================================================================

    #[test]
    fn test_velocity_warmup() {
        let features = extract_features(
            &[0.0; DEFAULT_NUM_BINS],
            None, // no lag
            None, // no prev distance
            1e6,
            1e6,
            &make_config(),
        );
        assert_eq!(
            features.hm_cluster_velocity, 0.0,
            "No previous → velocity = 0"
        );
    }

    // ========================================================================
    // F7: Mass-Weighted Distance
    // ========================================================================

    #[test]
    fn test_mwd_no_mass() {
        let heatmap = [0.0f64; DEFAULT_NUM_BINS];
        let mwd = compute_mass_weighted_distance(&heatmap, 0.10, 200);
        assert!((mwd - 0.10).abs() < 1e-10, "No mass → max distance");
    }

    #[test]
    fn test_mwd_concentrated_at_mid() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        heatmap[CENTRE_BIN] = 1_000_000.0;
        let mwd = compute_mass_weighted_distance(&heatmap, 0.10, 200);
        // Centre bin delta ≈ 0.0005 (half a bin from zero)
        assert!(mwd < 0.001, "Mass at centre → MWD near 0, got {}", mwd);
    }

    // ========================================================================
    // F8: Heatmap Entropy
    // ========================================================================

    #[test]
    fn test_entropy_single_bin() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        heatmap[50] = 1_000_000.0;
        let entropy = compute_heatmap_entropy(&heatmap);
        assert!(entropy.abs() < 1e-10, "Single bin → entropy = 0");
    }

    #[test]
    fn test_entropy_uniform() {
        let heatmap = [1_000.0f64; DEFAULT_NUM_BINS];
        let entropy = compute_heatmap_entropy(&heatmap);
        let max_entropy = (DEFAULT_NUM_BINS as f64).ln();
        assert!(
            (entropy - max_entropy).abs() < 1e-6,
            "Uniform → max entropy (ln(K)), got {}",
            entropy
        );
    }

    #[test]
    fn test_entropy_two_bins() {
        let mut heatmap = [0.0f64; DEFAULT_NUM_BINS];
        heatmap[50] = 500_000.0;
        heatmap[150] = 500_000.0;
        let entropy = compute_heatmap_entropy(&heatmap);
        let expected = 2.0f64.ln(); // ln(2)
        assert!(
            (entropy - expected).abs() < 1e-6,
            "Two equal bins → ln(2), got {}",
            entropy
        );
    }

    // ========================================================================
    // Buffer Integration Tests
    // ========================================================================

    #[test]
    fn test_buffer_warmup_produces_features() {
        let config = make_config();
        let mut buffer = HeatmapBuffer::new(config);
        let mid = 100_000.0;
        let positions = vec![
            make_position(2_000_000.0, mid * 0.97, true), // 3% below
            make_position(1_500_000.0, mid * 1.02, false), // 2% above
        ];

        let features = buffer.update_and_compute(&positions, mid, 5_000_000.0, 5_000_000.0);

        // Should have valid features even on first tick (except velocity)
        assert!(
            features.hm_nearest_cluster_dist < 0.10,
            "Should detect cluster"
        );
        assert!(
            features.hm_cluster_mass_ratio > 1.0,
            "Should have concentrated mass"
        );
        assert_eq!(
            features.hm_cluster_velocity, 0.0,
            "No velocity on first tick"
        );
    }

    #[test]
    fn test_buffer_velocity_after_warmup() {
        let mut config = make_config();
        config.velocity_lag_ticks = 2; // speed up for test
        let mut buffer = HeatmapBuffer::new(config);
        let mid = 100_000.0;

        // Tick 0: cluster at 3% below
        let pos1 = vec![make_position(2_000_000.0, mid * 0.97, true)];
        let _ = buffer.update_and_compute(&pos1, mid, 5e6, 5e6);

        // Tick 1: same position
        let _ = buffer.update_and_compute(&pos1, mid, 5e6, 5e6);

        // Tick 2: velocity lag boundary — prev_nearest_dist gets stored
        let features = buffer.update_and_compute(&pos1, mid, 5e6, 5e6);

        // After 3 ticks with velocity_lag=2, the lag snapshot fires at tick 0 and tick 2.
        // velocity should now be computable (distance hasn't changed → velocity ≈ 0)
        // The exact value depends on when prev_nearest_dist was stored
        assert!(
            features.hm_cluster_velocity.is_finite(),
            "Velocity should be finite after warmup"
        );
    }

    #[test]
    fn test_default_features_are_safe() {
        let features = HeatmapFeatures::default();
        let vec = features.to_vec();
        for (i, &v) in vec.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Default feature {} should be finite, got {}",
                i,
                v
            );
        }
    }
}
