//! Alert Condition Detection
//!
//! Checks features against configurable thresholds and generates alerts.
//! Implements cooldown to prevent alert spam.

use std::collections::HashMap;

use crate::features::Features;
use crate::redis_publisher::{AlertTrigger, AlertType, AlertSeverity};

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Whale flow z-score threshold for accumulation alert
    pub whale_accumulation_zscore: f64,
    /// Whale flow z-score threshold for distribution alert
    pub whale_distribution_zscore: f64,
    /// Liquidation cascade probability threshold
    pub cascade_probability: f64,
    /// Entropy drop threshold (below this = predictable market)
    pub entropy_low: f64,
    /// Regime clarity threshold for regime change alert
    pub regime_clarity: f64,
    /// Cooldown between same alert type (milliseconds)
    pub cooldown_ms: u64,
    /// Enable/disable specific alert types
    pub enabled_alerts: EnabledAlerts,
}

#[derive(Debug, Clone)]
pub struct EnabledAlerts {
    pub whale_accumulation: bool,
    pub whale_distribution: bool,
    pub liquidation_cluster: bool,
    pub regime_change: bool,
    pub entropy_drop: bool,
}

impl Default for EnabledAlerts {
    fn default() -> Self {
        Self {
            whale_accumulation: true,
            whale_distribution: true,
            liquidation_cluster: true,
            regime_change: true,
            entropy_drop: true,
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            whale_accumulation_zscore: 2.0,
            whale_distribution_zscore: -2.0,
            cascade_probability: 0.7,
            entropy_low: 0.5,
            regime_clarity: 0.6,
            cooldown_ms: 300_000, // 5 minutes
            enabled_alerts: EnabledAlerts::default(),
        }
    }
}

impl AlertConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        Self {
            whale_accumulation_zscore: std::env::var("ALERT_WHALE_ACC_ZSCORE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.0),
            whale_distribution_zscore: std::env::var("ALERT_WHALE_DIST_ZSCORE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(-2.0),
            cascade_probability: std::env::var("ALERT_CASCADE_PROB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            entropy_low: std::env::var("ALERT_ENTROPY_LOW")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.5),
            regime_clarity: std::env::var("ALERT_REGIME_CLARITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.6),
            cooldown_ms: std::env::var("ALERT_COOLDOWN_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300_000),
            enabled_alerts: EnabledAlerts::default(),
        }
    }
}

/// Alert state tracker (for cooldowns)
pub struct AlertTracker {
    config: AlertConfig,
    /// Map of alert_key -> last_triggered_timestamp_ms
    last_alerts: HashMap<String, u64>,
    /// Track previous regime to detect changes
    last_regime: HashMap<String, String>,
}

impl AlertTracker {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            last_alerts: HashMap::new(),
            last_regime: HashMap::new(),
        }
    }

    /// Check features and return any triggered alerts
    pub fn check(&mut self, features: &Features, symbol: &str, timestamp_ms: u64) -> Vec<AlertTrigger> {
        let mut alerts = Vec::new();

        // Check whale accumulation
        if self.config.enabled_alerts.whale_accumulation {
            if let Some(ref wf) = features.whale_flow {
                if wf.whale_flow_normalized_1h >= self.config.whale_accumulation_zscore {
                    if self.can_fire("whale_acc", symbol, timestamp_ms) {
                        alerts.push(AlertTrigger {
                            timestamp_ms,
                            symbol: symbol.to_string(),
                            alert_type: AlertType::WhaleAccumulation,
                            severity: if wf.whale_flow_normalized_1h >= 3.0 {
                                AlertSeverity::Critical
                            } else {
                                AlertSeverity::Warning
                            },
                            message: format!(
                                "Whale accumulation: z={:.2}, flow=${:.0}K/1h",
                                wf.whale_flow_normalized_1h,
                                wf.whale_net_flow_1h / 1000.0
                            ),
                            data: serde_json::json!({
                                "flow_1h": wf.whale_net_flow_1h,
                                "flow_zscore": wf.whale_flow_normalized_1h,
                                "flow_4h": wf.whale_net_flow_4h,
                                "flow_24h": wf.whale_net_flow_24h,
                                "intensity": wf.whale_flow_intensity,
                            }),
                        });
                        self.mark_fired("whale_acc", symbol, timestamp_ms);
                    }
                }
            }
        }

        // Check whale distribution
        if self.config.enabled_alerts.whale_distribution {
            if let Some(ref wf) = features.whale_flow {
                if wf.whale_flow_normalized_1h <= self.config.whale_distribution_zscore {
                    if self.can_fire("whale_dist", symbol, timestamp_ms) {
                        alerts.push(AlertTrigger {
                            timestamp_ms,
                            symbol: symbol.to_string(),
                            alert_type: AlertType::WhaleDistribution,
                            severity: if wf.whale_flow_normalized_1h <= -3.0 {
                                AlertSeverity::Critical
                            } else {
                                AlertSeverity::Warning
                            },
                            message: format!(
                                "Whale distribution: z={:.2}, flow=${:.0}K/1h",
                                wf.whale_flow_normalized_1h,
                                wf.whale_net_flow_1h / 1000.0
                            ),
                            data: serde_json::json!({
                                "flow_1h": wf.whale_net_flow_1h,
                                "flow_zscore": wf.whale_flow_normalized_1h,
                                "flow_4h": wf.whale_net_flow_4h,
                                "flow_24h": wf.whale_net_flow_24h,
                            }),
                        });
                        self.mark_fired("whale_dist", symbol, timestamp_ms);
                    }
                }
            }
        }

        // Check regime change
        if self.config.enabled_alerts.regime_change {
            if let Some(ref regime) = features.regime {
                if regime.regime_clarity() >= self.config.regime_clarity {
                    let current_regime = if regime.accumulation_score > regime.distribution_score {
                        "ACCUMULATION"
                    } else {
                        "DISTRIBUTION"
                    };

                    let regime_key = format!("regime:{}", symbol);
                    let previous_regime = self.last_regime.get(&regime_key).map(|s| s.as_str());

                    // Only alert if regime changed
                    let regime_changed = previous_regime
                        .map(|prev| prev != current_regime)
                        .unwrap_or(true); // First detection counts as change

                    if regime_changed && self.can_fire("regime", symbol, timestamp_ms) {
                        alerts.push(AlertTrigger {
                            timestamp_ms,
                            symbol: symbol.to_string(),
                            alert_type: AlertType::RegimeChange,
                            severity: AlertSeverity::Info,
                            message: format!(
                                "Regime: {} (clarity={:.0}%, range_pos={:.0}%)",
                                current_regime,
                                regime.regime_clarity() * 100.0,
                                regime.range_position_24h * 100.0
                            ),
                            data: serde_json::json!({
                                "regime": current_regime,
                                "previous_regime": previous_regime,
                                "accumulation_score": regime.accumulation_score,
                                "distribution_score": regime.distribution_score,
                                "clarity": regime.regime_clarity(),
                                "range_position_24h": regime.range_position_24h,
                                "absorption_zscore": regime.absorption_zscore,
                                "churn_zscore": regime.churn_zscore,
                            }),
                        });
                        self.mark_fired("regime", symbol, timestamp_ms);
                    }

                    // Update tracked regime
                    self.last_regime.insert(regime_key, current_regime.to_string());
                }
            }
        }

        // Check liquidation cascade risk
        // High asymmetry indicates concentrated liquidation risk in one direction
        if self.config.enabled_alerts.liquidation_cluster {
            if let Some(ref liq) = features.liquidation_risk {
                // Check for extreme asymmetry (>3x difference in directional risk)
                let asymmetry_abs = liq.liquidation_asymmetry.abs();
                if asymmetry_abs >= 3.0 || asymmetry_abs <= 0.33 {
                    if self.can_fire("cascade", symbol, timestamp_ms) {
                        let direction = if liq.liquidation_asymmetry > 1.0 {
                            "UPWARD (short squeeze risk)"
                        } else {
                            "DOWNWARD (long squeeze risk)"
                        };

                        alerts.push(AlertTrigger {
                            timestamp_ms,
                            symbol: symbol.to_string(),
                            alert_type: AlertType::LiquidationCluster,
                            severity: AlertSeverity::Critical,
                            message: format!(
                                "Liquidation asymmetry: {:.2}x, {}",
                                liq.liquidation_asymmetry,
                                direction
                            ),
                            data: serde_json::json!({
                                "asymmetry": liq.liquidation_asymmetry,
                                "direction": direction,
                                "risk_above_1pct": liq.liquidation_risk_above_1pct,
                                "risk_below_1pct": liq.liquidation_risk_below_1pct,
                                "risk_above_5pct": liq.liquidation_risk_above_5pct,
                                "risk_below_5pct": liq.liquidation_risk_below_5pct,
                            }),
                        });
                        self.mark_fired("cascade", symbol, timestamp_ms);
                    }
                }
            }
        }

        // Check entropy drop (market becoming predictable)
        if self.config.enabled_alerts.entropy_drop {
            if features.entropy.tick_entropy_1m < self.config.entropy_low {
                if self.can_fire("entropy", symbol, timestamp_ms) {
                    alerts.push(AlertTrigger {
                        timestamp_ms,
                        symbol: symbol.to_string(),
                        alert_type: AlertType::EntropyDrop,
                        severity: AlertSeverity::Info,
                        message: format!(
                            "Low entropy: {:.3} (market predictable)",
                            features.entropy.tick_entropy_1m
                        ),
                        data: serde_json::json!({
                            "entropy_1m": features.entropy.tick_entropy_1m,
                            "entropy_5s": features.entropy.tick_entropy_5s,
                            "entropy_30s": features.entropy.tick_entropy_30s,
                            "permutation_returns_16": features.entropy.permutation_returns_16,
                        }),
                    });
                    self.mark_fired("entropy", symbol, timestamp_ms);
                }
            }
        }

        alerts
    }

    /// Check if alert can fire (respects cooldown)
    fn can_fire(&self, alert_type: &str, symbol: &str, now_ms: u64) -> bool {
        let key = format!("{}:{}", alert_type, symbol);
        match self.last_alerts.get(&key) {
            Some(&last_time) => {
                let elapsed_ms = now_ms.saturating_sub(last_time);
                elapsed_ms >= self.config.cooldown_ms
            }
            None => true,
        }
    }

    /// Mark alert as fired
    fn mark_fired(&mut self, alert_type: &str, symbol: &str, now_ms: u64) {
        let key = format!("{}:{}", alert_type, symbol);
        self.last_alerts.insert(key, now_ms);
    }

    /// Get number of alerts fired (for stats)
    pub fn alerts_fired_count(&self) -> usize {
        self.last_alerts.len()
    }

    /// Clear old cooldowns (memory cleanup)
    pub fn cleanup_old_cooldowns(&mut self, current_ms: u64, max_age_ms: u64) {
        self.last_alerts.retain(|_, &mut last_time| {
            current_ms.saturating_sub(last_time) < max_age_ms
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::*;

    fn make_features_with_whale(zscore: f64) -> Features {
        let mut features = Features::default();
        // Set entropy above threshold to avoid triggering entropy alert
        features.entropy.tick_entropy_1m = 0.8;
        features.whale_flow = Some(WhaleFlowFeatures {
            whale_net_flow_1h: 1_000_000.0,
            whale_flow_normalized_1h: zscore,
            whale_net_flow_4h: 2_000_000.0,
            whale_net_flow_24h: 5_000_000.0,
            whale_flow_intensity: 0.3,
            ..Default::default()
        });
        features
    }

    #[test]
    fn test_whale_accumulation_alert() {
        let config = AlertConfig::default();
        let mut tracker = AlertTracker::new(config);

        let features = make_features_with_whale(2.5); // Above threshold
        let alerts = tracker.check(&features, "BTC", 1000);

        assert_eq!(alerts.len(), 1);
        assert!(matches!(alerts[0].alert_type, AlertType::WhaleAccumulation));
    }

    #[test]
    fn test_whale_distribution_alert() {
        let config = AlertConfig::default();
        let mut tracker = AlertTracker::new(config);

        let features = make_features_with_whale(-2.5); // Below threshold
        let alerts = tracker.check(&features, "BTC", 1000);

        assert_eq!(alerts.len(), 1);
        assert!(matches!(alerts[0].alert_type, AlertType::WhaleDistribution));
    }

    #[test]
    fn test_cooldown_prevents_spam() {
        let mut config = AlertConfig::default();
        config.cooldown_ms = 1000; // 1 second cooldown

        let mut tracker = AlertTracker::new(config);
        let features = make_features_with_whale(2.5);

        // First alert fires
        let alerts1 = tracker.check(&features, "BTC", 1000);
        assert_eq!(alerts1.len(), 1);

        // Second alert within cooldown doesn't fire
        let alerts2 = tracker.check(&features, "BTC", 1500);
        assert_eq!(alerts2.len(), 0);

        // Third alert after cooldown fires
        let alerts3 = tracker.check(&features, "BTC", 2500);
        assert_eq!(alerts3.len(), 1);
    }

    #[test]
    fn test_no_alert_below_threshold() {
        let config = AlertConfig::default();
        let mut tracker = AlertTracker::new(config);

        let features = make_features_with_whale(1.5); // Below threshold
        let alerts = tracker.check(&features, "BTC", 1000);

        assert_eq!(alerts.len(), 0);
    }

    #[test]
    fn test_regime_change_alert() {
        let config = AlertConfig::default();
        let mut tracker = AlertTracker::new(config);

        let mut features = Features::default();
        features.regime = Some(RegimeFeatures {
            accumulation_score: 0.8,
            distribution_score: 0.2,
            ..Default::default()
        });

        let alerts = tracker.check(&features, "BTC", 1000);

        // Should fire regime change on first detection
        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::RegimeChange)));
    }

    #[test]
    fn test_entropy_alert() {
        let mut config = AlertConfig::default();
        config.entropy_low = 0.5;

        let mut tracker = AlertTracker::new(config);

        let mut features = Features::default();
        features.entropy.tick_entropy_1m = 0.3; // Below threshold

        let alerts = tracker.check(&features, "BTC", 1000);

        assert!(alerts.iter().any(|a| matches!(a.alert_type, AlertType::EntropyDrop)));
    }

    #[test]
    fn test_different_symbols_independent() {
        let mut config = AlertConfig::default();
        config.cooldown_ms = 10000;

        let mut tracker = AlertTracker::new(config);
        let features = make_features_with_whale(2.5);

        // BTC alert fires
        let alerts_btc = tracker.check(&features, "BTC", 1000);
        assert_eq!(alerts_btc.len(), 1);

        // ETH alert also fires (different symbol)
        let alerts_eth = tracker.check(&features, "ETH", 1000);
        assert_eq!(alerts_eth.len(), 1);

        // BTC again within cooldown doesn't fire
        let alerts_btc2 = tracker.check(&features, "BTC", 2000);
        assert_eq!(alerts_btc2.len(), 0);
    }
}
