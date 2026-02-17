//! Metrics collection for monitoring

use metrics::{counter, histogram, describe_counter, describe_histogram, Unit};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use std::time::Duration;

/// Metrics collector
pub struct Metrics {
    _private: (),
}

impl Metrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        // Describe metrics
        describe_counter!(
            "ing_features_emitted_total",
            Unit::Count,
            "Total number of feature vectors emitted"
        );

        describe_counter!(
            "ing_errors_total",
            Unit::Count,
            "Total number of errors by type"
        );

        describe_histogram!(
            "ing_update_latency_seconds",
            Unit::Seconds,
            "Latency of processing WebSocket updates"
        );

        describe_histogram!(
            "ing_feature_latency_seconds",
            Unit::Seconds,
            "Latency of computing features"
        );

        Self { _private: () }
    }

    /// Record a feature emission
    pub fn record_feature_emitted(&self, symbol: &str) {
        counter!("ing_features_emitted_total", "symbol" => symbol.to_string()).increment(1);
    }

    /// Record an error
    pub fn record_error(&self, symbol: &str, error_type: &str) {
        counter!(
            "ing_errors_total",
            "symbol" => symbol.to_string(),
            "type" => error_type.to_string()
        ).increment(1);
    }

    /// Record update processing latency
    pub fn record_update_latency(&self, symbol: &str, duration: Duration) {
        histogram!(
            "ing_update_latency_seconds",
            "symbol" => symbol.to_string()
        ).record(duration.as_secs_f64());
    }

    /// Record feature computation latency
    pub fn record_feature_latency(&self, symbol: &str, duration: Duration) {
        histogram!(
            "ing_feature_latency_seconds",
            "symbol" => symbol.to_string()
        ).record(duration.as_secs_f64());
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Start the Prometheus exporter
pub fn start_prometheus_exporter(addr: SocketAddr) -> anyhow::Result<()> {
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()?;

    Ok(())
}
