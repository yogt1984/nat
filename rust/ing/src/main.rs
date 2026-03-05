//! ING - Hyperliquid Market Data Ingestor
//!
//! Real-time feature extraction from Hyperliquid order book and trade data.
//! Outputs features to Parquet files for downstream analysis.

mod config;
mod error;
mod features;
mod metrics;
mod output;
mod positions;
mod rest;
mod state;
mod whales;
mod ws;

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, error, Level};
use tracing_subscriber::{fmt, EnvFilter};

use crate::config::Config;
use crate::metrics::Metrics;
use crate::output::ParquetWriter;
use crate::state::MarketState;
use crate::ws::HyperliquidClient;

/// Feature vector emitted at regular intervals
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub sequence_id: u64,
    pub features: features::Features,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("Starting ING - Hyperliquid Ingestor");

    // Load configuration
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/ing.toml"));

    let config = Config::load(&config_path)?;
    info!(?config, "Configuration loaded");

    // Initialize metrics
    let metrics = Arc::new(Metrics::new());

    // Start Prometheus exporter if configured
    if let Some(addr) = &config.metrics.prometheus_addr {
        metrics::start_prometheus_exporter(addr.parse()?)?;
        info!(%addr, "Prometheus exporter started");
    }

    // Create channels for feature vectors
    let (feature_tx, feature_rx) = mpsc::channel::<FeatureVector>(10_000);

    // Initialize Parquet writer
    let writer = ParquetWriter::new(&config.output)?;
    let writer_handle = tokio::spawn(run_writer(writer, feature_rx));

    // Initialize market state for each symbol
    let mut handles = Vec::new();

    for symbol in &config.symbols.assets {
        let symbol = symbol.clone();
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let feature_tx = feature_tx.clone();

        let handle = tokio::spawn(async move {
            run_symbol_ingestor(symbol, config, metrics, feature_tx).await
        });

        handles.push(handle);
    }

    // Drop the original sender so writer knows when to stop
    drop(feature_tx);

    // Wait for all tasks
    for handle in handles {
        if let Err(e) = handle.await {
            error!(?e, "Symbol ingestor task failed");
        }
    }

    writer_handle.await??;

    info!("ING shutdown complete");
    Ok(())
}

/// Run ingestor for a single symbol
async fn run_symbol_ingestor(
    symbol: String,
    config: Config,
    metrics: Arc<Metrics>,
    feature_tx: mpsc::Sender<FeatureVector>,
) -> Result<()> {
    info!(%symbol, "Starting ingestor");

    let mut state = MarketState::new(&symbol, &config.features);
    let mut client = HyperliquidClient::new(&config.websocket, &symbol);
    let mut sequence_id: u64 = 0;

    // Feature emission interval
    let emission_interval = tokio::time::Duration::from_millis(
        config.features.emission_interval_ms
    );
    let mut emission_ticker = tokio::time::interval(emission_interval);

    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = client.recv() => {
                match msg {
                    Ok(Some(ws_msg)) => {
                        let start = std::time::Instant::now();

                        state.update(&ws_msg);

                        let elapsed = start.elapsed();
                        metrics.record_update_latency(&symbol, elapsed);
                    }
                    Ok(None) => {
                        warn!(%symbol, "WebSocket disconnected, reconnecting...");
                        client.reconnect().await?;
                    }
                    Err(e) => {
                        error!(%symbol, ?e, "WebSocket error");
                        metrics.record_error(&symbol, "websocket");
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        client.reconnect().await?;
                    }
                }
            }

            // Emit features at regular intervals
            _ = emission_ticker.tick() => {
                let start = std::time::Instant::now();

                if let Some(features) = state.compute_features() {
                    sequence_id += 1;

                    let feature_vector = FeatureVector {
                        timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
                        symbol: symbol.clone(),
                        sequence_id,
                        features,
                    };

                    if feature_tx.send(feature_vector).await.is_err() {
                        warn!(%symbol, "Feature channel closed");
                        break;
                    }

                    let elapsed = start.elapsed();
                    metrics.record_feature_latency(&symbol, elapsed);
                    metrics.record_feature_emitted(&symbol);
                }
            }
        }
    }

    Ok(())
}

/// Run the Parquet writer task
async fn run_writer(
    mut writer: ParquetWriter,
    mut feature_rx: mpsc::Receiver<FeatureVector>,
) -> Result<()> {
    info!("Parquet writer started");

    while let Some(feature_vector) = feature_rx.recv().await {
        writer.write(&feature_vector)?;
    }

    writer.flush()?;
    info!("Parquet writer shutdown complete");
    Ok(())
}
