//! ING - Hyperliquid Market Data Ingestor
//!
//! Real-time feature extraction from Hyperliquid order book and trade data.
//! Outputs features to Parquet files for downstream analysis.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use ing::alerts::{AlertConfig, AlertTracker};
use ing::config::Config;
use ing::dashboard::{BroadcastLayer, DashboardState, run_dashboard_server};
use ing::dashboard::state::FeaturesSummary;
use ing::metrics::Metrics;
use ing::output::ParquetWriter;
use ing::redis_publisher::{RedisConfig, RedisPublisher};
use ing::state::MarketState;
use ing::ws::HyperliquidClient;
use ing::FeatureVector;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration first (before logging setup)
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config/ing.toml"));

    let config = Config::load(&config_path)?;

    // Initialize dashboard state (needed for log broadcast layer)
    let dashboard_state = Arc::new(DashboardState::new());

    // Initialize logging with broadcast layer for dashboard
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_thread_ids(true);

    let broadcast_layer = BroadcastLayer::new(Arc::clone(&dashboard_state));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(broadcast_layer)
        .init();

    info!("Starting ING - Hyperliquid Ingestor");
    info!(?config, "Configuration loaded");

    // Initialize metrics
    let metrics = Arc::new(Metrics::new());

    // Start Prometheus exporter if configured
    if let Some(addr) = &config.metrics.prometheus_addr {
        ing::metrics::start_prometheus_exporter(addr.parse()?)?;
        info!(%addr, "Prometheus exporter started");
    }

    // Start dashboard server if enabled
    if config.dashboard.enabled {
        let addr = config.dashboard.addr.parse()?;
        let state = Arc::clone(&dashboard_state);

        // Spawn dashboard server
        tokio::spawn(async move {
            if let Err(e) = run_dashboard_server(addr, state).await {
                error!(?e, "Dashboard server error");
            }
        });

        // Spawn state broadcaster
        let state = Arc::clone(&dashboard_state);
        tokio::spawn(async move {
            ing::dashboard::server::run_state_broadcaster(state).await;
        });

        info!(addr = %config.dashboard.addr, "Dashboard enabled");
    }

    // Initialize Redis publisher (optional - gracefully degrades if unavailable)
    let redis_config = RedisConfig::from_env();
    let redis_publisher: Option<Arc<Mutex<RedisPublisher>>> =
        match RedisPublisher::try_new(redis_config).await {
            Some(publisher) => {
                info!("Redis publisher initialized");
                Some(Arc::new(Mutex::new(publisher)))
            }
            None => {
                info!("Redis publisher disabled (will continue without real-time streaming)");
                None
            }
        };

    // Load alert configuration
    let alert_config = AlertConfig::from_env();
    info!(?alert_config, "Alert configuration loaded");

    // Create channels for feature vectors
    let (feature_tx, feature_rx) = mpsc::channel::<FeatureVector>(10_000);

    // Initialize Parquet writer
    let writer = ParquetWriter::new(&config.output, config.data_dir())?;
    let writer_handle = tokio::spawn(run_writer(writer, feature_rx));

    // Initialize market state for each symbol
    let mut handles = Vec::new();

    for symbol in &config.symbols.assets {
        let symbol = symbol.clone();
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let feature_tx = feature_tx.clone();
        let dashboard_state = Arc::clone(&dashboard_state);
        let redis_publisher = redis_publisher.clone();
        let alert_config = alert_config.clone();

        let handle = tokio::spawn(async move {
            run_symbol_ingestor(
                symbol,
                config,
                metrics,
                feature_tx,
                dashboard_state,
                redis_publisher,
                alert_config,
            ).await
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
    dashboard_state: Arc<DashboardState>,
    redis_publisher: Option<Arc<Mutex<RedisPublisher>>>,
    alert_config: AlertConfig,
) -> Result<()> {
    info!(%symbol, "Starting ingestor");

    let mut state = MarketState::new(&symbol, &config.features);
    let mut client = HyperliquidClient::new(&config.websocket, &symbol);
    let mut sequence_id: u64 = 0;
    let mut message_count: u64 = 0;

    // Initialize alert tracker for this symbol
    let mut alert_tracker = AlertTracker::new(alert_config);

    // Connect to WebSocket before entering the main loop
    // This ensures connection isn't cancelled by the ticker
    loop {
        match client.connect().await {
            Ok(()) => {
                info!(%symbol, "WebSocket connected successfully");
                dashboard_state.update_symbol(&symbol, |s| {
                    s.connected = true;
                });
                break;
            }
            Err(e) => {
                error!(%symbol, ?e, "Failed to connect, retrying...");
                dashboard_state.update_symbol(&symbol, |s| {
                    s.connected = false;
                });
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }
    }

    // Feature emission interval
    let emission_interval = tokio::time::Duration::from_millis(
        config.features.emission_interval_ms
    );
    let mut emission_ticker = tokio::time::interval(emission_interval);

    loop {
        tokio::select! {
            // Bias towards WebSocket messages to ensure we don't miss data
            biased;
            // Handle incoming WebSocket messages
            msg = client.recv() => {
                match msg {
                    Ok(Some(ws_msg)) => {
                        let start = std::time::Instant::now();

                        state.update(&ws_msg);
                        message_count += 1;

                        // Update dashboard state
                        dashboard_state.update_symbol(&symbol, |s| {
                            s.message_count = message_count;
                            s.last_update_ms = chrono::Utc::now().timestamp_millis();
                            s.connected = true;
                        });

                        let elapsed = start.elapsed();
                        metrics.record_update_latency(&symbol, elapsed);
                    }
                    Ok(None) => {
                        // Ok(None) means no data message (ping/pong, unparseable, etc.)
                        // Only reconnect if connection was actually lost
                        if !client.is_connected() {
                            warn!(%symbol, "WebSocket disconnected, reconnecting...");
                            dashboard_state.update_symbol(&symbol, |s| {
                                s.connected = false;
                            });
                            client.reconnect().await?;
                            dashboard_state.update_symbol(&symbol, |s| {
                                s.connected = true;
                            });
                        }
                        // Otherwise just continue - this is normal for non-data messages
                    }
                    Err(e) => {
                        error!(%symbol, ?e, "WebSocket error");
                        metrics.record_error(&symbol, "websocket");
                        dashboard_state.update_symbol(&symbol, |s| {
                            s.connected = false;
                        });
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        client.reconnect().await?;
                        dashboard_state.update_symbol(&symbol, |s| {
                            s.connected = true;
                        });
                    }
                }
            }

            // Emit features at regular intervals
            _ = emission_ticker.tick() => {
                let start = std::time::Instant::now();

                if let Some(features) = state.compute_features() {
                    sequence_id += 1;
                    let timestamp_ms = chrono::Utc::now().timestamp_millis() as u64;

                    // Update dashboard with feature summary
                    dashboard_state.update_symbol(&symbol, |s| {
                        s.feature_count = sequence_id;
                        s.features = FeaturesSummary {
                            midprice: features.raw.midprice,
                            spread_bps: features.raw.spread_bps,
                            imbalance: features.imbalance.qty_l1,
                            volatility_1m: features.volatility.returns_1m,
                            vpin: features.toxicity.vpin_10,
                            whale_flow: features.whale_flow.as_ref().map(|wf| wf.whale_net_flow_1h).unwrap_or(0.0),
                            kyle_lambda: features.illiquidity.kyle_lambda_100,
                        };
                    });

                    // Publish features to Redis (if available)
                    if let Some(ref publisher) = redis_publisher {
                        let mut pub_guard = publisher.lock().await;
                        if let Err(e) = pub_guard.publish_features(&symbol, &features, timestamp_ms).await {
                            warn!(%symbol, ?e, "Failed to publish features to Redis");
                        }

                        // Check for alert conditions
                        let alerts = alert_tracker.check(&features, &symbol, timestamp_ms);
                        for alert in alerts {
                            info!(%symbol, alert_type = ?alert.alert_type, severity = ?alert.severity, "Alert triggered: {}", alert.message);
                            if let Err(e) = pub_guard.publish_alert(&alert).await {
                                warn!(%symbol, ?e, "Failed to publish alert to Redis");
                            }
                        }
                    }

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
