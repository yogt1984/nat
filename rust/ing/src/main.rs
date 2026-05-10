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
use ing::ws::{HyperliquidClient, WsMessage};
use ing::FeatureVector;

// Health thresholds — silent-failure detection.
//
// Background: on 2026-05-06 ~08:25 UTC, the orderbook stream silently froze
// for all three symbols simultaneously while trade ticks kept flowing. The
// existing message-based staleness check (`client.is_stale()`) did not fire
// because the WebSocket connection was alive; only the Book channel was dead.
// 69% of the data collected after that point had stale prices, undetected for
// four days. These thresholds let the periodic health timer notice the same
// failure mode within minutes instead of days.
const BOOK_STALE_WARN_SECS: u64 = 60;
const BOOK_STALE_ERROR_SECS: u64 = 300;
const TRADE_STALE_WARN_SECS: u64 = 120;
const PRICE_FROZEN_WARN_SECS: u64 = 60;
const PRICE_FROZEN_ERROR_SECS: u64 = 300;

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
    let redis_config = RedisConfig::from_env_with_toml_url(Some(&config.redis.url));
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

    // Health summary every 60 seconds
    let mut health_ticker = tokio::time::interval(tokio::time::Duration::from_secs(60));
    health_ticker.tick().await; // skip immediate first tick

    // Ping/keepalive ticker
    let ping_interval = tokio::time::Duration::from_millis(config.websocket.ping_interval_ms);
    let mut ping_ticker = tokio::time::interval(ping_interval);
    ping_ticker.tick().await; // skip immediate first tick

    let connect_time = std::time::Instant::now();
    let mut first_feature_logged = false;
    let mut no_data_warned = false;

    // Per-channel liveness tracking — Book and Trade streams must be checked
    // independently because the failure mode on 2026-05-06 was Book-only.
    let mut last_book_msg_at: Option<std::time::Instant> = None;
    let mut last_trade_msg_at: Option<std::time::Instant> = None;
    let mut book_msg_count: u64 = 0;
    let mut trade_msg_count: u64 = 0;

    // Midprice-change tracking — the actual data corruption signal. If midprice
    // does not change for many consecutive emissions, downstream features are
    // being written with stale orderbook state regardless of WS connection state.
    let mut last_midprice: Option<f64> = None;
    let mut last_midprice_change_at: Option<std::time::Instant> = None;

    // Latching flags so we emit ERROR-level lines once per stuck episode
    // rather than every health tick.
    let mut book_stale_error_logged = false;
    let mut price_frozen_error_logged = false;

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

                        // Track per-channel liveness independently — see
                        // BOOK_STALE_WARN_SECS comment for the failure mode this
                        // catches.
                        match &ws_msg {
                            WsMessage::Book(_) => {
                                last_book_msg_at = Some(std::time::Instant::now());
                                book_msg_count += 1;
                            }
                            WsMessage::Trades(_) => {
                                last_trade_msg_at = Some(std::time::Instant::now());
                                trade_msg_count += 1;
                            }
                            _ => {}
                        }

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

                // Warn if no data after 30 seconds
                if !no_data_warned && message_count == 0 && connect_time.elapsed().as_secs() >= 30 {
                    no_data_warned = true;
                    warn!(
                        %symbol,
                        elapsed_s = connect_time.elapsed().as_secs(),
                        "No WebSocket data received after 30s — possible network/firewall issue"
                    );
                }

                if let Some(features) = state.compute_features() {
                    sequence_id += 1;
                    let timestamp_ms = chrono::Utc::now().timestamp_millis() as u64;

                    if !first_feature_logged {
                        first_feature_logged = true;
                        info!(
                            %symbol,
                            elapsed_s = format!("{:.1}", connect_time.elapsed().as_secs_f64()),
                            "First feature vector computed"
                        );
                    }

                    // Midprice-change detection: any motion at all resets the
                    // freeze timer. Exact equality is intentional — orderbook
                    // ticks are integers in price-precision units, so genuine
                    // motion always yields a different f64.
                    let mid = features.raw.midprice;
                    if mid.is_finite() {
                        let changed = match last_midprice {
                            Some(prev) => prev != mid,
                            None => true,
                        };
                        if changed {
                            last_midprice = Some(mid);
                            last_midprice_change_at = Some(std::time::Instant::now());
                        }
                    }

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

            // Ping/keepalive — detect stale connections and reconnect
            _ = ping_ticker.tick() => {
                if client.is_connected() {
                    // Check if connection is stale or ping timed out
                    if client.is_stale() || client.ping_timed_out() {
                        let silence = client.seconds_since_last_message();
                        warn!(
                            %symbol,
                            silence_s = format!("{:.1}", silence),
                            stale = client.is_stale(),
                            ping_timeout = client.ping_timed_out(),
                            "Connection stale — forcing reconnect"
                        );
                        dashboard_state.update_symbol(&symbol, |s| {
                            s.connected = false;
                        });
                        client.reconnect().await?;
                        dashboard_state.update_symbol(&symbol, |s| {
                            s.connected = true;
                        });
                    } else {
                        // Send keepalive ping
                        if let Err(e) = client.send_ping().await {
                            warn!(%symbol, ?e, "Ping send failed, reconnecting");
                            client.reconnect().await?;
                        }
                    }
                }
            }

            // Health summary every 60 seconds (not biased — runs when other branches idle)
            _ = health_ticker.tick() => {
                let uptime = connect_time.elapsed().as_secs();
                let mins = uptime / 60;
                let secs = uptime % 60;
                let ws_msgs = client.message_count();

                if ws_msgs == 0 && uptime >= 30 {
                    warn!(
                        %symbol,
                        connected = client.is_connected(),
                        messages = ws_msgs,
                        features = sequence_id,
                        uptime = format!("{}m{}s", mins, secs),
                        "Health: NO DATA FLOWING"
                    );
                } else {
                    let now = std::time::Instant::now();
                    let book_age = last_book_msg_at
                        .map(|t| now.saturating_duration_since(t).as_secs());
                    let trade_age = last_trade_msg_at
                        .map(|t| now.saturating_duration_since(t).as_secs());
                    let price_age = last_midprice_change_at
                        .map(|t| now.saturating_duration_since(t).as_secs());

                    // Book-stream silence — the May-6 failure signature. Trade
                    // ticks may keep flowing while book updates are dead, so
                    // generic ws-staleness misses this.
                    match book_age {
                        Some(age) if age >= BOOK_STALE_WARN_SECS => {
                            warn!(
                                %symbol,
                                book_silence_s = age,
                                book_msgs = book_msg_count,
                                last_midprice = ?last_midprice,
                                "Health: BOOK STREAM STALE — orderbook updates have stopped"
                            );
                            if age >= BOOK_STALE_ERROR_SECS && !book_stale_error_logged {
                                book_stale_error_logged = true;
                                error!(
                                    %symbol,
                                    book_silence_s = age,
                                    threshold_s = BOOK_STALE_ERROR_SECS,
                                    "Health: BOOK STREAM DEAD — collected data is corrupted; \
                                     reconnect/resubscribe required"
                                );
                            }
                        }
                        Some(_) => {
                            book_stale_error_logged = false;
                        }
                        None if uptime >= 60 => {
                            warn!(
                                %symbol,
                                uptime_s = uptime,
                                "Health: NO BOOK MESSAGES received yet — \
                                 book subscription may have failed"
                            );
                        }
                        None => {}
                    }

                    // Trade-stream silence — secondary signal; trades are bursty,
                    // so the threshold is more lenient.
                    if let Some(age) = trade_age {
                        if age >= TRADE_STALE_WARN_SECS {
                            warn!(
                                %symbol,
                                trade_silence_s = age,
                                trade_msgs = trade_msg_count,
                                "Health: TRADE STREAM QUIET — no trades in over {}s",
                                TRADE_STALE_WARN_SECS
                            );
                        }
                    }

                    // Midprice-frozen — the actual data corruption signal. Even
                    // if Book messages keep arriving, if midprice doesn't move,
                    // downstream features are being written with stale state.
                    if let Some(age) = price_age {
                        if age >= PRICE_FROZEN_WARN_SECS {
                            warn!(
                                %symbol,
                                price_frozen_s = age,
                                last_midprice = ?last_midprice,
                                book_silence_s = ?book_age,
                                "Health: MIDPRICE FROZEN — features are being written with stale prices"
                            );
                            if age >= PRICE_FROZEN_ERROR_SECS && !price_frozen_error_logged {
                                price_frozen_error_logged = true;
                                error!(
                                    %symbol,
                                    price_frozen_s = age,
                                    threshold_s = PRICE_FROZEN_ERROR_SECS,
                                    "Health: MIDPRICE FROZEN past threshold — \
                                     collected data is unusable; investigate immediately"
                                );
                            }
                        } else {
                            price_frozen_error_logged = false;
                        }
                    }

                    info!(
                        %symbol,
                        connected = client.is_connected(),
                        messages = ws_msgs,
                        book_msgs = book_msg_count,
                        trade_msgs = trade_msg_count,
                        features = sequence_id,
                        book_age_s = ?book_age,
                        trade_age_s = ?trade_age,
                        price_change_age_s = ?price_age,
                        last_midprice = ?last_midprice,
                        uptime = format!("{}m{}s", mins, secs),
                        "Health summary"
                    );
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
