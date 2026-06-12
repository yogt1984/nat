//! ING - Hyperliquid Market Data Ingestor
//!
//! Real-time feature extraction from Hyperliquid order book and trade data.
//! Outputs features to Parquet files for downstream analysis.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use ing::alerts::{AlertConfig, AlertTracker};
use ing::config::Config;
use ing::dashboard::state::FeaturesSummary;
use ing::dashboard::{run_dashboard_server, BroadcastLayer, DashboardState};
use ing::features::CrossSymbolState;
use ing::metrics::Metrics;
use ing::output::{ParquetWriter, TradeParquetWriter, TradeRecord};
use ing::positions::{
    PositionTracker, PositionTrackerConfig, SharedPositionState, WalletDiscovery,
};
use ing::redis_publisher::{RedisConfig, RedisPublisher};
use ing::state::MarketState;
use ing::ws::{HyperliquidClient, WsMessage};
use ing::FeatureVector;

// Health thresholds moved to health.rs — see `ChannelHealth` and its tests.
use ing::health::{ChannelHealth, Severity};

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
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

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

    // Graceful shutdown: watch channel broadcasts to all symbol tasks
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        let ctrl_c = tokio::signal::ctrl_c();
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => info!("Received SIGINT (Ctrl+C), initiating shutdown..."),
            _ = sigterm.recv() => info!("Received SIGTERM, initiating shutdown..."),
        }
        let _ = shutdown_tx.send(true);
    });

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
    let redis_config =
        RedisConfig::from_env_with_toml(Some(&config.redis.url), config.redis.publish_interval_ms);
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

    // Initialize algorithms (dummy impls for now, real logic in Python)
    let alg_feature_names = {
        let probe = ing::algorithms::create_algorithms(&config.algorithms.enabled);
        ing::algorithms::all_alg_feature_names(&probe)
    };
    if !alg_feature_names.is_empty() {
        info!(algorithms = ?config.algorithms.enabled, n_alg_features = alg_feature_names.len(),
              "Algorithm features enabled");
    }

    // Initialize Parquet writer (with algorithm columns)
    let writer =
        ParquetWriter::new_with_alg_features(&config.output, config.data_dir(), alg_feature_names)?;
    let writer_handle = tokio::spawn(run_writer(writer, feature_rx));

    // Initialize trade writer (optional — writes raw trades to separate Parquet files)
    let (trade_tx, trade_rx) = mpsc::channel::<TradeRecord>(100_000);
    let trade_writer_handle = if config.trade_output.enabled {
        let default_trade_dir = "../data/trades".to_string();
        let trade_writer =
            TradeParquetWriter::new(&config.trade_output, &default_trade_dir)?;
        info!("Trade Parquet writer enabled");
        Some(tokio::spawn(run_trade_writer(trade_writer, trade_rx)))
    } else {
        info!("Trade Parquet writer disabled");
        drop(trade_rx); // drop receiver so senders don't block
        None
    };

    // Initialize position tracker (optional — polls wallet positions via REST)
    // tracker_wallets_handle is shared with wallet discovery in symbol tasks
    let mut tracker_wallets_handle: Option<Arc<parking_lot::RwLock<Vec<String>>>> = None;
    let shared_position_state: Option<Arc<SharedPositionState>> =
        if let Some(ref pt_config) = config.position_tracker {
            if pt_config.enabled {
                let shared = Arc::new(SharedPositionState::new());

                let (snapshot_tx, mut snapshot_rx) = mpsc::channel(1000);
                let (delta_tx, mut delta_rx) = mpsc::channel(1000);

                let tracker_config = PositionTrackerConfig {
                    poll_interval_secs: pt_config.poll_interval_secs,
                    symbols: config.symbols.assets.clone(),
                    max_concurrent_requests: pt_config.max_concurrent_requests,
                };
                let tracker = match PositionTracker::new(tracker_config) {
                    Ok(t) => t,
                    Err(e) => {
                        error!(?e, "Failed to create position tracker");
                        return Err(e);
                    }
                };
                let tracker = tracker
                    .with_snapshot_channel(snapshot_tx)
                    .with_delta_channel(delta_tx);

                if !pt_config.initial_wallets.is_empty() {
                    tracker.add_wallets(&pt_config.initial_wallets);
                    info!(
                        count = pt_config.initial_wallets.len(),
                        "Position tracker: loaded initial wallets"
                    );
                }

                // Extract wallet handle for discovery before moving tracker
                if pt_config.discover_from_trades {
                    tracker_wallets_handle = Some(tracker.wallets_handle());
                }

                // Spawn tracker polling loop
                tokio::spawn(async move {
                    if let Err(e) = tracker.run().await {
                        error!(?e, "Position tracker error");
                    }
                });

                // Spawn consumer: channels -> SharedPositionState
                let sps = Arc::clone(&shared);
                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            Some(snapshot) = snapshot_rx.recv() => {
                                sps.update_snapshot(snapshot);
                            }
                            Some(delta) = delta_rx.recv() => {
                                sps.update_delta(delta);
                            }
                            else => break,
                        }
                    }
                });

                info!(
                    poll_secs = pt_config.poll_interval_secs,
                    max_wallets = pt_config.max_tracked_wallets,
                    discover = pt_config.discover_from_trades,
                    "Position tracker started"
                );
                Some(shared)
            } else {
                None
            }
        } else {
            None
        };

    // Initialize market state for each symbol
    let mut handles = Vec::new();
    let cross_symbol_state = CrossSymbolState::new();

    for symbol in &config.symbols.assets {
        let symbol = symbol.clone();
        let config = config.clone();
        let metrics = Arc::clone(&metrics);
        let feature_tx = feature_tx.clone();
        let trade_tx = trade_tx.clone();
        let dashboard_state = Arc::clone(&dashboard_state);
        let redis_publisher = redis_publisher.clone();
        let alert_config = alert_config.clone();
        let cross_symbol = cross_symbol_state.clone();
        let shutdown_rx = shutdown_rx.clone();
        let position_state = shared_position_state.clone();
        let wallets_handle = tracker_wallets_handle.clone();

        let handle = tokio::spawn(async move {
            run_symbol_ingestor(
                symbol,
                config,
                metrics,
                feature_tx,
                trade_tx,
                dashboard_state,
                redis_publisher,
                alert_config,
                cross_symbol,
                shutdown_rx,
                position_state,
                wallets_handle,
            )
            .await
        });

        handles.push(handle);
    }

    // Drop the original senders so writers know when to stop
    // (once all symbol tasks also drop their clones)
    drop(feature_tx);
    drop(trade_tx);

    // Block until SIGINT/SIGTERM — this is where the process lives
    // during normal operation. Symbol tasks emit features in the
    // background; on signal they break their loops and drop senders.
    let _ = shutdown_rx.changed().await;

    // Wait for symbol tasks to exit (3s deadline)
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(3);
    for handle in handles {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(remaining, handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => error!(?e, "Symbol task error"),
            Ok(Err(e)) => error!(?e, "Symbol task panicked"),
            Err(_) => warn!("Symbol task did not stop in time"),
        }
    }

    info!("All symbol tasks stopped, waiting for writer flush...");

    // Writer's recv() returns None now that all senders are dropped → flushes buffer
    match tokio::time::timeout(tokio::time::Duration::from_secs(3), writer_handle).await {
        Ok(Ok(Ok(()))) => info!("Writer flushed successfully"),
        Ok(Ok(Err(e))) => error!(?e, "Writer error during shutdown"),
        Ok(Err(e)) => error!(?e, "Writer task panicked"),
        Err(_) => error!("Writer flush timed out — data may be lost"),
    }

    // Flush trade writer if enabled
    if let Some(handle) = trade_writer_handle {
        match tokio::time::timeout(tokio::time::Duration::from_secs(3), handle).await {
            Ok(Ok(Ok(()))) => info!("Trade writer flushed successfully"),
            Ok(Ok(Err(e))) => error!(?e, "Trade writer error during shutdown"),
            Ok(Err(e)) => error!(?e, "Trade writer task panicked"),
            Err(_) => error!("Trade writer flush timed out — data may be lost"),
        }
    }

    info!("ING shutdown complete");
    Ok(())
}

/// Run ingestor for a single symbol
async fn run_symbol_ingestor(
    symbol: String,
    config: Config,
    metrics: Arc<Metrics>,
    feature_tx: mpsc::Sender<FeatureVector>,
    trade_tx: mpsc::Sender<TradeRecord>,
    dashboard_state: Arc<DashboardState>,
    redis_publisher: Option<Arc<Mutex<RedisPublisher>>>,
    alert_config: AlertConfig,
    cross_symbol_state: CrossSymbolState,
    mut shutdown_rx: watch::Receiver<bool>,
    position_state: Option<Arc<SharedPositionState>>,
    wallets_handle: Option<Arc<parking_lot::RwLock<Vec<String>>>>,
) -> Result<()> {
    info!(%symbol, "Starting ingestor");

    let algorithms = ing::algorithms::create_algorithms(&config.algorithms.enabled);
    let mut state = MarketState::new_with_algorithms(&symbol, &config.features, algorithms);
    state.set_cross_symbol_state(cross_symbol_state);

    // Wire position tracker data if available
    if let Some(ref ps) = position_state {
        state.set_position_state(Arc::clone(ps));
        if let Some(ref pt_config) = config.position_tracker {
            state.set_concentration_whale_threshold(pt_config.whale_threshold_usd);
        }
        info!(%symbol, "Position tracker wired into feature computation");
    }

    // Initialize wallet discovery if tracker is active with discover_from_trades
    let mut wallet_discovery = wallets_handle.as_ref().map(|_| {
        let max_wallets = config
            .position_tracker
            .as_ref()
            .map(|pt| pt.max_tracked_wallets)
            .unwrap_or(50);
        WalletDiscovery::new(max_wallets)
    });
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
    let emission_interval =
        tokio::time::Duration::from_millis(config.features.emission_interval_ms);
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

    // Per-channel liveness tracking — see health.rs for thresholds and docs.
    let mut channel_health = ChannelHealth::new();

    // Wallet discovery promotion ticker (every 5 minutes)
    let mut discovery_ticker =
        tokio::time::interval(tokio::time::Duration::from_secs(300));
    discovery_ticker.tick().await; // skip immediate first tick

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

                        // Track per-channel liveness independently
                        match &ws_msg {
                            WsMessage::Book(_) => channel_health.on_book(),
                            WsMessage::Trades(trades) => {
                                channel_health.on_trade();
                                // Send raw trades to trade writer (if enabled)
                                if config.trade_output.enabled {
                                    for t in trades {
                                        let rec = TradeRecord {
                                            timestamp_ns: (t.time as i64) * 1_000_000, // ms → ns
                                            symbol: symbol.clone(),
                                            tid: t.tid,
                                            price: t.price(),
                                            size: t.size(),
                                            is_buy: t.is_buy(),
                                        };
                                        // Non-blocking: if channel is full, drop the trade
                                        let _ = trade_tx.try_send(rec);
                                    }
                                }
                                // Wallet discovery: observe maker/taker addresses
                                if let Some(ref mut discovery) = wallet_discovery {
                                    for t in trades {
                                        if let Some((maker, taker)) = &t.users {
                                            let first = discovery.observe_trade(maker, taker);
                                            if first {
                                                info!(
                                                    %symbol,
                                                    "WsTrade.users field IS populated — wallet discovery active"
                                                );
                                            }
                                        }
                                    }
                                }
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

                if let Some((features, alg_values)) = state.compute_features() {
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

                    // Midprice-change detection via ChannelHealth
                    channel_health.on_midprice(features.raw.midprice);

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
                        alg_values,
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
                    let hs = channel_health.check_health(uptime);

                    // Book-stream health
                    match hs.book.severity {
                        Severity::Error => {
                            warn!(
                                %symbol,
                                book_silence_s = ?hs.book.age_secs,
                                book_msgs = hs.book.msg_count,
                                last_midprice = ?channel_health.last_midprice,
                                "Health: BOOK STREAM STALE — orderbook updates have stopped"
                            );
                            if !channel_health.book_stale_error_logged() {
                                // Latch was just set by check_health
                            }
                            error!(
                                %symbol,
                                book_silence_s = ?hs.book.age_secs,
                                "Health: BOOK STREAM DEAD — collected data is corrupted"
                            );
                        }
                        Severity::Warn => {
                            warn!(
                                %symbol,
                                book_silence_s = ?hs.book.age_secs,
                                book_msgs = hs.book.msg_count,
                                last_midprice = ?channel_health.last_midprice,
                                "Health: BOOK STREAM STALE — orderbook updates have stopped"
                            );
                        }
                        Severity::Ok => {}
                    }

                    // Trade-stream health
                    if hs.trade.severity == Severity::Warn {
                        warn!(
                            %symbol,
                            trade_silence_s = ?hs.trade.age_secs,
                            trade_msgs = hs.trade.msg_count,
                            "Health: TRADE STREAM QUIET"
                        );
                    }

                    // Price-frozen health
                    match hs.price_frozen.severity {
                        Severity::Error => {
                            warn!(
                                %symbol,
                                price_frozen_s = ?hs.price_frozen.age_secs,
                                last_midprice = ?hs.price_frozen.last_midprice,
                                book_silence_s = ?hs.book.age_secs,
                                regime = hs.price_frozen.regime,
                                "Health: MIDPRICE FROZEN — features are being written with stale prices"
                            );
                            error!(
                                %symbol,
                                price_frozen_s = ?hs.price_frozen.age_secs,
                                regime = hs.price_frozen.regime,
                                "Health: MIDPRICE FROZEN past threshold — \
                                 collected data is unusable; investigate immediately"
                            );
                        }
                        Severity::Warn => {
                            warn!(
                                %symbol,
                                price_frozen_s = ?hs.price_frozen.age_secs,
                                last_midprice = ?hs.price_frozen.last_midprice,
                                book_silence_s = ?hs.book.age_secs,
                                regime = hs.price_frozen.regime,
                                "Health: MIDPRICE FROZEN — features are being written with stale prices"
                            );
                        }
                        Severity::Ok => {}
                    }

                    info!(
                        %symbol,
                        connected = client.is_connected(),
                        messages = ws_msgs,
                        book_msgs = channel_health.book_msg_count,
                        trade_msgs = channel_health.trade_msg_count,
                        features = sequence_id,
                        book_age_s = ?hs.book.age_secs,
                        trade_age_s = ?hs.trade.age_secs,
                        price_change_age_s = ?hs.price_frozen.age_secs,
                        last_midprice = ?channel_health.last_midprice,
                        uptime = format!("{}m{}s", mins, secs),
                        "Health summary"
                    );
                }
            }

            // Wallet discovery: promote top wallets to tracker every 5 minutes
            _ = discovery_ticker.tick() => {
                if let (Some(ref mut discovery), Some(ref wh)) = (&mut wallet_discovery, &wallets_handle) {
                    if discovery.users_field_available() {
                        discovery.promote_top(5, wh);
                    } else if connect_time.elapsed().as_secs() > 300 {
                        // After 5 minutes with no users field, warn once
                        warn!(
                            %symbol,
                            "WsTrade.users field not populated after 5min — \
                             wallet discovery unavailable, using initial_wallets only"
                        );
                    }
                }
            }

            // Graceful shutdown — lowest priority so in-flight work completes first
            _ = shutdown_rx.changed() => {
                info!(%symbol, "Shutdown signal received, closing WebSocket...");
                client.close().await;
                info!(%symbol, "Symbol ingestor stopped");
                break;
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

/// Run the trade Parquet writer task
async fn run_trade_writer(
    mut writer: TradeParquetWriter,
    mut trade_rx: mpsc::Receiver<TradeRecord>,
) -> Result<()> {
    info!("Trade Parquet writer started");

    while let Some(trade) = trade_rx.recv().await {
        writer.write(&trade)?;
    }

    writer.flush()?;
    info!("Trade Parquet writer shutdown complete");
    Ok(())
}
