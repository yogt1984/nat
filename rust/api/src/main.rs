//! NAT API Server
//!
//! REST API and WebSocket server for feature consumption.

use axum::{http::Method, routing::get, Router};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use nat_api::config::ApiConfig;
use nat_api::redis_client::RedisClient;
use nat_api::routes;
use nat_api::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("nat_api=info,tower_http=debug")),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load config
    let config = ApiConfig::from_env();
    info!(?config, "Configuration loaded");

    // Initialize Redis client
    let redis = RedisClient::new(&config.redis_url)
        .await
        .expect("Failed to connect to Redis");

    // Create app state
    let state = Arc::new(AppState::new(redis, config.clone()));

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(routes::health::health_check))
        // Feature endpoints
        .route("/api/features/:symbol", get(routes::features::get_latest))
        .route("/api/symbols", get(routes::features::get_symbols))
        // Whale endpoints
        .route("/api/whales/:symbol", get(routes::whales::get_whale_summary))
        // Regime endpoints
        .route("/api/regime/:symbol", get(routes::regime::get_regime_state))
        // Research endpoints
        .route("/api/research/hypotheses", get(routes::research::list_hypotheses))
        .route("/api/research/hypotheses/:id", get(routes::research::get_hypothesis))
        .route("/api/research/cycles", get(routes::research::list_cycles))
        .route("/api/research/signals", get(routes::research::list_signals))
        .route("/api/research/stats", get(routes::research::get_stats))
        .route("/api/research/heatmap", get(routes::research::get_heatmap))
        .route("/api/research/network", get(routes::research::get_network))
        // WebSocket streaming
        .route("/ws/stream/:symbol", get(routes::ws::websocket_handler))
        .route("/ws/alerts", get(routes::ws::alerts_websocket_handler))
        .route("/ws/research", get(routes::ws::research_websocket_handler))
        // Add state
        .with_state(state)
        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        );

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    info!("Starting NAT API server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
