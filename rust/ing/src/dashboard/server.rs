//! Dashboard HTTP/WebSocket server.

use anyhow::Result;
use axum::http::header;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tracing::info;

use super::handlers::{logs_handler, state_handler, symbols_handler};
use super::state::DashboardState;

/// Embedded HTML dashboard
const INDEX_HTML: &str = include_str!("../../static/index.html");

/// Serve the index page
async fn serve_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

/// Health check endpoint
async fn health() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain")],
        "ok",
    )
}

/// Run the dashboard server
pub async fn run_dashboard_server(
    addr: SocketAddr,
    state: Arc<DashboardState>,
) -> Result<()> {
    let app = Router::new()
        .route("/", get(serve_index))
        .route("/ws/logs", get(logs_handler))
        .route("/ws/state", get(state_handler))
        .route("/api/symbols", get(symbols_handler))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = TcpListener::bind(addr).await?;
    info!(%addr, "Dashboard server started");
    info!("Open http://{} in your browser", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the state broadcast loop
pub async fn run_state_broadcaster(state: Arc<DashboardState>) {
    let interval_ms = state.config.state_update_interval_ms;
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

    loop {
        interval.tick().await;
        state.broadcast_state();
    }
}
