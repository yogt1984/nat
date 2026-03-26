//! Health check endpoint

use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub timestamp: String,
}

/// GET /health
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}
