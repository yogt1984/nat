//! Feature API routes

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::sync::Arc;

use crate::redis_client::FeatureSnapshot;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// GET /api/features/:symbol
/// Returns latest feature snapshot for a symbol
pub async fn get_latest(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<FeatureSnapshot>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = symbol.to_uppercase();

    match state.redis.get_latest_features(&symbol).await {
        Ok(Some(snapshot)) => Ok(Json(snapshot)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No data for symbol: {}", symbol),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Redis error: {}", e),
            }),
        )),
    }
}

/// GET /api/symbols
/// Returns list of active symbols
pub async fn get_symbols(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    match state.redis.get_active_symbols().await {
        Ok(symbols) => Ok(Json(symbols)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Redis error: {}", e),
            }),
        )),
    }
}
