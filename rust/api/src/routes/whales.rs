//! Whale API routes

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::sync::Arc;

use crate::routes::features::ErrorResponse;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct WhaleResponse {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub net_flow_1h: f64,
    pub net_flow_1h_zscore: f64,
    pub net_flow_4h: f64,
    pub net_flow_24h: f64,
    pub intensity: f64,
    pub direction: String,
    pub interpretation: String,
}

/// GET /api/whales/:symbol
/// Returns whale activity summary
pub async fn get_whale_summary(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<WhaleResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = symbol.to_uppercase();

    match state.redis.get_latest_features(&symbol).await {
        Ok(Some(snapshot)) => {
            if let Some(whale) = snapshot.whale {
                let net_flow_1h = whale["net_flow_1h"].as_f64().unwrap_or(0.0);
                let net_flow_1h_zscore = whale["net_flow_1h_zscore"].as_f64().unwrap_or(0.0);
                let net_flow_4h = whale["net_flow_4h"].as_f64().unwrap_or(0.0);
                let net_flow_24h = whale["net_flow_24h"].as_f64().unwrap_or(0.0);
                let intensity = whale["intensity"].as_f64().unwrap_or(0.0);
                let direction = whale["direction"]
                    .as_str()
                    .unwrap_or("NEUTRAL")
                    .to_string();

                let interpretation = interpret_whale_activity(
                    net_flow_1h,
                    net_flow_1h_zscore,
                    net_flow_4h,
                    &direction,
                );

                Ok(Json(WhaleResponse {
                    timestamp_ms: snapshot.timestamp_ms,
                    symbol,
                    net_flow_1h,
                    net_flow_1h_zscore,
                    net_flow_4h,
                    net_flow_24h,
                    intensity,
                    direction,
                    interpretation,
                }))
            } else {
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "Whale data not yet available".to_string(),
                    }),
                ))
            }
        }
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

fn interpret_whale_activity(
    flow_1h: f64,
    zscore: f64,
    flow_4h: f64,
    direction: &str,
) -> String {
    if zscore.abs() < 1.0 {
        return "Whale activity within normal range. No significant directional bias.".to_string();
    }

    let strength = if zscore.abs() >= 3.0 {
        "Extreme"
    } else if zscore.abs() >= 2.0 {
        "Strong"
    } else {
        "Moderate"
    };

    let flow_str = format!("${:.0}K", flow_1h.abs() / 1000.0);

    match direction {
        "ACCUMULATING" => {
            let momentum = if flow_1h > flow_4h / 4.0 {
                "accelerating"
            } else {
                "steady"
            };
            format!(
                "{} whale accumulation detected ({} in 1h). Flow is {}. Smart money may be positioning long.",
                strength, flow_str, momentum
            )
        }
        "DISTRIBUTING" => {
            let momentum = if flow_1h.abs() > flow_4h.abs() / 4.0 {
                "accelerating"
            } else {
                "steady"
            };
            format!(
                "{} whale distribution detected ({} in 1h). Flow is {}. Smart money may be exiting positions.",
                strength, flow_str, momentum
            )
        }
        _ => format!(
            "Whale flow elevated (z={:.2}) but direction unclear. Watch for breakout.",
            zscore
        ),
    }
}
