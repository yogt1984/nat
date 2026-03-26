//! Regime API routes

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
pub struct RegimeResponse {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub regime_type: String,
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub clarity: f64,
    pub range_position_24h: f64,
    pub absorption_zscore: f64,
    pub divergence_zscore: f64,
    pub churn_zscore: f64,
    pub interpretation: String,
}

/// GET /api/regime/:symbol
/// Returns current regime state with interpretation
pub async fn get_regime_state(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<RegimeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = symbol.to_uppercase();

    match state.redis.get_latest_features(&symbol).await {
        Ok(Some(snapshot)) => {
            if let Some(regime) = snapshot.regime {
                let acc_score = regime["accumulation_score"].as_f64().unwrap_or(0.0);
                let dist_score = regime["distribution_score"].as_f64().unwrap_or(0.0);
                let clarity = regime["clarity"].as_f64().unwrap_or(0.0);
                let range_pos = regime["range_position_24h"].as_f64().unwrap_or(0.5);

                let (regime_type, interpretation) =
                    interpret_regime(acc_score, dist_score, clarity, range_pos);

                Ok(Json(RegimeResponse {
                    timestamp_ms: snapshot.timestamp_ms,
                    symbol,
                    regime_type,
                    accumulation_score: acc_score,
                    distribution_score: dist_score,
                    clarity,
                    range_position_24h: range_pos,
                    absorption_zscore: regime["absorption_zscore"].as_f64().unwrap_or(0.0),
                    divergence_zscore: regime["divergence_zscore"].as_f64().unwrap_or(0.0),
                    churn_zscore: regime["churn_zscore"].as_f64().unwrap_or(0.0),
                    interpretation,
                }))
            } else {
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "Regime data not yet available (need ~60 minutes of data)"
                            .to_string(),
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

fn interpret_regime(acc: f64, dist: f64, clarity: f64, range_pos: f64) -> (String, String) {
    if clarity < 0.3 {
        return (
            "UNCLEAR".to_string(),
            "No clear regime detected. Market in transition or ranging.".to_string(),
        );
    }

    if acc > dist {
        let regime = "ACCUMULATION".to_string();
        let interp = if range_pos < 0.3 {
            "Strong accumulation at range lows. Smart money likely buying. Potential markup phase approaching."
        } else if range_pos < 0.5 {
            "Accumulation detected in lower half of range. Buying pressure with room to run."
        } else {
            "Accumulation detected but price elevated. Watch for distribution transition."
        };
        (regime, interp.to_string())
    } else {
        let regime = "DISTRIBUTION".to_string();
        let interp = if range_pos > 0.7 {
            "Strong distribution at range highs. Smart money likely selling. Potential markdown phase approaching."
        } else if range_pos > 0.5 {
            "Distribution detected in upper half of range. Selling pressure building."
        } else {
            "Distribution detected but price already low. Watch for accumulation transition."
        };
        (regime, interp.to_string())
    }
}
