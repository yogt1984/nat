//! Research API routes — hypothesis history, cycle reports, signal registry, stats.
//!
//! Data source: the SQLite `research_output` table only (written by Python agents).
//! The legacy JSON-directory fallback was removed in Q1.3 — SQLite is the single
//! source of truth; endpoints return empty/404 when the DB is unavailable.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::routes::features::ErrorResponse;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Schema versioning
// ---------------------------------------------------------------------------

/// Current expected schema version for research output records.
const CURRENT_SCHEMA_VERSION: u64 = 1;

/// Normalize a research output record: ensure schema_version is present,
/// backfill missing fields for old records, warn on unknown versions.
fn normalize_record(mut record: serde_json::Value) -> serde_json::Value {
    let obj = match record.as_object_mut() {
        Some(o) => o,
        None => return record,
    };

    let version = obj
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if version == 0 {
        // Pre-versioning record: backfill defaults
        obj.entry("schema_version").or_insert(serde_json::json!(1));
        obj.entry("gates").or_insert(serde_json::json!([]));
        obj.entry("features").or_insert(serde_json::json!([]));
        obj.entry("math").or_insert(serde_json::json!(""));
        obj.entry("failure_reason")
            .or_insert(serde_json::Value::Null);
        obj.entry("parent_id").or_insert(serde_json::Value::Null);

        // Normalize timestamps: old records may have flat completed/created fields
        if obj.get("timestamps").is_none() {
            let created = obj
                .get("created")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let completed = obj
                .get("completed")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !created.is_empty() || !completed.is_empty() {
                obj.insert(
                    "timestamps".to_string(),
                    serde_json::json!({
                        "created": created,
                        "completed": completed,
                    }),
                );
            }
        }

        debug!(
            id = obj.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
            "Normalized v0 record to v1"
        );
    } else if version > CURRENT_SCHEMA_VERSION {
        warn!(
            version,
            id = obj.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
            "Unknown schema version — record may contain unrecognized fields"
        );
    }

    record
}

/// Apply normalization to a batch of records.
fn normalize_batch(items: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    items.into_iter().map(normalize_record).collect()
}

// ---------------------------------------------------------------------------
// Query parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct HypothesisQuery {
    pub agent: Option<String>,
    pub generator: Option<String>,
    pub status: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CycleQuery {
    pub agent: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T: Serialize> {
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
    pub items: Vec<T>,
}

#[derive(Debug, Default, Serialize)]
pub struct ResearchStats {
    pub total_hypotheses: usize,
    pub by_status: std::collections::HashMap<String, usize>,
    pub by_agent: std::collections::HashMap<String, usize>,
    pub by_generator: std::collections::HashMap<String, usize>,
    pub total_cycles: usize,
}

#[derive(Debug, Serialize)]
pub struct HeatmapEntry {
    pub feature: String,
    pub horizon_s: f64,
    pub ic: f64,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct HeatmapResponse {
    pub entries: Vec<HeatmapEntry>,
    pub features: Vec<String>,
    pub horizons: Vec<f64>,
}

// ---------------------------------------------------------------------------
// SQLite query helpers
// ---------------------------------------------------------------------------

/// Query research_output table with filters. Returns (items, total).
async fn query_db(
    state: &AppState,
    kind: &str,
    agent: Option<&str>,
    generator: Option<&str>,
    status: Option<&str>,
    limit: usize,
    offset: usize,
) -> Option<(Vec<serde_json::Value>, usize)> {
    let db = state.research_db.as_ref()?;
    let conn = db.lock().await;

    // Build WHERE clause with positional params
    let mut conditions = vec!["kind = ?".to_string()];
    let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(kind.to_string())];

    if let Some(a) = agent {
        conditions.push("agent = ?".to_string());
        params.push(Box::new(a.to_string()));
    }
    if let Some(g) = generator {
        conditions.push("generator = ?".to_string());
        params.push(Box::new(g.to_string()));
    }
    if let Some(s) = status {
        conditions.push("status = ?".to_string());
        params.push(Box::new(s.to_string()));
    }

    let where_clause = conditions.join(" AND ");
    let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    // Count total
    let count_sql = format!(
        "SELECT COUNT(*) FROM research_output WHERE {}",
        where_clause
    );
    let total: usize = conn
        .query_row(
            &count_sql,
            rusqlite::params_from_iter(param_refs.iter()),
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as usize;

    // Fetch page
    let mut fetch_params = params;
    fetch_params.push(Box::new(limit as i64));
    fetch_params.push(Box::new(offset as i64));
    let fetch_refs: Vec<&dyn rusqlite::ToSql> = fetch_params.iter().map(|p| p.as_ref()).collect();

    let fetch_sql = format!(
        "SELECT payload FROM research_output WHERE {} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        where_clause
    );

    let mut stmt = match conn.prepare(&fetch_sql) {
        Ok(s) => s,
        Err(e) => {
            warn!("SQLite query error: {}", e);
            return None;
        }
    };

    let items: Vec<serde_json::Value> = stmt
        .query_map(rusqlite::params_from_iter(fetch_refs.iter()), |row| {
            row.get::<_, String>(0)
        })
        .ok()
        .map(|rows| {
            rows.filter_map(|r| r.ok())
                .filter_map(|p| serde_json::from_str(&p).ok())
                .collect()
        })
        .unwrap_or_default();

    Some((normalize_batch(items), total))
}

/// Get a single record by ID from SQLite.
async fn get_db_record(state: &AppState, id: &str) -> Option<serde_json::Value> {
    let db = state.research_db.as_ref()?;
    let conn = db.lock().await;
    let result = conn.query_row(
        "SELECT payload FROM research_output WHERE id = ?1",
        [id],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(payload) => serde_json::from_str(&payload).ok().map(normalize_record),
        Err(_) => None,
    }
}

/// Get all records of a kind (for stats/heatmap aggregation).
async fn get_all_db(state: &AppState, kind: &str) -> Option<Vec<serde_json::Value>> {
    let db = state.research_db.as_ref()?;
    let conn = db.lock().await;
    let mut stmt = conn
        .prepare("SELECT payload FROM research_output WHERE kind = ?1 ORDER BY created_at DESC")
        .ok()?;
    let items: Vec<serde_json::Value> = stmt
        .query_map([kind], |row| row.get::<_, String>(0))
        .ok()?
        .filter_map(|r| r.ok())
        .filter_map(|p| serde_json::from_str(&p).ok())
        .collect();
    Some(normalize_batch(items))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/research/hypotheses
/// Paginated list of hypothesis records, filterable by agent/generator/status.
pub async fn list_hypotheses(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HypothesisQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);

    let (items, total) = query_db(
        &state,
        "hypothesis",
        q.agent.as_deref(),
        q.generator.as_deref(),
        q.status.as_deref(),
        limit,
        offset,
    )
    .await
    .unwrap_or((Vec::new(), 0));

    Ok(Json(PaginatedResponse {
        total,
        offset,
        limit,
        items,
    }))
}

/// GET /api/research/hypotheses/:id
/// Full detail for a single hypothesis.
pub async fn get_hypothesis(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match get_db_record(&state, &id).await {
        Some(record) => Ok(Json(record)),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Hypothesis not found: {}", id),
            }),
        )),
    }
}

/// GET /api/research/cycles
/// Paginated list of cycle summaries.
pub async fn list_cycles(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CycleQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);

    let (items, total) = query_db(
        &state,
        "cycle",
        q.agent.as_deref(),
        None, // cycles don't filter by generator
        None,
        limit,
        offset,
    )
    .await
    .unwrap_or((Vec::new(), 0));

    Ok(Json(PaginatedResponse {
        total,
        offset,
        limit,
        items,
    }))
}

/// GET /api/research/signals
/// Registered signals (hypotheses with status=replicated).
pub async fn list_signals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    let (items, _) = query_db(
        &state,
        "hypothesis",
        None,
        None,
        Some("replicated"),
        1000,
        0,
    )
    .await
    .unwrap_or((Vec::new(), 0));
    Ok(Json(items))
}

/// GET /api/research/stats
/// Aggregate statistics across all hypotheses and cycles.
pub async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ResearchStats>, (StatusCode, Json<ErrorResponse>)> {
    let stats = compute_stats_from_db(&state).await.unwrap_or_default();
    Ok(Json(stats))
}

async fn compute_stats_from_db(state: &AppState) -> Option<ResearchStats> {
    let db = state.research_db.as_ref()?;
    let conn = db.lock().await;

    // Total hypotheses
    let total_hypotheses: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM research_output WHERE kind = 'hypothesis'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as usize;

    // Total cycles
    let total_cycles: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM research_output WHERE kind = 'cycle'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as usize;

    // By status
    let mut by_status = std::collections::HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT status, COUNT(*) FROM research_output WHERE kind = 'hypothesis' GROUP BY status",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for r in rows.flatten() {
                by_status.insert(r.0, r.1 as usize);
            }
        }
    }

    // By agent
    let mut by_agent = std::collections::HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT agent, COUNT(*) FROM research_output WHERE kind = 'hypothesis' GROUP BY agent",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for r in rows.flatten() {
                by_agent.insert(r.0, r.1 as usize);
            }
        }
    }

    // By generator
    let mut by_generator = std::collections::HashMap::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT generator, COUNT(*) FROM research_output WHERE kind = 'hypothesis' AND generator IS NOT NULL GROUP BY generator",
    ) {
        if let Ok(rows) = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }) {
            for r in rows.flatten() {
                by_generator.insert(r.0, r.1 as usize);
            }
        }
    }

    Some(ResearchStats {
        total_hypotheses,
        by_status,
        by_agent,
        by_generator,
        total_cycles,
    })
}

/// GET /api/research/heatmap
/// Feature x horizon IC matrix from hypothesis gate results.
pub async fn get_heatmap(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HeatmapResponse>, (StatusCode, Json<ErrorResponse>)> {
    let all = get_all_db(&state, "hypothesis").await.unwrap_or_default();

    let mut entries = Vec::new();
    let mut features_set = std::collections::BTreeSet::new();
    let mut horizons_set = std::collections::BTreeSet::new();

    for h in &all {
        let status = h
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let horizon = h.get("horizon_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let feats = h
            .get("features")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        // Extract IC from first gate that has a metric
        let ic = h
            .get("gates")
            .and_then(|g| g.as_array())
            .and_then(|gates| {
                gates.iter().find_map(|gate| {
                    let name = gate.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if name == "IC" || name == "dIC" {
                        gate.get("metric").and_then(|v| v.as_f64())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(0.0);

        for feat_val in &feats {
            if let Some(feat) = feat_val.as_str() {
                features_set.insert(feat.to_string());
                horizons_set.insert(ordered_float::OrderedFloat(horizon));
                entries.push(HeatmapEntry {
                    feature: feat.to_string(),
                    horizon_s: horizon,
                    ic,
                    status: status.clone(),
                });
            }
        }
    }

    Ok(Json(HeatmapResponse {
        entries,
        features: features_set.into_iter().collect(),
        horizons: horizons_set.into_iter().map(|f| f.into_inner()).collect(),
    }))
}

// ---------------------------------------------------------------------------
// Network endpoint — feature interaction graph from IT engine + hypotheses
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct NetworkNode {
    pub id: String,
    pub category: String,
    pub mi: std::collections::HashMap<String, f64>,
    pub cmi: std::collections::HashMap<String, f64>,
    pub interaction: f64,
    pub cost_viable: bool,
    pub hypothesis_count: usize,
    pub selected: bool,
}

#[derive(Debug, Serialize)]
pub struct NetworkEdge {
    pub source: String,
    pub target: String,
    pub weight: usize,
}

#[derive(Debug, Serialize)]
pub struct NetworkMeta {
    pub symbol: String,
    pub n_samples: u64,
    pub last_updated: String,
    pub total_features: usize,
}

#[derive(Debug, Serialize)]
pub struct NetworkResponse {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub meta: NetworkMeta,
}

/// Derive a category from feature name prefix.
fn feature_category(name: &str) -> &'static str {
    if name.starts_with("spread_") {
        return "spread";
    }
    if name.starts_with("depth_") {
        return "depth";
    }
    if name.starts_with("imb_") {
        return "imbalance";
    }
    if name.starts_with("flow_") {
        return "flow";
    }
    if name.starts_with("vol_") || name.starts_with("volatility_") {
        return "volatility";
    }
    if name.starts_with("ent_") {
        return "entropy";
    }
    if name.starts_with("trend_") {
        return "trend";
    }
    if name.starts_with("illiq_") {
        return "illiquidity";
    }
    if name.starts_with("tox_") {
        return "toxicity";
    }
    if name.starts_with("whale_") {
        return "whale";
    }
    if name.starts_with("liquidation_")
        || name.starts_with("largest_position")
        || name.starts_with("nearest_cluster")
        || name.starts_with("positions_at_risk")
    {
        return "liquidation";
    }
    if name.starts_with("top")
        || name.starts_with("herfindahl_")
        || name.starts_with("gini_")
        || name.starts_with("theil_")
    {
        return "concentration";
    }
    if name.starts_with("ctx_") {
        return "context";
    }
    if name.starts_with("raw_") {
        return "raw";
    }
    if name.starts_with("regime_") || name.starts_with("gmm_") {
        return "regime";
    }
    if name.starts_with("cross_") {
        return "cross_symbol";
    }
    "other"
}

/// Read IT engine state file for a symbol.
fn read_it_engine_state(dir: &std::path::Path, symbol: &str) -> Option<serde_json::Value> {
    let path = dir.join(format!("state_{}.json", symbol));
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Build network graph from IT engine state and hypothesis data.
fn build_network(
    it_state: &serde_json::Value,
    hypotheses: &[serde_json::Value],
) -> NetworkResponse {
    let mi_matrix = it_state.get("mi_matrix").and_then(|v| v.as_object());
    let cmi_matrix = it_state.get("cmi_matrix").and_then(|v| v.as_object());
    let interaction_map = it_state.get("interaction").and_then(|v| v.as_object());
    let cost_viable_map = it_state.get("cost_viable").and_then(|v| v.as_object());
    let selected_features: Vec<&str> = it_state
        .get("selected_features")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    // Count hypotheses per feature
    let mut feature_hyp_count: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    // Track co-occurrence pairs
    let mut cooccurrence: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();

    for h in hypotheses {
        let feats: Vec<String> = h
            .get("features")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for f in &feats {
            *feature_hyp_count.entry(f.clone()).or_default() += 1;
        }
        // Co-occurrence edges (undirected, sorted pair)
        for i in 0..feats.len() {
            for j in (i + 1)..feats.len() {
                let (a, b) = if feats[i] < feats[j] {
                    (feats[i].clone(), feats[j].clone())
                } else {
                    (feats[j].clone(), feats[i].clone())
                };
                *cooccurrence.entry((a, b)).or_default() += 1;
            }
        }
    }

    // Build nodes from MI matrix keys
    let mut nodes = Vec::new();
    if let Some(mi) = mi_matrix {
        for (feature, mi_val) in mi {
            let mi_horizons: std::collections::HashMap<String, f64> = mi_val
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                        .collect()
                })
                .unwrap_or_default();

            let cmi_horizons: std::collections::HashMap<String, f64> = cmi_matrix
                .and_then(|c| c.get(feature))
                .and_then(|v| v.as_object())
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                        .collect()
                })
                .unwrap_or_default();

            let interaction = interaction_map
                .and_then(|m| m.get(feature))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            let cost_viable = cost_viable_map
                .and_then(|m| m.get(feature))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let hypothesis_count = feature_hyp_count
                .get(feature.as_str())
                .copied()
                .unwrap_or(0);
            let selected = selected_features.contains(&feature.as_str());

            nodes.push(NetworkNode {
                id: feature.clone(),
                category: feature_category(feature).to_string(),
                mi: mi_horizons,
                cmi: cmi_horizons,
                interaction,
                cost_viable,
                hypothesis_count,
                selected,
            });
        }
    }

    // Sort nodes by category then name for stable output
    nodes.sort_by(|a, b| a.category.cmp(&b.category).then(a.id.cmp(&b.id)));

    // Build edges
    let edges: Vec<NetworkEdge> = cooccurrence
        .into_iter()
        .map(|((source, target), weight)| NetworkEdge {
            source,
            target,
            weight,
        })
        .collect();

    let meta = NetworkMeta {
        symbol: it_state
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("BTC")
            .to_string(),
        n_samples: it_state
            .get("n_samples")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        last_updated: it_state
            .get("last_updated")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        total_features: nodes.len(),
    };

    NetworkResponse { nodes, edges, meta }
}

#[derive(Debug, Deserialize)]
pub struct NetworkQuery {
    pub symbol: Option<String>,
}

/// GET /api/research/network
/// Feature interaction graph from IT engine state + hypothesis co-occurrence.
pub async fn get_network(
    State(state): State<Arc<AppState>>,
    Query(q): Query<NetworkQuery>,
) -> Result<Json<NetworkResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = q.symbol.as_deref().unwrap_or("BTC");
    let it_dir = PathBuf::from(&state.config.it_engine_data_dir);

    let it_state = read_it_engine_state(&it_dir, symbol).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No IT engine state for symbol: {}", symbol),
            }),
        )
    })?;

    let hypotheses = get_all_db(&state, "hypothesis").await.unwrap_or_default();

    let network = build_network(&it_state, &hypotheses);
    Ok(Json(network))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::fs;
    use tokio::sync::Mutex;

    /// Create an in-memory SQLite DB with research_output table and return it wrapped.
    fn test_db() -> Arc<Mutex<Connection>> {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE research_output (
                id              TEXT PRIMARY KEY,
                kind            TEXT NOT NULL,
                agent           TEXT NOT NULL,
                generator       TEXT,
                status          TEXT,
                payload         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                schema_version  INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX idx_ro_agent ON research_output(agent);
            CREATE INDEX idx_ro_kind ON research_output(kind);
            CREATE INDEX idx_ro_created ON research_output(created_at DESC);
            CREATE INDEX idx_ro_status ON research_output(status);",
        )
        .unwrap();
        Arc::new(Mutex::new(conn))
    }

    fn insert_hypothesis(
        db: &Arc<Mutex<Connection>>,
        id: &str,
        agent: &str,
        gen: &str,
        status: &str,
    ) {
        let conn = db.blocking_lock();
        let record = serde_json::json!({
            "schema_version": 1,
            "id": id,
            "agent": agent,
            "generator": gen,
            "claim": format!("Test claim for {}", id),
            "math": "IC = ...",
            "status": status,
            "failure_reason": if status == "failed" { Some("no_effect") } else { None::<&str> },
            "gates": [{
                "name": "IC",
                "passed": status != "failed",
                "message": "PASS IC=0.08 vs min=0.03 p=0.001",
                "metric": 0.08,
                "threshold": 0.03,
                "p_value": 0.001
            }],
            "features": ["ent_book_shape"],
            "regime_gate": "ent_book_shape<0.4",
            "horizon_s": 5.0,
            "thresholds": {"horizon_s": 5.0},
            "parent_id": null,
            "timestamps": {
                "created": "2026-05-25T10:00:00+00:00",
                "completed": "2026-05-25T10:01:00+00:00"
            }
        });
        let payload = serde_json::to_string(&record).unwrap();
        conn.execute(
            "INSERT INTO research_output (id, kind, agent, generator, status, payload, created_at, schema_version)
             VALUES (?1, 'hypothesis', ?2, ?3, ?4, ?5, '2026-05-25T10:01:00+00:00', 1)",
            rusqlite::params![id, agent, gen, status, payload],
        ).unwrap();
    }

    fn insert_cycle(db: &Arc<Mutex<Connection>>, cycle_id: &str, agent: &str, n_tested: usize) {
        let conn = db.blocking_lock();
        let record = serde_json::json!({
            "schema_version": 1,
            "cycle_id": cycle_id,
            "agent": agent,
            "started": "2026-05-25T10:00:00+00:00",
            "completed": "2026-05-25T10:05:00+00:00",
            "duration_s": 300.0,
            "n_tested": n_tested,
            "n_registered": 1,
            "n_fdr_rejected": 0,
            "n_chained": 0,
            "fdr_q": 0.05,
            "hypotheses": [],
            "generator_stats": {}
        });
        let payload = serde_json::to_string(&record).unwrap();
        conn.execute(
            "INSERT INTO research_output (id, kind, agent, generator, status, payload, created_at, schema_version)
             VALUES (?1, 'cycle', ?2, NULL, NULL, ?3, '2026-05-25T10:05:00+00:00', 1)",
            rusqlite::params![cycle_id, agent, payload],
        ).unwrap();
    }

    /// Direct SQLite queries for testing — bypass AppState dependency.
    fn test_query_db(
        db: &Arc<Mutex<Connection>>,
        kind: &str,
        agent: Option<&str>,
        generator: Option<&str>,
        status: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> (Vec<serde_json::Value>, usize) {
        let conn = db.blocking_lock();

        // Build WHERE clause
        let mut conditions = vec!["kind = ?".to_string()];
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(kind.to_string())];

        if let Some(a) = agent {
            conditions.push("agent = ?".to_string());
            params_vec.push(Box::new(a.to_string()));
        }
        if let Some(g) = generator {
            conditions.push("generator = ?".to_string());
            params_vec.push(Box::new(g.to_string()));
        }
        if let Some(s) = status {
            conditions.push("status = ?".to_string());
            params_vec.push(Box::new(s.to_string()));
        }

        let where_clause = conditions.join(" AND ");
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        // Count
        let count_sql = format!(
            "SELECT COUNT(*) FROM research_output WHERE {}",
            where_clause
        );
        let total: i64 = conn
            .query_row(
                &count_sql,
                rusqlite::params_from_iter(params_refs.iter()),
                |row| row.get(0),
            )
            .unwrap_or(0);

        // Fetch
        let mut fetch_params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for p in &params_vec {
            // Re-extract string values
            fetch_params.push(Box::new(format!("{}", ""))); // placeholder
        }
        // Simpler approach: just build the full query with all params
        let mut all_params: Vec<Box<dyn rusqlite::ToSql>> = params_vec;
        all_params.push(Box::new(limit as i64));
        all_params.push(Box::new(offset as i64));
        let all_refs: Vec<&dyn rusqlite::ToSql> = all_params.iter().map(|p| p.as_ref()).collect();

        let fetch_sql = format!(
            "SELECT payload FROM research_output WHERE {} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            where_clause
        );
        let mut stmt = conn.prepare(&fetch_sql).unwrap();
        let items: Vec<serde_json::Value> = stmt
            .query_map(rusqlite::params_from_iter(all_refs.iter()), |row| {
                row.get::<_, String>(0)
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .filter_map(|p| serde_json::from_str(&p).ok())
            .collect();

        (items, total as usize)
    }

    fn test_get_record(db: &Arc<Mutex<Connection>>, id: &str) -> Option<serde_json::Value> {
        let conn = db.blocking_lock();
        let result = conn.query_row(
            "SELECT payload FROM research_output WHERE id = ?1",
            [id],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(payload) => serde_json::from_str(&payload).ok(),
            Err(_) => None,
        }
    }

    fn test_get_all(db: &Arc<Mutex<Connection>>, kind: &str) -> Vec<serde_json::Value> {
        let conn = db.blocking_lock();
        let mut stmt = conn
            .prepare("SELECT payload FROM research_output WHERE kind = ?1 ORDER BY created_at DESC")
            .unwrap();
        stmt.query_map([kind], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .filter_map(|p| serde_json::from_str(&p).ok())
            .collect()
    }

    /// Compute aggregate stats directly via SQL — mirrors `compute_stats_from_db`
    /// without the AppState dependency.
    fn test_compute_stats(db: &Arc<Mutex<Connection>>) -> ResearchStats {
        let conn = db.blocking_lock();

        let total_hypotheses: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM research_output WHERE kind = 'hypothesis'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as usize;
        let total_cycles: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM research_output WHERE kind = 'cycle'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0) as usize;

        let group_count = |sql: &str| -> std::collections::HashMap<String, usize> {
            let mut map = std::collections::HashMap::new();
            if let Ok(mut stmt) = conn.prepare(sql) {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                }) {
                    for r in rows.flatten() {
                        map.insert(r.0, r.1 as usize);
                    }
                }
            }
            map
        };

        ResearchStats {
            total_hypotheses,
            by_status: group_count(
                "SELECT status, COUNT(*) FROM research_output WHERE kind = 'hypothesis' GROUP BY status",
            ),
            by_agent: group_count(
                "SELECT agent, COUNT(*) FROM research_output WHERE kind = 'hypothesis' GROUP BY agent",
            ),
            by_generator: group_count(
                "SELECT generator, COUNT(*) FROM research_output WHERE kind = 'hypothesis' AND generator IS NOT NULL GROUP BY generator",
            ),
            total_cycles,
        }
    }

    // -----------------------------------------------------------------------
    // Schema normalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_v0_record_backfills_defaults() {
        let record = serde_json::json!({
            "id": "H-OLD-001",
            "agent": "micro",
            "status": "failed",
            "created": "2026-01-01T00:00:00",
            "completed": "2026-01-01T00:01:00"
        });
        let normalized = normalize_record(record);
        assert_eq!(normalized["schema_version"], 1);
        assert!(normalized["gates"].is_array());
        assert!(normalized["features"].is_array());
        assert_eq!(normalized["math"], "");
        assert!(normalized["failure_reason"].is_null());
        assert!(normalized["parent_id"].is_null());
        // Timestamps normalized from flat fields
        assert_eq!(normalized["timestamps"]["created"], "2026-01-01T00:00:00");
        assert_eq!(normalized["timestamps"]["completed"], "2026-01-01T00:01:00");
    }

    #[test]
    fn test_normalize_v1_record_unchanged() {
        let record = serde_json::json!({
            "schema_version": 1,
            "id": "H1",
            "agent": "micro",
            "gates": [{"name": "IC", "passed": true}],
            "features": ["ent_book_shape"],
            "timestamps": {"created": "t1", "completed": "t2"}
        });
        let normalized = normalize_record(record.clone());
        assert_eq!(normalized, record);
    }

    #[test]
    fn test_normalize_future_version_preserved() {
        let record = serde_json::json!({
            "schema_version": 99,
            "id": "H-FUTURE",
            "new_field": "future data"
        });
        let normalized = normalize_record(record.clone());
        // Should pass through unchanged (with a warning logged)
        assert_eq!(normalized["schema_version"], 99);
        assert_eq!(normalized["new_field"], "future data");
    }

    #[test]
    fn test_normalize_v0_preserves_existing_timestamps() {
        // If timestamps block already exists, don't overwrite from flat fields
        let record = serde_json::json!({
            "id": "H1",
            "timestamps": {"created": "orig", "completed": "orig"},
            "created": "flat",
            "completed": "flat"
        });
        let normalized = normalize_record(record);
        assert_eq!(normalized["timestamps"]["created"], "orig");
    }

    #[test]
    fn test_normalize_batch() {
        let items = vec![
            serde_json::json!({"id": "H1", "agent": "micro"}),
            serde_json::json!({"schema_version": 1, "id": "H2", "agent": "micro"}),
        ];
        let normalized = normalize_batch(items);
        assert_eq!(normalized[0]["schema_version"], 1); // backfilled
        assert_eq!(normalized[1]["schema_version"], 1); // already present
    }

    // -----------------------------------------------------------------------
    // SQLite query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_db_hypotheses() {
        let db = test_db();
        insert_hypothesis(&db, "H1", "micro", "systematic", "replicated");
        insert_hypothesis(&db, "H2", "micro", "spectral", "failed");
        insert_hypothesis(&db, "H3", "macro", "systematic", "replicated");

        // All hypotheses
        let (items, total) = test_query_db(&db, "hypothesis", None, None, None, 50, 0);
        assert_eq!(total, 3);
        assert_eq!(items.len(), 3);

        // Filter by agent
        let (items, total) = test_query_db(&db, "hypothesis", Some("micro"), None, None, 50, 0);
        assert_eq!(total, 2);
        assert_eq!(items.len(), 2);

        // Filter by status
        let (items, total) =
            test_query_db(&db, "hypothesis", None, None, Some("replicated"), 50, 0);
        assert_eq!(total, 2);
        assert_eq!(items.len(), 2);

        // Filter by generator
        let (items, total) =
            test_query_db(&db, "hypothesis", None, Some("systematic"), None, 50, 0);
        assert_eq!(total, 2);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_query_db_pagination() {
        let db = test_db();
        for i in 0..10 {
            insert_hypothesis(&db, &format!("H{}", i), "micro", "systematic", "failed");
        }

        let (items, total) = test_query_db(&db, "hypothesis", None, None, None, 3, 0);
        assert_eq!(total, 10);
        assert_eq!(items.len(), 3);

        let (items, _) = test_query_db(&db, "hypothesis", None, None, None, 3, 8);
        assert_eq!(items.len(), 2); // only 2 left
    }

    #[test]
    fn test_query_db_cycles() {
        let db = test_db();
        insert_cycle(&db, "CYC-1", "micro", 10);
        insert_cycle(&db, "CYC-2", "macro", 5);

        let (items, total) = test_query_db(&db, "cycle", None, None, None, 50, 0);
        assert_eq!(total, 2);
        assert_eq!(items.len(), 2);

        let (items, total) = test_query_db(&db, "cycle", Some("micro"), None, None, 50, 0);
        assert_eq!(total, 1);
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_get_db_record() {
        let db = test_db();
        insert_hypothesis(&db, "H-TEST-1", "micro", "systematic", "replicated");

        let record = test_get_record(&db, "H-TEST-1");
        assert!(record.is_some());
        assert_eq!(record.unwrap()["id"], "H-TEST-1");

        let missing = test_get_record(&db, "NONEXISTENT");
        assert!(missing.is_none());
    }

    #[test]
    fn test_get_all_db() {
        let db = test_db();
        insert_hypothesis(&db, "H1", "micro", "systematic", "replicated");
        insert_hypothesis(&db, "H2", "micro", "spectral", "failed");
        insert_cycle(&db, "C1", "micro", 5);

        let hyps = test_get_all(&db, "hypothesis");
        assert_eq!(hyps.len(), 2);

        let cycles = test_get_all(&db, "cycle");
        assert_eq!(cycles.len(), 1);
    }

    #[test]
    fn test_stats_aggregation_from_db() {
        let db = test_db();
        insert_hypothesis(&db, "H1", "micro", "systematic", "replicated");
        insert_hypothesis(&db, "H2", "micro", "spectral", "failed");
        insert_hypothesis(&db, "H3", "macro", "systematic", "replicated");
        insert_cycle(&db, "C1", "micro", 10);
        insert_cycle(&db, "C2", "macro", 5);

        let stats = test_compute_stats(&db);

        assert_eq!(stats.total_hypotheses, 3);
        assert_eq!(stats.total_cycles, 2);
        assert_eq!(stats.by_status["replicated"], 2);
        assert_eq!(stats.by_status["failed"], 1);
        assert_eq!(stats.by_agent["micro"], 2);
        assert_eq!(stats.by_agent["macro"], 1);
        assert_eq!(stats.by_generator["systematic"], 2);
        assert_eq!(stats.by_generator["spectral"], 1);
    }

    // -----------------------------------------------------------------------
    // Network / feature_category
    // -----------------------------------------------------------------------

    #[test]
    fn test_feature_category() {
        assert_eq!(feature_category("spread_best_bid_ask"), "spread");
        assert_eq!(feature_category("depth_total_bid"), "depth");
        assert_eq!(feature_category("imb_size_ratio"), "imbalance");
        assert_eq!(feature_category("flow_vwap_5s"), "flow");
        assert_eq!(feature_category("vol_realized_10s"), "volatility");
        assert_eq!(feature_category("ent_book_shape"), "entropy");
        assert_eq!(feature_category("trend_ema_short"), "trend");
        assert_eq!(feature_category("illiq_amihud"), "illiquidity");
        assert_eq!(feature_category("tox_vpin"), "toxicity");
        assert_eq!(feature_category("whale_net_flow_1h"), "whale");
        assert_eq!(feature_category("liquidation_intensity"), "liquidation");
        assert_eq!(feature_category("top5_concentration"), "concentration");
        assert_eq!(feature_category("unknown_feature"), "other");
    }

    #[test]
    fn test_build_network_basic() {
        let it_state = serde_json::json!({
            "mi_matrix": {
                "spread_ba": {"10t": 0.05, "50t": 0.03},
                "depth_bid": {"10t": 0.02, "50t": 0.01},
                "ent_shape": {"10t": 0.0, "50t": 0.0}
            },
            "cmi_matrix": {
                "spread_ba": {"10t": 0.04, "50t": 0.02},
                "depth_bid": {"10t": 0.01, "50t": 0.005}
            },
            "interaction": {"spread_ba": 0.003, "depth_bid": -0.001},
            "cost_viable": {"spread_ba": true, "depth_bid": false},
            "selected_features": ["spread_ba"],
            "symbol": "BTC",
            "n_samples": 6000,
            "last_updated": "2026-05-21T11:00:00"
        });

        let hypotheses = vec![
            serde_json::json!({"features": ["spread_ba", "depth_bid"]}),
            serde_json::json!({"features": ["spread_ba", "ent_shape"]}),
            serde_json::json!({"features": ["spread_ba"]}),
        ];

        let net = build_network(&it_state, &hypotheses);
        assert_eq!(net.nodes.len(), 3);
        assert_eq!(net.meta.symbol, "BTC");

        let spread_node = net.nodes.iter().find(|n| n.id == "spread_ba").unwrap();
        assert!(spread_node.cost_viable);
        assert!(spread_node.selected);
        assert_eq!(spread_node.hypothesis_count, 3);
    }

    #[test]
    fn test_build_network_empty() {
        let it_state = serde_json::json!({
            "mi_matrix": {},
            "symbol": "SOL",
            "n_samples": 0,
            "last_updated": ""
        });
        let net = build_network(&it_state, &[]);
        assert!(net.nodes.is_empty());
        assert!(net.edges.is_empty());
    }

    #[test]
    fn test_read_it_engine_state() {
        let tmp = tempfile::tempdir().unwrap();
        let state = serde_json::json!({"mi_matrix": {"feat_a": {"10t": 0.1}}, "symbol": "BTC"});
        fs::write(
            tmp.path().join("state_BTC.json"),
            serde_json::to_string(&state).unwrap(),
        )
        .unwrap();
        let loaded = read_it_engine_state(tmp.path(), "BTC");
        assert!(loaded.is_some());
        assert!(read_it_engine_state(tmp.path(), "ETH").is_none());
    }

    // -----------------------------------------------------------------------
    // API contract tests — verify response shapes match frontend TypeScript
    // interfaces defined in web/src/lib/api.ts
    // -----------------------------------------------------------------------

    /// Assert that a hypothesis record has every field the frontend Hypothesis
    /// interface expects, with correct types.
    fn assert_hypothesis_shape(h: &serde_json::Value) {
        assert!(h["id"].is_string(), "Hypothesis.id must be string");
        assert!(h["agent"].is_string(), "Hypothesis.agent must be string");
        assert!(
            h["generator"].is_string(),
            "Hypothesis.generator must be string"
        );
        assert!(h["claim"].is_string(), "Hypothesis.claim must be string");
        assert!(h["math"].is_string(), "Hypothesis.math must be string");
        assert!(h["status"].is_string(), "Hypothesis.status must be string");
        // failure_reason: string | null
        assert!(
            h["failure_reason"].is_string() || h["failure_reason"].is_null(),
            "Hypothesis.failure_reason must be string|null"
        );
        // gates: Gate[]
        assert!(h["gates"].is_array(), "Hypothesis.gates must be array");
        for gate in h["gates"].as_array().unwrap() {
            assert!(gate["name"].is_string(), "Gate.name must be string");
            assert!(gate["passed"].is_boolean(), "Gate.passed must be boolean");
            assert!(gate["message"].is_string(), "Gate.message must be string");
            assert!(
                gate["metric"].is_number() || gate["metric"].is_null(),
                "Gate.metric must be number|null"
            );
            assert!(
                gate["threshold"].is_number() || gate["threshold"].is_null(),
                "Gate.threshold must be number|null"
            );
            assert!(
                gate["p_value"].is_number() || gate["p_value"].is_null(),
                "Gate.p_value must be number|null"
            );
        }
        // features: string[]
        assert!(
            h["features"].is_array(),
            "Hypothesis.features must be array"
        );
        for f in h["features"].as_array().unwrap() {
            assert!(f.is_string(), "features[] items must be strings");
        }
        // regime_gate: string | null
        assert!(
            h["regime_gate"].is_string() || h["regime_gate"].is_null(),
            "Hypothesis.regime_gate must be string|null"
        );
        // horizon_s: number | null
        assert!(
            h["horizon_s"].is_number() || h["horizon_s"].is_null(),
            "Hypothesis.horizon_s must be number|null"
        );
        // thresholds: Record<string, unknown>
        assert!(
            h["thresholds"].is_object(),
            "Hypothesis.thresholds must be object"
        );
        // parent_id: string | null
        assert!(
            h["parent_id"].is_string() || h["parent_id"].is_null(),
            "Hypothesis.parent_id must be string|null"
        );
        // timestamps: { created: string; completed: string | null }
        assert!(
            h["timestamps"].is_object(),
            "Hypothesis.timestamps must be object"
        );
        assert!(
            h["timestamps"]["created"].is_string(),
            "timestamps.created must be string"
        );
        assert!(
            h["timestamps"]["completed"].is_string() || h["timestamps"]["completed"].is_null(),
            "timestamps.completed must be string|null"
        );
    }

    /// Assert that a cycle record has every field the frontend CycleSummary
    /// interface expects.
    fn assert_cycle_shape(c: &serde_json::Value) {
        assert!(
            c["cycle_id"].is_string(),
            "CycleSummary.cycle_id must be string"
        );
        assert!(c["agent"].is_string(), "CycleSummary.agent must be string");
        assert!(
            c["started"].is_string(),
            "CycleSummary.started must be string"
        );
        assert!(
            c["completed"].is_string(),
            "CycleSummary.completed must be string"
        );
        assert!(
            c["duration_s"].is_number(),
            "CycleSummary.duration_s must be number"
        );
        assert!(
            c["n_tested"].is_number(),
            "CycleSummary.n_tested must be number"
        );
        assert!(
            c["n_registered"].is_number(),
            "CycleSummary.n_registered must be number"
        );
        assert!(
            c["n_fdr_rejected"].is_number(),
            "CycleSummary.n_fdr_rejected must be number"
        );
        assert!(
            c["n_chained"].is_number(),
            "CycleSummary.n_chained must be number"
        );
        assert!(c["fdr_q"].is_number(), "CycleSummary.fdr_q must be number");
        assert!(
            c["hypotheses"].is_array(),
            "CycleSummary.hypotheses must be array"
        );
        assert!(
            c["generator_stats"].is_object(),
            "CycleSummary.generator_stats must be object"
        );
    }

    #[test]
    fn contract_hypothesis_record_shape() {
        let db = test_db();
        insert_hypothesis(&db, "H-CONTRACT-1", "micro", "systematic", "replicated");
        insert_hypothesis(&db, "H-CONTRACT-2", "macro", "spectral", "failed");

        let record = test_get_record(&db, "H-CONTRACT-1").unwrap();
        let normalized = normalize_record(record);
        assert_hypothesis_shape(&normalized);

        let failed = test_get_record(&db, "H-CONTRACT-2").unwrap();
        let normalized_failed = normalize_record(failed);
        assert_hypothesis_shape(&normalized_failed);
        assert!(normalized_failed["failure_reason"].is_string());
    }

    #[test]
    fn contract_cycle_record_shape() {
        let db = test_db();
        insert_cycle(&db, "CYC-CONTRACT-1", "micro", 10);

        let record = test_get_record(&db, "CYC-CONTRACT-1").unwrap();
        assert_cycle_shape(&record);
    }

    #[test]
    fn contract_paginated_response_shape() {
        let items: Vec<serde_json::Value> = (0..2).map(|i| serde_json::json!({"i": i})).collect();
        let resp = PaginatedResponse {
            total: 5,
            offset: 1,
            limit: 2,
            items,
        };
        let json = serde_json::to_value(&resp).unwrap();

        assert!(
            json["items"].is_array(),
            "PaginatedResponse.items must be array"
        );
        assert!(
            json["total"].is_number(),
            "PaginatedResponse.total must be number"
        );
        assert!(
            json["offset"].is_number(),
            "PaginatedResponse.offset must be number"
        );
        assert!(
            json["limit"].is_number(),
            "PaginatedResponse.limit must be number"
        );
        assert_eq!(json["total"], 5);
        assert_eq!(json["offset"], 1);
        assert_eq!(json["limit"], 2);
        assert_eq!(json["items"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn contract_stats_response_shape() {
        let db = test_db();
        insert_hypothesis(&db, "H1", "micro", "systematic", "replicated");
        insert_hypothesis(&db, "H2", "macro", "spectral", "failed");
        insert_cycle(&db, "C1", "micro", 5);
        insert_cycle(&db, "C2", "macro", 3);
        insert_cycle(&db, "C3", "micro", 7);
        let stats = test_compute_stats(&db);
        let json = serde_json::to_value(&stats).unwrap();

        assert!(
            json["total_hypotheses"].is_number(),
            "ResearchStats.total_hypotheses must be number"
        );
        assert!(
            json["total_cycles"].is_number(),
            "ResearchStats.total_cycles must be number"
        );
        assert!(
            json["by_status"].is_object(),
            "ResearchStats.by_status must be object"
        );
        assert!(
            json["by_agent"].is_object(),
            "ResearchStats.by_agent must be object"
        );
        assert!(
            json["by_generator"].is_object(),
            "ResearchStats.by_generator must be object"
        );
        // Values in maps must be numbers
        for (_, v) in json["by_status"].as_object().unwrap() {
            assert!(v.is_number(), "by_status values must be numbers");
        }
    }

    #[test]
    fn contract_heatmap_response_shape() {
        let resp = HeatmapResponse {
            entries: vec![HeatmapEntry {
                feature: "ent_book_shape".to_string(),
                horizon_s: 5.0,
                ic: 0.08,
                status: "replicated".to_string(),
            }],
            features: vec!["ent_book_shape".to_string()],
            horizons: vec![5.0],
        };
        let json = serde_json::to_value(&resp).unwrap();

        assert!(
            json["entries"].is_array(),
            "HeatmapResponse.entries must be array"
        );
        assert!(
            json["features"].is_array(),
            "HeatmapResponse.features must be array"
        );
        assert!(
            json["horizons"].is_array(),
            "HeatmapResponse.horizons must be array"
        );
        let entry = &json["entries"][0];
        assert!(
            entry["feature"].is_string(),
            "HeatmapEntry.feature must be string"
        );
        assert!(
            entry["horizon_s"].is_number(),
            "HeatmapEntry.horizon_s must be number"
        );
        assert!(entry["ic"].is_number(), "HeatmapEntry.ic must be number");
        assert!(
            entry["status"].is_string(),
            "HeatmapEntry.status must be string"
        );
    }

    #[test]
    fn contract_network_response_shape() {
        let it_state = serde_json::json!({
            "mi_matrix": {"spread_ba": {"10t": 0.05}},
            "cmi_matrix": {"spread_ba": {"10t": 0.04}},
            "interaction": {"spread_ba": 0.003},
            "cost_viable": {"spread_ba": true},
            "selected_features": ["spread_ba"],
            "symbol": "BTC",
            "n_samples": 6000,
            "last_updated": "2026-05-21T11:00:00"
        });
        let hypotheses = vec![serde_json::json!({"features": ["spread_ba"]})];
        let net = build_network(&it_state, &hypotheses);
        let json = serde_json::to_value(&net).unwrap();

        // NetworkResponse top level
        assert!(
            json["nodes"].is_array(),
            "NetworkResponse.nodes must be array"
        );
        assert!(
            json["edges"].is_array(),
            "NetworkResponse.edges must be array"
        );
        assert!(
            json["meta"].is_object(),
            "NetworkResponse.meta must be object"
        );

        // NetworkMeta
        let meta = &json["meta"];
        assert!(
            meta["symbol"].is_string(),
            "NetworkMeta.symbol must be string"
        );
        assert!(
            meta["n_samples"].is_number(),
            "NetworkMeta.n_samples must be number"
        );
        assert!(
            meta["last_updated"].is_string(),
            "NetworkMeta.last_updated must be string"
        );
        assert!(
            meta["total_features"].is_number(),
            "NetworkMeta.total_features must be number"
        );

        // NetworkNode
        let node = &json["nodes"][0];
        assert!(node["id"].is_string(), "NetworkNode.id must be string");
        assert!(
            node["category"].is_string(),
            "NetworkNode.category must be string"
        );
        assert!(node["mi"].is_object(), "NetworkNode.mi must be object");
        assert!(node["cmi"].is_object(), "NetworkNode.cmi must be object");
        assert!(
            node["interaction"].is_number(),
            "NetworkNode.interaction must be number"
        );
        assert!(
            node["cost_viable"].is_boolean(),
            "NetworkNode.cost_viable must be boolean"
        );
        assert!(
            node["hypothesis_count"].is_number(),
            "NetworkNode.hypothesis_count must be number"
        );
        assert!(
            node["selected"].is_boolean(),
            "NetworkNode.selected must be boolean"
        );
    }
}
