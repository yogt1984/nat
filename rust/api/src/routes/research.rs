//! Research API routes — hypothesis history, cycle reports, signal registry, stats.
//!
//! Primary data source: SQLite `research_output` table (written by Python agents).
//! Fallback: JSON file scanning from `data/research/` (legacy).

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

#[derive(Debug, Serialize)]
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
    let count_sql = format!("SELECT COUNT(*) FROM research_output WHERE {}", where_clause);
    let total: usize = conn
        .query_row(&count_sql, rusqlite::params_from_iter(param_refs.iter()), |row| {
            row.get::<_, i64>(0)
        })
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

    Some((items, total))
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
        Ok(payload) => serde_json::from_str(&payload).ok(),
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
    Some(items)
}

// ---------------------------------------------------------------------------
// JSON fallback helpers (used when SQLite is unavailable)
// ---------------------------------------------------------------------------

fn research_dir(state: &AppState) -> PathBuf {
    PathBuf::from(&state.config.research_data_dir)
}

/// Read and parse all JSON files in a directory.
fn read_json_dir(dir: &std::path::Path) -> Vec<serde_json::Value> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut items: Vec<serde_json::Value> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .filter(|e| {
            // Skip .tmp files (atomic write pattern)
            !e.path()
                .file_stem()
                .map(|s| s.to_string_lossy().ends_with(".json"))
                .unwrap_or(false)
        })
        .filter_map(|e| {
            let data = std::fs::read_to_string(e.path()).ok()?;
            serde_json::from_str(&data).ok()
        })
        .collect();

    // Sort by completed/started timestamp descending (newest first)
    items.sort_by(|a, b| {
        let ts_a = a
            .get("timestamps")
            .and_then(|t| t.get("completed"))
            .and_then(|v| v.as_str())
            .or_else(|| a.get("completed").and_then(|v| v.as_str()))
            .unwrap_or("");
        let ts_b = b
            .get("timestamps")
            .and_then(|t| t.get("completed"))
            .and_then(|v| v.as_str())
            .or_else(|| b.get("completed").and_then(|v| v.as_str()))
            .unwrap_or("");
        ts_b.cmp(ts_a)
    });

    items
}

/// Filter hypothesis records by query parameters.
fn filter_hypotheses(items: &[serde_json::Value], q: &HypothesisQuery) -> Vec<serde_json::Value> {
    items
        .iter()
        .filter(|h| {
            if let Some(ref agent) = q.agent {
                if h.get("agent").and_then(|v| v.as_str()) != Some(agent.as_str()) {
                    return false;
                }
            }
            if let Some(ref gen) = q.generator {
                if h.get("generator").and_then(|v| v.as_str()) != Some(gen.as_str()) {
                    return false;
                }
            }
            if let Some(ref status) = q.status {
                if h.get("status").and_then(|v| v.as_str()) != Some(status.as_str()) {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect()
}

/// Filter cycle records by query parameters.
fn filter_cycles(items: &[serde_json::Value], q: &CycleQuery) -> Vec<serde_json::Value> {
    items
        .iter()
        .filter(|c| {
            if let Some(ref agent) = q.agent {
                if c.get("agent").and_then(|v| v.as_str()) != Some(agent.as_str()) {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect()
}

fn paginate(items: Vec<serde_json::Value>, offset: usize, limit: usize) -> PaginatedResponse<serde_json::Value> {
    let total = items.len();
    let paged: Vec<serde_json::Value> = items.into_iter().skip(offset).take(limit).collect();
    PaginatedResponse {
        total,
        offset,
        limit,
        items: paged,
    }
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

    // Try SQLite first
    if let Some((items, total)) = query_db(
        &state,
        "hypothesis",
        q.agent.as_deref(),
        q.generator.as_deref(),
        q.status.as_deref(),
        limit,
        offset,
    ).await {
        debug!(total, "Served hypotheses from SQLite");
        return Ok(Json(PaginatedResponse { total, offset, limit, items }));
    }

    // Fallback to JSON files
    debug!("Falling back to JSON file scan for hypotheses");
    let hyp_dir = research_dir(&state).join("hypotheses");
    let all = read_json_dir(&hyp_dir);
    let filtered = filter_hypotheses(&all, &q);
    Ok(Json(paginate(filtered, offset, limit)))
}

/// GET /api/research/hypotheses/:id
/// Full detail for a single hypothesis.
pub async fn get_hypothesis(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Try SQLite first
    if let Some(record) = get_db_record(&state, &id).await {
        return Ok(Json(record));
    }

    // Fallback to direct file read
    let path = research_dir(&state).join("hypotheses").join(format!("{}.json", id));
    let data = std::fs::read_to_string(&path).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Hypothesis not found: {}", id),
            }),
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&data).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Invalid JSON: {}", e),
            }),
        )
    })?;
    Ok(Json(value))
}

/// GET /api/research/cycles
/// Paginated list of cycle summaries.
pub async fn list_cycles(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CycleQuery>,
) -> Result<Json<PaginatedResponse<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);

    // Try SQLite first
    if let Some((items, total)) = query_db(
        &state,
        "cycle",
        q.agent.as_deref(),
        None,  // cycles don't filter by generator
        None,
        limit,
        offset,
    ).await {
        debug!(total, "Served cycles from SQLite");
        return Ok(Json(PaginatedResponse { total, offset, limit, items }));
    }

    // Fallback to JSON files
    debug!("Falling back to JSON file scan for cycles");
    let cyc_dir = research_dir(&state).join("cycles");
    let all = read_json_dir(&cyc_dir);
    let filtered = filter_cycles(&all, &q);
    Ok(Json(paginate(filtered, offset, limit)))
}

/// GET /api/research/signals
/// Registered signals (hypotheses with status=replicated).
pub async fn list_signals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    // Try SQLite first
    if let Some((items, _)) = query_db(&state, "hypothesis", None, None, Some("replicated"), 1000, 0).await {
        return Ok(Json(items));
    }

    // Fallback
    let hyp_dir = research_dir(&state).join("hypotheses");
    let all = read_json_dir(&hyp_dir);
    let signals: Vec<serde_json::Value> = all
        .into_iter()
        .filter(|h| h.get("status").and_then(|v| v.as_str()) == Some("replicated"))
        .collect();
    Ok(Json(signals))
}

/// GET /api/research/stats
/// Aggregate statistics across all hypotheses and cycles.
pub async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ResearchStats>, (StatusCode, Json<ErrorResponse>)> {
    // Try SQLite aggregation
    if let Some(stats) = compute_stats_from_db(&state).await {
        return Ok(Json(stats));
    }

    // Fallback to JSON
    let hyp_dir = research_dir(&state).join("hypotheses");
    let cyc_dir = research_dir(&state).join("cycles");
    let hypotheses = read_json_dir(&hyp_dir);
    let cycles = read_json_dir(&cyc_dir);

    Ok(Json(compute_stats_from_items(&hypotheses, cycles.len())))
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

fn compute_stats_from_items(hypotheses: &[serde_json::Value], total_cycles: usize) -> ResearchStats {
    let mut by_status: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_agent: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_generator: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for h in hypotheses {
        if let Some(s) = h.get("status").and_then(|v| v.as_str()) {
            *by_status.entry(s.to_string()).or_default() += 1;
        }
        if let Some(a) = h.get("agent").and_then(|v| v.as_str()) {
            *by_agent.entry(a.to_string()).or_default() += 1;
        }
        if let Some(g) = h.get("generator").and_then(|v| v.as_str()) {
            *by_generator.entry(g.to_string()).or_default() += 1;
        }
    }

    ResearchStats {
        total_hypotheses: hypotheses.len(),
        by_status,
        by_agent,
        by_generator,
        total_cycles,
    }
}

/// GET /api/research/heatmap
/// Feature x horizon IC matrix from hypothesis gate results.
pub async fn get_heatmap(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HeatmapResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Get all hypotheses (from SQLite or fallback)
    let all = if let Some(items) = get_all_db(&state, "hypothesis").await {
        items
    } else {
        let hyp_dir = research_dir(&state).join("hypotheses");
        read_json_dir(&hyp_dir)
    };

    let mut entries = Vec::new();
    let mut features_set = std::collections::BTreeSet::new();
    let mut horizons_set = std::collections::BTreeSet::new();

    for h in &all {
        let status = h
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let horizon = h
            .get("horizon_s")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
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
    if name.starts_with("spread_") { return "spread"; }
    if name.starts_with("depth_") { return "depth"; }
    if name.starts_with("imb_") { return "imbalance"; }
    if name.starts_with("flow_") { return "flow"; }
    if name.starts_with("vol_") || name.starts_with("volatility_") { return "volatility"; }
    if name.starts_with("ent_") { return "entropy"; }
    if name.starts_with("trend_") { return "trend"; }
    if name.starts_with("illiq_") { return "illiquidity"; }
    if name.starts_with("tox_") { return "toxicity"; }
    if name.starts_with("whale_") { return "whale"; }
    if name.starts_with("liquidation_") || name.starts_with("largest_position")
        || name.starts_with("nearest_cluster") || name.starts_with("positions_at_risk") { return "liquidation"; }
    if name.starts_with("top") || name.starts_with("herfindahl_") || name.starts_with("gini_")
        || name.starts_with("theil_") { return "concentration"; }
    if name.starts_with("ctx_") { return "context"; }
    if name.starts_with("raw_") { return "raw"; }
    if name.starts_with("regime_") || name.starts_with("gmm_") { return "regime"; }
    if name.starts_with("cross_") { return "cross_symbol"; }
    "other"
}

/// Read IT engine state file for a symbol.
fn read_it_engine_state(dir: &std::path::Path, symbol: &str) -> Option<serde_json::Value> {
    let path = dir.join(format!("state_{}.json", symbol));
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Build network graph from IT engine state and hypothesis data.
fn build_network(it_state: &serde_json::Value, hypotheses: &[serde_json::Value]) -> NetworkResponse {
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
    let mut feature_hyp_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    // Track co-occurrence pairs
    let mut cooccurrence: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();

    for h in hypotheses {
        let feats: Vec<String> = h
            .get("features")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
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

            let hypothesis_count = feature_hyp_count.get(feature.as_str()).copied().unwrap_or(0);
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
        .map(|((source, target), weight)| NetworkEdge { source, target, weight })
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

    // Get hypotheses from SQLite or fallback
    let hypotheses = if let Some(items) = get_all_db(&state, "hypothesis").await {
        items
    } else {
        let hyp_dir = research_dir(&state).join("hypotheses");
        read_json_dir(&hyp_dir)
    };

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

    fn insert_hypothesis(db: &Arc<Mutex<Connection>>, id: &str, agent: &str, gen: &str, status: &str) {
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
        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        // Count
        let count_sql = format!("SELECT COUNT(*) FROM research_output WHERE {}", where_clause);
        let total: i64 = conn
            .query_row(&count_sql, rusqlite::params_from_iter(params_refs.iter()), |row| row.get(0))
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
        let (items, total) = test_query_db(&db, "hypothesis", None, None, Some("replicated"), 50, 0);
        assert_eq!(total, 2);
        assert_eq!(items.len(), 2);

        // Filter by generator
        let (items, total) = test_query_db(&db, "hypothesis", None, Some("systematic"), None, 50, 0);
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

        let items = test_get_all(&db, "hypothesis");
        let stats = compute_stats_from_items(&items, 2);

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
    // JSON fallback tests (existing)
    // -----------------------------------------------------------------------

    fn write_hypothesis_json(dir: &std::path::Path, id: &str, agent: &str, gen: &str, status: &str) {
        let hyp_dir = dir.join("hypotheses");
        fs::create_dir_all(&hyp_dir).unwrap();
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
        let path = hyp_dir.join(format!("{}.json", id));
        fs::write(path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    fn write_cycle_json(dir: &std::path::Path, cycle_id: &str, agent: &str, n_tested: usize) {
        let cyc_dir = dir.join("cycles");
        fs::create_dir_all(&cyc_dir).unwrap();
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
        let path = cyc_dir.join(format!("{}.json", cycle_id));
        fs::write(path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    }

    #[test]
    fn test_read_json_dir_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("empty");
        fs::create_dir_all(&dir).unwrap();
        let items = read_json_dir(&dir);
        assert!(items.is_empty());
    }

    #[test]
    fn test_read_json_dir_nonexistent() {
        let items = read_json_dir(std::path::Path::new("/nonexistent/path"));
        assert!(items.is_empty());
    }

    #[test]
    fn test_read_json_dir_ignores_non_json() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("readme.txt"), "not json").unwrap();
        fs::write(tmp.path().join("valid.json"), r#"{"id": "test"}"#).unwrap();
        let items = read_json_dir(tmp.path());
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_read_json_dir_ignores_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("bad.json"), "not valid json{{{").unwrap();
        fs::write(tmp.path().join("good.json"), r#"{"ok": true}"#).unwrap();
        let items = read_json_dir(tmp.path());
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_filter_by_agent() {
        let items = vec![
            serde_json::json!({"agent": "micro", "status": "replicated"}),
            serde_json::json!({"agent": "macro", "status": "failed"}),
            serde_json::json!({"agent": "micro", "status": "failed"}),
        ];
        let q = HypothesisQuery { agent: Some("micro".to_string()), generator: None, status: None, limit: None, offset: None };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_by_status() {
        let items = vec![
            serde_json::json!({"agent": "micro", "status": "replicated"}),
            serde_json::json!({"agent": "macro", "status": "failed"}),
        ];
        let q = HypothesisQuery { agent: None, generator: None, status: Some("replicated".to_string()), limit: None, offset: None };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_paginate_basic() {
        let items: Vec<serde_json::Value> = (0..10).map(|i| serde_json::json!({"i": i})).collect();
        let result = paginate(items, 0, 3);
        assert_eq!(result.total, 10);
        assert_eq!(result.items.len(), 3);
    }

    #[test]
    fn test_paginate_offset() {
        let items: Vec<serde_json::Value> = (0..10).map(|i| serde_json::json!({"i": i})).collect();
        let result = paginate(items, 5, 3);
        assert_eq!(result.total, 10);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.items[0]["i"], 5);
    }

    #[test]
    fn test_compute_stats_from_items() {
        let items = vec![
            serde_json::json!({"agent": "micro", "generator": "systematic", "status": "replicated"}),
            serde_json::json!({"agent": "micro", "generator": "spectral", "status": "failed"}),
            serde_json::json!({"agent": "macro", "generator": "systematic", "status": "replicated"}),
        ];
        let stats = compute_stats_from_items(&items, 2);
        assert_eq!(stats.total_hypotheses, 3);
        assert_eq!(stats.total_cycles, 2);
        assert_eq!(stats.by_status["replicated"], 2);
        assert_eq!(stats.by_agent["micro"], 2);
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
        ).unwrap();
        let loaded = read_it_engine_state(tmp.path(), "BTC");
        assert!(loaded.is_some());
        assert!(read_it_engine_state(tmp.path(), "ETH").is_none());
    }

    // -----------------------------------------------------------------------
    // End-to-end: JSON fallback still works
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_fallback_hypotheses() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis_json(tmp.path(), "H1", "micro", "systematic", "replicated");
        write_hypothesis_json(tmp.path(), "H2", "micro", "systematic", "failed");

        let dir = tmp.path().join("hypotheses");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 2);

        let q = HypothesisQuery { agent: Some("micro".to_string()), generator: None, status: Some("replicated".to_string()), limit: None, offset: None };
        let filtered = filter_hypotheses(&all, &q);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_json_fallback_cycles() {
        let tmp = tempfile::tempdir().unwrap();
        write_cycle_json(tmp.path(), "CYC-1", "micro", 10);
        write_cycle_json(tmp.path(), "CYC-2", "macro", 5);

        let dir = tmp.path().join("cycles");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 2);

        let q = CycleQuery { agent: Some("micro".to_string()), limit: None, offset: None };
        let filtered = filter_cycles(&all, &q);
        assert_eq!(filtered.len(), 1);
    }
}
