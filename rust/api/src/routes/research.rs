//! Research API routes — hypothesis history, cycle reports, signal registry, stats.
//!
//! Reads structured JSON files emitted by the Python agent (P2-1):
//!   data/research/hypotheses/{id}.json
//!   data/research/cycles/{cycle_id}.json

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

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
// Helpers
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
// Cache helpers
// ---------------------------------------------------------------------------

/// Ensure the research cache is fresh, reloading from disk if stale.
async fn ensure_cache(state: &AppState) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    // Fast path: read lock, check freshness
    {
        let cache = state.research_cache.read().await;
        if !cache.is_stale() {
            return (cache.hypotheses.clone(), cache.cycles.clone());
        }
    }

    // Slow path: write lock, reload from disk
    let mut cache = state.research_cache.write().await;
    // Double-check after acquiring write lock (another request may have refreshed)
    if !cache.is_stale() {
        return (cache.hypotheses.clone(), cache.cycles.clone());
    }

    let hyp_dir = research_dir(state).join("hypotheses");
    let cyc_dir = research_dir(state).join("cycles");
    cache.hypotheses = read_json_dir(&hyp_dir);
    cache.cycles = read_json_dir(&cyc_dir);
    cache.loaded_at = std::time::Instant::now();
    debug!(
        hypotheses = cache.hypotheses.len(),
        cycles = cache.cycles.len(),
        "Research cache refreshed"
    );

    (cache.hypotheses.clone(), cache.cycles.clone())
}

/// Invalidate the cache (called after research events).
pub async fn invalidate_cache(state: &AppState) {
    let mut cache = state.research_cache.write().await;
    cache.loaded_at = std::time::Instant::now() - cache.ttl - std::time::Duration::from_secs(1);
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
    let (all, _) = ensure_cache(&state).await;
    let filtered = filter_hypotheses(&all, &q);
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);
    Ok(Json(paginate(filtered, offset, limit)))
}

/// GET /api/research/hypotheses/:id
/// Full detail for a single hypothesis.
pub async fn get_hypothesis(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Single hypothesis: check cache first, fall back to direct read
    let (all, _) = ensure_cache(&state).await;
    if let Some(h) = all.iter().find(|h| h.get("id").and_then(|v| v.as_str()) == Some(&id)) {
        return Ok(Json(h.clone()));
    }
    // Not in cache — try direct file read (may be very new)
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
    let (_, all) = ensure_cache(&state).await;
    let filtered = filter_cycles(&all, &q);
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).min(200);
    Ok(Json(paginate(filtered, offset, limit)))
}

/// GET /api/research/signals
/// Registered signals (hypotheses with status=replicated).
pub async fn list_signals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, Json<ErrorResponse>)> {
    let (all, _) = ensure_cache(&state).await;
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
    let (hypotheses, cycles) = ensure_cache(&state).await;

    let mut by_status: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_agent: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut by_generator: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for h in &hypotheses {
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

    Ok(Json(ResearchStats {
        total_hypotheses: hypotheses.len(),
        by_status,
        by_agent,
        by_generator,
        total_cycles: cycles.len(),
    }))
}

/// GET /api/research/heatmap
/// Feature x horizon IC matrix from hypothesis gate results.
pub async fn get_heatmap(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HeatmapResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (all, _) = ensure_cache(&state).await;

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_hypothesis(dir: &std::path::Path, id: &str, agent: &str, gen: &str, status: &str) {
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
            "gates": [
                {
                    "name": "IC",
                    "passed": status != "failed",
                    "message": "PASS IC=0.08 vs min=0.03 p=0.001",
                    "metric": 0.08,
                    "threshold": 0.03,
                    "p_value": 0.001
                }
            ],
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

    fn write_cycle(dir: &std::path::Path, cycle_id: &str, agent: &str, n_tested: usize) {
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

    // -----------------------------------------------------------------------
    // read_json_dir
    // -----------------------------------------------------------------------

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
        fs::write(
            tmp.path().join("valid.json"),
            r#"{"id": "test"}"#,
        )
        .unwrap();
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
    fn test_read_json_dir_sorted_by_completed() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("a.json"),
            r#"{"timestamps": {"completed": "2026-05-25T10:00:00+00:00"}}"#,
        ).unwrap();
        fs::write(
            tmp.path().join("b.json"),
            r#"{"timestamps": {"completed": "2026-05-25T12:00:00+00:00"}}"#,
        ).unwrap();
        let items = read_json_dir(tmp.path());
        assert_eq!(items.len(), 2);
        // b (12:00) should come first (newest)
        let ts_first = items[0]["timestamps"]["completed"].as_str().unwrap();
        assert!(ts_first.contains("12:00"));
    }

    // -----------------------------------------------------------------------
    // filter_hypotheses
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_by_agent() {
        let items = vec![
            serde_json::json!({"agent": "micro", "status": "replicated"}),
            serde_json::json!({"agent": "macro", "status": "failed"}),
            serde_json::json!({"agent": "micro", "status": "failed"}),
        ];
        let q = HypothesisQuery {
            agent: Some("micro".to_string()),
            generator: None,
            status: None,
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_by_status() {
        let items = vec![
            serde_json::json!({"agent": "micro", "status": "replicated"}),
            serde_json::json!({"agent": "macro", "status": "failed"}),
        ];
        let q = HypothesisQuery {
            agent: None,
            generator: None,
            status: Some("replicated".to_string()),
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["status"], "replicated");
    }

    #[test]
    fn test_filter_by_generator() {
        let items = vec![
            serde_json::json!({"generator": "systematic", "status": "replicated"}),
            serde_json::json!({"generator": "spectral", "status": "replicated"}),
        ];
        let q = HypothesisQuery {
            agent: None,
            generator: Some("spectral".to_string()),
            status: None,
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["generator"], "spectral");
    }

    #[test]
    fn test_filter_combined() {
        let items = vec![
            serde_json::json!({"agent": "micro", "generator": "systematic", "status": "replicated"}),
            serde_json::json!({"agent": "micro", "generator": "systematic", "status": "failed"}),
            serde_json::json!({"agent": "macro", "generator": "systematic", "status": "replicated"}),
        ];
        let q = HypothesisQuery {
            agent: Some("micro".to_string()),
            generator: Some("systematic".to_string()),
            status: Some("replicated".to_string()),
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_filter_no_match() {
        let items = vec![
            serde_json::json!({"agent": "micro", "status": "failed"}),
        ];
        let q = HypothesisQuery {
            agent: Some("nonexistent".to_string()),
            generator: None,
            status: None,
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_no_filters() {
        let items = vec![
            serde_json::json!({"agent": "micro"}),
            serde_json::json!({"agent": "macro"}),
        ];
        let q = HypothesisQuery {
            agent: None,
            generator: None,
            status: None,
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&items, &q);
        assert_eq!(filtered.len(), 2);
    }

    // -----------------------------------------------------------------------
    // filter_cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_cycles_by_agent() {
        let items = vec![
            serde_json::json!({"agent": "micro", "cycle_id": "CYC-1"}),
            serde_json::json!({"agent": "macro", "cycle_id": "CYC-2"}),
        ];
        let q = CycleQuery {
            agent: Some("micro".to_string()),
            limit: None,
            offset: None,
        };
        let filtered = filter_cycles(&items, &q);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["cycle_id"], "CYC-1");
    }

    // -----------------------------------------------------------------------
    // paginate
    // -----------------------------------------------------------------------

    #[test]
    fn test_paginate_basic() {
        let items: Vec<serde_json::Value> = (0..10)
            .map(|i| serde_json::json!({"i": i}))
            .collect();
        let result = paginate(items, 0, 3);
        assert_eq!(result.total, 10);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.offset, 0);
        assert_eq!(result.limit, 3);
    }

    #[test]
    fn test_paginate_offset() {
        let items: Vec<serde_json::Value> = (0..10)
            .map(|i| serde_json::json!({"i": i}))
            .collect();
        let result = paginate(items, 5, 3);
        assert_eq!(result.total, 10);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.items[0]["i"], 5);
    }

    #[test]
    fn test_paginate_past_end() {
        let items: Vec<serde_json::Value> = (0..3)
            .map(|i| serde_json::json!({"i": i}))
            .collect();
        let result = paginate(items, 10, 5);
        assert_eq!(result.total, 3);
        assert!(result.items.is_empty());
    }

    #[test]
    fn test_paginate_empty() {
        let result = paginate(Vec::new(), 0, 10);
        assert_eq!(result.total, 0);
        assert!(result.items.is_empty());
    }

    // -----------------------------------------------------------------------
    // Integration: full file read + filter + paginate
    // -----------------------------------------------------------------------

    #[test]
    fn test_end_to_end_hypotheses() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis(tmp.path(), "HYP-SYS-001", "micro", "systematic", "replicated");
        write_hypothesis(tmp.path(), "HYP-SYS-002", "micro", "systematic", "failed");
        write_hypothesis(tmp.path(), "HYP-SPE-003", "micro", "spectral", "replicated");
        write_hypothesis(tmp.path(), "HYP-MAC-004", "macro", "funding_meanrev", "failed");

        let dir = tmp.path().join("hypotheses");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 4);

        // Filter: micro + replicated
        let q = HypothesisQuery {
            agent: Some("micro".to_string()),
            generator: None,
            status: Some("replicated".to_string()),
            limit: None,
            offset: None,
        };
        let filtered = filter_hypotheses(&all, &q);
        assert_eq!(filtered.len(), 2);

        // Paginate
        let page = paginate(filtered, 0, 1);
        assert_eq!(page.total, 2);
        assert_eq!(page.items.len(), 1);
    }

    #[test]
    fn test_end_to_end_cycles() {
        let tmp = tempfile::tempdir().unwrap();
        write_cycle(tmp.path(), "CYC-001", "micro", 10);
        write_cycle(tmp.path(), "CYC-002", "macro", 5);
        write_cycle(tmp.path(), "CYC-003", "micro", 8);

        let dir = tmp.path().join("cycles");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 3);

        let q = CycleQuery {
            agent: Some("micro".to_string()),
            limit: None,
            offset: None,
        };
        let filtered = filter_cycles(&all, &q);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_signals_filters_replicated_only() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis(tmp.path(), "H1", "micro", "systematic", "replicated");
        write_hypothesis(tmp.path(), "H2", "micro", "systematic", "failed");
        write_hypothesis(tmp.path(), "H3", "micro", "spectral", "replicated");

        let dir = tmp.path().join("hypotheses");
        let all = read_json_dir(&dir);
        let signals: Vec<_> = all
            .into_iter()
            .filter(|h| h.get("status").and_then(|v| v.as_str()) == Some("replicated"))
            .collect();
        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn test_stats_aggregation() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis(tmp.path(), "H1", "micro", "systematic", "replicated");
        write_hypothesis(tmp.path(), "H2", "micro", "spectral", "failed");
        write_hypothesis(tmp.path(), "H3", "macro", "systematic", "replicated");
        write_cycle(tmp.path(), "C1", "micro", 5);
        write_cycle(tmp.path(), "C2", "macro", 3);

        let hyp_dir = tmp.path().join("hypotheses");
        let cyc_dir = tmp.path().join("cycles");
        let hypotheses = read_json_dir(&hyp_dir);
        let cycles = read_json_dir(&cyc_dir);

        let mut by_status: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut by_agent: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut by_generator: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for h in &hypotheses {
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

        assert_eq!(hypotheses.len(), 3);
        assert_eq!(cycles.len(), 2);
        assert_eq!(by_status["replicated"], 2);
        assert_eq!(by_status["failed"], 1);
        assert_eq!(by_agent["micro"], 2);
        assert_eq!(by_agent["macro"], 1);
        assert_eq!(by_generator["systematic"], 2);
        assert_eq!(by_generator["spectral"], 1);
    }

    #[test]
    fn test_heatmap_extraction() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis(tmp.path(), "H1", "micro", "systematic", "replicated");
        write_hypothesis(tmp.path(), "H2", "micro", "systematic", "failed");

        let dir = tmp.path().join("hypotheses");
        let all = read_json_dir(&dir);

        let mut entries = Vec::new();
        let mut features_set = std::collections::BTreeSet::new();

        for h in &all {
            let status = h.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let horizon = h.get("horizon_s").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let feats = h.get("features").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let ic = h.get("gates")
                .and_then(|g| g.as_array())
                .and_then(|gates| {
                    gates.iter().find_map(|gate| {
                        if gate.get("name").and_then(|v| v.as_str()) == Some("IC") {
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
                    entries.push(HeatmapEntry {
                        feature: feat.to_string(),
                        horizon_s: horizon,
                        ic,
                        status: status.to_string(),
                    });
                }
            }
        }

        assert_eq!(entries.len(), 2);
        assert!(features_set.contains("ent_book_shape"));
        assert!((entries[0].ic - 0.08).abs() < 1e-6);
        assert_eq!(entries[0].horizon_s, 5.0);
    }

    #[test]
    fn test_hypothesis_schema_fields() {
        let tmp = tempfile::tempdir().unwrap();
        write_hypothesis(tmp.path(), "HYP-SYS-test", "micro", "systematic", "replicated");

        let dir = tmp.path().join("hypotheses");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 1);

        let h = &all[0];
        assert_eq!(h["schema_version"], 1);
        assert_eq!(h["id"], "HYP-SYS-test");
        assert_eq!(h["agent"], "micro");
        assert_eq!(h["generator"], "systematic");
        assert!(h["claim"].as_str().unwrap().len() > 0);
        assert!(h["math"].as_str().unwrap().len() > 0);
        assert_eq!(h["status"], "replicated");
        assert!(h["gates"].as_array().unwrap().len() > 0);
        assert!(h["features"].as_array().unwrap().len() > 0);
        assert!(h["timestamps"]["created"].as_str().is_some());
        assert!(h["timestamps"]["completed"].as_str().is_some());
    }

    #[test]
    fn test_cycle_schema_fields() {
        let tmp = tempfile::tempdir().unwrap();
        write_cycle(tmp.path(), "CYC-test01", "micro", 7);

        let dir = tmp.path().join("cycles");
        let all = read_json_dir(&dir);
        assert_eq!(all.len(), 1);

        let c = &all[0];
        assert_eq!(c["schema_version"], 1);
        assert_eq!(c["cycle_id"], "CYC-test01");
        assert_eq!(c["agent"], "micro");
        assert_eq!(c["n_tested"], 7);
        assert_eq!(c["n_registered"], 1);
        assert_eq!(c["fdr_q"], 0.05);
        assert!(c["started"].as_str().is_some());
        assert!(c["completed"].as_str().is_some());
    }

    #[test]
    fn test_empty_research_dir() {
        let tmp = tempfile::tempdir().unwrap();
        // No hypotheses or cycles dirs created
        let hyp_dir = tmp.path().join("hypotheses");
        let cyc_dir = tmp.path().join("cycles");
        assert!(read_json_dir(&hyp_dir).is_empty());
        assert!(read_json_dir(&cyc_dir).is_empty());
    }

    #[test]
    fn test_paginate_limit_clamped() {
        // Verify that limit > total still works
        let items: Vec<serde_json::Value> = (0..3)
            .map(|i| serde_json::json!({"i": i}))
            .collect();
        let result = paginate(items, 0, 100);
        assert_eq!(result.total, 3);
        assert_eq!(result.items.len(), 3);
        assert_eq!(result.limit, 100);
    }

    // -----------------------------------------------------------------------
    // Cache tests
    // -----------------------------------------------------------------------

    use crate::state::ResearchCache;
    use std::time::{Duration, Instant};

    #[test]
    fn test_cache_starts_stale() {
        let cache = ResearchCache::new(Duration::from_secs(30));
        assert!(cache.is_stale());
        assert!(cache.hypotheses.is_empty());
        assert!(cache.cycles.is_empty());
    }

    #[test]
    fn test_cache_fresh_after_load() {
        let mut cache = ResearchCache::new(Duration::from_secs(30));
        cache.hypotheses = vec![serde_json::json!({"id": "H1"})];
        cache.loaded_at = Instant::now();
        assert!(!cache.is_stale());
        assert_eq!(cache.hypotheses.len(), 1);
    }

    #[test]
    fn test_cache_stale_after_ttl() {
        let mut cache = ResearchCache::new(Duration::from_millis(1));
        cache.loaded_at = Instant::now() - Duration::from_millis(10);
        assert!(cache.is_stale());
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = ResearchCache::new(Duration::from_secs(30));
        cache.hypotheses = vec![serde_json::json!({"id": "H1"})];
        cache.loaded_at = Instant::now();
        assert!(!cache.is_stale());

        // Invalidate by pushing loaded_at back past TTL
        cache.loaded_at = Instant::now() - cache.ttl - Duration::from_secs(1);
        assert!(cache.is_stale());
    }
}
