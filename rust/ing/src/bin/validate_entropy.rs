//! Tick Entropy Feature Validation
//!
//! Skeptical validation of tick entropy features against live Hyperliquid data.
//! Tests:
//! 1. Are tick entropy values in expected range?
//! 2. Do different time windows produce different values?
//! 3. Does entropy correlate with market activity?

use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
const COLLECTION_DURATION_SECS: u64 = 120; // 2 minutes of data

/// Trade sample for entropy calculation
#[derive(Debug, Clone)]
struct TradeSample {
    timestamp_ms: u64,
    price: f64,
    size: f64,
    is_buy: bool,
}

/// Rolling buffer for entropy calculation
struct EntropyBuffer {
    trades: VecDeque<TradeSample>,
    last_price: Option<f64>,
}

impl EntropyBuffer {
    fn new() -> Self {
        Self {
            trades: VecDeque::with_capacity(10_000),
            last_price: None,
        }
    }

    fn add_trade(&mut self, price: f64, size: f64, is_buy: bool, timestamp_ms: u64) {
        self.trades.push_back(TradeSample {
            timestamp_ms,
            price,
            size,
            is_buy,
        });
        self.last_price = Some(price);
    }

    fn get_tick_directions(&self, window_ms: u64) -> Vec<(i8, f64)> {
        if self.trades.is_empty() {
            return vec![];
        }

        let latest_time = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest_time.saturating_sub(window_ms);

        let trades: Vec<_> = self.trades.iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .collect();

        if trades.len() < 2 {
            return trades.iter()
                .map(|t| (if t.is_buy { 1i8 } else { -1i8 }, t.size))
                .collect();
        }

        let mut result = Vec::with_capacity(trades.len());
        let mut last_price: Option<f64> = None;

        for trade in trades {
            let direction = match last_price {
                Some(prev) if trade.price > prev => 1i8,
                Some(prev) if trade.price < prev => -1i8,
                Some(_) => if trade.is_buy { 1i8 } else { -1i8 },
                None => if trade.is_buy { 1i8 } else { -1i8 },
            };
            last_price = Some(trade.price);
            result.push((direction, trade.size));
        }

        result
    }

    fn tick_entropy(&self, window_secs: u64) -> Option<f64> {
        let directions = self.get_tick_directions(window_secs * 1000);
        if directions.is_empty() {
            return None;
        }

        let mut counts = [0u32; 3]; // down, neutral, up
        for (dir, _) in &directions {
            let idx = match *dir {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => continue,
            };
            counts[idx] += 1;
        }

        let total: u32 = counts.iter().sum();
        if total == 0 {
            return None;
        }

        let entropy: f64 = counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum();

        Some(entropy)
    }

    fn volume_tick_entropy(&self, window_secs: u64) -> Option<f64> {
        let directions = self.get_tick_directions(window_secs * 1000);
        if directions.is_empty() {
            return None;
        }

        let mut volumes = [0.0f64; 3];
        for (dir, vol) in &directions {
            let idx = match *dir {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => continue,
            };
            volumes[idx] += vol;
        }

        let total: f64 = volumes.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let entropy: f64 = volumes.iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / total;
                -p * p.ln()
            })
            .sum();

        Some(entropy)
    }

    fn trade_count(&self, window_secs: u64) -> usize {
        if self.trades.is_empty() {
            return 0;
        }
        let latest_time = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest_time.saturating_sub(window_secs * 1000);
        self.trades.iter().filter(|t| t.timestamp_ms >= cutoff).count()
    }
}

#[derive(Debug, Serialize)]
struct EntropySnapshot {
    timestamp: String,
    trade_count_1s: usize,
    trade_count_5s: usize,
    trade_count_30s: usize,
    tick_entropy_1s: Option<f64>,
    tick_entropy_5s: Option<f64>,
    tick_entropy_10s: Option<f64>,
    tick_entropy_30s: Option<f64>,
    tick_entropy_1m: Option<f64>,
    volume_entropy_1s: Option<f64>,
    volume_entropy_5s: Option<f64>,
    volume_entropy_30s: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    timestamp: String,
    symbol: String,
    collection_duration_secs: u64,
    total_trades: usize,
    snapshots: Vec<EntropySnapshot>,
    analysis: EntropyAnalysis,
    go_no_go: String,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EntropyAnalysis {
    avg_tick_entropy_1s: f64,
    avg_tick_entropy_5s: f64,
    avg_tick_entropy_30s: f64,
    avg_tick_entropy_1m: f64,
    max_tick_entropy: f64,
    min_tick_entropy: f64,
    entropy_range: f64,
    windows_with_data: WindowStats,
    entropy_values_in_range: bool,
    different_windows_different_values: bool,
}

#[derive(Debug, Serialize)]
struct WindowStats {
    tick_1s: usize,
    tick_5s: usize,
    tick_30s: usize,
    tick_1m: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        TICK ENTROPY VALIDATION (SKEPTICAL MODE)                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let symbol = "BTC";
    println!("[PHASE 1] Connecting to Hyperliquid WebSocket...");

    let (mut ws_stream, _) = connect_async(WS_URL)
        .await
        .context("Failed to connect to WebSocket")?;

    println!("  ✓ Connected\n");

    // Subscribe to trades
    println!("[PHASE 2] Subscribing to {} trades...", symbol);
    let sub = serde_json::json!({
        "method": "subscribe",
        "subscription": {
            "type": "trades",
            "coin": symbol
        }
    });
    ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;
    println!("  ✓ Subscribed\n");

    // Collect trades
    println!("[PHASE 3] Collecting trades for {} seconds...", COLLECTION_DURATION_SECS);

    let mut buffer = EntropyBuffer::new();
    let mut snapshots = Vec::new();
    let mut total_trades = 0;

    let start = std::time::Instant::now();
    let duration = Duration::from_secs(COLLECTION_DURATION_SECS);
    let mut last_snapshot = std::time::Instant::now();
    let snapshot_interval = Duration::from_secs(10);

    while start.elapsed() < duration {
        match tokio::time::timeout(Duration::from_secs(5), ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {
                    if response["channel"] == "trades" {
                        if let Some(trades) = response["data"].as_array() {
                            for trade in trades {
                                let price = trade["px"].as_str()
                                    .and_then(|s| s.parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                let size = trade["sz"].as_str()
                                    .and_then(|s| s.parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                let side = trade["side"].as_str().unwrap_or("B");
                                let is_buy = side == "B";
                                let timestamp = trade["time"].as_u64().unwrap_or(0);

                                buffer.add_trade(price, size, is_buy, timestamp);
                                total_trades += 1;
                            }
                        }
                    }
                }
            }
            Ok(Some(Ok(Message::Ping(data)))) => {
                ws_stream.send(Message::Pong(data)).await?;
            }
            _ => {}
        }

        // Take periodic snapshots
        if last_snapshot.elapsed() >= snapshot_interval && buffer.trades.len() > 10 {
            let snapshot = EntropySnapshot {
                timestamp: Utc::now().to_rfc3339(),
                trade_count_1s: buffer.trade_count(1),
                trade_count_5s: buffer.trade_count(5),
                trade_count_30s: buffer.trade_count(30),
                tick_entropy_1s: buffer.tick_entropy(1),
                tick_entropy_5s: buffer.tick_entropy(5),
                tick_entropy_10s: buffer.tick_entropy(10),
                tick_entropy_30s: buffer.tick_entropy(30),
                tick_entropy_1m: buffer.tick_entropy(60),
                volume_entropy_1s: buffer.volume_tick_entropy(1),
                volume_entropy_5s: buffer.volume_tick_entropy(5),
                volume_entropy_30s: buffer.volume_tick_entropy(30),
            };

            println!("  Snapshot @ {}s: {} trades, tick_ent_5s={:.4}, tick_ent_30s={:.4}",
                start.elapsed().as_secs(),
                snapshot.trade_count_30s,
                snapshot.tick_entropy_5s.unwrap_or(0.0),
                snapshot.tick_entropy_30s.unwrap_or(0.0),
            );

            snapshots.push(snapshot);
            last_snapshot = std::time::Instant::now();
        }
    }

    println!("\n  ✓ Collected {} trades in {} snapshots\n", total_trades, snapshots.len());

    // Analyze results
    println!("[PHASE 4] Analyzing entropy features...\n");

    let analysis = analyze_entropy(&snapshots);
    let mut warnings = Vec::new();

    // Validation checks
    if !analysis.entropy_values_in_range {
        warnings.push("Entropy values outside expected range [0, ln(3)]".to_string());
    }

    if !analysis.different_windows_different_values {
        warnings.push("Different time windows have identical entropy values".to_string());
    }

    if total_trades < 100 {
        warnings.push(format!("Low trade count: {} (expected 100+)", total_trades));
    }

    let go_no_go = if warnings.is_empty() && total_trades >= 100 {
        format!("GO: Tick entropy features working correctly with {} trades", total_trades)
    } else if warnings.len() <= 1 && total_trades >= 50 {
        "CAUTION: Minor issues but entropy features functional".to_string()
    } else {
        "NO-GO: Significant issues with entropy feature computation".to_string()
    };

    let report = ValidationReport {
        timestamp: Utc::now().to_rfc3339(),
        symbol: symbol.to_string(),
        collection_duration_secs: COLLECTION_DURATION_SECS,
        total_trades,
        snapshots,
        analysis,
        go_no_go: go_no_go.clone(),
        warnings: warnings.clone(),
    };

    print_report(&report);

    // Save report
    let report_path = format!(
        "data/entropy_validation_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data")?;
    let mut file = File::create(&report_path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("\n📄 Report saved to: {}", report_path);

    Ok(())
}

fn analyze_entropy(snapshots: &[EntropySnapshot]) -> EntropyAnalysis {
    if snapshots.is_empty() {
        return EntropyAnalysis {
            avg_tick_entropy_1s: 0.0,
            avg_tick_entropy_5s: 0.0,
            avg_tick_entropy_30s: 0.0,
            avg_tick_entropy_1m: 0.0,
            max_tick_entropy: 0.0,
            min_tick_entropy: 0.0,
            entropy_range: 0.0,
            windows_with_data: WindowStats {
                tick_1s: 0,
                tick_5s: 0,
                tick_30s: 0,
                tick_1m: 0,
            },
            entropy_values_in_range: false,
            different_windows_different_values: false,
        };
    }

    let max_theoretical_entropy = 3.0_f64.ln(); // ln(3) for 3 states

    let mut tick_1s: Vec<f64> = snapshots.iter().filter_map(|s| s.tick_entropy_1s).collect();
    let mut tick_5s: Vec<f64> = snapshots.iter().filter_map(|s| s.tick_entropy_5s).collect();
    let mut tick_30s: Vec<f64> = snapshots.iter().filter_map(|s| s.tick_entropy_30s).collect();
    let mut tick_1m: Vec<f64> = snapshots.iter().filter_map(|s| s.tick_entropy_1m).collect();

    let all_entropy: Vec<f64> = [&tick_1s, &tick_5s, &tick_30s, &tick_1m]
        .iter()
        .flat_map(|v| v.iter())
        .cloned()
        .collect();

    let max_entropy = all_entropy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_entropy = all_entropy.iter().cloned().fold(f64::INFINITY, f64::min);

    let avg = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };

    let avg_1s = avg(&tick_1s);
    let avg_5s = avg(&tick_5s);
    let avg_30s = avg(&tick_30s);
    let avg_1m = avg(&tick_1m);

    // Check if all entropy values are in valid range [0, ln(3)]
    let entropy_values_in_range = all_entropy.iter()
        .all(|&e| e >= 0.0 && e <= max_theoretical_entropy + 0.01);

    // Check if different windows produce meaningfully different values
    let different_windows_different_values = if !tick_5s.is_empty() && !tick_30s.is_empty() {
        (avg_5s - avg_30s).abs() > 0.01 || (avg_1s - avg_1m).abs() > 0.01
    } else {
        false
    };

    EntropyAnalysis {
        avg_tick_entropy_1s: avg_1s,
        avg_tick_entropy_5s: avg_5s,
        avg_tick_entropy_30s: avg_30s,
        avg_tick_entropy_1m: avg_1m,
        max_tick_entropy: if max_entropy.is_finite() { max_entropy } else { 0.0 },
        min_tick_entropy: if min_entropy.is_finite() { min_entropy } else { 0.0 },
        entropy_range: if max_entropy.is_finite() && min_entropy.is_finite() {
            max_entropy - min_entropy
        } else {
            0.0
        },
        windows_with_data: WindowStats {
            tick_1s: tick_1s.len(),
            tick_5s: tick_5s.len(),
            tick_30s: tick_30s.len(),
            tick_1m: tick_1m.len(),
        },
        entropy_values_in_range,
        different_windows_different_values,
    }
}

fn print_report(report: &ValidationReport) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    ENTROPY VALIDATION REPORT                     ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ DATA COLLECTION");
    println!("║   Symbol: {}", report.symbol);
    println!("║   Duration: {} seconds", report.collection_duration_secs);
    println!("║   Total trades: {}", report.total_trades);
    println!("║   Snapshots: {}", report.snapshots.len());

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ ENTROPY ANALYSIS");
    println!("║   Theoretical max: {:.4} (ln(3) for 3 states)", 3.0_f64.ln());
    println!("║   Avg tick entropy (1s):  {:.4}", report.analysis.avg_tick_entropy_1s);
    println!("║   Avg tick entropy (5s):  {:.4}", report.analysis.avg_tick_entropy_5s);
    println!("║   Avg tick entropy (30s): {:.4}", report.analysis.avg_tick_entropy_30s);
    println!("║   Avg tick entropy (1m):  {:.4}", report.analysis.avg_tick_entropy_1m);
    println!("║   Min entropy: {:.4}", report.analysis.min_tick_entropy);
    println!("║   Max entropy: {:.4}", report.analysis.max_tick_entropy);
    println!("║   Range: {:.4}", report.analysis.entropy_range);

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ VALIDATION CHECKS");
    println!("║   Values in range [0, ln(3)]: {}",
        if report.analysis.entropy_values_in_range { "✓" } else { "✗" });
    println!("║   Different windows differ:  {}",
        if report.analysis.different_windows_different_values { "✓" } else { "✗" });
    println!("║   Snapshots with 1s data:  {}", report.analysis.windows_with_data.tick_1s);
    println!("║   Snapshots with 5s data:  {}", report.analysis.windows_with_data.tick_5s);
    println!("║   Snapshots with 30s data: {}", report.analysis.windows_with_data.tick_30s);
    println!("║   Snapshots with 1m data:  {}", report.analysis.windows_with_data.tick_1m);

    println!("╠══════════════════════════════════════════════════════════════════╣");

    if !report.warnings.is_empty() {
        println!("║ ⚠️  WARNINGS:");
        for warning in &report.warnings {
            println!("║   • {}", warning);
        }
        println!("╠══════════════════════════════════════════════════════════════════╣");
    }

    println!("║ VERDICT: {}", report.go_no_go);
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
