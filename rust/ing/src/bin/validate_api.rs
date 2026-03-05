//! Hyperliquid API Validation Tool
//!
//! Skeptical validation of API data availability and quality claims.
//! Tests critical assumptions before building on top of the API.

use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
const TEST_DURATION_SECS: u64 = 60; // 1 minute (quick validation)
const SYMBOLS: &[&str] = &["BTC", "ETH", "SOL"];

#[derive(Debug, Default)]
struct ValidationStats {
    // Connection stats
    connection_time_ms: u64,

    // Trade stats per symbol
    trade_count: HashMap<String, usize>,
    trades_with_wallet: HashMap<String, usize>,
    unique_makers: HashMap<String, HashSet<String>>,
    unique_takers: HashMap<String, HashSet<String>>,

    // Book stats per symbol
    book_update_count: HashMap<String, usize>,
    max_book_levels: HashMap<String, usize>,

    // Timing stats
    message_timestamps: Vec<u64>,
    max_gap_ms: u64,
    gaps_over_5s: usize,

    // Asset context stats
    asset_ctx_updates: HashMap<String, usize>,
    last_funding_rates: HashMap<String, f64>,
    last_open_interest: HashMap<String, f64>,

    // Error stats
    parse_errors: usize,
    connection_drops: usize,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    timestamp: String,
    test_duration_secs: u64,
    symbols_tested: Vec<String>,

    // Critical findings
    wallet_addresses_available: bool,
    wallet_coverage_pct: f64,
    unique_maker_count: usize,
    unique_taker_count: usize,

    // Data quality
    max_gap_ms: u64,
    gaps_over_5s: usize,
    data_quality_score: f64,

    // Volume stats
    total_trades: usize,
    trades_per_symbol: HashMap<String, usize>,

    // Recommendations
    go_no_go: String,
    warnings: Vec<String>,
    blockers: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SubscriptionRequest {
    method: String,
    subscription: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct WsResponse {
    channel: String,
    data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WsTrade {
    coin: String,
    side: String,
    px: String,
    sz: String,
    hash: String,
    time: u64,
    tid: u64,
    #[serde(default)]
    users: Option<(String, String)>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WsBook {
    coin: String,
    levels: (Vec<WsLevel>, Vec<WsLevel>),
    time: u64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WsLevel {
    px: String,
    sz: String,
    n: u32,
}

#[derive(Debug, Deserialize)]
struct WsAssetCtx {
    coin: String,
    ctx: AssetCtxData,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssetCtxData {
    funding: String,
    open_interest: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           HYPERLIQUID API VALIDATION (SKEPTICAL MODE)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut stats = ValidationStats::default();

    // Initialize per-symbol stats
    for symbol in SYMBOLS {
        stats.trade_count.insert(symbol.to_string(), 0);
        stats.trades_with_wallet.insert(symbol.to_string(), 0);
        stats.unique_makers.insert(symbol.to_string(), HashSet::new());
        stats.unique_takers.insert(symbol.to_string(), HashSet::new());
        stats.book_update_count.insert(symbol.to_string(), 0);
        stats.max_book_levels.insert(symbol.to_string(), 0);
        stats.asset_ctx_updates.insert(symbol.to_string(), 0);
    }

    // Test 1: Connection time
    println!("[TEST 1] Measuring connection time...");
    let conn_start = Instant::now();

    let (mut ws_stream, _) = connect_async(WS_URL)
        .await
        .context("Failed to connect to Hyperliquid WebSocket")?;

    stats.connection_time_ms = conn_start.elapsed().as_millis() as u64;
    println!("  ✓ Connected in {}ms", stats.connection_time_ms);

    // Test 2: Subscribe to all channels
    println!("\n[TEST 2] Subscribing to channels...");
    for symbol in SYMBOLS {
        // Subscribe to l2Book
        let sub = SubscriptionRequest {
            method: "subscribe".to_string(),
            subscription: serde_json::json!({
                "type": "l2Book",
                "coin": symbol
            }),
        };
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;

        // Subscribe to trades
        let sub = SubscriptionRequest {
            method: "subscribe".to_string(),
            subscription: serde_json::json!({
                "type": "trades",
                "coin": symbol
            }),
        };
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;

        // Subscribe to activeAssetCtx
        let sub = SubscriptionRequest {
            method: "subscribe".to_string(),
            subscription: serde_json::json!({
                "type": "activeAssetCtx",
                "coin": symbol
            }),
        };
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;

        println!("  ✓ Subscribed to {} (l2Book, trades, activeAssetCtx)", symbol);
    }

    // Test 3: Collect data for TEST_DURATION_SECS
    println!("\n[TEST 3] Collecting data for {} seconds...", TEST_DURATION_SECS);
    let test_start = Instant::now();
    let test_duration = Duration::from_secs(TEST_DURATION_SECS);

    let mut last_msg_time: Option<u64> = None;
    let mut progress_counter = 0u64;

    while test_start.elapsed() < test_duration {
        match timeout(Duration::from_secs(10), ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(response) = serde_json::from_str::<WsResponse>(&text) {
                    let now_ms = chrono::Utc::now().timestamp_millis() as u64;

                    // Track message gaps
                    if let Some(last) = last_msg_time {
                        let gap = now_ms.saturating_sub(last);
                        if gap > stats.max_gap_ms {
                            stats.max_gap_ms = gap;
                        }
                        if gap > 5000 {
                            stats.gaps_over_5s += 1;
                        }
                    }
                    last_msg_time = Some(now_ms);
                    stats.message_timestamps.push(now_ms);

                    match response.channel.as_str() {
                        "trades" => {
                            if let Ok(trades) = serde_json::from_value::<Vec<WsTrade>>(response.data) {
                                for trade in trades {
                                    let symbol = trade.coin.clone();
                                    *stats.trade_count.entry(symbol.clone()).or_insert(0) += 1;

                                    if let Some((maker, taker)) = &trade.users {
                                        *stats.trades_with_wallet.entry(symbol.clone()).or_insert(0) += 1;
                                        stats.unique_makers.entry(symbol.clone())
                                            .or_insert_with(HashSet::new)
                                            .insert(maker.clone());
                                        stats.unique_takers.entry(symbol.clone())
                                            .or_insert_with(HashSet::new)
                                            .insert(taker.clone());
                                    }
                                }
                            }
                        }
                        "l2Book" => {
                            if let Ok(book) = serde_json::from_value::<WsBook>(response.data) {
                                let symbol = book.coin.clone();
                                *stats.book_update_count.entry(symbol.clone()).or_insert(0) += 1;
                                let levels = book.levels.0.len().max(book.levels.1.len());
                                let max = stats.max_book_levels.entry(symbol).or_insert(0);
                                if levels > *max {
                                    *max = levels;
                                }
                            }
                        }
                        "activeAssetCtx" => {
                            if let Ok(ctx) = serde_json::from_value::<WsAssetCtx>(response.data) {
                                let symbol = ctx.coin.clone();
                                *stats.asset_ctx_updates.entry(symbol.clone()).or_insert(0) += 1;
                                if let Ok(funding) = ctx.ctx.funding.parse::<f64>() {
                                    stats.last_funding_rates.insert(symbol.clone(), funding);
                                }
                                if let Ok(oi) = ctx.ctx.open_interest.parse::<f64>() {
                                    stats.last_open_interest.insert(symbol, oi);
                                }
                            }
                        }
                        _ => {}
                    }
                } else {
                    stats.parse_errors += 1;
                }

                // Progress indicator
                progress_counter += 1;
                if progress_counter % 1000 == 0 {
                    let elapsed = test_start.elapsed().as_secs();
                    let total_trades: usize = stats.trade_count.values().sum();
                    print!("\r  {} messages | {} trades | {}s/{}s",
                        progress_counter, total_trades, elapsed, TEST_DURATION_SECS);
                    std::io::stdout().flush().ok();
                }
            }
            Ok(Some(Ok(Message::Ping(data)))) => {
                ws_stream.send(Message::Pong(data)).await?;
            }
            Ok(Some(Ok(Message::Close(_)))) => {
                println!("\n  ⚠ Connection closed by server, reconnecting...");
                stats.connection_drops += 1;
                let (new_stream, _) = connect_async(WS_URL).await?;
                ws_stream = new_stream;
            }
            Ok(Some(Err(e))) => {
                println!("\n  ⚠ WebSocket error: {:?}", e);
                stats.connection_drops += 1;
            }
            Ok(None) => {
                println!("\n  ⚠ Stream ended");
                break;
            }
            Err(_) => {
                println!("\n  ⚠ Timeout waiting for message");
            }
            _ => {}
        }
    }
    println!();

    // Generate report
    let report = generate_report(&stats);

    // Print report
    print_report(&report);

    // Save report to file
    let report_path = format!(
        "data/validation_report_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data")?;
    let mut file = File::create(&report_path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("\n📄 Report saved to: {}", report_path);

    Ok(())
}

fn generate_report(stats: &ValidationStats) -> ValidationReport {
    let total_trades: usize = stats.trade_count.values().sum();
    let total_with_wallet: usize = stats.trades_with_wallet.values().sum();

    let wallet_coverage = if total_trades > 0 {
        (total_with_wallet as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };

    let unique_makers: usize = stats.unique_makers.values()
        .map(|s| s.len())
        .sum();
    let unique_takers: usize = stats.unique_takers.values()
        .map(|s| s.len())
        .sum();

    // Data quality score: 0-100
    let mut quality_score = 100.0;
    if stats.gaps_over_5s > 0 {
        quality_score -= (stats.gaps_over_5s as f64).min(30.0);
    }
    if stats.parse_errors > 0 {
        quality_score -= (stats.parse_errors as f64 * 0.5).min(20.0);
    }
    if stats.connection_drops > 0 {
        quality_score -= (stats.connection_drops as f64 * 5.0).min(30.0);
    }
    quality_score = quality_score.max(0.0);

    let mut warnings = Vec::new();
    let mut blockers = Vec::new();

    // Evaluate findings
    if wallet_coverage < 100.0 {
        if wallet_coverage == 0.0 {
            blockers.push("CRITICAL: No wallet addresses in trade data. Public API may not include user field.".to_string());
        } else {
            warnings.push(format!("Wallet coverage only {:.1}% - some trades missing user info", wallet_coverage));
        }
    }

    if stats.max_gap_ms > 5000 {
        warnings.push(format!("Maximum message gap: {}ms (threshold: 5000ms)", stats.max_gap_ms));
    }

    if stats.gaps_over_5s > 0 {
        warnings.push(format!("{} gaps > 5 seconds detected", stats.gaps_over_5s));
    }

    if stats.connection_drops > 0 {
        warnings.push(format!("{} connection drops during test", stats.connection_drops));
    }

    if total_trades < 100 {
        warnings.push(format!("Low trade volume: {} trades in {} seconds", total_trades, TEST_DURATION_SECS));
    }

    // Determine go/no-go
    let go_no_go = if !blockers.is_empty() {
        "NO-GO: Critical blockers found".to_string()
    } else if warnings.len() > 3 {
        "CAUTION: Multiple warnings - proceed with caution".to_string()
    } else if wallet_coverage == 0.0 {
        "PIVOT: Wallet tracking unavailable in public API - consider L4 server".to_string()
    } else {
        "GO: API meets requirements".to_string()
    };

    ValidationReport {
        timestamp: Utc::now().to_rfc3339(),
        test_duration_secs: TEST_DURATION_SECS,
        symbols_tested: SYMBOLS.iter().map(|s| s.to_string()).collect(),
        wallet_addresses_available: total_with_wallet > 0,
        wallet_coverage_pct: wallet_coverage,
        unique_maker_count: unique_makers,
        unique_taker_count: unique_takers,
        max_gap_ms: stats.max_gap_ms,
        gaps_over_5s: stats.gaps_over_5s,
        data_quality_score: quality_score,
        total_trades,
        trades_per_symbol: stats.trade_count.clone(),
        go_no_go,
        warnings,
        blockers,
    }
}

fn print_report(report: &ValidationReport) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        VALIDATION REPORT                         ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ Test Duration: {} seconds", report.test_duration_secs);
    println!("║ Symbols: {:?}", report.symbols_tested);
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ CRITICAL: WALLET TRACKING");
    if report.wallet_addresses_available {
        println!("║   ✓ Wallet addresses: AVAILABLE");
        println!("║   ✓ Coverage: {:.1}%", report.wallet_coverage_pct);
        println!("║   ✓ Unique makers: {}", report.unique_maker_count);
        println!("║   ✓ Unique takers: {}", report.unique_taker_count);
    } else {
        println!("║   ✗ Wallet addresses: NOT AVAILABLE");
        println!("║   → Public API does not include 'users' field");
        println!("║   → Need L4 order book server for wallet tracking");
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ DATA QUALITY");
    println!("║   Score: {:.1}/100", report.data_quality_score);
    println!("║   Max gap: {}ms", report.max_gap_ms);
    println!("║   Gaps >5s: {}", report.gaps_over_5s);
    println!("║   Total trades: {}", report.total_trades);
    for (symbol, count) in &report.trades_per_symbol {
        println!("║     {}: {} trades", symbol, count);
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");

    if !report.blockers.is_empty() {
        println!("║ ❌ BLOCKERS:");
        for blocker in &report.blockers {
            println!("║   • {}", blocker);
        }
    }

    if !report.warnings.is_empty() {
        println!("║ ⚠️  WARNINGS:");
        for warning in &report.warnings {
            println!("║   • {}", warning);
        }
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ VERDICT: {}", report.go_no_go);
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
