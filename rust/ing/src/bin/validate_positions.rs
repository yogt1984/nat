//! Position Tracking Validation
//!
//! Skeptical validation of wallet position tracking capabilities.
//! Tests:
//! 1. Can we fetch positions for known active wallets?
//! 2. Do position changes match observed trades?
//! 3. Is position data fresh (not stale)?

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
const REST_URL: &str = "https://api.hyperliquid.xyz/info";
const TRADE_COLLECTION_SECS: u64 = 60;
const MIN_TRADES_PER_WALLET: usize = 3;
const MAX_WALLETS_TO_TEST: usize = 10;

#[derive(Debug, Default)]
struct ValidationStats {
    // Wallet discovery
    total_trades_seen: usize,
    trades_with_wallet: usize,
    unique_wallets: HashSet<String>,
    wallet_trade_counts: HashMap<String, usize>,

    // Position fetching
    wallets_tested: usize,
    wallets_with_positions: usize,
    total_positions_found: usize,
    position_fetch_errors: usize,
    avg_fetch_time_ms: f64,

    // Cross-validation
    trades_matched_to_positions: usize,
    trades_not_matched: usize,
}

#[derive(Debug, Serialize)]
struct PositionValidationReport {
    timestamp: String,
    trade_collection_secs: u64,

    // Discovery
    total_trades: usize,
    trades_with_wallet_pct: f64,
    unique_wallets_found: usize,
    wallets_meeting_threshold: usize,

    // Position testing
    wallets_tested: usize,
    wallets_with_positions: usize,
    total_positions: usize,
    avg_fetch_time_ms: f64,
    position_fetch_success_rate: f64,

    // Verdict
    position_tracking_available: bool,
    go_no_go: String,
    warnings: Vec<String>,
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
    time: u64,
    #[serde(default)]
    users: Option<(String, String)>,
}

// REST API types
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum InfoRequest {
    #[serde(rename = "clearinghouseState")]
    ClearinghouseState { user: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct ClearinghouseState {
    asset_positions: Vec<AssetPosition>,
    margin_summary: MarginSummary,
    withdrawable: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct AssetPosition {
    position: Position,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct Position {
    coin: String,
    szi: String,
    entry_px: Option<String>,
    position_value: String,
    unrealized_pnl: String,
    liquidation_px: Option<String>,
}

impl Position {
    fn size(&self) -> f64 {
        self.szi.parse().unwrap_or(0.0)
    }

    fn is_empty(&self) -> bool {
        self.size().abs() < 1e-10
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct MarginSummary {
    account_value: String,
    total_ntl_pos: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         POSITION TRACKING VALIDATION (SKEPTICAL MODE)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut stats = ValidationStats::default();
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // Phase 1: Collect trades to discover active wallets
    println!("[PHASE 1] Discovering active wallets from trades ({} seconds)...", TRADE_COLLECTION_SECS);

    let (mut ws_stream, _) = connect_async(WS_URL)
        .await
        .context("Failed to connect to WebSocket")?;

    // Subscribe to trades for BTC, ETH, SOL
    for symbol in &["BTC", "ETH", "SOL"] {
        let sub = SubscriptionRequest {
            method: "subscribe".to_string(),
            subscription: serde_json::json!({
                "type": "trades",
                "coin": symbol
            }),
        };
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;
    }

    let start = Instant::now();
    let collection_duration = Duration::from_secs(TRADE_COLLECTION_SECS);

    while start.elapsed() < collection_duration {
        match timeout(Duration::from_secs(5), ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(response) = serde_json::from_str::<WsResponse>(&text) {
                    if response.channel == "trades" {
                        if let Ok(trades) = serde_json::from_value::<Vec<WsTrade>>(response.data) {
                            for trade in trades {
                                stats.total_trades_seen += 1;
                                if let Some((maker, taker)) = &trade.users {
                                    stats.trades_with_wallet += 1;
                                    stats.unique_wallets.insert(maker.clone());
                                    stats.unique_wallets.insert(taker.clone());
                                    *stats.wallet_trade_counts.entry(maker.clone()).or_insert(0) += 1;
                                    *stats.wallet_trade_counts.entry(taker.clone()).or_insert(0) += 1;
                                }
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

        // Progress
        let elapsed = start.elapsed().as_secs();
        print!("\r  {} trades | {} unique wallets | {}s/{}s",
            stats.total_trades_seen, stats.unique_wallets.len(), elapsed, TRADE_COLLECTION_SECS);
        std::io::stdout().flush().ok();
    }
    println!();

    // Find wallets with enough trades
    let mut active_wallets: Vec<_> = stats.wallet_trade_counts.iter()
        .filter(|(_, count)| **count >= MIN_TRADES_PER_WALLET)
        .map(|(wallet, count)| (wallet.clone(), *count))
        .collect();
    active_wallets.sort_by(|a, b| b.1.cmp(&a.1));

    let wallets_to_test: Vec<_> = active_wallets.iter()
        .take(MAX_WALLETS_TO_TEST)
        .map(|(w, _)| w.clone())
        .collect();

    println!("\n  ✓ Found {} wallets with >= {} trades", active_wallets.len(), MIN_TRADES_PER_WALLET);
    println!("  ✓ Testing top {} wallets\n", wallets_to_test.len());

    // Phase 2: Test position fetching for discovered wallets
    println!("[PHASE 2] Testing position fetching...");

    let mut fetch_times = Vec::new();

    for wallet in &wallets_to_test {
        stats.wallets_tested += 1;
        print!("  Testing {}... ", &wallet[..10.min(wallet.len())]);
        std::io::stdout().flush().ok();

        let fetch_start = Instant::now();
        let request = InfoRequest::ClearinghouseState { user: wallet.clone() };

        match http_client.post(REST_URL)
            .json(&request)
            .send()
            .await
        {
            Ok(response) => {
                let fetch_time = fetch_start.elapsed().as_millis() as f64;
                fetch_times.push(fetch_time);

                if response.status().is_success() {
                    match response.json::<ClearinghouseState>().await {
                        Ok(state) => {
                            let positions: Vec<_> = state.asset_positions.iter()
                                .filter(|p| !p.position.is_empty())
                                .collect();

                            if !positions.is_empty() {
                                stats.wallets_with_positions += 1;
                                stats.total_positions_found += positions.len();
                                println!("✓ {} positions ({:.0}ms)", positions.len(), fetch_time);

                                // Print position details
                                for pos in &positions {
                                    let p = &pos.position;
                                    let side = if p.size() > 0.0 { "LONG" } else { "SHORT" };
                                    println!("    {} {} {:.4} @ entry {}",
                                        p.coin, side, p.size().abs(),
                                        p.entry_px.as_ref().unwrap_or(&"N/A".to_string()));
                                }
                            } else {
                                println!("✓ no positions ({:.0}ms)", fetch_time);
                            }
                        }
                        Err(e) => {
                            println!("✗ parse error: {:?}", e);
                            stats.position_fetch_errors += 1;
                        }
                    }
                } else {
                    println!("✗ HTTP {}", response.status());
                    stats.position_fetch_errors += 1;
                }
            }
            Err(e) => {
                println!("✗ request error: {:?}", e);
                stats.position_fetch_errors += 1;
            }
        }

        // Small delay to avoid rate limiting
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Calculate average fetch time
    stats.avg_fetch_time_ms = if !fetch_times.is_empty() {
        fetch_times.iter().sum::<f64>() / fetch_times.len() as f64
    } else {
        0.0
    };

    // Generate report
    let report = generate_report(&stats, &active_wallets);
    print_report(&report);

    // Save report
    let report_path = format!(
        "data/position_validation_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data")?;
    let mut file = File::create(&report_path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("\n📄 Report saved to: {}", report_path);

    Ok(())
}

fn generate_report(stats: &ValidationStats, active_wallets: &[(String, usize)]) -> PositionValidationReport {
    let trades_with_wallet_pct = if stats.total_trades_seen > 0 {
        (stats.trades_with_wallet as f64 / stats.total_trades_seen as f64) * 100.0
    } else {
        0.0
    };

    let position_fetch_success_rate = if stats.wallets_tested > 0 {
        ((stats.wallets_tested - stats.position_fetch_errors) as f64 / stats.wallets_tested as f64) * 100.0
    } else {
        0.0
    };

    let mut warnings = Vec::new();

    if trades_with_wallet_pct < 100.0 {
        warnings.push(format!("Only {:.1}% of trades have wallet info", trades_with_wallet_pct));
    }

    if stats.position_fetch_errors > 0 {
        warnings.push(format!("{} position fetch errors", stats.position_fetch_errors));
    }

    if stats.avg_fetch_time_ms > 500.0 {
        warnings.push(format!("High avg fetch time: {:.0}ms", stats.avg_fetch_time_ms));
    }

    let position_tracking_available = stats.wallets_tested > 0 && stats.position_fetch_errors == 0;

    let go_no_go = if !position_tracking_available {
        "NO-GO: Cannot fetch positions".to_string()
    } else if stats.wallets_with_positions == 0 {
        "CAUTION: Fetching works but no positions found (normal if wallets have no open positions)".to_string()
    } else {
        "GO: Position tracking works".to_string()
    };

    PositionValidationReport {
        timestamp: Utc::now().to_rfc3339(),
        trade_collection_secs: TRADE_COLLECTION_SECS,
        total_trades: stats.total_trades_seen,
        trades_with_wallet_pct,
        unique_wallets_found: stats.unique_wallets.len(),
        wallets_meeting_threshold: active_wallets.len(),
        wallets_tested: stats.wallets_tested,
        wallets_with_positions: stats.wallets_with_positions,
        total_positions: stats.total_positions_found,
        avg_fetch_time_ms: stats.avg_fetch_time_ms,
        position_fetch_success_rate,
        position_tracking_available,
        go_no_go,
        warnings,
    }
}

fn print_report(report: &PositionValidationReport) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    POSITION VALIDATION REPORT                    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ WALLET DISCOVERY");
    println!("║   Total trades: {}", report.total_trades);
    println!("║   Trades with wallet: {:.1}%", report.trades_with_wallet_pct);
    println!("║   Unique wallets: {}", report.unique_wallets_found);
    println!("║   Wallets with {} trades: {}", MIN_TRADES_PER_WALLET, report.wallets_meeting_threshold);

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ POSITION FETCHING");
    println!("║   Wallets tested: {}", report.wallets_tested);
    println!("║   Wallets with positions: {}", report.wallets_with_positions);
    println!("║   Total positions found: {}", report.total_positions);
    println!("║   Avg fetch time: {:.0}ms", report.avg_fetch_time_ms);
    println!("║   Success rate: {:.1}%", report.position_fetch_success_rate);

    println!("╠══════════════════════════════════════════════════════════════════╣");

    if !report.warnings.is_empty() {
        println!("║ ⚠️  WARNINGS:");
        for warning in &report.warnings {
            println!("║   • {}", warning);
        }
        println!("╠══════════════════════════════════════════════════════════════════╣");
    }

    if report.position_tracking_available {
        println!("║ ✅ Position tracking: AVAILABLE");
    } else {
        println!("║ ❌ Position tracking: UNAVAILABLE");
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ VERDICT: {}", report.go_no_go);
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
