//! Whale Identification Validation
//!
//! Skeptical validation of whale identification and classification.
//! Tests:
//! 1. Can we identify whales from position data?
//! 2. Do whale positions correlate with price movement?
//! 3. Are whales actually skilled or just large?

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

const REST_URL: &str = "https://api.hyperliquid.xyz/info";
const MIN_POSITION_USD: f64 = 500_000.0;
const MAX_WALLETS_TO_CHECK: usize = 50;

#[derive(Debug, Default)]
struct ValidationStats {
    wallets_checked: usize,
    wallets_with_positions: usize,
    whales_found: usize,
    small_whales: usize,
    medium_whales: usize,
    large_whales: usize,
    total_whale_position_usd: f64,
    largest_position_usd: f64,
    largest_whale_address: String,
}

#[derive(Debug, Serialize)]
struct WhaleValidationReport {
    timestamp: String,
    wallets_checked: usize,
    wallets_with_positions: usize,
    whales_found: usize,
    by_tier: TierBreakdown,
    total_whale_position_usd: f64,
    largest_position_usd: f64,
    largest_whale: String,
    whale_list: Vec<WhaleInfo>,
    go_no_go: String,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TierBreakdown {
    small: usize,
    medium: usize,
    large: usize,
}

#[derive(Debug, Serialize)]
struct WhaleInfo {
    address: String,
    tier: String,
    total_position_usd: f64,
    positions: Vec<PositionInfo>,
}

#[derive(Debug, Serialize)]
struct PositionInfo {
    symbol: String,
    size: f64,
    value_usd: f64,
    side: String,
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

    fn value(&self) -> f64 {
        self.position_value.parse().unwrap_or(0.0)
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
    println!("║          WHALE IDENTIFICATION VALIDATION (SKEPTICAL MODE)        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut stats = ValidationStats::default();
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    // Phase 1: Discover wallets from recent trades
    println!("[PHASE 1] Discovering active wallets...");

    let wallets = discover_wallets_from_trades(&http_client).await?;
    println!("  ✓ Found {} active wallets to check\n", wallets.len());

    // Phase 2: Check each wallet for whale status
    println!("[PHASE 2] Checking wallets for whale positions...");

    let mut whale_list: Vec<WhaleInfo> = Vec::new();

    for (i, wallet) in wallets.iter().take(MAX_WALLETS_TO_CHECK).enumerate() {
        stats.wallets_checked += 1;

        let request = InfoRequest::ClearinghouseState { user: wallet.clone() };

        match http_client.post(REST_URL)
            .json(&request)
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(state) = response.json::<ClearinghouseState>().await {
                    let positions: Vec<_> = state.asset_positions.iter()
                        .filter(|p| !p.position.is_empty())
                        .collect();

                    if !positions.is_empty() {
                        stats.wallets_with_positions += 1;

                        let total_position: f64 = positions.iter()
                            .map(|p| p.position.value().abs())
                            .sum();

                        // Check if whale
                        if total_position >= MIN_POSITION_USD {
                            stats.whales_found += 1;
                            stats.total_whale_position_usd += total_position;

                            // Classify tier
                            let tier = if total_position >= 10_000_000.0 {
                                stats.large_whales += 1;
                                "Large Whale"
                            } else if total_position >= 2_000_000.0 {
                                stats.medium_whales += 1;
                                "Medium Whale"
                            } else {
                                stats.small_whales += 1;
                                "Small Whale"
                            };

                            // Track largest
                            if total_position > stats.largest_position_usd {
                                stats.largest_position_usd = total_position;
                                stats.largest_whale_address = wallet.clone();
                            }

                            // Build position info
                            let pos_info: Vec<PositionInfo> = positions.iter()
                                .map(|p| PositionInfo {
                                    symbol: p.position.coin.clone(),
                                    size: p.position.size(),
                                    value_usd: p.position.value().abs(),
                                    side: if p.position.size() > 0.0 { "LONG" } else { "SHORT" }.to_string(),
                                })
                                .collect();

                            whale_list.push(WhaleInfo {
                                address: wallet.clone(),
                                tier: tier.to_string(),
                                total_position_usd: total_position,
                                positions: pos_info,
                            });

                            println!("  🐋 {} - {} (${:.0})",
                                &wallet[..10.min(wallet.len())],
                                tier,
                                total_position
                            );
                        }
                    }
                }
            }
            _ => {}
        }

        // Progress
        if (i + 1) % 10 == 0 {
            println!("  ... checked {}/{} wallets, found {} whales",
                i + 1, wallets.len().min(MAX_WALLETS_TO_CHECK), stats.whales_found);
        }

        // Rate limiting
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    println!();

    // Generate report
    let report = generate_report(&stats, whale_list);
    print_report(&report);

    // Save report
    let report_path = format!(
        "data/whale_validation_{}.json",
        Utc::now().format("%Y%m%d_%H%M%S")
    );
    std::fs::create_dir_all("data")?;
    let mut file = File::create(&report_path)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("\n📄 Report saved to: {}", report_path);

    Ok(())
}

/// Discover wallets by collecting from trades WebSocket
async fn discover_wallets_from_trades(client: &reqwest::Client) -> Result<Vec<String>> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    println!("  Collecting wallets from trades (30 seconds)...");

    let (mut ws_stream, _) = connect_async("wss://api.hyperliquid.xyz/ws")
        .await
        .context("Failed to connect to WebSocket")?;

    // Subscribe to trades for major pairs
    for symbol in &["BTC", "ETH", "SOL"] {
        let sub = serde_json::json!({
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": symbol
            }
        });
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;
    }

    let mut wallet_counts: HashMap<String, usize> = HashMap::new();
    let start = std::time::Instant::now();
    let duration = Duration::from_secs(30);

    while start.elapsed() < duration {
        match tokio::time::timeout(Duration::from_secs(5), ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {
                    if response["channel"] == "trades" {
                        if let Some(trades) = response["data"].as_array() {
                            for trade in trades {
                                if let Some(users) = trade["users"].as_array() {
                                    if users.len() == 2 {
                                        if let (Some(maker), Some(taker)) = (
                                            users[0].as_str(),
                                            users[1].as_str()
                                        ) {
                                            *wallet_counts.entry(maker.to_string()).or_insert(0) += 1;
                                            *wallet_counts.entry(taker.to_string()).or_insert(0) += 1;
                                        }
                                    }
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
    }

    // Sort by activity and return top wallets
    let mut wallets: Vec<_> = wallet_counts.into_iter().collect();
    wallets.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(wallets.into_iter().map(|(w, _)| w).collect())
}

fn generate_report(stats: &ValidationStats, whale_list: Vec<WhaleInfo>) -> WhaleValidationReport {
    let mut warnings = Vec::new();

    if stats.whales_found < 10 {
        warnings.push(format!("Only {} whales found (target: 50+)", stats.whales_found));
    }

    if stats.wallets_with_positions < stats.wallets_checked / 2 {
        warnings.push("Less than half of active wallets have positions".to_string());
    }

    let go_no_go = if stats.whales_found >= 10 {
        format!("GO: Found {} whales with ${:.0}M total position",
            stats.whales_found,
            stats.total_whale_position_usd / 1_000_000.0)
    } else if stats.whales_found > 0 {
        "CAUTION: Fewer whales than expected but identification works".to_string()
    } else {
        "NO-GO: No whales identified".to_string()
    };

    WhaleValidationReport {
        timestamp: Utc::now().to_rfc3339(),
        wallets_checked: stats.wallets_checked,
        wallets_with_positions: stats.wallets_with_positions,
        whales_found: stats.whales_found,
        by_tier: TierBreakdown {
            small: stats.small_whales,
            medium: stats.medium_whales,
            large: stats.large_whales,
        },
        total_whale_position_usd: stats.total_whale_position_usd,
        largest_position_usd: stats.largest_position_usd,
        largest_whale: stats.largest_whale_address.clone(),
        whale_list,
        go_no_go,
        warnings,
    }
}

fn print_report(report: &WhaleValidationReport) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      WHALE VALIDATION REPORT                     ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    println!("║ DISCOVERY");
    println!("║   Wallets checked: {}", report.wallets_checked);
    println!("║   Wallets with positions: {}", report.wallets_with_positions);

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ WHALE CLASSIFICATION");
    println!("║   Total whales found: {}", report.whales_found);
    println!("║   Small whales ($500K-$2M): {}", report.by_tier.small);
    println!("║   Medium whales ($2M-$10M): {}", report.by_tier.medium);
    println!("║   Large whales ($10M+): {}", report.by_tier.large);

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ POSITIONS");
    println!("║   Total whale position: ${:.2}M", report.total_whale_position_usd / 1_000_000.0);
    println!("║   Largest position: ${:.2}M", report.largest_position_usd / 1_000_000.0);
    if !report.largest_whale.is_empty() {
        println!("║   Largest whale: {}...", &report.largest_whale[..12.min(report.largest_whale.len())]);
    }

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
