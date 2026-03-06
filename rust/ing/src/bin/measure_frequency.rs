//! Measure Hyperliquid WebSocket update frequency
//!
//! Empirically measures how fast data arrives from Hyperliquid.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
const MEASUREMENT_DURATION_SECS: u64 = 15;

#[derive(Default)]
struct ChannelStats {
    message_count: usize,
    timestamps: Vec<Instant>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║      HYPERLIQUID WEBSOCKET FREQUENCY MEASUREMENT                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let (mut ws_stream, _) = connect_async(WS_URL)
        .await
        .context("Failed to connect")?;

    println!("Connected to Hyperliquid WebSocket\n");

    // Subscribe to multiple channels
    for coin in &["BTC", "ETH"] {
        // Trades
        let sub = serde_json::json!({
            "method": "subscribe",
            "subscription": {"type": "trades", "coin": coin}
        });
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;

        // L2 Book
        let sub = serde_json::json!({
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": coin}
        });
        ws_stream.send(Message::Text(serde_json::to_string(&sub)?)).await?;
    }

    println!("Subscribed to BTC & ETH trades + l2Book");
    println!("Measuring for {} seconds...\n", MEASUREMENT_DURATION_SECS);

    let mut stats: HashMap<String, ChannelStats> = HashMap::new();
    let start = Instant::now();
    let duration = Duration::from_secs(MEASUREMENT_DURATION_SECS);

    while start.elapsed() < duration {
        match tokio::time::timeout(Duration::from_millis(100), ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => {
                let now = Instant::now();
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                    let channel = data["channel"].as_str().unwrap_or("unknown");
                    let coin = data["data"]["coin"].as_str()
                        .or_else(|| data["data"].get("coin").and_then(|c| c.as_str()))
                        .unwrap_or("");

                    let key = if coin.is_empty() {
                        channel.to_string()
                    } else {
                        format!("{}:{}", channel, coin)
                    };

                    let entry = stats.entry(key).or_default();
                    entry.message_count += 1;
                    entry.timestamps.push(now);
                }
            }
            Ok(Some(Ok(Message::Ping(data)))) => {
                ws_stream.send(Message::Pong(data)).await?;
            }
            _ => {}
        }
    }

    // Calculate and print results
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    FREQUENCY ANALYSIS RESULTS                    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut channels: Vec<_> = stats.iter().collect();
    channels.sort_by_key(|(k, _)| k.as_str());

    let mut total_messages = 0;

    for (channel, data) in &channels {
        if data.timestamps.len() < 2 {
            continue;
        }

        total_messages += data.message_count;

        let elapsed = data.timestamps.last().unwrap().duration_since(*data.timestamps.first().unwrap());
        let freq = data.message_count as f64 / elapsed.as_secs_f64();

        // Calculate intervals
        let intervals: Vec<f64> = data.timestamps.windows(2)
            .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
            .collect();

        let avg_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let min_interval = intervals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_interval = intervals.iter().cloned().fold(0.0, f64::max);

        println!("║");
        println!("║ {}", channel);
        println!("║   Messages:      {:>8}", data.message_count);
        println!("║   Frequency:     {:>8.1} Hz (msg/sec)", freq);
        println!("║   Avg interval:  {:>8.1} ms", avg_interval);
        println!("║   Min interval:  {:>8.1} ms", min_interval);
        println!("║   Max interval:  {:>8.1} ms", max_interval);
    }

    let total_freq = total_messages as f64 / MEASUREMENT_DURATION_SECS as f64;

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ TOTALS");
    println!("║   Total messages:     {:>8}", total_messages);
    println!("║   Combined frequency: {:>8.1} Hz", total_freq);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ INTERPRETATION");
    println!("║");

    // Find trades frequency
    let btc_trades_freq = channels.iter()
        .find(|(k, _)| k.contains("trades:BTC"))
        .map(|(_, d)| {
            let elapsed = d.timestamps.last().unwrap().duration_since(*d.timestamps.first().unwrap());
            d.message_count as f64 / elapsed.as_secs_f64()
        })
        .unwrap_or(0.0);

    let book_freq = channels.iter()
        .find(|(k, _)| k.contains("l2Book:BTC"))
        .map(|(_, d)| {
            let elapsed = d.timestamps.last().unwrap().duration_since(*d.timestamps.first().unwrap());
            d.message_count as f64 / elapsed.as_secs_f64()
        })
        .unwrap_or(0.0);

    println!("║   Trades stream:  ~{:.0} Hz (event-driven, varies with activity)", btc_trades_freq);
    println!("║   L2 Book stream: ~{:.0} Hz (batched by block)", book_freq);
    println!("║");
    println!("║   Your 1 Hz display is ARTIFICIAL - raw data is much faster!");
    println!("║   For HFT: Use 10-100 Hz feature emission");
    println!("║   For signals: 1-10 Hz is typically sufficient");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
