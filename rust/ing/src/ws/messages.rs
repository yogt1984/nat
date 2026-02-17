//! Hyperliquid WebSocket message types

use serde::{Deserialize, Serialize};

/// Subscription request
#[derive(Debug, Serialize)]
pub struct SubscriptionRequest {
    pub method: String,
    pub subscription: Subscription,
}

/// Subscription types
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum Subscription {
    #[serde(rename = "l2Book")]
    L2Book { coin: String },
    #[serde(rename = "trades")]
    Trades { coin: String },
    #[serde(rename = "activeAssetCtx")]
    ActiveAssetCtx { coin: String },
}

impl SubscriptionRequest {
    pub fn l2_book(coin: &str) -> Self {
        Self {
            method: "subscribe".to_string(),
            subscription: Subscription::L2Book {
                coin: coin.to_string(),
            },
        }
    }

    pub fn trades(coin: &str) -> Self {
        Self {
            method: "subscribe".to_string(),
            subscription: Subscription::Trades {
                coin: coin.to_string(),
            },
        }
    }

    pub fn active_asset_ctx(coin: &str) -> Self {
        Self {
            method: "subscribe".to_string(),
            subscription: Subscription::ActiveAssetCtx {
                coin: coin.to_string(),
            },
        }
    }
}

/// Incoming WebSocket message wrapper
#[derive(Debug, Deserialize)]
pub struct WsResponse {
    pub channel: String,
    pub data: serde_json::Value,
}

/// Parsed WebSocket message
#[derive(Debug, Clone)]
pub enum WsMessage {
    Book(WsBook),
    Trades(Vec<WsTrade>),
    AssetCtx(WsAssetCtx),
    Unknown(String),
}

/// Level 2 Order Book update
#[derive(Debug, Clone, Deserialize)]
pub struct WsBook {
    pub coin: String,
    pub levels: (Vec<WsLevel>, Vec<WsLevel>), // (bids, asks)
    pub time: u64,
}

/// Single price level in the order book
#[derive(Debug, Clone, Deserialize)]
pub struct WsLevel {
    /// Price as string for precision
    pub px: String,
    /// Size as string for precision
    pub sz: String,
    /// Number of orders at this level
    pub n: u32,
}

impl WsLevel {
    /// Parse price as f64
    pub fn price(&self) -> f64 {
        self.px.parse().unwrap_or(0.0)
    }

    /// Parse size as f64
    pub fn size(&self) -> f64 {
        self.sz.parse().unwrap_or(0.0)
    }
}

/// Trade execution
#[derive(Debug, Clone, Deserialize)]
pub struct WsTrade {
    pub coin: String,
    /// "A" = sell aggressor, "B" = buy aggressor
    pub side: String,
    /// Price
    pub px: String,
    /// Size
    pub sz: String,
    /// Transaction hash
    pub hash: String,
    /// Timestamp (ms)
    pub time: u64,
    /// Trade ID
    pub tid: u64,
}

impl WsTrade {
    /// Parse price as f64
    pub fn price(&self) -> f64 {
        self.px.parse().unwrap_or(0.0)
    }

    /// Parse size as f64
    pub fn size(&self) -> f64 {
        self.sz.parse().unwrap_or(0.0)
    }

    /// Returns true if buy aggressor
    pub fn is_buy(&self) -> bool {
        self.side == "B"
    }
}

/// Asset context (funding, OI, etc.)
#[derive(Debug, Clone, Deserialize)]
pub struct WsAssetCtx {
    pub coin: String,
    pub ctx: AssetCtxData,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetCtxData {
    /// 24h notional volume
    pub day_ntl_vlm: String,
    /// Current funding rate
    pub funding: String,
    /// Open interest
    pub open_interest: String,
    /// Oracle price
    pub oracle_px: String,
    /// Previous day price
    pub prev_day_px: String,
    /// Mark price
    pub mark_px: Option<String>,
    /// Premium (mark - oracle)
    pub premium: Option<String>,
}

impl AssetCtxData {
    pub fn funding_rate(&self) -> f64 {
        self.funding.parse().unwrap_or(0.0)
    }

    pub fn open_interest(&self) -> f64 {
        self.open_interest.parse().unwrap_or(0.0)
    }

    pub fn oracle_price(&self) -> f64 {
        self.oracle_px.parse().unwrap_or(0.0)
    }

    pub fn mark_price(&self) -> f64 {
        self.mark_px.as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| self.oracle_price())
    }

    pub fn volume_24h(&self) -> f64 {
        self.day_ntl_vlm.parse().unwrap_or(0.0)
    }

    pub fn premium(&self) -> f64 {
        self.premium.as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    }
}

/// Parse a WebSocket message
pub fn parse_ws_message(text: &str) -> Option<WsMessage> {
    let response: WsResponse = serde_json::from_str(text).ok()?;

    match response.channel.as_str() {
        "l2Book" => {
            let book: WsBook = serde_json::from_value(response.data).ok()?;
            Some(WsMessage::Book(book))
        }
        "trades" => {
            let trades: Vec<WsTrade> = serde_json::from_value(response.data).ok()?;
            Some(WsMessage::Trades(trades))
        }
        "activeAssetCtx" => {
            let ctx: WsAssetCtx = serde_json::from_value(response.data).ok()?;
            Some(WsMessage::AssetCtx(ctx))
        }
        _ => Some(WsMessage::Unknown(response.channel)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_book_message() {
        let json = r#"{
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "50000.0", "sz": "1.5", "n": 3}],
                    [{"px": "50001.0", "sz": "2.0", "n": 2}]
                ],
                "time": 1704067200000
            }
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Book(book) => {
                assert_eq!(book.coin, "BTC");
                assert_eq!(book.levels.0.len(), 1);
                assert_eq!(book.levels.0[0].price(), 50000.0);
            }
            _ => panic!("Expected Book message"),
        }
    }

    #[test]
    fn test_parse_trade_message() {
        let json = r#"{
            "channel": "trades",
            "data": [{
                "coin": "BTC",
                "side": "B",
                "px": "50000.0",
                "sz": "0.1",
                "hash": "0x123",
                "time": 1704067200000,
                "tid": 12345
            }]
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => {
                assert_eq!(trades.len(), 1);
                assert!(trades[0].is_buy());
                assert_eq!(trades[0].price(), 50000.0);
            }
            _ => panic!("Expected Trades message"),
        }
    }
}
