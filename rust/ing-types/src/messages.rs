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
    /// [maker_address, taker_address] - Hyperliquid unique!
    /// May not always be present in public API
    #[serde(default)]
    pub users: Option<(String, String)>,
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

    /// Get maker wallet address (if available)
    pub fn maker_address(&self) -> Option<&str> {
        self.users.as_ref().map(|(maker, _)| maker.as_str())
    }

    /// Get taker wallet address (if available)
    pub fn taker_address(&self) -> Option<&str> {
        self.users.as_ref().map(|(_, taker)| taker.as_str())
    }

    /// Check if wallet addresses are present
    pub fn has_wallet_info(&self) -> bool {
        self.users.is_some()
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
        self.mark_px
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| self.oracle_price())
    }

    pub fn volume_24h(&self) -> f64 {
        self.day_ntl_vlm.parse().unwrap_or(0.0)
    }

    pub fn premium(&self) -> f64 {
        self.premium
            .as_ref()
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

    // ========================================================================
    // Message parsing: edge cases and malformed input
    // ========================================================================

    #[test]
    fn test_parse_empty_string() {
        assert!(parse_ws_message("").is_none());
    }

    #[test]
    fn test_parse_invalid_json() {
        assert!(parse_ws_message("{not valid json}").is_none());
    }

    #[test]
    fn test_parse_valid_json_missing_channel() {
        assert!(parse_ws_message(r#"{"data": {}}"#).is_none());
    }

    #[test]
    fn test_parse_valid_json_missing_data() {
        assert!(parse_ws_message(r#"{"channel": "l2Book"}"#).is_none());
    }

    #[test]
    fn test_parse_unknown_channel() {
        let json = r#"{"channel": "someNewChannel", "data": {}}"#;
        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Unknown(ch) => assert_eq!(ch, "someNewChannel"),
            _ => panic!("Expected Unknown message"),
        }
    }

    #[test]
    fn test_parse_book_wrong_data_shape() {
        // channel is l2Book but data is not a valid WsBook
        let json = r#"{"channel": "l2Book", "data": {"wrong": "shape"}}"#;
        assert!(parse_ws_message(json).is_none());
    }

    #[test]
    fn test_parse_trades_wrong_data_shape() {
        // channel is trades but data is not a valid Vec<WsTrade>
        let json = r#"{"channel": "trades", "data": "not_an_array"}"#;
        assert!(parse_ws_message(json).is_none());
    }

    #[test]
    fn test_parse_asset_ctx_wrong_data_shape() {
        let json = r#"{"channel": "activeAssetCtx", "data": 42}"#;
        assert!(parse_ws_message(json).is_none());
    }

    #[test]
    fn test_parse_asset_ctx_message() {
        let json = r#"{
            "channel": "activeAssetCtx",
            "data": {
                "coin": "ETH",
                "ctx": {
                    "dayNtlVlm": "5000000.0",
                    "funding": "0.00015",
                    "openInterest": "2500000.0",
                    "oraclePx": "3200.0",
                    "prevDayPx": "3100.0",
                    "markPx": "3201.5",
                    "premium": "1.5"
                }
            }
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::AssetCtx(ctx) => {
                assert_eq!(ctx.coin, "ETH");
                assert!((ctx.ctx.funding_rate() - 0.00015).abs() < 1e-10);
                assert!((ctx.ctx.open_interest() - 2500000.0).abs() < 1e-6);
                assert!((ctx.ctx.oracle_price() - 3200.0).abs() < 1e-6);
                assert!((ctx.ctx.mark_price() - 3201.5).abs() < 1e-6);
                assert!((ctx.ctx.volume_24h() - 5000000.0).abs() < 1e-6);
                assert!((ctx.ctx.premium() - 1.5).abs() < 1e-6);
            }
            _ => panic!("Expected AssetCtx message"),
        }
    }

    #[test]
    fn test_parse_asset_ctx_optional_fields_missing() {
        let json = r#"{
            "channel": "activeAssetCtx",
            "data": {
                "coin": "SOL",
                "ctx": {
                    "dayNtlVlm": "100000.0",
                    "funding": "-0.0001",
                    "openInterest": "500000.0",
                    "oraclePx": "150.0",
                    "prevDayPx": "148.0"
                }
            }
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::AssetCtx(ctx) => {
                assert_eq!(ctx.coin, "SOL");
                // mark_px missing -> falls back to oracle_px
                assert!((ctx.ctx.mark_price() - 150.0).abs() < 1e-6);
                // premium missing -> 0.0
                assert!((ctx.ctx.premium() - 0.0).abs() < 1e-10);
            }
            _ => panic!("Expected AssetCtx"),
        }
    }

    #[test]
    fn test_parse_trade_sell_aggressor() {
        let json = r#"{
            "channel": "trades",
            "data": [{
                "coin": "BTC",
                "side": "A",
                "px": "49999.5",
                "sz": "2.5",
                "hash": "0xabc",
                "time": 1704067200000,
                "tid": 99999
            }]
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => {
                assert!(!trades[0].is_buy());
                assert_eq!(trades[0].side, "A");
                assert!((trades[0].size() - 2.5).abs() < 1e-10);
            }
            _ => panic!("Expected Trades"),
        }
    }

    #[test]
    fn test_parse_trade_with_wallet_info() {
        let json = r#"{
            "channel": "trades",
            "data": [{
                "coin": "BTC",
                "side": "B",
                "px": "50000.0",
                "sz": "1.0",
                "hash": "0x123",
                "time": 1704067200000,
                "tid": 1,
                "users": ["0xmaker_addr", "0xtaker_addr"]
            }]
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => {
                assert!(trades[0].has_wallet_info());
                assert_eq!(trades[0].maker_address(), Some("0xmaker_addr"));
                assert_eq!(trades[0].taker_address(), Some("0xtaker_addr"));
            }
            _ => panic!("Expected Trades"),
        }
    }

    #[test]
    fn test_parse_trade_without_wallet_info() {
        let json = r#"{
            "channel": "trades",
            "data": [{
                "coin": "BTC",
                "side": "B",
                "px": "50000.0",
                "sz": "1.0",
                "hash": "0x123",
                "time": 1704067200000,
                "tid": 1
            }]
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => {
                assert!(!trades[0].has_wallet_info());
                assert_eq!(trades[0].maker_address(), None);
                assert_eq!(trades[0].taker_address(), None);
            }
            _ => panic!("Expected Trades"),
        }
    }

    #[test]
    fn test_parse_multiple_trades() {
        let json = r#"{
            "channel": "trades",
            "data": [
                {"coin": "BTC", "side": "B", "px": "50000", "sz": "1.0", "hash": "0x1", "time": 100, "tid": 1},
                {"coin": "BTC", "side": "A", "px": "49999", "sz": "0.5", "hash": "0x2", "time": 101, "tid": 2},
                {"coin": "BTC", "side": "B", "px": "50001", "sz": "2.0", "hash": "0x3", "time": 102, "tid": 3}
            ]
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => {
                assert_eq!(trades.len(), 3);
                assert!(trades[0].is_buy());
                assert!(!trades[1].is_buy());
                assert!(trades[2].is_buy());
                assert_eq!(trades[0].tid, 1);
                assert_eq!(trades[2].tid, 3);
            }
            _ => panic!("Expected Trades"),
        }
    }

    #[test]
    fn test_parse_empty_trades_array() {
        let json = r#"{"channel": "trades", "data": []}"#;
        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Trades(trades) => assert!(trades.is_empty()),
            _ => panic!("Expected Trades"),
        }
    }

    #[test]
    fn test_parse_book_multi_level() {
        let json = r#"{
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [
                        {"px": "50000.0", "sz": "1.0", "n": 3},
                        {"px": "49999.5", "sz": "2.0", "n": 5},
                        {"px": "49999.0", "sz": "3.0", "n": 2}
                    ],
                    [
                        {"px": "50001.0", "sz": "1.5", "n": 4},
                        {"px": "50001.5", "sz": "0.5", "n": 1}
                    ]
                ],
                "time": 1704067200000
            }
        }"#;

        let msg = parse_ws_message(json).unwrap();
        match msg {
            WsMessage::Book(book) => {
                assert_eq!(book.levels.0.len(), 3); // 3 bid levels
                assert_eq!(book.levels.1.len(), 2); // 2 ask levels
                // Best bid/ask
                assert!((book.levels.0[0].price() - 50000.0).abs() < 1e-6);
                assert!((book.levels.1[0].price() - 50001.0).abs() < 1e-6);
                // Deeper level
                assert_eq!(book.levels.0[1].n, 5);
                assert!((book.levels.0[2].size() - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected Book"),
        }
    }

    #[test]
    fn test_ws_level_unparseable_defaults_to_zero() {
        let level = WsLevel {
            px: "not_a_number".to_string(),
            sz: "also_bad".to_string(),
            n: 1,
        };
        assert_eq!(level.price(), 0.0);
        assert_eq!(level.size(), 0.0);
    }

    #[test]
    fn test_parse_nested_json_garbage_in_data() {
        // Valid JSON structure but data is a nested object that doesn't match any schema
        let json = r#"{"channel": "l2Book", "data": {"deeply": {"nested": {"garbage": true}}}}"#;
        assert!(parse_ws_message(json).is_none());
    }

    #[test]
    fn test_parse_empty_object() {
        assert!(parse_ws_message("{}").is_none());
    }

    #[test]
    fn test_parse_null_literal() {
        assert!(parse_ws_message("null").is_none());
    }

    #[test]
    fn test_parse_array_instead_of_object() {
        assert!(parse_ws_message("[1, 2, 3]").is_none());
    }

    #[test]
    fn test_parse_very_long_string_no_crash() {
        // 100KB of garbage — must not panic or OOM
        let long = "x".repeat(100_000);
        assert!(parse_ws_message(&long).is_none());
    }

    #[test]
    fn test_subscription_request_serialization() {
        let req = SubscriptionRequest::l2_book("ETH");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("subscribe"));
        assert!(json.contains("l2Book"));
        assert!(json.contains("ETH"));

        let req = SubscriptionRequest::trades("SOL");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("trades"));
        assert!(json.contains("SOL"));

        let req = SubscriptionRequest::active_asset_ctx("BTC");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("activeAssetCtx"));
        assert!(json.contains("BTC"));
    }
}
