//! Hyperliquid REST API client

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

const INFO_URL: &str = "https://api.hyperliquid.xyz/info";

/// REST API client for Hyperliquid
#[derive(Clone)]
pub struct HyperliquidRestClient {
    client: Client,
    base_url: String,
}

impl HyperliquidRestClient {
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: INFO_URL.to_string(),
        })
    }

    /// Fetch clearinghouse state for a wallet
    pub async fn get_clearinghouse_state(&self, wallet: &str) -> Result<ClearinghouseState> {
        let request = InfoRequest::ClearinghouseState {
            user: wallet.to_string(),
        };

        let response = self.client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .context("Failed to send clearinghouse request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Clearinghouse request failed: {} - {}", status, body);
        }

        let state: ClearinghouseState = response
            .json()
            .await
            .context("Failed to parse clearinghouse response")?;

        Ok(state)
    }

    /// Fetch user fills (recent trades for a wallet)
    pub async fn get_user_fills(&self, wallet: &str) -> Result<Vec<UserFill>> {
        let request = InfoRequest::UserFills {
            user: wallet.to_string(),
        };

        let response = self.client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .context("Failed to send user fills request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("User fills request failed: {} - {}", status, body);
        }

        let fills: Vec<UserFill> = response
            .json()
            .await
            .context("Failed to parse user fills response")?;

        Ok(fills)
    }

    /// Fetch metadata for all assets
    pub async fn get_meta(&self) -> Result<Meta> {
        let request = InfoRequest::Meta;

        let response = self.client
            .post(&self.base_url)
            .json(&request)
            .send()
            .await
            .context("Failed to send meta request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Meta request failed: {} - {}", status, body);
        }

        let meta: Meta = response
            .json()
            .await
            .context("Failed to parse meta response")?;

        Ok(meta)
    }
}

impl Default for HyperliquidRestClient {
    fn default() -> Self {
        Self::new().expect("Failed to create REST client")
    }
}

// Request types
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum InfoRequest {
    #[serde(rename = "clearinghouseState")]
    ClearinghouseState { user: String },
    #[serde(rename = "userFills")]
    UserFills { user: String },
    #[serde(rename = "meta")]
    Meta,
}

// Response types

/// Clearinghouse state for a user
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearinghouseState {
    pub asset_positions: Vec<AssetPosition>,
    pub margin_summary: MarginSummary,
    #[serde(default)]
    pub cross_margin_summary: Option<CrossMarginSummary>,
    pub withdrawable: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetPosition {
    pub position: Position,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub coin: String,
    /// Signed size (negative = short)
    pub szi: String,
    pub leverage: Leverage,
    pub entry_px: Option<String>,
    pub position_value: String,
    pub unrealized_pnl: String,
    pub liquidation_px: Option<String>,
    #[serde(default)]
    pub margin_used: Option<String>,
    #[serde(default)]
    pub max_trade_szs: Option<(String, String)>,
}

impl Position {
    /// Get signed position size as f64
    pub fn size(&self) -> f64 {
        self.szi.parse().unwrap_or(0.0)
    }

    /// Get entry price as f64
    pub fn entry_price(&self) -> f64 {
        self.entry_px.as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    }

    /// Get liquidation price as f64 (None if no liquidation price)
    pub fn liquidation_price(&self) -> Option<f64> {
        self.liquidation_px.as_ref()
            .and_then(|s| s.parse().ok())
    }

    /// Get unrealized PnL as f64
    pub fn unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl.parse().unwrap_or(0.0)
    }

    /// Get position value as f64
    pub fn position_value(&self) -> f64 {
        self.position_value.parse().unwrap_or(0.0)
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size() > 0.0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size() < 0.0
    }

    /// Check if position is empty
    pub fn is_empty(&self) -> bool {
        self.size().abs() < 1e-10
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Leverage {
    #[serde(rename = "type")]
    pub leverage_type: String,
    pub value: f64,
    #[serde(default)]
    pub raw_usd: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarginSummary {
    pub account_value: String,
    pub total_ntl_pos: String,
    pub total_raw_usd: String,
    pub total_margin_used: String,
}

impl MarginSummary {
    pub fn account_value(&self) -> f64 {
        self.account_value.parse().unwrap_or(0.0)
    }

    pub fn total_position_value(&self) -> f64 {
        self.total_ntl_pos.parse().unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CrossMarginSummary {
    pub account_value: String,
    pub total_ntl_pos: String,
}

/// User fill (trade execution)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserFill {
    pub coin: String,
    pub px: String,
    pub sz: String,
    pub side: String,
    pub time: u64,
    pub start_position: String,
    pub dir: String,
    pub closed_pnl: String,
    pub hash: String,
    pub oid: u64,
    pub crossed: bool,
    pub fee: String,
    pub tid: u64,
}

impl UserFill {
    pub fn price(&self) -> f64 {
        self.px.parse().unwrap_or(0.0)
    }

    pub fn size(&self) -> f64 {
        self.sz.parse().unwrap_or(0.0)
    }

    pub fn is_buy(&self) -> bool {
        self.side == "B"
    }
}

/// Asset metadata
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub universe: Vec<AssetMeta>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetMeta {
    pub name: String,
    pub sz_decimals: u8,
    #[serde(default)]
    pub max_leverage: Option<f64>,
    #[serde(default)]
    pub only_isolated: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_size_parsing() {
        let position = Position {
            coin: "BTC".to_string(),
            szi: "-1.5".to_string(),
            leverage: Leverage {
                leverage_type: "cross".to_string(),
                value: 10.0,
                raw_usd: None,
            },
            entry_px: Some("50000.0".to_string()),
            position_value: "75000.0".to_string(),
            unrealized_pnl: "1000.0".to_string(),
            liquidation_px: Some("45000.0".to_string()),
            margin_used: None,
            max_trade_szs: None,
        };

        assert_eq!(position.size(), -1.5);
        assert!(position.is_short());
        assert!(!position.is_long());
        assert_eq!(position.entry_price(), 50000.0);
        assert_eq!(position.liquidation_price(), Some(45000.0));
    }
}
