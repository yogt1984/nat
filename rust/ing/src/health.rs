//! Per-channel health tracking for WebSocket data streams.
//!
//! Extracted from main.rs to enable unit testing. Tracks book/trade
//! channel liveness and midprice freeze detection independently.
//!
//! Background: on 2026-05-06 ~08:25 UTC, the orderbook stream silently
//! froze while trade ticks kept flowing. Generic WS staleness missed it
//! because the connection was alive. These checks would have caught it
//! within minutes.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

pub const BOOK_STALE_WARN_SECS: u64 = 60;
pub const BOOK_STALE_ERROR_SECS: u64 = 300;
pub const TRADE_STALE_WARN_SECS: u64 = 120;

// Midprice-frozen detection is two-tiered:
//
// (a) Compound freeze: book channel also stale (May-6 signature) — fire fast.
// (b) Pure freeze: book alive but L1 hasn't moved. Empirically 70–84s
//     happens naturally on BTC/ETH at low-vol moments. Use higher threshold.
//
// BOOK_STALE_GATE_SECS: book_age above which we treat it as compound.
pub const PRICE_FROZEN_WITH_BOOK_STALE_WARN_SECS: u64 = 60;
pub const PRICE_FROZEN_WITH_BOOK_STALE_ERROR_SECS: u64 = 300;
pub const PRICE_FROZEN_BOOK_ALIVE_WARN_SECS: u64 = 300;
pub const PRICE_FROZEN_BOOK_ALIVE_ERROR_SECS: u64 = 900;
pub const BOOK_STALE_GATE_SECS: u64 = 30;

// ---------------------------------------------------------------------------
// Health status
// ---------------------------------------------------------------------------

/// Severity level for a health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Warn,
    Error,
}

/// Result of a health check tick.
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub book: ChannelStatus,
    pub trade: ChannelStatus,
    pub price_frozen: PriceFrozenStatus,
}

#[derive(Debug, Clone)]
pub struct ChannelStatus {
    pub severity: Severity,
    pub age_secs: Option<u64>,
    pub msg_count: u64,
}

#[derive(Debug, Clone)]
pub struct PriceFrozenStatus {
    pub severity: Severity,
    pub age_secs: Option<u64>,
    pub last_midprice: Option<f64>,
    /// "book_stale" or "book_alive" — determines which thresholds apply
    pub regime: &'static str,
}

// ---------------------------------------------------------------------------
// ChannelHealth
// ---------------------------------------------------------------------------

/// Per-symbol channel health tracker.
///
/// Tracks book/trade message timestamps and midprice changes independently
/// so that silent single-channel failures are detected.
pub struct ChannelHealth {
    pub last_book_msg_at: Option<Instant>,
    pub last_trade_msg_at: Option<Instant>,
    pub book_msg_count: u64,
    pub trade_msg_count: u64,
    pub last_midprice: Option<f64>,
    pub last_midprice_change_at: Option<Instant>,
    book_stale_error_logged: bool,
    price_frozen_error_logged: bool,
}

impl ChannelHealth {
    pub fn new() -> Self {
        Self {
            last_book_msg_at: None,
            last_trade_msg_at: None,
            book_msg_count: 0,
            trade_msg_count: 0,
            last_midprice: None,
            last_midprice_change_at: None,
            book_stale_error_logged: false,
            price_frozen_error_logged: false,
        }
    }

    /// Record a book message received.
    pub fn on_book(&mut self) {
        self.last_book_msg_at = Some(Instant::now());
        self.book_msg_count += 1;
    }

    /// Record a trade message received.
    pub fn on_trade(&mut self) {
        self.last_trade_msg_at = Some(Instant::now());
        self.trade_msg_count += 1;
    }

    /// Record a midprice observation. Resets freeze timer on change.
    pub fn on_midprice(&mut self, price: f64) {
        if !price.is_finite() {
            return;
        }
        let changed = match self.last_midprice {
            Some(prev) => prev != price,
            None => true,
        };
        if changed {
            self.last_midprice = Some(price);
            self.last_midprice_change_at = Some(Instant::now());
        }
    }

    /// Seconds since last book message, or None if never received.
    pub fn book_age_secs(&self) -> Option<u64> {
        self.last_book_msg_at
            .map(|t| t.elapsed().as_secs())
    }

    /// Seconds since last trade message, or None if never received.
    pub fn trade_age_secs(&self) -> Option<u64> {
        self.last_trade_msg_at
            .map(|t| t.elapsed().as_secs())
    }

    /// Seconds since last midprice change, or None if never changed.
    pub fn price_age_secs(&self) -> Option<u64> {
        self.last_midprice_change_at
            .map(|t| t.elapsed().as_secs())
    }

    /// Run a health check and return status. Also manages error-logged latches
    /// so that ERROR severity fires once per stuck episode.
    pub fn check_health(&mut self, uptime_secs: u64) -> HealthStatus {
        let book_age = self.book_age_secs();
        let trade_age = self.trade_age_secs();
        let price_age = self.price_age_secs();

        // --- Book channel ---
        let book_severity = match book_age {
            Some(age) if age >= BOOK_STALE_ERROR_SECS => {
                if !self.book_stale_error_logged {
                    self.book_stale_error_logged = true;
                }
                Severity::Error
            }
            Some(age) if age >= BOOK_STALE_WARN_SECS => Severity::Warn,
            Some(_) => {
                self.book_stale_error_logged = false;
                Severity::Ok
            }
            None if uptime_secs >= 60 => Severity::Warn,
            None => Severity::Ok,
        };

        // --- Trade channel ---
        let trade_severity = match trade_age {
            Some(age) if age >= TRADE_STALE_WARN_SECS => Severity::Warn,
            _ => Severity::Ok,
        };

        // --- Price frozen ---
        let book_stale = book_age.is_some_and(|b| b >= BOOK_STALE_GATE_SECS);
        let (price_severity, regime) = match price_age {
            Some(age) => {
                let (warn_thresh, error_thresh, regime) = if book_stale {
                    (
                        PRICE_FROZEN_WITH_BOOK_STALE_WARN_SECS,
                        PRICE_FROZEN_WITH_BOOK_STALE_ERROR_SECS,
                        "book_stale",
                    )
                } else {
                    (
                        PRICE_FROZEN_BOOK_ALIVE_WARN_SECS,
                        PRICE_FROZEN_BOOK_ALIVE_ERROR_SECS,
                        "book_alive",
                    )
                };

                if age >= error_thresh {
                    if !self.price_frozen_error_logged {
                        self.price_frozen_error_logged = true;
                    }
                    (Severity::Error, regime)
                } else if age >= warn_thresh {
                    (Severity::Warn, regime)
                } else {
                    self.price_frozen_error_logged = false;
                    (Severity::Ok, regime)
                }
            }
            None => {
                self.price_frozen_error_logged = false;
                (Severity::Ok, "book_alive")
            }
        };

        HealthStatus {
            book: ChannelStatus {
                severity: book_severity,
                age_secs: book_age,
                msg_count: self.book_msg_count,
            },
            trade: ChannelStatus {
                severity: trade_severity,
                age_secs: trade_age,
                msg_count: self.trade_msg_count,
            },
            price_frozen: PriceFrozenStatus {
                severity: price_severity,
                age_secs: price_age,
                last_midprice: self.last_midprice,
                regime,
            },
        }
    }

    /// Whether the book-stale error has already been logged (latch).
    pub fn book_stale_error_logged(&self) -> bool {
        self.book_stale_error_logged
    }

    /// Whether the price-frozen error has already been logged (latch).
    pub fn price_frozen_error_logged(&self) -> bool {
        self.price_frozen_error_logged
    }
}

impl Default for ChannelHealth {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_initial_state() {
        let h = ChannelHealth::new();
        assert!(h.last_book_msg_at.is_none());
        assert!(h.last_trade_msg_at.is_none());
        assert_eq!(h.book_msg_count, 0);
        assert_eq!(h.trade_msg_count, 0);
        assert!(h.last_midprice.is_none());
        assert!(h.last_midprice_change_at.is_none());
        assert!(!h.book_stale_error_logged());
        assert!(!h.price_frozen_error_logged());
    }

    #[test]
    fn test_on_book_updates_timestamp_and_count() {
        let mut h = ChannelHealth::new();
        h.on_book();
        assert!(h.last_book_msg_at.is_some());
        assert_eq!(h.book_msg_count, 1);
        h.on_book();
        assert_eq!(h.book_msg_count, 2);
    }

    #[test]
    fn test_on_trade_updates_timestamp_and_count() {
        let mut h = ChannelHealth::new();
        h.on_trade();
        assert!(h.last_trade_msg_at.is_some());
        assert_eq!(h.trade_msg_count, 1);
    }

    #[test]
    fn test_on_midprice_tracks_changes() {
        let mut h = ChannelHealth::new();

        // First observation: always a change
        h.on_midprice(50000.0);
        assert_eq!(h.last_midprice, Some(50000.0));
        assert!(h.last_midprice_change_at.is_some());
        let first_change = h.last_midprice_change_at;

        // Same price: no change
        h.on_midprice(50000.0);
        assert_eq!(h.last_midprice, Some(50000.0));
        assert_eq!(h.last_midprice_change_at, first_change);

        // Different price: resets timer
        h.on_midprice(50001.0);
        assert_eq!(h.last_midprice, Some(50001.0));
        // Can't easily assert time changed (too fast), but the field was updated
    }

    #[test]
    fn test_on_midprice_ignores_nan() {
        let mut h = ChannelHealth::new();
        h.on_midprice(f64::NAN);
        assert!(h.last_midprice.is_none());
        assert!(h.last_midprice_change_at.is_none());
    }

    #[test]
    fn test_on_midprice_ignores_inf() {
        let mut h = ChannelHealth::new();
        h.on_midprice(f64::INFINITY);
        assert!(h.last_midprice.is_none());
    }

    #[test]
    fn test_book_age_none_when_no_messages() {
        let h = ChannelHealth::new();
        assert!(h.book_age_secs().is_none());
    }

    #[test]
    fn test_book_age_zero_when_just_received() {
        let mut h = ChannelHealth::new();
        h.on_book();
        // Just received — age should be 0
        assert_eq!(h.book_age_secs(), Some(0));
    }

    #[test]
    fn test_book_stale_after_silence() {
        let mut h = ChannelHealth::new();
        // Simulate message received 90s ago
        h.last_book_msg_at = Some(Instant::now() - Duration::from_secs(90));
        h.book_msg_count = 1;

        let status = h.check_health(120);
        assert_eq!(status.book.severity, Severity::Warn);
        assert!(status.book.age_secs.unwrap() >= 89); // allow 1s tolerance
    }

    #[test]
    fn test_book_stale_error_after_long_silence() {
        let mut h = ChannelHealth::new();
        h.last_book_msg_at = Some(Instant::now() - Duration::from_secs(350));
        h.book_msg_count = 1;

        let status = h.check_health(400);
        assert_eq!(status.book.severity, Severity::Error);
        assert!(h.book_stale_error_logged());
    }

    #[test]
    fn test_book_stale_warns_when_no_messages_after_uptime() {
        let h = &mut ChannelHealth::new();
        // No book messages at all, uptime 120s → should warn
        let status = h.check_health(120);
        assert_eq!(status.book.severity, Severity::Warn);
    }

    #[test]
    fn test_book_ok_when_no_messages_early_uptime() {
        let h = &mut ChannelHealth::new();
        // No book messages, but only 10s uptime → ok (startup grace)
        let status = h.check_health(10);
        assert_eq!(status.book.severity, Severity::Ok);
    }

    #[test]
    fn test_trade_stale_after_silence() {
        let mut h = ChannelHealth::new();
        h.last_trade_msg_at = Some(Instant::now() - Duration::from_secs(150));
        h.trade_msg_count = 1;

        let status = h.check_health(200);
        assert_eq!(status.trade.severity, Severity::Warn);
    }

    #[test]
    fn test_trade_ok_when_recent() {
        let mut h = ChannelHealth::new();
        h.on_trade();

        let status = h.check_health(60);
        assert_eq!(status.trade.severity, Severity::Ok);
    }

    #[test]
    fn test_book_alive_trade_stale_divergence() {
        let mut h = ChannelHealth::new();
        // Book just updated, trades silent for 150s
        h.on_book();
        h.last_trade_msg_at = Some(Instant::now() - Duration::from_secs(150));
        h.trade_msg_count = 5;

        let status = h.check_health(200);
        assert_eq!(status.book.severity, Severity::Ok);
        assert_eq!(status.trade.severity, Severity::Warn);
    }

    #[test]
    fn test_midprice_frozen_with_book_stale_tight_threshold() {
        let mut h = ChannelHealth::new();
        // Book stale (60s), price frozen (90s) — compound: warn at 60s
        h.last_book_msg_at = Some(Instant::now() - Duration::from_secs(60));
        h.last_midprice_change_at = Some(Instant::now() - Duration::from_secs(90));
        h.last_midprice = Some(50000.0);
        h.book_msg_count = 1;

        let status = h.check_health(120);
        assert_eq!(status.price_frozen.regime, "book_stale");
        assert_eq!(status.price_frozen.severity, Severity::Warn);
    }

    #[test]
    fn test_midprice_frozen_book_alive_lenient_threshold() {
        let mut h = ChannelHealth::new();
        // Book alive (just received), price frozen for 90s — pure: ok (threshold 300s)
        h.on_book();
        h.last_midprice_change_at = Some(Instant::now() - Duration::from_secs(90));
        h.last_midprice = Some(50000.0);

        let status = h.check_health(120);
        assert_eq!(status.price_frozen.regime, "book_alive");
        assert_eq!(status.price_frozen.severity, Severity::Ok);
    }

    #[test]
    fn test_midprice_frozen_book_alive_warns_at_300s() {
        let mut h = ChannelHealth::new();
        h.on_book();
        h.last_midprice_change_at = Some(Instant::now() - Duration::from_secs(350));
        h.last_midprice = Some(50000.0);

        let status = h.check_health(400);
        assert_eq!(status.price_frozen.regime, "book_alive");
        assert_eq!(status.price_frozen.severity, Severity::Warn);
    }

    #[test]
    fn test_error_logged_latch_fires_once() {
        let mut h = ChannelHealth::new();
        h.last_book_msg_at = Some(Instant::now() - Duration::from_secs(350));
        h.book_msg_count = 1;

        // First check: error, latch set
        let status = h.check_health(400);
        assert_eq!(status.book.severity, Severity::Error);
        assert!(h.book_stale_error_logged());

        // Latch stays set on subsequent checks (same situation)
        let _status2 = h.check_health(460);
        assert!(h.book_stale_error_logged());
    }

    #[test]
    fn test_error_latch_clears_on_recovery() {
        let mut h = ChannelHealth::new();
        h.last_book_msg_at = Some(Instant::now() - Duration::from_secs(350));
        h.book_msg_count = 1;

        // Trigger error latch
        h.check_health(400);
        assert!(h.book_stale_error_logged());

        // Recovery: fresh book message
        h.on_book();
        let status = h.check_health(400);
        assert_eq!(status.book.severity, Severity::Ok);
        assert!(!h.book_stale_error_logged());
    }

    #[test]
    fn test_all_ok_when_healthy() {
        let mut h = ChannelHealth::new();
        h.on_book();
        h.on_trade();
        h.on_midprice(50000.0);

        let status = h.check_health(60);
        assert_eq!(status.book.severity, Severity::Ok);
        assert_eq!(status.trade.severity, Severity::Ok);
        assert_eq!(status.price_frozen.severity, Severity::Ok);
    }
}
