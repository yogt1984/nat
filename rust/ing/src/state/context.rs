//! Market context (funding, open interest, etc.)

use crate::ws::WsAssetCtx;
use crate::state::RingBuffer;

/// Market context state
#[derive(Debug)]
pub struct MarketContext {
    /// Current funding rate
    funding_rate: f64,
    /// Historical funding rates for z-score calculation
    funding_history: RingBuffer<f64>,
    /// Current open interest
    open_interest: f64,
    /// Historical OI for change calculation
    oi_history: RingBuffer<f64>,
    /// Oracle price
    oracle_price: f64,
    /// Mark price
    mark_price: f64,
    /// 24h volume
    volume_24h: f64,
    /// Historical volume samples
    volume_history: RingBuffer<f64>,
    /// Premium (mark - oracle)
    premium: f64,
    /// Last update timestamp
    _last_update: u64,
    /// Whether context has been initialized
    initialized: bool,
}

impl MarketContext {
    /// Create a new market context
    pub fn new() -> Self {
        Self {
            funding_rate: 0.0,
            funding_history: RingBuffer::new(288),  // 24h at 5-min intervals
            open_interest: 0.0,
            oi_history: RingBuffer::new(60),  // 5 minutes of samples
            oracle_price: 0.0,
            mark_price: 0.0,
            volume_24h: 0.0,
            volume_history: RingBuffer::new(288),
            premium: 0.0,
            _last_update: 0,
            initialized: false,
        }
    }

    /// Update from WebSocket message
    pub fn update(&mut self, ctx: &WsAssetCtx) {
        self.funding_rate = ctx.ctx.funding_rate();
        self.funding_history.push(self.funding_rate);

        self.open_interest = ctx.ctx.open_interest();
        self.oi_history.push(self.open_interest);

        self.oracle_price = ctx.ctx.oracle_price();
        self.mark_price = ctx.ctx.mark_price();
        self.volume_24h = ctx.ctx.volume_24h();
        self.volume_history.push(self.volume_24h);

        // Calculate premium in basis points
        if self.oracle_price > 0.0 {
            self.premium = (self.mark_price - self.oracle_price) / self.oracle_price * 10000.0;
        }

        self.initialized = true;
    }

    /// Get current funding rate
    pub fn funding_rate(&self) -> f64 {
        self.funding_rate
    }

    /// Get funding rate z-score relative to history
    pub fn funding_zscore(&self) -> f64 {
        if self.funding_history.len() < 2 {
            return 0.0;
        }

        let mean = self.funding_history.mean();
        let std = self.funding_history.std();

        if std > 0.0 {
            (self.funding_rate - mean) / std
        } else {
            0.0
        }
    }

    /// Get current open interest
    pub fn open_interest(&self) -> f64 {
        self.open_interest
    }

    /// Get OI change over last N samples
    pub fn oi_change(&self, samples: usize) -> f64 {
        if self.oi_history.len() < samples + 1 {
            return 0.0;
        }

        let current = self.open_interest;
        let past = self.oi_history.get(self.oi_history.len().saturating_sub(samples))
            .copied()
            .unwrap_or(current);

        current - past
    }

    /// Get OI percent change over last N samples
    pub fn oi_change_pct(&self, samples: usize) -> f64 {
        if self.oi_history.len() < samples + 1 {
            return 0.0;
        }

        let current = self.open_interest;
        let past = self.oi_history.get(self.oi_history.len().saturating_sub(samples))
            .copied()
            .unwrap_or(current);

        if past > 0.0 {
            (current - past) / past * 100.0
        } else {
            0.0
        }
    }

    /// Get oracle price
    pub fn oracle_price(&self) -> f64 {
        self.oracle_price
    }

    /// Get mark price
    pub fn mark_price(&self) -> f64 {
        self.mark_price
    }

    /// Get premium in basis points
    pub fn premium_bps(&self) -> f64 {
        self.premium
    }

    /// Get 24h volume
    pub fn volume_24h(&self) -> f64 {
        self.volume_24h
    }

    /// Get volume ratio (current vs average)
    pub fn volume_ratio(&self) -> f64 {
        if self.volume_history.len() < 2 {
            return 1.0;
        }

        let mean = self.volume_history.mean();
        if mean > 0.0 {
            self.volume_24h / mean
        } else {
            1.0
        }
    }

    /// Get mark-oracle divergence persistence
    /// Returns how long the premium has been in the same direction
    pub fn mark_oracle_divergence(&self) -> f64 {
        // Simplified: just return current premium
        // Could be extended to track persistence over time
        self.premium.abs()
    }

    /// Get funding rate momentum: current - 8h ago (Hyperliquid 8h settlement cycle)
    /// 8h = 96 samples at 5-min intervals
    pub fn funding_momentum_8h(&self) -> f64 {
        let lookback = 96; // 8h / 5min
        if self.funding_history.len() < lookback + 1 {
            return 0.0;
        }
        let past = self.funding_history.get(
            self.funding_history.len().saturating_sub(lookback)
        ).copied().unwrap_or(self.funding_rate);
        self.funding_rate - past
    }

    /// Get funding acceleration: momentum change (finite difference of momentum)
    /// Approximated as current_zscore - previous_zscore_equivalent
    pub fn funding_acceleration(&self) -> f64 {
        let lookback_current = 96;
        let lookback_prev = 108; // 9h ago — gives 1h difference in momentum
        if self.funding_history.len() < lookback_prev + 1 {
            return 0.0;
        }
        let current = self.funding_rate;
        let past_8h = self.funding_history.get(
            self.funding_history.len().saturating_sub(lookback_current)
        ).copied().unwrap_or(current);
        let past_9h = self.funding_history.get(
            self.funding_history.len().saturating_sub(lookback_prev)
        ).copied().unwrap_or(current);

        // funding_1h_ago (approximated from 9h-ago minus 8h-ago momentum)
        let prev_funding = self.funding_history.get(
            self.funding_history.len().saturating_sub(12)  // 1h ago
        ).copied().unwrap_or(current);

        let momentum_now = current - past_8h;
        let momentum_prev = prev_funding - past_9h;
        momentum_now - momentum_prev
    }

    /// Get OI momentum over 1 hour (~720 samples at 5s updates, capped to buffer)
    pub fn oi_momentum_1h(&self) -> f64 {
        // oi_history has 60 samples (~5 min). Use full buffer for best approximation.
        let n = self.oi_history.len();
        if n < 2 {
            return 0.0;
        }
        let current = self.open_interest;
        let oldest = self.oi_history.first().copied().unwrap_or(current);
        if oldest > 0.0 {
            (current - oldest) / oldest * 100.0
        } else {
            0.0
        }
    }

    /// Check if context has been initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for MarketContext {
    fn default() -> Self {
        Self::new()
    }
}
