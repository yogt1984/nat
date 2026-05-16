//! Order Book Resilience Feature Extraction
//!
//! Measures the order book's ability to recover depth after large trades.
//! Resilient books refill quickly; fragile books stay depleted.
//!
//! # Features (3 total)
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Recovery time 50%** | Ticks to recover 50% of depth after a large take | [0, +inf) | Lower = more resilient |
//! | **Depth impact ratio** | Max depth drop / pre-take depth | [0, 1] | Higher = larger impact |
//! | **Recovery speed** | Depth recovery per tick (normalized) | [0, +inf) | Higher = faster refill |
//!
//! # Algorithm
//!
//! Track total L5 depth. When depth drops by >20% in one tick (large take),
//! record pre-take depth and measure how quickly it recovers to 50%.
//!
//! # References
//!
//! - Biais, Hillion & Spatt (1995) — An Empirical Analysis of the Limit Order Book

use std::collections::VecDeque;

/// Tracks depth recovery events for resilience measurement
#[derive(Debug, Clone)]
pub struct ResilienceTracker {
    /// Pre-take depth levels for active recovery events
    events: VecDeque<ResilienceEvent>,
    /// Previous tick's total depth (L5 bid + ask)
    prev_depth: f64,
    /// Maximum events to track simultaneously
    max_events: usize,
}

#[derive(Debug, Clone)]
struct ResilienceEvent {
    /// Depth before the large take
    pre_take_depth: f64,
    /// Ticks elapsed since the take
    ticks_elapsed: u32,
    /// Whether 50% recovery has been reached
    recovered: bool,
}

impl ResilienceTracker {
    pub fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(16),
            prev_depth: 0.0,
            max_events: 10,
        }
    }

    /// Update tracker with current depth and return resilience features
    pub fn update(&mut self, current_depth: f64) -> ResilienceFeatures {
        let depth_change = if self.prev_depth > 0.0 {
            (current_depth - self.prev_depth) / self.prev_depth
        } else {
            0.0
        };

        // Detect large take: depth drops by >20%
        if depth_change < -0.20 && self.prev_depth > 0.0 {
            if self.events.len() >= self.max_events {
                self.events.pop_front();
            }
            self.events.push_back(ResilienceEvent {
                pre_take_depth: self.prev_depth,
                ticks_elapsed: 0,
                recovered: false,
            });
        }

        // Update all active events
        let mut total_recovery_time = 0.0;
        let mut total_impact = 0.0;
        let mut total_speed = 0.0;
        let mut n_events = 0u32;

        for event in self.events.iter_mut() {
            event.ticks_elapsed += 1;

            let depth_ratio = current_depth / event.pre_take_depth;
            if depth_ratio >= 0.5 && !event.recovered {
                event.recovered = true;
            }

            if event.ticks_elapsed <= 100 {
                // Only count recent events (last 10s at 100ms)
                let impact = 1.0 - (current_depth / event.pre_take_depth).min(1.0);
                let speed = if event.ticks_elapsed > 0 {
                    depth_ratio / event.ticks_elapsed as f64
                } else {
                    0.0
                };
                let recovery_time = if event.recovered {
                    event.ticks_elapsed as f64
                } else {
                    100.0 // cap at 100 ticks if not yet recovered
                };

                total_recovery_time += recovery_time;
                total_impact += impact;
                total_speed += speed;
                n_events += 1;
            }
        }

        // Evict old events
        self.events.retain(|e| e.ticks_elapsed <= 200);

        self.prev_depth = current_depth;

        if n_events > 0 {
            let n = n_events as f64;
            ResilienceFeatures {
                recovery_time_50: total_recovery_time / n,
                depth_impact_ratio: total_impact / n,
                recovery_speed: total_speed / n,
            }
        } else {
            ResilienceFeatures::default()
        }
    }
}

impl Default for ResilienceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Order book resilience features (3 features)
#[derive(Debug, Clone, Default)]
pub struct ResilienceFeatures {
    /// Average ticks to 50% depth recovery after large takes
    pub recovery_time_50: f64,
    /// Average depth impact ratio from large takes
    pub depth_impact_ratio: f64,
    /// Average depth recovery speed (depth_ratio / ticks)
    pub recovery_speed: f64,
}

impl ResilienceFeatures {
    pub fn count() -> usize { 3 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "resilience_recovery_time_50",
            "resilience_depth_impact_ratio",
            "resilience_recovery_speed",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.recovery_time_50,
            self.depth_impact_ratio,
            self.recovery_speed,
        ]
    }
}
