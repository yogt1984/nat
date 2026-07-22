# 04 — Wallet Discovery from Trade Stream

## What

Automatically discover whale wallet addresses from the WebSocket trade stream's
`users` field, then add them to `PositionTracker` for polling.

## Why

- `PositionTracker` needs wallet addresses to poll
- Hardcoding addresses is fragile — whales rotate wallets
- `WsTrade.users: Option<(String, String)>` contains maker/taker addresses
- The existing `discover_wallets_from_trades()` in `tracker.rs` (line 318) is a
  blocking reference implementation — we need a non-blocking production version

## Risk

**`WsTrade.users` may not be populated by Hyperliquid's public WebSocket.**
The field exists with `#[serde(default)]` but we don't know if the exchange sends it.

Mitigation:
- Log on first occurrence of `Some` vs `None` to diagnose
- If never populated, fall back to `initial_wallets` in config
- Phase 1 whale flow (trade-size classification) works regardless

## Changes

### 1. Wallet accumulator — new file or section in `positions/state.rs`

```rust
pub struct WalletDiscovery {
    /// Wallet address -> trade count
    counts: RwLock<HashMap<String, usize>>,
    /// Already-added wallets (avoid re-adding)
    added: RwLock<HashSet<String>>,
    max_wallets: usize,
}

impl WalletDiscovery {
    pub fn observe_trade(&self, maker: &str, taker: &str) {
        let mut counts = self.counts.write();
        *counts.entry(maker.to_string()).or_insert(0) += 1;
        *counts.entry(taker.to_string()).or_insert(0) += 1;
    }

    /// Returns top-N wallets not yet added, sorted by trade count
    pub fn top_new_wallets(&self, n: usize) -> Vec<String> {
        let counts = self.counts.read();
        let added = self.added.read();
        let mut candidates: Vec<_> = counts.iter()
            .filter(|(w, _)| !added.contains(*w))
            .collect();
        candidates.sort_by(|a, b| b.1.cmp(a.1));
        candidates.into_iter().take(n).map(|(w, _)| w.clone()).collect()
    }

    pub fn mark_added(&self, wallets: &[String]) {
        let mut added = self.added.write();
        for w in wallets { added.insert(w.clone()); }
    }
}
```

### 2. Feed trades — `rust/ing/src/main.rs` or `state/mod.rs`

In the trade processing path, when `discover_from_trades = true`:

```rust
if let Some((maker, taker)) = &trade.users {
    wallet_discovery.observe_trade(maker, taker);
}
```

### 3. Periodic promotion — spawned task in `main.rs`

Every 5 minutes:
```rust
let new_wallets = discovery.top_new_wallets(5);
if !new_wallets.is_empty() {
    tracker.add_wallets(&new_wallets);
    discovery.mark_added(&new_wallets);
    info!(count = new_wallets.len(), "Discovered new whale wallets");
}
```

Cap total tracked wallets at `max_tracked_wallets`.

### 4. Diagnostics

Log on startup whether `users` field is being received:
```rust
static LOGGED_USERS_STATUS: AtomicBool = AtomicBool::new(false);
// In trade handler:
if !LOGGED_USERS_STATUS.swap(true, Ordering::Relaxed) {
    if trade.users.is_some() {
        info!("WsTrade.users field IS populated — wallet discovery active");
    } else {
        warn!("WsTrade.users field is None — wallet discovery unavailable, using initial_wallets only");
    }
}
```

## Verify

```bash
cd rust && cargo test -p ing
# Run ingestor, check logs for users field status
# If populated: verify wallet count grows over time
# If not populated: verify fallback to initial_wallets works
```

## Fallback

If `users` is never available, the user must manually curate `initial_wallets`
in `config/ing.toml`. Known whale addresses can be sourced from:
- Hyperliquid leaderboard (manual)
- On-chain explorers
- Historical trade analysis
