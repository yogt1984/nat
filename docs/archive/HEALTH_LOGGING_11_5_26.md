# Health Logging & Calibration — 2026-05-11

Diagnostic notes from the silent-corruption investigation and the
ingestor-health patches that followed.

## What we found

The walk-forward audit on 2026-05-10 exposed silent data corruption:
**from 2026-05-06 08:25 UTC onward, 69% of collected bars had frozen
prices** on all three symbols simultaneously. 283 of 551 float feature
columns went constant. The artifact was masked by the legacy single-split
forward-test, which reported "OOS IC = 0 for every feature" — a methodological
collapse, not a feature problem.

```
BTC : last price change 2026-05-06 08:25:00  → frozen at 81,529.50 for 1,754 bars
ETH : last price change 2026-05-06 08:25:00  → frozen at  2,374.65 for 1,754 bars
SOL : last price change 2026-05-06 08:24:00  → frozen at     87.91 for 1,755 bars
```

All three within one minute → shared-resource failure (WebSocket connection
or shared dispatch loop), not per-symbol orderbook bug.

## Root cause

Commit `d975d50` (2026-05-06 08:17 UTC) added WebSocket ping/pong keepalive
and stale-connection detection — `client.is_stale()`, `client.send_ping()`,
auto-reconnect on staleness.

The freeze happened at 08:25 UTC, **8 minutes after the commit landed**.
The running binary on su-35 was still pre-`d975d50`. No keepalive, no
stale detection, no reconnect. Ingestor sat idle for 4 days writing stale
prices into parquet.

This was a **deployment lag**, not an open bug. The fix already existed in
the codebase.

## What we added

| commit | what | why |
|---|---|---|
| `67b563d` | walk-forward k-fold in `scalping_profiler.py` | the legacy 70/30 split produced degenerate OOS metrics; walk-forward gives per-feature OOS-IC distribution and sign-consistency. This is the diagnostic tool that surfaced the corruption in the first place. |
| `5d9a6e4` | per-channel health logging in `rust/ing/src/main.rs` | existing message-count health check missed the May-6 mode entirely (trade ticks kept flowing while book channel was dead). New logging tracks book/trade msg ages and midprice-change age independently. |
| `0a321ca` | two-tier MIDPRICE FROZEN gated on book health | 60s threshold tripped ~5x/hour on benign quiet inside-quote periods; calibrated against observed distribution. |

## Calibrated thresholds (live)

```rust
// book channel alive (book_age < 30s): benign quiet period — generous
const PRICE_FROZEN_BOOK_ALIVE_WARN_SECS:  u64 = 300;
const PRICE_FROZEN_BOOK_ALIVE_ERROR_SECS: u64 = 900;

// book channel stale: the May-6 signature — keep tight
const PRICE_FROZEN_WITH_BOOK_STALE_WARN_SECS:  u64 = 60;
const PRICE_FROZEN_WITH_BOOK_STALE_ERROR_SECS: u64 = 300;
const BOOK_STALE_GATE_SECS: u64 = 30;
```

Health log lines now carry `regime=book_alive | book_stale` so the two
modes are unambiguous.

## Live validation

Two 1-hour watches on the patched binary:

**Watch 1 (2026-05-10 17:30 → 18:30 UTC):**
- 5 MIDPRICE FROZEN events, all `book_silence_s=0` (book channel alive)
- Durations 70–84s — normal market microstructure for liquid majors at
  low-volatility moments
- This confirmed the 60s threshold was too tight → calibration

**Watch 2 (2026-05-11 04:31 → 05:31 UTC):**
- **Caught the May-6 failure mode live at 04:31:52 UTC.** All three symbols
  hit `BOOK STREAM STALE` + `MIDPRICE FROZEN regime=book_stale` simultaneously
  (book_silence_s=94 across all). Existing `client.is_stale()` forced
  reconnect, all three resubscribed in ~3 seconds. No data loss beyond the
  ~94 s freeze itself.
- 0 false-positive `book_alive` WARNs (calibration verified)
- `price_change_age_s` distribution over the watch: p50=4s, p90=27s,
  p95=42s, p99=95s, max=96s. New 300 s threshold sits ~7× above noise.
- Observed reconnect rate: ~1 per 22 min mixed single-symbol and
  synchronized; each ~3 s recovery.

## Data hygiene going forward

- **Quarantine pre-2026-05-10 21:32 UTC bars.** They include the 4-day
  freeze residue. For now, filter by `(raw_midprice_close != lag(raw_midprice_close))`
  in any analysis touching that range, or just exclude entirely.
- **Mask reconnect spans in healthy data.** Each freeze incident contributes
  up to ~94 s of stale-feature bars. Join feature bars against the
  `BOOK STREAM STALE` / `Connection stale` log timestamps to drop those
  ranges, or filter rows where `price_change_age_s >= 60` once we log
  that column to the parquet (not currently emitted to disk).

## Open items

- Audit thread (walk-forward sweep BTC/ETH/SOL × {1m, 2m, 5m}) was blocked
  on this; now unblocked. Bank ~5–7 days of post-2026-05-10-21:32 data
  before running it.
- Hyperliquid disconnect rate (~1/22min observed) is worth tracking over
  the week — if it climbs, that's a network/upstream signal rather than
  a code fix.
- Auto-reconnect-on-`BOOK STREAM DEAD` is intentionally not implemented;
  current `client.is_stale()` already handles the same trigger at 90 s,
  which is faster than our 300 s ERROR threshold. The new logging is
  visibility, not recovery.

## Files touched

```
config/pipeline.toml              (walk-forward config keys)
rust/ing/src/main.rs              (health logging + calibration)
scripts/scalping_profiler.py      (walk-forward k-fold)
```

## Tmux session for the running ingestor

```bash
tmux a -t ing                                      # attach
tail -F logs/ingestor_20260510_213245.log          # passive watch
```
