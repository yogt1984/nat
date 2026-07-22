# CURRENT ISSUE: Ingestor produces 0-byte parquet files on su-35

**Date:** 2026-04-19
**Machine:** su-35 (second computer)
**Status:** OPEN — data collection not working on this machine

## Problem

The ingestor connects to Hyperliquid WebSocket successfully for all 3 symbols (BTC, ETH, SOL), opens a parquet file, but the file remains 0 bytes indefinitely. After 11+ minutes in release mode, no data is flushed.

## Evidence

### What works (dev machine — onat's primary workstation)
- WebSocket connects, data flows, features are computed
- Debug logging confirmed: `Feature computed, sending to writer symbol=ETH seq=1`
- Parquet files fill up: `20260419_185502.parquet` = 8,914 rows (2.7MB)
- Buffer flush happens at ~5.5 minutes (10,000 rows at ~30 rows/sec)

### What fails (su-35)
- WebSocket connects successfully (same logs as dev machine)
- Parquet file opens at correct path (`../data/features/2026-04-19/`)
- File stays 0 bytes after 11+ minutes in release mode, 20+ minutes in debug mode
- No error messages, no warnings after "Opening new Parquet file"
- Process is alive, consuming CPU (20.7% in debug, active in release)
- Multiple attempts across the day, all same result
- Both debug and release builds tested

### Log output (su-35, release mode)
```
2026-04-19T20:42:59.194081Z  INFO ThreadId(09) ing::ws::client: All subscriptions complete symbol=BTC
2026-04-19T20:42:59.194051Z  INFO ThreadId(07) ing::ws::client: All subscriptions complete symbol=ETH
2026-04-19T20:42:59.194118Z  INFO ThreadId(09) ing: WebSocket connected successfully symbol=BTC
2026-04-19T20:42:59.194121Z  INFO ThreadId(02) ing: WebSocket connected successfully symbol=SOL
2026-04-19T20:42:59.194127Z  INFO ThreadId(07) ing: WebSocket connected successfully symbol=ETH
2026-04-19T20:42:59.596091Z  INFO ThreadId(07) ing::output::writer: Opening new Parquet file path="../data/features/2026-04-19/20260419_204259.parquet"
```
Then complete silence. No further log output.

### File listing (su-35, 22:53 local time)
```
-rw-rw-r-- 1 onat onat 2802613 Apr 19 21:09 20260419_185502.parquet  # copied from dev machine
-rw-rw-r-- 1 onat onat  304749 Apr 19 21:09 20260419_185925.parquet  # copied from dev machine
-rw-rw-r-- 1 onat onat       0 Apr 19 22:13 20260419_201310.parquet
-rw-rw-r-- 1 onat onat       0 Apr 19 22:13 20260419_201339.parquet
-rw-rw-r-- 1 onat onat       0 Apr 19 22:14 20260419_201404.parquet
-rw-rw-r-- 1 onat onat       0 Apr 19 22:17 20260419_201756.parquet
-rw-rw-r-- 1 onat onat       0 Apr 19 22:42 20260419_204259.parquet  # current run, 11 min old
```

## Root Cause Hypotheses

### H1: WebSocket connected but no data messages arriving (MOST LIKELY)
- The WebSocket handshake succeeds (TCP + TLS + HTTP upgrade to WS)
- Subscription requests are sent and acknowledged
- But Hyperliquid may not be sending actual l2Book/trades data
- Network/firewall on su-35 may allow the handshake but throttle/block streaming data
- On dev machine, debug logs confirmed messages arrive: `stream.next() returned a message` and `Parsed: Book coin=ETH`
- This was NOT tested on su-35 with debug logging

### H2: Messages arrive but parse to Unknown/None
- If Hyperliquid changed their API response format, `parse_ws_message()` could return `None`
- This would mean `state.update()` is never called with a `Book` message
- `initialized` flag stays `false`, `compute_features()` returns `None`
- No features sent to writer, buffer stays empty

### H3: tokio::select! biased starvation
- The main loop uses `biased` select, always checking WebSocket first
- If WebSocket messages arrive fast enough, the emission ticker branch never runs
- But this would mean data IS arriving (contradicting H1)

### H4: DNS or proxy difference on su-35
- `wss://api.hyperliquid.xyz/ws` may resolve differently on su-35
- CloudFront CDN edge may behave differently for that IP/region

## Diagnostic Steps to Run on su-35

### Step 1: Verify WebSocket data flow with debug logging
```bash
cd rust
RUST_LOG=debug timeout 20 ./target/release/ing ../config/ing.toml 2>&1 | \
  grep -E "stream\.next|Received|Parsed:|Emission tick|Feature computed|initialized"
```
Expected output if working: lines like `Parsed: Book coin=BTC`, `Feature computed`
Expected output if broken: only `Emission tick ... initialized=false` or nothing

### Step 2: Test raw WebSocket with Python
```bash
python3 -c "
import asyncio, json, websockets
async def test():
    async with websockets.connect('wss://api.hyperliquid.xyz/ws') as ws:
        await ws.send(json.dumps({'method':'subscribe','subscription':{'type':'trades','coin':'BTC'}}))
        for i in range(3):
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            print(f'MSG {i}: {msg[:120]}...')
        print('OK - data flowing')
asyncio.run(test())
"
```

### Step 3: Check network differences
```bash
curl -s https://api.hyperliquid.xyz/info -d '{"type":"meta"}' | head -100
ping -c 3 api.hyperliquid.xyz
traceroute api.hyperliquid.xyz
```

### Step 4: Check if file is actually being written (inotify)
```bash
inotifywait -m ../data/features/2026-04-19/ -e modify -e close_write
```

## Acceptance Criteria

The issue is resolved when ALL of the following are true:

1. `make run` on su-35 produces non-zero parquet files within 7 minutes of startup
2. `python3 -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/features/<date>/<file>.parquet').num_rows)"` returns > 0
3. The dashboard shows `n_files > 0` and `total_bytes > 0` for newly created files
4. The ingestor can run for 1+ hour without crashing, producing hourly file rotations
5. Data contains valid 194-column schema (3 metadata + 191 features)
6. At least 10 rows/second sustained throughput (30 rows/sec expected for 3 symbols)

## Files Involved

- `rust/ing/src/main.rs` — main loop with tokio::select!, feature emission
- `rust/ing/src/ws/client.rs` — WebSocket recv(), message parsing dispatch
- `rust/ing/src/ws/messages.rs` — parse_ws_message(), WsMessage enum
- `rust/ing/src/state/mod.rs` — MarketState.update(), compute_features(), initialized flag
- `rust/ing/src/output/writer.rs` — ParquetWriter buffer (10K rows), flush, file rotation
- `config/ing.toml` — data_dir, emission_interval_ms, row_group_size

## Workaround

Run the ingestor on the dev machine (where it works) and copy parquet files to su-35 for analysis:
```bash
# On dev machine
make run  # let it run

# Copy data to su-35
rsync -avz data/features/ su-35:~/nat/data/features/
```

## Next Step: Built-in Pipeline Traceability

The ingestor currently goes silent after "Opening new Parquet file" — there is no way to tell whether data is flowing without running debug mode. The following logging milestones should be added so that failures are diagnosable in release mode.

### Milestone 1: WebSocket Connection (already exists)
**File:** `rust/ing/src/main.rs` (lines 177-193)
- Already logs `WebSocket connected successfully` — no change needed.
- On failure, already retries with error log.

### Milestone 2: First Data Message Received
**File:** `rust/ing/src/ws/client.rs`
- Add a `first_message_logged: bool` flag to `HyperliquidClient`.
- On the first successful `stream.next()` that returns data, log at INFO level:
  ```
  INFO First WebSocket data message received symbol=BTC elapsed_since_connect=1.2s
  ```
- If no message arrives within 30 seconds of connection, log a WARNING:
  ```
  WARN No WebSocket data received after 30s — possible network/firewall issue symbol=BTC
  ```

### Milestone 3: State Initialization
**File:** `rust/ing/src/state/mod.rs`
- When `initialized` transitions from `false` to `true`, log at INFO:
  ```
  INFO Market state initialized (first Book message processed) symbol=ETH
  ```
- This confirms the message was not only received but successfully parsed as a Book update.

### Milestone 4: First Feature Computed
**File:** `rust/ing/src/main.rs` (inside `run_symbol_ingestor`, emission_ticker branch)
- On the first successful `compute_features()` (sequence_id == 1), log at INFO:
  ```
  INFO First feature vector computed symbol=ETH elapsed_since_connect=3.4s
  ```
- If `compute_features()` returns `None` for 30+ seconds after connection, log a WARNING:
  ```
  WARN No features computed after 30s — state not initialized? symbol=BTC messages_received=0
  ```

### Milestone 5: Buffer Fill Progress
**File:** `rust/ing/src/output/writer.rs`
- Log buffer progress at 25%, 50%, 75%, and 100% of capacity:
  ```
  INFO Writer buffer 25% full (2500/10000 rows) elapsed_since_open=1m23s
  ```
- On flush, log confirmation:
  ```
  INFO Buffer flushed to disk rows=10000 file=20260419_204259.parquet file_size=2.7MB
  ```

### Milestone 6: Periodic Health Summary (every 60s)
**File:** `rust/ing/src/main.rs` (inside `run_symbol_ingestor`)
- Add a second `tokio::time::interval(Duration::from_secs(60))` in the select loop.
- Every 60 seconds, log a one-line health summary at INFO:
  ```
  INFO Health: symbol=BTC connected=true messages=1847 features=612 buffer=612/10000 uptime=5m12s
  ```
- If `messages == 0` and uptime > 30s, escalate to WARN:
  ```
  WARN Health: symbol=BTC connected=true messages=0 features=0 — NO DATA FLOWING uptime=2m30s
  ```

### Milestone 7: File Write Confirmation
**File:** `rust/ing/src/output/writer.rs`
- After `writer.write(&batch)`, verify file size is non-zero and log:
  ```
  INFO Parquet write confirmed rows_in_file=10000 file_size=2.7MB path=../data/features/2026-04-19/20260419_204259.parquet
  ```
- On file rotation (close + open), log the closed file's final stats:
  ```
  INFO Closed Parquet file rows=30000 size=8.1MB duration=55m path=...
  INFO Opening new Parquet file path=...
  ```

### Implementation Notes

- All milestone logs use INFO level (visible in release mode by default).
- Warnings use WARN level for conditions that indicate probable failure.
- No debug-level logs — everything must be visible in production.
- Elapsed times should use `Instant::now()` captured at connection time.
- The health summary timer should be a third branch in the `tokio::select!` loop (not biased).
- Buffer progress logging should use a simple threshold check (e.g., `last_logged_pct`) to avoid spamming.

### Expected Output (working system)
```
INFO Starting ING - Hyperliquid Ingestor
INFO WebSocket connected successfully symbol=BTC
INFO WebSocket connected successfully symbol=ETH
INFO WebSocket connected successfully symbol=SOL
INFO Opening new Parquet file path=../data/features/2026-04-19/20260419_204259.parquet
INFO First WebSocket data message received symbol=BTC elapsed=0.4s
INFO First WebSocket data message received symbol=ETH elapsed=0.4s
INFO First WebSocket data message received symbol=SOL elapsed=0.5s
INFO Market state initialized symbol=BTC
INFO Market state initialized symbol=ETH
INFO Market state initialized symbol=SOL
INFO First feature vector computed symbol=BTC elapsed=1.2s
INFO First feature vector computed symbol=ETH elapsed=1.3s
INFO First feature vector computed symbol=SOL elapsed=1.4s
INFO Health: symbol=BTC connected=true messages=312 features=104 buffer=312/10000 uptime=1m0s
INFO Writer buffer 25% full (2500/10000 rows) elapsed=1m23s
INFO Health: symbol=BTC connected=true messages=1847 features=612 buffer=2500/10000 uptime=2m0s
INFO Writer buffer 50% full (5000/10000 rows) elapsed=2m46s
INFO Writer buffer 75% full (7500/10000 rows) elapsed=4m10s
INFO Buffer flushed to disk rows=10000 file_size=2.7MB
```

### Expected Output (broken system — su-35 scenario)
```
INFO Starting ING - Hyperliquid Ingestor
INFO WebSocket connected successfully symbol=BTC
INFO WebSocket connected successfully symbol=ETH
INFO WebSocket connected successfully symbol=SOL
INFO Opening new Parquet file path=../data/features/2026-04-19/20260419_204259.parquet
WARN No WebSocket data received after 30s symbol=BTC
WARN No WebSocket data received after 30s symbol=ETH
WARN No WebSocket data received after 30s symbol=SOL
WARN Health: symbol=BTC connected=true messages=0 features=0 — NO DATA FLOWING uptime=1m0s
WARN Health: symbol=ETH connected=true messages=0 features=0 — NO DATA FLOWING uptime=1m0s
WARN Health: symbol=SOL connected=true messages=0 features=0 — NO DATA FLOWING uptime=1m0s
```

This makes the failure point immediately obvious without needing debug mode.
