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
