# NAT Cloud Research Agent — Technical Specification

**Date**: 2026-06-01
**Status**: PROPOSED
**Purpose**: Autonomous 24/7 quant research agent deployed in the cloud

---

## 1. Overview

A persistent Claude-powered agent that runs nightly research cycles on the
NAT platform: calibrating algorithms, evaluating hypotheses, running gauntlet
sweeps, and delivering reports via Telegram. The agent operates autonomously
on a schedule, with the full NAT codebase and data available as its workspace.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Host: su-35 (or cloud VM)                               │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Docker Container: nat-research-agent              │  │
│  │                                                    │  │
│  │  daemon.py (scheduler)                             │  │
│  │    │                                               │  │
│  │    ├── 23:00 UTC ── Nightly Research Cycle         │  │
│  │    │   ├── Data freshness check                    │  │
│  │    │   ├── nat calibrate (param learning)          │  │
│  │    │   ├── nat gauntlet run (OOS validation)       │  │
│  │    │   ├── Hypothesis evaluation                   │  │
│  │    │   └── Report generation + Telegram delivery   │  │
│  │    │                                               │  │
│  │    ├── 06:00 UTC ── Morning Health Check           │  │
│  │    │   ├── Ingestor status                         │  │
│  │    │   ├── Parquet file count vs expected           │  │
│  │    │   └── Alert if data gap > 2h                  │  │
│  │    │                                               │  │
│  │    └── On-demand ── API trigger for ad-hoc tasks   │  │
│  │                                                    │  │
│  │  ClaudeSDKClient                                   │  │
│  │    ├── Tools: Read, Bash, Edit, Glob, Grep         │  │
│  │    ├── permission_mode: acceptEdits                │  │
│  │    ├── max_turns: 50 per cycle                     │  │
│  │    └── cwd: /home/onat/nat                         │  │
│  │                                                    │  │
│  │  SessionStore → Redis (session persistence)        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Volumes:                                                │
│    /home/onat/nat/data/features/  ← parquet (read)       │
│    /home/onat/nat/data/research/  ← outputs (read/write) │
│    /home/onat/nat/config/         ← algo config (r/w)    │
│    /home/onat/nat/reports/        ← reports (write)      │
└──────────────────────────────────────────────────────────┘
```

## 3. Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| Agent runtime       | Claude Agent SDK (Python)           |
| Model               | Claude Opus 4 / Sonnet 4            |
| Scheduler           | Python `schedule` library           |
| Container           | Docker (single container)           |
| Session persistence | Redis (local) or S3 (cloud)         |
| Reporting           | Telegram Bot API                    |
| Authentication      | ANTHROPIC_API_KEY env var           |
| Monitoring          | OpenTelemetry spans + log file      |
| Host                | su-35 (primary) or cloud VM         |

## 4. Agent SDK Core Pattern

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

SYSTEM_PROMPT = """
You are NAT's autonomous research agent. Your workspace is /home/onat/nat.

Available commands:
  nat status              — system health
  nat gauntlet run        — multi-day OOS sweep
  nat gauntlet report     — latest results
  nat calibrate           — nightly param learning
  nat daily               — single-day OOS eval
  nat algorithm evaluate  — per-algorithm analysis

Your data is in data/features/ (parquet, 100ms ticks, 3 symbols).
Your config is in config/ (TOML files).
Your outputs go to data/research/ and reports/.

Rules:
  - Never modify algorithm source code
  - Never push to git without explicit approval
  - Always verify data freshness before analysis
  - Report findings as structured JSON + human-readable summary
  - Flag any parameter shift > 50% for manual review
"""

async def run_cycle(task: str, session_id: str = None):
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash", "Edit", "Glob", "Grep"],
        permission_mode="acceptEdits",
        system_prompt=SYSTEM_PROMPT,
        max_turns=50,
        cwd="/home/onat/nat",
    )
    if session_id:
        options.resume = session_id

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)
        result_text = ""
        async for msg in client.receive_response():
            if msg.is_result:
                result_text = msg.result
                new_session_id = msg.session_id
        return result_text, new_session_id
```

## 5. Nightly Research Cycle

The primary cycle runs at 23:00 UTC daily. It issues a sequence of queries
to the agent within a single session, preserving context across steps.

```python
async def nightly_cycle():
    # Step 1: Data health
    result, sid = await run_cycle(
        "Check data freshness: list today's parquet files in data/features/, "
        "count rows per symbol, verify no gaps > 2 hours. "
        "Report as JSON: {symbol: {files: N, rows: N, gaps: [...]}}."
    )
    report = {"data_health": result}

    # Step 2: Algorithm calibration
    result, sid = await run_cycle(
        "Run parameter calibration for optimal_entry, hawkes_intensity, "
        "and funding_reversion. Use estimate_ou_params() on the last 7 days "
        "of imbalance data. Report per-symbol theta, sigma_process, sigma_obs. "
        "Flag any parameter shift > 50% vs current config.",
        session_id=sid
    )
    report["calibration"] = result

    # Step 3: Gauntlet evaluation
    result, sid = await run_cycle(
        "Run nat daily on today's data for all symbols. "
        "Parse the output and rank algorithms by total PnL. "
        "Compare to yesterday's ranking if available.",
        session_id=sid
    )
    report["gauntlet"] = result

    # Step 4: Hypothesis check
    result, sid = await run_cycle(
        "Check the hypothesis queue in data/agent/. "
        "Are there any hypotheses ready for promotion (passed all 5 gates)? "
        "Any that should be retired (failed 3+ consecutive cycles)? "
        "Summarize the active hypothesis pipeline.",
        session_id=sid
    )
    report["hypotheses"] = result

    # Step 5: Generate and send report
    result, sid = await run_cycle(
        "Compose a daily research report combining all findings from this "
        "session. Format: markdown with sections for Data Health, Calibration, "
        "Algorithm Performance, and Hypothesis Pipeline. "
        "Write to reports/daily_YYYY-MM-DD.md. "
        "Also output a 10-line Telegram summary.",
        session_id=sid
    )
    
    send_telegram(extract_telegram_summary(result))
    save_session(sid)
```

## 6. Telegram Integration

```python
import requests

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send_telegram(message: str):
    """Send report to Telegram. Splits if > 4096 chars."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i+4096] for i in range(0, len(message), 4096)]
    for chunk in chunks:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
        })
```

## 7. Scheduler Daemon

```python
import asyncio
import schedule
import logging

logging.basicConfig(filename="logs/research_agent.log", level=logging.INFO)

def schedule_cycles():
    schedule.every().day.at("23:00").do(
        lambda: asyncio.run(nightly_cycle())
    )
    schedule.every().day.at("06:00").do(
        lambda: asyncio.run(morning_health_check())
    )

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_cycles()
```

## 8. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Claude Agent SDK and dependencies
RUN pip install --no-cache-dir \
    claude-agent-sdk \
    schedule \
    requests

# Copy agent code
COPY scripts/cloud_agent/ /app/

# Environment (secrets injected at runtime)
ENV ANTHROPIC_API_KEY=""
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""

# Mount points (bound at docker run)
# /home/onat/nat/data      -> /nat/data
# /home/onat/nat/config    -> /nat/config
# /home/onat/nat/reports   -> /nat/reports
# /home/onat/nat/scripts   -> /nat/scripts

CMD ["python", "daemon.py"]
```

### Docker Run Command

```bash
docker run -d \
    --name nat-research-agent \
    --restart unless-stopped \
    -e ANTHROPIC_API_KEY="sk-ant-..." \
    -e TELEGRAM_BOT_TOKEN="..." \
    -e TELEGRAM_CHAT_ID="..." \
    -v /home/onat/nat:/nat:rw \
    nat-research-agent:latest
```

## 9. Session Persistence

Sessions allow the agent to maintain context across nightly cycles.
Yesterday's findings inform today's analysis.

```python
import json
from pathlib import Path

SESSION_FILE = Path("data/research/agent_sessions.json")

def save_session(session_id: str, cycle_date: str):
    sessions = json.loads(SESSION_FILE.read_text()) if SESSION_FILE.exists() else {}
    sessions[cycle_date] = session_id
    # Keep last 7 days
    keys = sorted(sessions.keys())[-7:]
    sessions = {k: sessions[k] for k in keys}
    SESSION_FILE.write_text(json.dumps(sessions, indent=2))

def get_last_session() -> str | None:
    if not SESSION_FILE.exists():
        return None
    sessions = json.loads(SESSION_FILE.read_text())
    if sessions:
        return sessions[max(sessions.keys())]
    return None
```

For production resilience, replace JSON with Redis:

```python
from claude_agent_sdk import RedisSessionStore

store = RedisSessionStore(url="redis://localhost:6379/1")
```

## 10. Monitoring and Alerting

```python
HEALTH_CHECKS = {
    "data_freshness": {
        "check": "ls data/features/$(date +%Y-%m-%d)/*.parquet | wc -l",
        "threshold": 20,  # expect 20+ files per day (hourly rotation x symbols)
        "alert": "Data gap detected — ingestor may be down",
    },
    "disk_usage": {
        "check": "df -h /home/onat/nat/data | tail -1 | awk '{print $5}'",
        "threshold": "85%",
        "alert": "Disk usage above 85% — rotate old parquet",
    },
    "ingestor_alive": {
        "check": "pgrep -f 'target/release/ing' | wc -l",
        "threshold": 1,
        "alert": "Ingestor process not running",
    },
}

async def morning_health_check():
    result, _ = await run_cycle(
        "Run health checks: "
        "1. Count today's parquet files (expect 20+). "
        "2. Check disk usage on /home/onat/nat/data (alert if >85%). "
        "3. Verify ingestor process is running (pgrep ing). "
        "4. Check last gauntlet report date (alert if >3 days stale). "
        "Report status as: OK / WARNING / CRITICAL per check."
    )
    if "CRITICAL" in result or "WARNING" in result:
        send_telegram(f"NAT Health Alert:\n{result}")
```

## 11. Security Constraints

| Rule | Enforcement |
|------|------------|
| No git push without approval | System prompt + `permission_mode` callback |
| No algorithm source modification | System prompt rule + read-only mount option |
| No exchange API calls | No exchange credentials in environment |
| API key rotation | Kubernetes secret or Docker secret, not hardcoded |
| Session data encrypted at rest | Redis AUTH + TLS, or S3 server-side encryption |
| Max spend per cycle | `max_turns=50` caps API usage (~$5-8 per cycle) |

## 12. Cost Estimate

| Component | Monthly cost |
|-----------|-------------|
| Claude API (nightly 50-turn cycle x 30) | ~$150-200 |
| Claude API (morning health x 30) | ~$15-20 |
| Claude API (ad-hoc queries, ~10/month) | ~$30-50 |
| Infrastructure (Docker on su-35) | $0 (existing hardware) |
| Infrastructure (cloud VM alternative) | ~$50/month (4GB RAM) |
| Redis (local) | $0 |
| Telegram Bot API | $0 |
| **Total (su-35 deployment)** | **~$200-270/month** |
| **Total (cloud VM deployment)** | **~$250-320/month** |

## 13. File Structure

```
scripts/cloud_agent/
    __init__.py
    daemon.py              # Scheduler + cycle orchestration
    agent.py               # ClaudeSDKClient wrapper + session management
    telegram.py            # Telegram Bot API reporter
    health.py              # Health check definitions
    Dockerfile             # Container definition

config/
    cloud_agent.toml       # Schedule times, max_turns, alert thresholds

docs/agent_specifications/
    agent_specs_1.md       # This document
```

## 14. Implementation Phases

### Phase 1: Minimal Viable Agent (1 day)
- daemon.py with schedule loop
- Single nightly cycle: `nat daily` + Telegram report
- Run directly on su-35 (no Docker yet)
- Manual start: `python scripts/cloud_agent/daemon.py &`

### Phase 2: Full Nightly Cycle (1 day)
- Multi-step cycle: health → calibrate → gauntlet → hypotheses → report
- Session persistence (JSON file)
- Morning health check
- Structured report format

### Phase 3: Containerization (0.5 day)
- Dockerfile + docker-compose
- Volume mounts for data/config/reports
- Environment-based secrets
- `--restart unless-stopped` for resilience

### Phase 4: Observability (0.5 day)
- OpenTelemetry integration for cost tracking
- Log rotation
- Cycle duration and token usage metrics
- Failure alerting via Telegram

## 15. Comparison: Agent SDK vs Routines

| Capability | Agent SDK (self-hosted) | Routines (Anthropic-hosted) |
|---|---|---|
| Access to local files | Full (mounted volumes) | No (repo clone only) |
| Access to parquet data | Yes | No |
| Run nat CLI commands | Yes | No (no local binary) |
| Custom scheduling | Full control | Cron-style only |
| Session persistence | Your infrastructure | Anthropic-managed |
| Telegram integration | Direct API calls | Via HTTP in routine code |
| Cost control | max_turns + monitoring | Daily run allowance |
| Setup effort | 1-2 days | 5 minutes |
| Maintenance | You manage Docker/host | Zero |

**Recommendation**: Agent SDK on su-35 for NAT. The agent needs local file access
(parquet data, nat CLI, config files) which Routines cannot provide. Routines are
better suited for repo-level tasks (PR review, code generation) where local data
access is not required.

## 16. References

- [Claude Agent SDK Overview](https://code.claude.com/docs/en/agent-sdk/overview)
- [Agent SDK Quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart)
- [Agent SDK Sessions](https://code.claude.com/docs/en/agent-sdk/sessions)
- [Agent SDK Hosting](https://code.claude.com/docs/en/agent-sdk/hosting)
- [Routines Documentation](https://code.claude.com/docs/en/routines)
