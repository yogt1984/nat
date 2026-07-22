# LLM Bridge and Skill Definition Spec

This document defines how to build an effective bridge between LLMs (OpenAI/Anthropic) and local project skills, and how to standardize skill definitions.

## 1) Purpose

The `llm-bridge` module is a translation and orchestration layer that:

- discovers skills from `agents/*/SKILL.md`
- converts skill schemas into provider-specific tool formats
- receives tool calls from the LLM
- routes execution to the correct skill service
- returns results to the LLM in the required format

This keeps skills provider-agnostic and keeps LLM logic centralized.

## 2) Recommended Monorepo Conventions

- `agents/<skill-name>/SKILL.md`: skill metadata and schemas
- `agents/<skill-name>/...`: skill runtime code
- `llm-bridge/`: discovery, schema mapping, and invocation router
- `shared/`: common schema validation, Redis helpers, and error models

## 3) Skill Definition Standard (`SKILL.md`)

Each skill must include YAML front matter in `SKILL.md`.

### Required fields

- `name`: globally unique tool name (snake_case recommended)
- `description`: clear usage description (when to use, when not to use)
- `input_schema`: JSON Schema object (request shape)

### Strongly recommended fields

- `output_schema`: JSON Schema object (response shape)
- `transport`: `http` | `redis` | `nats`
- `endpoint`: HTTP URL if `transport: http`
- `timeouts_ms`: timeout configuration
- `version`: semantic version
- `owner`: owning team or person

### Example `SKILL.md`

```yaml
---
name: backtester_run
description: "Runs historical backtests for a selected strategy and returns performance metrics."
version: "1.0.0"
owner: "quant-research"
transport: "http"
endpoint: "http://backtester:8000/backtest"
timeouts_ms:
  connect: 500
  read: 120000
input_schema:
  type: object
  additionalProperties: false
  properties:
    strategy:
      type: string
      enum: ["trend_following", "mean_reversion", "entropy_gated"]
    symbol:
      type: string
    start_date:
      type: string
      format: date
    end_date:
      type: string
      format: date
    parameters:
      type: object
      additionalProperties: true
  required: ["strategy", "symbol", "start_date", "end_date"]
output_schema:
  type: object
  additionalProperties: false
  properties:
    run_id:
      type: string
    status:
      type: string
      enum: ["completed", "failed"]
    metrics:
      type: object
      additionalProperties: true
  required: ["run_id", "status", "metrics"]
---
Implementation notes for developers:
- Input must be validated against input_schema.
- Return compact, deterministic JSON.
- Include error details without leaking secrets.
```

## 4) LLM Bridge Responsibilities

The bridge should implement these core functions:

1. `discover_skills()`
   - scan `agents/*/SKILL.md`
   - parse YAML front matter
   - validate required fields

2. `build_tool_definitions(provider)`
   - map canonical skill spec to provider-specific tool format
   - OpenAI uses `parameters`
   - Anthropic uses `input_schema`

3. `invoke_skill(name, arguments)`
   - resolve skill from registry
   - validate arguments against `input_schema`
   - call skill over configured transport
   - validate result against `output_schema` (if provided)

4. `tool_loop(chat_request)`
   - send tools to model
   - handle one or multiple tool calls
   - append tool results back to model
   - return final assistant answer

## 5) API Contract for `llm-bridge`

Recommended endpoints:

- `GET /skills`
  - returns canonical skill metadata from registry
- `GET /tools?provider=openai|anthropic`
  - returns provider-specific tool definitions
- `POST /invoke`
  - deterministic direct skill invocation for testing and UI
- `POST /chat` (optional proxy mode)
  - full chat + tool orchestration loop

## 6) Standards and Compatibility

- **JSON Schema**: use for all `input_schema` and `output_schema`
- **OpenAPI 3.1**: expose bridge API contract (FastAPI supports this)
- **OpenAI function tools**: `type=function` + JSON schema parameters
- **Anthropic tools**: `name`, `description`, `input_schema`

Practical guidance:

- prefer strict schemas (`additionalProperties: false` where possible)
- keep tool names stable and version skill behavior explicitly
- include concise but specific descriptions to improve tool selection quality

## 7) Validation and Error Rules

- reject invalid skill front matter at startup
- reject invalid tool arguments with clear 4xx responses
- enforce invocation timeout per skill
- classify errors:
  - `validation_error`
  - `skill_timeout`
  - `skill_unavailable`
  - `skill_execution_error`
- return machine-readable error JSON

## 8) Performance and Reliability

- cache parsed skill registry in memory and refresh on interval (or startup-only)
- use connection pooling for HTTP skill calls
- support async and parallel tool calls when safe
- add circuit breaker/retry policy for flaky skills
- include idempotency keys for long-running invocations

## 9) Security Baseline

- never pass provider API keys to skill services
- authenticate bridge endpoints (service token/JWT)
- sanitize logs (no secrets, no raw credentials)
- enforce per-skill allowlist and rate limits
- isolate execution permissions by skill type

## 10) Observability

Track at minimum:

- tool call latency (p50/p95/p99)
- success/failure rate per skill
- schema validation failures
- retries/timeouts
- token usage and cost per request path

## 11) Implementation Checklist

- [ ] Create canonical Pydantic models for skill metadata
- [ ] Implement `agents/*/SKILL.md` scanner + parser
- [ ] Add JSON Schema validation for inputs/outputs
- [ ] Implement provider mappers (OpenAI, Anthropic)
- [ ] Implement invocation transports (HTTP first; Redis optional)
- [ ] Add `/tools`, `/invoke`, `/skills`, and optional `/chat`
- [ ] Add structured logs, metrics, and health checks
- [ ] Add integration tests with at least one sample skill

---

If a new skill follows this spec, it becomes LLM-callable without changing bridge business logic.
