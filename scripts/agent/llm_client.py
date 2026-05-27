"""
LLM Client — Claude wrapper with cost control and logging.

Supports two modes:
  - "cli"  (default): calls `claude -p` using Max subscription — zero API cost
  - "api": uses anthropic Python SDK — requires ANTHROPIC_API_KEY, pay-per-token

Every call is logged to SQLite (llm_calls table) for reproducibility.
Per-cycle budget enforced at the client level.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone

log = logging.getLogger("nat.llm")


class LLMClient:
    """Claude wrapper with budget enforcement and call logging."""

    def __init__(self, config: dict, store=None):
        """
        config: from agent.toml [agent.llm]
        store:  StateStore for logging (optional — if None, logs to logger only)
        """
        self._mode = config.get("mode", "cli")
        self._model = config.get("model", "claude-sonnet-4-20250514")
        self._max_calls = config.get("max_calls_per_cycle", 3)
        self._max_tokens = config.get("max_tokens_per_call", 4096)
        self._temperature = config.get("temperature", 0.7)
        self._store = store
        self._calls_this_cycle = 0
        self._agent = config.get("agent_name", "unknown")

        if self._mode == "api":
            api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"LLM client in api mode requires {api_key_env} environment variable"
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            # CLI mode — verify claude is available
            self._client = None
            self._claude_bin = config.get("claude_bin", "claude")
            try:
                r = subprocess.run(
                    [self._claude_bin, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                log.info("LLM client using CLI mode: %s", r.stdout.strip())
            except FileNotFoundError:
                raise EnvironmentError(
                    f"LLM client in cli mode requires '{self._claude_bin}' in PATH"
                )

    def call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tag: str = "",
    ) -> str | None:
        """
        Single Claude call with budget enforcement.
        Returns response text or None if budget exhausted.
        """
        if self._calls_this_cycle >= self._max_calls:
            log.info("LLM budget exhausted (%d/%d calls this cycle)",
                     self._calls_this_cycle, self._max_calls)
            return None

        if self._mode == "api":
            return self._call_api(system, user, max_tokens=max_tokens,
                                  temperature=temperature, tag=tag)
        else:
            return self._call_cli(system, user, max_tokens=max_tokens, tag=tag)

    def _call_api(self, system, user, *, max_tokens=None, temperature=None,
                  tag="") -> str | None:
        """Call via anthropic SDK (pay-per-token)."""
        max_tok = max_tokens or self._max_tokens
        temp = temperature if temperature is not None else self._temperature

        t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tok,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text
            latency_ms = int((time.monotonic() - t0) * 1000)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            self._calls_this_cycle += 1

            log.info("LLM api [%s]: %d in / %d out tokens, %dms, %d/%d budget",
                     tag, input_tokens, output_tokens, latency_ms,
                     self._calls_this_cycle, self._max_calls)

            if self._store:
                self._log_call(tag, system, user, text, input_tokens,
                               output_tokens, latency_ms)
            return text

        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.error("LLM api [%s] failed after %dms: %s", tag, latency_ms, e)
            if self._store:
                self._log_call(tag, system, user, f"ERROR: {e}", 0, 0, latency_ms)
            return None

    def _call_cli(self, system, user, *, max_tokens=None, tag="") -> str | None:
        """Call via claude CLI (uses Max subscription, no per-token cost)."""
        # Build prompt: prepend system instructions to user message
        prompt = f"{system}\n\n---\n\n{user}"
        max_tok = max_tokens or self._max_tokens

        cmd = [
            self._claude_bin, "-p", prompt,
            "--output-format", "text",
            "--model", self._model,
            "--max-turns", "1",
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 min timeout per call
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            if result.returncode != 0:
                err = result.stderr.strip()[:200]
                log.error("LLM cli [%s] failed (rc=%d) after %dms: %s",
                          tag, result.returncode, latency_ms, err)
                if self._store:
                    self._log_call(tag, system, user, f"ERROR: rc={result.returncode} {err}",
                                   0, 0, latency_ms)
                return None

            text = result.stdout.strip()
            self._calls_this_cycle += 1

            # Estimate tokens (CLI doesn't report exact counts)
            est_in = len(prompt) // 4
            est_out = len(text) // 4

            log.info("LLM cli [%s]: ~%d in / ~%d out tokens (est), %dms, %d/%d budget",
                     tag, est_in, est_out, latency_ms,
                     self._calls_this_cycle, self._max_calls)

            if self._store:
                self._log_call(tag, system, user, text, est_in, est_out, latency_ms)

            return text

        except subprocess.TimeoutExpired:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.error("LLM cli [%s] timed out after %dms", tag, latency_ms)
            if self._store:
                self._log_call(tag, system, user, "ERROR: timeout", 0, 0, latency_ms)
            return None
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.error("LLM cli [%s] failed after %dms: %s", tag, latency_ms, e)
            if self._store:
                self._log_call(tag, system, user, f"ERROR: {e}", 0, 0, latency_ms)
            return None

    def _log_call(self, tag, system, user, response, in_tok, out_tok, latency_ms):
        """Write call record to SQLite."""
        try:
            conn = self._store._conn
            with conn:
                conn.execute(
                    "INSERT INTO llm_calls "
                    "(agent, tag, system, user_msg, response, model, "
                    " input_tokens, output_tokens, latency_ms, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self._agent, tag, system, user, response, self._model,
                     in_tok, out_tok, latency_ms,
                     datetime.now(timezone.utc).isoformat()),
                )
        except Exception as e:
            log.debug("Failed to log LLM call: %s", e)

    @property
    def calls_this_cycle(self) -> int:
        return self._calls_this_cycle

    @property
    def budget_remaining(self) -> int:
        return max(0, self._max_calls - self._calls_this_cycle)

    def reset_cycle(self) -> None:
        """Reset per-cycle call counter. Called at cycle start."""
        self._calls_this_cycle = 0
