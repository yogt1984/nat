"""
LLM Client — Claude API wrapper with cost control and logging.

Every call is logged to SQLite (llm_calls table) for reproducibility.
Per-cycle budget enforced at the client level.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("nat.llm")


class LLMClient:
    """Claude API wrapper with budget enforcement and call logging."""

    def __init__(self, config: dict, store=None):
        """
        config: from agent.toml [agent.llm]
        store:  StateStore for logging (optional — if None, logs to logger only)
        """
        self._model = config.get("model", "claude-sonnet-4-20250514")
        self._max_calls = config.get("max_calls_per_cycle", 3)
        self._max_tokens = config.get("max_tokens_per_call", 4096)
        self._temperature = config.get("temperature", 0.7)
        self._store = store
        self._calls_this_cycle = 0
        self._agent = config.get("agent_name", "unknown")

        api_key_env = config.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"LLM client requires {api_key_env} environment variable"
            )

        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

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
        Single Claude API call with budget enforcement.
        Returns response text or None if budget exhausted.
        """
        if self._calls_this_cycle >= self._max_calls:
            log.info("LLM budget exhausted (%d/%d calls this cycle)",
                     self._calls_this_cycle, self._max_calls)
            return None

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

            log.info("LLM call [%s]: %d in / %d out tokens, %dms, %d/%d budget",
                     tag, input_tokens, output_tokens, latency_ms,
                     self._calls_this_cycle, self._max_calls)

            # Log to SQLite
            if self._store:
                self._log_call(tag, system, user, text, input_tokens,
                               output_tokens, latency_ms)

            return text

        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            log.error("LLM call [%s] failed after %dms: %s", tag, latency_ms, e)
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
