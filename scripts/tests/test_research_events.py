"""Tests for research event publishing via Redis Streams.

Verifies:
- publish_research_event() serialises and publishes to correct stream
- _get_redis() handles missing Redis gracefully
- set_redis() allows dependency injection (no mock.patch needed)
- BaseRunner._publish_event() is best-effort (no exceptions)
- ResearchAgent._publish_event() is best-effort
- Event payloads match the 5 specified types
"""

import json
from pathlib import Path
from unittest.mock import MagicMock


import pytest

from agent.research_output import (
    publish_research_event,
    set_redis,
    _RESEARCH_STREAM,
)


# ===========================================================================
# publish_research_event unit tests
# ===========================================================================

class TestPublishResearchEvent:
    def setup_method(self):
        """Reset the global Redis connection between tests."""
        set_redis(None)

    def teardown_method(self):
        set_redis(None)

    def test_publish_sends_to_correct_stream(self):
        mock_conn = MagicMock()
        set_redis(mock_conn)

        result = publish_research_event("hypothesis_started", {
            "id": "h_001", "agent": "micro", "claim": "test",
        })

        assert result is True
        mock_conn.xadd.assert_called_once()
        call_args = mock_conn.xadd.call_args
        assert call_args[0][0] == _RESEARCH_STREAM
        payload = json.loads(call_args[0][1]["event"])
        assert payload["event"] == "hypothesis_started"
        assert payload["id"] == "h_001"

    def test_publish_returns_false_when_no_redis(self):
        set_redis(None)
        # _get_redis will try to connect and fail — returns None
        import agent.research_output as mod
        mod._redis_conn = None  # ensure lazy-connect path
        from unittest.mock import patch
        with patch("agent.research_output.os.environ.get",
                   return_value="redis://127.0.0.1:59999"):
            result = publish_research_event("gate_passed", {"id": "h_002"})
        assert result is False

    def test_publish_returns_false_on_redis_error(self):
        mock_conn = MagicMock()
        mock_conn.xadd.side_effect = ConnectionError("Redis down")
        set_redis(mock_conn)

        result = publish_research_event("gate_failed", {"id": "h_003"})
        assert result is False

    def test_payload_serialises_all_event_types(self):
        mock_conn = MagicMock()
        set_redis(mock_conn)

        events = [
            ("hypothesis_started", {"id": "h_001", "agent": "micro", "claim": "x"}),
            ("gate_passed", {"id": "h_001", "gate": "G1_discovery", "msg": "IC=0.05"}),
            ("gate_failed", {"id": "h_001", "gate": "G3_symbol", "reason": "1/3"}),
            ("hypothesis_registered", {"id": "h_001", "agent": "micro", "ic": 0.042}),
            ("cycle_completed", {"agent": "micro", "tested": 8, "passed": 1, "cycle": 42}),
        ]

        for event_type, payload in events:
            publish_research_event(event_type, payload)

        assert mock_conn.xadd.call_count == 5

        for call in mock_conn.xadd.call_args_list:
            raw = call[0][1]["event"]
            parsed = json.loads(raw)
            assert "event" in parsed


# ===========================================================================
# _get_redis graceful fallback
# ===========================================================================

class TestGetRedis:
    def setup_method(self):
        set_redis(None)

    def teardown_method(self):
        set_redis(None)

    def test_returns_none_when_redis_unavailable(self):
        from agent.research_output import _get_redis
        from unittest.mock import patch
        with patch("agent.research_output.os.environ.get",
                   return_value="redis://127.0.0.1:59999"):
            result = _get_redis()
        assert result is None

    def test_returns_injected_connection(self):
        mock_conn = MagicMock()
        set_redis(mock_conn)
        from agent.research_output import _get_redis
        assert _get_redis() is mock_conn


# ===========================================================================
# BaseRunner._publish_event integration
# ===========================================================================

class TestBaseRunnerPublishEvent:
    def setup_method(self):
        set_redis(None)

    def teardown_method(self):
        set_redis(None)

    def test_publish_event_best_effort(self):
        """_publish_event never raises, even when Redis is down."""
        from agent.base import BaseRunner

        class FakeRunner(BaseRunner):
            TIMEFRAME = None
            def _check_gates(self, report):
                return True, "ok"

        h = MagicMock()
        h.id = "h_test"
        h.test_protocol = []
        runner = FakeRunner(h, {})

        # No Redis set → should not raise
        runner._publish_event("gate_passed", {"id": "h_test", "gate": "G1"})

    def test_publish_event_calls_redis(self):
        from agent.base import BaseRunner

        class FakeRunner(BaseRunner):
            TIMEFRAME = None
            def _check_gates(self, report):
                return True, "ok"

        h = MagicMock()
        h.id = "h_test"
        runner = FakeRunner(h, {})

        mock_conn = MagicMock()
        set_redis(mock_conn)
        runner._publish_event("gate_failed", {"id": "h_test", "gate": "G2"})

        mock_conn.xadd.assert_called_once()
        payload = json.loads(mock_conn.xadd.call_args[0][1]["event"])
        assert payload["event"] == "gate_failed"


# ===========================================================================
# ResearchAgent._publish_event integration
# ===========================================================================

class TestResearchAgentPublishEvent:
    def setup_method(self):
        set_redis(None)

    def teardown_method(self):
        set_redis(None)

    def _make_agent(self):
        from agent.base import ResearchAgent

        class StubAgent(ResearchAgent):
            agent_type = "test_agent"
            config_section = "agent"
            generator_module_prefix = "agent.generators"
            default_generators = []
            def create_runner(self, h, m):
                return MagicMock()

        agent = StubAgent.__new__(StubAgent)
        agent.agent_type = "test_agent"
        return agent

    def test_publish_event_best_effort(self):
        """ResearchAgent._publish_event never raises."""
        agent = self._make_agent()
        agent._publish_event("cycle_completed", {"agent": "test", "cycle": 1})

    def test_publish_event_sends_to_redis(self):
        """ResearchAgent._publish_event forwards to Redis Stream."""
        agent = self._make_agent()
        mock_conn = MagicMock()
        set_redis(mock_conn)
        agent._publish_event("hypothesis_started", {
            "id": "h_test", "agent": "test", "claim": "test claim",
        })

        mock_conn.xadd.assert_called_once()
        payload = json.loads(mock_conn.xadd.call_args[0][1]["event"])
        assert payload["event"] == "hypothesis_started"
        assert payload["id"] == "h_test"
