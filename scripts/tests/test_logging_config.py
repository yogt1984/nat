"""Tests for the centralized logging configuration.

Covers:
- setup_logging creates file + stderr handlers
- JSON formatter produces valid JSON with required fields
- Human formatter includes context when present
- Correlation context (set/get/clear) is thread-local
- File handler creates log files and rotates
- Repeated setup_logging calls don't duplicate handlers
- file_only mode suppresses stderr
- json_stderr mode uses JSON on stderr
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from logging_config import (
    setup_logging,
    set_context,
    clear_context,
    get_context,
    JSONFormatter,
    HumanFormatter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_context():
    """Ensure context is cleared between tests."""
    clear_context()
    yield
    clear_context()


@pytest.fixture
def log_dir(tmp_path):
    """Provide a temp log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


# ===========================================================================
# JSON Formatter
# ===========================================================================

class TestJSONFormatter:
    def test_produces_valid_json(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["msg"] == "test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "nat.test"
        assert "ts" in parsed

    def test_includes_context(self):
        set_context(hypothesis_id="HYP-001", agent="micro")
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="gate passed", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["ctx"]["hypothesis_id"] == "HYP-001"
        assert parsed["ctx"]["agent"] == "micro"

    def test_no_context_no_ctx_key(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="no ctx", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert "ctx" not in parsed

    def test_includes_exception(self):
        fmt = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="nat.test", level=logging.ERROR, pathname="", lineno=0,
            msg="error occurred", args=(), exc_info=exc_info,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_timestamp_is_utc(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="ts test", args=(), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["ts"].endswith("+00:00")

    def test_message_with_args(self):
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="IC=%.3f for %s", args=(0.152, "BTC"), exc_info=None,
        )
        line = fmt.format(record)
        parsed = json.loads(line)
        assert parsed["msg"] == "IC=0.152 for BTC"


# ===========================================================================
# Human Formatter
# ===========================================================================

class TestHumanFormatter:
    def test_basic_format(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        line = fmt.format(record)
        assert "[INFO]" in line
        assert "nat.test" in line
        assert "hello" in line

    def test_includes_context_suffix(self):
        set_context(cycle_id="CYC-001")
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        line = fmt.format(record)
        assert "cycle_id=CYC-001" in line

    def test_no_context_no_suffix(self):
        fmt = HumanFormatter()
        record = logging.LogRecord(
            name="nat.test", level=logging.INFO, pathname="", lineno=0,
            msg="clean", args=(), exc_info=None,
        )
        line = fmt.format(record)
        assert "[" not in line.split("clean")[1]  # no context bracket after msg


# ===========================================================================
# Context management
# ===========================================================================

class TestContext:
    def test_set_and_get(self):
        set_context(cycle_id="C1", hypothesis_id="H1")
        ctx = get_context()
        assert ctx["cycle_id"] == "C1"
        assert ctx["hypothesis_id"] == "H1"

    def test_clear(self):
        set_context(cycle_id="C1")
        clear_context()
        assert get_context() == {}

    def test_update_preserves_existing(self):
        set_context(cycle_id="C1")
        set_context(hypothesis_id="H1")
        ctx = get_context()
        assert ctx["cycle_id"] == "C1"
        assert ctx["hypothesis_id"] == "H1"

    def test_overwrite_key(self):
        set_context(hypothesis_id="H1")
        set_context(hypothesis_id="H2")
        assert get_context()["hypothesis_id"] == "H2"

    def test_thread_isolation(self):
        """Context is thread-local — different threads don't see each other."""
        set_context(thread="main")
        results = {}

        def worker():
            set_context(thread="worker")
            results["worker"] = get_context().get("thread")

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert get_context()["thread"] == "main"
        assert results["worker"] == "worker"

    def test_thread_isolation_no_leak(self):
        """A thread that doesn't set context sees empty context."""
        clear_context()
        results = {}

        def worker():
            results["ctx"] = get_context()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert results["ctx"] == {}


# ===========================================================================
# setup_logging
# ===========================================================================

class TestSetupLogging:
    def test_creates_log_file(self, log_dir):
        logger = setup_logging("nat.test_file", log_dir=log_dir)
        logger.info("test message")
        # Flush handlers
        for h in logger.handlers:
            h.flush()

        log_file = log_dir / "nat.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

    def test_log_file_is_valid_json(self, log_dir):
        logger = setup_logging("nat.test_json", log_dir=log_dir)
        logger.info("line one")
        logger.warning("line two")
        for h in logger.handlers:
            h.flush()

        lines = (log_dir / "nat.jsonl").read_text().strip().split("\n")
        for line in lines:
            parsed = json.loads(line)
            assert "ts" in parsed
            assert "level" in parsed
            assert "msg" in parsed

    def test_no_duplicate_handlers(self, log_dir):
        """Calling setup_logging twice doesn't duplicate handlers."""
        setup_logging("nat.test_dup", log_dir=log_dir)
        setup_logging("nat.test_dup", log_dir=log_dir)
        logger = logging.getLogger("nat.test_dup")
        # Should have exactly 2 handlers (file + stderr)
        assert len(logger.handlers) == 2

    def test_file_only_mode(self, log_dir):
        logger = setup_logging("nat.test_fileonly", log_dir=log_dir, file_only=True)
        # Only file handler, no stderr
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0],
                          logging.handlers.TimedRotatingFileHandler)

    def test_json_stderr_mode(self, log_dir):
        logger = setup_logging("nat.test_jsonstderr", log_dir=log_dir,
                               json_stderr=True)
        # Both handlers exist
        assert len(logger.handlers) == 2
        # stderr handler should use JSONFormatter
        stderr_handler = [h for h in logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.handlers.TimedRotatingFileHandler)]
        assert len(stderr_handler) == 1
        assert isinstance(stderr_handler[0].formatter, JSONFormatter)

    def test_custom_level(self, log_dir):
        logger = setup_logging("nat.test_level", log_dir=log_dir,
                               level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_returns_logger(self, log_dir):
        result = setup_logging("nat.test_return", log_dir=log_dir)
        assert isinstance(result, logging.Logger)
        assert result.name == "nat.test_return"

    def test_propagate_false(self, log_dir):
        """Logger doesn't propagate to root (prevents double output)."""
        logger = setup_logging("nat.test_prop", log_dir=log_dir)
        assert logger.propagate is False

    def test_context_in_log_file(self, log_dir):
        """Correlation context appears in JSON log output."""
        logger = setup_logging("nat.test_ctx", log_dir=log_dir)
        set_context(hypothesis_id="HYP-TEST-001", agent="micro")
        logger.info("gate passed")
        for h in logger.handlers:
            h.flush()

        lines = (log_dir / "nat.jsonl").read_text().strip().split("\n")
        last = json.loads(lines[-1])
        assert last["ctx"]["hypothesis_id"] == "HYP-TEST-001"
        assert last["ctx"]["agent"] == "micro"

    def test_log_dir_created_if_missing(self, tmp_path):
        """setup_logging creates log_dir if it doesn't exist."""
        new_dir = tmp_path / "deep" / "nested" / "logs"
        assert not new_dir.exists()
        setup_logging("nat.test_mkdir", log_dir=new_dir)
        assert new_dir.exists()


# ===========================================================================
# Integration with agent base
# ===========================================================================

class TestAgentIntegration:
    def test_cli_main_uses_setup_logging(self, tmp_path):
        """cli_main uses setup_logging instead of basicConfig."""
        from agent.base import cli_main
        import inspect
        source = inspect.getsource(cli_main)
        assert "setup_logging" in source
        assert "basicConfig" not in source
