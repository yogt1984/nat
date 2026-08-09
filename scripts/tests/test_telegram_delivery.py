"""REL-4 — planted tests for verifiable Telegram delivery.

The zombie-ingestor postmortem's third gap was "no Telegram". REL-1/2/3 shipped the
detection and remediation; what REL-4 adds is the property that a page CANNOT fail
silently. Three failure modes are planted:

- **formatting kills the page**: Telegram rejects Markdown with unmatched entities
  (400) and the old sender returned False and moved on — an alert whose text happens
  to contain `_` would never page. The sender must fall back to plain text before
  giving up, and gap alerts send plain by default (formatting is never worth a page);
- **the outcome vanishes**: `_send` used to fire-and-forget; now every alert line in
  alerts.log carries its channel outcome (`telegram=ok|FAILED|unconfigured`) and a
  configured-but-failed send logs at WARNING, so a broken token is visible the first
  time it eats a page, not months later;
- **"verified" without a send**: the `test` subcommand pages through the daemon's own
  `_send` path — same code that fires on a real gap — and exits non-zero unless the
  Telegram API accepted the message. No mock passes that; only delivery does.

No test here touches the network: the API interaction is monkeypatched. The real
end-to-end send (the phone that buzzes) is `nat gap test`, run once with live creds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (suite convention)
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ops.gap_alert as ga  # noqa: E402
import tournament.report as report  # noqa: E402


# ── send_telegram: plain-text fallback ───────────────────────────────────────────
class _Resp:
    def __init__(self, ok: bool, text: str = ""):
        self.ok = ok
        self.text = text


class TestSendTelegramFallback:
    @pytest.fixture(autouse=True)
    def _creds(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t0k3n")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def test_markdown_rejection_falls_back_to_plain(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json)
            if "parse_mode" in json:
                return _Resp(False, "Bad Request: can't parse entities")
            return _Resp(True)

        monkeypatch.setattr(report, "requests", type("R", (), {"post": staticmethod(fake_post)}))
        assert report.send_telegram("gap_state.json has _underscores_") is True
        assert len(calls) == 2
        assert "parse_mode" in calls[0] and "parse_mode" not in calls[1]

    def test_plain_rejection_is_a_failure(self, monkeypatch):
        monkeypatch.setattr(
            report, "requests",
            type("R", (), {"post": staticmethod(lambda *a, **k: _Resp(False, "403"))}))
        assert report.send_telegram("x") is False

    def test_plain_mode_sends_no_parse_mode(self, monkeypatch):
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append(json)
            return _Resp(True)

        monkeypatch.setattr(report, "requests", type("R", (), {"post": staticmethod(fake_post)}))
        assert report.send_telegram("plain", parse_mode=None) is True
        assert calls and "parse_mode" not in calls[0]

    def test_unconfigured_never_touches_network(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
        monkeypatch.setattr(
            report, "requests",
            type("R", (), {"post": staticmethod(
                lambda *a, **k: (_ for _ in ()).throw(AssertionError("network hit")))}))
        assert report.send_telegram("x") is False


# ── _send: outcomes on the record ────────────────────────────────────────────────
def _alerter(tmp_path) -> ga.GapAlerter:
    cfg = ga.load_config()
    cfg.update({
        "data_dirs": [str(tmp_path / "data")],
        "state_path": str(tmp_path / "gap_state.json"),
        "heartbeat_path": str(tmp_path / "hb"),
        "pid_file": str(tmp_path / "pid"),
        "alert_log": str(tmp_path / "alerts.log"),
        "pause_file": str(tmp_path / "paused"),
        "auto_restart": False,
    })
    return ga.GapAlerter(config=cfg)


class TestSendOutcomeRecording:
    def test_unconfigured_is_tagged(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        a = _alerter(tmp_path)
        out = a._send("planted alert")
        assert out["telegram"] == "unconfigured"
        line = (tmp_path / "alerts.log").read_text()
        assert "planted alert" in line and "telegram=unconfigured" in line

    def test_delivered_is_tagged_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(report, "send_telegram", lambda m, parse_mode="Markdown": True)
        a = _alerter(tmp_path)
        out = a._send("planted alert")
        assert out["telegram"] == "ok"
        assert "telegram=ok" in (tmp_path / "alerts.log").read_text()

    def test_failed_send_is_loud_and_tagged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(report, "send_telegram", lambda m, parse_mode="Markdown": False)
        warned = []
        # spy the module logger directly — the repo's JSON logger does not
        # propagate to root, so caplog cannot see it
        monkeypatch.setattr(ga.log, "warning",
                            lambda msg, *a, **k: warned.append(msg % a if a else msg))
        out = _alerter(tmp_path)._send("planted alert")
        assert out["telegram"] == "FAILED"
        assert "telegram=FAILED" in (tmp_path / "alerts.log").read_text()
        assert any("FAILED" in m for m in warned)

    def test_gap_alerts_page_plain_never_markdown(self, tmp_path, monkeypatch):
        """Formatting must never cost a page: the daemon sends parse_mode=None."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        seen = {}

        def spy(m, parse_mode="Markdown"):
            seen["parse_mode"] = parse_mode
            return True

        monkeypatch.setattr(report, "send_telegram", spy)
        _alerter(tmp_path)._send("x")
        assert seen["parse_mode"] is None


# ── the delivery test itself ─────────────────────────────────────────────────────
class TestDeliveryTest:
    def test_pass_requires_api_acceptance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(report, "send_telegram", lambda m, parse_mode="Markdown": True)
        out = ga.delivery_test(_alerter(tmp_path))
        assert out["telegram"] == "ok" and out["telegram_configured"] is True
        assert out["passed"] is True

    def test_unconfigured_fails_the_test(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        out = ga.delivery_test(_alerter(tmp_path))
        assert out["passed"] is False and out["telegram"] == "unconfigured"

    def test_api_rejection_fails_the_test(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(report, "send_telegram", lambda m, parse_mode="Markdown": False)
        out = ga.delivery_test(_alerter(tmp_path))
        assert out["passed"] is False and out["telegram"] == "FAILED"
