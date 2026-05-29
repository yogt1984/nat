"""Tests for utils.health — daemon heartbeat system."""

import json
import time

import pytest

from utils.health import HealthWriter, check_health


@pytest.fixture
def health_dir(tmp_path):
    return tmp_path / "health"


class TestHealthWriter:
    def test_beat_creates_file(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="RUN", cycle=1)
        path = health_dir / "test_daemon.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["daemon"] == "test_daemon"
        assert data["phase"] == "RUN"
        assert data["cycle"] == 1
        assert "epoch" in data

    def test_beat_updates_in_place(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="INIT", cycle=0)
        hw.beat(phase="EXECUTE", cycle=5)
        data = json.loads((health_dir / "test_daemon.json").read_text())
        assert data["phase"] == "EXECUTE"
        assert data["cycle"] == 5

    def test_beat_extra(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="RUN", cycle=1, extra={"hypotheses": 3})
        data = json.loads((health_dir / "test_daemon.json").read_text())
        assert data["extra"]["hypotheses"] == 3

    def test_shutdown(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="RUN", cycle=1)
        hw.shutdown()
        data = json.loads((health_dir / "test_daemon.json").read_text())
        assert data["phase"] == "STOPPED"
        assert data["cycle"] == -1

    def test_path_property(self, health_dir):
        hw = HealthWriter("my_daemon", health_dir=health_dir)
        assert hw.path == health_dir / "my_daemon.json"


class TestCheckHealth:
    def test_healthy(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="RUN", cycle=1)
        ok, msg = check_health("test_daemon", max_age_s=10, health_dir=health_dir)
        assert ok is True
        assert "healthy" in msg

    def test_no_heartbeat(self, health_dir):
        health_dir.mkdir(parents=True, exist_ok=True)
        ok, msg = check_health("missing", health_dir=health_dir)
        assert ok is False
        assert "no heartbeat" in msg

    def test_stopped_daemon(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.shutdown()
        ok, msg = check_health("test_daemon", health_dir=health_dir)
        assert ok is False
        assert "stopped" in msg.lower() or "STOPPED" in msg

    def test_stale_heartbeat(self, health_dir):
        hw = HealthWriter("test_daemon", health_dir=health_dir)
        hw.beat(phase="RUN", cycle=1)
        # Backdate the epoch
        path = health_dir / "test_daemon.json"
        data = json.loads(path.read_text())
        data["epoch"] = time.time() - 300
        path.write_text(json.dumps(data))
        ok, msg = check_health("test_daemon", max_age_s=60, health_dir=health_dir)
        assert ok is False
        assert "stale" in msg
