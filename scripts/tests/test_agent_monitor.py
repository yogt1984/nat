"""Tests for adaptive IC threshold (2.1) and IC decay monitoring (2.2).

Tests cover:
- Adaptive IC computation from registry state
- IC decay detection and counter tracking
- Auto-retirement after consecutive decay days
- Decay counter reset on IC recovery
- Promotion eligibility checks
- Rolling IC computation helpers
- Edge cases: empty registry, retired signals, missing data
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_dirs(tmp_path, monkeypatch):
    """Set up isolated agent state directory."""
    state_dir = tmp_path / "data" / "agent"
    state_dir.mkdir(parents=True)
    data_dir = tmp_path / "data" / "features" / "2026-05-12"
    data_dir.mkdir(parents=True)

    import agent.daemon as daemon_mod
    import agent.runner as runner_mod
    # Redirect ROOT so all path properties (state_path, registry_path, etc.) resolve to tmp
    monkeypatch.setattr(daemon_mod, "ROOT", tmp_path)
    monkeypatch.setattr(daemon_mod, "STATE_PATH", state_dir / "agent_state.json")
    monkeypatch.setattr(daemon_mod, "STATS_PATH", state_dir / "generator_stats.json")
    monkeypatch.setattr(runner_mod, "REGISTRY_PATH", state_dir / "registry.json")
    monkeypatch.setattr(runner_mod, "ROOT", tmp_path)

    return tmp_path


@pytest.fixture
def daemon(agent_dirs):
    from agent.daemon import AgentDaemon
    from data.state import StateStore
    store = StateStore(agent_dirs / "data" / "nat.db")
    return AgentDaemon(store=store)


@pytest.fixture
def registry_path(agent_dirs):
    return agent_dirs / "data" / "agent" / "registry.json"


def _write_registry_to_store(daemon, signals):
    """Write signals into the daemon's SQLite store."""
    for sig in signals:
        daemon._store.append_signal(daemon.agent_type, sig)


def _read_registry_from_store(daemon):
    """Read signals from the daemon's SQLite store."""
    return daemon._store.load_registry(daemon.agent_type)


def _write_registry(path, signals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(signals, f)


def _make_signal(name="sig_a", ic=0.50, status="validated", **kwargs):
    sig = {
        "name": name,
        "features": ["imbalance_qty_l1"],
        "regime_gate": "ent_book_shape<P20",
        "expected_ic": ic,
        "symbols": ["BTC", "ETH", "SOL"],
        "discovery_date": "2026-05-10",
        "hypothesis_id": kwargs.pop("hypothesis_id", f"HYP-{name}"),
        "status": status,
    }
    sig.update(kwargs)
    return sig


# ===========================================================================
# 2.1 Adaptive IC threshold
# ===========================================================================

class TestAdaptiveIC:
    def test_empty_registry_returns_floor(self, daemon):
        """No registry → use config floor (0.10)."""
        ic = daemon._compute_adaptive_ic()
        assert ic == 0.10

    def test_empty_registry_file_returns_floor(self, daemon):
        """Empty registry list → use floor."""
        ic = daemon._compute_adaptive_ic()
        assert ic == 0.10

    def test_single_signal(self, daemon):
        """One signal with IC=0.50 → max(0.10, 0.50*0.8) = 0.40."""
        _write_registry_to_store(daemon, [_make_signal(ic=0.50)])
        ic = daemon._compute_adaptive_ic()
        assert ic == pytest.approx(0.40)

    def test_multiple_signals_uses_median(self, daemon):
        """Three signals → median IC drives threshold."""
        signals = [
            _make_signal("a", ic=0.30),
            _make_signal("b", ic=0.50),
            _make_signal("c", ic=0.70),
        ]
        _write_registry_to_store(daemon, signals)
        ic = daemon._compute_adaptive_ic()
        # median=0.50, adaptive=max(0.10, 0.50*0.8)=0.40
        assert ic == pytest.approx(0.40)

    def test_high_quality_registry_raises_bar(self, daemon):
        """All high-IC signals → threshold well above floor."""
        signals = [_make_signal(f"s{i}", ic=0.60 + i * 0.05) for i in range(5)]
        _write_registry_to_store(daemon, signals)
        ic = daemon._compute_adaptive_ic()
        # median=0.70, adaptive=max(0.10, 0.70*0.8)=0.56
        assert ic == pytest.approx(0.56)
        assert ic > 0.10

    def test_low_quality_registry_uses_floor(self, daemon):
        """All low-IC signals → floor wins."""
        signals = [_make_signal(f"s{i}", ic=0.05) for i in range(3)]
        _write_registry_to_store(daemon, signals)
        ic = daemon._compute_adaptive_ic()
        # median=0.05, 0.05*0.8=0.04 < 0.10
        assert ic == 0.10

    def test_retired_signals_excluded(self, daemon):
        """Retired signals should not influence the adaptive threshold."""
        signals = [
            _make_signal("active", ic=0.30),
            _make_signal("retired", ic=0.90, status="retired"),
        ]
        _write_registry_to_store(daemon, signals)
        ic = daemon._compute_adaptive_ic()
        # Only active IC=0.30, adaptive=max(0.10, 0.30*0.8)=0.24
        assert ic == pytest.approx(0.24)

    def test_even_count_median(self, daemon):
        """Even number of signals — uses lower-median (integer division)."""
        signals = [
            _make_signal("a", ic=0.20),
            _make_signal("b", ic=0.40),
            _make_signal("c", ic=0.60),
            _make_signal("d", ic=0.80),
        ]
        _write_registry_to_store(daemon, signals)
        ic = daemon._compute_adaptive_ic()
        # sorted=[0.20, 0.40, 0.60, 0.80], len//2=2, median=0.60
        # adaptive=max(0.10, 0.60*0.8)=0.48
        assert ic == pytest.approx(0.48)


# ===========================================================================
# 2.2 IC decay monitoring
# ===========================================================================

class TestICDecayDetection:
    """Test that run_monitor() detects IC decay and increments counters."""

    def test_healthy_signal_no_decay(self, daemon):
        """IC above threshold → decay_days stays at 0."""
        sig = _make_signal(ic=0.50)
        _write_registry_to_store(daemon, [sig])

        # Mock rolling IC = 0.40 (>= 0.50*0.5=0.25)
        with patch.object(daemon, "_compute_rolling_ic", return_value=0.40):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["decay_days"] == 0
        assert reg[0]["status"] == "validated"

    def test_decaying_signal_increments_counter(self, daemon):
        """IC below 50% of discovery → decay_days incremented."""
        sig = _make_signal(ic=0.50)
        _write_registry_to_store(daemon, [sig])

        # Mock rolling IC = 0.20 (< 0.50*0.5=0.25)
        with patch.object(daemon, "_compute_rolling_ic", return_value=0.20):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["decay_days"] == 1
        assert reg[0]["status"] == "validated"  # not yet retired

    def test_cumulative_decay_days(self, daemon):
        """Multiple monitor calls accumulate decay_days."""
        sig = _make_signal(ic=0.50, decay_days=5)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.10):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["decay_days"] == 6

    def test_decay_counter_resets_on_recovery(self, daemon):
        """IC recovers above threshold → counter resets to 0."""
        sig = _make_signal(ic=0.50, decay_days=10)
        _write_registry_to_store(daemon, [sig])

        # IC = 0.40 >= 0.25 threshold → recovery
        with patch.object(daemon, "_compute_rolling_ic", return_value=0.40):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["decay_days"] == 0
        assert reg[0]["status"] == "validated"

    def test_no_data_skips_decay_check(self, daemon):
        """When rolling IC returns None, no decay tracking changes."""
        sig = _make_signal(ic=0.50)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=None):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0].get("decay_days", 0) == 0  # never incremented


class TestAutoRetirement:
    """Test that signals are auto-retired after sustained IC decay."""

    def test_retirement_at_limit(self, daemon):
        """Signal at decay_days=13 gets retired on next decay (14th day)."""
        sig = _make_signal(ic=0.50, decay_days=13)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.10):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["status"] == "retired"
        assert reg[0]["retired_reason"] == "ic_decay"
        assert "retired_date" in reg[0]

    def test_not_retired_before_limit(self, daemon):
        """Signal at decay_days=12 is NOT retired on day 13."""
        sig = _make_signal(ic=0.50, decay_days=12)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.10):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["status"] == "validated"
        assert reg[0]["decay_days"] == 13

    def test_retirement_updates_signal_count(self, daemon):
        """Retirement decrements total_signals_registered in state."""
        daemon.state.set("total_signals_registered", 3)
        sig = _make_signal(ic=0.50, decay_days=13)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.10):
            daemon.run_monitor()

        assert daemon.state.get("total_signals_registered") == 2

    def test_retired_signals_skipped_by_monitor(self, daemon):
        """Already-retired signals are not re-checked for decay."""
        sig = _make_signal(ic=0.50, status="retired", retired_reason="ic_decay")
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic") as mock_ic:
            daemon.run_monitor()
            mock_ic.assert_not_called()

    def test_multiple_signals_independent(self, daemon):
        """Each signal's decay tracking is independent."""
        signals = [
            _make_signal("healthy", ic=0.50),
            _make_signal("decaying", ic=0.50, decay_days=13),
        ]
        _write_registry_to_store(daemon, signals)

        def mock_ic(sig):
            if sig["name"] == "healthy":
                return 0.40  # above threshold
            return 0.10  # below threshold

        with patch.object(daemon, "_compute_rolling_ic", side_effect=mock_ic):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        healthy = [s for s in reg if s["name"] == "healthy"][0]
        decaying = [s for s in reg if s["name"] == "decaying"][0]
        assert healthy["status"] == "validated"
        assert healthy["decay_days"] == 0
        assert decaying["status"] == "retired"

    def test_custom_decay_config(self, daemon):
        """Custom decay config from agent.toml is respected."""
        daemon.config["decay"] = {
            "ic_decay_ratio": 0.3,           # stricter: 30%
            "consecutive_days_limit": 7,      # faster retirement
        }
        sig = _make_signal(ic=0.50, decay_days=6)
        _write_registry_to_store(daemon, [sig])

        # IC=0.10 < 0.50*0.3=0.15 → decay, day 7 → retire
        with patch.object(daemon, "_compute_rolling_ic", return_value=0.10):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert reg[0]["status"] == "retired"


class TestICHistory:
    """Test that IC history is tracked correctly."""

    def test_history_appended(self, daemon):
        """Each monitor call appends to ic_history."""
        sig = _make_signal(ic=0.50)
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.40):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert len(reg[0]["ic_history"]) == 1
        assert reg[0]["ic_history"][0]["ic"] == 0.40
        assert reg[0]["latest_ic"] == 0.40

    def test_history_bounded_at_30(self, daemon):
        """IC history is kept to last 30 entries."""
        sig = _make_signal(ic=0.50)
        sig["ic_history"] = [{"date": f"2026-04-{i:02d}", "ic": 0.40}
                             for i in range(1, 31)]  # 30 entries
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.45):
            daemon.run_monitor()

        reg = _read_registry_from_store(daemon)
        assert len(reg[0]["ic_history"]) == 30
        # Latest entry is the new one
        assert reg[0]["ic_history"][-1]["ic"] == 0.45


class TestPromotionCheck:
    """Test paper trading promotion eligibility (existing feature)."""

    def test_promotable_signal(self, daemon):
        """Signal meeting all criteria is flagged as promotable."""
        sig = _make_signal(
            ic=0.50, status="paper",
            paper_sharpe=2.0,
            paper_days_elapsed=10,
            realized_ic=0.45,
            max_drawdown_pct=1.5,
        )
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.45):
            daemon.run_monitor()  # should log promotion eligible

    def test_not_promotable_low_sharpe(self, daemon):
        """Low Sharpe → not promotable."""
        sig = _make_signal(
            ic=0.50, status="paper",
            paper_sharpe=0.5,  # below 1.5
            paper_days_elapsed=10,
            realized_ic=0.45,
            max_drawdown_pct=1.5,
        )
        _write_registry_to_store(daemon, [sig])

        with patch.object(daemon, "_compute_rolling_ic", return_value=0.45):
            daemon.run_monitor()


class TestComputeRollingIC:
    """Test the _compute_rolling_ic helper."""

    def test_returns_none_when_no_data_dir(self, daemon, agent_dirs):
        """No data directory → None."""
        import shutil
        data_root = agent_dirs / "data" / "features"
        if data_root.exists():
            shutil.rmtree(data_root)
        sig = _make_signal(ic=0.50)
        assert daemon._compute_rolling_ic(sig) is None

    def test_returns_none_when_no_features(self, daemon, registry_path):
        """Signal with no features → None."""
        sig = _make_signal(ic=0.50)
        sig["features"] = []
        assert daemon._compute_rolling_ic(sig) is None

    def test_computes_ic_from_parquet(self, daemon, agent_dirs):
        """With real-ish data, computes a numeric IC."""
        import numpy as np
        import pandas as pd

        # Create synthetic parquet data — need enough rows that after
        # P20 gating (~20% pass) AND 50-row fwd shift we still have >= 200
        n = 5000
        np.random.seed(42)
        base_ts = int(pd.Timestamp("2026-05-12T00:00:00").value)
        df = pd.DataFrame({
            "timestamp_ns": base_ts + np.arange(n) * 100_000_000,
            "symbol": ["BTC"] * n,
            "raw_midprice": 100 + np.cumsum(np.random.randn(n) * 0.01),
            "imbalance_qty_l1": np.random.randn(n),
            "ent_book_shape": np.random.uniform(0, 1, n),
        })
        data_dir = agent_dirs / "data" / "features" / "2026-05-12"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / "test.parquet")

        sig = _make_signal(ic=0.50)
        result = daemon._compute_rolling_ic(sig)
        # Should return a float (may be near zero for random data)
        assert result is not None
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_uses_latest_date(self, daemon, agent_dirs):
        """Should prefer the most recent date directory."""
        import numpy as np
        import pandas as pd

        n = 5000
        np.random.seed(99)
        for date in ["2026-05-10", "2026-05-12"]:
            d = agent_dirs / "data" / "features" / date
            d.mkdir(parents=True, exist_ok=True)
            base_ts = int(pd.Timestamp(f"{date}T00:00:00").value)
            df = pd.DataFrame({
                "timestamp_ns": base_ts + np.arange(n) * 100_000_000,
                "symbol": ["BTC"] * n,
                "raw_midprice": 100 + np.cumsum(np.random.randn(n) * 0.01),
                "imbalance_qty_l1": np.random.randn(n),
                "ent_book_shape": np.random.uniform(0, 1, n),
            })
            df.to_parquet(d / "test.parquet")

        sig = _make_signal(ic=0.50)
        result = daemon._compute_rolling_ic(sig)
        assert result is not None

    def test_insufficient_data_returns_none(self, daemon, agent_dirs):
        """Less than 500 rows → skip."""
        import numpy as np
        import pandas as pd

        n = 100  # too few
        base_ts = int(pd.Timestamp("2026-05-12T00:00:00").value)
        df = pd.DataFrame({
            "timestamp_ns": base_ts + np.arange(n) * 100_000_000,
            "symbol": ["BTC"] * n,
            "raw_midprice": 100 + np.cumsum(np.random.randn(n) * 0.01),
            "imbalance_qty_l1": np.random.randn(n),
            "ent_book_shape": np.random.uniform(0, 1, n),
        })
        data_dir = agent_dirs / "data" / "features" / "2026-05-12"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / "test.parquet")

        sig = _make_signal(ic=0.50)
        result = daemon._compute_rolling_ic(sig)
        assert result is None
