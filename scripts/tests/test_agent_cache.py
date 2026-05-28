"""Tests for agent.cache — computation cache for deterministic nat commands.

Skeptical tests: verify correctness, edge cases, TTL expiry, corruption
handling, key stability, and integration with the runner.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


import pytest
from agent.cache import ReportCache, _cache_key, _is_cacheable, CACHEABLE_PREFIXES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path):
    """Fresh temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def cache(cache_dir):
    """ReportCache with short TTL for testing."""
    return ReportCache(cache_dir, ttl_seconds=60)


@pytest.fixture
def sample_report():
    return {
        "baseline_ic_filt_5s": 0.488,
        "single_factors": [
            {"label": "ent_book_shape<P20", "ic_filt_5s": 0.569, "n_obs": 28800}
        ],
        "n_rows": 144000,
    }


# ---------------------------------------------------------------------------
# _is_cacheable
# ---------------------------------------------------------------------------

class TestIsCacheable:
    def test_regime_command_is_cacheable(self):
        assert _is_cacheable(["spannung", "regime", "--data", "x", "--symbol", "BTC"])

    def test_spectral_command_is_cacheable(self):
        assert _is_cacheable(["spannung", "spectral", "--data", "x", "--symbol", "BTC"])

    def test_backtest_command_is_cacheable(self):
        assert _is_cacheable(["spannung", "backtest", "--data", "x", "--symbol", "BTC"])

    def test_profile_scalp_is_cacheable(self):
        assert _is_cacheable(["profile", "scalp", "--symbol", "BTC", "--data", "x"])

    def test_non_cacheable_command(self):
        assert not _is_cacheable(["status"])

    def test_agent_command_not_cacheable(self):
        assert not _is_cacheable(["agent", "status"])

    def test_empty_command_not_cacheable(self):
        assert not _is_cacheable([])

    def test_partial_match_not_cacheable(self):
        """'spannung' alone is not in CACHEABLE_PREFIXES."""
        assert not _is_cacheable(["spannung", "--data", "x"])

    def test_all_prefixes_covered(self):
        """Every entry in CACHEABLE_PREFIXES should be testable."""
        for prefix in CACHEABLE_PREFIXES:
            parts = prefix.split() + ["--data", "x", "--symbol", "BTC"]
            assert _is_cacheable(parts), f"{prefix} should be cacheable"


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        parts = ["spannung", "regime", "--data", "data/features/2026-05-12", "--symbol", "BTC"]
        assert _cache_key(parts) == _cache_key(parts)

    def test_flag_order_invariant(self):
        """--data X --symbol Y should hash the same as --symbol Y --data X."""
        k1 = _cache_key(["spannung", "regime", "--data", "dir1", "--symbol", "BTC"])
        k2 = _cache_key(["spannung", "regime", "--symbol", "BTC", "--data", "dir1"])
        assert k1 == k2

    def test_different_symbols_different_keys(self):
        k1 = _cache_key(["spannung", "regime", "--data", "dir1", "--symbol", "BTC"])
        k2 = _cache_key(["spannung", "regime", "--data", "dir1", "--symbol", "ETH"])
        assert k1 != k2

    def test_different_data_dirs_different_keys(self):
        k1 = _cache_key(["spannung", "regime", "--data", "data/2026-05-12", "--symbol", "BTC"])
        k2 = _cache_key(["spannung", "regime", "--data", "data/2026-05-15", "--symbol", "BTC"])
        assert k1 != k2

    def test_different_commands_different_keys(self):
        k1 = _cache_key(["spannung", "regime", "--data", "dir", "--symbol", "BTC"])
        k2 = _cache_key(["spannung", "backtest", "--data", "dir", "--symbol", "BTC"])
        assert k1 != k2

    def test_key_is_hex_string(self):
        k = _cache_key(["spannung", "regime", "--data", "dir", "--symbol", "BTC"])
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_extra_flags_change_key(self):
        k1 = _cache_key(["spannung", "regime", "--data", "dir", "--symbol", "BTC"])
        k2 = _cache_key(["spannung", "regime", "--data", "dir", "--symbol", "BTC", "--forward-test"])
        assert k1 != k2

    def test_boolean_flag_no_value(self):
        """Boolean flags (no value) should still be captured."""
        k1 = _cache_key(["profile", "scalp", "--forward-test", "--symbol", "BTC"])
        k2 = _cache_key(["profile", "scalp", "--symbol", "BTC"])
        assert k1 != k2


# ---------------------------------------------------------------------------
# ReportCache — basic operations
# ---------------------------------------------------------------------------

class TestReportCacheBasic:
    def test_miss_on_empty_cache(self, cache):
        result = cache.get(["spannung", "regime", "--data", "x", "--symbol", "BTC"])
        assert result is None

    def test_put_then_get(self, cache, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        result = cache.get(cmd)
        assert result == sample_report

    def test_put_returns_none_for_non_cacheable(self, cache, sample_report):
        cmd = ["status"]
        cache.put(cmd, sample_report)
        assert cache.get(cmd) is None

    def test_get_returns_none_for_non_cacheable(self, cache):
        assert cache.get(["agent", "status"]) is None

    def test_different_commands_dont_collide(self, cache, sample_report):
        cmd1 = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cmd2 = ["spannung", "regime", "--data", "x", "--symbol", "ETH"]
        report2 = {"different": True}
        cache.put(cmd1, sample_report)
        cache.put(cmd2, report2)
        assert cache.get(cmd1) == sample_report
        assert cache.get(cmd2) == report2

    def test_overwrite_existing_entry(self, cache, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        new_report = {"updated": True}
        cache.put(cmd, new_report)
        assert cache.get(cmd) == new_report

    def test_flag_order_hits_same_entry(self, cache, sample_report):
        cmd1 = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cmd2 = ["spannung", "regime", "--symbol", "BTC", "--data", "x"]
        cache.put(cmd1, sample_report)
        assert cache.get(cmd2) == sample_report


# ---------------------------------------------------------------------------
# ReportCache — TTL expiry
# ---------------------------------------------------------------------------

class TestReportCacheTTL:
    def test_expired_entry_returns_none(self, cache_dir, sample_report):
        cache = ReportCache(cache_dir, ttl_seconds=1)
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        assert cache.get(cmd) == sample_report
        time.sleep(1.1)
        assert cache.get(cmd) is None

    def test_non_expired_entry_returns_data(self, cache_dir, sample_report):
        cache = ReportCache(cache_dir, ttl_seconds=3600)
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        assert cache.get(cmd) == sample_report

    def test_evict_expired_removes_old_entries(self, cache_dir, sample_report):
        cache = ReportCache(cache_dir, ttl_seconds=1)
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        time.sleep(1.1)
        evicted = cache.evict_expired()
        assert evicted == 1
        # Files should be gone
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_evict_keeps_fresh_entries(self, cache_dir, sample_report):
        cache = ReportCache(cache_dir, ttl_seconds=3600)
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        evicted = cache.evict_expired()
        assert evicted == 0
        assert cache.get(cmd) == sample_report


# ---------------------------------------------------------------------------
# ReportCache — corruption resilience
# ---------------------------------------------------------------------------

class TestReportCacheCorruption:
    def test_corrupted_meta_returns_miss(self, cache, cache_dir, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        # Corrupt the meta file
        key = _cache_key(cmd)
        meta_path = cache_dir / f"{key}.meta.json"
        meta_path.write_text("not json{{{")
        assert cache.get(cmd) is None

    def test_corrupted_data_returns_miss(self, cache, cache_dir, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        key = _cache_key(cmd)
        data_path = cache_dir / f"{key}.json"
        data_path.write_text("corrupted!!!")
        assert cache.get(cmd) is None

    def test_missing_data_file_returns_miss(self, cache, cache_dir, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        key = _cache_key(cmd)
        (cache_dir / f"{key}.json").unlink()
        assert cache.get(cmd) is None

    def test_missing_meta_file_returns_miss(self, cache, cache_dir, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        key = _cache_key(cmd)
        (cache_dir / f"{key}.meta.json").unlink()
        assert cache.get(cmd) is None

    def test_evict_handles_corrupted_meta(self, cache_dir):
        cache = ReportCache(cache_dir, ttl_seconds=1)
        # Write a corrupted meta file
        (cache_dir / "badkey.meta.json").write_text("{{{not json")
        evicted = cache.evict_expired()
        assert evicted == 1  # should clean up without crashing


# ---------------------------------------------------------------------------
# ReportCache — invalidate and clear
# ---------------------------------------------------------------------------

class TestReportCacheInvalidate:
    def test_invalidate_removes_entry(self, cache, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        assert cache.invalidate(cmd) is True
        assert cache.get(cmd) is None

    def test_invalidate_nonexistent_returns_false(self, cache):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        assert cache.invalidate(cmd) is False

    def test_invalidate_doesnt_affect_other_entries(self, cache, sample_report):
        cmd1 = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cmd2 = ["spannung", "regime", "--data", "x", "--symbol", "ETH"]
        cache.put(cmd1, sample_report)
        cache.put(cmd2, {"other": True})
        cache.invalidate(cmd1)
        assert cache.get(cmd1) is None
        assert cache.get(cmd2) == {"other": True}

    def test_clear_removes_all(self, cache, sample_report):
        cmd1 = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cmd2 = ["spannung", "regime", "--data", "x", "--symbol", "ETH"]
        cache.put(cmd1, sample_report)
        cache.put(cmd2, {"other": True})
        count = cache.clear()
        assert count == 2
        assert cache.get(cmd1) is None
        assert cache.get(cmd2) is None

    def test_clear_empty_cache(self, cache):
        assert cache.clear() == 0


# ---------------------------------------------------------------------------
# ReportCache — stats
# ---------------------------------------------------------------------------

class TestReportCacheStats:
    def test_initial_stats(self, cache):
        s = cache.stats
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0
        assert s["entries"] == 0

    def test_stats_after_miss(self, cache):
        cache.get(["spannung", "regime", "--data", "x", "--symbol", "BTC"])
        s = cache.stats
        assert s["misses"] == 1
        assert s["hits"] == 0

    def test_stats_after_hit(self, cache, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        cache.get(cmd)
        s = cache.stats
        assert s["hits"] == 1
        assert s["entries"] == 1

    def test_hit_rate_calculation(self, cache, sample_report):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        cache.get(cmd)  # hit
        cache.get(["spannung", "regime", "--data", "y", "--symbol", "BTC"])  # miss
        s = cache.stats
        assert s["hit_rate"] == pytest.approx(0.5)

    def test_non_cacheable_not_counted(self, cache):
        """Non-cacheable commands should not affect hit/miss stats."""
        cache.get(["status"])
        s = cache.stats
        assert s["hits"] == 0
        assert s["misses"] == 0


# ---------------------------------------------------------------------------
# ReportCache — filesystem edge cases
# ---------------------------------------------------------------------------

class TestReportCacheFilesystem:
    def test_cache_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "deeply" / "nested" / "cache"
        cache = ReportCache(new_dir)
        assert new_dir.exists()

    def test_concurrent_put_same_key(self, cache, sample_report):
        """Two puts with the same key — last write wins."""
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cache.put(cmd, sample_report)
        cache.put(cmd, {"second": True})
        assert cache.get(cmd) == {"second": True}

    def test_large_report(self, cache):
        """Cache should handle large JSON reports."""
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        large_report = {"factors": [{"i": i, "data": "x" * 1000} for i in range(500)]}
        cache.put(cmd, large_report)
        result = cache.get(cmd)
        assert len(result["factors"]) == 500

    def test_unicode_in_report(self, cache):
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        report = {"note": "price \u2191 sharply", "symbols": ["\u20bf", "\u039e"]}
        cache.put(cmd, report)
        assert cache.get(cmd) == report


# ---------------------------------------------------------------------------
# Integration: run_nat_cached
# ---------------------------------------------------------------------------

class TestRunNatCached:
    """Integration tests for run_nat_cached with mocked subprocess."""

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path):
        """Set up a fresh cache for each test."""
        from agent import runner
        self._orig_cache = runner._report_cache
        runner.set_cache(ReportCache(tmp_path / "cache"))
        yield
        runner.set_cache(self._orig_cache)

    def test_cache_miss_runs_command(self):
        from agent.runner import run_nat_cached
        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mock_parse.return_value = {"ic": 0.5}

            result, report = run_nat_cached(
                ["spannung", "regime", "--data", "x", "--symbol", "BTC"],
                symbol="BTC"
            )
            assert result.returncode == 0
            assert report == {"ic": 0.5}
            mock_run.assert_called_once()

    def test_cache_hit_skips_command(self):
        from agent.runner import run_nat_cached, get_cache
        # Pre-populate cache
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        get_cache().put(cmd, {"ic": 0.5})

        with patch("agent.runner.run_nat") as mock_run:
            result, report = run_nat_cached(cmd, symbol="BTC")
            assert result.returncode == 0
            assert report == {"ic": 0.5}
            mock_run.assert_not_called()  # no subprocess spawned

    def test_cache_stores_after_miss(self):
        from agent.runner import run_nat_cached, get_cache
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]

        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0)
            mock_parse.return_value = {"ic": 0.5}
            run_nat_cached(cmd, symbol="BTC")

        # Second call should hit cache
        with patch("agent.runner.run_nat") as mock_run2:
            result, report = run_nat_cached(cmd, symbol="BTC")
            mock_run2.assert_not_called()
            assert report == {"ic": 0.5}

    def test_failed_command_not_cached(self):
        from agent.runner import run_nat_cached, get_cache
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]

        with patch("agent.runner.run_nat") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            result, report = run_nat_cached(cmd, symbol="BTC")
            assert result.returncode == 1
            assert report is None

        # Should not be cached
        assert get_cache().get(cmd) is None

    def test_non_cacheable_always_runs(self):
        from agent.runner import run_nat_cached
        cmd = ["status"]

        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0)
            mock_parse.return_value = None
            run_nat_cached(cmd, symbol="BTC")
            run_nat_cached(cmd, symbol="BTC")
            assert mock_run.call_count == 2  # both calls ran the command

    def test_cached_result_has_synthetic_process(self):
        from agent.runner import run_nat_cached, get_cache
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        get_cache().put(cmd, {"ic": 0.5})

        with patch("agent.runner.run_nat"):
            result, _ = run_nat_cached(cmd, symbol="BTC")
            assert result.returncode == 0
            assert result.stdout == "[cached]"
            assert result.stderr == ""

    def test_none_report_not_cached(self):
        """If parse_report returns None, don't cache it."""
        from agent.runner import run_nat_cached, get_cache
        cmd = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]

        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0)
            mock_parse.return_value = None
            run_nat_cached(cmd, symbol="BTC")

        assert get_cache().get(cmd) is None

    def test_different_symbols_cached_separately(self):
        from agent.runner import run_nat_cached, get_cache
        cmd_btc = ["spannung", "regime", "--data", "x", "--symbol", "BTC"]
        cmd_eth = ["spannung", "regime", "--data", "x", "--symbol", "ETH"]
        get_cache().put(cmd_btc, {"symbol": "BTC"})
        get_cache().put(cmd_eth, {"symbol": "ETH"})

        with patch("agent.runner.run_nat"):
            _, report_btc = run_nat_cached(cmd_btc, symbol="BTC")
            _, report_eth = run_nat_cached(cmd_eth, symbol="ETH")
            assert report_btc["symbol"] == "BTC"
            assert report_eth["symbol"] == "ETH"


# ---------------------------------------------------------------------------
# Integration: cache across full experiment
# ---------------------------------------------------------------------------

class TestCacheIntegration:
    """Verify cache reduces subprocess calls across multiple hypotheses
    sharing the same (date, symbol)."""

    @pytest.fixture(autouse=True)
    def setup_cache(self, tmp_path):
        from agent import runner
        self._orig_cache = runner._report_cache
        runner.set_cache(ReportCache(tmp_path / "cache"))
        yield
        runner.set_cache(self._orig_cache)

    def test_second_hypothesis_hits_cache(self):
        """Two hypotheses on same (date, symbol) should share cached reports."""
        from agent.runner import run_nat_cached, get_cache

        cmd = ["spannung", "regime", "--data", "data/features/2026-05-12", "--symbol", "BTC"]

        call_count = 0
        orig_report = {"baseline_ic_filt_5s": 0.488, "single_factors": [], "n_rows": 144000}

        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0)
            mock_parse.return_value = orig_report

            # First hypothesis — cache miss
            r1, report1 = run_nat_cached(cmd, symbol="BTC")
            assert mock_run.call_count == 1

            # Second hypothesis — cache hit
            r2, report2 = run_nat_cached(cmd, symbol="BTC")
            assert mock_run.call_count == 1  # still 1, no new subprocess

        assert report1 == report2 == orig_report
        stats = get_cache().stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_survives_across_different_gates(self):
        """Different gate hypotheses on same symbol/date share the regime report."""
        from agent.runner import run_nat_cached, get_cache

        # Both use the same spannung regime command (gate is checked post-hoc)
        cmd = ["spannung", "regime", "--data", "data/features/2026-05-12", "--symbol", "BTC"]
        report = {"baseline_ic_filt_5s": 0.488, "single_factors": [
            {"label": "ent_book_shape<P20", "ic_filt_5s": 0.569},
            {"label": "ent_book_shape<P40", "ic_filt_5s": 0.543},
        ]}

        with patch("agent.runner.run_nat") as mock_run, \
             patch("agent.runner.parse_report") as mock_parse:
            mock_run.return_value = MagicMock(returncode=0)
            mock_parse.return_value = report

            # Hypothesis 1: ent_book_shape<P20
            run_nat_cached(cmd, symbol="BTC")
            # Hypothesis 2: ent_book_shape<P40 (same command, different gate check)
            _, r = run_nat_cached(cmd, symbol="BTC")

            assert mock_run.call_count == 1
            assert r["single_factors"][1]["label"] == "ent_book_shape<P40"
