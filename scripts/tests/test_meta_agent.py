"""Tests for the Meta-Agent orchestrator (4.4).

Tests cover:
- Agent-level stats: loading, aggregation, Thompson weights
- Budget allocation: Thompson sampling, normalization
- Cross-agent correlation: detection, flagging
- Portfolio: risk parity weights, metrics, redundancy filtering
- Promotion: Sharpe evaluation, consecutive days
- MetaAgent: lifecycle, config, state isolation
- Backward compatibility: all 3 research agents unaffected
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    """Redirect all meta-agent paths to tmp_path."""
    import agent.meta_daemon as meta_mod

    state_dir = tmp_path / "data" / "agent_meta"
    state_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(meta_mod, "META_STATE_PATH", state_dir / "meta_state.json")
    monkeypatch.setattr(meta_mod, "AGENT_STATS_PATH", state_dir / "agent_stats.json")
    monkeypatch.setattr(meta_mod, "CORRELATION_PATH", state_dir / "correlation.json")
    monkeypatch.setattr(meta_mod, "PORTFOLIO_PATH", state_dir / "portfolio.json")

    # Create isolated registry paths
    for agent_name in ["agent", "agent_mf", "agent_macro"]:
        d = tmp_path / "data" / agent_name
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(meta_mod, "AGENT_REGISTRY_PATHS", {
        "microstructure": tmp_path / "data" / "agent" / "registry.json",
        "medium_freq": tmp_path / "data" / "agent_mf" / "registry.json",
        "macro": tmp_path / "data" / "agent_macro" / "registry.json",
    })
    monkeypatch.setattr(meta_mod, "AGENT_GEN_STATS_PATHS", {
        "microstructure": tmp_path / "data" / "agent" / "generator_stats.json",
        "medium_freq": tmp_path / "data" / "agent_mf" / "generator_stats.json",
        "macro": tmp_path / "data" / "agent_macro" / "generator_stats.json",
    })
    monkeypatch.setattr(meta_mod, "AGENT_STATE_PATHS", {
        "microstructure": tmp_path / "data" / "agent" / "agent_state.json",
        "medium_freq": tmp_path / "data" / "agent_mf" / "agent_state.json",
        "macro": tmp_path / "data" / "agent_macro" / "agent_state.json",
    })

    return tmp_path


@pytest.fixture
def meta_agent(tmp_state):
    from agent.meta_daemon import MetaAgent
    return MetaAgent()


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ===========================================================================
# Agent stats tests
# ===========================================================================

class TestAgentStats:
    def test_load_empty_stats(self, meta_agent, tmp_state):
        stats = meta_agent.update_agent_stats()
        for agent_name in ["microstructure", "medium_freq", "macro"]:
            assert stats[agent_name]["attempts"] == 0
            assert stats[agent_name]["successes"] == 0

    def test_load_existing_stats(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        stats_path = meta_mod.AGENT_GEN_STATS_PATHS["microstructure"]
        _write_json(stats_path, {
            "systematic": {"attempts": 10, "successes": 3},
            "spectral": {"attempts": 5, "successes": 1},
        })
        stats = meta_agent.update_agent_stats()
        assert stats["microstructure"]["attempts"] == 15
        assert stats["microstructure"]["successes"] == 4

    def test_compute_agent_level_weight(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        stats_path = meta_mod.AGENT_GEN_STATS_PATHS["macro"]
        _write_json(stats_path, {
            "funding_meanrev": {"attempts": 20, "successes": 5},
        })
        stats = meta_agent.update_agent_stats()
        # weight = (5+1) / (20+2) = 6/22 ≈ 0.2727
        assert abs(stats["macro"]["weight"] - 6 / 22) < 0.01

    def test_separate_stats_per_agent(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_GEN_STATS_PATHS["microstructure"], {
            "systematic": {"attempts": 10, "successes": 5},
        })
        _write_json(meta_mod.AGENT_GEN_STATS_PATHS["medium_freq"], {
            "momentum": {"attempts": 8, "successes": 2},
        })
        stats = meta_agent.update_agent_stats()
        assert stats["microstructure"]["attempts"] == 10
        assert stats["medium_freq"]["attempts"] == 8
        assert stats["macro"]["attempts"] == 0

    def test_save_and_reload_roundtrip(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_GEN_STATS_PATHS["microstructure"], {
            "systematic": {"attempts": 10, "successes": 3},
        })
        meta_agent.update_agent_stats()
        # Verify persisted
        with open(meta_mod.AGENT_STATS_PATH) as f:
            loaded = json.load(f)
        assert loaded["microstructure"]["attempts"] == 10


# ===========================================================================
# Budget allocation tests
# ===========================================================================

class TestBudgetAllocation:
    def test_allocate_equal_when_no_data(self, meta_agent):
        stats = {
            "microstructure": {"attempts": 0, "successes": 0, "weight": 0.5},
            "medium_freq": {"attempts": 0, "successes": 0, "weight": 0.5},
            "macro": {"attempts": 0, "successes": 0, "weight": 0.5},
        }
        budget = meta_agent.allocate_budget(stats)
        for share in budget.values():
            assert abs(share - 1.0 / 3) < 0.01

    def test_allocate_favors_higher_hit_rate(self, meta_agent):
        stats = {
            "microstructure": {"attempts": 20, "successes": 10, "weight": 0.5},
            "medium_freq": {"attempts": 20, "successes": 2, "weight": 0.1364},
            "macro": {"attempts": 20, "successes": 2, "weight": 0.1364},
        }
        budget = meta_agent.allocate_budget(stats)
        assert budget["microstructure"] > budget["medium_freq"]
        assert budget["microstructure"] > budget["macro"]

    def test_allocate_sums_to_one(self, meta_agent):
        stats = {
            "microstructure": {"weight": 0.3},
            "medium_freq": {"weight": 0.5},
            "macro": {"weight": 0.2},
        }
        budget = meta_agent.allocate_budget(stats)
        assert abs(sum(budget.values()) - 1.0) < 0.01

    def test_allocate_with_one_agent_zero_attempts(self, meta_agent):
        stats = {
            "microstructure": {"weight": 0.6},
            "medium_freq": {"weight": 0.5},
            "macro": {"weight": 0.5},
        }
        budget = meta_agent.allocate_budget(stats)
        assert all(v > 0 for v in budget.values())


# ===========================================================================
# Cross-correlation tests
# ===========================================================================

class TestCrossCorrelation:
    def test_empty_registries_no_flags(self, meta_agent):
        flagged = meta_agent.monitor_cross_correlation()
        assert flagged == []

    def test_single_agent_signals_skipped(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["microstructure"], [
            {"name": "sig_a", "features": ["f1"], "status": "validated"},
            {"name": "sig_b", "features": ["f2"], "status": "validated"},
        ])
        flagged = meta_agent.monitor_cross_correlation()
        # Both signals from same agent — no cross-agent pairs to check
        assert flagged == []

    def test_cross_agent_pair_checked(self, meta_agent, tmp_state):
        """Two signals from different agents — correlation computed (or None if no data)."""
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["microstructure"], [
            {"name": "micro_sig", "features": ["f1"], "status": "validated"},
        ])
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["macro"], [
            {"name": "macro_sig", "features": ["f2"], "status": "validated"},
        ])
        flagged = meta_agent.monitor_cross_correlation()
        # No data available → correlation is None → not flagged
        assert len(flagged) == 0
        # But correlation file should exist
        with open(meta_mod.CORRELATION_PATH) as f:
            data = json.load(f)
        assert len(data["pairs"]) == 1
        assert data["pairs"][0]["signal_a"] == "micro_sig"
        assert data["pairs"][0]["agent_b"] == "macro"

    def test_retired_signals_excluded(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["microstructure"], [
            {"name": "active", "features": ["f1"], "status": "validated"},
            {"name": "old", "features": ["f2"], "status": "retired"},
        ])
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["macro"], [
            {"name": "macro_sig", "features": ["f3"], "status": "validated"},
        ])
        flagged = meta_agent.monitor_cross_correlation()
        with open(meta_mod.CORRELATION_PATH) as f:
            data = json.load(f)
        # Only 1 cross-agent pair (active × macro_sig), retired excluded
        assert len(data["pairs"]) == 1

    def test_correlation_matrix_structure(self, meta_agent, tmp_state):
        import agent.meta_daemon as meta_mod
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["microstructure"], [
            {"name": "s1", "features": ["f1"], "status": "validated"},
        ])
        _write_json(meta_mod.AGENT_REGISTRY_PATHS["medium_freq"], [
            {"name": "s2", "features": ["f2"], "status": "validated"},
        ])
        meta_agent.monitor_cross_correlation()
        with open(meta_mod.CORRELATION_PATH) as f:
            data = json.load(f)
        assert "computed_at" in data
        assert "pairs" in data
        assert "flagged" in data


# ===========================================================================
# Portfolio tests
# ===========================================================================

class TestPortfolio:
    def test_risk_parity_weights_inverse_vol(self):
        from agent.meta_portfolio import compute_risk_parity_weights
        signals = [
            {"ic_history": [0.5, 0.5, 0.5, 0.5]},  # low vol → higher weight
            {"ic_history": [0.1, 0.9, 0.1, 0.9]},  # high vol → lower weight
        ]
        weights = compute_risk_parity_weights(signals)
        assert len(weights) == 2
        assert weights[0] > weights[1]
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_equal_weights_fallback(self):
        from agent.meta_portfolio import compute_risk_parity_weights
        signals = [
            {"name": "a"},  # no ic_history
            {"name": "b"},
            {"name": "c"},
        ]
        weights = compute_risk_parity_weights(signals)
        for w in weights:
            assert abs(w - 1.0 / 3) < 1e-6

    def test_filter_redundant_keeps_higher_ic(self):
        from agent.meta_portfolio import filter_redundant_signals
        signals = [
            {"name": "high", "expected_ic": 0.5},
            {"name": "low", "expected_ic": 0.2},
            {"name": "other", "expected_ic": 0.3},
        ]
        flagged = [{"signal_a": "high", "signal_b": "low", "correlation": 0.85}]
        filtered = filter_redundant_signals(signals, flagged)
        names = [s["name"] for s in filtered]
        assert "high" in names
        assert "low" not in names
        assert "other" in names

    def test_portfolio_metrics_weighted_ic(self):
        from agent.meta_portfolio import compute_portfolio_metrics
        signals = [
            {"expected_ic": 0.4},
            {"expected_ic": 0.2},
        ]
        weights = [0.6, 0.4]
        metrics = compute_portfolio_metrics(signals, weights)
        # 0.6*0.4 + 0.4*0.2 = 0.24 + 0.08 = 0.32
        assert abs(metrics["portfolio_ic"] - 0.32) < 1e-4

    def test_effective_n_computation(self):
        from agent.meta_portfolio import compute_portfolio_metrics
        signals = [{"expected_ic": 0.3}] * 4
        weights = [0.25] * 4
        metrics = compute_portfolio_metrics(signals, weights)
        # effective_n = 1 / (4 * 0.25^2) = 1 / 0.25 = 4.0
        assert abs(metrics["effective_n"] - 4.0) < 0.1

    def test_min_signals_check(self, meta_agent, tmp_state):
        """Portfolio empty when fewer signals than min_portfolio_signals."""
        portfolio = meta_agent.assemble_portfolio([])
        assert portfolio["total_signals"] == 0


# ===========================================================================
# Promotion tests
# ===========================================================================

class TestPromotion:
    def test_no_promotion_below_threshold(self):
        from agent.meta_portfolio import evaluate_promotion
        # Noisy IC with mean near zero → low Sharpe
        result = evaluate_promotion([0.05, -0.04, 0.03, -0.06, 0.02, -0.03, 0.01],
                                    paper_sharpe_min=5.0, paper_days=7)
        assert result["recommended"] is False

    def test_promotion_recommended_above_threshold(self):
        from agent.meta_portfolio import evaluate_promotion
        # High, consistent IC → high Sharpe
        result = evaluate_promotion([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                      0.5, 0.5, 0.5],
                                    paper_sharpe_min=0.5, paper_days=3)
        # All identical → std≈0 → Sharpe very high (or infinite)
        # Need some variance for meaningful test
        result2 = evaluate_promotion([0.48, 0.52, 0.49, 0.51, 0.50, 0.49, 0.51],
                                     paper_sharpe_min=1.0, paper_days=3)
        # Mean ~0.5, std ~0.014, Sharpe ~ 0.5/0.014 * sqrt(252) ≈ very high
        assert result2["recommended"] is True

    def test_promotion_requires_consecutive_days(self):
        from agent.meta_portfolio import evaluate_promotion
        result = evaluate_promotion([0.5, 0.5, 0.5],
                                    paper_sharpe_min=1.5, paper_days=7)
        assert result["recommended"] is False
        assert "insufficient data" in result["reason"]


# ===========================================================================
# MetaAgent tests
# ===========================================================================

class TestMetaAgent:
    def test_is_not_research_agent(self):
        from agent.meta_daemon import MetaAgent
        from agent.base import ResearchAgent
        assert not issubclass(MetaAgent, ResearchAgent)

    def test_state_in_agent_meta_dir(self):
        from agent.meta_daemon import META_STATE_PATH, PORTFOLIO_PATH
        assert "agent_meta" in str(META_STATE_PATH)
        assert "agent_meta" in str(PORTFOLIO_PATH)

    def test_run_cycle_calls_all_steps(self, meta_agent, tmp_state):
        calls = []
        meta_agent.update_agent_stats = lambda: (calls.append("stats"),
            {"microstructure": {"weight": 0.5},
             "medium_freq": {"weight": 0.5},
             "macro": {"weight": 0.5}})[1]
        meta_agent.allocate_budget = lambda s: (calls.append("budget"),
            {"microstructure": 0.33, "medium_freq": 0.33, "macro": 0.34})[1]
        meta_agent.monitor_cross_correlation = lambda: (calls.append("corr"), [])[1]
        meta_agent.assemble_portfolio = lambda f: (calls.append("portfolio"), {})[1]
        meta_agent.evaluate_promotions = lambda: (calls.append("promote"), {})[1]
        meta_agent.run_cycle()
        assert calls == ["stats", "budget", "corr", "portfolio", "promote"]

    def test_config_loads_meta_agent_section(self):
        from agent.meta_daemon import load_config
        config = load_config()
        assert config["cycle_interval_s"] == 21600
        assert config["correlation_threshold"] == 0.70

    def test_agents_list_has_three_entries(self):
        from agent.meta_daemon import MetaAgent
        assert len(MetaAgent.AGENTS) == 3

    def test_cli_status_smoke(self, meta_agent, capsys):
        meta_agent.print_status()
        out = capsys.readouterr().out
        assert "Phase:" in out
        assert "Cycles:" in out


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestMetaBackcompat:
    def test_micro_agent_unaffected(self):
        from agent.daemon import MicrostructureAgent, AgentDaemon
        assert AgentDaemon is MicrostructureAgent
        assert MicrostructureAgent.agent_type == "microstructure"

    def test_mf_agent_unaffected(self):
        from agent.mf_daemon import MediumFrequencyAgent
        assert MediumFrequencyAgent.agent_type == "medium_freq"

    def test_macro_agent_unaffected(self):
        from agent.macro_daemon import MacroAgent
        assert MacroAgent.agent_type == "macro"
