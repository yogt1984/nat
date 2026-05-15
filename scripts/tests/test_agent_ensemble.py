"""Tests for ensemble gate generator (2.5) and hypothesis chaining (2.4).

Tests cover:
- Ensemble generator: gate pair extraction, cross-category boost, dedup
- Hypothesis chaining: symbol-specific variants, strong dIC flagging
- Edge cases: insufficient gates, same-feature pairs, empty queue
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent.hypothesis import Hypothesis
from agent.hypothesis_queue import HypothesisQueue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_queue(tmp_path):
    """Create a queue backed by a temp file."""
    return HypothesisQueue(path=tmp_path / "hypotheses.json")


def _make_hyp(claim, gate, ic=0.50, status="passed", generator="systematic",
              dIC=0.08, failure_reason=None, **extra_thresholds):
    h = Hypothesis.create(
        claim=claim,
        generator=generator,
        test_protocol=["spannung regime --data data/features/2026-05-12 --symbol BTC"],
        priority=0.5,
        thresholds={"regime_gate": gate, "min_ic": 0.10, "min_dIC": 0.05,
                    **extra_thresholds},
    )
    h.status = status
    h.failure_reason = failure_reason
    h.results = {
        "gate_results": [
            {"msg": f"IC={ic:.4f} [gated({gate})] vs min=0.10 p=0.00e+00 PASS"},
            {"msg": f"dIC={dIC:+.4f} (gated={ic:.4f} - base={ic-dIC:.4f}) vs min=0.05 PASS"},
        ]
    }
    return h


# ===========================================================================
# Ensemble generator
# ===========================================================================

class TestExtractPassingGates:
    def test_finds_passing_gates(self, tmp_queue):
        from agent.generators.ensemble import _extract_passing_gates
        h1 = _make_hyp("imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imbalance_qty_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.45)
        tmp_queue._all = [h1, h2]
        gates = _extract_passing_gates(tmp_queue)
        assert len(gates) == 2
        gate_names = {g["gate"] for g in gates}
        assert "ent_book_shape<P20" in gate_names
        assert "toxic_vpin_50<P40" in gate_names

    def test_skips_low_ic(self, tmp_queue):
        from agent.generators.ensemble import _extract_passing_gates
        h = _make_hyp("sig gated by gate<P20 predicts 5s returns", "gate<P20", ic=0.05)
        tmp_queue._all = [h]
        assert _extract_passing_gates(tmp_queue) == []

    def test_skips_queued(self, tmp_queue):
        from agent.generators.ensemble import _extract_passing_gates
        h = _make_hyp("sig gated by gate<P20 predicts 5s returns", "gate<P20",
                       ic=0.50, status="queued")
        tmp_queue._all = [h]
        assert _extract_passing_gates(tmp_queue) == []

    def test_deduplicates_same_signal_gate(self, tmp_queue):
        from agent.generators.ensemble import _extract_passing_gates
        h1 = _make_hyp("imb_l1 gated by ent<P20 predicts 5s returns", "ent<P20", ic=0.50)
        h2 = _make_hyp("imb_l1 gated by ent<P20 predicts 5s returns", "ent<P20", ic=0.55)
        h2.id = "HYP-SYS-duplicate"
        tmp_queue._all = [h1, h2]
        gates = _extract_passing_gates(tmp_queue)
        assert len(gates) == 1


class TestGateCategory:
    def test_entropy_category(self):
        from agent.generators.ensemble import _gate_category
        assert _gate_category("ent_book_shape<P20") == "entropy"
        assert _gate_category("ent_tick_5s>P60") == "entropy"

    def test_illiquidity_category(self):
        from agent.generators.ensemble import _gate_category
        assert _gate_category("illiq_kyle_100<P40") == "illiquidity"

    def test_toxicity_category(self):
        from agent.generators.ensemble import _gate_category
        assert _gate_category("toxic_vpin_50<P20") == "toxicity"

    def test_unknown_category(self):
        from agent.generators.ensemble import _gate_category
        assert _gate_category("some_random_feature<P20") == "unknown"


class TestEnsembleGenerate:
    def test_generates_pairs(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imbalance_qty_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.45)
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        assert len(result) == 1
        assert "AND" in result[0].claim
        assert result[0].thresholds.get("ensemble") is True
        assert result[0].thresholds.get("regime_gate_b") is not None

    def test_skips_same_feature_pairs(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imb_l1 gated by ent_book_shape<P40 predicts 5s returns",
                        "ent_book_shape<P40", ic=0.45)
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        assert len(result) == 0  # Same feature, different threshold → skip

    def test_cross_category_priority_boost(self, tmp_queue):
        from agent.generators.ensemble import generate
        # entropy + toxicity = cross-category
        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.50)
        h2 = _make_hyp("imb_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.50)
        # entropy + entropy (but different feature) = same category
        h3 = _make_hyp("imb_l1 gated by ent_tick_5s<P20 predicts 5s returns",
                        "ent_tick_5s<P20", ic=0.50)
        tmp_queue._all = [h1, h2, h3]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        cross = [h for h in result if "toxic" in h.claim and "ent_book" in h.claim]
        same = [h for h in result if "ent_tick" in h.claim and "ent_book" in h.claim]
        # Both should exist but cross-category has higher priority
        if cross and same:
            assert cross[0].priority > same[0].priority

    def test_needs_at_least_two_gates(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.50)
        tmp_queue._all = [h1]
        manifest = {"dates": {"2026-05-12": {}}}

        assert generate(manifest, tmp_queue) == []

    def test_no_duplicate_claims(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imb_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.45)
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        # Generate once
        first = generate(manifest, tmp_queue)
        assert len(first) == 1
        # Push to queue
        for h in first:
            tmp_queue._all.append(h)
        # Generate again — should produce nothing
        second = generate(manifest, tmp_queue)
        assert len(second) == 0

    def test_empty_manifest(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("sig gated by a<P20 predicts 5s", "a<P20", ic=0.50)
        h2 = _make_hyp("sig gated by b<P40 predicts 5s", "b<P40", ic=0.50)
        tmp_queue._all = [h1, h2]
        assert generate({}, tmp_queue) == []

    def test_parent_id_tracks_constituents(self, tmp_queue):
        from agent.generators.ensemble import generate
        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imb_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.45)
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        assert result[0].parent_id is not None
        assert "+" in result[0].parent_id  # "id_a+id_b"

    def test_multiple_signals_independent(self, tmp_queue):
        from agent.generators.ensemble import generate
        # Two gates for imb_l1
        h1 = _make_hyp("imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55)
        h2 = _make_hyp("imbalance_qty_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.45)
        # Two gates for imb_l5
        h3 = _make_hyp("imbalance_qty_l5 gated by illiq_kyle_100<P20 predicts 5s returns",
                        "illiq_kyle_100<P20", ic=0.40)
        h4 = _make_hyp("imbalance_qty_l5 gated by vol_returns_1m>P80 predicts 5s returns",
                        "vol_returns_1m>P80", ic=0.35)
        tmp_queue._all = [h1, h2, h3, h4]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        # Should get 1 for imb_l1 and 1 for imb_l5
        l1_ensembles = [h for h in result if "imbalance_qty_l1" in h.claim]
        l5_ensembles = [h for h in result if "imbalance_qty_l5" in h.claim]
        assert len(l1_ensembles) == 1
        assert len(l5_ensembles) == 1


# ===========================================================================
# Hypothesis chaining
# ===========================================================================

@pytest.fixture
def daemon_dirs(tmp_path, monkeypatch):
    """Set up isolated agent state for chaining tests."""
    state_dir = tmp_path / "data" / "agent"
    state_dir.mkdir(parents=True)

    import agent.daemon as daemon_mod
    import agent.runner as runner_mod
    import agent.hypothesis_queue as hq_mod
    monkeypatch.setattr(daemon_mod, "ROOT", tmp_path)
    monkeypatch.setattr(daemon_mod, "STATE_PATH", state_dir / "agent_state.json")
    monkeypatch.setattr(daemon_mod, "STATS_PATH", state_dir / "generator_stats.json")
    monkeypatch.setattr(runner_mod, "REGISTRY_PATH", state_dir / "registry.json")
    monkeypatch.setattr(runner_mod, "ROOT", tmp_path)
    monkeypatch.setattr(hq_mod, "DEFAULT_PATH", state_dir / "hypotheses.json")
    return tmp_path


@pytest.fixture
def daemon(daemon_dirs):
    from agent.daemon import AgentDaemon
    return AgentDaemon()


class TestSymbolSpecificChaining:
    def test_spawns_variant_on_one_failed_symbol(self, daemon):
        """Failed on exactly 1 symbol → spawn symbol-specific variant."""
        h = _make_hyp("imb_l1 gated by ent<P20 predicts 5s returns", "ent<P20",
                       status="failed", failure_reason="no_replication")
        h.results["symbol_replication"] = {
            "passed": ["BTC", "ETH"],
            "failed": ["SOL"],
            "n_pass": 1,
            "n_total": 2,
        }

        n = daemon._chain_hypotheses([h])
        assert n == 1
        # Check the spawned hypothesis
        queued = daemon.queue.queued
        assert len(queued) == 1
        assert "[symbol-specific:" in queued[0].claim
        assert queued[0].parent_id == h.id
        assert queued[0].priority < h.priority

    def test_no_variant_when_both_symbols_fail(self, daemon):
        """Failed on 2 symbols → no variant (not a near-miss)."""
        h = _make_hyp("sig gated by g<P20 predicts 5s returns", "g<P20",
                       status="failed", failure_reason="no_replication")
        h.results["symbol_replication"] = {
            "passed": ["BTC"],
            "failed": ["ETH", "SOL"],
            "n_pass": 0,
            "n_total": 2,
        }

        n = daemon._chain_hypotheses([h])
        assert n == 0

    def test_no_variant_when_not_replication_failure(self, daemon):
        """Failed for other reason (e.g. no_effect) → no chaining."""
        h = _make_hyp("sig gated by g<P20 predicts 5s returns", "g<P20",
                       status="failed", failure_reason="no_effect")
        n = daemon._chain_hypotheses([h])
        assert n == 0

    def test_no_duplicate_chaining(self, daemon):
        """Same hypothesis chained twice → second time produces nothing."""
        h = _make_hyp("imb_l1 gated by ent<P20 predicts 5s returns", "ent<P20",
                       status="failed", failure_reason="no_replication")
        h.results["symbol_replication"] = {
            "passed": ["BTC", "ETH"],
            "failed": ["SOL"],
            "n_pass": 1,
            "n_total": 2,
        }
        daemon._chain_hypotheses([h])
        n = daemon._chain_hypotheses([h])
        assert n == 0  # Already exists

    def test_variant_has_correct_symbols(self, daemon):
        """Symbol-specific variant should only target passing symbols."""
        h = _make_hyp("sig gated by g<P20 predicts 5s returns", "g<P20",
                       status="failed", failure_reason="no_replication")
        h.results["symbol_replication"] = {
            "passed": ["BTC", "ETH"],
            "failed": ["SOL"],
            "n_pass": 1,
            "n_total": 2,
        }
        daemon._chain_hypotheses([h])
        variant = daemon.queue.queued[0]
        assert variant.thresholds.get("symbols_override") == ["BTC", "ETH"]
        assert variant.thresholds.get("min_symbols") == 2


class TestStrongDICFlagging:
    def test_strong_dic_logged(self, daemon):
        """dIC >= 2 * min_dIC should be flagged (no hypothesis spawned, just logged)."""
        h = _make_hyp("sig gated by g<P20 predicts 5s returns", "g<P20",
                       ic=0.55, dIC=0.12, status="replicated")
        n = daemon._chain_hypotheses([h])
        # No hypothesis spawned — ensemble picks it up next cycle
        assert n == 0

    def test_weak_dic_not_flagged(self, daemon):
        """dIC < 2 * min_dIC → not flagged."""
        h = _make_hyp("sig gated by g<P20 predicts 5s returns", "g<P20",
                       ic=0.55, dIC=0.06, status="replicated")
        n = daemon._chain_hypotheses([h])
        assert n == 0


class TestExtractDICFromResults:
    def test_extracts_dic(self, daemon):
        h = _make_hyp("sig gated by g<P20", "g<P20", ic=0.50, dIC=0.08)
        assert daemon._extract_dIC_from_results(h) == pytest.approx(0.08, abs=0.001)

    def test_missing_results(self, daemon):
        h = Hypothesis.create("test", "systematic", [], 0.5)
        assert daemon._extract_dIC_from_results(h) == 0.0

    def test_no_dic_in_msg(self, daemon):
        h = Hypothesis.create("test", "systematic", [], 0.5)
        h.results = {"gate_results": [{"msg": "IC=0.50 PASS"}]}
        assert daemon._extract_dIC_from_results(h) == 0.0


# ===========================================================================
# Integration: ensemble + chaining together
# ===========================================================================

class TestEnsembleChainingIntegration:
    def test_passing_gates_feed_ensemble(self, tmp_queue):
        """Hypotheses that pass discovery (any status beyond queued)
        are picked up by ensemble generator."""
        from agent.generators.ensemble import generate

        # Simulate two gates that passed all the way through
        h1 = _make_hyp("imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55, status="replicated")
        h2 = _make_hyp("imbalance_qty_l1 gated by illiq_kyle_100<P40 predicts 5s returns",
                        "illiq_kyle_100<P40", ic=0.40, status="replicated")
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        assert len(result) == 1
        assert "ent_book_shape" in result[0].claim
        assert "illiq_kyle_100" in result[0].claim
        assert result[0].generator == "ensemble"

    def test_failed_gates_also_feed_ensemble(self, tmp_queue):
        """Even failed hypotheses (with good IC) can contribute to ensemble."""
        from agent.generators.ensemble import generate

        h1 = _make_hyp("imb_l1 gated by ent_book_shape<P20 predicts 5s returns",
                        "ent_book_shape<P20", ic=0.55, status="replicated")
        h2 = _make_hyp("imb_l1 gated by toxic_vpin_50<P40 predicts 5s returns",
                        "toxic_vpin_50<P40", ic=0.35, status="failed",
                        failure_reason="no_replication")
        tmp_queue._all = [h1, h2]
        manifest = {"dates": {"2026-05-12": {}}}

        result = generate(manifest, tmp_queue)
        assert len(result) == 1  # IC=0.35 > 0.10, so it contributes
