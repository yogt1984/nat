"""
Tests for Task 6.2: Cross-Symbol Consistency Check.

Covers: pairwise ARI, consensus voting, disagreement detection,
edge cases, and validation errors.
"""

import numpy as np
import pytest

from cluster_pipeline.validate import CrossSymbolResult, cross_symbol_consistency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identical_labels(n=100, k=3, n_symbols=3):
    """All symbols have identical labels."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, k, size=n)
    return {f"SYM{i}": labels.copy() for i in range(n_symbols)}


def _random_labels(n=200, k=4, n_symbols=3, seed=99):
    """Each symbol has independently random labels."""
    rng = np.random.default_rng(seed)
    return {f"SYM{i}": rng.integers(0, k, size=n) for i in range(n_symbols)}


def _two_agree_one_disagrees(n=100):
    """BTC and ETH agree perfectly, SOL is random."""
    rng = np.random.default_rng(7)
    shared = rng.integers(0, 3, size=n)
    return {
        "BTC": shared.copy(),
        "ETH": shared.copy(),
        "SOL": rng.integers(0, 3, size=n),
    }


# ---------------------------------------------------------------------------
# TestAgreementMatrix
# ---------------------------------------------------------------------------


class TestAgreementMatrix:
    """Tests for the pairwise ARI agreement matrix."""

    def test_identical_labels_ari_one(self):
        labels = _identical_labels(n=100, k=3, n_symbols=3)
        result = cross_symbol_consistency(labels)
        # All off-diagonal should be 1.0
        n = 3
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert result.agreement_matrix[i, j] == 1.0
                else:
                    assert result.agreement_matrix[i, j] == pytest.approx(1.0)

    def test_random_labels_low_ari(self):
        labels = _random_labels(n=500, k=5, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert result.mean_agreement < 0.1

    def test_matrix_is_symmetric(self):
        labels = _two_agree_one_disagrees(n=100)
        result = cross_symbol_consistency(labels)
        np.testing.assert_array_almost_equal(
            result.agreement_matrix, result.agreement_matrix.T
        )

    def test_diagonal_is_one(self):
        labels = _random_labels(n=100, k=3, n_symbols=4)
        result = cross_symbol_consistency(labels)
        for i in range(4):
            assert result.agreement_matrix[i, i] == 1.0

    def test_matrix_shape(self):
        labels = _random_labels(n=50, k=2, n_symbols=5)
        result = cross_symbol_consistency(labels)
        assert result.agreement_matrix.shape == (5, 5)

    def test_two_agree_one_random(self):
        labels = _two_agree_one_disagrees(n=200)
        result = cross_symbol_consistency(labels)
        # BTC-ETH pair should have ARI ~1.0
        sym_idx = {s: i for i, s in enumerate(result.symbol_names)}
        btc_eth_ari = result.agreement_matrix[sym_idx["BTC"], sym_idx["ETH"]]
        assert btc_eth_ari == pytest.approx(1.0)
        # BTC-SOL and ETH-SOL should be low
        btc_sol_ari = result.agreement_matrix[sym_idx["BTC"], sym_idx["SOL"]]
        assert btc_sol_ari < 0.3

    def test_ari_bounded(self):
        """ARI is bounded in [-0.5, 1.0] for most cases."""
        labels = _random_labels(n=100, k=3, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert np.all(result.agreement_matrix >= -1.0)
        assert np.all(result.agreement_matrix <= 1.0)


# ---------------------------------------------------------------------------
# TestMeanAgreement
# ---------------------------------------------------------------------------


class TestMeanAgreement:
    """Tests for mean agreement and above_random flag."""

    def test_perfect_agreement_mean_one(self):
        labels = _identical_labels(n=50, k=2, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert result.mean_agreement == pytest.approx(1.0)

    def test_random_not_above_random(self):
        labels = _random_labels(n=500, k=5, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert not result.above_random

    def test_identical_above_random(self):
        labels = _identical_labels(n=50, k=3, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert result.above_random

    def test_custom_random_threshold(self):
        labels = _two_agree_one_disagrees(n=200)
        result = cross_symbol_consistency(labels, random_threshold=0.9)
        # Mean will be < 0.9 (since one symbol is random)
        # but let's just check the flag respects threshold
        if result.mean_agreement <= 0.9:
            assert not result.above_random
        else:
            assert result.above_random


# ---------------------------------------------------------------------------
# TestConsensus
# ---------------------------------------------------------------------------


class TestConsensus:
    """Tests for majority-vote consensus labels."""

    def test_perfect_agreement_consensus_matches(self):
        labels = _identical_labels(n=50, k=3, n_symbols=3)
        result = cross_symbol_consistency(labels)
        # Consensus should match any symbol's labels
        any_labels = list(labels.values())[0]
        np.testing.assert_array_equal(result.consensus_labels, any_labels)

    def test_majority_two_of_three(self):
        """When 2/3 agree, consensus follows majority."""
        n = 10
        labels = {
            "BTC": np.array([0, 0, 0, 1, 1, 2, 2, 0, 1, 0]),
            "ETH": np.array([0, 0, 0, 1, 1, 2, 2, 0, 1, 0]),
            "SOL": np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1]),
        }
        result = cross_symbol_consistency(labels)
        # BTC and ETH agree at all bars → consensus = BTC/ETH labels
        expected = np.array([0, 0, 0, 1, 1, 2, 2, 0, 1, 0])
        np.testing.assert_array_equal(result.consensus_labels, expected)

    def test_complete_disagreement_marks_uncertain(self):
        """3 symbols, 3 different labels → uncertain (-1)."""
        n = 5
        labels = {
            "BTC": np.array([0, 0, 0, 0, 0]),
            "ETH": np.array([1, 1, 1, 1, 1]),
            "SOL": np.array([2, 2, 2, 2, 2]),
        }
        result = cross_symbol_consistency(labels)
        np.testing.assert_array_equal(result.consensus_labels, np.full(5, -1))

    def test_disagreement_rate_all_disagree(self):
        labels = {
            "BTC": np.array([0, 0, 0, 0]),
            "ETH": np.array([1, 1, 1, 1]),
            "SOL": np.array([2, 2, 2, 2]),
        }
        result = cross_symbol_consistency(labels)
        assert result.disagreement_rate == 1.0

    def test_disagreement_rate_all_agree(self):
        labels = _identical_labels(n=100, k=2, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert result.disagreement_rate == 0.0

    def test_disagreement_rate_partial(self):
        """Mix of agreement and disagreement bars."""
        labels = {
            "BTC": np.array([0, 0, 1, 0]),
            "ETH": np.array([0, 1, 1, 1]),
            "SOL": np.array([0, 2, 1, 2]),
        }
        # Bar 0: all 0 → consensus 0
        # Bar 1: 0,1,2 → no majority → -1
        # Bar 2: all 1 → consensus 1
        # Bar 3: 0,1,2 → no majority → -1
        result = cross_symbol_consistency(labels)
        expected = np.array([0, -1, 1, -1])
        np.testing.assert_array_equal(result.consensus_labels, expected)
        assert result.disagreement_rate == 0.5

    def test_consensus_length_matches_input(self):
        labels = _random_labels(n=77, k=3, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert len(result.consensus_labels) == 77

    def test_two_symbols_majority(self):
        """With 2 symbols, both must agree for majority (>50%)."""
        labels = {
            "BTC": np.array([0, 0, 1, 1, 0]),
            "ETH": np.array([0, 1, 1, 0, 0]),
        }
        result = cross_symbol_consistency(labels)
        # Bar 0: both 0 → consensus 0
        # Bar 1: 0,1 → no majority (1 is not > 1) → -1
        # Bar 2: both 1 → consensus 1
        # Bar 3: 1,0 → no majority → -1
        # Bar 4: both 0 → consensus 0
        expected = np.array([0, -1, 1, -1, 0])
        np.testing.assert_array_equal(result.consensus_labels, expected)

    def test_four_symbols_majority(self):
        """With 4 symbols, need 3+ to agree."""
        labels = {
            "A": np.array([0, 0, 0]),
            "B": np.array([0, 0, 1]),
            "C": np.array([0, 1, 1]),
            "D": np.array([1, 1, 1]),
        }
        # Bar 0: 0,0,0,1 → 3 zeros, majority=0
        # Bar 1: 0,0,1,1 → tie (2 each), no majority → -1
        # Bar 2: 0,1,1,1 → 3 ones, majority=1
        result = cross_symbol_consistency(labels)
        expected = np.array([0, -1, 1])
        np.testing.assert_array_equal(result.consensus_labels, expected)


# ---------------------------------------------------------------------------
# TestReturnType
# ---------------------------------------------------------------------------


class TestReturnType:
    """Tests for return type and structure."""

    def test_returns_cross_symbol_result(self):
        labels = _identical_labels(n=20, k=2, n_symbols=2)
        result = cross_symbol_consistency(labels)
        assert isinstance(result, CrossSymbolResult)

    def test_has_all_fields(self):
        labels = _random_labels(n=30, k=2, n_symbols=3)
        result = cross_symbol_consistency(labels)
        assert hasattr(result, "agreement_matrix")
        assert hasattr(result, "mean_agreement")
        assert hasattr(result, "above_random")
        assert hasattr(result, "consensus_labels")
        assert hasattr(result, "disagreement_rate")
        assert hasattr(result, "per_symbol_labels")
        assert hasattr(result, "symbol_names")

    def test_symbol_names_sorted(self):
        labels = {"SOL": np.array([0, 1]), "BTC": np.array([0, 1]), "ETH": np.array([0, 1])}
        result = cross_symbol_consistency(labels)
        assert result.symbol_names == ["BTC", "ETH", "SOL"]

    def test_per_symbol_labels_preserved(self):
        original = {"A": np.array([0, 1, 2]), "B": np.array([2, 1, 0])}
        result = cross_symbol_consistency(original)
        np.testing.assert_array_equal(result.per_symbol_labels["A"], [0, 1, 2])
        np.testing.assert_array_equal(result.per_symbol_labels["B"], [2, 1, 0])


# ---------------------------------------------------------------------------
# TestValidationErrors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Tests for input validation."""

    def test_single_symbol_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            cross_symbol_consistency({"BTC": np.array([0, 1, 2])})

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            cross_symbol_consistency({})

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="different lengths"):
            cross_symbol_consistency({
                "BTC": np.array([0, 1, 2]),
                "ETH": np.array([0, 1]),
            })

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cross_symbol_consistency({
                "BTC": np.array([], dtype=int),
                "ETH": np.array([], dtype=int),
            })


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_bar(self):
        labels = {"BTC": np.array([0]), "ETH": np.array([0]), "SOL": np.array([0])}
        result = cross_symbol_consistency(labels)
        assert result.consensus_labels[0] == 0
        assert result.disagreement_rate == 0.0

    def test_binary_labels(self):
        labels = {
            "BTC": np.array([0, 0, 1, 1, 0]),
            "ETH": np.array([0, 0, 1, 1, 0]),
            "SOL": np.array([0, 0, 1, 1, 0]),
        }
        result = cross_symbol_consistency(labels)
        assert result.mean_agreement == pytest.approx(1.0)

    def test_many_states(self):
        """Works with many unique label values."""
        rng = np.random.default_rng(123)
        labels = {f"S{i}": rng.integers(0, 20, size=200) for i in range(3)}
        result = cross_symbol_consistency(labels)
        assert result.agreement_matrix.shape == (3, 3)
        assert 0 <= result.disagreement_rate <= 1.0

    def test_negative_labels(self):
        """Handles negative label values (e.g., -1 for uncertain)."""
        labels = {
            "BTC": np.array([-1, 0, 1, -1, 0]),
            "ETH": np.array([-1, 0, 1, -1, 0]),
            "SOL": np.array([-1, 0, 1, -1, 0]),
        }
        result = cross_symbol_consistency(labels)
        assert result.mean_agreement == pytest.approx(1.0)

    def test_large_n_symbols(self):
        """Works with many symbols."""
        rng = np.random.default_rng(55)
        shared = rng.integers(0, 3, size=100)
        labels = {f"S{i}": shared.copy() for i in range(10)}
        result = cross_symbol_consistency(labels)
        assert result.agreement_matrix.shape == (10, 10)
        assert result.mean_agreement == pytest.approx(1.0)

    def test_all_same_label(self):
        """All bars are same state for all symbols."""
        labels = {
            "BTC": np.zeros(50, dtype=int),
            "ETH": np.zeros(50, dtype=int),
            "SOL": np.zeros(50, dtype=int),
        }
        result = cross_symbol_consistency(labels)
        # ARI is 1.0 when labels are identical (even if only one cluster)
        assert result.consensus_labels[0] == 0
        assert result.disagreement_rate == 0.0


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        labels = _random_labels(n=100, k=3, n_symbols=3, seed=42)
        r1 = cross_symbol_consistency(labels)
        r2 = cross_symbol_consistency(labels)
        np.testing.assert_array_equal(r1.agreement_matrix, r2.agreement_matrix)
        np.testing.assert_array_equal(r1.consensus_labels, r2.consensus_labels)
        assert r1.mean_agreement == r2.mean_agreement
        assert r1.disagreement_rate == r2.disagreement_rate

    def test_symbol_order_doesnt_matter(self):
        """Result is consistent regardless of dict insertion order."""
        labels_v1 = {"BTC": np.array([0, 1, 0]), "ETH": np.array([0, 1, 1])}
        labels_v2 = {"ETH": np.array([0, 1, 1]), "BTC": np.array([0, 1, 0])}
        r1 = cross_symbol_consistency(labels_v1)
        r2 = cross_symbol_consistency(labels_v2)
        assert r1.mean_agreement == r2.mean_agreement
        np.testing.assert_array_equal(r1.consensus_labels, r2.consensus_labels)
