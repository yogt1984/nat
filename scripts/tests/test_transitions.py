"""
Skeptical tests for cluster_pipeline.transitions — empirical transition modeling.

Test philosophy:
  - Known sequences with hand-computable transition matrices
  - Row-stochastic property (rows sum to 1)
  - Duration distributions match hand-counted run lengths
  - Edge cases: single state, alternating, single bar, long runs
  - Sparse/non-contiguous state IDs
  - Determinism: same input → same output
  - Validation: invalid inputs rejected
  - Consistency: mean_duration ≈ 1/(1 - self_transition_rate) for geometric
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_pipeline.transitions import (
    TransitionModel,
    empirical_transitions,
    _compute_duration_distributions,
)


# ===========================================================================
# Row Stochastic Property
# ===========================================================================


class TestRowStochastic:
    """Every row in the transition matrix must sum to 1.0."""

    def test_two_states(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 1])
        model = empirical_transitions(labels)
        row_sums = model.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_three_states(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        model = empirical_transitions(labels)
        row_sums = model.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_five_states(self):
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 5, size=1000)
        model = empirical_transitions(labels)
        row_sums = model.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_single_state(self):
        labels = np.array([0, 0, 0, 0, 0])
        model = empirical_transitions(labels)
        np.testing.assert_allclose(model.matrix, [[1.0]], atol=1e-10)

    def test_very_long_sequence(self):
        rng = np.random.RandomState(123)
        labels = rng.randint(0, 3, size=100_000)
        model = empirical_transitions(labels)
        row_sums = model.matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ===========================================================================
# Known Transition Probabilities
# ===========================================================================


class TestKnownTransitions:
    """Hand-computable sequences with known transition matrices."""

    def test_perfect_persistence(self):
        """[0,0,0,0,0,1,1,1,1,1] → T[0,0]=0.8, T[0,1]=0.2, T[1,1]=1.0."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        model = empirical_transitions(labels)
        # State 0: 4 self-transitions + 1 transition to 1 = 5 total from state 0
        # T[0,0] = 4/5 = 0.8, T[0,1] = 1/5 = 0.2
        np.testing.assert_allclose(model.matrix[0, 0], 0.8, atol=1e-10)
        np.testing.assert_allclose(model.matrix[0, 1], 0.2, atol=1e-10)
        # State 1: only self-transitions (last 4 pairs: 1→1)
        np.testing.assert_allclose(model.matrix[1, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 0], 0.0, atol=1e-10)

    def test_alternating(self):
        """[0,1,0,1,0,1] → T[0,1]=1.0, T[1,0]=1.0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        model = empirical_transitions(labels)
        np.testing.assert_allclose(model.matrix[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[0, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 1], 0.0, atol=1e-10)

    def test_three_state_cycle(self):
        """[0,1,2,0,1,2,0] → T[0,1]=1, T[1,2]=1, T[2,0]=1."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0])
        model = empirical_transitions(labels)
        np.testing.assert_allclose(model.matrix[0, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 2], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[2, 0], 1.0, atol=1e-10)

    def test_two_bars(self):
        """Minimal: [0, 1] → T[0,1]=1.0, T[1,*] has no transitions."""
        labels = np.array([0, 1])
        model = empirical_transitions(labels)
        np.testing.assert_allclose(model.matrix[0, 1], 1.0, atol=1e-10)
        # State 1 only appears at end — no outgoing transitions
        # Row should still sum to 1 (handled by zero-row fallback)
        row_sum = model.matrix[1, :].sum()
        np.testing.assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_symmetric_two_state(self):
        """[0,0,1,1,0,0,1,1] → transitions counted from pairs."""
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        model = empirical_transitions(labels)
        # Pairs: (0,0),(0,1),(1,1),(1,0),(0,0),(0,1),(1,1)
        # From state 0: 0→0 twice, 0→1 twice = T[0,0]=0.5, T[0,1]=0.5
        np.testing.assert_allclose(model.matrix[0, 0], 0.5, atol=1e-10)
        np.testing.assert_allclose(model.matrix[0, 1], 0.5, atol=1e-10)
        # From state 1: 1→1 twice, 1→0 once = T[1,0]=1/3, T[1,1]=2/3
        np.testing.assert_allclose(model.matrix[1, 0], 1.0 / 3, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 1], 2.0 / 3, atol=1e-10)


# ===========================================================================
# Self-Transition Rates
# ===========================================================================


class TestSelfTransitionRates:
    """Self-transition rates must equal the matrix diagonal."""

    def test_matches_diagonal(self):
        labels = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1])
        model = empirical_transitions(labels)
        for s, rate in model.self_transition_rates.items():
            idx = sorted(np.unique(labels).tolist()).index(s)
            assert rate == pytest.approx(model.matrix[idx, idx])

    def test_alternating_zero_str(self):
        """Alternating sequence → self_transition_rate = 0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert model.self_transition_rates[0] == pytest.approx(0.0)
        assert model.self_transition_rates[1] == pytest.approx(0.0)

    def test_all_same_str_one(self):
        """Single state → self_transition_rate = 1.0."""
        labels = np.array([2, 2, 2, 2, 2])
        model = empirical_transitions(labels)
        assert model.self_transition_rates[2] == pytest.approx(1.0)

    def test_high_persistence(self):
        """Long runs → high self-transition rate."""
        labels = np.array([0]*100 + [1]*100)
        model = empirical_transitions(labels)
        # State 0: 99 self-transitions, 1 transition to 1 → 99/100
        assert model.self_transition_rates[0] == pytest.approx(99.0 / 100)


# ===========================================================================
# Row Entropy
# ===========================================================================


class TestRowEntropy:
    """Row entropy measures predictability of next state."""

    def test_deterministic_zero_entropy(self):
        """If transitions are deterministic → entropy ≈ 0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        model = empirical_transitions(labels)
        # T[0,1]=1.0 → row is [0, 1] → entropy = -1*log(1) = 0
        assert model.row_entropy[0] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_max_entropy(self):
        """Equal probability → maximum entropy."""
        # Construct sequence where state 0 transitions equally to 0,1,2
        labels = np.array([0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2])
        model = empirical_transitions(labels)
        # Check entropy is positive for state 0 (has multiple successors)
        assert model.row_entropy[0] > 0

    def test_single_state_zero_entropy(self):
        """Single state → T=[[1]] → entropy = -1*log(1) ≈ 0."""
        labels = np.array([0, 0, 0, 0])
        model = empirical_transitions(labels)
        assert model.row_entropy[0] == pytest.approx(0.0, abs=1e-10)

    def test_entropy_non_negative(self):
        """Entropy is always >= 0."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 4, size=500)
        model = empirical_transitions(labels)
        for s, ent in model.row_entropy.items():
            assert ent >= -1e-10

    def test_entropy_bounded_by_log_k(self):
        """Entropy <= log(k) for k states."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 5, size=1000)
        model = empirical_transitions(labels)
        k = len(model.state_names)
        max_entropy = np.log(k)
        for s, ent in model.row_entropy.items():
            assert ent <= max_entropy + 1e-10


# ===========================================================================
# Most Likely Successor
# ===========================================================================


class TestMostLikelySuccessor:
    """Most likely successor is argmax of off-diagonal row."""

    def test_alternating(self):
        """[0,1,0,1] → successor of 0 is 1, successor of 1 is 0."""
        labels = np.array([0, 1, 0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert model.most_likely_successor[0] == 1
        assert model.most_likely_successor[1] == 0

    def test_cycle(self):
        """[0,1,2,0,1,2] → 0→1, 1→2, 2→0."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0])
        model = empirical_transitions(labels)
        assert model.most_likely_successor[0] == 1
        assert model.most_likely_successor[1] == 2
        assert model.most_likely_successor[2] == 0

    def test_single_state_successor_is_self(self):
        """Only one state → successor is itself."""
        labels = np.array([0, 0, 0, 0])
        model = empirical_transitions(labels)
        assert model.most_likely_successor[0] == 0

    def test_asymmetric(self):
        """State 0 → prefers 1 over 2."""
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 2])
        model = empirical_transitions(labels)
        # 0→1 happens 3 times, 0→2 happens 1 time, 0→0 happens 3 times
        # Off-diagonal: 1 has 3, 2 has 1 → successor is 1
        assert model.most_likely_successor[0] == 1


# ===========================================================================
# Duration Distributions
# ===========================================================================


class TestDurationDistributions:
    """Duration distributions must match hand-counted run lengths."""

    def test_known_durations(self):
        """[0,0,0,1,1,0,0] → durations[0]=[3,2], durations[1]=[2]."""
        labels = np.array([0, 0, 0, 1, 1, 0, 0])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[0], [3, 2])
        np.testing.assert_array_equal(model.duration_distributions[1], [2])

    def test_single_run(self):
        """All same → single run."""
        labels = np.array([0, 0, 0, 0, 0])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[0], [5])

    def test_alternating_unit_runs(self):
        """Alternating → all runs of length 1."""
        labels = np.array([0, 1, 0, 1, 0])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[0], [1, 1, 1])
        np.testing.assert_array_equal(model.duration_distributions[1], [1, 1])

    def test_multiple_regimes(self):
        """Three states with known run pattern."""
        labels = np.array([0, 0, 1, 1, 1, 2, 0, 0, 0, 0])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[0], [2, 4])
        np.testing.assert_array_equal(model.duration_distributions[1], [3])
        np.testing.assert_array_equal(model.duration_distributions[2], [1])

    def test_single_bar_state(self):
        """State appears only once, run=1."""
        labels = np.array([0, 1, 0])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[1], [1])

    def test_sum_equals_total_bars(self):
        """Sum of all durations == total bars."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 3, size=500)
        model = empirical_transitions(labels)
        total = sum(d.sum() for d in model.duration_distributions.values())
        assert total == 500


# ===========================================================================
# Mean Durations
# ===========================================================================


class TestMeanDurations:
    """Mean duration is the average run length per state."""

    def test_known_mean(self):
        """[0,0,0,1,1,0,0] → mean_dur[0]=(3+2)/2=2.5, mean_dur[1]=2."""
        labels = np.array([0, 0, 0, 1, 1, 0, 0])
        model = empirical_transitions(labels)
        assert model.mean_durations[0] == pytest.approx(2.5)
        assert model.mean_durations[1] == pytest.approx(2.0)

    def test_single_state_mean(self):
        """All same → mean = total length."""
        labels = np.array([0, 0, 0, 0, 0, 0, 0])
        model = empirical_transitions(labels)
        assert model.mean_durations[0] == pytest.approx(7.0)

    def test_geometric_approximation(self):
        """For persistent states: mean ≈ 1/(1-STR) (geometric distribution)."""
        # Create highly persistent sequence
        rng = np.random.RandomState(42)
        labels = []
        state = 0
        for _ in range(10000):
            labels.append(state)
            # P(stay) = 0.95 → expected duration = 20
            if rng.random() < 0.05:
                state = 1 - state
        labels = np.array(labels)
        model = empirical_transitions(labels)
        # Check geometric approximation
        for s in [0, 1]:
            str_rate = model.self_transition_rates[s]
            if str_rate < 1.0:
                expected_geo = 1.0 / (1.0 - str_rate)
                actual = model.mean_durations[s]
                # Should be roughly close (within 20% for large samples)
                assert abs(actual - expected_geo) / expected_geo < 0.2


# ===========================================================================
# State Names
# ===========================================================================


class TestStateNames:
    """State names handling."""

    def test_default_names(self):
        labels = np.array([0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert model.state_names == ["S0", "S1"]

    def test_custom_names(self):
        labels = np.array([0, 1, 2, 0])
        model = empirical_transitions(labels, state_names=["calm", "volatile", "trending"])
        assert model.state_names == ["calm", "volatile", "trending"]

    def test_wrong_length_names_raises(self):
        labels = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="state_names length"):
            empirical_transitions(labels, state_names=["a", "b", "c"])

    def test_non_contiguous_state_ids(self):
        """States 0, 5, 10 → default names S0, S5, S10."""
        labels = np.array([0, 5, 10, 0, 5, 10])
        model = empirical_transitions(labels)
        assert model.state_names == ["S0", "S5", "S10"]
        assert model.matrix.shape == (3, 3)


# ===========================================================================
# Non-Contiguous State IDs
# ===========================================================================


class TestNonContiguousStates:
    """States don't have to be 0,1,2,... — can be sparse."""

    def test_sparse_ids_matrix_shape(self):
        """States [2, 7] → 2x2 matrix."""
        labels = np.array([2, 2, 7, 7, 2, 7])
        model = empirical_transitions(labels)
        assert model.matrix.shape == (2, 2)

    def test_sparse_ids_transition_correct(self):
        """[2, 7, 2, 7] → T[2→7]=1, T[7→2]=1."""
        labels = np.array([2, 7, 2, 7])
        model = empirical_transitions(labels)
        # State 2 is index 0, state 7 is index 1
        np.testing.assert_allclose(model.matrix[0, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(model.matrix[1, 0], 1.0, atol=1e-10)

    def test_sparse_ids_duration(self):
        """Non-contiguous IDs still compute durations correctly."""
        labels = np.array([3, 3, 3, 8, 8, 3])
        model = empirical_transitions(labels)
        np.testing.assert_array_equal(model.duration_distributions[3], [3, 1])
        np.testing.assert_array_equal(model.duration_distributions[8], [2])

    def test_sparse_ids_self_transition_keys(self):
        """Keys in self_transition_rates are original state IDs, not indices."""
        labels = np.array([10, 10, 20, 20, 10])
        model = empirical_transitions(labels)
        assert 10 in model.self_transition_rates
        assert 20 in model.self_transition_rates

    def test_sparse_ids_successor_keys(self):
        """most_likely_successor uses original IDs."""
        labels = np.array([5, 5, 9, 9, 5, 9])
        model = empirical_transitions(labels)
        assert model.most_likely_successor[5] == 9
        assert model.most_likely_successor[9] == 5


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    """Invalid inputs raise appropriate errors."""

    def test_empty_labels(self):
        with pytest.raises(ValueError, match="empty"):
            empirical_transitions(np.array([], dtype=int))

    def test_2d_labels(self):
        with pytest.raises(ValueError, match="1-D"):
            empirical_transitions(np.array([[0, 1], [1, 0]]))

    def test_single_bar(self):
        """Single bar → no transitions, but shouldn't crash."""
        model = empirical_transitions(np.array([0]))
        assert model.matrix.shape == (1, 1)
        # No transitions exist, row gets normalized to avoid NaN
        np.testing.assert_allclose(model.matrix.sum(axis=1), 1.0, atol=1e-10)

    def test_state_names_wrong_length(self):
        with pytest.raises(ValueError, match="state_names length"):
            empirical_transitions(np.array([0, 1, 0]), state_names=["a"])


# ===========================================================================
# Return Type
# ===========================================================================


class TestReturnType:
    """empirical_transitions returns TransitionModel with correct types."""

    def test_return_type(self):
        labels = np.array([0, 1, 0, 1, 0])
        model = empirical_transitions(labels)
        assert isinstance(model, TransitionModel)

    def test_matrix_is_ndarray(self):
        labels = np.array([0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert isinstance(model.matrix, np.ndarray)

    def test_matrix_dtype_float(self):
        labels = np.array([0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert model.matrix.dtype == np.float64

    def test_state_names_is_list(self):
        labels = np.array([0, 1, 0, 1])
        model = empirical_transitions(labels)
        assert isinstance(model.state_names, list)

    def test_duration_distributions_values_are_arrays(self):
        labels = np.array([0, 0, 1, 1, 0])
        model = empirical_transitions(labels)
        for s, durs in model.duration_distributions.items():
            assert isinstance(durs, np.ndarray)

    def test_matrix_shape(self):
        labels = np.array([0, 1, 2, 0, 1])
        model = empirical_transitions(labels)
        assert model.matrix.shape == (3, 3)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same input → identical output."""

    def test_deterministic(self):
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        m1 = empirical_transitions(labels)
        m2 = empirical_transitions(labels)
        np.testing.assert_array_equal(m1.matrix, m2.matrix)
        assert m1.self_transition_rates == m2.self_transition_rates
        assert m1.most_likely_successor == m2.most_likely_successor
        assert m1.mean_durations == m2.mean_durations


# ===========================================================================
# Matrix Properties
# ===========================================================================


class TestMatrixProperties:
    """Mathematical properties of transition matrices."""

    def test_non_negative(self):
        """All entries >= 0."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 4, size=500)
        model = empirical_transitions(labels)
        assert np.all(model.matrix >= 0)

    def test_max_one(self):
        """All entries <= 1."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 4, size=500)
        model = empirical_transitions(labels)
        assert np.all(model.matrix <= 1.0 + 1e-10)

    def test_diagonal_dominant_for_persistent(self):
        """Persistent sequences have diagonal > off-diagonal."""
        labels = np.array([0]*50 + [1]*50 + [0]*50 + [1]*50)
        model = empirical_transitions(labels)
        for i in range(model.matrix.shape[0]):
            diag = model.matrix[i, i]
            off_diag_max = max(
                model.matrix[i, j] for j in range(model.matrix.shape[1]) if j != i
            )
            assert diag > off_diag_max

    def test_stationary_distribution_exists(self):
        """For ergodic chains, a stationary distribution exists (left eigenvector)."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 3, size=5000)
        model = empirical_transitions(labels)
        # Power method: T^n converges
        T = model.matrix
        pi = np.ones(T.shape[0]) / T.shape[0]
        for _ in range(100):
            pi = pi @ T
        # pi should be positive and sum to 1
        assert np.all(pi > 0)
        assert pi.sum() == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Unusual but valid inputs."""

    def test_all_same_large(self):
        """10000 bars, all same state."""
        labels = np.zeros(10000, dtype=int)
        model = empirical_transitions(labels)
        assert model.matrix.shape == (1, 1)
        assert model.matrix[0, 0] == pytest.approx(1.0)
        assert model.mean_durations[0] == pytest.approx(10000.0)

    def test_rapid_switching(self):
        """Every bar switches: 0,1,2,0,1,2,..."""
        labels = np.tile([0, 1, 2], 100)
        model = empirical_transitions(labels)
        # All self-transitions should be 0
        for s in [0, 1, 2]:
            assert model.self_transition_rates[s] == pytest.approx(0.0)

    def test_state_at_end_only(self):
        """State appears only at the very end → no outgoing transitions."""
        labels = np.array([0, 0, 0, 0, 1])
        model = empirical_transitions(labels)
        # State 1 has no outgoing transitions — row should still sum to 1
        row_sum = model.matrix[1, :].sum()
        np.testing.assert_allclose(row_sum, 1.0, atol=1e-10)

    def test_float_labels_converted(self):
        """Float labels should work (converted to int internally)."""
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        model = empirical_transitions(labels)
        assert model.matrix.shape == (2, 2)

    def test_negative_labels(self):
        """Negative state IDs are valid."""
        labels = np.array([-1, -1, 0, 0, -1])
        model = empirical_transitions(labels)
        assert -1 in model.self_transition_rates
        assert 0 in model.self_transition_rates

    def test_large_state_ids(self):
        """Very large state IDs (e.g. 1000000)."""
        labels = np.array([0, 1000000, 0, 1000000])
        model = empirical_transitions(labels)
        assert model.matrix.shape == (2, 2)
        assert 1000000 in model.most_likely_successor


# ===========================================================================
# Duration Distribution Helper
# ===========================================================================


class TestComputeDurationDistributions:
    """Direct tests for _compute_duration_distributions."""

    def test_empty(self):
        result = _compute_duration_distributions(np.array([]), [0, 1])
        assert len(result[0]) == 0
        assert len(result[1]) == 0

    def test_single_element(self):
        result = _compute_duration_distributions(np.array([0]), [0])
        np.testing.assert_array_equal(result[0], [1])

    def test_known_pattern(self):
        result = _compute_duration_distributions(
            np.array([0, 0, 1, 0, 0, 0, 1, 1]), [0, 1]
        )
        np.testing.assert_array_equal(result[0], [2, 3])
        np.testing.assert_array_equal(result[1], [1, 2])

    def test_state_never_appears(self):
        """State in unique_states but not in labels → empty array."""
        result = _compute_duration_distributions(np.array([0, 0, 0]), [0, 1])
        np.testing.assert_array_equal(result[0], [3])
        np.testing.assert_array_equal(result[1], [])


# ===========================================================================
# Consistency Checks
# ===========================================================================


class TestConsistency:
    """Internal consistency between fields."""

    def test_duration_count_matches_transitions(self):
        """Number of runs = number of entering transitions + 1 (for first run)."""
        labels = np.array([0, 0, 1, 1, 0, 0, 1])
        model = empirical_transitions(labels)
        # State 0 has 2 runs, state 1 has 2 runs
        assert len(model.duration_distributions[0]) == 2
        assert len(model.duration_distributions[1]) == 2

    def test_all_keys_present(self):
        """All unique states appear in all dictionaries."""
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 5, size=200)
        model = empirical_transitions(labels)
        unique = set(np.unique(labels))
        assert set(model.self_transition_rates.keys()) == unique
        assert set(model.row_entropy.keys()) == unique
        assert set(model.most_likely_successor.keys()) == unique
        assert set(model.mean_durations.keys()) == unique
        assert set(model.duration_distributions.keys()) == unique

    def test_n_states_matches_matrix(self):
        """Matrix dimensions match number of unique states."""
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        model = empirical_transitions(labels)
        assert model.matrix.shape[0] == 4
        assert model.matrix.shape[1] == 4
        assert len(model.state_names) == 4
