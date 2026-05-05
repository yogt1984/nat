"""
Skeptical tests for cluster_pipeline.characterize — state profiling.

Test philosophy:
  - Centroid must equal manual mean of derivatives for that state
  - Elevated/suppressed features are disjoint
  - All states are profiled
  - Duration stats match hand-computed values
  - Successor probs match transition matrix
  - Edge cases: single state, single bar, constant columns, NaN
  - Input validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cluster_pipeline.characterize import (
    StateProfile,
    characterize_states,
)
from cluster_pipeline.hierarchy import HierarchicalLabels, StructureTest
from cluster_pipeline.transitions import empirical_transitions, TransitionModel


# ===========================================================================
# Helpers
# ===========================================================================


def _make_hierarchy(macro_labels: np.ndarray, micro_labels: np.ndarray,
                    label_map: dict, n_micro_total: int,
                    n_micro_per_regime: dict) -> HierarchicalLabels:
    """Build a HierarchicalLabels for testing."""
    composite = np.array([
        f"R{label_map[m][0]}_S{label_map[m][1]}" for m in micro_labels
    ])
    return HierarchicalLabels(
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        composite_labels=composite,
        n_macro=len(np.unique(macro_labels)),
        n_micro_per_regime=n_micro_per_regime,
        n_micro_total=n_micro_total,
        label_map=label_map,
    )


def _simple_setup(n=100, seed=42):
    """
    Create a simple 2-regime, 3-micro-state scenario:
      - State 0: regime 0, local 0 → columns high
      - State 1: regime 0, local 1 → columns low
      - State 2: regime 1, local 0 → columns medium
    """
    rng = np.random.RandomState(seed)

    # Create derivatives with known separation
    data = np.zeros((n, 5))
    micro_labels = np.zeros(n, dtype=int)
    macro_labels = np.zeros(n, dtype=int)

    n0 = n // 3
    n1 = n // 3
    n2 = n - n0 - n1

    # State 0: high values
    data[:n0, :] = rng.normal(3, 0.5, (n0, 5))
    micro_labels[:n0] = 0
    macro_labels[:n0] = 0

    # State 1: low values
    data[n0:n0+n1, :] = rng.normal(-3, 0.5, (n1, 5))
    micro_labels[n0:n0+n1] = 1
    macro_labels[n0:n0+n1] = 0

    # State 2: medium values
    data[n0+n1:, :] = rng.normal(0, 0.5, (n2, 5))
    micro_labels[n0+n1:] = 2
    macro_labels[n0+n1:] = 1

    columns = [f"feat_{i}" for i in range(5)]
    derivatives = pd.DataFrame(data, columns=columns)

    label_map = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    hierarchy = _make_hierarchy(
        macro_labels=macro_labels,
        micro_labels=micro_labels,
        label_map=label_map,
        n_micro_total=3,
        n_micro_per_regime={0: 2, 1: 1},
    )

    transition_model = empirical_transitions(micro_labels)

    return derivatives, hierarchy, transition_model


# ===========================================================================
# Centroid Is Mean
# ===========================================================================


class TestCentroidIsMean:
    """Profile centroid must equal manual mean of derivatives where label==state."""

    def test_centroid_state_0(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        mask = hierarchy.micro_labels == 0
        expected = derivatives.loc[mask].mean()
        for col in derivatives.columns:
            assert profiles[0].centroid[col] == pytest.approx(expected[col], abs=1e-10)

    def test_centroid_state_1(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        mask = hierarchy.micro_labels == 1
        expected = derivatives.loc[mask].mean()
        for col in derivatives.columns:
            assert profiles[1].centroid[col] == pytest.approx(expected[col], abs=1e-10)

    def test_centroid_state_2(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        mask = hierarchy.micro_labels == 2
        expected = derivatives.loc[mask].mean()
        for col in derivatives.columns:
            assert profiles[2].centroid[col] == pytest.approx(expected[col], abs=1e-10)

    def test_centroid_single_bar(self):
        """State with single bar → centroid = that bar's values."""
        derivatives = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        micro_labels = np.array([0, 1, 2])
        macro_labels = np.array([0, 0, 0])
        label_map = {0: (0, 0), 1: (0, 1), 2: (0, 2)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 3, {0: 3})
        tm = empirical_transitions(micro_labels)
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert profiles[0].centroid["a"] == pytest.approx(1.0)
        assert profiles[0].centroid["b"] == pytest.approx(4.0)


# ===========================================================================
# Elevated and Suppressed Disjoint
# ===========================================================================


class TestElevatedSuppressedDisjoint:
    """No overlap between top_elevated and top_suppressed."""

    def test_disjoint(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            elevated_cols = {col for col, _ in profile.top_elevated}
            suppressed_cols = {col for col, _ in profile.top_suppressed}
            assert elevated_cols.isdisjoint(suppressed_cols), (
                f"State {state_id}: overlap in elevated/suppressed"
            )

    def test_elevated_positive_z(self):
        """All elevated features have positive z-score."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            for col, z in profile.top_elevated:
                assert z > 0, f"State {state_id}, {col}: z={z} should be > 0"

    def test_suppressed_negative_z(self):
        """All suppressed features have negative z-score."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            for col, z in profile.top_suppressed:
                assert z < 0, f"State {state_id}, {col}: z={z} should be < 0"

    def test_high_state_has_elevated(self):
        """State 0 (high values) should have elevated features."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert len(profiles[0].top_elevated) > 0

    def test_low_state_has_suppressed(self):
        """State 1 (low values) should have suppressed features."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert len(profiles[1].top_suppressed) > 0


# ===========================================================================
# All States Profiled
# ===========================================================================


class TestAllStatesProfiled:
    """Every state gets a profile."""

    def test_count_equals_n_micro_total(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert len(profiles) == hierarchy.n_micro_total

    def test_keys_are_state_ids(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert set(profiles.keys()) == set(range(hierarchy.n_micro_total))

    def test_five_states(self):
        """5 micro states → 5 profiles."""
        rng = np.random.RandomState(42)
        n = 200
        data = rng.normal(0, 1, (n, 3))
        micro_labels = np.array([i % 5 for i in range(n)])
        macro_labels = np.array([i % 5 // 3 for i in range(n)])
        label_map = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 5, {0: 3, 1: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(data, columns=["a", "b", "c"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert len(profiles) == 5


# ===========================================================================
# Duration Statistics
# ===========================================================================


class TestDurationStatistics:
    """Duration stats come from transition model."""

    def test_known_durations(self):
        """Hand-construct labels with known run lengths."""
        # State 0: runs of 5, 5 → mean=5, median=5, p90=5
        # State 1: runs of 3, 7 → mean=5, median=5
        # Sequence: [0]*5 + [1]*3 + [0]*5 + [1]*7 (the [1]*3+[1]*4 merges)
        micro_labels = np.array([0]*5 + [1]*3 + [0]*5 + [1]*7)
        macro_labels = np.zeros(20, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert profiles[0].duration_mean == pytest.approx(5.0)
        assert profiles[0].duration_median == pytest.approx(5.0)
        assert profiles[1].duration_mean == pytest.approx(5.0)

    def test_single_long_run(self):
        """All same state → single run = total length."""
        n = 50
        micro_labels = np.zeros(n, dtype=int)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 1, {0: 1})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(np.random.randn(n, 2), columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert profiles[0].duration_mean == pytest.approx(50.0)
        assert profiles[0].duration_median == pytest.approx(50.0)
        assert profiles[0].duration_p90 == pytest.approx(50.0)

    def test_alternating_unit_durations(self):
        """Alternating → all runs of length 1."""
        micro_labels = np.array([0, 1] * 20)
        macro_labels = np.zeros(40, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(np.random.randn(40, 2), columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert profiles[0].duration_mean == pytest.approx(1.0)
        assert profiles[1].duration_mean == pytest.approx(1.0)


# ===========================================================================
# Successor Probabilities
# ===========================================================================


class TestSuccessorProbs:
    """Successor probs match transition matrix (off-diagonal)."""

    def test_alternating_full_transition(self):
        """[0,1,0,1,...] → P(0→1)=1, P(1→0)=1."""
        micro_labels = np.array([0, 1, 0, 1, 0, 1])
        macro_labels = np.zeros(6, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(np.random.randn(6, 2), columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert profiles[0].successor_probs[1] == pytest.approx(1.0)
        assert profiles[1].successor_probs[0] == pytest.approx(1.0)

    def test_no_self_in_successor(self):
        """Successor probs should not include self-transitions."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            assert state_id not in profile.successor_probs

    def test_successor_probs_sum_with_self(self):
        """Off-diagonal probs + self-transition rate = 1 (approximately)."""
        micro_labels = np.array([0]*10 + [1]*5 + [0]*10 + [2]*5 + [0]*10)
        macro_labels = np.zeros(40, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1), 2: (0, 2)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 3, {0: 3})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(np.random.randn(40, 2), columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            off_diag_sum = sum(profile.successor_probs.values())
            self_rate = tm.self_transition_rates.get(state_id, 0)
            assert off_diag_sum + self_rate == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# State Profile Metadata
# ===========================================================================


class TestProfileMetadata:
    """Metadata fields are correct."""

    def test_regime_id_correct(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            expected_regime, expected_local = hierarchy.label_map[state_id]
            assert profile.regime_id == expected_regime
            assert profile.local_state_id == expected_local

    def test_n_bars_correct(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            expected = int(np.sum(hierarchy.micro_labels == state_id))
            assert profile.n_bars == expected

    def test_state_id_correct(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for state_id, profile in profiles.items():
            assert profile.state_id == state_id

    def test_n_bars_sum_equals_total(self):
        """Sum of all state n_bars == total number of bars."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        total = sum(p.n_bars for p in profiles.values())
        assert total == len(derivatives)


# ===========================================================================
# Return Type
# ===========================================================================


class TestReturnType:
    """characterize_states returns Dict[int, StateProfile]."""

    def test_return_type(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert isinstance(profiles, dict)

    def test_values_are_state_profiles(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for v in profiles.values():
            assert isinstance(v, StateProfile)

    def test_centroid_is_dict(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for v in profiles.values():
            assert isinstance(v.centroid, dict)

    def test_elevated_is_list_of_tuples(self):
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm)
        for v in profiles.values():
            assert isinstance(v.top_elevated, list)
            for item in v.top_elevated:
                assert isinstance(item, tuple)
                assert len(item) == 2


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    """Invalid inputs raise appropriate errors."""

    def test_length_mismatch(self):
        """derivatives rows != label length → ValueError."""
        derivatives = pd.DataFrame({"a": [1, 2, 3]})
        micro_labels = np.array([0, 1])
        macro_labels = np.array([0, 0])
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        with pytest.raises(ValueError, match="derivatives rows"):
            characterize_states(derivatives, hierarchy, tm)

    def test_empty_derivatives(self):
        """Empty DataFrame → ValueError."""
        derivatives = pd.DataFrame()
        micro_labels = np.array([], dtype=int)
        macro_labels = np.array([], dtype=int)
        label_map = {}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 0, {})
        tm = empirical_transitions(np.array([0]))  # dummy
        with pytest.raises(ValueError, match="empty"):
            characterize_states(derivatives, hierarchy, tm)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Unusual but valid inputs."""

    def test_single_state(self):
        """Only one state → one profile."""
        n = 50
        data = np.random.randn(n, 3)
        micro_labels = np.zeros(n, dtype=int)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 1, {0: 1})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(data, columns=["a", "b", "c"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        assert len(profiles) == 1
        assert profiles[0].n_bars == 50

    def test_constant_column(self):
        """Constant column → z-score = 0, not in elevated/suppressed."""
        n = 60
        micro_labels = np.array([0]*30 + [1]*30)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        data = np.column_stack([
            np.concatenate([np.ones(30) * 5, np.ones(30) * -5]),  # varies
            np.ones(n) * 3.14,  # constant
        ])
        derivatives = pd.DataFrame(data, columns=["varies", "constant"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        # Constant column should not appear in elevated or suppressed
        # (z-score is 0 because centroid == global mean for constant)
        for profile in profiles.values():
            elevated_cols = {col for col, _ in profile.top_elevated}
            suppressed_cols = {col for col, _ in profile.top_suppressed}
            assert "constant" not in elevated_cols
            assert "constant" not in suppressed_cols

    def test_many_columns(self):
        """50 columns → top_n limits output."""
        n = 100
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, (n, 50))
        data[:50, :] += 2  # state 0 elevated
        micro_labels = np.array([0]*50 + [1]*50)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(data, columns=[f"f_{i}" for i in range(50)])
        profiles = characterize_states(derivatives, hierarchy, tm, top_n=5)
        # top_n=5 limits output
        assert len(profiles[0].top_elevated) <= 5
        assert len(profiles[0].top_suppressed) <= 5

    def test_nan_in_derivatives(self):
        """NaN values handled gracefully (pandas mean ignores NaN)."""
        n = 40
        micro_labels = np.array([0]*20 + [1]*20)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        data = np.random.randn(n, 3)
        data[0, 0] = np.nan
        data[5, 1] = np.nan
        derivatives = pd.DataFrame(data, columns=["a", "b", "c"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        # Should not crash; centroid computed ignoring NaN
        assert len(profiles) == 2

    def test_top_n_zero(self):
        """top_n=0 → empty elevated/suppressed."""
        derivatives, hierarchy, tm = _simple_setup()
        profiles = characterize_states(derivatives, hierarchy, tm, top_n=0)
        for profile in profiles.values():
            assert len(profile.top_elevated) == 0
            assert len(profile.top_suppressed) == 0


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same inputs → identical output."""

    def test_deterministic(self):
        derivatives, hierarchy, tm = _simple_setup()
        p1 = characterize_states(derivatives, hierarchy, tm)
        p2 = characterize_states(derivatives, hierarchy, tm)
        for state_id in p1:
            assert p1[state_id].centroid == p2[state_id].centroid
            assert p1[state_id].top_elevated == p2[state_id].top_elevated
            assert p1[state_id].top_suppressed == p2[state_id].top_suppressed


# ===========================================================================
# Z-Score Logic
# ===========================================================================


class TestZScoreLogic:
    """Z-scores correctly identify distinguishing features."""

    def test_state_with_higher_mean_has_elevated(self):
        """If state 0 has col 'a' much higher than global mean → elevated."""
        n = 100
        data = np.zeros((n, 2))
        data[:50, 0] = 10  # state 0, col a = 10
        data[50:, 0] = -10  # state 1, col a = -10
        data[:, 1] = 0  # col b neutral for both
        micro_labels = np.array([0]*50 + [1]*50)
        macro_labels = np.zeros(n, dtype=int)
        label_map = {0: (0, 0), 1: (0, 1)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 2, {0: 2})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(data, columns=["a", "b"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        # State 0 should have 'a' elevated
        elevated_cols_0 = {col for col, _ in profiles[0].top_elevated}
        assert "a" in elevated_cols_0
        # State 1 should have 'a' suppressed
        suppressed_cols_1 = {col for col, _ in profiles[1].top_suppressed}
        assert "a" in suppressed_cols_1

    def test_neutral_state_no_extreme_z(self):
        """State with mean ≈ global mean → no strong elevated/suppressed."""
        n = 150
        rng = np.random.RandomState(42)
        # State 0: mean=5, State 1: mean=-5, State 2: mean=0 (≈ global mean)
        data = np.zeros((n, 1))
        data[:50] = 5 + rng.normal(0, 0.1, (50, 1))
        data[50:100] = -5 + rng.normal(0, 0.1, (50, 1))
        data[100:] = rng.normal(0, 0.1, (50, 1))
        micro_labels = np.array([0]*50 + [1]*50 + [2]*50)
        macro_labels = np.array([0]*50 + [0]*50 + [1]*50)
        label_map = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
        hierarchy = _make_hierarchy(macro_labels, micro_labels, label_map, 3, {0: 2, 1: 1})
        tm = empirical_transitions(micro_labels)
        derivatives = pd.DataFrame(data, columns=["f"])
        profiles = characterize_states(derivatives, hierarchy, tm)
        # State 2 (mean≈0, global mean ≈0) should have small z-scores
        # It may still have entries if slightly above/below, but they'll be small
        if profiles[2].top_elevated:
            _, z = profiles[2].top_elevated[0]
            assert abs(z) < abs(profiles[0].top_elevated[0][1])
