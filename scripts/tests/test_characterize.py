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
    TransitionSignature,
    compute_signatures,
    ReturnProfile,
    return_profile,
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


# ===========================================================================
# Task 5.2: Entry and Exit Signatures
# ===========================================================================


def _make_signature_data(n=100, seed=42):
    """Create data with known entry/exit patterns for state 0."""
    rng = np.random.RandomState(seed)
    # Pattern: [1]*10 + [0]*10 + [1]*10 + [0]*10 + ... (repeating)
    # This gives predictable entry/exit counts
    labels = np.array(([1]*10 + [0]*10) * (n // 20))
    data = rng.normal(0, 1, (len(labels), 3))
    # Add a ramp before entry to state 0 (derivative increases before switching)
    for t in range(len(labels)):
        if labels[t] == 0 and (t == 0 or labels[t-1] != 0):
            # Entry at t — make preceding bars ramp up
            for offset in range(1, 6):
                if t - offset >= 0:
                    data[t - offset, 0] += offset * 0.5  # ramp on col 0
    derivatives = pd.DataFrame(data, columns=["a", "b", "c"])
    return derivatives, labels


class TestEntryCount:
    """Entry count must match hand-counted transitions INTO the state."""

    def test_known_entry_count(self):
        """[1,0,0,1,0,0,1,0] → state 0 entered at index 1 and 4 → 2 entries."""
        labels = np.array([1, 0, 0, 1, 0, 0, 1, 0])
        derivatives = pd.DataFrame(np.random.randn(8, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=1, min_events=1)
        assert sig is not None
        # Entries at indices 1, 4, 7 (where labels[t]==0 and labels[t-1]!=0)
        # t=1: labels[1]=0, labels[0]=1 ✓
        # t=4: labels[4]=0, labels[3]=1 ✓
        # t=7: labels[7]=0, labels[6]=1 ✓
        # But entry_count only counts those with enough lookback
        # With lookback=1: t=1 needs t=0 (ok), t=4 needs t=3 (ok), t=7 needs t=6 (ok)
        assert sig.entry_count == 3

    def test_first_bar_is_entry(self):
        """If label[0] == state_id, it counts as an entry but no lookback available."""
        labels = np.array([0, 0, 1, 0, 0])
        derivatives = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        # Entry at t=0 (no lookback), t=3 (lookback=2 → needs t=1,2)
        # t=0: start=0-2=-2 < 0 → skip
        # t=3: start=3-2=1 → window=[1,2] → ok
        assert sig is not None
        assert sig.entry_count == 1

    def test_entry_count_long_lookback(self):
        """Longer lookback → fewer valid entries (need more history)."""
        labels = np.array([1]*5 + [0]*5 + [1]*5 + [0]*5)
        derivatives = pd.DataFrame(np.random.randn(20, 2), columns=["a", "b"])
        sig_short = compute_signatures(derivatives, labels, state_id=0,
                                       lookback=3, min_events=1)
        sig_long = compute_signatures(derivatives, labels, state_id=0,
                                      lookback=10, min_events=1)
        # Short lookback: entry at t=5 (start=2 ok), t=15 (start=12 ok) → 2
        assert sig_short is not None
        assert sig_short.entry_count == 2
        # Long lookback: entry at t=5 (start=-5 < 0 skip), t=15 (start=5 ok) → 1
        assert sig_long is not None
        assert sig_long.entry_count == 1


class TestExitCount:
    """Exit count must match hand-counted transitions OUT OF the state."""

    def test_known_exit_count(self):
        """[0,0,0,1,1,0,0,1] → state 0 exits at index 2, 6."""
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 1])
        derivatives = pd.DataFrame(np.random.randn(8, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=1, min_events=1)
        # Exit at t where labels[t]==0 and labels[t+1]!=0
        # t=2: labels[2]=0, labels[3]=1 ✓ → exit window = [t+1:t+2] = [3:4] ok
        # t=6: labels[6]=0, labels[7]=1 ✓ → exit window = [t+1:t+2] = [7:8] ok
        assert sig is not None
        assert sig.exit_count == 2

    def test_last_bar_is_exit_no_window(self):
        """If state ends at last bar, it's an exit but no forward window."""
        labels = np.array([1, 1, 0, 0, 0])
        derivatives = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        # Exit at t=4 (last bar) → need t+1:t+3 = [5:7] → out of bounds → skip
        # Exit at t=2? No, labels[3]=0, so t=2 is not exit
        # Actually exit at t=4: labels[4]=0 and t==n-1 → it IS exit
        # But window t+1:t+1+2 = [5:7] > len=5 → skip
        # So exit_count = 0
        # Entry at t=2: start=2-2=0, window=[0:2] ok → entry_count=1
        assert sig is not None
        assert sig.exit_count == 0
        assert sig.entry_count == 1


class TestInsufficientEventsReturnsNone:
    """Returns None when both entry and exit counts < min_events."""

    def test_too_few_events(self):
        """State with only 2 entries, min_events=5 → None."""
        labels = np.array([1]*10 + [0]*5 + [1]*10 + [0]*5)
        derivatives = pd.DataFrame(np.random.randn(30, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=5)
        assert sig is None

    def test_exactly_min_events_entry(self):
        """Exactly min_events entries → returns result."""
        # Create 5 entry events with lookback=1
        labels = np.array(([1, 0]) * 6)  # 12 bars, 6 entries to state 0
        derivatives = pd.DataFrame(np.random.randn(12, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=1, min_events=5)
        assert sig is not None
        assert sig.entry_count >= 5

    def test_entry_sufficient_exit_not(self):
        """Entry ≥ min_events but exit < min_events → still returns (entry ok)."""
        # 10 entries with lookback but exits need forward window
        labels = np.array(([1]*3 + [0]*3) * 10)  # 60 bars
        derivatives = pd.DataFrame(np.random.randn(60, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=5)
        assert sig is not None


class TestTrajectoryShape:
    """Trajectory DataFrames have correct shape."""

    def test_entry_trajectory_shape(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert sig.entry_trajectory.shape == (5, 3)

    def test_exit_trajectory_shape(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert sig.exit_trajectory.shape == (5, 3)

    def test_entry_std_shape(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert sig.entry_std.shape == (5, 3)

    def test_exit_std_shape(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert sig.exit_std.shape == (5, 3)

    def test_custom_lookback(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert sig is not None
        assert sig.entry_trajectory.shape == (3, 3)
        assert sig.exit_trajectory.shape == (3, 3)


class TestTrajectoryIndex:
    """Trajectory index uses relative time."""

    def test_entry_index_negative(self):
        """Entry trajectory index: [-lookback, ..., -1]."""
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert list(sig.entry_trajectory.index) == [-5, -4, -3, -2, -1]

    def test_exit_index_positive(self):
        """Exit trajectory index: [1, 2, ..., lookback]."""
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=5, min_events=1)
        assert sig is not None
        assert list(sig.exit_trajectory.index) == [1, 2, 3, 4, 5]


class TestTrajectoryContent:
    """Trajectory values are correct averages."""

    def test_entry_is_mean_of_windows(self):
        """Entry trajectory == mean of all lookback windows before entry."""
        # Simple case: constant values before entry
        labels = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0])
        data = np.arange(20).reshape(10, 2).astype(float)
        derivatives = pd.DataFrame(data, columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        assert sig is not None
        # Entry at t=3: window = derivatives[1:3] = [[2,3],[4,5]]
        # Entry at t=8: window = derivatives[6:8] = [[12,13],[14,15]]
        # Mean: [[(2+12)/2, (3+13)/2], [(4+14)/2, (5+15)/2]] = [[7,8],[9,10]]
        expected = np.array([[7.0, 8.0], [9.0, 10.0]])
        np.testing.assert_allclose(sig.entry_trajectory.values, expected)

    def test_exit_is_mean_of_windows(self):
        """Exit trajectory == mean of all lookback windows after exit."""
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        data = np.arange(20).reshape(10, 2).astype(float)
        derivatives = pd.DataFrame(data, columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        # Exit at t where labels[t]==0 and labels[t+1]!=0
        # t=1: labels[1]=0, labels[2]=1 ✓ → window = [2:4] = [[4,5],[6,7]]
        # t=5: labels[5]=0, labels[6]=1 ✓ → window = [6:8] = [[12,13],[14,15]]
        # Mean: [[(4+12)/2, (5+13)/2], [(6+14)/2, (7+15)/2]] = [[8,9],[10,11]]
        assert sig is not None
        expected = np.array([[8.0, 9.0], [10.0, 11.0]])
        np.testing.assert_allclose(sig.exit_trajectory.values, expected)

    def test_single_event_no_std(self):
        """Single event → std = 0."""
        labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        data = np.ones((8, 2))
        derivatives = pd.DataFrame(data, columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        assert sig is not None
        # Single entry event → std across events = 0
        assert np.all(sig.entry_std.values == 0)


class TestSignatureValidation:
    """Invalid inputs raise errors."""

    def test_length_mismatch(self):
        derivatives = pd.DataFrame({"a": [1, 2, 3]})
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="derivatives rows"):
            compute_signatures(derivatives, labels, state_id=0)

    def test_lookback_zero(self):
        derivatives = pd.DataFrame({"a": [1, 2, 3]})
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            compute_signatures(derivatives, labels, state_id=0, lookback=0)

    def test_negative_lookback(self):
        derivatives = pd.DataFrame({"a": [1, 2, 3]})
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            compute_signatures(derivatives, labels, state_id=0, lookback=-1)


class TestSignatureReturnType:
    """Return type checks."""

    def test_returns_transition_signature(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert isinstance(sig, TransitionSignature)

    def test_trajectories_are_dataframes(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert isinstance(sig.entry_trajectory, pd.DataFrame)
        assert isinstance(sig.exit_trajectory, pd.DataFrame)
        assert isinstance(sig.entry_std, pd.DataFrame)
        assert isinstance(sig.exit_std, pd.DataFrame)

    def test_state_id_stored(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert sig.state_id == 0


class TestSignatureEdgeCases:
    """Edge cases."""

    def test_state_never_appears(self):
        """State not in labels → None (0 events)."""
        labels = np.array([0, 0, 0, 0])
        derivatives = pd.DataFrame(np.random.randn(4, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=99,
                                 lookback=1, min_events=1)
        assert sig is None

    def test_state_all_bars(self):
        """State occupies all bars → 1 entry at t=0, but no lookback."""
        labels = np.array([0, 0, 0, 0, 0])
        derivatives = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=2, min_events=1)
        # Entry at t=0 → need t-2:t → negative → skip
        # No exits (state runs to end without switching)
        # exit_count = 0, entry_count = 0 → None
        assert sig is None

    def test_lookback_one(self):
        """Minimal lookback=1 works."""
        labels = np.array([1, 0, 1, 0, 1, 0])
        derivatives = pd.DataFrame(np.random.randn(6, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=1, min_events=1)
        assert sig is not None
        assert sig.entry_trajectory.shape == (1, 2)

    def test_many_columns(self):
        """50 derivative columns work correctly."""
        rng = np.random.RandomState(42)
        labels = np.array(([1]*5 + [0]*5) * 10)
        data = rng.normal(0, 1, (100, 50))
        derivatives = pd.DataFrame(data, columns=[f"f_{i}" for i in range(50)])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert sig is not None
        assert sig.entry_trajectory.shape[1] == 50

    def test_min_events_zero(self):
        """min_events=0 → always returns (as long as state exists)."""
        labels = np.array([1, 0, 1])
        derivatives = pd.DataFrame(np.random.randn(3, 2), columns=["a", "b"])
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=1, min_events=0)
        # entry at t=1, lookback=1 → window=[0:1] ok → entry_count=1
        assert sig is not None


class TestSignatureDeterminism:
    """Same inputs → same results."""

    def test_deterministic(self):
        derivatives, labels = _make_signature_data()
        s1 = compute_signatures(derivatives, labels, state_id=0, lookback=3, min_events=1)
        s2 = compute_signatures(derivatives, labels, state_id=0, lookback=3, min_events=1)
        pd.testing.assert_frame_equal(s1.entry_trajectory, s2.entry_trajectory)
        pd.testing.assert_frame_equal(s1.exit_trajectory, s2.exit_trajectory)


class TestSignatureColumns:
    """Trajectory columns match derivatives columns."""

    def test_columns_match(self):
        derivatives, labels = _make_signature_data()
        sig = compute_signatures(derivatives, labels, state_id=0,
                                 lookback=3, min_events=1)
        assert list(sig.entry_trajectory.columns) == list(derivatives.columns)
        assert list(sig.exit_trajectory.columns) == list(derivatives.columns)
        assert list(sig.entry_std.columns) == list(derivatives.columns)


# ===========================================================================
# Task 5.3: Forward Return Profiling (Multi-Horizon)
# ===========================================================================


class TestReturnComputation:
    """Forward log returns computed correctly."""

    def test_log_return_single_horizon(self):
        """log(price[t+h] / price[t]) computed correctly."""
        prices = np.array([100.0, 110.0, 121.0, 133.1, 146.41])
        labels = np.array([0, 0, 0, 0, 0])
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        # All bars are state 0, horizon=1
        # returns: log(110/100), log(121/110), log(133.1/121), log(146.41/133.1)
        expected = np.log(np.array([110, 121, 133.1, 146.41]) /
                          np.array([100, 110, 121, 133.1]))
        assert rp.horizons[1]["mean"] == pytest.approx(np.mean(expected), abs=1e-10)
        assert rp.horizons[1]["n"] == 4

    def test_horizon_2(self):
        """Horizon 2: log(price[t+2]/price[t])."""
        prices = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
        labels = np.array([0, 0, 0, 0, 0])
        rp = return_profile(labels, prices, state_id=0, horizons=[2])
        # returns: log(400/100), log(800/200), log(1600/400)
        expected = np.log(np.array([400, 800, 1600]) /
                          np.array([100, 200, 400]))
        assert rp.horizons[2]["mean"] == pytest.approx(np.mean(expected), abs=1e-10)
        assert rp.horizons[2]["n"] == 3

    def test_only_state_bars_used(self):
        """Only bars where labels==state_id contribute returns."""
        prices = np.array([100.0, 110.0, 105.0, 115.0, 120.0])
        labels = np.array([0, 1, 0, 1, 0])
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        # State 0 at t=0,2,4. Horizon 1: t=0→1, t=2→3 (t=4 has no t+1)
        expected = np.array([np.log(110/100), np.log(115/105)])
        assert rp.horizons[1]["mean"] == pytest.approx(np.mean(expected), abs=1e-10)
        assert rp.horizons[1]["n"] == 2

    def test_positive_return(self):
        """Uptrend → positive mean return."""
        prices = np.exp(np.linspace(0, 1, 100))  # exponential growth
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[5])
        assert rp.horizons[5]["mean"] > 0

    def test_negative_return(self):
        """Downtrend → negative mean return."""
        prices = np.exp(np.linspace(0, -1, 100))  # exponential decline
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[5])
        assert rp.horizons[5]["mean"] < 0


class TestMultipleHorizons:
    """Result contains entries for all requested horizons."""

    def test_default_horizons(self):
        """Default horizons [1, 5, 10, 20] all present."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 100).cumsum() + 5)
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0)
        assert set(rp.horizons.keys()) == {1, 5, 10, 20}

    def test_custom_horizons(self):
        """Custom horizons [3, 7, 15] all present."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 100).cumsum() + 5)
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[3, 7, 15])
        assert set(rp.horizons.keys()) == {3, 7, 15}

    def test_single_horizon(self):
        prices = np.ones(50) * 100
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert set(rp.horizons.keys()) == {1}

    def test_all_stats_present(self):
        """Each horizon has all required stat keys."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 100).cumsum() + 5)
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1, 5])
        expected_keys = {"mean", "median", "std", "skew", "kurtosis", "p5", "p95", "sharpe", "n"}
        for h in [1, 5]:
            assert set(rp.horizons[h].keys()) == expected_keys


class TestMeanDurationAdded:
    """mean_duration auto-added to horizons."""

    def test_adds_mean_duration(self):
        """mean_duration=8, horizons=[1,5] → result has 1,5,8."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 50).cumsum() + 5)
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0,
                            horizons=[1, 5], mean_duration=8)
        assert 8 in rp.horizons
        assert set(rp.horizons.keys()) == {1, 5, 8}

    def test_mean_duration_already_in_horizons(self):
        """mean_duration already in horizons → no duplicate."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 50).cumsum() + 5)
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0,
                            horizons=[1, 5, 10], mean_duration=5)
        assert set(rp.horizons.keys()) == {1, 5, 10}

    def test_mean_duration_none(self):
        """mean_duration=None → no extra horizon."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 50).cumsum() + 5)
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0,
                            horizons=[1, 5], mean_duration=None)
        assert set(rp.horizons.keys()) == {1, 5}

    def test_mean_duration_zero_ignored(self):
        """mean_duration=0 → not added."""
        prices = np.exp(np.random.RandomState(42).normal(0, 0.01, 50).cumsum() + 5)
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0,
                            horizons=[1, 5], mean_duration=0)
        assert set(rp.horizons.keys()) == {1, 5}


class TestReturnStatistics:
    """Statistics are mathematically correct."""

    def test_std_positive(self):
        """Non-constant prices → positive std."""
        rng = np.random.RandomState(42)
        prices = np.exp(rng.normal(0, 0.02, 200).cumsum() + 5)
        labels = np.zeros(200, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["std"] > 0

    def test_constant_prices_zero_return(self):
        """Constant prices → mean return = 0, std = 0."""
        prices = np.ones(50) * 100
        labels = np.zeros(50, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["mean"] == pytest.approx(0.0, abs=1e-15)
        assert rp.horizons[1]["std"] == pytest.approx(0.0, abs=1e-15)

    def test_sharpe_sign_matches_mean(self):
        """Sharpe has same sign as mean return."""
        rng = np.random.RandomState(42)
        # Uptrend with noise so std > 0
        prices = np.exp(np.linspace(0, 0.5, 100) + rng.normal(0, 0.01, 100))
        labels = np.zeros(100, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["sharpe"] > 0

    def test_p5_less_than_p95(self):
        """p5 < p95 for non-degenerate returns."""
        rng = np.random.RandomState(42)
        prices = np.exp(rng.normal(0, 0.02, 200).cumsum() + 5)
        labels = np.zeros(200, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["p5"] < rp.horizons[1]["p95"]

    def test_median_between_p5_and_p95(self):
        """Median is between p5 and p95."""
        rng = np.random.RandomState(42)
        prices = np.exp(rng.normal(0, 0.02, 200).cumsum() + 5)
        labels = np.zeros(200, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        stats = rp.horizons[1]
        assert stats["p5"] <= stats["median"] <= stats["p95"]

    def test_n_correct(self):
        """n counts how many valid returns were computed."""
        prices = np.ones(10) * 100
        labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        # State 0 at t=0,1,2,5,6,7. Horizon 1: need t+1<10
        # Valid: t=0,1,2,5,6,7 → all have t+1<10 except... all valid
        # But t+1 must exist: t=7→8 ok. All 6 valid.
        assert rp.horizons[1]["n"] == 6


class TestReturnValidation:
    """Invalid inputs raise errors."""

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="labels length"):
            return_profile(np.array([0, 1]), np.array([100.0]), state_id=0)

    def test_negative_price(self):
        with pytest.raises(ValueError, match="strictly positive"):
            return_profile(np.array([0, 0]), np.array([100.0, -50.0]), state_id=0)

    def test_zero_price(self):
        with pytest.raises(ValueError, match="strictly positive"):
            return_profile(np.array([0, 0]), np.array([100.0, 0.0]), state_id=0)

    def test_2d_labels(self):
        with pytest.raises(ValueError, match="1-D"):
            return_profile(np.array([[0, 1]]), np.array([100.0, 110.0]), state_id=0)

    def test_2d_prices(self):
        with pytest.raises(ValueError, match="1-D"):
            return_profile(np.array([0, 0]), np.array([[100.0, 110.0]]), state_id=0)


class TestReturnEdgeCases:
    """Edge cases."""

    def test_state_not_in_labels(self):
        """State never appears → n=0, stats are NaN."""
        prices = np.array([100.0, 110.0, 120.0])
        labels = np.array([0, 0, 0])
        rp = return_profile(labels, prices, state_id=99, horizons=[1])
        assert rp.horizons[1]["n"] == 0
        assert np.isnan(rp.horizons[1]["mean"])

    def test_horizon_larger_than_data(self):
        """Horizon > data length → n=0."""
        prices = np.array([100.0, 110.0, 120.0])
        labels = np.array([0, 0, 0])
        rp = return_profile(labels, prices, state_id=0, horizons=[100])
        assert rp.horizons[100]["n"] == 0

    def test_state_only_at_end(self):
        """State only at last bar → no forward return possible."""
        prices = np.array([100.0, 110.0, 120.0])
        labels = np.array([1, 1, 0])
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["n"] == 0

    def test_single_bar_in_state(self):
        """Single bar with forward data → n=1."""
        prices = np.array([100.0, 110.0, 120.0])
        labels = np.array([0, 1, 1])
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert rp.horizons[1]["n"] == 1
        assert rp.horizons[1]["mean"] == pytest.approx(np.log(110/100))

    def test_very_large_horizon(self):
        """Large horizon with enough data."""
        prices = np.exp(np.linspace(0, 1, 1000))
        labels = np.zeros(1000, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[500])
        assert rp.horizons[500]["n"] == 500
        # log(exp(0.5+...)/exp(0+...)) ≈ 0.5
        assert rp.horizons[500]["mean"] == pytest.approx(0.5005, abs=0.01)


class TestReturnReturnType:
    """Return type checks."""

    def test_returns_return_profile(self):
        prices = np.ones(20) * 100
        labels = np.zeros(20, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert isinstance(rp, ReturnProfile)

    def test_state_id_stored(self):
        prices = np.ones(20) * 100
        labels = np.zeros(20, dtype=int)
        rp = return_profile(labels, prices, state_id=7, horizons=[1])
        assert rp.state_id == 7

    def test_horizons_is_dict(self):
        prices = np.ones(20) * 100
        labels = np.zeros(20, dtype=int)
        rp = return_profile(labels, prices, state_id=0, horizons=[1])
        assert isinstance(rp.horizons, dict)


class TestReturnDeterminism:
    """Same inputs → same results."""

    def test_deterministic(self):
        rng = np.random.RandomState(42)
        prices = np.exp(rng.normal(0, 0.01, 100).cumsum() + 5)
        labels = np.array([0]*50 + [1]*50)
        r1 = return_profile(labels, prices, state_id=0, horizons=[1, 5, 10])
        r2 = return_profile(labels, prices, state_id=0, horizons=[1, 5, 10])
        for h in [1, 5, 10]:
            assert r1.horizons[h] == r2.horizons[h]
