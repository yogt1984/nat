"""Unit tests for Wave 2 decision gate logic."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_wave2_gate import (
    evaluate_wave2, compute_pairwise_correlations,
    CASE_A, CASE_B, CASE_C, CASE_D, CORRELATION_THRESHOLD,
)


# --- Decision matrix tests ---

def test_case_a():
    """3+ positive algos, no correlated pairs -> CASE_A."""
    assert evaluate_wave2(n_positive=3, max_rho=0.3, n_correlated_pairs=0) == CASE_A
    assert evaluate_wave2(n_positive=4, max_rho=0.1, n_correlated_pairs=0) == CASE_A
    assert evaluate_wave2(n_positive=5, max_rho=0.0, n_correlated_pairs=0) == CASE_A


def test_case_b_from_correlation():
    """3+ positive but correlated pairs -> CASE_B (downgraded from A)."""
    assert evaluate_wave2(n_positive=3, max_rho=0.6, n_correlated_pairs=1) == CASE_B
    assert evaluate_wave2(n_positive=4, max_rho=0.8, n_correlated_pairs=2) == CASE_B


def test_case_b_from_count():
    """Exactly 2 positive algos -> CASE_B regardless of correlation."""
    assert evaluate_wave2(n_positive=2, max_rho=0.1, n_correlated_pairs=0) == CASE_B
    assert evaluate_wave2(n_positive=2, max_rho=0.7, n_correlated_pairs=1) == CASE_B


def test_case_c():
    """1 positive algo -> CASE_C."""
    assert evaluate_wave2(n_positive=1, max_rho=0.0, n_correlated_pairs=0) == CASE_C


def test_case_d():
    """0 positive algos -> CASE_D."""
    assert evaluate_wave2(n_positive=0, max_rho=0.0, n_correlated_pairs=0) == CASE_D


# --- Boundary tests ---

def test_boundary_2_to_3():
    """Boundary between CASE_B (2 positive) and CASE_A (3 positive)."""
    assert evaluate_wave2(n_positive=2, max_rho=0.1, n_correlated_pairs=0) == CASE_B
    assert evaluate_wave2(n_positive=3, max_rho=0.1, n_correlated_pairs=0) == CASE_A


def test_boundary_0_to_1():
    """Boundary between CASE_D (0 positive) and CASE_C (1 positive)."""
    assert evaluate_wave2(n_positive=0, max_rho=0.0, n_correlated_pairs=0) == CASE_D
    assert evaluate_wave2(n_positive=1, max_rho=0.0, n_correlated_pairs=0) == CASE_C


# --- Correlation computation tests ---

def test_correlation_flag():
    """Two algos with |rho| > 0.5 flagged as correlated."""
    n = 200
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, n)

    signals = {
        "algo_a": base,
        "algo_b": base + rng.normal(0, 0.1, n),  # highly correlated with a
        "algo_c": rng.normal(0, 1, n),            # uncorrelated
    }

    rho_matrix, max_rho, correlated = compute_pairwise_correlations(signals)

    assert max_rho > CORRELATION_THRESHOLD
    assert ("algo_a", "algo_b") in correlated
    assert ("algo_a", "algo_c") not in correlated


def test_correlation_nan_with_few_samples():
    """Pairs with < 30 valid samples get NaN correlation, not flagged."""
    signals = {
        "algo_x": np.full(20, np.nan),
        "algo_y": np.full(20, np.nan),
    }

    rho_matrix, max_rho, correlated = compute_pairwise_correlations(signals)

    assert max_rho == 0.0
    assert len(correlated) == 0


def test_correlation_independent_signals():
    """Independent signals have |rho| < threshold."""
    n = 500
    rng = np.random.default_rng(123)
    signals = {
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    }

    _, max_rho, correlated = compute_pairwise_correlations(signals)

    assert max_rho < CORRELATION_THRESHOLD
    assert len(correlated) == 0
