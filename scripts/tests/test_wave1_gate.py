"""Unit tests for Wave 1 decision gate logic."""

from evaluate_wave1_gate import (
    CASE_A, CASE_B, CASE_C, CASE_D,
    evaluate_momentum_gate,
    evaluate_cpd,
)


def test_case_a_full_proceed():
    """OOS Sharpe=0.7, symbols_positive=2 -> CASE_A"""
    assert evaluate_momentum_gate(0.7, 2) == CASE_A


def test_case_b_cautious():
    """OOS Sharpe=0.6, symbols_positive=1 -> CASE_B"""
    assert evaluate_momentum_gate(0.6, 1) == CASE_B


def test_case_c_investigate():
    """OOS Sharpe=0.3, symbols_positive=2 -> CASE_C"""
    assert evaluate_momentum_gate(0.3, 2) == CASE_C


def test_case_d_stop():
    """OOS Sharpe=-0.2, symbols_positive=0 -> CASE_D"""
    assert evaluate_momentum_gate(-0.2, 0) == CASE_D


def test_case_boundary_sharpe_zero():
    """OOS Sharpe=0.0, symbols_positive=3 -> CASE_C (not D, boundary inclusive)"""
    assert evaluate_momentum_gate(0.0, 3) == CASE_C


def test_case_boundary_sharpe_half():
    """OOS Sharpe=0.5, symbols_positive=2 -> CASE_A (boundary inclusive)"""
    assert evaluate_momentum_gate(0.5, 2) == CASE_A


def test_cpd_keep():
    """CPD with vol_corr=0.20 and variance=0.05 -> KEEP"""
    assert evaluate_cpd(0.05, 0.20) == "KEEP"


def test_cpd_retire():
    """CPD with vol_corr=0.05 and variance=0.002 -> RETIRE"""
    assert evaluate_cpd(0.002, 0.05) == "RETIRE"


def test_cpd_low_variance_retire():
    """CPD with good correlation but low variance -> RETIRE"""
    assert evaluate_cpd(0.005, 0.30) == "RETIRE"


def test_cpd_low_corr_retire():
    """CPD with good variance but low correlation -> RETIRE"""
    assert evaluate_cpd(0.05, 0.05) == "RETIRE"
