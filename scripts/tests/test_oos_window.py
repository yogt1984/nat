"""Planted tests for the `nat oos --window` longitudinal validator.

Synthetic daily-P&L series with KNOWN properties, so the estimators
(max-drawdown, walk-forward holdout, deflated Sharpe wiring, the daily-P&L
matrix loader) are verified before any real-data use — per METHODOLOGY, the
non-negotiable planted-test-first rule.
"""
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]  # .../scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from alpha import oos_window as ow  # noqa: E402


# ── max drawdown ─────────────────────────────────────────────────────────────

def test_max_drawdown_known_path():
    # cum = [10, 5, 15]; peak = [10, 10, 15]; dd = [0, 5, 0]  -> max 5
    assert abs(ow.max_drawdown_bps(np.array([10.0, -5.0, 10.0])) - 5.0) < 1e-9


def test_max_drawdown_monotonic_up_is_zero():
    assert ow.max_drawdown_bps(np.array([1.0, 2.0, 3.0])) == 0.0


def test_max_drawdown_empty():
    assert ow.max_drawdown_bps(np.array([])) == 0.0


# ── Sharpe sign ──────────────────────────────────────────────────────────────

def test_sharpe_positive_drift():
    # mean 1.5, std 0.5 -> clearly positive Sharpe
    st = ow.strategy_stats(np.array([2.0, 1.0] * 15))
    assert st["sharpe"] > 0


def test_sharpe_zero_mean_is_zero():
    # symmetric zero-mean series -> Sharpe ~ 0 (and no div-by-zero crash)
    st = ow.strategy_stats(np.array([1.0, -1.0] * 15))
    assert abs(st["sharpe"]) < 1e-9


# ── walk-forward holdout (the overfit signal) ────────────────────────────────

def test_walk_forward_detects_overfit():
    # good in train (first 20), bad in test (last 10) -> OOS collapses
    pnl = np.concatenate([np.array([2.0, 1.0] * 10), np.array([-2.0, -1.0] * 5)])
    wf = ow.walk_forward_holdout(pnl, train_frac=0.67)
    assert wf["n_train"] >= 2 and wf["n_test"] >= 2
    assert wf["is_sharpe"] > 0
    assert wf["oos_sharpe"] < 0
    assert wf["oos_is_ratio"] < 0.7  # fails a G4-style robustness bar


def test_walk_forward_stable_signal():
    # same distribution throughout -> OOS/IS ~ 1
    wf = ow.walk_forward_holdout(np.array([2.0, 1.0] * 15), train_frac=0.67)
    assert wf["oos_is_ratio"] > 0.7


def test_walk_forward_too_few_days():
    wf = ow.walk_forward_holdout(np.array([1.0, 2.0]), train_frac=0.67)
    assert wf["n_train"] == 0 and wf["n_test"] == 0  # gracefully declines


# ── deflated Sharpe: must get HARDER with more trials ────────────────────────

def test_deflated_sharpe_decreases_with_more_trials():
    from backtest.walk_forward import compute_deflated_sharpe
    p_few = compute_deflated_sharpe(observed_sharpe=1.0, n_trials=3)
    p_many = compute_deflated_sharpe(observed_sharpe=1.0, n_trials=50)
    # More strategies searched -> higher bar -> lower probability it's real.
    assert p_many < p_few


def test_strategy_stats_reports_deflated():
    st = ow.strategy_stats(np.array([2.0, 1.0] * 15), n_trials=15)
    assert "deflated_sharpe_prob" in st
    assert st["deflated_sharpe_prob"] is None or 0.0 <= st["deflated_sharpe_prob"] <= 1.0


# ── daily-P&L matrix loader: windowing + dedup ───────────────────────────────

def _write_report(path, daily):
    path.write_text(json.dumps({"daily": daily}))


def test_load_matrix_windows_and_dedups(tmp_path):
    # file A: dates 01,02,03 ; file B (sorts later): overrides date 02
    _write_report(tmp_path / "gauntlet_a.json", [
        {"date": "2026-01-01", "algorithms": {"jump_detector": {"BTC": {"total_net_bps": 1.0}}}},
        {"date": "2026-01-02", "algorithms": {"jump_detector": {"BTC": {"total_net_bps": 2.0}}}},
        {"date": "2026-01-03", "algorithms": {"jump_detector": {"BTC": {"total_net_bps": 3.0}}}},
    ])
    _write_report(tmp_path / "gauntlet_b.json", [
        {"date": "2026-01-02", "algorithms": {"jump_detector": {"BTC": {"total_net_bps": 99.0}}}},
    ])
    window_dates, all_dates, matrix = ow.load_daily_matrix(
        tmp_path, window_days=2, symbols=["BTC"], algos=["jump_detector"])
    # windowing: last 2 of {01,02,03}
    assert window_dates == ["2026-01-02", "2026-01-03"]
    assert len(all_dates) == 3
    # dedup: file B (later filename) wins for 2026-01-02 -> 99.0
    np.testing.assert_array_equal(matrix["jump_detector"]["BTC"], np.array([99.0, 3.0]))


def test_load_matrix_missing_cells_default_zero(tmp_path):
    _write_report(tmp_path / "gauntlet_x.json", [
        {"date": "2026-02-01", "algorithms": {}},  # no algo data
    ])
    _, _, matrix = ow.load_daily_matrix(
        tmp_path, window_days=5, symbols=["ETH"], algos=["optimal_entry"])
    np.testing.assert_array_equal(matrix["optimal_entry"]["ETH"], np.array([0.0]))


def test_build_report_json_safe(tmp_path):
    # a single day -> walk_forward declines (ratio nan); ensure _json_safe nulls it
    _write_report(tmp_path / "gauntlet_z.json", [
        {"date": "2026-03-01", "algorithms": {"3f_liquidity": {"BTC": {"total_net_bps": 5.0}}}},
    ])
    report = ow.build_report(30, ["BTC"], ["3f_liquidity"], tmp_path, 0.67)
    safe = ow._json_safe(report)
    # must serialize without ValueError on NaN/Inf
    json.loads(json.dumps(safe))
    assert report["days_available"] == 1
